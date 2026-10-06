#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from hparams_search_scripts.gpu_resources import gpu_device_tokens, resolve_gpu_concurrency


@dataclass(frozen=True)
class SchedulerTask:
    task_id: str
    combo_key: str
    pool: str
    command: list[str]
    env: dict[str, str]
    cwd: Path
    log_path: Path
    retry_count: int
    label: str
    log_index_path: Path | None = None
    concurrency_group: str | None = None
    work_units: int = 1


@dataclass(frozen=True)
class TaskResult:
    task: SchedulerTask
    success: bool
    attempts: int
    total_seconds: float
    return_code: int
    error_message: str | None
    last_log_path: Path
    worker_id: int


@dataclass
class _RunningTask:
    task: SchedulerTask
    process: subprocess.Popen
    log_handle: TextIO
    log_path: Path
    worker_id: int
    attempt: int
    total_seconds: float
    start_time: float


class MultiPoolScheduler:
    def __init__(
        self,
        *,
        worker_ids: list[int],
        max_parallel_per_worker: int | dict[int, int],
        poll_interval_sec: float,
        launch_interval_sec: float,
        device: str,
    ) -> None:
        if len(worker_ids) == 0:
            raise ValueError("worker_ids must be non-empty")
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("worker_ids must be unique")
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be > 0")
        if launch_interval_sec < 0:
            raise ValueError("launch_interval_sec must be >= 0")
        if device not in {"cpu", "gpu"}:
            raise ValueError("device must be one of {'cpu', 'gpu'}")

        self._worker_ids = list(worker_ids)
        self._poll_interval_sec = float(poll_interval_sec)
        self._launch_interval_sec = float(launch_interval_sec)
        self._device = device
        limits = resolve_gpu_concurrency(self._worker_ids, max_parallel_per_worker)
        self._parallel_limit_by_worker = (
            dict(limits) if isinstance(limits, dict)
            else {worker_id: limits for worker_id in self._worker_ids}
        )
        self._gpu_tokens = gpu_device_tokens(self._worker_ids) if device == "gpu" else {}
        if device == "gpu":
            print(
                f"[Scheduler] GPU concurrency: {self._parallel_limit_by_worker}",
                flush=True,
            )

        self._pool_queues: dict[str, deque[SchedulerTask]] = {}
        self._pool_order: list[str] = []
        self._round_robin_cursor = 0

        self._running: dict[int, _RunningTask] = {}
        self._launched_count_by_worker = {worker_id: 0 for worker_id in self._worker_ids}
        self._group_parallel_limit_per_worker: dict[str, int] = {}
        self._cancelled_combos: set[str] = set()
        self._last_launch_monotonic = 0.0

    def enqueue(self, task: SchedulerTask) -> None:
        if task.combo_key in self._cancelled_combos:
            return
        if task.pool not in self._pool_queues:
            self._pool_queues[task.pool] = deque()
            self._pool_order.append(task.pool)
        self._pool_queues[task.pool].append(task)

    def cancel_combo(self, combo_key: str) -> None:
        self._cancelled_combos.add(combo_key)
        for pool_name in self._pool_order:
            queue = self._pool_queues[pool_name]
            self._pool_queues[pool_name] = deque(
                task for task in queue if task.combo_key != combo_key
            )

    def set_group_parallel_limit_per_worker(self, group: str, limit: int) -> None:
        if limit < 1:
            raise ValueError("group parallel limit per worker must be >= 1")
        self._group_parallel_limit_per_worker[group] = int(limit)

    def run(self, on_result: Callable[[TaskResult], list[SchedulerTask] | None]) -> None:
        while self._has_pending_work() or len(self._running) > 0:
            self._poll_running(on_result=on_result)
            self._launch_pending()
            if self._has_pending_work() or len(self._running) > 0:
                time.sleep(self._poll_interval_sec)

    def _has_pending_work(self) -> bool:
        return any(len(queue) > 0 for queue in self._pool_queues.values())

    def _running_count_by_worker(self) -> dict[int, int]:
        counts = {worker_id: 0 for worker_id in self._worker_ids}
        for running in self._running.values():
            counts[running.worker_id] += 1
        return counts

    def _running_count_by_group_worker(self) -> dict[tuple[str, int], int]:
        counts: dict[tuple[str, int], int] = {}
        for running in self._running.values():
            group = running.task.concurrency_group
            if group is None:
                continue
            key = (group, running.worker_id)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _next_pool_name(self) -> str | None:
        if len(self._pool_order) == 0:
            return None

        total_pools = len(self._pool_order)
        for _ in range(total_pools):
            pool_index = self._round_robin_cursor % total_pools
            pool_name = self._pool_order[pool_index]
            self._round_robin_cursor = (self._round_robin_cursor + 1) % total_pools
            if len(self._pool_queues[pool_name]) > 0:
                return pool_name
        return None

    def _choose_worker_for_task(
        self,
        task: SchedulerTask,
        *,
        running_count_by_worker: dict[int, int],
        running_count_by_group_worker: dict[tuple[str, int], int],
    ) -> int | None:
        group_limit: int | None = None
        if task.concurrency_group is not None:
            group_limit = self._group_parallel_limit_per_worker.get(task.concurrency_group)

        candidates = [
            worker_id
            for worker_id in self._worker_ids
            if running_count_by_worker[worker_id] < self._parallel_limit_by_worker[worker_id]
            and (
                group_limit is None
                or task.concurrency_group is None
                or running_count_by_group_worker.get((task.concurrency_group, worker_id), 0) < group_limit
            )
        ]
        if len(candidates) == 0:
            return None

        candidates.sort(
            key=lambda worker_id: (
                running_count_by_worker[worker_id],
                self._launched_count_by_worker[worker_id],
                worker_id,
            )
        )
        return candidates[0]

    def _attempt_log_path(self, task: SchedulerTask, attempt: int) -> Path:
        suffix = task.log_path.suffix or ".log"
        return task.log_path.with_name(f"{task.log_path.stem}_attempt_{attempt}{suffix}")

    def _wait_for_launch_interval(self) -> None:
        if self._launch_interval_sec <= 0:
            return

        since_last_launch = time.monotonic() - self._last_launch_monotonic
        remaining = self._launch_interval_sec - since_last_launch
        if remaining > 0:
            time.sleep(remaining)

    def _launch_pending(self) -> None:
        while True:
            running_count_by_worker = self._running_count_by_worker()
            running_count_by_group_worker = self._running_count_by_group_worker()
            launched = False

            if len(self._pool_order) == 0:
                return

            for _ in range(len(self._pool_order)):
                pool_name = self._next_pool_name()
                if pool_name is None:
                    return

                queue = self._pool_queues[pool_name]
                while len(queue) > 0 and queue[0].combo_key in self._cancelled_combos:
                    queue.popleft()
                if len(queue) == 0:
                    continue

                queue_len = len(queue)
                for _ in range(queue_len):
                    if len(queue) == 0:
                        break
                    task = queue[0]
                    worker_id = self._choose_worker_for_task(
                        task,
                        running_count_by_worker=running_count_by_worker,
                        running_count_by_group_worker=running_count_by_group_worker,
                    )
                    if worker_id is not None:
                        queue.popleft()
                        self._wait_for_launch_interval()
                        self._launch_task(
                            task=task,
                            worker_id=worker_id,
                            attempt=1,
                            elapsed_before=0.0,
                        )
                        launched = True
                        break
                    queue.rotate(-1)
                    while len(queue) > 0 and queue[0].combo_key in self._cancelled_combos:
                        queue.popleft()

                if launched:
                    break

            if not launched:
                return

    def _launch_task(
        self,
        *,
        task: SchedulerTask,
        worker_id: int,
        attempt: int,
        elapsed_before: float,
    ) -> None:
        log_path = self._attempt_log_path(task, attempt)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.update(task.env)

        temp_root = env.get("TMPDIR") or env.get("TMP") or env.get("TEMP")
        if temp_root is None or str(temp_root).strip() == "":
            temp_root = str(Path.home() / ".cache" / "tmp")
        temp_root_path = Path(temp_root).expanduser()
        temp_root_path.mkdir(parents=True, exist_ok=True)
        temp_root = str(temp_root_path)
        env["TMPDIR"] = temp_root
        env.setdefault("TMP", temp_root)
        env.setdefault("TEMP", temp_root)

        if self._device == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = self._gpu_tokens[worker_id]
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"

        log_handle = log_path.open("w", encoding="utf-8")
        log_handle.write(f"cwd: {task.cwd}\n")
        log_handle.write(f"worker_id: {worker_id}\n")
        env_preview = " ".join(f"{key}={env[key]!r}" for key in sorted(env.keys()))
        log_handle.write(f"env: {env_preview}\n")
        log_handle.write(f"command: {' '.join(task.command)}\n")
        log_handle.flush()

        process = subprocess.Popen(
            task.command,
            cwd=str(task.cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )

        self._running[process.pid] = _RunningTask(
            task=task,
            process=process,
            log_handle=log_handle,
            log_path=log_path,
            worker_id=worker_id,
            attempt=attempt,
            total_seconds=elapsed_before,
            start_time=time.monotonic(),
        )
        self._launched_count_by_worker[worker_id] += 1
        self._last_launch_monotonic = time.monotonic()

    @staticmethod
    def _safe_close(handle: TextIO) -> None:
        if not handle.closed:
            handle.close()

    def _append_log_index(self, result: TaskResult) -> None:
        path = result.task.log_index_path
        if path is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        need_header = not path.is_file()
        with path.open("a", encoding="utf-8") as handle:
            if need_header:
                handle.write(
                    "task_id,pool,label,status,attempts,return_code,seconds,last_log_path,error_message\n"
                )
            status = "completed" if result.success else "failed"
            error_message = "" if result.error_message is None else result.error_message.replace("\n", " ")
            handle.write(
                f"{result.task.task_id},{result.task.pool},{result.task.label},{status},"
                f"{result.attempts},{result.return_code},{result.total_seconds:.6f},"
                f"{result.last_log_path},{error_message}\n"
            )

    def _poll_running(self, on_result: Callable[[TaskResult], list[SchedulerTask] | None]) -> None:
        for pid, running in list(self._running.items()):
            returncode = running.process.poll()
            if returncode is None:
                continue

            elapsed = time.monotonic() - running.start_time
            total_elapsed = running.total_seconds + elapsed
            self._safe_close(running.log_handle)
            del self._running[pid]

            if (
                returncode != 0
                and running.attempt <= running.task.retry_count
                and running.task.combo_key not in self._cancelled_combos
            ):
                self._wait_for_launch_interval()
                self._launch_task(
                    task=running.task,
                    worker_id=running.worker_id,
                    attempt=running.attempt + 1,
                    elapsed_before=total_elapsed,
                )
                continue

            if running.task.combo_key in self._cancelled_combos:
                continue

            success = returncode == 0
            error_message = None
            if not success:
                total_attempts = 1 + running.task.retry_count
                error_message = (
                    f"{running.task.label} attempt {running.attempt}/{total_attempts} "
                    f"failed with exit code {returncode}; log={running.log_path}"
                )

            result = TaskResult(
                task=running.task,
                success=success,
                attempts=running.attempt,
                total_seconds=total_elapsed,
                return_code=int(returncode),
                error_message=error_message,
                last_log_path=running.log_path,
                worker_id=running.worker_id,
            )
            self._append_log_index(result)
            spawned = on_result(result) or []
            for task in spawned:
                self.enqueue(task)
