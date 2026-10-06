#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from hparams_search_scripts import mechanism_stage_utils
    from hparams_search_scripts.multi_pool_scheduler import MultiPoolScheduler, SchedulerTask, TaskResult
except ModuleNotFoundError:
    import mechanism_stage_utils  # type: ignore
    from multi_pool_scheduler import MultiPoolScheduler, SchedulerTask, TaskResult  # type: ignore


MANIFEST_COLUMNS = [
    "job_id",
    "job_dir",
    *mechanism_stage_utils.OUTER_AXIS_NAMES,
    "search_status",
    "grid_total",
    "grid_done",
    "verify_total",
    "verify_done",
    "search_attempts",
    "search_seconds",
    "logs_dir",
    "best_rank",
    "best_candidate_id",
    "best_x_steps",
    "best_learning_rate",
    "best_weight_decay",
    "best_dropout",
    "best_tao2",
    "best_verify_val_acc_mean",
    "best_verify_val_acc_std",
    "best_verify_test_acc_mean",
    "best_verify_test_acc_std",
    "best_config_path",
    "recommended_command_path",
    "error_message",
]

DEFAULT_STAGE_RETRY_COUNT = 0


class SuiteError(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchJob:
    job_id: str
    job_dir: Path
    display_name: str
    job_spec: dict[str, object]


@dataclass(frozen=True)
class BatchSpec:
    output_root: Path
    execution: mechanism_stage_utils.ExecutionSettings
    jobs: list[BatchJob]
    config_copy_source: Path


@dataclass
class SearchState:
    job: BatchJob
    row: dict[str, str]
    logs_dir: Path
    log_index_path: Path
    training_total: int
    training_accounted: int
    grid_total: int
    verify_total: int
    grid_pending_total: int = 0
    grid_completed_runtime: int = 0
    verify_pending_total: int = 0
    verify_completed_runtime: int = 0
    failed: bool = False


@dataclass
class ProgressBar:
    total: int
    width: int = 30
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    _rendered: bool = False
    _last_text: str = ""

    @property
    def done(self) -> int:
        return self.completed + self.skipped + self.failed

    def _line(self) -> str:
        ratio = 1.0 if self.total <= 0 else min(1.0, self.done / self.total)
        filled = int(round(self.width * ratio))
        bar = "#" * filled + "-" * (self.width - filled)
        percent = ratio * 100.0
        return (
            f"Progress [{bar}] {self.done}/{self.total} ({percent:5.1f}%) "
            f"done={self.completed} cached={self.skipped} failed={self.failed}"
        )

    def render(self, *, force: bool = False) -> None:
        line = self._line()
        if not force and self._rendered and line == self._last_text:
            return
        padded = line
        if len(self._last_text) > len(line):
            padded = line + (" " * (len(self._last_text) - len(line)))
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()
        self._rendered = True
        self._last_text = line

    def log(self, message: str) -> None:
        if self._rendered:
            sys.stdout.write("\r" + (" " * len(self._last_text)) + "\r")
            sys.stdout.flush()
        print(message)
        self._rendered = False
        self._last_text = ""
        self.render(force=True)

    def mark_completed(self, *, count: int = 1) -> None:
        if count > 0:
            self.completed += count
            self.render()

    def mark_skipped(self, *, count: int = 1) -> None:
        if count > 0:
            self.skipped += count
            self.render()

    def mark_failed(self, *, count: int = 1) -> None:
        if count > 0:
            self.failed += count
            self.render()

    def finalize(self) -> None:
        self.render(force=True)
        if self._rendered:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._rendered = False
        self._last_text = ""


def _format_seconds(value: float) -> str:
    return f"{float(value):.6f}"


def _default_row(job: BatchJob) -> dict[str, str]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row["job_id"] = job.job_id
    row["job_dir"] = str(job.job_dir)

    fixed_params = job.job_spec["fixed_params"]
    if not isinstance(fixed_params, dict):
        raise SuiteError("job_spec.fixed_params must be a mapping")

    for axis_name in mechanism_stage_utils.OUTER_AXIS_NAMES:
        value = fixed_params.get(axis_name)
        row[axis_name] = mechanism_stage_utils.canonical_search_value(value)

    row["search_status"] = "not_started"
    row["grid_total"] = "0"
    row["grid_done"] = "0"
    row["verify_total"] = "0"
    row["verify_done"] = "0"
    row["search_attempts"] = "0"
    row["search_seconds"] = "0.000000"
    row["logs_dir"] = str(mechanism_stage_utils.logs_dir(job.job_dir))
    return row


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})
    tmp_path.replace(path)


def _accumulate_metrics(row: dict[str, str], attempts: int, seconds: float) -> None:
    previous_attempts = int(str(row.get("search_attempts", "0")).strip() or "0")
    previous_seconds = float(str(row.get("search_seconds", "0")).strip() or "0")
    row["search_attempts"] = str(previous_attempts + int(attempts))
    row["search_seconds"] = _format_seconds(previous_seconds + float(seconds))


def _load_best_outputs_into_row(row: dict[str, str], job_dir: Path) -> None:
    best_config = mechanism_stage_utils.load_best_config(job_dir)
    best_candidate = best_config.get("best_candidate")
    verify_metrics = best_config.get("verify_metrics")
    artifacts = best_config.get("artifacts")

    if not isinstance(best_candidate, dict) or not isinstance(verify_metrics, dict) or not isinstance(artifacts, dict):
        raise SuiteError(f"best_config.yaml has an invalid shape under {job_dir}")

    val_metrics = verify_metrics.get("val_acc")
    test_metrics = verify_metrics.get("test_acc")
    if not isinstance(val_metrics, dict) or not isinstance(test_metrics, dict):
        raise SuiteError(f"best_config.yaml is missing verify metric summaries under {job_dir}")

    row["best_rank"] = mechanism_stage_utils.canonical_search_value(best_config.get("best_rank"))
    row["best_candidate_id"] = mechanism_stage_utils.canonical_search_value(best_candidate.get("candidate_id"))
    row["best_x_steps"] = mechanism_stage_utils.canonical_search_value(best_candidate.get("x_steps"))
    row["best_learning_rate"] = mechanism_stage_utils.canonical_search_value(best_candidate.get("learning_rate"))
    row["best_weight_decay"] = mechanism_stage_utils.canonical_search_value(best_candidate.get("weight_decay"))
    row["best_dropout"] = mechanism_stage_utils.canonical_search_value(best_candidate.get("dropout"))
    row["best_tao2"] = mechanism_stage_utils.canonical_search_value(best_candidate.get("tao2"))
    row["best_verify_val_acc_mean"] = mechanism_stage_utils.canonical_search_value(val_metrics.get("mean"))
    row["best_verify_val_acc_std"] = mechanism_stage_utils.canonical_search_value(val_metrics.get("std"))
    row["best_verify_test_acc_mean"] = mechanism_stage_utils.canonical_search_value(test_metrics.get("mean"))
    row["best_verify_test_acc_std"] = mechanism_stage_utils.canonical_search_value(test_metrics.get("std"))
    row["best_config_path"] = str(mechanism_stage_utils.best_config_path(job_dir))
    row["recommended_command_path"] = str(mechanism_stage_utils.recommended_command_path(job_dir))


def _normalized_job_spec_for_comparison(job_spec: dict[str, object]) -> dict[str, object]:
    normalized = dict(job_spec)
    normalized.pop("job_id", None)
    # These fields describe the current machine/environment, not the
    # experiment. They must not invalidate a reusable job when moving between
    # lab nodes or renaming the Conda environment.
    normalized.pop("python_bin", None)
    normalized.pop("pre_smoothing_feature_cache_root", None)

    fixed_params = normalized.get("fixed_params")
    if isinstance(fixed_params, dict):
        fixed_params_normalized = {
            axis_name: fixed_params.get(axis_name)
            for axis_name in mechanism_stage_utils.OUTER_AXIS_NAMES
        }
        normalized["fixed_params"] = fixed_params_normalized

    return normalized


def _ensure_job_spec(job: BatchJob) -> None:
    mechanism_stage_utils.ensure_job_directories(job.job_dir)
    path = mechanism_stage_utils.job_spec_path(job.job_dir)
    if not path.exists():
        mechanism_stage_utils.write_yaml_file(path, job.job_spec)
        return

    existing = mechanism_stage_utils.load_job_spec(job.job_dir)
    if mechanism_stage_utils.canonical_yaml_text(existing) == mechanism_stage_utils.canonical_yaml_text(job.job_spec):
        return

    existing_normalized = _normalized_job_spec_for_comparison(existing)
    expected_normalized = _normalized_job_spec_for_comparison(job.job_spec)
    if mechanism_stage_utils.canonical_yaml_text(existing_normalized) == mechanism_stage_utils.canonical_yaml_text(expected_normalized):
        mechanism_stage_utils.write_yaml_file(path, job.job_spec)
        return

    raise SuiteError(
        f"Existing job_spec.yaml does not match the current configuration: {path}"
    )


def _count_existing_grid(job_spec: dict[str, object], job_dir: Path) -> tuple[list[mechanism_stage_utils.CandidateSpec], int]:
    try:
        base_seed = int(job_spec["base_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SuiteError("job_spec.base_seed must be an integer") from exc

    missing: list[mechanism_stage_utils.CandidateSpec] = []
    done = 0
    for candidate in mechanism_stage_utils.job_candidates(job_spec):
        output_dir = mechanism_stage_utils.grid_candidate_output_dir(job_dir, candidate)
        if mechanism_stage_utils.candidate_result_exists(
            output_dir,
            candidate,
            expected_seed=base_seed,
        ):
            done += 1
        else:
            missing.append(candidate)
    return missing, done


def _count_existing_verify(
    job_spec: dict[str, object],
    job_dir: Path,
    ranked_candidates: list[mechanism_stage_utils.RankedCandidate],
) -> tuple[list[tuple[mechanism_stage_utils.RankedCandidate, int]], int]:
    verify_stage = job_spec["defaults"]["stage"]["verify"]  # type: ignore[index]
    base_seed = int(job_spec["base_seed"])
    repeats = int(verify_stage["repeats"])

    missing: list[tuple[mechanism_stage_utils.RankedCandidate, int]] = []
    done = 0
    for ranked in ranked_candidates:
        for repeat_id in range(1, repeats + 1):
            seed = base_seed + repeat_id - 1
            output_dir = mechanism_stage_utils.verify_candidate_output_dir(job_dir, ranked, repeat_id)
            if mechanism_stage_utils.candidate_result_exists(
                output_dir,
                ranked.candidate,
                expected_seed=seed,
            ):
                done += 1
            else:
                missing.append((ranked, repeat_id))
    return missing, done


def _bool_fixed_param(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _uses_figure3_grouped_state_runner(config_path: Path, fixed_params: dict[str, Any]) -> bool:
    if config_path.parent.name == 'figure30':
        figure_name = 'figure30'
        config_group = str(fixed_params.get('backbone', '')).strip().lower()
    else:
        figure_name = config_path.parent.parent.name
        config_group = config_path.parent.name
    if figure_name == 'figure3':
        allowed_groups = {'gcn', 'gat'}
        allowed_backbones = {'gcn', 'gat'}
    elif figure_name == 'figure32':
        allowed_groups = {'gcn', 'gat', 'sage'}
        allowed_backbones = {'gcn', 'gat', 'sage'}
    elif figure_name == 'main_add':
        allowed_groups = {'gcn', 'gat', 'sage'}
        allowed_backbones = {'gcn', 'gat', 'sage'}
    elif figure_name == 'figure30':
        allowed_groups = {'gat'}
        allowed_backbones = {'gat'}
    else:
        return False
    if config_group not in allowed_groups:
        return False
    if str(fixed_params.get('feature', '')).strip().lower() != 'raw':
        return False
    if str(fixed_params.get('smoother', '')).strip().lower() not in {'hoa', 'kprop'}:
        return False
    if str(fixed_params.get('backbone', '')).strip().lower() not in allowed_backbones:
        return False
    return True


def _uses_semantic_raw_grouped_state_runner(fixed_params: dict[str, Any]) -> bool:
    if str(fixed_params.get('feature', '')).strip().lower() != 'raw':
        return False
    if str(fixed_params.get('backbone', '')).strip().lower() not in {'sage', 'gcn', 'gat'}:
        return False
    if str(fixed_params.get('smoother', '')).strip().lower() not in {'hoa', 'kprop'}:
        return False
    if (
        _bool_fixed_param(fixed_params.get('use_nfr', False))
        and str(fixed_params.get('mechanism', '')).strip().lower() not in {'mbm', 'pm', 'hds'}
    ):
        return False
    return True


def _uses_semantic_random_normal_grouped_state_runner(fixed_params: dict[str, Any]) -> bool:
    if str(fixed_params.get('feature', '')).strip().lower() != 'random_normal':
        return False
    if _bool_fixed_param(fixed_params.get('use_nfr', False)):
        return False
    if str(fixed_params.get('backbone', '')).strip().lower() not in {'sage', 'gcn', 'gat'}:
        return False
    if str(fixed_params.get('smoother', '')).strip().lower() not in {'hoa', 'kprop'}:
        return False
    return True


def _uses_semantic_operator_grouped_runner(fixed_params: dict[str, Any]) -> bool:
    if str(fixed_params.get('feature', '')).strip().lower() != 'operator':
        return False
    if _bool_fixed_param(fixed_params.get('use_nfr', False)):
        return False
    if str(fixed_params.get('backbone', '')).strip().lower() not in {'sage', 'gcn', 'gat'}:
        return False
    return True


def _uses_semantic_sim_grouped_state_runner(
    job_spec: dict[str, Any],
    fixed_params: dict[str, Any],
) -> bool:
    if str(fixed_params.get('feature', '')).strip().lower() != 'sim':
        return False
    if _bool_fixed_param(fixed_params.get('use_nfr', False)):
        return False
    if str(fixed_params.get('backbone', '')).strip().lower() not in {'sage', 'gcn', 'gat'}:
        return False
    if str(fixed_params.get('smoother', '')).strip().lower() not in {'hoa', 'kprop'}:
        return False
    defaults = job_spec.get('defaults')
    if not isinstance(defaults, dict):
        return False
    trainer_defaults = defaults.get('trainer')
    if not isinstance(trainer_defaults, dict):
        return False
    return not _bool_fixed_param(trainer_defaults.get('sim_epoch_refresh', False))


def _job_grouped_runner_mode(
    config_path: Path,
    job_spec: dict[str, Any],
) -> str | None:
    fixed_params = job_spec.get('fixed_params')
    if not isinstance(fixed_params, dict):
        return None
    if _uses_semantic_operator_grouped_runner(fixed_params):
        return 'operator'
    if (
        str(fixed_params.get('feature', '')).strip().lower() == 'raw'
        and _bool_fixed_param(fixed_params.get('use_nfr', False))
    ):
        return (
            'materialized'
            if _uses_semantic_raw_grouped_state_runner(fixed_params)
            else None
        )
    if (
        _uses_figure3_grouped_state_runner(config_path, fixed_params)
        or _uses_semantic_raw_grouped_state_runner(fixed_params)
        or _uses_semantic_random_normal_grouped_state_runner(fixed_params)
        or _uses_semantic_sim_grouped_state_runner(job_spec, fixed_params)
    ):
        return 'materialized'
    return None


def _grouped_runner_mode(spec: BatchSpec) -> str | None:
    if len(spec.jobs) == 0:
        return None

    config_path = spec.config_copy_source.resolve()
    modes = [
        _job_grouped_runner_mode(config_path, job.job_spec)
        for job in spec.jobs
    ]
    selected_mode = modes[0]
    if selected_mode is None or any(mode != selected_mode for mode in modes[1:]):
        return None
    return selected_mode


def _group_grid_candidates(
    candidates: list[mechanism_stage_utils.CandidateSpec],
) -> list[list[mechanism_stage_utils.CandidateSpec]]:
    grouped: dict[tuple[int, str], list[mechanism_stage_utils.CandidateSpec]] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            int(item.x_steps),
            mechanism_stage_utils.canonical_optional_float_text(item.tao2),
            int(item.candidate_id),
        ),
    ):
        key = (int(candidate.x_steps), mechanism_stage_utils.canonical_optional_float_text(candidate.tao2))
        grouped.setdefault(key, []).append(candidate)
    return list(grouped.values())


def _group_verify_work(
    items: list[tuple[mechanism_stage_utils.RankedCandidate, int]],
) -> list[list[tuple[mechanism_stage_utils.RankedCandidate, int]]]:
    grouped: dict[tuple[int, int, str], list[tuple[mechanism_stage_utils.RankedCandidate, int]]] = {}
    for ranked, repeat_id in sorted(
        items,
        key=lambda item: (
            int(item[1]),
            int(item[0].candidate.x_steps),
            mechanism_stage_utils.canonical_optional_float_text(item[0].candidate.tao2),
            int(item[0].candidate_id),
        ),
    ):
        key = (
            int(repeat_id),
            int(ranked.candidate.x_steps),
            mechanism_stage_utils.canonical_optional_float_text(ranked.candidate.tao2),
        )
        grouped.setdefault(key, []).append((ranked, repeat_id))
    return list(grouped.values())


def _stage_scripts(repo_root: Path) -> dict[str, Path]:
    scripts = {
        "grid_task": repo_root / "hparams_search_scripts" / "run_mechanism_grid_task.py",
        "group_task": repo_root / "hparams_search_scripts" / "run_mechanism_group_task.py",
        "operator_group_task": repo_root / "hparams_search_scripts" / "run_mechanism_operator_group_task.py",
        "grid_rank": repo_root / "hparams_search_scripts" / "run_mechanism_grid_rank.py",
        "verify_task": repo_root / "hparams_search_scripts" / "run_mechanism_verify_task.py",
        "verify_finalize": repo_root / "hparams_search_scripts" / "run_mechanism_verify_finalize.py",
    }
    for name, path in scripts.items():
        if not path.is_file():
            raise SuiteError(f"Stage script not found ({name}): {path}")
    return scripts


def run_batch_search(
    spec: BatchSpec,
    *,
    repo_root: Path,
) -> tuple[int, int, int, Path]:
    if not (repo_root / "main.py").is_file():
        raise SuiteError(f"main.py not found under repo root: {repo_root}")

    stage_scripts = _stage_scripts(repo_root)
    grouped_runner_mode = _grouped_runner_mode(spec)
    use_grouped_state_runner = grouped_runner_mode is not None
    grouped_stage_script = stage_scripts["operator_group_task"] if grouped_runner_mode == "operator" else stage_scripts["group_task"]
    spec.output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spec.config_copy_source, spec.output_root / mechanism_stage_utils.INPUT_CONFIG_COPY_FILENAME)

    manifest_path = spec.output_root / mechanism_stage_utils.ROOT_MANIFEST_FILENAME
    rows_by_job_id: dict[str, dict[str, str]] = {}
    states: list[SearchState] = []
    completed_jobs = 0
    skipped_jobs = 0
    failed_jobs = 0

    total_training_tasks = 0
    for job in spec.jobs:
        candidate_count = len(mechanism_stage_utils.job_candidates(job.job_spec))
        verify_repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])  # type: ignore[index]
        verify_total = min(mechanism_stage_utils.VERIFY_TOPK, candidate_count) * verify_repeats
        total_training_tasks += candidate_count + verify_total

    progress = ProgressBar(total=total_training_tasks)
    progress.render(force=True)

    for job in spec.jobs:
        _ensure_job_spec(job)
        row = _default_row(job)
        rows_by_job_id[job.job_id] = row

        if mechanism_stage_utils.job_outputs_complete(job.job_dir):
            _load_best_outputs_into_row(row, job.job_dir)
            row["search_status"] = "skipped_existing_result"
            skipped_jobs += 1
            completed_jobs += 1

            candidate_count = len(mechanism_stage_utils.job_candidates(job.job_spec))
            verify_repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])  # type: ignore[index]
            verify_total = min(mechanism_stage_utils.VERIFY_TOPK, candidate_count) * verify_repeats
            progress.mark_skipped(count=candidate_count + verify_total)
            continue

        logs_dir = mechanism_stage_utils.logs_dir(job.job_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_index_path = logs_dir / "task_index.csv"

        missing_grid, existing_grid_done = _count_existing_grid(job.job_spec, job.job_dir)
        candidate_count = len(mechanism_stage_utils.job_candidates(job.job_spec))
        verify_repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])  # type: ignore[index]
        verify_total = min(mechanism_stage_utils.VERIFY_TOPK, candidate_count) * verify_repeats
        accounted = existing_grid_done
        verify_done = 0
        initial_verify_missing: list[tuple[mechanism_stage_utils.RankedCandidate, int]] | None = None
        initial_enqueue_rank = False
        initial_enqueue_finalize = False

        topk_path = mechanism_stage_utils.verify_topk_path(job.job_dir)
        if len(missing_grid) == 0:
            if topk_path.is_file():
                ranked = mechanism_stage_utils.load_ranked_candidates(topk_path)
                initial_verify_missing, verify_done = _count_existing_verify(job.job_spec, job.job_dir, ranked)
                accounted += verify_done
                if len(initial_verify_missing) == 0:
                    initial_enqueue_finalize = True
            else:
                initial_enqueue_rank = True

        if accounted > 0:
            progress.mark_skipped(count=accounted)

        row["search_status"] = "in_progress"
        row["grid_total"] = str(candidate_count)
        row["grid_done"] = str(existing_grid_done)
        row["verify_total"] = str(verify_total)
        row["verify_done"] = str(verify_done)

        states.append(
            SearchState(
                job=job,
                row=row,
                logs_dir=logs_dir,
                log_index_path=log_index_path,
                training_total=candidate_count + verify_total,
                training_accounted=accounted,
                grid_total=candidate_count,
                verify_total=verify_total,
                grid_pending_total=len(missing_grid),
                verify_pending_total=0 if initial_verify_missing is None else len(initial_verify_missing),
            )
        )

        state = states[-1]
        state.row["_initial_rank"] = "true" if initial_enqueue_rank else "false"
        state.row["_initial_finalize"] = "true" if initial_enqueue_finalize else "false"
        state.row["_missing_grid_ids"] = ",".join(str(candidate.candidate_id) for candidate in missing_grid)
        state.row["_initial_verify_work"] = ",".join(
            f"{ranked.rank}:{repeat_id}" for ranked, repeat_id in (initial_verify_missing or [])
        )

    _write_manifest(manifest_path, list(rows_by_job_id.values()))

    if len(states) == 0:
        progress.finalize()
        print()
        print(f"Batch finished: completed={completed_jobs} skipped={skipped_jobs} failed={failed_jobs}")
        print(f"Manifest: {manifest_path}")
        return completed_jobs, skipped_jobs, failed_jobs, manifest_path

    scheduler = MultiPoolScheduler(
        worker_ids=spec.execution.worker_ids,
        max_parallel_per_worker=spec.execution.max_parallel_per_worker,
        poll_interval_sec=spec.execution.poll_interval_sec,
        launch_interval_sec=spec.execution.launch_interval_sec,
        device=spec.execution.device,
    )
    state_by_job_id = {state.job.job_id: state for state in states}
    stage_runner_python = sys.executable

    def build_grid_task(state: SearchState, candidate: mechanism_stage_utils.CandidateSpec) -> SchedulerTask:
        task_id = f"grid_candidate_{candidate.candidate_id:04d}"
        return SchedulerTask(
            task_id=task_id,
            combo_key=state.job.job_id,
            pool="grid",
            command=[
                stage_runner_python,
                str(stage_scripts["grid_task"]),
                str(state.job.job_dir),
                "--candidate_id",
                str(candidate.candidate_id),
            ],
            env={},
            cwd=repo_root,
            log_path=state.logs_dir / f"{task_id}.log",
            retry_count=DEFAULT_STAGE_RETRY_COUNT,
            label=task_id,
            log_index_path=state.log_index_path,
            work_units=1,
        )

    def build_grid_group_task(
        state: SearchState,
        grouped_candidates: list[mechanism_stage_utils.CandidateSpec],
    ) -> SchedulerTask:
        first = grouped_candidates[0]
        tao2_token = mechanism_stage_utils.path_token(first.tao2)
        task_id = f"grid_group_xsteps={int(first.x_steps)}__tao2={tao2_token}"
        command = [
            stage_runner_python,
            str(grouped_stage_script),
            str(state.job.job_dir),
            "--stage",
            "grid",
        ]
        for candidate in grouped_candidates:
            command.extend(["--candidate_id", str(candidate.candidate_id)])
        return SchedulerTask(
            task_id=task_id,
            combo_key=state.job.job_id,
            pool="grid",
            command=command,
            env={},
            cwd=repo_root,
            log_path=state.logs_dir / f"{task_id}.log",
            retry_count=DEFAULT_STAGE_RETRY_COUNT,
            label=task_id,
            log_index_path=state.log_index_path,
            work_units=len(grouped_candidates),
        )

    def build_rank_task(state: SearchState) -> SchedulerTask:
        return SchedulerTask(
            task_id="grid_rank",
            combo_key=state.job.job_id,
            pool="verify",
            command=[stage_runner_python, str(stage_scripts["grid_rank"]), str(state.job.job_dir)],
            env={},
            cwd=repo_root,
            log_path=state.logs_dir / "grid_rank.log",
            retry_count=DEFAULT_STAGE_RETRY_COUNT,
            label="grid_rank",
            log_index_path=state.log_index_path,
        )

    def build_verify_task(
        state: SearchState,
        ranked: mechanism_stage_utils.RankedCandidate,
        repeat_id: int,
    ) -> SchedulerTask:
        task_id = f"verify_rank{ranked.rank:02d}_repeat{repeat_id:02d}"
        return SchedulerTask(
            task_id=task_id,
            combo_key=state.job.job_id,
            pool="verify",
            command=[
                stage_runner_python,
                str(stage_scripts["verify_task"]),
                str(state.job.job_dir),
                "--rank",
                str(ranked.rank),
                "--repeat_id",
                str(repeat_id),
            ],
            env={},
            cwd=repo_root,
            log_path=state.logs_dir / f"{task_id}.log",
            retry_count=DEFAULT_STAGE_RETRY_COUNT,
            label=task_id,
            log_index_path=state.log_index_path,
            work_units=1,
        )

    def build_verify_group_task(
        state: SearchState,
        grouped_items: list[tuple[mechanism_stage_utils.RankedCandidate, int]],
    ) -> SchedulerTask:
        first_ranked, repeat_id = grouped_items[0]
        tao2_token = mechanism_stage_utils.path_token(first_ranked.candidate.tao2)
        task_id = (
            f"verify_group_repeat={int(repeat_id):02d}__xsteps={int(first_ranked.candidate.x_steps)}__tao2={tao2_token}"
        )
        command = [
            stage_runner_python,
            str(grouped_stage_script),
            str(state.job.job_dir),
            "--stage",
            "verify",
            "--repeat_id",
            str(repeat_id),
        ]
        for ranked, _ in grouped_items:
            command.extend(["--candidate_id", str(ranked.candidate_id)])
        return SchedulerTask(
            task_id=task_id,
            combo_key=state.job.job_id,
            pool="verify",
            command=command,
            env={},
            cwd=repo_root,
            log_path=state.logs_dir / f"{task_id}.log",
            retry_count=DEFAULT_STAGE_RETRY_COUNT,
            label=task_id,
            log_index_path=state.log_index_path,
            work_units=len(grouped_items),
        )

    def build_finalize_task(state: SearchState) -> SchedulerTask:
        return SchedulerTask(
            task_id="verify_finalize",
            combo_key=state.job.job_id,
            pool="verify",
            command=[stage_runner_python, str(stage_scripts["verify_finalize"]), str(state.job.job_dir)],
            env={},
            cwd=repo_root,
            log_path=state.logs_dir / "verify_finalize.log",
            retry_count=DEFAULT_STAGE_RETRY_COUNT,
            label="verify_finalize",
            log_index_path=state.log_index_path,
        )

    for state in states:
        job_spec = state.job.job_spec
        missing_grid_ids = [chunk for chunk in state.row.pop("_missing_grid_ids").split(",") if chunk]
        initial_rank = state.row.pop("_initial_rank") == "true"
        initial_finalize = state.row.pop("_initial_finalize") == "true"
        initial_verify_work = [chunk for chunk in state.row.pop("_initial_verify_work").split(",") if chunk]

        if len(missing_grid_ids) > 0:
            missing_candidates = [
                mechanism_stage_utils.job_candidate_by_id(job_spec, int(candidate_id_text))
                for candidate_id_text in missing_grid_ids
            ]
            if use_grouped_state_runner:
                for grouped_candidates in _group_grid_candidates(missing_candidates):
                    scheduler.enqueue(build_grid_group_task(state, grouped_candidates))
            else:
                for candidate in missing_candidates:
                    scheduler.enqueue(build_grid_task(state, candidate))
            continue

        if initial_rank:
            scheduler.enqueue(build_rank_task(state))
            continue

        if len(initial_verify_work) > 0:
            pending_verify_items: list[tuple[mechanism_stage_utils.RankedCandidate, int]] = []
            for item in initial_verify_work:
                rank_text, repeat_text = item.split(":", 1)
                ranked = mechanism_stage_utils.ranked_candidate_by_rank(
                    mechanism_stage_utils.verify_topk_path(state.job.job_dir),
                    int(rank_text),
                )
                pending_verify_items.append((ranked, int(repeat_text)))
            if use_grouped_state_runner:
                for grouped_items in _group_verify_work(pending_verify_items):
                    scheduler.enqueue(build_verify_group_task(state, grouped_items))
            else:
                for ranked, repeat_id in pending_verify_items:
                    scheduler.enqueue(build_verify_task(state, ranked, int(repeat_id)))
            continue

        if initial_finalize:
            scheduler.enqueue(build_finalize_task(state))

    def mark_failed(state: SearchState, message: str) -> None:
        nonlocal failed_jobs
        if state.failed:
            return

        state.failed = True
        state.row["search_status"] = "failed"
        state.row["error_message"] = " ".join(str(message).split())
        failed_jobs += 1

        remaining = max(0, state.training_total - state.training_accounted)
        if remaining > 0:
            progress.mark_failed(count=remaining)
            state.training_accounted += remaining

        scheduler.cancel_combo(state.job.job_id)
        _write_manifest(manifest_path, list(rows_by_job_id.values()))
        progress.log(f"failed [{state.job.display_name}]: {state.row['error_message']}")

    def handle_result(result: TaskResult) -> list[SchedulerTask]:
        nonlocal completed_jobs

        state = state_by_job_id[result.task.combo_key]
        _accumulate_metrics(state.row, result.attempts, result.total_seconds)
        if state.failed:
            return []

        if not result.success:
            mark_failed(state, result.error_message or f"{result.task.label} failed")
            return []

        if result.task.task_id.startswith("grid_candidate_") or result.task.task_id.startswith("grid_group_"):
            completed_units = int(result.task.work_units)
            progress.mark_completed(count=completed_units)
            state.training_accounted += completed_units
            state.grid_completed_runtime += completed_units
            current_grid_done = int(state.row["grid_done"]) + completed_units
            state.row["grid_done"] = str(current_grid_done)
            if state.grid_completed_runtime == state.grid_pending_total:
                return [build_rank_task(state)]
            return []

        if result.task.task_id == "grid_rank":
            try:
                ranked = mechanism_stage_utils.load_ranked_candidates(
                    mechanism_stage_utils.verify_topk_path(state.job.job_dir)
                )
            except Exception as exc:  # noqa: BLE001
                mark_failed(state, f"failed to load verify_topk.csv: {exc}")
                return []

            missing_verify, existing_verify_done = _count_existing_verify(
                state.job.job_spec,
                state.job.job_dir,
                ranked,
            )
            already_done = int(state.row["verify_done"])
            if existing_verify_done > already_done:
                delta = existing_verify_done - already_done
                progress.mark_skipped(count=delta)
                state.training_accounted += delta
                state.row["verify_done"] = str(existing_verify_done)

            state.verify_pending_total = len(missing_verify)
            if len(missing_verify) == 0:
                return [build_finalize_task(state)]

            if use_grouped_state_runner:
                return [build_verify_group_task(state, grouped_items) for grouped_items in _group_verify_work(missing_verify)]
            return [build_verify_task(state, ranked_candidate, repeat_id) for ranked_candidate, repeat_id in missing_verify]

        if result.task.task_id.startswith("verify_rank") or result.task.task_id.startswith("verify_group_"):
            completed_units = int(result.task.work_units)
            progress.mark_completed(count=completed_units)
            state.training_accounted += completed_units
            state.verify_completed_runtime += completed_units
            current_verify_done = int(state.row["verify_done"]) + completed_units
            state.row["verify_done"] = str(current_verify_done)
            if state.verify_completed_runtime == state.verify_pending_total:
                return [build_finalize_task(state)]
            return []

        if result.task.task_id == "verify_finalize":
            try:
                _load_best_outputs_into_row(state.row, state.job.job_dir)
            except Exception as exc:  # noqa: BLE001
                mark_failed(state, f"failed to parse final outputs: {exc}")
                return []

            state.row["search_status"] = "completed"
            state.row["error_message"] = ""
            completed_jobs += 1
            _write_manifest(manifest_path, list(rows_by_job_id.values()))
            return []

        mark_failed(state, f"unexpected task id: {result.task.task_id}")
        return []

    scheduler.run(handle_result)

    _write_manifest(manifest_path, list(rows_by_job_id.values()))
    progress.finalize()
    print()
    print(f"Batch finished: completed={completed_jobs} skipped={skipped_jobs} failed={failed_jobs}")
    print(f"Manifest: {manifest_path}")
    return completed_jobs, skipped_jobs, failed_jobs, manifest_path
