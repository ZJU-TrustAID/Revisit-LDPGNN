#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from hparams_search_scripts import mechanism_stage_utils
    from hparams_search_scripts import run_mechanism_hparam_search as search_runner
except ModuleNotFoundError:
    import mechanism_stage_utils  # type: ignore
    import run_mechanism_hparam_search as search_runner  # type: ignore

from datasets import load_dataset, resolve_dataset_name
from transforms import load_or_build_operator_normalized_adj


REPO_LOCAL_FEATURE_REWRITE_ROOT = REPO_ROOT
REPO_LOCAL_FEATURE_REWRITE_SRC = REPO_ROOT / "src"
_repo_local_feature_rewrite_module = None


@dataclass(frozen=True)
class CacheTask:
    dataset: str
    feature: str
    params: dict[str, Any]
    rewrite_seed: int | None
    effective_seed: int | None
    source_configs: tuple[str, ...]


@dataclass(frozen=True)
class TaskResult:
    task: CacheTask
    status: str
    cache_path: str | None
    elapsed_sec: float
    detail: str | None = None


def _module_origin_path(module_name: str, module: Any) -> Path:
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        raise RuntimeError(f"Imported {module_name} has no __file__; unable to verify import origin.")
    return Path(module_path).resolve()


def _assert_module_under_repo_root(module_name: str, module: Any) -> None:
    module_path = _module_origin_path(module_name, module)
    repo_root = REPO_LOCAL_FEATURE_REWRITE_ROOT.resolve()
    if module_path != repo_root and repo_root not in module_path.parents:
        raise RuntimeError(
            f"Imported {module_name} from unexpected location: {module_path}. "
            f"Expected path under {repo_root}."
        )


def _purge_modules(prefix: str) -> None:
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(f"{prefix}."):
            del sys.modules[module_name]


def _load_repo_local_feature_rewrite_module():
    global _repo_local_feature_rewrite_module

    if _repo_local_feature_rewrite_module is not None:
        return _repo_local_feature_rewrite_module

    if not REPO_LOCAL_FEATURE_REWRITE_ROOT.is_dir():
        raise RuntimeError(
            f"artificial-node-feature-generator repository root not found at {REPO_LOCAL_FEATURE_REWRITE_ROOT}. "
            "Please initialize/update the repository checkout first."
        )
    if not REPO_LOCAL_FEATURE_REWRITE_SRC.is_dir():
        raise RuntimeError(
            f"artificial-node-feature-generator source directory not found at {REPO_LOCAL_FEATURE_REWRITE_SRC}. "
            "Expected a repo-local source tree."
        )

    repo_local_src = str(REPO_LOCAL_FEATURE_REWRITE_SRC)
    if repo_local_src in sys.path:
        sys.path.remove(repo_local_src)
    sys.path.insert(0, repo_local_src)

    _purge_modules("artificial_node_feature_generator")

    try:
        module = importlib.import_module("artificial_node_feature_generator")
    except Exception as exc:
        raise ImportError(
            "Failed to import artificial_node_feature_generator from repo-local source path. "
            f"Expected path: {REPO_LOCAL_FEATURE_REWRITE_SRC}"
        ) from exc

    _assert_module_under_repo_root("artificial_node_feature_generator", module)
    if not hasattr(module, "rewrite_features"):
        raise RuntimeError("artificial_node_feature_generator is missing expected attribute: rewrite_features")

    _repo_local_feature_rewrite_module = module
    return _repo_local_feature_rewrite_module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute repo-local artificial feature caches for a suite of YAML-driven search jobs "
            "without launching GPU training."
        )
    )
    parser.add_argument(
        "--suite-script",
        type=str,
        default=None,
        help="shell script whose lines contain `python -m ... --config <yaml>` commands",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="search configuration YAML (may be passed multiple times)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(12, os.cpu_count() or 1),
        help="maximum number of CPU worker processes",
    )
    parser.add_argument(
        "--max-eigen-dense-gib",
        type=float,
        default=8.0,
        help=(
            "skip eigen/eigen_norm cache jobs whose dense adjacency matrix estimate exceeds this size; "
            "use a negative value to disable the guard"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild caches even when the cache file already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate cache tasks and exit without generating caches",
    )
    return parser.parse_args()


def _read_suite_script_config_paths(path: Path) -> list[Path]:
    config_paths: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        parts = shlex.split(stripped)
        for index, part in enumerate(parts):
            if part == "--config" and index + 1 < len(parts):
                config_paths.append((path.parent / parts[index + 1]).resolve())
                break
    return config_paths


def _collect_config_paths(args: argparse.Namespace) -> list[Path]:
    config_paths: list[Path] = []
    if args.suite_script is not None:
        suite_path = Path(args.suite_script).resolve()
        config_paths.extend(_read_suite_script_config_paths(suite_path))
    for raw in args.config:
        config_paths.append(Path(raw).resolve())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in config_paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    if len(unique) == 0:
        raise ValueError("at least one --config or --suite-script must be provided")
    return unique


def _is_rewrite_feature(feature: str) -> bool:
    return feature in set(search_runner.SUPPORTED_REWRITE_FEATURES)


def _rewrite_params_from_fixed_params(
    fixed_params: dict[str, Any],
    *,
    candidate_x_steps: int | None = None,
) -> dict[str, Any]:
    feature = str(fixed_params["feature"])
    params: dict[str, Any] = {}

    if fixed_params.get("feature_dim") is not None:
        params["feature_dim"] = int(fixed_params["feature_dim"])

    if feature == "random_normal":
        if fixed_params.get("random_normal_mean") is not None:
            params["mean"] = float(fixed_params["random_normal_mean"])
        if fixed_params.get("random_normal_std") is not None:
            params["std"] = float(fixed_params["random_normal_std"])
    elif feature == "shared":
        if fixed_params.get("shared_value") is not None:
            params["value"] = float(fixed_params["shared_value"])
    elif feature == "degree_bucket_range":
        if fixed_params.get("degree_bucket_num_buckets") is not None:
            params["num_buckets"] = int(fixed_params["degree_bucket_num_buckets"])
        if fixed_params.get("degree_bucket_range_max") is not None:
            params["range_max"] = int(fixed_params["degree_bucket_range_max"])
    elif feature == "degree_bucket_distribution":
        if fixed_params.get("degree_bucket_num_buckets") is not None:
            params["num_buckets"] = int(fixed_params["degree_bucket_num_buckets"])
    elif feature == "deepwalk":
        if fixed_params.get("deepwalk_walk_length") is not None:
            params["walk_length"] = int(fixed_params["deepwalk_walk_length"])
        if fixed_params.get("deepwalk_number_walks") is not None:
            params["number_walks"] = int(fixed_params["deepwalk_number_walks"])
        if fixed_params.get("deepwalk_window_size") is not None:
            params["window_size"] = int(fixed_params["deepwalk_window_size"])
        if fixed_params.get("deepwalk_workers") is not None:
            params["workers"] = int(fixed_params["deepwalk_workers"])
        if fixed_params.get("deepwalk_undirected") is not None:
            params["undirected"] = bool(fixed_params["deepwalk_undirected"])
    elif feature == "operator":
        if candidate_x_steps is None:
            raise RuntimeError("operator cache precompute requires candidate_x_steps")
        params["x_steps"] = int(candidate_x_steps)

    return params


def _rewrite_seeds_for_job(job_spec: dict[str, Any]) -> list[int | None]:
    try:
        base_seed = int(job_spec["base_seed"])
        verify_repeats = int(job_spec["defaults"]["stage"]["verify"]["repeats"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("job_spec is missing base_seed or defaults.stage.verify.repeats") from exc
    return [base_seed + offset for offset in range(verify_repeats)]


def _task_key(dataset: str, feature: str, params: dict[str, Any], effective_seed: int | None) -> str:
    return json.dumps(
        {
            "dataset": dataset,
            "feature": feature,
            "params": params,
            "effective_seed": effective_seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _collect_cache_tasks(config_paths: list[Path]) -> list[CacheTask]:
    _load_repo_local_feature_rewrite_module()
    from artificial_node_feature_generator.registry import get_provider

    task_map: dict[str, CacheTask] = {}
    for config_path in config_paths:
        search_config = search_runner.load_search_config(config_path)
        batch_spec = search_runner.build_batch_spec(
            search_config=search_config,
            output_root=_repo_root() / ".cache_precompute_probe" / config_path.stem,
            config_copy_source=config_path,
        )

        for job in batch_spec.jobs:
            fixed_params = job.job_spec["fixed_params"]
            if not isinstance(fixed_params, dict):
                raise RuntimeError("job_spec.fixed_params must be a mapping")

            feature = str(fixed_params["feature"])
            if not _is_rewrite_feature(feature):
                continue

            provider = get_provider(feature)
            if not bool(getattr(provider, "cacheable", False)):
                continue

            dataset_name = resolve_dataset_name(str(fixed_params["dataset"]))
            if feature == "operator":
                candidate_x_steps_values = [None]
                rewrite_seeds = [None]
            else:
                candidate_x_steps_values = [None]
                rewrite_seeds = _rewrite_seeds_for_job(job.job_spec)

            for candidate_x_steps in candidate_x_steps_values:
                if feature == "operator":
                    params = {"cache_kind": "normalized_adjacency"}
                else:
                    params = _rewrite_params_from_fixed_params(
                        fixed_params,
                        candidate_x_steps=candidate_x_steps,
                    )
                for rewrite_seed in rewrite_seeds:
                    effective_seed = provider.cache_seed(seed=rewrite_seed, params=params)
                    key = _task_key(dataset_name, feature, params, effective_seed)
                    if key not in task_map:
                        task_map[key] = CacheTask(
                            dataset=dataset_name,
                            feature=feature,
                            params=dict(params),
                            rewrite_seed=rewrite_seed,
                            effective_seed=effective_seed,
                            source_configs=(str(config_path),),
                        )
                    else:
                        merged_sources = tuple(
                            sorted(set(task_map[key].source_configs) | {str(config_path)})
                        )
                        task_map[key] = CacheTask(
                            dataset=task_map[key].dataset,
                            feature=task_map[key].feature,
                            params=dict(task_map[key].params),
                            rewrite_seed=task_map[key].rewrite_seed,
                            effective_seed=task_map[key].effective_seed,
                            source_configs=merged_sources,
                        )

    tasks = list(task_map.values())
    tasks.sort(
        key=lambda task: (
            task.feature,
            task.dataset.casefold(),
            json.dumps(task.params, sort_keys=True),
            -1 if task.effective_seed is None else int(task.effective_seed),
        )
    )
    return tasks


def _estimate_dense_matrix_gib(num_nodes: int) -> float:
    dense_bytes = int(num_nodes) * int(num_nodes) * 8
    return dense_bytes / (1024 ** 3)


def _run_task(task: CacheTask, *, force: bool, max_eigen_dense_gib: float) -> TaskResult:
    started = time.time()
    data = load_dataset(task.dataset)
    if task.feature == "operator":
        _, cache_path, built = load_or_build_operator_normalized_adj(data, force=force)
        return TaskResult(
            task=task,
            status="built" if built else "hit",
            cache_path=str(cache_path),
            elapsed_sec=time.time() - started,
        )

    _load_repo_local_feature_rewrite_module()
    from artificial_node_feature_generator.cache import feature_cache_path, save_cached_features
    from artificial_node_feature_generator.graph import graph_fingerprint
    from artificial_node_feature_generator.registry import get_provider

    provider = get_provider(task.feature)
    graph_key = graph_fingerprint(data)
    cache_path = feature_cache_path(
        feature=task.feature,
        seed=task.effective_seed,
        params=task.params,
        graph_key=graph_key,
    )
    if cache_path.is_file() and not force:
        return TaskResult(
            task=task,
            status="hit",
            cache_path=str(cache_path),
            elapsed_sec=time.time() - started,
        )

    output = provider.build(
        data,
        params=task.params,
        seed=task.rewrite_seed,
    )
    save_cached_features(
        cache_path,
        features=output.features,
        params=task.params,
        seed=task.effective_seed,
        graph_key=graph_key,
    )
    return TaskResult(
        task=task,
        status="built",
        cache_path=str(cache_path),
        elapsed_sec=time.time() - started,
    )


def _format_task(task: CacheTask) -> str:
    params_text = json.dumps(task.params, sort_keys=True, separators=(",", ":"))
    return (
        f"dataset={task.dataset}, feature={task.feature}, params={params_text}, "
        f"rewrite_seed={task.rewrite_seed}, effective_seed={task.effective_seed}"
    )


def main() -> int:
    args = _parse_args()
    try:
        config_paths = _collect_config_paths(args)
        tasks = _collect_cache_tasks(config_paths)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Configs: {len(config_paths)}")
    for path in config_paths:
        print(f"  - {path}")
    print(f"Unique cache tasks: {len(tasks)}")
    if len(tasks) == 0:
        return 0

    feature_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    for task in tasks:
        feature_counts[task.feature] = feature_counts.get(task.feature, 0) + 1
        dataset_counts[task.dataset] = dataset_counts.get(task.dataset, 0) + 1

    print("By feature:")
    for feature_name in sorted(feature_counts):
        print(f"  - {feature_name}: {feature_counts[feature_name]}")
    print("By dataset:")
    for dataset_name in sorted(dataset_counts, key=str.casefold):
        print(f"  - {dataset_name}: {dataset_counts[dataset_name]}")

    if args.dry_run:
        print("Dry run tasks:")
        for task in tasks:
            print(f"  - {_format_task(task)}")
        return 0

    workers = max(1, int(args.workers))
    if workers > len(tasks):
        workers = len(tasks)
    print(f"Workers: {workers}")
    print(f"Eigen dense threshold: {args.max_eigen_dense_gib} GiB")

    results: list[TaskResult] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(
                _run_task,
                task,
                force=bool(args.force),
                max_eigen_dense_gib=float(args.max_eigen_dense_gib),
            ): task
            for task in tasks
        }
        completed = 0
        total = len(future_to_task)
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = TaskResult(
                    task=task,
                    status="failed",
                    cache_path=None,
                    elapsed_sec=0.0,
                    detail=str(exc),
                )
            results.append(result)
            suffix = f" ({result.detail})" if result.detail else ""
            print(
                f"[{completed}/{total}] {result.status}: {_format_task(task)}"
                f" in {result.elapsed_sec:.2f}s{suffix}"
            )

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    print("Summary:")
    for status_name in sorted(status_counts):
        print(f"  - {status_name}: {status_counts[status_name]}")

    failed = [result for result in results if result.status == "failed"]
    if failed:
        print("Failures:")
        for result in failed:
            print(f"  - {_format_task(result.task)}: {result.detail}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
