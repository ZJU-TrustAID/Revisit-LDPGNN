from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from hparams_search_scripts import run_mechanism_hparam_search as search_impl
from hparams_search_scripts import table_search_suite_core as core
from hparams_search_scripts import mechanism_stage_utils
from .paths import parse_gpu_ids, max_parallel_per_gpu, WORK_ROOT, normalize_mode, result_root
from .table_settings import TABLE6_FEATURE_DIM, table6_feature_dims

REPO_ROOT=Path(__file__).resolve().parents[2]

PIPELINE_AXES = {
    "figure3_pipeline1": {"mechanism": "mbm", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline2": {"mechanism": "hds", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline3": {"mechanism": "mbm", "smoother": "hoa", "use_nfr": "true"},
    "figure3_pipeline4": {"mechanism": "pm", "smoother": "hoa", "use_nfr": "true"},
}


def _configs(figure_id: int) -> list[Path]:
    roots={1:REPO_ROOT/"configs_AEC/figure1",3:REPO_ROOT/"configs_AEC/figure3",4:REPO_ROOT/"configs_AEC/figure4",5:REPO_ROOT/"configs_AEC/figure5",6:REPO_ROOT/"configs_AEC/figure6",7:REPO_ROOT/"configs_AEC/figure7","table4":REPO_ROOT/"configs_AEC/table4","table6":REPO_ROOT/"configs_AEC/table6"}
    if figure_id == 5:
        # Figure 5 is one tao2 sweep. Keep its AEC inputs flat; the old
        # figure6/figure6_add/figure6_add_again directories were author-side
        # export batches and are not part of the runnable configuration.
        return sorted((REPO_ROOT / "configs_AEC/figure5").glob("tao2=*.yaml"))
    return sorted(roots[figure_id].rglob("*.yaml")) if figure_id in roots else []

def _claim_paths(figure_id):
    root=REPO_ROOT
    if figure_id==1:
        paths=[]
        paths += sorted((root/"configs_AEC/figure1/FeatFree/cora").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/FeatFree/facebook").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/LDPGNN").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure1/Non-private").rglob("*.yaml"))
        return paths
    if figure_id==6:
        paths=[]
        paths += sorted((root/"configs_AEC/figure6/FeatFree/actor/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/FeatFree/flickr/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/LDPGNN/sage").rglob("*.yaml"))
        paths += sorted((root/"configs_AEC/figure6/Non-private").rglob("*.yaml"))
        return paths
    if figure_id==5: return _configs(5)
    return _configs(figure_id)

def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle: return yaml.safe_load(handle)

def _config_token(path: Path) -> str:
    return path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "__")


def _search_root_name(figure_id: int | str) -> str:
    if isinstance(figure_id, str) and figure_id.startswith("table"):
        return figure_id
    return f"figure{figure_id}"


def _runtime_config_data(
    path: Path, *, mode: str, configure_devices: bool, feature_dims: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    mode = normalize_mode(mode)
    if mode not in {"scaled", "full"}:
        raise ValueError(f"Search mode must be scaled or full: {mode}")
    data = load_config(path)
    if feature_dims is not None:
        data["search_space"]["feature_transformation"]["feature_dim"] = list(feature_dims)
    datasets = data.get("search_space", {}).get("dataset", {}).get("datasets", [])
    path_text = path.as_posix().lower()
    if mode == "scaled":
        if "configs_aec/figure1" in path_text:
            data["search_space"]["dataset"]["datasets"] = [
                value for value in datasets if str(value).lower() in {"cora", "facebook"}
            ]
        elif "configs_aec/figure6" in path_text:
            data["search_space"]["dataset"]["datasets"] = [
                value for value in datasets if str(value).lower() in {"actor", "flickr"}
            ]
        # Preserve each configuration's verification repeat count.

    device = data["device"]
    if device["device"] == "gpu":
        if configure_devices:
            device["gpu_ids"] = parse_gpu_ids()
            device["max_parallel_per_gpu"] = max_parallel_per_gpu(device["gpu_ids"])
        device["gpu_launch_interval_sec"] = 0.01
    return data


def _runtime_config(
    path: Path, *, mode: str, output: Path, feature_dims: tuple[int, ...] | None = None,
) -> Path:
    data = _runtime_config_data(path, mode=mode, configure_devices=True, feature_dims=feature_dims)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output


def scaled_config(path: Path, *, output: Path) -> Path:
    return _runtime_config(path, mode="scaled", output=output)


def _build_runtime_batch(
    config: Path,
    *,
    mode: str,
    output_root: Path,
    feature_dims: tuple[int, ...] | None = None,
) -> tuple[Path, core.BatchSpec]:
    token = _config_token(config)
    actual = _runtime_config(
        config,
        mode=mode,
        output=WORK_ROOT / "runtime_configs" / mode / f"{token}.yaml",
        feature_dims=feature_dims,
    )
    search_config = search_impl.load_search_config(actual)
    batch = search_impl.build_batch_spec(
        search_config=search_config,
        output_root=output_root / token,
        config_copy_source=actual,
    )
    return actual, batch


def run_search(
    config: Path,
    *,
    mode: str = "scaled",
    output_root: Path | None = None,
    dry_run: bool = True,
) -> dict[str, object]:
    output_root = output_root or WORK_ROOT / "search" / config.stem / mode
    actual, batch = _build_runtime_batch(config, mode=mode, output_root=output_root.parent)
    result = {
        "mode": mode,
        "config": str(actual),
        "output_root": str(batch.output_root),
        "jobs_planned": len(batch.jobs),
        "training_tasks_planned": sum(
            len(job.job_spec["candidates"])
            + min(
                mechanism_stage_utils.VERIFY_TOPK,
                len(job.job_spec["candidates"]),
            )
            * int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])
            for job in batch.jobs
        ),
        "dry_run": dry_run,
    }
    if not dry_run:
        completed, skipped, failed, manifest = core.run_batch_search(
            batch,
            repo_root=REPO_ROOT,
        )
        result.update(
            completed=completed,
            skipped=skipped,
            failed=failed,
            manifest=str(manifest),
        )
    return result

def _select_paths(figure_id, paths, mode):
    mode = normalize_mode(mode)
    if mode not in {"scaled", "full"}:
        raise ValueError(f"Search mode must be scaled or full: {mode}")
    if mode == "full":
        return paths
    selected=[]
    for path in paths:
        text=path.as_posix().lower()
        if figure_id == 1:
            if "ldpgnn" in text or ("featfree" in text and any(f"/{d}/" in text for d in ("cora","facebook"))) or "non-private" in text:
                selected.append(path)
        elif figure_id == 3:
            if path.name in {"LDP.yaml","SIM.yaml"}: selected.append(path)
        elif figure_id == 5:
            selected.append(path)
        elif figure_id == 6:
            if "ldpgnn/sage" in text or ("featfree" in text and "/sage/" in text) or "non-private" in text:
                selected.append(path)
        elif figure_id in {4,7,"table4","table6"}:
            selected.append(path)
    if not selected:
        raise ValueError(f"No search configurations selected for {figure_id} mode={mode}")
    return selected


def _planned_task_count(batch: core.BatchSpec) -> int:
    total = 0
    for job in batch.jobs:
        candidate_count = len(job.job_spec["candidates"])
        repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])
        total += candidate_count + min(mechanism_stage_utils.VERIFY_TOPK, candidate_count) * repeats
    return total


def _table_setting_from_config(path: Path) -> str:
    # Configs are stored as table4_FeatFree-P/... and
    # table6_FeatFree-HOA/...; the backbone directory is not the table setting.
    for part in path.parts:
        for prefix in ("table4_", "table6_"):
            if part.startswith(prefix):
                return part[len(prefix):]
    raise ValueError(f"Cannot derive table setting from config path: {path}")


@dataclass(frozen=True)
class SearchPlan:
    target: int | str
    mode: str
    batch: core.BatchSpec
    sources: dict[str, Path]
    configs: list[dict[str, object]]
    feature_dims: tuple[int, ...] | None = None


def build_search_plan(
    figure_id: int | str, mode: str, *, materialize: bool = False,
    feature_dim: int | list[int] = TABLE6_FEATURE_DIM,
) -> SearchPlan:
    mode = normalize_mode(mode)
    feature_dims = table6_feature_dims(feature_dim) if figure_id == "table6" else None
    paths = _configs(figure_id)
    selected = _select_paths(figure_id, paths, mode)
    root = result_root(figure_id, mode)
    jobs: list[core.BatchJob] = []
    execution = None
    config_results: list[dict[str, object]] = []
    sources: dict[str, Path] = {}

    for config in selected:
        if materialize:
            actual, batch = _build_runtime_batch(config, mode=mode, output_root=root, feature_dims=feature_dims)
        else:
            actual = config
            data = _runtime_config_data(config, mode=mode, configure_devices=False, feature_dims=feature_dims)
            parsed = search_impl.parse_search_config(data, source=str(config))
            batch = search_impl.build_batch_spec(
                search_config=parsed, output_root=root / _config_token(config), config_copy_source=config,
            )
        if execution is None:
            execution = batch.execution
        elif materialize and execution != batch.execution:
            raise RuntimeError(
                f"Selected configs do not share one execution pool: {config}"
            )
        # Give each configuration its own job_id prefix to keep candidate spaces independent.
        namespaced_jobs = []
        config_token = _config_token(config)
        for job in batch.jobs:
            namespaced_id = f"{config_token}__{job.job_id}"
            if namespaced_id in sources:
                raise ValueError(f"Duplicate search job_id: {namespaced_id}")
            sources[namespaced_id] = config
            job_spec = dict(job.job_spec)
            job_spec["job_id"] = namespaced_id
            namespaced_jobs.append(
                core.BatchJob(
                    job_id=namespaced_id,
                    job_dir=job.job_dir,
                    display_name=job.display_name,
                    job_spec=job_spec,
                )
            )
        jobs.extend(namespaced_jobs)
        config_results.append(
            {
                "source_config": str(config),
                "runtime_config": str(actual),
                "jobs_planned": len(batch.jobs),
                "training_tasks_planned": _planned_task_count(batch),
            }
        )

    if execution is None or not jobs:
        raise ValueError(f"No search jobs for {figure_id} mode={mode}")

    batch = core.BatchSpec(
        output_root=root,
        execution=execution,
        jobs=jobs,
        config_copy_source=Path(config_results[0]["runtime_config"]),
    )
    return SearchPlan(figure_id, mode, batch, sources, config_results, feature_dims)


def run_search_for_figure(
    figure_id: int | str,
    *,
    mode: str = "scaled",
    execute: bool = False,
    feature_dim: int | list[int] = TABLE6_FEATURE_DIM,
) -> dict[str, object]:
    mode = normalize_mode(mode)
    if mode not in {"scaled", "full"}:
        raise ValueError(f"Search mode must be scaled or full: {mode}")
    if figure_id in (2, 8):
        return {"figure_id": figure_id, "mode": "analytic", "configs": 0}
    from .search_results import collect_search_outputs, validate_search_plan

    plan = build_search_plan(figure_id, mode, materialize=True, feature_dim=feature_dim)
    validate_search_plan(plan)
    result: dict[str, object] = {
        "figure_id": figure_id,
        "mode": mode,
        "configs_total": len(_configs(figure_id)),
        "configs_selected": len(plan.configs),
        "jobs_planned": len(plan.batch.jobs),
        "training_tasks_planned": _planned_task_count(plan.batch),
        "configs": plan.configs,
        "execute": execute,
    }
    if plan.feature_dims is not None:
        result["feature_dims"] = list(plan.feature_dims)
    if execute:
        completed, skipped, failed, manifest = core.run_batch_search(
            plan.batch,
            repo_root=REPO_ROOT,
        )
        result.update(
            completed=completed,
            skipped=skipped,
            failed=failed,
            manifest=str(manifest),
        )
        if failed:
            raise RuntimeError(f"Search failed for {failed} job(s); see {manifest}")
        output = collect_search_outputs(figure_id, mode, plan=plan)
        result["seed_rows" if isinstance(figure_id, str) else "plot_data"] = str(output)
    return result
