from __future__ import annotations

import csv
import math
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hparams_search_scripts import run_mechanism_hparam_search as search_impl
from hparams_search_scripts import table_search_suite_core as core
from hparams_search_scripts import mechanism_stage_utils
from hparams_search_scripts.gpu_resources import GpuConcurrency, resolve_gpu_concurrency

from .paths import FIXED_ROOT, REFERENCE_ROOT, WORK_ROOT, max_parallel_per_gpu, parse_gpu_ids
from .result_io import read_csv
from .table_settings import TABLE6_FEATURE_DIM

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _FixedPointJob:
    point: dict[str, Any]
    job: core.BatchJob


def fixed_points(figure_id: int | str) -> list[dict[str, Any]]:
    path = FIXED_ROOT / f"figure{figure_id}.yaml"
    if not path.is_file():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("points", []))


def plan_fixed_jobs(
    figure_id: int,
    *,
    repeats: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    repeats = reference_repeats(figure_id) if repeats is None else repeats
    points = fixed_points(figure_id)
    jobs = [
        {
            "figure_id": figure_id,
            "point_id": point.get("point_id"),
            "dataset": point.get("fixed_params", {}).get("dataset"),
            "backbone": point.get("fixed_params", {}).get("backbone"),
            "mechanism": point.get("fixed_params", {}).get("mechanism"),
            "x_eps": point.get("fixed_params", {}).get("x_eps"),
            "repeats": repeats,
        }
        for point in points
    ]
    return jobs if limit is None else jobs[:limit]


def reference_repeats(target: int | str) -> int:
    """Return the repeat count used by the corresponding paper reference."""
    if target in {2, 8}:
        raise ValueError(
            f"Figure {target} is analytic and has no fixed-training reference repeats; "
            "render it with scripts.aec.figures.render_reference."
        )
    if isinstance(target, str) and target.startswith("table"):
        path = REFERENCE_ROOT / f"{target}_seed_rows.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError(f"Reference table is empty: {path}")
        # Every fixed table point in the published reference uses the same
        # seed count; count one complete setting rather than guessing a value.
        first_key = tuple(rows[0].get(key, "") for key in ("setting", "dataset", "backbone", "feature_dim"))
        count = sum(
            tuple(row.get(key, "") for key in ("setting", "dataset", "backbone", "feature_dim")) == first_key
            for row in rows
        )
        return int(count)

    path = REFERENCE_ROOT / f"figure{target}_plot_data.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        values = {int(row["n"]) for row in csv.DictReader(handle) if row.get("n")}
    if len(values) != 1:
        raise RuntimeError(f"Reference Figure {target} has inconsistent repeat counts: {sorted(values)}")
    return values.pop()


def _one_candidate_config(
    point: dict[str, Any],
    repeats: int,
    gpu_ids: list[int] | None = None,
    max_parallel: GpuConcurrency | None = None,
) -> dict[str, Any]:
    fixed = point.get("fixed_params", point)
    candidate = point.get("candidate", {})
    defaults = point.get("defaults", {}) or {}
    feature = str(fixed.get("feature", "raw"))

    feature_cfg: dict[str, list[Any]] = {
        key: []
        for key in (
            "sim_reference_eps",
            "feature_dim",
            "scale",
            "feature_preprojection",
            "preprojection_output_dim",
            "random_normal_mean",
            "random_normal_std",
            "shared_value",
            "degree_bucket_num_buckets",
            "degree_bucket_range_max",
            "deepwalk_walk_length",
            "deepwalk_number_walks",
            "deepwalk_window_size",
            "deepwalk_workers",
            "deepwalk_undirected",
        )
    }
    feature_cfg["features"] = [feature]
    scale = fixed.get("scale")
    feature_cfg["scale"] = [1 if scale is None else scale]
    for key in feature_cfg:
        if key not in {"features", "scale"} and fixed.get(key) is not None:
            feature_cfg[key] = [fixed[key]]
    if feature == "sim":
        feature_cfg["sim_reference_eps"] = [fixed.get("sim_reference_eps")]

    perturbation = {
        "mechanisms": [fixed.get("mechanism")],
        "x_eps": [fixed.get("x_eps")],
        "m": [fixed.get("m", "best")],
    }
    norm = bool(fixed.get("norm", False))
    calibrator = {
        "norm": [norm],
        "norm_scale": [fixed.get("norm_scale", "none")] if norm else [],
        "x_steps": [candidate.get("x_steps", 0)],
        "smoother": [
            fixed.get("smoother")
            if fixed.get("smoother") not in (None, "none", "")
            else "hoa"
        ],
    }
    use_nfr = bool(fixed.get("use_nfr", False))
    nfr = {
        "use_nfr": [use_nfr],
        "tao2": [candidate.get("tao2")] if use_nfr else [],
    }

    selected_gpu_ids = parse_gpu_ids() if gpu_ids is None else list(gpu_ids)
    selected_parallel = (
        max_parallel_per_gpu(selected_gpu_ids) if max_parallel is None
        else resolve_gpu_concurrency(selected_gpu_ids, max_parallel)
    )
    device = {
        "device": "gpu",
        "cpu_worker_count": None,
        "gpu_ids": selected_gpu_ids,
        "max_parallel_per_gpu": selected_parallel,
        "gpu_launch_interval_sec": 0.01,
    }

    clean_defaults = dict(defaults)
    clean_stage = dict(clean_defaults.get("stage", {}))
    clean_grid = dict(clean_stage.get("grid", {}))
    clean_verify = dict(clean_stage.get("verify", {}))
    clean_grid.pop("max_epochs", None)
    clean_verify.pop("max_epochs", None)
    clean_grid["repeats"] = 1
    clean_verify["repeats"] = repeats
    clean_stage["grid"] = clean_grid
    clean_stage["verify"] = clean_verify
    clean_defaults["stage"] = clean_stage

    return {
        "seed": int(point.get("base_seed", 12345)),
        "device": device,
        "defaults": clean_defaults,
        "search_space": {
            "dataset": {"datasets": [fixed.get("dataset")]},
            "feature_transformation": feature_cfg,
            "feature_perturbation": perturbation,
            "calibrator": calibrator,
            "model": {
                "backbones": [fixed.get("backbone")],
                "dropout": [candidate.get("dropout", 0.5)],
            },
            "trainer": {
                "learning_rate": [candidate.get("learning_rate", 0.001)],
                "weight_decay": [candidate.get("weight_decay", 0.0)],
            },
            "nfr": nfr,
        },
    }


def _write_point_configs(
    points: list[dict[str, Any]],
    *,
    root: Path,
    repeats: int,
    device: dict[str, Any] | None = None,
) -> list[Path]:
    config_root = root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    configs: list[Path] = []
    if not points:
        return configs
    gpu_ids = parse_gpu_ids() if device is None else parse_gpu_ids(device["gpu_ids"])
    parallel = max_parallel_per_gpu(gpu_ids) if device is None else device["max_parallel_per_gpu"]
    for index, point in enumerate(points):
        path = config_root / f"point_{index:05d}.yaml"
        config = _one_candidate_config(point, repeats, gpu_ids=gpu_ids, max_parallel=parallel)
        if device is not None:
            config["device"] = dict(device)
        path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        configs.append(path)
    return configs


def _build_fixed_batch(
    points: list[dict[str, Any]],
    *,
    root: Path,
    repeats: int,
    config_root: Path | None = None,
    device: dict[str, Any] | None = None,
) -> tuple[core.BatchSpec | None, list[Path], dict[str, _FixedPointJob]]:
    configs = _write_point_configs(points, root=config_root or root, repeats=repeats, device=device)
    if not configs:
        return None, configs, {}

    jobs: list[core.BatchJob] = []
    point_jobs: dict[str, _FixedPointJob] = {}
    existing_dirs: dict[str, Path] = {}
    manifest_path = root / mechanism_stage_utils.ROOT_MANIFEST_FILENAME
    if manifest_path.is_file():
        for record in read_csv(manifest_path):
            job_id = record["job_id"]
            directory = Path(record["job_dir"])
            if job_id in existing_dirs:
                raise ValueError(f"Duplicate job_id {job_id!r} in {manifest_path}")
            if not directory.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Job directory is outside the fixed result directory: {directory}")
            existing_dirs[job_id] = directory
    assigned_dirs: set[Path] = set()
    execution = None
    if len(points) != len(configs):
        raise RuntimeError(f"Fixed point/config count mismatch: points={len(points)} configs={len(configs)}")
    for index, (point, config) in enumerate(zip(points, configs)):
        search_config = search_impl.load_search_config(config)
        point_root = root / f"result_{index:05d}"
        point_batch = search_impl.build_batch_spec(
            search_config=search_config,
            output_root=point_root,
            config_copy_source=config,
        )
        if len(point_batch.jobs) != 1:
            raise RuntimeError(
                f"Fixed point {index} expanded to {len(point_batch.jobs)} jobs; expected exactly one"
            )
        if execution is None:
            execution = point_batch.execution
        elif execution != point_batch.execution:
            raise RuntimeError("Fixed-point configurations disagree on execution settings")
        # point_id distinguishes points with identical fixed parameters but different candidate hyperparameters.
        point_id = str(point.get("point_id") or f"point_{index:05d}")
        for job in point_batch.jobs:
            namespaced_id = f"{point_id}__{job.job_id}"
            if namespaced_id in point_jobs:
                raise ValueError(f"Duplicate fixed job_id {namespaced_id!r} at point {index}")
            if len(mechanism_stage_utils.job_candidates(job.job_spec)) != 1:
                raise ValueError(f"Fixed point {point_id!r} must have exactly one candidate")
            job_spec = dict(job.job_spec)
            job_spec["job_id"] = namespaced_id
            directory = existing_dirs.get(namespaced_id, job.job_dir)
            if directory.resolve() in assigned_dirs:
                raise ValueError(f"Multiple fixed jobs use the same directory: {directory}")
            assigned_dirs.add(directory.resolve())
            fixed_job = core.BatchJob(
                job_id=namespaced_id,
                job_dir=directory,
                display_name=job.display_name,
                job_spec=job_spec,
            )
            jobs.append(fixed_job)
            # The full job_id also distinguishes HOA and Kprop entries sharing a point_id in Table 6.
            point_jobs[namespaced_id] = _FixedPointJob(point=point, job=fixed_job)

    assert execution is not None
    return (
        core.BatchSpec(
            output_root=root,
            execution=execution,
            jobs=jobs,
            config_copy_source=configs[0],
        ),
        configs,
        point_jobs,
    )


def _run_fixed_batch(
    points: list[dict[str, Any]],
    *,
    root: Path,
    repeats: int,
    execute: bool,
) -> dict[str, Any]:
    batch, configs, point_jobs = _build_fixed_batch(points, root=root, repeats=repeats)
    if batch is None:
        print("planned 0 fixed jobs; execute=False")
        return {"jobs": [], "configs": [], "point_jobs": {}, "completed": 0, "skipped": 0, "failed": 0}

    if not execute:
        print(
            f"planned {len(batch.jobs)} fixed jobs in one global task pool; execute=False"
        )
        return {
            "jobs": batch.jobs,
            "configs": configs,
            "point_jobs": point_jobs,
            "completed": 0,
            "skipped": 0,
            "failed": 0,
        }

    completed, skipped, failed, manifest = core.run_batch_search(
        batch,
        repo_root=REPO_ROOT,
    )
    summary = {
        "jobs": batch.jobs,
        "configs": configs,
        "point_jobs": point_jobs,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "manifest": manifest,
    }
    print(
        f"global fixed batch finished: jobs={len(batch.jobs)} "
        f"completed={completed} skipped={skipped} failed={failed}"
    )
    if failed:
        raise RuntimeError(f"Fixed batch failed for {failed} job(s); see {manifest}")
    return summary


def run_fixed(
    figure_id: int,
    *,
    repeats: int | None = None,
    limit: int | None = None,
    execute: bool = False,
) -> list[dict[str, Any]]:
    from .fixed_results import collect_fixed_output, figure_templates

    repeats = reference_repeats(figure_id) if repeats is None else repeats
    points = fixed_points(figure_id)
    jobs = plan_fixed_jobs(figure_id, repeats=repeats, limit=limit)
    selected_ids = {job["point_id"] for job in jobs}
    selected_points = [point for point in points if point.get("point_id") in selected_ids]
    if execute:
        figure_templates(figure_id, selected_points, repeats=repeats)
    root = WORK_ROOT / "search" / f"figure{figure_id}" / "fixed"
    summary = _run_fixed_batch(selected_points, root=root, repeats=repeats, execute=execute)
    if execute:
        collect_fixed_output(figure_id, summary["point_jobs"], root)
    return jobs


def _table_job_rows(binding: _FixedPointJob, manifest: dict[str, str]) -> list[dict[str, str]]:
    """Read a fixed job only after checking its resolved config and every repeat."""
    job = binding.job
    spec = job.job_spec
    fixed = spec["fixed_params"]
    if manifest.get("search_status") not in {"completed", "skipped_existing_result"}:
        raise ValueError(f"job is not complete: search_status={manifest.get('search_status')!r}")
    if not manifest.get("job_dir") or Path(manifest["job_dir"]).resolve() != job.job_dir.resolve():
        raise ValueError(
            f"manifest job_dir does not match the planned directory: {manifest.get('job_dir')!r}"
        )

    # An ID does not include the candidate or repeat count. Retain the same
    # compatibility rules as resume, but never rewrite a spec while collecting.
    saved_spec = mechanism_stage_utils.load_job_spec(job.job_dir)
    expected_spec = core._normalized_job_spec_for_comparison(spec)
    existing_spec = core._normalized_job_spec_for_comparison(saved_spec)
    if (
        mechanism_stage_utils.canonical_yaml_text(existing_spec)
        != mechanism_stage_utils.canonical_yaml_text(expected_spec)
    ):
        raise ValueError(f"saved job_spec.yaml does not match the current configuration: {job.job_dir}")

    for key in mechanism_stage_utils.OUTER_AXIS_NAMES:
        expected = mechanism_stage_utils.canonical_search_value(fixed.get(key))
        if manifest.get(key, "") != expected:
            raise ValueError(f"manifest field {key!r}: expected {expected!r}, got {manifest.get(key)!r}")

    candidate, = mechanism_stage_utils.job_candidates(spec)
    best = mechanism_stage_utils.load_best_config(job.job_dir)
    if mechanism_stage_utils.CandidateSpec.from_dict(best["best_candidate"]) != candidate:
        raise ValueError("best_config.yaml candidate does not match the planned fixed candidate")
    # Older reusable artifacts may have an unnamespaced best_config.job_id;
    # compare their experiment data instead of requiring that label to change.
    for key in ("fixed_params", "defaults"):
        saved = core._normalized_job_spec_for_comparison({key: best[key]})
        expected = core._normalized_job_spec_for_comparison({key: spec[key]})
        if (
            mechanism_stage_utils.canonical_yaml_text(saved)
            != mechanism_stage_utils.canonical_yaml_text(expected)
        ):
            raise ValueError(f"best_config.yaml {key} does not match the planned configuration")

    repeats = int(spec["defaults"]["stage"]["verify"]["repeats"])
    base_seed = int(spec["base_seed"])
    records: dict[int, dict[str, str]] = {}
    # Cached manifest rows can have verify_done=0. The resolved job spec and
    # actual CSVs, not those progress counters, determine completeness.
    for child in sorted(mechanism_stage_utils.verify_stage_dir(job.job_dir).glob("rank=*")):
        match = re.match(r"rank=(\d+)__repeat=(\d+)__candidate=(\d+)__", child.name)
        if not match:
            raise ValueError(f"invalid verify result directory: {child}")
        rank, repeat, candidate_id = map(int, match.groups())
        if rank != 1 or candidate_id != candidate.candidate_id:
            raise ValueError(f"unexpected rank or candidate in fixed result: {child}")
        if repeat not in range(1, repeats + 1):
            raise ValueError(f"unexpected repeat {repeat}; expected 1..{repeats}: {child}")
        if repeat in records:
            raise ValueError(f"duplicate repeat {repeat}: {child}")
        files = sorted(child.glob("*.csv"))
        if len(files) != 1:
            raise ValueError(f"expected exactly one result CSV, found {len(files)}: {child}")
        with files[0].open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            record = next(reader, None)
            if record is None or next(reader, None) is not None:
                raise ValueError(f"expected exactly one seed row: {files[0]}")
        expected_seed = base_seed + repeat - 1
        try:
            valid_seed = float(record.get("seed", "")) == expected_seed
        except (TypeError, ValueError):
            valid_seed = False
        if not valid_seed:
            raise ValueError(
                f"repeat {repeat} expected seed {expected_seed}, got {record.get('seed')!r}: {files[0]}"
            )
        try:
            valid_candidate = (
                float(record["x_steps"]) == candidate.x_steps
                and mechanism_stage_utils.row_matches_candidate(record, candidate, expected_seed=expected_seed)
            )
        except (KeyError, TypeError, ValueError, mechanism_stage_utils.StageError):
            valid_candidate = False
        if not valid_candidate:
            raise ValueError(f"result candidate does not match the planned fixed candidate: {files[0]}")
        for key in ("val/acc", "test/acc"):
            try:
                valid = math.isfinite(float(record[key]))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(f"missing or non-finite metric {key!r}: {files[0]}")
        records[repeat] = record

    missing = sorted(set(range(1, repeats + 1)) - records.keys())
    if missing:
        raise ValueError(f"missing verify repeats {missing}; expected {repeats}, found {len(records)}")

    return [
        {
            "setting": str(binding.point["setting"]),
            "dataset": str(fixed["dataset"]),
            "backbone": str(fixed["backbone"]),
            "feature_dim": mechanism_stage_utils.canonical_search_value(fixed.get("feature_dim")),
            "seed": record["seed"],
            "val_acc": record["val/acc"],
            "test_acc": record["test/acc"],
        }
        for _, record in sorted(records.items())
    ]


def _unselected_table6_job(manifest: dict[str, str], point_jobs: dict[str, _FixedPointJob]) -> bool:
    dimension = manifest.get("feature_dim")
    if dimension not in {"800", "3200"}:
        return False
    for binding in point_jobs.values():
        fixed = {**binding.job.job_spec["fixed_params"], "feature_dim": int(dimension)}
        point_id = f"table6_{fixed['dataset']}_{fixed['backbone']}_{dimension}"
        job_id = f"{point_id}__{mechanism_stage_utils.stable_job_id(fixed)}"
        if manifest.get("job_id") == job_id and all(
            manifest.get(key, "") == mechanism_stage_utils.canonical_search_value(fixed.get(key))
            for key in mechanism_stage_utils.OUTER_AXIS_NAMES
        ):
            return True
    return False


def _collect_table_outputs(
    table: str,
    point_jobs: dict[str, _FixedPointJob],
    root: Path,
) -> Path:
    if not point_jobs:
        raise ValueError(f"No fixed jobs were planned for {table}")
    manifest_path = root / mechanism_stage_utils.ROOT_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Batch manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifests = list(csv.DictReader(handle))

    by_id: dict[str, dict[str, str]] = {}
    for manifest in manifests:
        job_id = manifest.get("job_id", "")
        if job_id in by_id:
            raise RuntimeError(f"Duplicate job_id {job_id!r} in {manifest_path}")
        by_id[job_id] = manifest
        if job_id not in point_jobs:
            if table == "table6" and _unselected_table6_job(manifest, point_jobs):
                continue
            raise RuntimeError(f"Unexpected job_id {job_id!r} in {manifest_path}")

    rows: list[dict[str, str]] = []
    for job_id, binding in point_jobs.items():
        try:
            if job_id not in by_id:
                raise ValueError("missing manifest row")
            rows.extend({"table": table, **row} for row in _table_job_rows(binding, by_id[job_id]))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Cannot collect {table} point={binding.point.get('point_id')!r} "
                f"job_id={job_id!r} under {binding.job.job_dir}: {exc}"
            ) from exc

    # Validate the entire table before atomically replacing a previous export.
    output = root / f"{table}_seed_rows.csv"
    try:
        output_mode = stat.S_IMODE(output.stat().st_mode)
    except FileNotFoundError:
        output_mode = None
    temporary_path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary: Path | None = None
    try:
        # Exclusive creation uses normal file permissions (including umask)
        # instead of NamedTemporaryFile's 0600. Preserve an existing export's
        # mode so recollection does not remove access for collaborators.
        with temporary_path.open("x", newline="", encoding="utf-8") as handle:
            temporary = temporary_path
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if output_mode is not None:
            temporary.chmod(output_mode)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _fixed_table_points(table: str) -> list[dict[str, Any]]:
    path = FIXED_ROOT / f"{table}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    points = data.get("points") if isinstance(data, dict) else None
    if not isinstance(points, list) or not points:
        raise ValueError(f"Fixed table must contain a nonempty points list: {path}")
    for index, point in enumerate(points):
        if (
            not isinstance(point, dict)
            or not isinstance(point.get("fixed_params"), dict)
            or not isinstance(point.get("setting"), str)
            or not point["setting"].strip()
        ):
            raise ValueError(
                f"Fixed table point {index} requires fixed_params and a nonempty setting: {path}"
            )
    if table == "table6":
        if any(point["fixed_params"]["feature_dim"] != TABLE6_FEATURE_DIM for point in points):
            raise ValueError(f"All fixed Table 6 points must use feature_dim={TABLE6_FEATURE_DIM}: {path}")
    return points


def collect_fixed_table(table: str, *, repeats: int | None = None) -> Path:
    """Validate and export existing results without scheduling or changing jobs.

    By default, expect the paper's seed count, as used by Notebook 2. For a
    custom run, pass the same repeats value originally given to run_fixed_table.
    """
    points = _fixed_table_points(table)
    root = WORK_ROOT / "search" / table / "fixed"
    expected_repeats = reference_repeats(table) if repeats is None else repeats
    saved_config = mechanism_stage_utils.read_yaml_file(root / mechanism_stage_utils.INPUT_CONFIG_COPY_FILENAME)
    # Collection uses the experiment's saved device configuration and requires no local GPU.
    with tempfile.TemporaryDirectory(prefix=f".collect-{table}-", dir=root) as tmp:
        _, _, point_jobs = _build_fixed_batch(
            points,
            root=root,
            repeats=expected_repeats,
            config_root=Path(tmp),
            device=saved_config["device"],
        )
        return _collect_table_outputs(table, point_jobs, root)


def run_fixed_table(
    table: str,
    *,
    repeats: int = 3,
    execute: bool = False,
) -> dict[str, Any]:
    points = _fixed_table_points(table)
    root = WORK_ROOT / "search" / table / "fixed"
    summary = _run_fixed_batch(points, root=root, repeats=repeats, execute=execute)
    if execute:
        summary["seed_rows"] = _collect_table_outputs(table, summary["point_jobs"], root)
    summary["table"] = table
    return summary
