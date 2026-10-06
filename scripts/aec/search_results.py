from __future__ import annotations

import math
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from hparams_search_scripts import mechanism_stage_utils as stage
from hparams_search_scripts import table_search_suite_core as core

from .paths import REFERENCE_ROOT, normalize_mode, result_root
from .result_io import VerifiedResult, atomic_write_csv, manifest_index, read_completed_job, read_csv
from .search_runner import PIPELINE_AXES, SearchPlan, _table_setting_from_config, build_search_plan
from .table_settings import TABLE6_FEATURE_DIM


TARGETS = (1, 3, 4, 5, 6, 7, "table4", "table6")
PRESENTATION_FIELDS = (
    "source", "backbone", "dataset", "x_eps", "x_eps_label", "x_index", "line_label",
    "pipeline", "mechanism", "mechanism_label", "epsilon", "epsilon_label",
    "scale_exponent", "norm_scale", "norm_scale_label", "tao2", "delta", "delta_label", "point_id",
)
PIPELINES = {
    "LPGNN": "figure3_pipeline1", "PrivGE": "figure3_pipeline2",
    "UPGNET-MBM": "figure3_pipeline3", "UPGNET-PM": "figure3_pipeline4",
}


def number(value: object) -> str:
    return stage.canonical_float_text(value)


def table_datasets(table: str, mode: str) -> tuple[str, ...]:
    if table == "table4" and normalize_mode(mode) == "scaled":
        return ("cora", "facebook")
    return ("cora", "lastfm", "citeseer", "facebook")


def plot_templates(target: int, mode: str) -> list[dict[str, str]]:
    rows = read_csv(REFERENCE_ROOT / f"figure{target}_plot_data.csv")
    if normalize_mode(mode) == "scaled":
        if target == 1:
            rows = [row for row in rows if row["dataset"] in {"cora", "facebook"}]
        elif target == 6:
            rows = [row for row in rows if row["backbone"] == "sage"]
    return [{key: row[key] for key in PRESENTATION_FIELDS if key in row} for row in rows]


def plot_key(target: int, row: dict[str, Any], *, binding: bool = False) -> tuple[str, ...]:
    if target in {1, 6}:
        axis = "baseline" if binding and row["pipeline"] in {"featfree", "clean_reference"} else number(row["x_eps"])
        return (row["dataset"], row["backbone"], row["pipeline"], axis)
    if target == 3:
        return (row["source"], row["mechanism"], number(row["x_eps"]))
    if target in {4, 7}:
        return (number(row["epsilon"]), number(row["norm_scale"]))
    if target == 5:
        return (number(row["epsilon"]), number(row["tao2"]))
    raise ValueError(f"Unsupported figure: {target}")


def _job_plot_key(plan: SearchPlan, job: core.BatchJob) -> tuple[str, ...]:
    fixed = job.job_spec["fixed_params"]
    if plan.target in {1, 6}:
        source = plan.sources[job.job_id]
        if "FeatFree" in source.parts:
            pipeline, axis = "featfree", "baseline"
        elif "Non-private" in source.parts:
            pipeline, axis = "clean_reference", "baseline"
        else:
            pipeline, axis = PIPELINES[source.stem], number(fixed["x_eps"])
            for key, expected in PIPELINE_AXES[pipeline].items():
                if stage.canonical_search_value(fixed[key]) != expected:
                    raise ValueError(f"Pipeline {pipeline} has inconsistent {key}: {source}")
        return (fixed["dataset"], fixed["backbone"], pipeline, axis)
    if plan.target == 3:
        source = {"raw": "ldp", "sim": "sim"}[fixed["feature"]]
        epsilon = fixed["sim_reference_eps"] if source == "sim" else fixed["x_eps"]
        return (source, fixed["mechanism"], number(epsilon))
    if plan.target in {4, 7}:
        return (number(fixed["x_eps"]), number(fixed["norm_scale"]))
    if plan.target == 5:
        values = job.job_spec["candidate_space"]["tao2"]
        if len(values) != 1:
            raise ValueError(f"Figure 5 requires one tao2 value per job: {job.job_id}")
        return (number(fixed["x_eps"]), number(values[0]))
    raise ValueError(f"Unsupported plot target: {plan.target}")


def plot_bindings(plan: SearchPlan) -> list[tuple[dict[str, str], core.BatchJob]]:
    templates = plot_templates(plan.target, plan.mode)
    if len({plot_key(plan.target, row) for row in templates}) != len(templates):
        raise ValueError(f"Duplicate plot coordinates for Figure {plan.target}")
    by_key = {}
    for job in plan.batch.jobs:
        key = _job_plot_key(plan, job)
        if key in by_key:
            raise ValueError(f"Multiple jobs map to Figure {plan.target} point {key}")
        by_key[key] = job
    bindings = []
    for row in templates:
        key = plot_key(plan.target, row, binding=True)
        if key not in by_key:
            raise RuntimeError(f"Search plan cannot produce Figure {plan.target} point {key}")
        bindings.append((row, by_key[key]))
    return bindings


def table_groups(
    table: str, mode: str, *, feature_dims: tuple[int, ...] | None = None,
) -> set[tuple[str, ...]]:
    datasets = table_datasets(table, mode)
    groups = {
        (row["setting"], row["dataset"], row["backbone"], row.get("feature_dim", ""))
        for row in read_csv(REFERENCE_ROOT / f"{table}_seed_rows.csv") if row["dataset"] in datasets
    }
    if table == "table6" and feature_dims is not None:
        return {(*group[:3], str(dim)) for group in groups for dim in feature_dims}
    return groups


def _table_group(plan: SearchPlan, job: core.BatchJob) -> tuple[str, ...]:
    fixed = job.job_spec["fixed_params"]
    return (
        _table_setting_from_config(plan.sources[job.job_id]), fixed["dataset"], fixed["backbone"],
        stage.canonical_search_value(fixed.get("feature_dim")),
    )


def validate_search_plan(plan: SearchPlan) -> None:
    if isinstance(plan.target, int):
        plot_bindings(plan)
        expected_repeats = {int(row["n"]) for row in read_csv(REFERENCE_ROOT / f"figure{plan.target}_plot_data.csv")}
    else:
        expected = table_groups(plan.target, plan.mode, feature_dims=plan.feature_dims)
        actual = Counter(_table_group(plan, job) for job in plan.batch.jobs)
        missing = expected - actual.keys()
        unexpected = actual.keys() - expected if plan.target == "table6" else set()
        repeated = [key for key in expected if actual[key] > 1]
        if missing or unexpected or repeated:
            raise RuntimeError(
                f"Invalid {plan.target} coverage: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}, duplicate={repeated}"
            )
        counts = Counter(
            (row["setting"], row["dataset"], row["backbone"], row.get("feature_dim", ""))
            for row in read_csv(REFERENCE_ROOT / f"{plan.target}_seed_rows.csv")
        )
        expected_repeats = set(counts.values())
    for job in plan.batch.jobs:
        repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])
        if repeats not in expected_repeats:
            raise ValueError(f"Unexpected repeat count {repeats} for {plan.target}: {job.job_id}")


def _load_results(
    plan: SearchPlan, cache: dict[tuple[str, str], VerifiedResult], job_ids: set[str] | None = None,
) -> dict[str, VerifiedResult]:
    indexed = manifest_index(plan.batch.output_root, plan.batch.jobs)
    results = {}
    for job in plan.batch.jobs:
        if job_ids is not None and job.job_id not in job_ids:
            continue
        key = (str(job.job_dir), job.job_id)
        if key not in cache:
            cache[key] = read_completed_job(job, indexed[job.job_id])
        results[job.job_id] = cache[key]
    return results


def _baseline_results(target: int, mode: str, cache, *, pipeline: str, datasets: tuple[str, ...], backbone: str | None = None):
    plan = build_search_plan(target, mode)
    validate_search_plan(plan)
    selected = {}
    for template, job in plot_bindings(plan):
        if template["dataset"] not in datasets or (backbone is not None and template["backbone"] != backbone):
            continue
        if pipeline == "ldp":
            if template["pipeline"] not in PIPELINE_AXES or number(template["x_eps"]) != "10":
                continue
        elif template["pipeline"] != pipeline:
            continue
        selected[job.job_id] = template
    if not selected:
        raise RuntimeError(f"No {pipeline} baseline in Figure {target} mode={mode}")
    return _load_results(plan, cache, set(selected)), selected


def _statistics(target: int, template: dict[str, str], result: VerifiedResult) -> dict[str, Any]:
    values = {key: [record[key] for record in result.records] for key in ("val_acc", "test_acc")}
    statistics = {}
    for key in values:
        for name, value in result.metrics[key].items():
            if name != "n":
                statistics[f"{key}_{name}"] = value
    if target in {1, 6}:
        from draw_figure.draw_figure1 import metric_stats

        axis = "baseline" if template["pipeline"] in {"featfree", "clean_reference"} else number(template["x_eps"])
        group_key = (template["dataset"], template["backbone"], template["pipeline"], axis)
        for key in values:
            stats = metric_stats(values[key], bootstrap_samples=1000, bootstrap_seed=12345, key=(key, *group_key))
            statistics[f"{key}_mean"] = stats["mean"]
            statistics[f"{key}_ci_low"] = stats["ci_low"]
            statistics[f"{key}_ci_high"] = stats["ci_high"]
    else:
        if target == 3:
            from draw_figure.draw_figure3 import bootstrap_ci as bootstrap

            group_key = (template["source"], template["mechanism"], Decimal(template["x_eps"]))
        elif target == 5:
            from draw_figure.draw_figure5 import bootstrap_ci as bootstrap

            group_key = (Decimal(template["epsilon"]), Decimal(template["tao2"]))
        else:
            from draw_figure.draw_figure4 import bootstrap_ci as bootstrap

            group_key = (Decimal(template["epsilon"]), int(template["scale_exponent"]))
        average, low, high = bootstrap(values["test_acc"], samples=1000, seed=12345, key=group_key)
        statistics.update(test_acc_mean=average, test_acc_ci_low=low, test_acc_ci_high=high)
    statistics["n"] = len(result.records)
    return statistics


def plot_row(
    target: int, template: dict[str, str], result: VerifiedResult, source: Path, *, mode: str = "search",
) -> dict[str, Any]:
    fixed = result.job.job_spec["fixed_params"]
    row = {**template, **_statistics(target, template, result), **result.candidate.to_dict()}
    if target == 5:
        row["tao2"] = template["tao2"]
    row.update(
        source=template["source"] if target == 3 else mode,
        dataset=fixed["dataset"], backbone=fixed["backbone"],
        smoother=stage.canonical_search_value(fixed.get("smoother")),
        feature_dim=stage.canonical_search_value(fixed.get("feature_dim")),
        candidate_tao2=result.candidate.tao2,
        verify_rank=result.rank, verify_ranks="|".join(str(result.rank) for _ in result.records),
        job_id=result.job.job_id, source_job_dir=str(result.job.job_dir), source_config=str(source),
    )
    return row


def _seed_rows(table: str, setting: str, result: VerifiedResult) -> list[dict[str, Any]]:
    fixed = result.job.job_spec["fixed_params"]
    return [{
        "table": table, "setting": setting, "dataset": fixed["dataset"], "backbone": fixed["backbone"],
        "feature_dim": stage.canonical_search_value(fixed.get("feature_dim")),
        "smoother": stage.canonical_search_value(fixed.get("smoother")),
        **record, **result.candidate.to_dict(), "job_id": result.job.job_id,
        "source_job_dir": str(result.job.job_dir),
    } for record in result.records]


def build_output_rows(plan: SearchPlan, *, cache: dict | None = None) -> list[dict[str, Any]]:
    validate_search_plan(plan)
    cache = {} if cache is None else cache
    results = _load_results(plan, cache)
    if isinstance(plan.target, int):
        rows = [plot_row(plan.target, template, results[job.job_id], plan.sources[job.job_id]) for template, job in plot_bindings(plan)]
        if plan.target in {4, 5, 7}:
            target, dataset = (6, "flickr") if plan.target == 7 else (1, "cora")
            baselines, _ = _baseline_results(target, plan.mode, cache, pipeline="featfree", datasets=(dataset,), backbone="sage")
            if len(baselines) != 1:
                raise RuntimeError(f"Expected one FeatFree baseline for Figure {plan.target}")
            baseline = next(iter(baselines.values()))
            for row in rows:
                row.update(featfree_acc=baseline.metrics["test_acc"]["mean"], baseline_job_id=baseline.job.job_id)
        return rows

    datasets = table_datasets(plan.target, plan.mode)
    rows = []
    for job in plan.batch.jobs:
        group = _table_group(plan, job)
        if group[1] in datasets:
            rows.extend(_seed_rows(plan.target, group[0], results[job.job_id]))
    if plan.target == "table4":
        baselines, templates = _baseline_results(1, plan.mode, cache, pipeline="ldp", datasets=datasets)
        grouped = defaultdict(list)
        for job_id, baseline in baselines.items():
            fixed = baseline.job.job_spec["fixed_params"]
            grouped[(fixed["dataset"], fixed["backbone"])].append(baseline)
        for dataset in datasets:
            for backbone in ("gcn", "sage", "gat"):
                candidates = grouped[(dataset, backbone)]
                if len(candidates) != len(PIPELINE_AXES):
                    raise RuntimeError(f"Incomplete Best-LDP comparison for {dataset}/{backbone}")
                winner = min(candidates, key=lambda item: (-item.metrics["val_acc"]["mean"], templates[item.job.job_id]["pipeline"]))
                rows.extend(_seed_rows("table4", "Best-LDP", winner))
    return rows


def output_path(target: int | str, mode: str) -> Path:
    return result_root(target, mode) / (f"{target}_seed_rows.csv" if isinstance(target, str) else "plot_data.csv")


def collect_search_outputs(
    target: int | str, mode: str, *, plan: SearchPlan | None = None,
    feature_dim: int | list[int] = TABLE6_FEATURE_DIM,
) -> Path:
    mode = normalize_mode(mode)
    plan = build_search_plan(target, mode, feature_dim=feature_dim) if plan is None else plan
    if plan.target != target or plan.mode != mode:
        raise ValueError("Search plan target/mode does not match the requested output")
    rows = build_output_rows(plan)
    path = output_path(target, plan.mode)
    atomic_write_csv(path, rows)
    return path


def _output_key(target: int | str, row: dict[str, Any]) -> tuple[str, ...]:
    if isinstance(target, int):
        return plot_key(target, row)
    return (row["setting"], row["dataset"], row["backbone"], str(row["feature_dim"]), str(row["seed"]))


def validate_output_rows(target: int | str, expected: list[dict[str, Any]], actual: list[dict[str, str]], path: Path) -> None:
    indexed = {_output_key(target, row): row for row in actual}
    expected_keys = {_output_key(target, row) for row in expected}
    if len(actual) != len(indexed) or indexed.keys() != expected_keys:
        raise RuntimeError(f"Incomplete or duplicate output rows in {path}: expected {len(expected)}, found {len(actual)}")
    for row in expected:
        found = indexed[_output_key(target, row)]
        for field, value in row.items():
            if isinstance(value, float):
                matches = field in found and math.isclose(float(found[field]), value, rel_tol=1e-10, abs_tol=1e-9)
            else:
                matches = found.get(field) == str(value)
            if not matches:
                raise RuntimeError(f"Output differs from training results: {path}, point={_output_key(target, row)}, field={field}")


def validate_search_output(
    target: int | str, mode: str, *, cache: dict | None = None,
    feature_dim: int | list[int] = TABLE6_FEATURE_DIM,
) -> dict[str, Any]:
    plan = build_search_plan(target, mode, feature_dim=feature_dim)
    expected = build_output_rows(plan, cache=cache)
    path = output_path(target, mode)
    actual = read_csv(path)
    validate_output_rows(target, expected, actual, path)
    return {"target": target, "mode": plan.mode, "jobs": len(plan.batch.jobs), "rows": len(actual), "path": str(path)}
