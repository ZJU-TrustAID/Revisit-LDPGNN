from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from hparams_search_scripts import mechanism_stage_utils as stage

from .fixed_runner import _FixedPointJob, _build_fixed_batch, fixed_points, reference_repeats
from .paths import FIXED_ROOT, result_root
from .result_io import atomic_write_csv, manifest_index, read_completed_job, read_csv
from .search_results import plot_row, plot_templates, validate_output_rows


def figure_templates(
    figure_id: int, points: list[dict[str, Any]], *, repeats: int,
) -> list[dict[str, str]]:
    templates = plot_templates(figure_id, "full")
    ids = [point.get("point_id") for point in points]
    if any(not isinstance(point_id, str) or not point_id for point_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Figure {figure_id} requires unique, nonempty point IDs")
    baselines = [point for point in points if point.get("scalar_baseline")]
    expected_baselines = 1 if figure_id in {4, 5, 7} else 0
    if len(baselines) != expected_baselines:
        raise ValueError(f"Figure {figure_id} requires {expected_baselines} scalar baseline(s)")
    expected = {row["point_id"] for row in templates} | {point["point_id"] for point in baselines}
    actual = set(ids)
    if actual != expected:
        raise ValueError(
            f"Incomplete fixed configuration for Figure {figure_id}: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    if repeats != reference_repeats(figure_id):
        raise ValueError(f"Figure {figure_id} requires {reference_repeats(figure_id)} verify repeats, got {repeats}")
    if baselines:
        fixed = baselines[0]["fixed_params"]
        expected_dataset = "flickr" if figure_id == 7 else "cora"
        if (fixed["dataset"], fixed["backbone"], fixed["feature"]) != (expected_dataset, "sage", "random_normal"):
            raise ValueError(f"Incorrect FeatFree baseline configuration for Figure {figure_id}")
    return templates


def build_fixed_rows(
    figure_id: int, point_jobs: dict[str, _FixedPointJob], root: Path,
) -> list[dict[str, Any]]:
    if not point_jobs:
        raise ValueError(f"No fixed jobs for Figure {figure_id}")
    bindings = list(point_jobs.values())
    repeats = {int(item.job.job_spec["defaults"]["stage"]["verify"]["repeats"]) for item in bindings}
    if len(repeats) != 1:
        raise ValueError(f"Inconsistent repeat counts for Figure {figure_id}")
    templates = figure_templates(figure_id, [item.point for item in bindings], repeats=repeats.pop())
    manifests = manifest_index(root, [item.job for item in bindings])
    results = {
        item.point["point_id"]: read_completed_job(item.job, manifests[item.job.job_id])
        for item in bindings
    }
    source = FIXED_ROOT / f"figure{figure_id}.yaml"
    rows = []
    for template in templates:
        row = plot_row(figure_id, template, results[template["point_id"]], source, mode="fixed")
        row["run_mode"] = "fixed"
        rows.append(row)
    if figure_id in {4, 5, 7}:
        baseline, = [results[item.point["point_id"]] for item in bindings if item.point.get("scalar_baseline")]
        for row in rows:
            row.update(featfree_acc=baseline.metrics["test_acc"]["mean"], baseline_job_id=baseline.job.job_id)
    return rows


def collect_fixed_output(figure_id: int, point_jobs: dict[str, _FixedPointJob], root: Path) -> Path:
    rows = build_fixed_rows(figure_id, point_jobs, root)
    path = root / "plot_data.csv"
    atomic_write_csv(path, rows)
    return path


def validate_fixed_output(figure_id: int, *, root: Path | None = None) -> dict[str, Any]:
    root = result_root(figure_id, "fixed") if root is None else root
    points = fixed_points(figure_id)
    repeats = reference_repeats(figure_id)
    figure_templates(figure_id, points, repeats=repeats)
    saved = stage.read_yaml_file(root / stage.INPUT_CONFIG_COPY_FILENAME)
    with tempfile.TemporaryDirectory(prefix=".fixed-validate-", dir=root) as temporary:
        _, _, point_jobs = _build_fixed_batch(
            points, root=root, repeats=repeats, config_root=Path(temporary), device=saved["device"],
        )
        expected = build_fixed_rows(figure_id, point_jobs, root)
    path = root / "plot_data.csv"
    actual = read_csv(path)
    validate_output_rows(figure_id, expected, actual, path)
    return {"figure_id": figure_id, "mode": "fixed", "jobs": len(point_jobs), "rows": len(actual), "path": str(path)}
