from __future__ import annotations

import ast
import csv
import math
import re
import stat
import uuid
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from hparams_search_scripts import mechanism_stage_utils as stage
from hparams_search_scripts import table_search_suite_core as core


@dataclass(frozen=True)
class VerifiedResult:
    job: core.BatchJob
    candidate: stage.CandidateSpec
    rank: int
    records: list[dict[str, Any]]
    metrics: dict[str, dict[str, float | int]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot export an empty result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    created = False
    try:
        with temporary.open("x", newline="", encoding="utf-8") as handle:
            created = True
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def manifest_index(root: Path, jobs: list[core.BatchJob]) -> dict[str, dict[str, str]]:
    path = root / stage.ROOT_MANIFEST_FILENAME
    expected = {job.job_id for job in jobs}
    if len(expected) != len(jobs):
        raise ValueError(f"Duplicate planned job IDs under {root}")
    indexed = {}
    for row in read_csv(path):
        job_id = row.get("job_id", "")
        if job_id in indexed:
            raise RuntimeError(f"Duplicate job_id={job_id!r} in {path}")
        if job_id not in expected:
            raise RuntimeError(f"Unexpected job_id={job_id!r} in {path}")
        indexed[job_id] = row
    missing = expected - indexed.keys()
    if missing:
        raise RuntimeError(f"Missing {len(missing)} jobs in {path}: {sorted(missing)[:5]}")
    return indexed


def _metrics(records: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    values = [record[key] for record in records]
    return {
        "mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values), "n": len(values),
    }


def _same_number(actual: object, expected: object) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-9)


def read_completed_job(job: core.BatchJob, manifest: dict[str, str]) -> VerifiedResult:
    location = f"job_id={job.job_id!r}, directory={job.job_dir}"

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"{message}; {location}")

    require(manifest.get("job_id") == job.job_id, "Manifest job ID mismatch")
    require(manifest.get("search_status") in {"completed", "skipped_existing_result"}, "Job is incomplete")
    require(Path(manifest["job_dir"]).resolve() == job.job_dir.resolve(), "Manifest job directory mismatch")
    saved = stage.load_job_spec(job.job_dir)
    expected = core._normalized_job_spec_for_comparison(job.job_spec)
    require(
        stage.canonical_yaml_text(core._normalized_job_spec_for_comparison(saved))
        == stage.canonical_yaml_text(expected), "Saved experiment configuration mismatch",
    )
    fixed = job.job_spec["fixed_params"]
    for key in stage.OUTER_AXIS_NAMES:
        require(manifest.get(key, "") == stage.canonical_search_value(fixed.get(key)), f"Manifest field mismatch: {key}")

    candidates = {item.candidate_id: item for item in stage.job_candidates(job.job_spec)}
    ranked = stage.load_ranked_candidates(stage.verify_topk_path(job.job_dir))
    count = min(int(job.job_spec["verify_topk"]), len(candidates))
    require(sorted(item.rank for item in ranked) == list(range(1, count + 1)), "Incomplete verify candidate ranks")
    require(len({item.candidate_id for item in ranked}) == count, "Duplicate verify candidate")
    by_rank = {item.rank: item for item in ranked}
    for item in ranked:
        require(candidates.get(item.candidate_id) == item.candidate, "Verify candidate differs from job specification")

    repeats = int(job.job_spec["defaults"]["stage"]["verify"]["repeats"])
    base_seed = int(job.job_spec["base_seed"])
    grouped: dict[int, dict[int, dict[str, Any]]] = {item.candidate_id: {} for item in ranked}
    for child in sorted(stage.verify_stage_dir(job.job_dir).glob("rank=*")):
        match = re.match(r"rank=(\d+)__repeat=(\d+)__candidate=(\d+)__", child.name)
        require(match is not None, f"Invalid verify directory: {child.name}")
        rank, repeat, candidate_id = map(int, match.groups())
        require(rank in by_rank and by_rank[rank].candidate_id == candidate_id, f"Unexpected candidate: {child.name}")
        require(1 <= repeat <= repeats, f"Unexpected repeat: {child.name}")
        require(repeat not in grouped[candidate_id], f"Duplicate repeat: {child.name}")
        files = sorted(child.glob("*.csv"))
        require(len(files) == 1, f"Expected one CSV in {child}")
        raw = read_csv(files[0])
        require(len(raw) == 1, f"Expected one seed row in {files[0]}")
        row = raw[0]
        candidate = candidates[candidate_id]
        numeric_fields = {
            "x_eps", "sim_reference_eps", "feature_dim", "scale", "norm_scale",
            "preprojection_output_dim", "random_normal_mean", "random_normal_std", "shared_value",
            "degree_bucket_num_buckets", "degree_bucket_range_max", "deepwalk_walk_length",
            "deepwalk_number_walks", "deepwalk_window_size", "deepwalk_workers",
        }
        for key, value in fixed.items():
            if value is None or (key == "norm_scale" and not fixed["norm"]):
                continue
            column = "model" if key == "backbone" else key
            require(column in row, f"Missing parameter {column} in {files[0]}")
            actual = row[column]
            equal = Decimal(actual) == Decimal(str(value)) if key in numeric_fields else actual.lower() == str(value).lower()
            require(equal, f"Incorrect parameter {column} in {files[0]}")
        defaults = job.job_spec["defaults"]
        run_params = {
            **defaults["model"], **defaults["trainer"],
            "val_ratio": defaults["dataset"]["val_ratio"], "test_ratio": defaults["dataset"]["test_ratio"],
            "max_epochs": defaults["stage"]["verify"]["max_epochs"],
            "patience": defaults["stage"]["verify"]["patience"], "repeats": 1,
        }
        for key, value in run_params.items():
            require(key in row, f"Missing training parameter {key} in {files[0]}")
            equal = Decimal(row[key]) == Decimal(str(value)) if type(value) in {int, float} else row[key].lower() == str(value).lower()
            require(equal, f"Incorrect training parameter {key} in {files[0]}")
        require(list(ast.literal_eval(row["data_range"])) == defaults["dataset"]["data_range"], f"Incorrect data_range in {files[0]}")
        require(Decimal(row["seed"]) == base_seed + repeat - 1, f"Incorrect seed in {files[0]}")
        require(Decimal(row["x_steps"]) == candidate.x_steps, f"Incorrect x_steps in {files[0]}")
        require(stage.row_matches_candidate(row, candidate), f"Incorrect candidate parameters in {files[0]}")
        val_acc, test_acc = float(row["val/acc"]), float(row["test/acc"])
        require(math.isfinite(val_acc) and math.isfinite(test_acc), f"Non-finite metrics in {files[0]}")
        grouped[candidate_id][repeat] = {
            "seed": base_seed + repeat - 1, "repeat": repeat, "rank": rank,
            "val_acc": val_acc, "test_acc": test_acc, "result_csv": str(files[0]),
        }

    metrics = {}
    for candidate_id, records in grouped.items():
        missing = set(range(1, repeats + 1)) - records.keys()
        require(not missing, f"Candidate {candidate_id} is missing repeats {sorted(missing)}")
        ordered = [records[index] for index in sorted(records)]
        metrics[candidate_id] = {key: _metrics(ordered, key) for key in ("val_acc", "test_acc")}

    # Select the highest mean validation accuracy across verified candidates; break ties by candidate_id.
    winner = min(metrics, key=lambda candidate_id: (-metrics[candidate_id]["val_acc"]["mean"], candidate_id))
    best = stage.load_best_config(job.job_dir)
    require(stage.CandidateSpec.from_dict(best["best_candidate"]) == candidates[winner], "Best candidate disagrees with validation results")
    for key in ("fixed_params", "defaults"):
        require(
            stage.canonical_yaml_text(core._normalized_job_spec_for_comparison({key: best[key]}))
            == stage.canonical_yaml_text(core._normalized_job_spec_for_comparison({key: job.job_spec[key]})),
            f"Best configuration mismatch: {key}",
        )
    require(int(manifest["best_candidate_id"]) == winner, "Manifest winner mismatch")
    summary = read_csv(stage.verify_summary_path(job.job_dir))
    require(len(summary) == count and len({int(row["candidate_id"]) for row in summary}) == count, "Incomplete verify summary")
    for row in summary:
        candidate_id = int(row["candidate_id"])
        require(candidate_id in metrics, "Unknown verify summary candidate")
        require(stage.CandidateSpec.from_dict(row) == candidates[candidate_id], "Verify summary candidate mismatch")
        require(int(row["n"]) == repeats, "Verify summary seed count mismatch")
        for key in ("val_acc", "test_acc"):
            for stat_name in ("mean", "std", "min", "max"):
                require(_same_number(row[f"{key}_{stat_name}"], metrics[candidate_id][key][stat_name]), f"Verify summary mismatch: {key}_{stat_name}")
    for key in ("val_acc", "test_acc"):
        for stat_name, value in metrics[winner][key].items():
            require(_same_number(best["verify_metrics"][key][stat_name], value), f"Best metric mismatch: {key}.{stat_name}")
        for stat_name in ("mean", "std"):
            require(_same_number(manifest[f"best_verify_{key}_{stat_name}"], metrics[winner][key][stat_name]), f"Manifest metric mismatch: {key}.{stat_name}")
    records = [grouped[winner][repeat] for repeat in range(1, repeats + 1)]
    return VerifiedResult(job, candidates[winner], records[0]["rank"], records, metrics[winner])
