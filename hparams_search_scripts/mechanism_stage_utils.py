#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml

SCHEMA_VERSION = 1
VERIFY_TOPK = 5
DEFAULT_BASE_SEED = 12345
GRID_MAX_EPOCHS = 300
VERIFY_MAX_EPOCHS = 500
POLL_INTERVAL_SEC = 0.01

CANDIDATE_AXIS_NAMES = (
    "x_steps",
    "learning_rate",
    "weight_decay",
    "dropout",
    "tao2",
)

OUTER_AXIS_NAMES = (
    "dataset",
    "feature",
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
    "mechanism",
    "x_eps",
    "m",
    "norm",
    "norm_scale",
    "smoother",
    "backbone",
    "use_nfr",
)

SUMMARY_METRICS = (
    ("val/acc", "val_acc", True),
    ("test/acc", "test_acc", True),
)

GRID_STAGE_DIRNAME = "grid"
VERIFY_STAGE_DIRNAME = "verify_top5"
LOGS_DIRNAME = "logs"
JOB_SPEC_FILENAME = "job_spec.yaml"
GRID_RANKING_FILENAME = "grid_ranking.csv"
VERIFY_TOPK_FILENAME = "verify_topk.csv"
VERIFY_SUMMARY_FILENAME = "verify_top5_summary.csv"
BEST_CONFIG_FILENAME = "best_config.yaml"
RECOMMENDED_COMMAND_FILENAME = "recommended_command.txt"
INPUT_CONFIG_COPY_FILENAME = "input_config.yaml"
ROOT_MANIFEST_FILENAME = "manifest.csv"

_PATH_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9._=-]+")


class StageError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: int
    x_steps: int
    learning_rate: str
    weight_decay: str
    dropout: str
    tao2: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            str(int(self.x_steps)),
            canonical_float_text(self.learning_rate),
            canonical_float_text(self.weight_decay),
            canonical_float_text(self.dropout),
            canonical_optional_float_text(self.tao2),
        )

    @property
    def token(self) -> str:
        return "__".join(
            [
                f"candidate={self.candidate_id:04d}",
                f"x_steps={path_token(self.x_steps)}",
                f"learning_rate={path_token(self.learning_rate)}",
                f"weight_decay={path_token(self.weight_decay)}",
                f"dropout={path_token(self.dropout)}",
                f"tao2={path_token(self.tao2)}",
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "x_steps": int(self.x_steps),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "tao2": self.tao2,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CandidateSpec":
        try:
            candidate_id = int(raw["candidate_id"])
            x_steps = int(raw["x_steps"])
            learning_rate = canonical_float_text(raw["learning_rate"])
            weight_decay = canonical_float_text(raw["weight_decay"])
            dropout = canonical_float_text(raw["dropout"])
            tao2 = canonical_optional_float_text(raw.get("tao2", "none"))
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError(f"Invalid candidate spec: {raw}") from exc

        return cls(
            candidate_id=candidate_id,
            x_steps=x_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
            tao2=tao2,
        )


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate: CandidateSpec

    @property
    def candidate_id(self) -> int:
        return self.candidate.candidate_id

    @property
    def x_steps(self) -> int:
        return self.candidate.x_steps

    @property
    def learning_rate(self) -> str:
        return self.candidate.learning_rate

    @property
    def weight_decay(self) -> str:
        return self.candidate.weight_decay

    @property
    def dropout(self) -> str:
        return self.candidate.dropout

    @property
    def tao2(self) -> str:
        return self.candidate.tao2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RankedCandidate":
        try:
            rank = int(raw["rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError(f"Invalid ranked candidate row: {raw}") from exc
        return cls(rank=rank, candidate=CandidateSpec.from_dict(raw))


@dataclass(frozen=True)
class ExecutionSettings:
    device: str
    worker_ids: list[int]
    max_parallel_per_worker: int | dict[int, int]
    launch_interval_sec: float
    poll_interval_sec: float = POLL_INTERVAL_SEC


def read_yaml_file(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise StageError(f"Invalid YAML syntax in {path}") from exc


def write_yaml_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def canonical_yaml_text(data: Any) -> str:
    return yaml.safe_dump(data, sort_keys=True, allow_unicode=False)


def canonical_float_text(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a finite float, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Expected a finite float, got {value!r}")
    return f"{parsed:.12g}"


def canonical_optional_float_text(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() == "none":
            return "none"
        value = stripped
    return canonical_float_text(value)


def is_default_rewrite_scale_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return canonical_float_text(value) == canonical_float_text(1.0)
    except ValueError:
        return False


def is_default_feature_preprojection_value(value: Any) -> bool:
    return value in (None, False)


def normalized_outer_fixed_params(fixed_params: dict[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for name in OUTER_AXIS_NAMES:
        value = fixed_params.get(name)
        if name == "norm_scale" and not bool(fixed_params.get("norm")):
            value = None
        if name == "scale" and is_default_rewrite_scale_value(value):
            value = None
        if name == "feature_preprojection" and is_default_feature_preprojection_value(value):
            value = None
        if name == "preprojection_output_dim" and not bool(fixed_params.get("feature_preprojection")):
            value = None
        ordered[name] = value
    return ordered


def canonical_search_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isfinite(value):
            return f"{value:.12g}"
        return str(value)
    return str(value)


def path_token(value: Any) -> str:
    text = canonical_search_value(value)
    if text == "":
        return "none"
    normalized = _PATH_TOKEN_PATTERN.sub("-", text).strip("-")
    return normalized or "none"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def stable_job_id(fixed_params: dict[str, Any]) -> str:
    ordered = normalized_outer_fixed_params(fixed_params)
    payload = canonical_yaml_text(ordered).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def build_candidate_specs(
    *,
    x_steps_values: list[int],
    learning_rate_values: list[str],
    weight_decay_values: list[str],
    dropout_values: list[str],
    tao2_values: list[str],
) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    candidate_id = 1
    for x_steps in x_steps_values:
        for learning_rate in learning_rate_values:
            for weight_decay in weight_decay_values:
                for dropout in dropout_values:
                    for tao2 in tao2_values:
                        candidates.append(
                            CandidateSpec(
                                candidate_id=candidate_id,
                                x_steps=int(x_steps),
                                learning_rate=canonical_float_text(learning_rate),
                                weight_decay=canonical_float_text(weight_decay),
                                dropout=canonical_float_text(dropout),
                                tao2=canonical_optional_float_text(tao2),
                            )
                        )
                        candidate_id += 1
    return candidates


def job_spec_path(job_dir: Path) -> Path:
    return job_dir / JOB_SPEC_FILENAME


def load_job_spec(job_dir: Path) -> dict[str, Any]:
    path = job_spec_path(job_dir)
    if not path.is_file():
        raise StageError(f"Job spec not found: {path}")
    raw = read_yaml_file(path)
    if not isinstance(raw, dict):
        raise StageError(f"Job spec must be a mapping: {path}")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise StageError(
            f"Unsupported job spec schema_version={raw.get('schema_version')!r} in {path}"
        )
    return raw


def job_candidates(job_spec: dict[str, Any]) -> list[CandidateSpec]:
    raw_candidates = job_spec.get("candidates")
    if not isinstance(raw_candidates, list):
        raise StageError("job_spec.candidates must be a list")
    return [CandidateSpec.from_dict(raw) for raw in raw_candidates]


def job_candidate_by_id(job_spec: dict[str, Any], candidate_id: int) -> CandidateSpec:
    for candidate in job_candidates(job_spec):
        if candidate.candidate_id == candidate_id:
            return candidate
    raise StageError(f"Candidate id {candidate_id} not found in job spec")


def load_ranked_candidates(path: Path) -> list[RankedCandidate]:
    if not path.is_file():
        raise StageError(f"Ranked candidate file not found: {path}")

    ranked: list[RankedCandidate] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ranked.append(RankedCandidate.from_dict(row))
    if len(ranked) == 0:
        raise StageError(f"Ranked candidate file is empty: {path}")
    return ranked


def ranked_candidate_by_rank(path: Path, rank: int) -> RankedCandidate:
    for ranked in load_ranked_candidates(path):
        if ranked.rank == rank:
            return ranked
    raise StageError(f"Rank {rank} not found in ranked candidates: {path}")


def grid_stage_dir(job_dir: Path) -> Path:
    return job_dir / GRID_STAGE_DIRNAME


def verify_stage_dir(job_dir: Path) -> Path:
    return job_dir / VERIFY_STAGE_DIRNAME


def logs_dir(job_dir: Path) -> Path:
    return job_dir / LOGS_DIRNAME


def grid_ranking_path(job_dir: Path) -> Path:
    return job_dir / GRID_RANKING_FILENAME


def verify_topk_path(job_dir: Path) -> Path:
    return job_dir / VERIFY_TOPK_FILENAME


def verify_summary_path(job_dir: Path) -> Path:
    return job_dir / VERIFY_SUMMARY_FILENAME


def best_config_path(job_dir: Path) -> Path:
    return job_dir / BEST_CONFIG_FILENAME


def recommended_command_path(job_dir: Path) -> Path:
    return job_dir / RECOMMENDED_COMMAND_FILENAME


def grid_candidate_output_dir(job_dir: Path, candidate: CandidateSpec) -> Path:
    return grid_stage_dir(job_dir) / candidate.token


def verify_candidate_output_dir(job_dir: Path, ranked: RankedCandidate, repeat_id: int) -> Path:
    return verify_stage_dir(job_dir) / (
        f"rank={ranked.rank:02d}__repeat={repeat_id:02d}__{ranked.candidate.token}"
    )


def ensure_job_directories(job_dir: Path) -> None:
    grid_stage_dir(job_dir).mkdir(parents=True, exist_ok=True)
    verify_stage_dir(job_dir).mkdir(parents=True, exist_ok=True)
    logs_dir(job_dir).mkdir(parents=True, exist_ok=True)


def _read_first_csv_row(csv_path: Path) -> dict[str, str] | None:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except Exception:
        return None

    if row is None:
        return None
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def _normalize_seed_text(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return ""
    try:
        return str(int(float(raw_value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid seed value: {raw_value!r}") from exc


def candidate_key_from_row(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    try:
        return (
            str(int(float(row["x_steps"]))),
            canonical_float_text(row["learning_rate"]),
            canonical_float_text(row["weight_decay"]),
            canonical_float_text(row["dropout"]),
            canonical_optional_float_text(row.get("tao2")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageError(f"CSV row is missing candidate columns: {row}") from exc


def row_matches_candidate(
    row: dict[str, str],
    candidate: CandidateSpec,
    *,
    expected_seed: int | None = None,
) -> bool:
    if candidate_key_from_row(row) != candidate.key:
        return False
    if expected_seed is None:
        return True
    return _normalize_seed_text(row.get("seed")) == str(int(expected_seed))


def find_matching_result_csv(
    output_dir: Path,
    candidate: CandidateSpec,
    *,
    expected_seed: int | None = None,
) -> Path | None:
    if not output_dir.is_dir():
        return None
    for csv_path in sorted(output_dir.glob("*.csv")):
        row = _read_first_csv_row(csv_path)
        if row is None:
            continue
        try:
            if row_matches_candidate(row, candidate, expected_seed=expected_seed):
                return csv_path
        except StageError:
            continue
    return None


def candidate_result_exists(
    output_dir: Path,
    candidate: CandidateSpec,
    *,
    expected_seed: int | None = None,
) -> bool:
    return find_matching_result_csv(
        output_dir,
        candidate,
        expected_seed=expected_seed,
    ) is not None


def load_stage_result_rows(stage_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not stage_dir.is_dir():
        return rows
    for csv_path in sorted(stage_dir.rglob("*.csv")):
        row = _read_first_csv_row(csv_path)
        if row is not None:
            rows.append(row)
    return rows


def _candidate_lookup(job_spec: dict[str, Any]) -> dict[tuple[str, str, str, str, str], CandidateSpec]:
    lookup: dict[tuple[str, str, str, str, str], CandidateSpec] = {}
    for candidate in job_candidates(job_spec):
        lookup[candidate.key] = candidate
    return lookup


def _metric_summary(values: list[float]) -> dict[str, Any]:
    if len(values) == 0:
        raise StageError("Cannot summarize an empty metric list")
    return {
        "mean": float(mean(values)),
        "std": 0.0 if len(values) == 1 else float(stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "n": int(len(values)),
    }


def _summary_fieldnames() -> list[str]:
    return [
        "rank",
        "candidate_id",
        *CANDIDATE_AXIS_NAMES,
        "val_acc_mean",
        "val_acc_std",
        "val_acc_min",
        "val_acc_max",
        "test_acc_mean",
        "test_acc_std",
        "test_acc_min",
        "test_acc_max",
        "n",
    ]


def _metric_value_from_row(row: dict[str, str], column_name: str, *, required: bool) -> str | None:
    raw_value = row.get(column_name)
    if raw_value is None:
        if required:
            raise StageError(f"Result CSV rows must contain numeric {column_name!r} values")
        return None

    text = str(raw_value).strip()
    if text == "":
        if required:
            raise StageError(f"Result CSV rows must contain numeric {column_name!r} values")
        return None
    return text


def _metric_summaries_from_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, Any] | None]:
    summaries: dict[str, dict[str, Any] | None] = {}
    for column_name, metric_name, required in SUMMARY_METRICS:
        values: list[float] = []
        missing_optional = False
        for row in rows:
            raw_value = _metric_value_from_row(row, column_name, required=required)
            if raw_value is None:
                missing_optional = True
                continue
            try:
                values.append(float(raw_value))
            except ValueError as exc:
                raise StageError(
                    f"Result CSV rows must contain numeric {column_name!r} values"
                ) from exc

        if required:
            summaries[metric_name] = _metric_summary(values)
            continue

        if len(values) == 0:
            summaries[metric_name] = None
            continue
        if missing_optional or len(values) != len(rows):
            raise StageError(
                f"Optional metric {column_name!r} must be present for every row of a candidate or absent for every row"
            )
        summaries[metric_name] = _metric_summary(values)
    return summaries


def _sorted_aggregate_rows(
    grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, str]]],
    *,
    job_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_lookup = _candidate_lookup(job_spec)
    aggregated: list[dict[str, Any]] = []

    for candidate_key, rows in grouped_rows.items():
        candidate = candidate_lookup.get(candidate_key)
        if candidate is None:
            raise StageError(f"Result rows contain an unknown candidate key: {candidate_key}")
        metric_summaries = _metric_summaries_from_rows(rows)
        val_summary = metric_summaries["val_acc"]
        test_summary = metric_summaries["test_acc"]
        if val_summary is None or test_summary is None:
            raise StageError("Required metrics are missing from summary rows")
        aggregated.append(
            {
                "candidate_id": candidate.candidate_id,
                "x_steps": candidate.x_steps,
                "learning_rate": candidate.learning_rate,
                "weight_decay": candidate.weight_decay,
                "dropout": candidate.dropout,
                "tao2": candidate.tao2,
                "val_acc_mean": val_summary["mean"],
                "val_acc_std": val_summary["std"],
                "val_acc_min": val_summary["min"],
                "val_acc_max": val_summary["max"],
                "test_acc_mean": test_summary["mean"],
                "test_acc_std": test_summary["std"],
                "test_acc_min": test_summary["min"],
                "test_acc_max": test_summary["max"],
                "n": int(val_summary["n"]),
            }
        )
    aggregated.sort(
        key=lambda row: (
            -float(row["val_acc_mean"]),
            int(row["candidate_id"]),
        )
    )
    return aggregated


def aggregate_grid_results(job_spec: dict[str, Any], job_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in load_stage_result_rows(grid_stage_dir(job_dir)):
        grouped.setdefault(candidate_key_from_row(row), []).append(row)
    if len(grouped) == 0:
        raise StageError(f"No grid CSV rows found under {grid_stage_dir(job_dir)}")
    return _sorted_aggregate_rows(grouped, job_spec=job_spec)


def aggregate_verify_results(job_spec: dict[str, Any], job_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in load_stage_result_rows(verify_stage_dir(job_dir)):
        grouped.setdefault(candidate_key_from_row(row), []).append(row)
    if len(grouped) == 0:
        raise StageError(f"No verify CSV rows found under {verify_stage_dir(job_dir)}")
    return _sorted_aggregate_rows(grouped, job_spec=job_spec)


def write_grid_ranking(job_dir: Path, ranking_rows: list[dict[str, Any]]) -> None:
    ranking_path = grid_ranking_path(job_dir)
    with ranking_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = _summary_fieldnames()
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(ranking_rows, start=1):
            payload = {
                "rank": index,
                "candidate_id": row["candidate_id"],
                "x_steps": row["x_steps"],
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "dropout": row["dropout"],
                "tao2": row["tao2"],
                "val_acc_mean": canonical_float_text(row["val_acc_mean"]),
                "val_acc_std": canonical_float_text(row["val_acc_std"]),
                "val_acc_min": canonical_float_text(row["val_acc_min"]),
                "val_acc_max": canonical_float_text(row["val_acc_max"]),
                "test_acc_mean": canonical_float_text(row["test_acc_mean"]),
                "test_acc_std": canonical_float_text(row["test_acc_std"]),
                "test_acc_min": canonical_float_text(row["test_acc_min"]),
                "test_acc_max": canonical_float_text(row["test_acc_max"]),
                "n": row["n"],
            }
            writer.writerow(payload)


def write_verify_topk(job_dir: Path, ranking_rows: list[dict[str, Any]], *, topk: int) -> None:
    topk_count = min(int(topk), len(ranking_rows))
    if topk_count < 1:
        raise StageError("Top-k candidate count must be >= 1")

    path = verify_topk_path(job_dir)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["rank", "candidate_id", *CANDIDATE_AXIS_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(ranking_rows[:topk_count], start=1):
            writer.writerow(
                {
                    "rank": index,
                    "candidate_id": row["candidate_id"],
                    "x_steps": row["x_steps"],
                    "learning_rate": row["learning_rate"],
                    "weight_decay": row["weight_decay"],
                    "dropout": row["dropout"],
                    "tao2": row["tao2"],
                }
            )


def write_verify_summary(job_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    path = verify_summary_path(job_dir)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = _summary_fieldnames()
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(summary_rows, start=1):
            payload = {
                "rank": index,
                "candidate_id": row["candidate_id"],
                "x_steps": row["x_steps"],
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "dropout": row["dropout"],
                "tao2": row["tao2"],
                "val_acc_mean": canonical_float_text(row["val_acc_mean"]),
                "val_acc_std": canonical_float_text(row["val_acc_std"]),
                "val_acc_min": canonical_float_text(row["val_acc_min"]),
                "val_acc_max": canonical_float_text(row["val_acc_max"]),
                "test_acc_mean": canonical_float_text(row["test_acc_mean"]),
                "test_acc_std": canonical_float_text(row["test_acc_std"]),
                "test_acc_min": canonical_float_text(row["test_acc_min"]),
                "test_acc_max": canonical_float_text(row["test_acc_max"]),
                "n": row["n"],
            }
            writer.writerow(payload)


def load_best_config(job_dir: Path) -> dict[str, Any]:
    path = best_config_path(job_dir)
    if not path.is_file():
        raise StageError(f"Best config file not found: {path}")
    raw = read_yaml_file(path)
    if not isinstance(raw, dict):
        raise StageError(f"Best config must be a mapping: {path}")
    return raw


def job_outputs_complete(job_dir: Path) -> bool:
    return (
        verify_summary_path(job_dir).is_file()
        and best_config_path(job_dir).is_file()
        and recommended_command_path(job_dir).is_file()
    )
