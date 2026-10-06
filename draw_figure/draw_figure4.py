#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.ticker import MultipleLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "aec" / "reference"
DEFAULT_FIGURE5_ROOT = PAPER_ROOT / "figure5_final"
DEFAULT_FIGURE5_ADD_ROOT = PAPER_ROOT / "figure5_final_add"
DEFAULT_FIGURE5_ADD_AGAIN_ROOT = PAPER_ROOT / "figure5_final_add_again"
DEFAULT_OUTPUT_DIR = DEFAULT_FIGURE5_ROOT / "plots"
DEFAULT_FEATFREE_MANIFEST = (
    REPO_ROOT / "scripts" / "aec" / "reference" / "reference_metrics.yaml"
)
FEATFREE_LABEL = r"$\mathsf{FeatFree}$"

EPSILON_SPECS = (
    (Decimal("0.0001"), r"$\epsilon=10^{-4}$"),
    (Decimal("0.001"), r"$\epsilon=10^{-3}$"),
    (Decimal("0.01"), r"$\epsilon=10^{-2}$"),
    (Decimal("0.1"), r"$\epsilon=10^{-1}$"),
    (Decimal("1.0"), r"$\epsilon=10^{0}$"),
)
EPSILON_TO_INDEX = {epsilon: idx for idx, (epsilon, _) in enumerate(EPSILON_SPECS)}

SCALE_EXPONENTS = tuple(range(-6, 18))
ADD_SOURCE_EXPONENTS = {-6, -4}
ADD_AGAIN_SOURCE_EXPONENTS = {-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17}

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 12345
EXPECTED_REPEATS = 10

VERIFY_DIR_RE = re.compile(
    r"^rank=(?P<rank>\d+)__repeat=(?P<repeat>\d+)__candidate=(?P<candidate>\d+)__"
    r"x_steps=(?P<x_steps>[^_]+)__learning_rate=(?P<learning_rate>[^_]+)__"
    r"weight_decay=(?P<weight_decay>[^_]+)__dropout=(?P<dropout>[^_]+)__tao2=(?P<tao2>.+)$"
)

# --- Visual Style Settings, aligned with the other final panel scripts ---
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
BACKGROUND_COLOR = "#f5f5f5"  # LaTeX xcolor: black!4
plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR

TITLE_FONTSIZE = 18
FONTSIZE = 31
X_LABEL_FONTSIZE = 0.60 * 30
TICKLABEL_FONTSIZE = 22
X_TICKLABEL_FONTSIZE = 0.75 * TICKLABEL_FONTSIZE
LEGEND_FONTSIZE = TICKLABEL_FONTSIZE
LINEWIDTH = 2.5
MARKERSIZE = 11
FIGSIZE_X = 13.2
FIGSIZE_Y = 7.1
BOTTOM = 0.36
COORDINATE_AREA_HEIGHT_SCALE = 0.75
TOP = BOTTOM + (0.99 - BOTTOM) * COORDINATE_AREA_HEIGHT_SCALE
LEFT = 0.08
RIGHT = 0.985

STYLE_CONFIGS = {
    Decimal("0.0001"): {"color": "#5372ab", "marker": "s", "linestyle": "-"},
    Decimal("0.001"): {"color": "#936bb9", "marker": "^", "linestyle": "-"},
    Decimal("0.01"): {"color": "#6aa56e", "marker": "o", "linestyle": "-"},
    Decimal("0.1"): {"color": "#c9b97d", "marker": "x", "linestyle": "-"},
    Decimal("1.0"): {"color": "#b75555", "marker": "v", "linestyle": "-"},
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    x_steps: str
    learning_rate: str
    weight_decay: str
    dropout: str
    tao2: str


@dataclass(frozen=True)
class VerifyRecord:
    source: str
    epsilon: Decimal
    epsilon_label: str
    scale_exponent: int
    scale_value: Decimal
    x_index: int
    repeat: int
    rank: int
    candidate: Candidate
    seed: str
    val_acc: float
    test_acc: float
    result_csv_path: Path
    source_job_dir: Path


def scale_from_exponent(exponent: int) -> Decimal:
    if exponent >= 0:
        return Decimal(2) ** exponent
    return Decimal(1) / (Decimal(2) ** abs(exponent))


SCALE_SPECS = tuple((exponent, scale_from_exponent(exponent), rf"$2^{{{exponent}}}$") for exponent in SCALE_EXPONENTS)
SCALE_TO_EXPONENT = {scale: exponent for exponent, scale, _ in SCALE_SPECS}
SCALE_TO_INDEX = {scale: idx for idx, (_, scale, _) in enumerate(SCALE_SPECS)}
SCALE_LABELS = [label for _, _, label in SCALE_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5 final norm-scale curves.")
    parser.add_argument(
        "--figure5-root",
        type=Path,
        default=DEFAULT_FIGURE5_ROOT,
        help="Root containing the main figure5.yaml result directory.",
    )
    parser.add_argument(
        "--figure5-add-root",
        type=Path,
        default=DEFAULT_FIGURE5_ADD_ROOT,
        help="Root containing the add-on figure5.yaml result directory for low r values.",
    )
    parser.add_argument(
        "--figure5-add-again-root",
        type=Path,
        default=DEFAULT_FIGURE5_ADD_AGAIN_ROOT,
        help="Root containing the second add-on figure5.yaml result directory for odd log2 r values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDF/PNG/CSV outputs will be written.",
    )
    parser.add_argument(
        "--featfree-manifest",
        type=Path,
        default=DEFAULT_FEATFREE_MANIFEST,
        help="Manifest for the Cora GraphSAGE FeatFree reference.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help="Number of bootstrap resamples for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def decimal_key(value: Any, *, field: str) -> Decimal:
    if value is None:
        raise ValueError(f"missing {field}")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        raise ValueError(f"missing {field}")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc


def normalize_scalar(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "none"
    try:
        dec = Decimal(text)
    except InvalidOperation:
        return text.lower()
    normalized = format(dec.normalize(), "f").rstrip("0").rstrip(".")
    return normalized or "0"


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest_row(source: str, row: dict[str, str]) -> None:
    expected = {
        "dataset": "cora",
        "feature": "raw",
        "mechanism": "mbm",
        "m": "best",
        "smoother": "hoa",
        "backbone": "sage",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{source}: expected {field}={value}, found {row.get(field)!r}")
    if row.get("norm") not in {"True", "true", "1"}:
        raise ValueError(f"{source}: expected norm=true, found {row.get('norm')!r}")
    if row.get("use_nfr") not in {"False", "false", "0"}:
        raise ValueError(f"{source}: expected use_nfr=false, found {row.get('use_nfr')!r}")


def build_manifest_index(source: str, manifest_path: Path, *, expected_rows: int) -> dict[tuple[Decimal, Decimal], dict[str, str]]:
    rows = load_manifest(manifest_path)
    if len(rows) != expected_rows:
        raise ValueError(f"{manifest_path}: expected {expected_rows} rows, found {len(rows)}")

    index: dict[tuple[Decimal, Decimal], dict[str, str]] = {}
    for row in rows:
        validate_manifest_row(source, row)
        epsilon = decimal_key(row.get("x_eps"), field="x_eps")
        scale = decimal_key(row.get("norm_scale"), field="norm_scale")
        key = (epsilon, scale)
        if key in index:
            raise ValueError(f"{source}: duplicate manifest key {key}")
        index[key] = row
    return index


def load_best_candidate(job_dir: Path) -> Candidate:
    path = job_dir / "best_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing best_config.yaml: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    raw = data.get("best_candidate")
    if not isinstance(raw, dict):
        raise ValueError(f"best_config.yaml has no best_candidate mapping: {path}")
    return Candidate(
        candidate_id=int(raw["candidate_id"]),
        x_steps=normalize_scalar(raw.get("x_steps")),
        learning_rate=normalize_scalar(raw.get("learning_rate")),
        weight_decay=normalize_scalar(raw.get("weight_decay")),
        dropout=normalize_scalar(raw.get("dropout")),
        tao2=normalize_scalar(raw.get("tao2")),
    )


def parse_verify_dir(path: Path) -> dict[str, str | int] | None:
    match = VERIFY_DIR_RE.match(path.name)
    if match is None:
        return None
    groups = match.groupdict()
    return {
        "rank": int(groups["rank"]),
        "repeat": int(groups["repeat"]),
        "candidate_id": int(groups["candidate"]),
        "x_steps": normalize_scalar(groups["x_steps"]),
        "learning_rate": normalize_scalar(groups["learning_rate"]),
        "weight_decay": normalize_scalar(groups["weight_decay"]),
        "dropout": normalize_scalar(groups["dropout"]),
        "tao2": normalize_scalar(groups["tao2"]),
    }


def candidate_matches_dir(candidate: Candidate, parsed: dict[str, str | int]) -> bool:
    return (
        int(parsed["candidate_id"]) == candidate.candidate_id
        and parsed["x_steps"] == candidate.x_steps
        and parsed["learning_rate"] == candidate.learning_rate
        and parsed["weight_decay"] == candidate.weight_decay
        and parsed["dropout"] == candidate.dropout
        and parsed["tao2"] == candidate.tao2
    )


def read_single_result_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in result CSV, found {len(rows)}: {path}")
    return rows[0]


def require_float(row: dict[str, str], field: str, path: Path) -> float:
    value = row.get(field, "")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}={value!r} in {path}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}={value!r} in {path}")
    return result


def collect_verify_records(
    source: str,
    epsilon: Decimal,
    scale_exponent: int,
    scale_value: Decimal,
    job_dir: Path,
) -> list[VerifyRecord]:
    candidate = load_best_candidate(job_dir)
    verify_root = job_dir / "verify_top5"
    if not verify_root.is_dir():
        raise FileNotFoundError(f"Missing verify_top5 directory: {verify_root}")

    records: list[VerifyRecord] = []
    for verify_dir in sorted(verify_root.glob("rank=*__repeat=*__candidate=*__*")):
        if not verify_dir.is_dir():
            continue
        parsed = parse_verify_dir(verify_dir)
        if parsed is None or not candidate_matches_dir(candidate, parsed):
            continue

        csv_paths = sorted(verify_dir.glob("*.csv"))
        if len(csv_paths) != 1:
            raise ValueError(f"Expected one result CSV in {verify_dir}, found {len(csv_paths)}")
        csv_path = csv_paths[0]
        result_row = read_single_result_csv(csv_path)
        records.append(
            VerifyRecord(
                source=source,
                epsilon=epsilon,
                epsilon_label=dict(EPSILON_SPECS)[epsilon],
                scale_exponent=scale_exponent,
                scale_value=scale_value,
                x_index=SCALE_TO_INDEX[scale_value],
                repeat=int(parsed["repeat"]),
                rank=int(parsed["rank"]),
                candidate=candidate,
                seed=result_row.get("seed", ""),
                val_acc=require_float(result_row, "val/acc", csv_path),
                test_acc=require_float(result_row, "test/acc", csv_path),
                result_csv_path=csv_path,
                source_job_dir=job_dir,
            )
        )

    repeats = sorted(record.repeat for record in records)
    expected_repeats = list(range(1, EXPECTED_REPEATS + 1))
    if repeats != expected_repeats:
        raise ValueError(
            f"{source}/eps={epsilon}/r=2^{scale_exponent}: expected repeats {expected_repeats}, "
            f"found {repeats} for candidate {candidate.candidate_id} in {job_dir}"
        )
    return records


def source_for_scale_exponent(scale_exponent: int) -> str:
    if scale_exponent in ADD_SOURCE_EXPONENTS:
        return "figure5_final_add"
    if scale_exponent in ADD_AGAIN_SOURCE_EXPONENTS:
        return "figure5_final_add_again"
    return "figure5_final"


def load_long_records(figure5_root: Path, figure5_add_root: Path, figure5_add_again_root: Path) -> list[VerifyRecord]:
    main_index = build_manifest_index(
        "figure5_final",
        figure5_root / "figure5.yaml" / "manifest.csv",
        expected_rows=96,
    )
    add_index = build_manifest_index(
        "figure5_final_add",
        figure5_add_root / "figure5.yaml" / "manifest.csv",
        expected_rows=12,
    )
    add_again_index = build_manifest_index(
        "figure5_final_add_again",
        figure5_add_again_root / "figure5.yaml" / "manifest.csv",
        expected_rows=72,
    )

    records: list[VerifyRecord] = []
    seen_points: set[tuple[Decimal, int]] = set()
    for epsilon, _ in EPSILON_SPECS:
        for scale_exponent, scale_value, _ in SCALE_SPECS:
            source = source_for_scale_exponent(scale_exponent)
            if source == "figure5_final_add":
                index = add_index
            elif source == "figure5_final_add_again":
                index = add_again_index
            else:
                index = main_index
            key = (epsilon, scale_value)
            if key not in index:
                raise ValueError(f"{source}: missing manifest row for eps={epsilon}, r=2^{scale_exponent}")
            if (epsilon, scale_exponent) in seen_points:
                raise ValueError(f"Duplicate target point eps={epsilon}, r=2^{scale_exponent}")
            seen_points.add((epsilon, scale_exponent))
            records.extend(
                collect_verify_records(
                    source=source,
                    epsilon=epsilon,
                    scale_exponent=scale_exponent,
                    scale_value=scale_value,
                    job_dir=Path(index[key]["job_dir"]),
                )
            )

    expected_points = len(EPSILON_SPECS) * len(SCALE_SPECS)
    if len(seen_points) != expected_points:
        raise ValueError(f"Expected {expected_points} target points, found {len(seen_points)}")
    expected_records = expected_points * EXPECTED_REPEATS
    if len(records) != expected_records:
        raise ValueError(f"Expected {expected_records} long records, found {len(records)}")
    return records


def bootstrap_ci(values: list[float], *, samples: int, seed: int, key: tuple[Any, ...]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size != EXPECTED_REPEATS:
        raise ValueError(f"Expected {EXPECTED_REPEATS} values for bootstrap, found {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values for bootstrap key={key}")
    mean = float(array.mean())
    key_text = "|".join([str(seed), *(str(item) for item in key)])
    key_seed = int(hashlib.sha256(key_text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(key_seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return mean, float(low), float(high)


def build_plot_rows(records: list[VerifyRecord], *, bootstrap_samples: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[Decimal, int], list[VerifyRecord]] = {}
    for record in records:
        grouped.setdefault((record.epsilon, record.scale_exponent), []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (EPSILON_TO_INDEX[item[0][0]], item[0][1])):
        epsilon, scale_exponent = key
        if len(group) != EXPECTED_REPEATS:
            raise ValueError(f"{key}: expected {EXPECTED_REPEATS} records, found {len(group)}")
        first = group[0]
        val_values = [record.val_acc for record in group]
        test_values = [record.test_acc for record in group]
        test_mean, ci_low, ci_high = bootstrap_ci(
            test_values,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            key=key,
        )
        val_array = np.asarray(val_values, dtype=float)
        test_array = np.asarray(test_values, dtype=float)
        rows.append(
            {
                "source": first.source,
                "epsilon": str(epsilon),
                "epsilon_label": first.epsilon_label,
                "scale_exponent": scale_exponent,
                "norm_scale": str(first.scale_value),
                "norm_scale_label": rf"$2^{{{scale_exponent}}}$",
                "x_index": first.x_index,
                "n": len(group),
                "candidate_id": first.candidate.candidate_id,
                "verify_rank": first.rank,
                "x_steps": first.candidate.x_steps,
                "learning_rate": first.candidate.learning_rate,
                "weight_decay": first.candidate.weight_decay,
                "dropout": first.candidate.dropout,
                "tao2": first.candidate.tao2,
                "val_acc_mean": float(val_array.mean()),
                "val_acc_std": float(val_array.std(ddof=1)),
                "test_acc_mean": test_mean,
                "test_acc_std": float(test_array.std(ddof=1)),
                "test_acc_ci_low": ci_low,
                "test_acc_ci_high": ci_high,
                "source_job_dir": str(first.source_job_dir),
            }
        )
    return rows


def write_long_csv(path: Path, records: list[VerifyRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "epsilon",
        "epsilon_label",
        "scale_exponent",
        "norm_scale",
        "norm_scale_label",
        "x_index",
        "repeat",
        "verify_rank",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "seed",
        "val_acc",
        "test_acc",
        "source_job_dir",
        "result_csv_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (EPSILON_TO_INDEX[r.epsilon], r.scale_exponent, r.repeat)):
            writer.writerow(
                {
                    "source": record.source,
                    "epsilon": str(record.epsilon),
                    "epsilon_label": record.epsilon_label,
                    "scale_exponent": record.scale_exponent,
                    "norm_scale": str(record.scale_value),
                    "norm_scale_label": rf"$2^{{{record.scale_exponent}}}$",
                    "x_index": record.x_index,
                    "repeat": record.repeat,
                    "verify_rank": record.rank,
                    "candidate_id": record.candidate.candidate_id,
                    "x_steps": record.candidate.x_steps,
                    "learning_rate": record.candidate.learning_rate,
                    "weight_decay": record.candidate.weight_decay,
                    "dropout": record.candidate.dropout,
                    "tao2": record.candidate.tao2,
                    "seed": record.seed,
                    "val_acc": record.val_acc,
                    "test_acc": record.test_acc,
                    "source_job_dir": record.source_job_dir,
                    "result_csv_path": record.result_csv_path,
                }
            )


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "epsilon",
        "epsilon_label",
        "scale_exponent",
        "norm_scale",
        "norm_scale_label",
        "x_index",
        "n",
        "candidate_id",
        "verify_rank",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "val_acc_mean",
        "val_acc_std",
        "test_acc_mean",
        "test_acc_std",
        "test_acc_ci_low",
        "test_acc_ci_high",
        "source_job_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (EPSILON_TO_INDEX[Decimal(str(r["epsilon"]))], int(r["scale_exponent"]))):
            writer.writerow(row)


def validate_plot_rows(rows: list[dict[str, Any]]) -> None:
    expected = len(EPSILON_SPECS) * len(SCALE_SPECS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} plot rows, found {len(rows)}")
    seen = {(Decimal(str(row["epsilon"])), int(row["scale_exponent"])) for row in rows}
    if len(seen) != expected:
        raise RuntimeError(f"Duplicate plot rows detected: unique={len(seen)} expected={expected}")
    for epsilon, _ in EPSILON_SPECS:
        curve = [row for row in rows if Decimal(str(row["epsilon"])) == epsilon]
        if len(curve) != len(SCALE_SPECS):
            raise RuntimeError(f"{epsilon}: expected {len(SCALE_SPECS)} scale points, found {len(curve)}")
    for row in rows:
        mean = float(row["test_acc_mean"])
        low = float(row["test_acc_ci_low"])
        high = float(row["test_acc_ci_high"])
        if int(row["n"]) != EXPECTED_REPEATS:
            raise RuntimeError(f"Expected n={EXPECTED_REPEATS} for {row}")
        if not (low <= mean <= high):
            raise RuntimeError(f"CI does not contain mean for {row}")


def load_featfree_reference(manifest_path: Path) -> float:
    manifest_rows = load_manifest(manifest_path)
    if len(manifest_rows) != 1:
        raise RuntimeError(
            f"Expected one FeatFree row in {manifest_path}, found {len(manifest_rows)}"
        )
    manifest_row = manifest_rows[0]
    expected_manifest = {
        "dataset": "cora",
        "feature": "random_normal",
        "feature_dim": "1433",
        "mechanism": "mbm",
        "x_eps": "inf",
        "m": "best",
        "norm": "false",
        "smoother": "hoa",
        "backbone": "sage",
        "use_nfr": "false",
    }
    for field, expected in expected_manifest.items():
        actual = manifest_row.get(field, "").strip().lower()
        if actual != expected:
            raise RuntimeError(
                f"Unexpected FeatFree {field} in {manifest_path}: "
                f"expected {expected!r}, found {actual!r}"
            )
    if manifest_row.get("search_status", "") not in {"completed", "skipped_existing_result"}:
        raise RuntimeError(
            f"Incomplete FeatFree result in {manifest_path}: "
            f"{manifest_row.get('search_status')!r}"
        )

    job_dir = Path(manifest_row["job_dir"])
    candidate = load_best_candidate(job_dir)
    if candidate.candidate_id != int(manifest_row["best_candidate_id"]):
        raise RuntimeError(
            f"FeatFree candidate mismatch: best_config={candidate.candidate_id}, "
            f"manifest={manifest_row['best_candidate_id']}"
        )
    expected_hparams = {
        "x_steps": candidate.x_steps,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "dropout": candidate.dropout,
        "tao2": candidate.tao2,
    }
    verify_root = job_dir / "verify_top5"
    if not verify_root.is_dir():
        raise FileNotFoundError(f"Missing FeatFree verify_top5 directory: {verify_root}")

    repeat_dirs: dict[int, Path] = {}
    for verify_dir in sorted(verify_root.glob("rank=*__repeat=*__candidate=*__*")):
        if not verify_dir.is_dir():
            continue
        parsed = parse_verify_dir(verify_dir)
        if parsed is None or not candidate_matches_dir(candidate, parsed):
            continue
        repeat = int(parsed["repeat"])
        if repeat in repeat_dirs:
            raise RuntimeError(f"Duplicate FeatFree repeat {repeat} in {verify_root}")
        repeat_dirs[repeat] = verify_dir
    if sorted(repeat_dirs) != list(range(1, EXPECTED_REPEATS + 1)):
        raise RuntimeError(
            f"Expected FeatFree repeats 1..{EXPECTED_REPEATS}, found {sorted(repeat_dirs)}"
        )

    test_accs: list[float] = []
    seeds: list[str] = []
    for repeat in range(1, EXPECTED_REPEATS + 1):
        csv_paths = sorted(repeat_dirs[repeat].glob("*.csv"))
        if len(csv_paths) != 1:
            raise RuntimeError(
                f"Expected one FeatFree CSV in {repeat_dirs[repeat]}, found {len(csv_paths)}"
            )
        csv_path = csv_paths[0]
        result = read_single_result_csv(csv_path)
        expected_result = {
            "dataset": "cora",
            "feature": "random_normal",
            "feature_dim": "1433",
            "smoother": "hoa",
            "model": "sage",
        }
        for field, expected in expected_result.items():
            if result.get(field, "").strip().lower() != expected:
                raise RuntimeError(f"Unexpected FeatFree {field} in {csv_path}")
        for field, expected in expected_hparams.items():
            if normalize_scalar(result.get(field)) != expected:
                raise RuntimeError(f"FeatFree hyperparameter mismatch for {field} in {csv_path}")
        test_accs.append(require_float(result, "test/acc", csv_path))
        seeds.append(result.get("seed", ""))

    if len(set(seeds)) != EXPECTED_REPEATS:
        raise RuntimeError(f"FeatFree verify seeds are not distinct: {seeds}")
    featfree_mean = float(np.mean(test_accs))
    featfree_std = float(np.std(test_accs, ddof=1))
    for field, actual in (
        ("best_verify_test_acc_mean", featfree_mean),
        ("best_verify_test_acc_std", featfree_std),
    ):
        expected = float(manifest_row[field])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"FeatFree manifest/raw mismatch for {field}: manifest={expected}, raw={actual}"
            )
    print(
        f"FeatFree reference: {featfree_mean:.12f} "
        f"(std={featfree_std:.12f}, n={len(test_accs)})"
    )
    return featfree_mean


def plot_curves(
    rows: list[dict[str, Any]], featfree_acc: float, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(FIGSIZE_X, FIGSIZE_Y))
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT)

    for epsilon, epsilon_label in EPSILON_SPECS:
        sub = sorted(
            [row for row in rows if Decimal(str(row["epsilon"])) == epsilon],
            key=lambda row: int(row["x_index"]),
        )
        if len(sub) != len(SCALE_SPECS):
            raise RuntimeError(f"{epsilon}: expected {len(SCALE_SPECS)} plot points, found {len(sub)}")
        style = STYLE_CONFIGS[epsilon]
        x = np.asarray([float(row["norm_scale"]) for row in sub], dtype=float)
        y = np.asarray([float(row["test_acc_mean"]) for row in sub], dtype=float)
        low = np.asarray([float(row["test_acc_ci_low"]) for row in sub], dtype=float)
        high = np.asarray([float(row["test_acc_ci_high"]) for row in sub], dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - low, high - y]),
            label=epsilon_label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            capsize=0,
            markerfacecolor="none",
            markeredgewidth=2,
        )

    ax.axhline(
        featfree_acc,
        color="black",
        linestyle="--",
        linewidth=2.4,
        alpha=0.95,
        zorder=1,
        label=FEATFREE_LABEL,
    )

    data_y_bottom = min(
        featfree_acc,
        min(float(row["test_acc_ci_low"]) for row in rows),
    )
    data_y_top = max(
        featfree_acc,
        max(float(row["test_acc_ci_high"]) for row in rows),
    )
    ax.set_ylim(bottom=data_y_bottom, top=data_y_top)

    scale_values = [float(scale_value) for _, scale_value, _ in SCALE_SPECS]
    ax.set_xscale("log", base=2)
    ax.set_xticks(scale_values)
    ax.set_xticklabels(SCALE_LABELS, rotation=35, ha="right")
    ax.axvline(float(Decimal("1")), color="black", linestyle="--", linewidth=2.4, alpha=0.95, zorder=0)
    ax.set_xlabel(
        r"$r$",
        fontsize=X_LABEL_FONTSIZE,
        fontweight="medium",
        labelpad=0,
    )
    ax.set_ylabel("Test Accuracy", fontsize=FONTSIZE, fontweight="medium")
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
    ax.tick_params(axis="x", which="major", labelsize=X_TICKLABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=X_TICKLABEL_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    epsilon_labels = [epsilon_label for _, epsilon_label in EPSILON_SPECS]
    first_labels = epsilon_labels[:3]
    second_labels = [*epsilon_labels[3:], FEATFREE_LABEL]
    first_legend = ax.legend(
        [handles_by_label[label] for label in first_labels],
        first_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.39),
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0.2,
        handlelength=1.5,
        columnspacing=0.85,
    )
    ax.add_artist(first_legend)
    ax.legend(
        [handles_by_label[label] for label in second_labels],
        second_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.52),
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0.2,
        handlelength=1.5,
        columnspacing=0.85,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure5_final_norm_scale_curves.pdf"
    png_path = output_dir / "figure5_final_norm_scale_curves.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    records = load_long_records(args.figure5_root, args.figure5_add_root, args.figure5_add_again_root)
    plot_rows = build_plot_rows(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    validate_plot_rows(plot_rows)

    long_csv_path = args.output_dir / "figure5_final_norm_scale_long.csv"
    plot_data_path = args.output_dir / "figure5_final_norm_scale_plot_data.csv"
    write_long_csv(long_csv_path, records)
    write_plot_data(plot_data_path, plot_rows)
    featfree_acc = load_featfree_reference(args.featfree_manifest)
    plot_curves(plot_rows, featfree_acc, args.output_dir)
    print(f"Saved {long_csv_path}")
    print(f"Saved {plot_data_path}")
    print(f"Long rows: {len(records)}")
    print(f"Plot rows: {len(plot_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
