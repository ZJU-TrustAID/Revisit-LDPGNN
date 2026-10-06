#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "scripts" / "aec" / "reference"
DEFAULT_MAIN_LONG_CSV = PAPER_ROOT / "main_add_10seed_backfill" / "test_acc_long.csv"
DEFAULT_FEATFREE_ROOT = REPO_ROOT / "scripts" / "aec" / "reference"
DEFAULT_CLEAN_REFERENCE_MANIFEST = PAPER_ROOT / "clean_reference" / "manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rebuttal_figure"
OUTPUT_STEM = "figure1"
PLOT_DATA_FILENAME = f"{OUTPUT_STEM}_plot_data.csv"

FROZEN_STYLE_VERSION = "paper_figure1_reference_v1"
EXPECTED_PYTHON_VERSION = "3.10.20"
EXPECTED_MATPLOTLIB_VERSION = "3.10.9"
EXPECTED_ENV_PREFIX = None

BACKBONES = ("gcn", "sage", "gat")
BACKBONE_LABELS = MappingProxyType({"gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"})
DATASETS = ("cora", "lastfm", "citeseer", "facebook")
DATASET_LABELS = MappingProxyType(
    {
        "cora": "Cora",
        "lastfm": "Lastfm",
        "citeseer": "Citeseer",
        "facebook": "Facebook",
    }
)
X_EPS_VALUES = ("0.001", "0.01", "0.1", "1.0", "2.0", "3.0", "4.0", "6.0", "8.0", "10.0")
PIPELINE_LABELS = MappingProxyType(
    {
        "figure3_pipeline1": r"$\mathsf{LPGNN}$",
        "figure3_pipeline2": r"$\mathsf{PrivGE}$",
        "figure3_pipeline3": r"$\mathsf{UPGNet\text{-}MBM}$",
        "figure3_pipeline4": r"$\mathsf{UPGNet\text{-}PM}$",
    }
)
FEATFREE_LABEL = r"$\mathsf{FeatFree}$"
NON_PRIVATE_LABEL = r"$\mathsf{Non\text{-}private}$"
MAJORITY_LABEL = r"$\mathsf{Majority\text{-}Class}$"
MAJORITY_BASELINE_PCT = MappingProxyType(
    {
        "cora": 100 * 818 / 2708,
        "lastfm": 100 * 1572 / 7083,
        "citeseer": 100 * 701 / 3327,
        "facebook": 100 * 6880 / 22470,
    }
)
VERIFY_DIR_RE = re.compile(r"^rank=(\d+)__repeat=(\d+)__candidate=(\d+)__")

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 12345

# --- Visual Style Settings, aligned with plot/*.py examples ---
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
BACKGROUND_COLOR = "#f5f5f5"  # LaTeX xcolor: black!4
plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR

TITLE_FONTSIZE = 18
FONTSIZE = 18
X_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 24
TICKLABEL_FONTSIZE = 15
LINEWIDTH = 2
MARKERSIZE = 8
FIGSIZE_X = 20
FIGSIZE_Y = 11.5
BOTTOM = 0.14
TOP = 0.92
LEFT = 0.065
RIGHT = 0.99
HSPACE = 0.34
WSPACE = 0.16
TITLE_PAD = 18
Y_LABEL_PAD = 14
MARKEREDGEWIDTH = 2
GRID_COLOR = "white"
GRID_LINESTYLE = "-"
GRID_LINEWIDTH = 1
GRID_ALPHA = 1.0
LEGEND_BBOX = (0.5, 0.005)
LEGEND_BORDERPAD = 0.2
LEGEND_HANDLELENGTH = 1.3
LEGEND_COLUMNSPACING = 0.75
SAVE_DPI = 300
SAVE_PAD_INCHES = 0.02
MAIN_AXIS_HEIGHT_FRACTION = 0.90
BASELINE_AXIS_HEIGHT_FRACTION = 0.075
AXIS_BREAK_GAP_FRACTION = 0.025
BASELINE_AXIS_HALF_RANGE = 1.0
MAJORITY_BASELINE_COLOR = "#000000"
MAJORITY_BASELINE_LINESTYLE = "--"
MAJORITY_BASELINE_LINEWIDTH = 2
MAJORITY_BASELINE_ZORDER = 1.5
AXIS_BREAK_MARK_SIZE = 0.012
AXIS_BREAK_MARK_LINEWIDTH = 1.2

STYLE_CONFIGS = MappingProxyType(
    {
        PIPELINE_LABELS["figure3_pipeline1"]: MappingProxyType(
            {"color": "#5372ab", "marker": "s", "linestyle": "-"}
        ),
        PIPELINE_LABELS["figure3_pipeline2"]: MappingProxyType(
            {"color": "#936bb9", "marker": "^", "linestyle": "-"}
        ),
        PIPELINE_LABELS["figure3_pipeline3"]: MappingProxyType(
            {"color": "#6aa56e", "marker": "o", "linestyle": "-"}
        ),
        PIPELINE_LABELS["figure3_pipeline4"]: MappingProxyType(
            {"color": "#c9b97d", "marker": "x", "linestyle": "-"}
        ),
        FEATFREE_LABEL: MappingProxyType(
            {"color": "#b75555", "marker": "D", "linestyle": "--"}
        ),
        NON_PRIVATE_LABEL: MappingProxyType(
            {"color": "#2f7fb8", "marker": "P", "linestyle": "--"}
        ),
    }
)
LINE_ORDER = (
    PIPELINE_LABELS["figure3_pipeline1"],
    PIPELINE_LABELS["figure3_pipeline2"],
    PIPELINE_LABELS["figure3_pipeline3"],
    PIPELINE_LABELS["figure3_pipeline4"],
    FEATFREE_LABEL,
    NON_PRIVATE_LABEL,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    x_steps: int
    learning_rate: str
    weight_decay: str
    dropout: str
    tao2: str


@dataclass
class FeatFreeChoice:
    backbone: str
    dataset: str
    smoother: str
    feature_dim: int
    candidate: Candidate
    val_accs: list[float]
    test_accs: list[float]
    source_job_dir: Path
    actual_verify_rank: int | None

    @property
    def val_mean(self) -> float:
        return float(np.mean(self.val_accs))

    @property
    def test_mean(self) -> float:
        return float(np.mean(self.test_accs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the frozen 3x4 main-add 10-seed panels with HOA FeatFree rerun data."
    )
    parser.add_argument("--main-long-csv", type=Path, default=DEFAULT_MAIN_LONG_CSV)
    parser.add_argument("--featfree-root", type=Path, default=DEFAULT_FEATFREE_ROOT)
    parser.add_argument("--clean-reference-manifest", type=Path, default=DEFAULT_CLEAN_REFERENCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def validate_render_environment() -> None:
    problems: list[str] = []
    actual_prefix = Path(sys.prefix).resolve()
    if EXPECTED_ENV_PREFIX is not None and actual_prefix != EXPECTED_ENV_PREFIX.resolve():
        problems.append(f"environment prefix is {actual_prefix}, expected {EXPECTED_ENV_PREFIX}")
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        problems.append(
            f"Python is {platform.python_version()}, expected {EXPECTED_PYTHON_VERSION}"
        )
    if matplotlib.__version__ != EXPECTED_MATPLOTLIB_VERSION:
        problems.append(
            f"Matplotlib is {matplotlib.__version__}, expected {EXPECTED_MATPLOTLIB_VERSION}"
        )
    if problems:
        detail = "; ".join(problems)
        raise RuntimeError(
            f"Frozen render environment mismatch ({FROZEN_STYLE_VERSION}): {detail}. "
            f"Run with {EXPECTED_ENV_PREFIX / 'bin' / 'python'}."
        )


def canonical_x_eps(raw_value: Any) -> str:
    value = float(raw_value)
    for candidate in X_EPS_VALUES:
        if math.isclose(value, float(candidate), rel_tol=0.0, abs_tol=1e-12):
            return candidate
    raise ValueError(f"Unsupported x_eps value: {raw_value!r}")


def metric_stats(values: list[float], *, bootstrap_samples: int, bootstrap_seed: int, key: tuple[Any, ...]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size != 10:
        raise ValueError(f"Expected 10 values for {key}, found {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite values for {key}: {values}")

    mean = float(np.mean(arr))
    digest = hashlib.sha256("|".join(map(str, key)).encode("utf-8")).hexdigest()
    seed = (int(digest[:8], 16) ^ int(bootstrap_seed)) % (2**32)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, arr.size, size=(bootstrap_samples, arr.size))
    sample_means = arr[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(sample_means, [2.5, 97.5])
    return {
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def candidate_from_best_config(path: Path) -> Candidate:
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid best_config.yaml: {path}")
    raw = data["best_candidate"]
    return Candidate(
        candidate_id=int(raw["candidate_id"]),
        x_steps=int(raw["x_steps"]),
        learning_rate=f"{float(raw['learning_rate']):.12g}",
        weight_decay=f"{float(raw['weight_decay']):.12g}",
        dropout=f"{float(raw['dropout']):.12g}",
        tao2="none" if str(raw.get("tao2", "none")).lower() == "none" else f"{float(raw['tao2']):.12g}",
    )


def row_float(row: dict[str, str], name: str, *, csv_path: Path) -> float:
    try:
        value = float(row[name])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"Invalid {name!r} in {csv_path}: {row.get(name)!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {name!r} in {csv_path}: {value}")
    return value


def row_candidate_matches(row: dict[str, str], candidate: Candidate, *, csv_path: Path) -> bool:
    tao2_raw = row.get("tao2", "").strip()
    tao2 = "none" if not tao2_raw or tao2_raw.lower() == "none" else f"{float(tao2_raw):.12g}"
    row_candidate = Candidate(
        candidate_id=candidate.candidate_id,
        x_steps=int(row_float(row, "x_steps", csv_path=csv_path)),
        learning_rate=f"{row_float(row, 'learning_rate', csv_path=csv_path):.12g}",
        weight_decay=f"{row_float(row, 'weight_decay', csv_path=csv_path):.12g}",
        dropout=f"{row_float(row, 'dropout', csv_path=csv_path):.12g}",
        tao2=tao2,
    )
    return row_candidate == candidate


def collect_reference_repeats(job_dir: Path, candidate: Candidate) -> tuple[list[float], list[float], int | None]:
    verify_dir = job_dir / "verify_top5"
    if not verify_dir.is_dir():
        raise RuntimeError(f"Missing verify_top5 directory: {verify_dir}")

    repeat_dirs: dict[int, tuple[int, Path]] = {}
    for child in sorted(verify_dir.iterdir()):
        if not child.is_dir():
            continue
        match = VERIFY_DIR_RE.match(child.name)
        if match is None:
            continue
        rank = int(match.group(1))
        repeat_id = int(match.group(2))
        candidate_id = int(match.group(3))
        if candidate_id != candidate.candidate_id:
            continue
        if repeat_id in repeat_dirs:
            raise RuntimeError(f"Duplicate verify repeat {repeat_id} for candidate {candidate.candidate_id} in {verify_dir}")
        repeat_dirs[repeat_id] = (rank, child)

    if sorted(repeat_dirs) != list(range(1, 11)):
        raise RuntimeError(
            f"Expected repeats 1..10 for candidate {candidate.candidate_id} in {verify_dir}, got {sorted(repeat_dirs)}"
        )

    val_accs: list[float] = []
    test_accs: list[float] = []
    observed_ranks: set[int] = set()
    for repeat_id in range(1, 11):
        rank, repeat_dir = repeat_dirs[repeat_id]
        observed_ranks.add(rank)
        csv_files = sorted(repeat_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(f"Expected exactly one CSV in {repeat_dir}, found {len(csv_files)}")
        csv_path = csv_files[0]
        rows = read_csv_rows(csv_path)
        if len(rows) != 1:
            raise RuntimeError(f"Expected exactly one row in {csv_path}, found {len(rows)}")
        row = rows[0]
        if not row_candidate_matches(row, candidate, csv_path=csv_path):
            raise RuntimeError(f"Candidate hyperparameter mismatch in {csv_path}")
        val_accs.append(row_float(row, "val/acc", csv_path=csv_path))
        test_accs.append(row_float(row, "test/acc", csv_path=csv_path))

    actual_rank = next(iter(observed_ranks)) if len(observed_ranks) == 1 else None
    return val_accs, test_accs, actual_rank


def load_main_curves(
    path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    if len(df) != 6000:
        raise RuntimeError(f"Expected 6000 rows in {path}, found {len(df)}")

    df = df.copy()
    df["x_eps"] = df["x_eps"].map(canonical_x_eps)
    df = df[
        df["backbone"].isin(BACKBONES)
        & df["dataset"].isin(DATASETS)
        & df["pipeline"].isin(PIPELINE_LABELS)
    ]
    expected_groups = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES) * len(PIPELINE_LABELS)
    grouped = df.groupby(["backbone", "pipeline", "dataset", "x_eps"], sort=False)
    if len(grouped) != expected_groups:
        raise RuntimeError(f"Expected {expected_groups} main groups, found {len(grouped)}")

    rows: list[dict[str, Any]] = []
    for (backbone, pipeline, dataset, x_eps), group in grouped:
        if len(group) != 10:
            raise RuntimeError(f"Expected 10 rows for {(backbone, pipeline, dataset, x_eps)}, found {len(group)}")
        stats = metric_stats(
            [float(v) for v in group["test_acc"]],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("main", backbone, pipeline, dataset, x_eps),
        )
        val_stats = metric_stats(
            [float(v) for v in group["val_acc"]],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("main-val", backbone, pipeline, dataset, x_eps),
        )
        first = group.iloc[0]
        rows.append(
            {
                "source": "main_add_10seed",
                "backbone": backbone,
                "dataset": dataset,
                "x_eps": x_eps,
                "x_index": X_EPS_VALUES.index(x_eps),
                "line_label": PIPELINE_LABELS[pipeline],
                "pipeline": pipeline,
                "smoother": str(first["smoother"]),
                "feature_dim": "",
                "candidate_id": int(first["candidate_id"]),
                "x_steps": int(first["x_steps"]),
                "learning_rate": first["learning_rate"],
                "weight_decay": first["weight_decay"],
                "dropout": first["dropout"],
                "tao2": first["tao2"],
                "test_acc_mean": stats["mean"],
                "test_acc_ci_low": stats["ci_low"],
                "test_acc_ci_high": stats["ci_high"],
                "test_acc_std": stats["std"],
                "test_acc_min": stats["min"],
                "test_acc_max": stats["max"],
                "val_acc_mean": val_stats["mean"],
                "val_acc_ci_low": val_stats["ci_low"],
                "val_acc_ci_high": val_stats["ci_high"],
                "n": stats["n"],
            }
        )
    return rows


def assert_manifest_stat_matches(
    row: dict[str, str],
    field: str,
    actual: float,
    *,
    manifest_path: Path,
) -> None:
    expected = row_float(row, field, csv_path=manifest_path)
    if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Manifest/raw mismatch for {field!r} in {manifest_path}: "
            f"manifest={expected} raw={actual}"
        )


def load_featfree_choices(featfree_root: Path) -> dict[tuple[str, str], FeatFreeChoice]:
    choices: dict[tuple[str, str], FeatFreeChoice] = {}
    complete_statuses = {"completed", "skipped_existing_result"}

    for backbone in BACKBONES:
        for dataset in DATASETS:
            manifest_path = (
                featfree_root / dataset / backbone / "random_projected.yaml" / "manifest.csv"
            )
            manifest_rows = read_csv_rows(manifest_path)
            if len(manifest_rows) != 1:
                raise RuntimeError(
                    f"Expected exactly one FeatFree row in {manifest_path}, found {len(manifest_rows)}"
                )
            row = manifest_rows[0]
            expected_fields = {
                "dataset": dataset,
                "backbone": backbone,
                "feature": "random_normal",
                "smoother": "hoa",
            }
            for field, expected in expected_fields.items():
                actual = row.get(field, "").lower()
                if actual != expected:
                    raise RuntimeError(
                        f"Unexpected {field} in {manifest_path}: expected {expected!r}, got {actual!r}"
                    )
            if row.get("search_status", "") not in complete_statuses:
                raise RuntimeError(
                    f"Incomplete FeatFree job in {manifest_path}: {row.get('search_status', '')!r}"
                )

            feature_dim = int(row_float(row, "feature_dim", csv_path=manifest_path))
            job_dir = Path(row["job_dir"])
            best_config_path = Path(row["best_config_path"])
            if not job_dir.is_dir():
                raise RuntimeError(f"Missing FeatFree job directory: {job_dir}")
            if not best_config_path.is_file():
                raise RuntimeError(f"Missing FeatFree best_config.yaml: {best_config_path}")
            if best_config_path.parent.resolve() != job_dir.resolve():
                raise RuntimeError(
                    f"FeatFree best_config_path is outside job_dir in {manifest_path}: "
                    f"{best_config_path} vs {job_dir}"
                )

            candidate = candidate_from_best_config(best_config_path)
            manifest_candidate_id = int(
                row_float(row, "best_candidate_id", csv_path=manifest_path)
            )
            if candidate.candidate_id != manifest_candidate_id:
                raise RuntimeError(
                    f"FeatFree candidate mismatch in {manifest_path}: "
                    f"manifest={manifest_candidate_id} best_config={candidate.candidate_id}"
                )
            val_accs, test_accs, actual_rank = collect_reference_repeats(job_dir, candidate)
            assert_manifest_stat_matches(
                row,
                "best_verify_val_acc_mean",
                float(np.mean(val_accs)),
                manifest_path=manifest_path,
            )
            assert_manifest_stat_matches(
                row,
                "best_verify_test_acc_mean",
                float(np.mean(test_accs)),
                manifest_path=manifest_path,
            )
            assert_manifest_stat_matches(
                row,
                "best_verify_test_acc_std",
                float(np.std(test_accs, ddof=1)),
                manifest_path=manifest_path,
            )
            choices[(backbone, dataset)] = FeatFreeChoice(
                backbone=backbone,
                dataset=dataset,
                smoother="hoa",
                feature_dim=feature_dim,
                candidate=candidate,
                val_accs=val_accs,
                test_accs=test_accs,
                source_job_dir=job_dir,
                actual_verify_rank=actual_rank,
            )

    expected_count = len(BACKBONES) * len(DATASETS)
    if len(choices) != expected_count:
        raise RuntimeError(f"Expected {expected_count} FeatFree choices, found {len(choices)}")
    return choices


def featfree_plot_rows(
    choices: dict[tuple[str, str], FeatFreeChoice],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (backbone, dataset), choice in sorted(choices.items()):
        test_stats = metric_stats(
            choice.test_accs,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("reference", backbone, dataset, choice.smoother, choice.feature_dim),
        )
        val_stats = metric_stats(
            choice.val_accs,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            key=("reference-val", backbone, dataset, choice.smoother, choice.feature_dim),
        )
        for x_eps in X_EPS_VALUES:
            rows.append(
                {
                    "source": "featfree_homo_rerun_hoa",
                    "backbone": backbone,
                    "dataset": dataset,
                    "x_eps": x_eps,
                    "x_index": X_EPS_VALUES.index(x_eps),
                    "line_label": FEATFREE_LABEL,
                    "pipeline": "featfree",
                    "smoother": choice.smoother,
                    "feature_dim": choice.feature_dim,
                    "candidate_id": choice.candidate.candidate_id,
                    "x_steps": choice.candidate.x_steps,
                    "learning_rate": choice.candidate.learning_rate,
                    "weight_decay": choice.candidate.weight_decay,
                    "dropout": choice.candidate.dropout,
                    "tao2": choice.candidate.tao2,
                    "test_acc_mean": test_stats["mean"],
                    "test_acc_ci_low": test_stats["ci_low"],
                    "test_acc_ci_high": test_stats["ci_high"],
                    "test_acc_std": test_stats["std"],
                    "test_acc_min": test_stats["min"],
                    "test_acc_max": test_stats["max"],
                    "val_acc_mean": val_stats["mean"],
                    "val_acc_ci_low": val_stats["ci_low"],
                    "val_acc_ci_high": val_stats["ci_high"],
                    "n": test_stats["n"],
                    "source_job_dir": str(choice.source_job_dir),
                    "actual_verify_rank": "" if choice.actual_verify_rank is None else choice.actual_verify_rank,
                }
            )
    return rows


def clean_reference_plot_rows(
    manifest_path: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(manifest_path)
    expected_count = len(BACKBONES) * len(DATASETS)
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} clean-reference rows in {manifest_path}, found {len(rows)}")

    by_key = {(row["backbone"], row["dataset"]): row for row in rows}
    missing = [
        (backbone, dataset)
        for backbone in BACKBONES
        for dataset in DATASETS
        if (backbone, dataset) not in by_key
    ]
    if missing:
        raise RuntimeError(f"Missing clean-reference entries: {missing}")

    plot_rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            row = by_key[(backbone, dataset)]
            if row["search_status"] not in {"completed", "skipped_existing_result"}:
                raise RuntimeError(f"Incomplete clean-reference entry for {(backbone, dataset)}: {row['search_status']}")
            mean = float(row["best_verify_test_acc_mean"])
            std = float(row["best_verify_test_acc_std"])
            n = 10
            # The clean-reference manifest only stores aggregate statistics.
            # Approximate the requested mean CI from std and n.
            half_width = 1.96 * std / math.sqrt(n)
            for x_eps in X_EPS_VALUES:
                plot_rows.append(
                    {
                        "source": "clean_reference",
                        "backbone": backbone,
                        "dataset": dataset,
                        "x_eps": x_eps,
                        "x_index": X_EPS_VALUES.index(x_eps),
                        "line_label": NON_PRIVATE_LABEL,
                        "pipeline": "clean_reference",
                        "smoother": row["smoother"],
                        "feature_dim": "",
                        "candidate_id": row["best_candidate_id"],
                        "x_steps": row["best_x_steps"],
                        "learning_rate": row["best_learning_rate"],
                        "weight_decay": row["best_weight_decay"],
                        "dropout": row["best_dropout"],
                        "tao2": row["best_tao2"],
                        "test_acc_mean": mean,
                        "test_acc_ci_low": mean - half_width,
                        "test_acc_ci_high": mean + half_width,
                        "test_acc_std": std,
                        "test_acc_min": "",
                        "test_acc_max": "",
                        "val_acc_mean": row["best_verify_val_acc_mean"],
                        "val_acc_ci_low": "",
                        "val_acc_ci_high": "",
                        "n": n,
                        "source_job_dir": row["job_dir"],
                        "actual_verify_rank": row["best_rank"],
                        "ci_method": "normal_from_manifest_std",
                    }
                )
    return plot_rows


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "backbone",
        "dataset",
        "x_eps",
        "x_index",
        "line_label",
        "pipeline",
        "smoother",
        "feature_dim",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "tao2",
        "test_acc_mean",
        "test_acc_ci_low",
        "test_acc_ci_high",
        "test_acc_std",
        "test_acc_min",
        "test_acc_max",
        "val_acc_mean",
        "val_acc_ci_low",
        "val_acc_ci_high",
        "n",
        "source_job_dir",
        "actual_verify_rank",
        "ci_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_plot_rows(rows: list[dict[str, Any]]) -> None:
    expected = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES) * len(LINE_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} plot rows, found {len(rows)}")
    seen = {(row["backbone"], row["dataset"], row["x_eps"], row["line_label"]) for row in rows}
    if len(seen) != expected:
        raise RuntimeError(f"Duplicate plot rows detected: unique={len(seen)} expected={expected}")
    expected_per_line = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES)
    for label in LINE_ORDER:
        count = sum(row["line_label"] == label for row in rows)
        if count != expected_per_line:
            raise RuntimeError(
                f"Expected {expected_per_line} rows for {label}, found {count}"
            )
    featfree_sources = {
        row["source"] for row in rows if row["line_label"] == FEATFREE_LABEL
    }
    if featfree_sources != {"featfree_homo_rerun_hoa"}:
        raise RuntimeError(f"Unexpected FeatFree sources: {sorted(featfree_sources)}")
    for row in rows:
        mean = float(row["test_acc_mean"])
        low = float(row["test_acc_ci_low"])
        high = float(row["test_acc_ci_high"])
        if int(row["n"]) != 10:
            raise RuntimeError(f"Expected n=10 for {row}")
        if not (low <= mean <= high):
            raise RuntimeError(f"CI does not contain mean for {row}")


def build_figure(rows: list[dict[str, Any]], *, datasets=DATASETS, backbones=BACKBONES) -> tuple[Any, np.ndarray]:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(len(backbones), len(datasets), figsize=(FIGSIZE_X, FIGSIZE_Y), sharex=False, sharey=False, squeeze=False)
    baseline_axes = np.empty_like(axes)
    fig.subplots_adjust(
        bottom=BOTTOM,
        top=TOP,
        left=LEFT,
        right=RIGHT,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    for row_idx, backbone in enumerate(backbones):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                letter = chr(97 + col_idx)
                majority_baseline = MAJORITY_BASELINE_PCT[dataset]
                ax.set_title(
                    f"({letter}) {DATASET_LABELS[dataset]}"
                    f" (Majority-Class: {majority_baseline:.2f}%)",
                    fontsize=TITLE_FONTSIZE,
                    fontweight="medium",
                    pad=TITLE_PAD,
                )

            panel = df[(df["backbone"] == backbone) & (df["dataset"] == dataset)]
            for label in LINE_ORDER:
                sub = panel[panel["line_label"] == label].sort_values("x_index")
                if sub.empty:
                    continue
                style = STYLE_CONFIGS[label]
                y = sub["test_acc_mean"].to_numpy(dtype=float)
                yerr = np.vstack(
                    [
                        y - sub["test_acc_ci_low"].to_numpy(dtype=float),
                        sub["test_acc_ci_high"].to_numpy(dtype=float) - y,
                    ]
                )
                ax.errorbar(
                    sub["x_index"].to_numpy(dtype=float),
                    y,
                    yerr=yerr,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markersize=MARKERSIZE,
                    linewidth=LINEWIDTH,
                    capsize=0,
                    markerfacecolor="none",
                    markeredgewidth=MARKEREDGEWIDTH,
                )

            ax.set_xticks(range(len(X_EPS_VALUES)))
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.grid(
                True,
                color=GRID_COLOR,
                linestyle=GRID_LINESTYLE,
                linewidth=GRID_LINEWIDTH,
                alpha=GRID_ALPHA,
            )
            ax.tick_params(axis="both", which="major", labelsize=TICKLABEL_FONTSIZE)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{BACKBONE_LABELS[backbone]}\nAccuracy",
                    fontsize=FONTSIZE,
                    fontweight="medium",
                    labelpad=Y_LABEL_PAD,
                )
            else:
                ax.set_ylabel("")

            x0, y0, width, height = ax.get_position().bounds
            baseline_height = height * BASELINE_AXIS_HEIGHT_FRACTION
            break_gap = height * AXIS_BREAK_GAP_FRACTION
            main_height = height * MAIN_AXIS_HEIGHT_FRACTION
            ax.set_position(
                [x0, y0 + baseline_height + break_gap, width, main_height]
            )
            ax.spines["bottom"].set_visible(False)

            baseline_ax = fig.add_axes(
                [x0, y0, width, baseline_height],
                sharex=ax,
                label=f"majority-{row_idx}-{col_idx}",
            )
            baseline_axes[row_idx, col_idx] = baseline_ax
            majority_baseline = MAJORITY_BASELINE_PCT[dataset]
            baseline_ax.axhline(
                majority_baseline,
                color=MAJORITY_BASELINE_COLOR,
                linestyle=MAJORITY_BASELINE_LINESTYLE,
                linewidth=MAJORITY_BASELINE_LINEWIDTH,
                marker=None,
                zorder=MAJORITY_BASELINE_ZORDER,
                label=MAJORITY_LABEL,
            )
            baseline_ax.set_ylim(
                majority_baseline - BASELINE_AXIS_HALF_RANGE,
                majority_baseline + BASELINE_AXIS_HALF_RANGE,
            )
            baseline_ax.set_yticks([majority_baseline])
            baseline_ax.set_yticklabels([f"{majority_baseline:.2f}"])
            baseline_ax.set_xticks(range(len(X_EPS_VALUES)))
            baseline_ax.set_xticklabels(X_EPS_VALUES, rotation=30, ha="right")
            baseline_ax.set_xlabel(
                r"$\epsilon$",
                fontsize=X_LABEL_FONTSIZE,
                fontweight="medium",
            )
            baseline_ax.grid(
                True,
                color=GRID_COLOR,
                linestyle=GRID_LINESTYLE,
                linewidth=GRID_LINEWIDTH,
                alpha=GRID_ALPHA,
            )
            baseline_ax.tick_params(
                axis="both",
                which="major",
                labelsize=TICKLABEL_FONTSIZE,
            )
            baseline_ax.spines["top"].set_visible(False)

            for break_ax, break_y in ((ax, 0.0), (baseline_ax, 1.0)):
                for break_x in (0.0, 1.0):
                    break_path = MplPath(
                        [
                            (
                                break_x - AXIS_BREAK_MARK_SIZE,
                                break_y - AXIS_BREAK_MARK_SIZE,
                            ),
                            (
                                break_x + AXIS_BREAK_MARK_SIZE,
                                break_y + AXIS_BREAK_MARK_SIZE,
                            ),
                        ]
                    )
                    break_ax.add_patch(
                        PathPatch(
                            break_path,
                            transform=break_ax.transAxes,
                            facecolor="none",
                            edgecolor="black",
                            linewidth=AXIS_BREAK_MARK_LINEWIDTH,
                            clip_on=False,
                            zorder=10,
                        )
                    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    baseline_handles, baseline_labels = baseline_axes[0, 0].get_legend_handles_labels()
    handles.extend(baseline_handles)
    labels.extend(baseline_labels)
    expected_labels = [*LINE_ORDER, MAJORITY_LABEL]
    if labels != expected_labels:
        raise RuntimeError(f"Expected legend labels {expected_labels}, found {labels}")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=LEGEND_BBOX,
        ncol=len(expected_labels),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=LEGEND_BORDERPAD,
        handlelength=LEGEND_HANDLELENGTH,
        columnspacing=LEGEND_COLUMNSPACING,
    )
    fig._figure1_baseline_axes = baseline_axes
    return fig, axes


def plot_panels(rows: list[dict[str, Any]], output_dir: Path, *, datasets=DATASETS, backbones=BACKBONES) -> None:
    fig, _ = build_figure(rows, datasets=datasets, backbones=backbones)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(
        pdf_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    fig.savefig(
        png_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    validate_render_environment()
    main_rows = load_main_curves(
        args.main_long_csv,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    featfree_choices = load_featfree_choices(args.featfree_root)
    featfree_rows = featfree_plot_rows(
        featfree_choices,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    non_private_rows = clean_reference_plot_rows(
        args.clean_reference_manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    plot_rows = main_rows + featfree_rows + non_private_rows
    validate_plot_rows(plot_rows)
    plot_data_path = args.output_dir / PLOT_DATA_FILENAME
    write_plot_data(plot_data_path, plot_rows)
    plot_panels(plot_rows, args.output_dir)
    print(f"Saved {plot_data_path}")
    print(f"Plot rows: {len(plot_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
