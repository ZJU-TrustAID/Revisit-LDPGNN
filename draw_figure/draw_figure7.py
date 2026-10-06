#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from draw_figure.draw_figure4 import (  # noqa: E402
    EPSILON_SPECS,
    EXPECTED_REPEATS,
    SCALE_LABELS,
    SCALE_SPECS,
    STYLE_CONFIGS,
    VerifyRecord,
    build_plot_rows,
    collect_verify_records,
    decimal_key,
    validate_plot_rows,
)

BACKGROUND_COLOR = "#f5f5f5"  # LaTeX xcolor: black!4
plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR


DEFAULT_FIGURE5_ROOT = REPO_ROOT / "scripts" / "aec" / "reference" / "figure7_plot_data.csv"
DEFAULT_FEATFREE_MANIFEST = REPO_ROOT / "scripts" / "aec" / "reference" / "reference_metrics.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rebuttal_figure"
EXPECTED_MANIFEST_ROWS = 168
FEATFREE_LABEL = r"$\mathsf{FeatFree}$"
VERIFY_DIR_RE = re.compile(r"^rank=(\d+)__repeat=(\d+)__candidate=(\d+)__")

# These are the exact layout/style values embedded in the reference PDF.
FIGSIZE_X = 13.2
FIGSIZE_Y = 7.1
BOTTOM = 0.36
BASE_TOP = 0.99
COORDINATE_AREA_HEIGHT_SCALE = 0.75
TOP = BOTTOM + (BASE_TOP - BOTTOM) * COORDINATE_AREA_HEIGHT_SCALE
LEFT = 0.08
RIGHT = 0.985
FONTSIZE = 31
X_LABEL_FONTSIZE = 30
X_LABELPAD = -6
LEGEND_FONTSIZE = 0.9 * 30
TICKLABEL_FONTSIZE = 22
LINEWIDTH = 2.5
MARKERSIZE = 11
Y_TICKS = (20, 30, 40, 50, 60)
FIRST_LEGEND_ANCHOR = (0.5, -0.4975)
SECOND_LEGEND_ANCHOR = (0.5, -0.6675)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 5 norm-scale curves on Flickr.")
    parser.add_argument(
        "--figure5-root",
        type=Path,
        default=DEFAULT_FIGURE5_ROOT,
        help="Root containing the Flickr figure5.yaml result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figure5_heter.pdf and figure5_heter.png are written.",
    )
    parser.add_argument(
        "--featfree-manifest",
        type=Path,
        default=DEFAULT_FEATFREE_MANIFEST,
        help="Manifest for the Flickr GraphSAGE FeatFree reference.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for 95%% confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=12345,
        help="Random seed for bootstrap resampling.",
    )
    return parser.parse_args()


def load_manifest_index(manifest_path: Path) -> dict[tuple[Decimal, Decimal], dict[str, str]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_MANIFEST_ROWS:
        raise ValueError(
            f"{manifest_path}: expected {EXPECTED_MANIFEST_ROWS} rows, found {len(rows)}"
        )

    expected = {
        "dataset": "flickr",
        "feature": "raw",
        "mechanism": "mbm",
        "m": "best",
        "smoother": "hoa",
        "backbone": "sage",
    }
    index: dict[tuple[Decimal, Decimal], dict[str, str]] = {}
    for row in rows:
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(
                    f"{manifest_path}: expected {field}={value}, found {row.get(field)!r}"
                )
        if row.get("norm") not in {"True", "true", "1"}:
            raise ValueError(f"{manifest_path}: expected norm=true, found {row.get('norm')!r}")
        if row.get("use_nfr") not in {"False", "false", "0"}:
            raise ValueError(
                f"{manifest_path}: expected use_nfr=false, found {row.get('use_nfr')!r}"
            )

        key = (
            decimal_key(row.get("x_eps"), field="x_eps"),
            decimal_key(row.get("norm_scale"), field="norm_scale"),
        )
        if key in index:
            raise ValueError(f"Duplicate manifest key {key} in {manifest_path}")
        index[key] = row
    return index


def load_records(figure5_root: Path) -> list[VerifyRecord]:
    manifest_path = figure5_root / "manifest.csv"
    index = load_manifest_index(manifest_path)
    records: list[VerifyRecord] = []

    for epsilon, _ in EPSILON_SPECS:
        for scale_exponent, scale_value, _ in SCALE_SPECS:
            key = (epsilon, scale_value)
            if key not in index:
                raise ValueError(
                    f"{manifest_path}: missing row for eps={epsilon}, r=2^{scale_exponent}"
                )
            records.extend(
                collect_verify_records(
                    source="figure5_heter",
                    epsilon=epsilon,
                    scale_exponent=scale_exponent,
                    scale_value=scale_value,
                    job_dir=Path(index[key]["job_dir"]),
                )
            )

    expected_records = len(EPSILON_SPECS) * len(SCALE_SPECS) * EXPECTED_REPEATS
    if len(records) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, found {len(records)}")
    return records


def load_featfree_reference(manifest_path: Path) -> float:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"FeatFree manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 1:
        raise RuntimeError(
            f"Expected one FeatFree row in {manifest_path}, found {len(manifest_rows)}"
        )

    manifest_row = manifest_rows[0]
    expected_manifest = {
        "dataset": "flickr",
        "feature": "random_normal",
        "feature_dim": "12047",
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

    candidate_id = int(manifest_row["best_candidate_id"])
    expected_hparams = {
        "x_steps": float(manifest_row["best_x_steps"]),
        "learning_rate": float(manifest_row["best_learning_rate"]),
        "weight_decay": float(manifest_row["best_weight_decay"]),
        "dropout": float(manifest_row["best_dropout"]),
    }
    job_dir = Path(manifest_row["job_dir"])
    verify_dir = job_dir / "verify_top5"
    if not verify_dir.is_dir():
        raise RuntimeError(f"Missing FeatFree verify directory: {verify_dir}")

    repeat_dirs: dict[int, Path] = {}
    for child in sorted(verify_dir.iterdir()):
        if not child.is_dir():
            continue
        match = VERIFY_DIR_RE.match(child.name)
        if match is None or int(match.group(3)) != candidate_id:
            continue
        repeat_id = int(match.group(2))
        if repeat_id in repeat_dirs:
            raise RuntimeError(f"Duplicate FeatFree repeat {repeat_id} in {verify_dir}")
        repeat_dirs[repeat_id] = child
    if sorted(repeat_dirs) != list(range(1, EXPECTED_REPEATS + 1)):
        raise RuntimeError(
            f"Expected FeatFree repeats 1..{EXPECTED_REPEATS}, found {sorted(repeat_dirs)}"
        )

    test_accs: list[float] = []
    seeds: list[int] = []
    for repeat_id in range(1, EXPECTED_REPEATS + 1):
        csv_files = sorted(repeat_dirs[repeat_id].glob("*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(
                f"Expected one FeatFree CSV in {repeat_dirs[repeat_id]}, found {len(csv_files)}"
            )
        with csv_files[0].open("r", encoding="utf-8", newline="") as handle:
            result_rows = list(csv.DictReader(handle))
        if len(result_rows) != 1:
            raise RuntimeError(f"Expected one row in {csv_files[0]}, found {len(result_rows)}")
        result = result_rows[0]
        expected_result = {
            "dataset": "flickr",
            "feature": "random_normal",
            "smoother": "hoa",
            "model": "sage",
        }
        for field, expected in expected_result.items():
            if result.get(field, "").strip().lower() != expected:
                raise RuntimeError(f"Unexpected FeatFree {field} in {csv_files[0]}")
        for field, expected in expected_hparams.items():
            if not math.isclose(float(result[field]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"FeatFree hyperparameter mismatch for {field} in {csv_files[0]}")
        test_acc = float(result["test/acc"])
        if not math.isfinite(test_acc):
            raise RuntimeError(f"Non-finite FeatFree test accuracy in {csv_files[0]}")
        test_accs.append(test_acc)
        seeds.append(int(result["seed"]))

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
    rows: list[dict[str, object]], featfree_acc: float, output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(FIGSIZE_X, FIGSIZE_Y))
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT)

    for epsilon, epsilon_label in EPSILON_SPECS:
        sub = sorted(
            [row for row in rows if Decimal(str(row["epsilon"])) == epsilon],
            key=lambda row: int(row["x_index"]),
        )
        if len(sub) != len(SCALE_SPECS):
            raise RuntimeError(
                f"{epsilon}: expected {len(SCALE_SPECS)} plot points, found {len(sub)}"
            )

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

    scale_values = [float(scale_value) for _, scale_value, _ in SCALE_SPECS]
    ax.set_xscale("log", base=2)
    ax.set_xticks(scale_values)
    ax.set_xticklabels(SCALE_LABELS, rotation=35, ha="right")
    ax.set_yticks(Y_TICKS)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=2.4, alpha=0.95, zorder=0)
    ax.set_xlabel(
        r"$r$",
        fontsize=X_LABEL_FONTSIZE,
        fontweight="medium",
        labelpad=X_LABELPAD,
    )
    ax.set_ylabel("Test Accuracy", fontsize=FONTSIZE, fontweight="medium")
    ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
    ax.tick_params(axis="both", which="major", labelsize=TICKLABEL_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    epsilon_labels = [epsilon_label for _, epsilon_label in EPSILON_SPECS]
    first_labels = epsilon_labels[:3]
    second_labels = [*epsilon_labels[3:], FEATFREE_LABEL]
    first_legend = ax.legend(
        [handles_by_label[label] for label in first_labels],
        first_labels,
        loc="lower center",
        bbox_to_anchor=FIRST_LEGEND_ANCHOR,
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
        bbox_to_anchor=SECOND_LEGEND_ANCHOR,
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0.2,
        handlelength=1.5,
        columnspacing=0.85,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure5_heter.pdf"
    png_path = output_dir / "figure5_heter.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    records = load_records(args.figure5_root)
    rows = build_plot_rows(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    validate_plot_rows(rows)
    featfree_acc = load_featfree_reference(args.featfree_manifest)
    plot_curves(rows, featfree_acc, args.output_dir)
    print(f"Long rows: {len(records)}")
    print(f"Plot rows: {len(rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
