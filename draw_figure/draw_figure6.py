#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BACKGROUND_COLOR = "#f5f5f5"  # LaTeX xcolor: black!4
plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from draw_figure import draw_figure1 as reference


DEFAULT_MAIN_ROOT = REPO_ROOT / "scripts" / "aec" / "reference"
DEFAULT_FEATFREE_ROOT = REPO_ROOT / "scripts" / "aec" / "reference"
DEFAULT_CLEAN_ROOT = REPO_ROOT / "scripts" / "aec" / "reference"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rebuttal_figure"
OUTPUT_STEM = "figure1_heter"
PLOT_DATA_FILENAME = f"{OUTPUT_STEM}_plot_data.csv"

BACKBONES = reference.BACKBONES
BACKBONE_LABELS = reference.BACKBONE_LABELS
DATASETS = ("actor", "flickr")
DATASET_LABELS = {
    "actor": "Actor",
    "flickr": "Flickr",
}
# Global label-prior baselines audited from the raw dataset labels.  Keep these
# separate from the plotted experiment rows: they are dataset statistics, not
# trained-method results.
MAJORITY_BASELINE_PCT = {
    "actor": 100.0 * 1965 / 7600,
    "flickr": 100.0 * 888 / 7575,
}
X_EPS_VALUES = reference.X_EPS_VALUES
PIPELINE_LABELS = reference.PIPELINE_LABELS
FEATFREE_LABEL = reference.FEATFREE_LABEL
NON_PRIVATE_LABEL = reference.NON_PRIVATE_LABEL
LINE_ORDER = reference.LINE_ORDER
STYLE_CONFIGS = reference.STYLE_CONFIGS
MAJORITY_LABEL = r"$\mathsf{Majority\text{-}Class}$"
LEGEND_TOP_ROW = LINE_ORDER[:4]
LEGEND_BOTTOM_ROW = (*LINE_ORDER[4:], MAJORITY_LABEL)
LEGEND_DISPLAY_LABELS = {
    reference.NON_PRIVATE_LABEL: r"$\mathsf{Non\text{-}Private}$",
}

COMPLETE_STATUSES = {"completed", "skipped_existing_result"}
PIPELINE_CONTRACT = {
    "figure3_pipeline1": {"mechanism": "mbm", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline2": {"mechanism": "hds", "smoother": "kprop", "use_nfr": "false"},
    "figure3_pipeline3": {"mechanism": "mbm", "smoother": "hoa", "use_nfr": "true"},
    "figure3_pipeline4": {"mechanism": "pm", "smoother": "hoa", "use_nfr": "true"},
}

BOOTSTRAP_SAMPLES = reference.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = reference.BOOTSTRAP_SEED

# Preserve Figure 1's per-panel styling while adapting its 3x4 canvas to 3x2.
FIGSIZE_X = 12
FIGSIZE_Y = reference.FIGSIZE_Y
BOTTOM = 0.19
TOP = reference.TOP
LEFT = 0.10
RIGHT = reference.RIGHT
HSPACE = 0.44
WSPACE = reference.WSPACE
LEGEND_TOP_BBOX = (0.5, 0.055)
LEGEND_BOTTOM_BBOX = (0.5, 0.006)
LEGEND_COLUMNSPACING = 0.45
LEGEND_HANDLETEXTPAD = 0.5

# Figure-specific typography.  The 3x2 layout is narrower than the reference
# 3x4 figure, so keep these settings local instead of changing the shared style.
AXIS_LABEL_FONTSIZE = 22
X_LABEL_PAD = -1
TICKLABEL_FONTSIZE = 20
PANEL_HEADING_FONTSIZE = 24
BACKBONE_LABEL_PAD = 4
BACKBONE_LABEL_X = -0.108
MAJORITY_BASELINE_COLOR = "#000000"
MAJORITY_BASELINE_LINESTYLE = "--"
MAJORITY_BASELINE_LINEWIDTH = 2
MAJORITY_BASELINE_ZORDER = 1.5
PANEL_Y_TICKS = {
    ("gcn", "actor"): (25.0, 27.5, 30.0),
    ("gcn", "flickr"): (25.0, 37.5, 50.0, 62.5),
    ("gat", "flickr"): (20.0, 30.0, 40.0, 50.0),
}


@dataclass(frozen=True)
class VerifiedChoice:
    candidate: reference.Candidate
    val_accs: list[float]
    test_accs: list[float]
    job_dir: Path
    verify_rank: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Figure 1-style panels for Actor and AttributedGraph-Flickr."
    )
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--featfree-root", type=Path, default=DEFAULT_FEATFREE_ROOT)
    parser.add_argument("--clean-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def require_field(
    row: dict[str, str],
    field: str,
    expected: str,
    *,
    manifest_path: Path,
) -> None:
    actual = row.get(field, "").strip().lower()
    if actual != expected.lower():
        raise RuntimeError(
            f"Unexpected {field} in {manifest_path}: expected {expected!r}, got {actual!r}"
        )


def verified_choice(row: dict[str, str], *, manifest_path: Path) -> VerifiedChoice:
    if row.get("search_status", "") not in COMPLETE_STATUSES:
        raise RuntimeError(
            f"Incomplete result in {manifest_path}: {row.get('search_status', '')!r}"
        )

    job_dir = Path(row["job_dir"])
    best_config_path = Path(row["best_config_path"])
    if not job_dir.is_dir():
        raise RuntimeError(f"Missing job directory from {manifest_path}: {job_dir}")
    if not best_config_path.is_file():
        raise RuntimeError(f"Missing best config from {manifest_path}: {best_config_path}")
    if best_config_path.parent.resolve() != job_dir.resolve():
        raise RuntimeError(
            f"best_config_path is outside job_dir in {manifest_path}: "
            f"{best_config_path} vs {job_dir}"
        )

    candidate = reference.candidate_from_best_config(best_config_path)
    manifest_candidate_id = int(
        reference.row_float(row, "best_candidate_id", csv_path=manifest_path)
    )
    if candidate.candidate_id != manifest_candidate_id:
        raise RuntimeError(
            f"Candidate mismatch in {manifest_path}: manifest={manifest_candidate_id}, "
            f"best_config={candidate.candidate_id}"
        )

    val_accs, test_accs, verify_rank = reference.collect_reference_repeats(
        job_dir, candidate
    )
    reference.assert_manifest_stat_matches(
        row,
        "best_verify_val_acc_mean",
        float(np.mean(val_accs)),
        manifest_path=manifest_path,
    )
    reference.assert_manifest_stat_matches(
        row,
        "best_verify_test_acc_mean",
        float(np.mean(test_accs)),
        manifest_path=manifest_path,
    )
    reference.assert_manifest_stat_matches(
        row,
        "best_verify_test_acc_std",
        float(np.std(test_accs, ddof=1)),
        manifest_path=manifest_path,
    )
    return VerifiedChoice(
        candidate=candidate,
        val_accs=val_accs,
        test_accs=test_accs,
        job_dir=job_dir,
        verify_rank=verify_rank,
    )


def stats_for_choice(
    choice: VerifiedChoice,
    *,
    key: tuple[Any, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    test_stats = reference.metric_stats(
        choice.test_accs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        key=("test", *key),
    )
    val_stats = reference.metric_stats(
        choice.val_accs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        key=("val", *key),
    )
    return test_stats, val_stats


def plot_row(
    *,
    source: str,
    backbone: str,
    dataset: str,
    x_eps: str,
    line_label: str,
    pipeline: str,
    smoother: str,
    feature_dim: str | int,
    choice: VerifiedChoice,
    test_stats: dict[str, float],
    val_stats: dict[str, float],
    manifest_path: Path,
) -> dict[str, Any]:
    candidate = choice.candidate
    return {
        "source": source,
        "source_manifest": str(manifest_path),
        "source_job_dir": str(choice.job_dir),
        "backbone": backbone,
        "dataset": dataset,
        "x_eps": x_eps,
        "x_index": X_EPS_VALUES.index(x_eps),
        "line_label": line_label,
        "pipeline": pipeline,
        "smoother": smoother,
        "feature_dim": feature_dim,
        "candidate_id": candidate.candidate_id,
        "x_steps": candidate.x_steps,
        "learning_rate": candidate.learning_rate,
        "weight_decay": candidate.weight_decay,
        "dropout": candidate.dropout,
        "tao2": candidate.tao2,
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
        "actual_verify_rank": "" if choice.verify_rank is None else choice.verify_rank,
        "ci_method": "bootstrap_95pct",
    }


def load_main_rows(
    root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            for pipeline, contract in PIPELINE_CONTRACT.items():
                manifest_path = root / dataset / backbone / f"{pipeline}.yaml" / "manifest.csv"
                manifest_rows = reference.read_csv_rows(manifest_path)
                if len(manifest_rows) != len(X_EPS_VALUES):
                    raise RuntimeError(
                        f"Expected {len(X_EPS_VALUES)} rows in {manifest_path}, "
                        f"found {len(manifest_rows)}"
                    )
                by_eps: dict[str, dict[str, str]] = {}
                for manifest_row in manifest_rows:
                    require_field(manifest_row, "dataset", dataset, manifest_path=manifest_path)
                    require_field(manifest_row, "backbone", backbone, manifest_path=manifest_path)
                    require_field(manifest_row, "feature", "raw", manifest_path=manifest_path)
                    for field, expected in contract.items():
                        require_field(manifest_row, field, expected, manifest_path=manifest_path)
                    x_eps = reference.canonical_x_eps(manifest_row["x_eps"])
                    if x_eps in by_eps:
                        raise RuntimeError(f"Duplicate epsilon {x_eps} in {manifest_path}")
                    by_eps[x_eps] = manifest_row
                if tuple(sorted(by_eps, key=X_EPS_VALUES.index)) != X_EPS_VALUES:
                    raise RuntimeError(f"Incomplete epsilon set in {manifest_path}: {sorted(by_eps)}")

                for x_eps in X_EPS_VALUES:
                    manifest_row = by_eps[x_eps]
                    choice = verified_choice(manifest_row, manifest_path=manifest_path)
                    test_stats, val_stats = stats_for_choice(
                        choice,
                        key=("figure1_heter", backbone, dataset, pipeline, x_eps),
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed,
                    )
                    rows.append(
                        plot_row(
                            source="figure1_heter",
                            backbone=backbone,
                            dataset=dataset,
                            x_eps=x_eps,
                            line_label=PIPELINE_LABELS[pipeline],
                            pipeline=pipeline,
                            smoother=manifest_row["smoother"],
                            feature_dim="",
                            choice=choice,
                            test_stats=test_stats,
                            val_stats=val_stats,
                            manifest_path=manifest_path,
                        )
                    )
    return rows


def load_featfree_rows(
    root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backbone in BACKBONES:
        for dataset in DATASETS:
            manifest_path = root / dataset / backbone / "random_projected.yaml" / "manifest.csv"
            manifest_rows = reference.read_csv_rows(manifest_path)
            if len(manifest_rows) != 1:
                raise RuntimeError(
                    f"Expected one FeatFree row in {manifest_path}, found {len(manifest_rows)}"
                )
            manifest_row = manifest_rows[0]
            require_field(manifest_row, "dataset", dataset, manifest_path=manifest_path)
            require_field(manifest_row, "backbone", backbone, manifest_path=manifest_path)
            require_field(manifest_row, "feature", "random_normal", manifest_path=manifest_path)
            require_field(manifest_row, "smoother", "hoa", manifest_path=manifest_path)
            feature_dim = int(
                reference.row_float(manifest_row, "feature_dim", csv_path=manifest_path)
            )
            choice = verified_choice(manifest_row, manifest_path=manifest_path)
            test_stats, val_stats = stats_for_choice(
                choice,
                key=("featfree_heter_hoa_random_projected", backbone, dataset, feature_dim),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for x_eps in X_EPS_VALUES:
                rows.append(
                    plot_row(
                        source="featfree_heter_hoa_random_projected",
                        backbone=backbone,
                        dataset=dataset,
                        x_eps=x_eps,
                        line_label=FEATFREE_LABEL,
                        pipeline="featfree",
                        smoother="hoa",
                        feature_dim=feature_dim,
                        choice=choice,
                        test_stats=test_stats,
                        val_stats=val_stats,
                        manifest_path=manifest_path,
                    )
                )
    return rows


def load_clean_rows(
    root: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        manifest_path = root / dataset / "figure1_clean.yaml" / "manifest.csv"
        manifest_rows = reference.read_csv_rows(manifest_path)
        if len(manifest_rows) != len(BACKBONES):
            raise RuntimeError(
                f"Expected {len(BACKBONES)} clean rows in {manifest_path}, "
                f"found {len(manifest_rows)}"
            )
        by_backbone = {row["backbone"].strip().lower(): row for row in manifest_rows}
        if set(by_backbone) != set(BACKBONES):
            raise RuntimeError(
                f"Unexpected clean backbones in {manifest_path}: {sorted(by_backbone)}"
            )
        for backbone in BACKBONES:
            manifest_row = by_backbone[backbone]
            require_field(manifest_row, "dataset", dataset, manifest_path=manifest_path)
            require_field(manifest_row, "feature", "raw", manifest_path=manifest_path)
            choice = verified_choice(manifest_row, manifest_path=manifest_path)
            test_stats, val_stats = stats_for_choice(
                choice,
                key=("clean_heter", backbone, dataset),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            for x_eps in X_EPS_VALUES:
                rows.append(
                    plot_row(
                        source="clean_heter",
                        backbone=backbone,
                        dataset=dataset,
                        x_eps=x_eps,
                        line_label=NON_PRIVATE_LABEL,
                        pipeline="clean_reference",
                        smoother=manifest_row["smoother"],
                        feature_dim="",
                        choice=choice,
                        test_stats=test_stats,
                        val_stats=val_stats,
                        manifest_path=manifest_path,
                    )
                )
    return rows


def validate_plot_rows(rows: list[dict[str, Any]]) -> None:
    expected = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES) * len(LINE_ORDER)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} plot rows, found {len(rows)}")
    keys = {
        (row["backbone"], row["dataset"], row["x_eps"], row["line_label"])
        for row in rows
    }
    if len(keys) != expected:
        raise RuntimeError(f"Duplicate plot rows: unique={len(keys)}, expected={expected}")
    per_line = len(BACKBONES) * len(DATASETS) * len(X_EPS_VALUES)
    for label in LINE_ORDER:
        count = sum(row["line_label"] == label for row in rows)
        if count != per_line:
            raise RuntimeError(f"Expected {per_line} rows for {label}, found {count}")
    expected_sources = {
        PIPELINE_LABELS["figure3_pipeline1"]: "figure1_heter",
        PIPELINE_LABELS["figure3_pipeline2"]: "figure1_heter",
        PIPELINE_LABELS["figure3_pipeline3"]: "figure1_heter",
        PIPELINE_LABELS["figure3_pipeline4"]: "figure1_heter",
        FEATFREE_LABEL: "featfree_heter_hoa_random_projected",
        NON_PRIVATE_LABEL: "clean_heter",
    }
    for label, expected_source in expected_sources.items():
        sources = {row["source"] for row in rows if row["line_label"] == label}
        if sources != {expected_source}:
            raise RuntimeError(f"Unexpected sources for {label}: {sorted(sources)}")
    for row in rows:
        mean = float(row["test_acc_mean"])
        low = float(row["test_acc_ci_low"])
        high = float(row["test_acc_ci_high"])
        if int(row["n"]) != 10 or not (low <= mean <= high):
            raise RuntimeError(f"Invalid statistics in row: {row}")


def write_plot_data(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "source_manifest",
        "source_job_dir",
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
        "actual_verify_rank",
        "ci_method",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_figure(rows: list[dict[str, Any]], *, datasets=DATASETS, backbones=BACKBONES) -> tuple[Any, np.ndarray]:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(
        len(backbones),
        len(datasets),
        figsize=(FIGSIZE_X, FIGSIZE_Y),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.subplots_adjust(
        bottom=BOTTOM,
        top=TOP,
        left=LEFT,
        right=RIGHT,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    for row_index, backbone in enumerate(backbones):
        for column_index, dataset in enumerate(datasets):
            ax = axes[row_index, column_index]
            if row_index == 0:
                majority_baseline = MAJORITY_BASELINE_PCT[dataset]
                ax.set_title(
                    f"({chr(97 + column_index)}) {DATASET_LABELS[dataset]}"
                    f" (Majority Class: {majority_baseline:.2f}%)",
                    fontsize=PANEL_HEADING_FONTSIZE,
                    fontweight="medium",
                    pad=reference.TITLE_PAD,
                )

            panel = df[(df["backbone"] == backbone) & (df["dataset"] == dataset)]
            for label in LINE_ORDER:
                subset = panel[panel["line_label"] == label].sort_values("x_index")
                style = STYLE_CONFIGS[label]
                y = subset["test_acc_mean"].to_numpy(dtype=float)
                yerr = np.vstack(
                    [
                        y - subset["test_acc_ci_low"].to_numpy(dtype=float),
                        subset["test_acc_ci_high"].to_numpy(dtype=float) - y,
                    ]
                )
                ax.errorbar(
                    subset["x_index"].to_numpy(dtype=float),
                    y,
                    yerr=yerr,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markersize=reference.MARKERSIZE,
                    linewidth=reference.LINEWIDTH,
                    capsize=0,
                    markerfacecolor="none",
                    markeredgewidth=reference.MARKEREDGEWIDTH,
                )

            majority_baseline = MAJORITY_BASELINE_PCT[dataset]
            ax.axhline(
                majority_baseline,
                color=MAJORITY_BASELINE_COLOR,
                linestyle=MAJORITY_BASELINE_LINESTYLE,
                linewidth=MAJORITY_BASELINE_LINEWIDTH,
                marker=None,
                zorder=MAJORITY_BASELINE_ZORDER,
                label=MAJORITY_LABEL,
            )

            panel_y_ticks = PANEL_Y_TICKS.get((backbone, dataset))
            if panel_y_ticks is not None:
                original_y_limits = ax.get_ylim()
                ax.set_yticks(panel_y_ticks)
                ax.set_ylim(original_y_limits)

            ax.set_xticks(range(len(X_EPS_VALUES)))
            ax.set_xticklabels(X_EPS_VALUES, rotation=30, ha="right")
            ax.set_xlabel(
                r"$\epsilon$",
                fontsize=AXIS_LABEL_FONTSIZE,
                fontweight="medium",
                labelpad=X_LABEL_PAD,
            )
            ax.grid(
                True,
                color=reference.GRID_COLOR,
                linestyle=reference.GRID_LINESTYLE,
                linewidth=reference.GRID_LINEWIDTH,
                alpha=reference.GRID_ALPHA,
            )
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=TICKLABEL_FONTSIZE,
            )
            if column_index == 0:
                ax.set_ylabel(
                    BACKBONE_LABELS[backbone],
                    fontsize=PANEL_HEADING_FONTSIZE,
                    fontweight="medium",
                    labelpad=BACKBONE_LABEL_PAD,
                )
                # Use a fixed axes-relative position so wider decimal tick labels
                # cannot push one backbone label farther left than the others.
                ax.yaxis.set_label_coords(BACKBONE_LABEL_X, 0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    expected_labels = [*LINE_ORDER, MAJORITY_LABEL]
    if len(labels) != len(expected_labels) or set(labels) != set(expected_labels):
        raise RuntimeError(f"Unexpected legend order: {labels}")
    handles_by_label = dict(zip(labels, handles))
    for legend_row, legend_bbox in (
        (LEGEND_TOP_ROW, LEGEND_TOP_BBOX),
        (LEGEND_BOTTOM_ROW, LEGEND_BOTTOM_BBOX),
    ):
        fig.legend(
            [handles_by_label[label] for label in legend_row],
            [LEGEND_DISPLAY_LABELS.get(label, label) for label in legend_row],
            loc="lower center",
            bbox_to_anchor=legend_bbox,
            ncol=len(legend_row),
            fontsize=reference.LEGEND_FONTSIZE,
            frameon=False,
            shadow=False,
            borderpad=reference.LEGEND_BORDERPAD,
            handlelength=reference.LEGEND_HANDLELENGTH,
            columnspacing=LEGEND_COLUMNSPACING,
            handletextpad=LEGEND_HANDLETEXTPAD,
        )
    return fig, axes


def save_figure(rows: list[dict[str, Any]], output_dir: Path, *, datasets=DATASETS, backbones=BACKBONES) -> tuple[Path, Path]:
    fig, _ = build_figure(rows, datasets=datasets, backbones=backbones)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    for path in (pdf_path, png_path):
        fig.savefig(
            path,
            dpi=reference.SAVE_DPI,
            bbox_inches="tight",
            pad_inches=reference.SAVE_PAD_INCHES,
        )
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    reference.validate_render_environment()
    main_rows = load_main_rows(
        args.main_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    featfree_rows = load_featfree_rows(
        args.featfree_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    clean_rows = load_clean_rows(
        args.clean_root,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    rows = main_rows + featfree_rows + clean_rows
    validate_plot_rows(rows)
    plot_data_path = args.output_dir / PLOT_DATA_FILENAME
    write_plot_data(plot_data_path, rows)
    pdf_path, png_path = save_figure(rows, args.output_dir)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    print(f"Saved {plot_data_path}")
    print(
        f"Plot rows: {len(rows)} "
        f"(main={len(main_rows)}, featfree={len(featfree_rows)}, clean={len(clean_rows)})"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
