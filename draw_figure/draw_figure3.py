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
from matplotlib.lines import Line2D
import numpy as np
import yaml


PAPER_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "aec" / "reference"
DEFAULT_FIGURE4_ROOT = PAPER_ROOT / "figure4_final"
DEFAULT_OUTPUT_DIR = DEFAULT_FIGURE4_ROOT / "plots"

EPSILON_SPECS = (
    (Decimal("0.00001"), "1e-5"),
    (Decimal("0.0001"), "1e-4"),
    (Decimal("0.001"), "0.001"),
    (Decimal("0.01"), "0.01"),
    (Decimal("0.1"), "0.1"),
    (Decimal("1.0"), "1"),
    (Decimal("2.0"), "2"),
    (Decimal("3.0"), "3"),
    (Decimal("4.0"), "4"),
    (Decimal("5.0"), "5"),
    (Decimal("6.0"), "6"),
    (Decimal("7.0"), "7"),
    (Decimal("8.0"), "8"),
    (Decimal("9.0"), "9"),
    (Decimal("10.0"), "10"),
)
EPSILON_TO_INDEX = {epsilon: idx for idx, (epsilon, _) in enumerate(EPSILON_SPECS)}
EPSILON_LABELS = [label for _, label in EPSILON_SPECS]
EPSILON_TICK_LABELS = [
    r"$10^{-5}$",
    r"$10^{-4}$",
    r"$10^{-3}$",
    r"$10^{-2}$",
    r"$10^{-1}$",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
]

MECHANISMS = ("mbm", "pm", "hds")
MECHANISM_LABELS = {"mbm": "MBM", "pm": "PM", "hds": "HDS"}
SOURCES = {
    "ldp": {
        "subdir": "figure4_ori.yaml",
        "eps_field": "x_eps",
        "label_prefix": "LDP",
        "expected_feature": "raw",
    },
    "sim": {
        "subdir": "figure4_sim.yaml",
        "eps_field": "sim_reference_eps",
        "label_prefix": "SIM",
        "expected_feature": "sim",
    },
}

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 12345
EXPECTED_REPEATS = 20

VERIFY_DIR_RE = re.compile(
    r"^rank=(?P<rank>\d+)__repeat=(?P<repeat>\d+)__candidate=(?P<candidate>\d+)__"
    r"x_steps=(?P<x_steps>[^_]+)__learning_rate=(?P<learning_rate>[^_]+)__"
    r"weight_decay=(?P<weight_decay>[^_]+)__dropout=(?P<dropout>[^_]+)__tao2=(?P<tao2>.+)$"
)

# --- Visual Style Settings, aligned with the main panel plotting script ---
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

LEGEND_FONTSIZE = 28
TITLE_FONTSIZE = LEGEND_FONTSIZE
TITLE_PAD = 8
FONTSIZE = 28
X_LABEL_FONTSIZE = 28
X_LABELPAD = -6
TICKLABEL_FONTSIZE = 20
LINEWIDTH = 2.4
MARKERSIZE = 10
FIGSIZE_X = 22.0
FIGSIZE_Y = 6.4
BOTTOM = 0.30
BASE_TOP = 0.90
COORDINATE_AREA_HEIGHT_SCALE = 0.75
TOP = BOTTOM + (BASE_TOP - BOTTOM) * COORDINATE_AREA_HEIGHT_SCALE
LEFT = 0.04
RIGHT = 0.96
LEGEND_ANCHOR = (0.465, 0.0525)
EXPORT_PAD_INCHES = 0

STYLE_CONFIGS = {
    "LDP-MBM": {"color": "#5372ab", "marker": "s", "linestyle": "-"},
    "SIM-MBM": {"color": "#8ab6d6", "marker": "o", "linestyle": "--"},
    "LDP-PM": {"color": "#6aa56e", "marker": "^", "linestyle": "-"},
    "SIM-PM": {"color": "#b5c98b", "marker": "D", "linestyle": "--"},
    "LDP-HDS": {"color": "#936bb9", "marker": "v", "linestyle": "-"},
    "SIM-HDS": {"color": "#d3a1c8", "marker": "P", "linestyle": "--"},
}
LINE_ORDER = ("LDP-MBM", "SIM-MBM", "LDP-PM", "SIM-PM", "LDP-HDS", "SIM-HDS")
Y_TICKS = {
    "mbm": (76, 78, 80, 82, 84),
    "pm": (76, 78, 80, 82, 84, 86),
    "hds": (76, 78, 80, 82, 84),
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
    line_label: str
    mechanism: str
    epsilon: Decimal
    epsilon_label: str
    x_index: int
    repeat: int
    rank: int
    candidate: Candidate
    seed: str
    val_acc: float
    test_acc: float
    result_csv_path: Path
    source_job_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 4 final LDP/SIM mechanism panels.")
    parser.add_argument(
        "--figure4-root",
        type=Path,
        default=DEFAULT_FIGURE4_ROOT,
        help="Root containing figure4_ori.yaml and figure4_sim.yaml result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDF/PNG/CSV outputs will be written.",
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


def decimal_key(value: Any) -> Decimal:
    if value is None:
        raise ValueError("missing epsilon value")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        raise ValueError("missing epsilon value")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid epsilon value: {value!r}") from exc


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
    return format(dec.normalize(), "f").rstrip("0").rstrip(".") or "0"


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    line_label: str,
    mechanism: str,
    epsilon: Decimal,
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
                line_label=line_label,
                mechanism=mechanism,
                epsilon=epsilon,
                epsilon_label=EPSILON_SPECS[EPSILON_TO_INDEX[epsilon]][1],
                x_index=EPSILON_TO_INDEX[epsilon],
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
            f"{source}/{mechanism}/eps={epsilon}: expected repeats {expected_repeats}, found {repeats} "
            f"for candidate {candidate.candidate_id} in {job_dir}"
        )
    return records


def validate_manifest_row(source: str, row: dict[str, str], source_config: dict[str, str]) -> None:
    expected_feature = source_config["expected_feature"]
    if row.get("dataset") != "cora":
        raise ValueError(f"{source}: expected dataset=cora, found {row.get('dataset')}")
    if row.get("feature") != expected_feature:
        raise ValueError(f"{source}: expected feature={expected_feature}, found {row.get('feature')}")
    if row.get("smoother") != "hoa":
        raise ValueError(f"{source}: expected smoother=hoa, found {row.get('smoother')}")
    if row.get("backbone") != "sage":
        raise ValueError(f"{source}: expected backbone=sage, found {row.get('backbone')}")
    if row.get("use_nfr") not in {"False", "false", "0"}:
        raise ValueError(f"{source}: expected use_nfr=false, found {row.get('use_nfr')}")


def load_long_records(figure4_root: Path) -> list[VerifyRecord]:
    records: list[VerifyRecord] = []
    seen_points: set[tuple[str, str, Decimal]] = set()
    for source, config in SOURCES.items():
        manifest_path = figure4_root / config["subdir"] / "manifest.csv"
        rows = load_manifest(manifest_path)
        if len(rows) != len(MECHANISMS) * len(EPSILON_SPECS):
            raise ValueError(f"{manifest_path}: expected 45 rows, found {len(rows)}")

        for row in rows:
            validate_manifest_row(source, row, config)
            mechanism = row.get("mechanism", "").strip().lower()
            if mechanism not in MECHANISMS:
                raise ValueError(f"{source}: unexpected mechanism={row.get('mechanism')!r}")
            epsilon = decimal_key(row.get(config["eps_field"]))
            if epsilon not in EPSILON_TO_INDEX:
                raise ValueError(f"{source}/{mechanism}: unexpected epsilon={epsilon}")
            key = (source, mechanism, epsilon)
            if key in seen_points:
                raise ValueError(f"Duplicate manifest point: {key}")
            seen_points.add(key)

            prefix = config["label_prefix"]
            line_label = f"{prefix}-{MECHANISM_LABELS[mechanism]}"
            records.extend(
                collect_verify_records(
                    source=source,
                    line_label=line_label,
                    mechanism=mechanism,
                    epsilon=epsilon,
                    job_dir=Path(row["job_dir"]),
                )
            )

    expected_points = len(SOURCES) * len(MECHANISMS) * len(EPSILON_SPECS)
    if len(seen_points) != expected_points:
        raise ValueError(f"Expected {expected_points} manifest points, found {len(seen_points)}")
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
    grouped: dict[tuple[str, str, Decimal], list[VerifyRecord]] = {}
    for record in records:
        grouped.setdefault((record.source, record.mechanism, record.epsilon), []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0], EPSILON_TO_INDEX[item[0][2]])):
        source, mechanism, epsilon = key
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
                "source": source,
                "line_label": first.line_label,
                "mechanism": mechanism,
                "mechanism_label": MECHANISM_LABELS[mechanism],
                "x_eps": str(epsilon),
                "x_eps_label": first.epsilon_label,
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
        "line_label",
        "mechanism",
        "mechanism_label",
        "x_eps",
        "x_eps_label",
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
        for record in sorted(records, key=lambda r: (r.mechanism, r.source, r.x_index, r.repeat)):
            writer.writerow(
                {
                    "source": record.source,
                    "line_label": record.line_label,
                    "mechanism": record.mechanism,
                    "mechanism_label": MECHANISM_LABELS[record.mechanism],
                    "x_eps": str(record.epsilon),
                    "x_eps_label": record.epsilon_label,
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
        "line_label",
        "mechanism",
        "mechanism_label",
        "x_eps",
        "x_eps_label",
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
        for row in sorted(rows, key=lambda r: (r["mechanism"], r["source"], int(r["x_index"]))):
            writer.writerow(row)


def validate_plot_rows(rows: list[dict[str, Any]]) -> None:
    expected = len(SOURCES) * len(MECHANISMS) * len(EPSILON_SPECS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} plot rows, found {len(rows)}")
    seen = {(row["source"], row["mechanism"], row["x_eps"]) for row in rows}
    if len(seen) != expected:
        raise RuntimeError(f"Duplicate plot rows detected: unique={len(seen)} expected={expected}")
    for row in rows:
        mean = float(row["test_acc_mean"])
        low = float(row["test_acc_ci_low"])
        high = float(row["test_acc_ci_high"])
        if int(row["n"]) != EXPECTED_REPEATS:
            raise RuntimeError(f"Expected n={EXPECTED_REPEATS} for {row}")
        if not (low <= mean <= high):
            raise RuntimeError(f"CI does not contain mean for {row}")


def plot_panels(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(MECHANISMS), figsize=(FIGSIZE_X, FIGSIZE_Y), sharex=False, sharey=False)
    fig.subplots_adjust(bottom=BOTTOM, top=TOP, left=LEFT, right=RIGHT, wspace=0.10)

    for col_idx, mechanism in enumerate(MECHANISMS):
        ax = axes[col_idx]
        ax.set_title(
            f"({chr(97 + col_idx)}) {MECHANISM_LABELS[mechanism]}",
            fontsize=TITLE_FONTSIZE,
            fontweight="medium",
            pad=TITLE_PAD,
        )
        panel = [row for row in rows if row["mechanism"] == mechanism]
        for source in ("ldp", "sim"):
            line_label = f"{SOURCES[source]['label_prefix']}-{MECHANISM_LABELS[mechanism]}"
            sub = sorted([row for row in panel if row["line_label"] == line_label], key=lambda row: int(row["x_index"]))
            if len(sub) != len(EPSILON_SPECS):
                raise RuntimeError(f"{mechanism}/{source}: expected {len(EPSILON_SPECS)} rows, found {len(sub)}")
            style = STYLE_CONFIGS[line_label]
            x = np.asarray([int(row["x_index"]) for row in sub], dtype=float)
            y = np.asarray([float(row["test_acc_mean"]) for row in sub], dtype=float)
            low = np.asarray([float(row["test_acc_ci_low"]) for row in sub], dtype=float)
            high = np.asarray([float(row["test_acc_ci_high"]) for row in sub], dtype=float)
            ax.errorbar(
                x,
                y,
                yerr=np.vstack([y - low, high - y]),
                label=line_label,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=MARKERSIZE,
                linewidth=LINEWIDTH,
                capsize=0,
                markerfacecolor="none",
                markeredgewidth=2,
            )

        ax.set_xticks(range(len(EPSILON_LABELS)))
        ax.set_xticklabels(EPSILON_TICK_LABELS, rotation=35, ha="right")
        ax.set_yticks(Y_TICKS[mechanism])
        ax.set_xlabel(
            r"$\epsilon$",
            fontsize=X_LABEL_FONTSIZE,
            fontweight="medium",
            labelpad=X_LABELPAD,
        )
        ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
        ax.tick_params(axis="both", which="major", labelsize=TICKLABEL_FONTSIZE)
        if col_idx == 0:
            ax.set_ylabel("Test Accuracy", fontsize=FONTSIZE, fontweight="medium")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STYLE_CONFIGS[label]["color"],
            marker=STYLE_CONFIGS[label]["marker"],
            linestyle=STYLE_CONFIGS[label]["linestyle"],
            markersize=MARKERSIZE,
            linewidth=LINEWIDTH,
            markerfacecolor="none",
            markeredgewidth=2,
            label=label,
        )
        for label in LINE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=LEGEND_ANCHOR,
        ncol=len(LINE_ORDER),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0,
        handlelength=1.5,
        columnspacing=0.9,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure4_final_ldp_sim_panels.pdf"
    png_path = output_dir / "figure4_final_ldp_sim_panels.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=EXPORT_PAD_INCHES)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    records = load_long_records(args.figure4_root)
    plot_rows = build_plot_rows(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    validate_plot_rows(plot_rows)

    long_csv_path = args.output_dir / "figure4_final_panel_long.csv"
    plot_data_path = args.output_dir / "figure4_final_panel_plot_data.csv"
    write_long_csv(long_csv_path, records)
    write_plot_data(plot_data_path, plot_rows)
    plot_panels(plot_rows, args.output_dir)
    print(f"Saved {long_csv_path}")
    print(f"Saved {plot_data_path}")
    print(f"Long rows: {len(records)}")
    print(f"Plot rows: {len(plot_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
