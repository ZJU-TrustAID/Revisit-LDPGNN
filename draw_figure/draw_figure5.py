#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from draw_figure.draw_figure4 import (  # noqa: E402
    DEFAULT_FEATFREE_MANIFEST,
    FEATFREE_LABEL,
    load_featfree_reference,
)


PAPER_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "aec" / "reference"
DEFAULT_FIGURE6_ROOT = PAPER_ROOT / "figure6"
DEFAULT_FIGURE6_ADD_ROOT = PAPER_ROOT / "figure6_add"
DEFAULT_FIGURE6_ADD_AGAIN_ROOT = PAPER_ROOT / "figure6_add_again"
DEFAULT_OUTPUT_DIR = DEFAULT_FIGURE6_ROOT / "plots"

EPSILON_SPECS = (
    (Decimal("0.0001"), r"$\epsilon=10^{-4}$"),
    (Decimal("0.001"), r"$\epsilon=10^{-3}$"),
    (Decimal("0.01"), r"$\epsilon=10^{-2}$"),
    (Decimal("0.1"), r"$\epsilon=10^{-1}$"),
    (Decimal("1.0"), r"$\epsilon=10^{0}$"),
)
EPSILON_TO_INDEX = {epsilon: idx for idx, (epsilon, _) in enumerate(EPSILON_SPECS)}

TAO2_SPECS = (
    (Decimal("0.5"), Decimal("0.5"), r"$2^{-1}$"),
    (Decimal("0.75"), Decimal("0.25"), r"$2^{-2}$"),
    (Decimal("0.875"), Decimal("0.125"), r"$2^{-3}$"),
    (Decimal("0.9375"), Decimal("0.0625"), r"$2^{-4}$"),
    (Decimal("0.96875"), Decimal("0.03125"), r"$2^{-5}$"),
    (Decimal("0.984375"), Decimal("0.015625"), r"$2^{-6}$"),
    (Decimal("0.9921875"), Decimal("0.0078125"), r"$2^{-7}$"),
    (Decimal("0.99609375"), Decimal("0.00390625"), r"$2^{-8}$"),
    (Decimal("0.998046875"), Decimal("0.001953125"), r"$2^{-9}$"),
    (Decimal("0.9990234375"), Decimal("0.0009765625"), r"$2^{-10}$"),
    (Decimal("0.99951171875"), Decimal("0.00048828125"), r"$2^{-11}$"),
    (Decimal("0.999755859375"), Decimal("0.000244140625"), r"$2^{-12}$"),
    (Decimal("0.9998779296875"), Decimal("0.0001220703125"), r"$2^{-13}$"),
    (Decimal("0.99993896484375"), Decimal("0.00006103515625"), r"$2^{-14}$"),
    (Decimal("0.999969482421875"), Decimal("0.000030517578125"), r"$2^{-15}$"),
    (Decimal("0.9999847412109375"), Decimal("0.0000152587890625"), r"$2^{-16}$"),
)
TAO2_TO_SPEC = {tao2: (idx, delta, label) for idx, (tao2, delta, label) in enumerate(TAO2_SPECS)}
IGNORED_TAO2_VALUES = {Decimal("0.125"), Decimal("0.25")}

BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 12345
EXPECTED_REPEATS = 10

VERIFY_DIR_RE = re.compile(
    r"^rank=(?P<rank>\d+)__repeat=(?P<repeat>\d+)__candidate=(?P<candidate>\d+)__"
    r"x_steps=(?P<x_steps>[^_]+)__learning_rate=(?P<learning_rate>[^_]+)__"
    r"weight_decay=(?P<weight_decay>[^_]+)__dropout=(?P<dropout>[^_]+)__tao2=(?P<tao2>.+)$"
)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
BACKGROUND_COLOR = "#f5f5f5"  # LaTeX xcolor: black!4
plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR

FONTSIZE = 31
LEGEND_FONTSIZE = 26
X_LABEL_FONTSIZE = LEGEND_FONTSIZE
TICKLABEL_FONTSIZE = 22
X_TICKLABEL_FONTSIZE = 0.8 * TICKLABEL_FONTSIZE
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
    tao2: Decimal
    delta: Decimal
    delta_label: str
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
    parser = argparse.ArgumentParser(description="Plot Figure 6 tao2 ablation curves.")
    parser.add_argument("--figure6-root", type=Path, default=DEFAULT_FIGURE6_ROOT)
    parser.add_argument("--figure6-add-root", type=Path, default=DEFAULT_FIGURE6_ADD_ROOT)
    parser.add_argument("--figure6-add-again-root", type=Path, default=DEFAULT_FIGURE6_ADD_AGAIN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--featfree-manifest",
        type=Path,
        default=DEFAULT_FEATFREE_MANIFEST,
        help="Manifest for the matching Cora GraphSAGE FeatFree reference.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
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


def parse_tao2_from_result_dir(path: Path) -> Decimal:
    name = path.name
    if not (name.startswith("tao2=") and name.endswith(".yaml")):
        raise ValueError(f"Unexpected tao2 result directory name: {path}")
    return decimal_key(name[len("tao2=") : -len(".yaml")], field="tao2")


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
    if row.get("norm") not in {"False", "false", "0"}:
        raise ValueError(f"{source}: expected norm=false, found {row.get('norm')!r}")
    if row.get("use_nfr") not in {"True", "true", "1"}:
        raise ValueError(f"{source}: expected use_nfr=true, found {row.get('use_nfr')!r}")


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


def read_single_result_csv(path: Path) -> tuple[str, float, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    row = rows[0]
    seed = row.get("seed", "")
    return seed, float(row["val/acc"]), float(row["test/acc"])


def collect_best_candidate_records(
    *,
    source: str,
    row: dict[str, str],
    tao2: Decimal,
    delta: Decimal,
    delta_label: str,
    x_index: int,
) -> list[VerifyRecord]:
    epsilon = decimal_key(row.get("x_eps"), field="x_eps")
    if epsilon not in EPSILON_TO_INDEX:
        raise ValueError(f"{source}: unexpected x_eps={epsilon}")
    epsilon_label = EPSILON_SPECS[EPSILON_TO_INDEX[epsilon]][1]

    job_dir = Path(row["job_dir"])
    candidate_id = int(row["best_candidate_id"])
    verify_root = job_dir / "verify_top5"
    if not verify_root.is_dir():
        raise FileNotFoundError(f"Missing verify_top5 directory: {verify_root}")

    records: list[VerifyRecord] = []
    for verify_dir in sorted(verify_root.iterdir()):
        if not verify_dir.is_dir():
            continue
        parsed = parse_verify_dir(verify_dir)
        if parsed is None or int(parsed["candidate_id"]) != candidate_id:
            continue
        csv_paths = sorted(verify_dir.glob("*.csv"))
        if len(csv_paths) != 1:
            raise ValueError(f"Expected one CSV under {verify_dir}, found {len(csv_paths)}")
        seed, val_acc, test_acc = read_single_result_csv(csv_paths[0])
        candidate = Candidate(
            candidate_id=candidate_id,
            x_steps=str(parsed["x_steps"]),
            learning_rate=str(parsed["learning_rate"]),
            weight_decay=str(parsed["weight_decay"]),
            dropout=str(parsed["dropout"]),
            tao2=str(parsed["tao2"]),
        )
        records.append(
            VerifyRecord(
                source=source,
                epsilon=epsilon,
                epsilon_label=epsilon_label,
                tao2=tao2,
                delta=delta,
                delta_label=delta_label,
                x_index=x_index,
                repeat=int(parsed["repeat"]),
                rank=int(parsed["rank"]),
                candidate=candidate,
                seed=seed,
                val_acc=val_acc,
                test_acc=test_acc,
                result_csv_path=csv_paths[0],
                source_job_dir=job_dir,
            )
        )

    if len(records) != EXPECTED_REPEATS:
        raise ValueError(
            f"{source} tao2={tao2} eps={epsilon}: expected {EXPECTED_REPEATS} records for "
            f"candidate_id={candidate_id}, found {len(records)}"
        )
    return sorted(records, key=lambda record: record.repeat)


def load_long_records(figure6_root: Path, figure6_add_root: Path, figure6_add_again_root: Path) -> list[VerifyRecord]:
    roots = (
        ("figure6", figure6_root),
        ("figure6_add", figure6_add_root),
        ("figure6_add_again", figure6_add_again_root),
    )
    records: list[VerifyRecord] = []
    seen: set[tuple[Decimal, Decimal]] = set()
    for source, root in roots:
        for manifest_path in sorted(root.glob("tao2=*.yaml/manifest.csv"), key=lambda path: parse_tao2_from_result_dir(path.parent)):
            tao2 = parse_tao2_from_result_dir(manifest_path.parent)
            if tao2 not in TAO2_TO_SPEC:
                if tao2 in IGNORED_TAO2_VALUES:
                    continue
                raise ValueError(f"Unexpected tao2={tao2} in {manifest_path}")
            x_index, delta, delta_label = TAO2_TO_SPEC[tao2]
            rows = load_manifest(manifest_path)
            if len(rows) != len(EPSILON_SPECS):
                raise ValueError(f"{manifest_path}: expected {len(EPSILON_SPECS)} rows, found {len(rows)}")
            for row in rows:
                validate_manifest_row(source, row)
                epsilon = decimal_key(row.get("x_eps"), field="x_eps")
                key = (tao2, epsilon)
                if key in seen:
                    raise ValueError(f"Duplicate tao2/epsilon key: {key}")
                seen.add(key)
                records.extend(
                    collect_best_candidate_records(
                        source=source,
                        row=row,
                        tao2=tao2,
                        delta=delta,
                        delta_label=delta_label,
                        x_index=x_index,
                    )
                )
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
    grouped: dict[tuple[Decimal, Decimal], list[VerifyRecord]] = {}
    for record in records:
        grouped.setdefault((record.epsilon, record.tao2), []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (EPSILON_TO_INDEX[item[0][0]], TAO2_TO_SPEC[item[0][1]][0])):
        epsilon, tao2 = key
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
                "tao2": str(tao2),
                "delta": str(first.delta),
                "delta_label": first.delta_label,
                "x_index": first.x_index,
                "n": len(group),
                "candidate_id": first.candidate.candidate_id,
                "verify_ranks": "|".join(str(record.rank) for record in sorted(group, key=lambda record: record.repeat)),
                "x_steps": first.candidate.x_steps,
                "learning_rate": first.candidate.learning_rate,
                "weight_decay": first.candidate.weight_decay,
                "dropout": first.candidate.dropout,
                "candidate_tao2": first.candidate.tao2,
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
        "tao2",
        "delta",
        "delta_label",
        "x_index",
        "repeat",
        "verify_rank",
        "candidate_id",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "candidate_tao2",
        "seed",
        "val_acc",
        "test_acc",
        "source_job_dir",
        "result_csv_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: (EPSILON_TO_INDEX[r.epsilon], r.x_index, r.repeat)):
            writer.writerow(
                {
                    "source": record.source,
                    "epsilon": str(record.epsilon),
                    "epsilon_label": record.epsilon_label,
                    "tao2": str(record.tao2),
                    "delta": str(record.delta),
                    "delta_label": record.delta_label,
                    "x_index": record.x_index,
                    "repeat": record.repeat,
                    "verify_rank": record.rank,
                    "candidate_id": record.candidate.candidate_id,
                    "x_steps": record.candidate.x_steps,
                    "learning_rate": record.candidate.learning_rate,
                    "weight_decay": record.candidate.weight_decay,
                    "dropout": record.candidate.dropout,
                    "candidate_tao2": record.candidate.tao2,
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
        "tao2",
        "delta",
        "delta_label",
        "x_index",
        "n",
        "candidate_id",
        "verify_ranks",
        "x_steps",
        "learning_rate",
        "weight_decay",
        "dropout",
        "candidate_tao2",
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
        for row in sorted(rows, key=lambda r: (EPSILON_TO_INDEX[Decimal(str(r["epsilon"]))], int(r["x_index"]))):
            writer.writerow(row)


def validate_plot_rows(rows: list[dict[str, Any]]) -> None:
    expected = len(EPSILON_SPECS) * len(TAO2_SPECS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} plot rows, found {len(rows)}")
    seen = {(Decimal(str(row["epsilon"])), Decimal(str(row["tao2"]))) for row in rows}
    if len(seen) != expected:
        raise RuntimeError(f"Duplicate plot rows detected: unique={len(seen)} expected={expected}")
    for epsilon, _ in EPSILON_SPECS:
        curve = [row for row in rows if Decimal(str(row["epsilon"])) == epsilon]
        if len(curve) != len(TAO2_SPECS):
            raise RuntimeError(f"{epsilon}: expected {len(TAO2_SPECS)} tao2 points, found {len(curve)}")
    for row in rows:
        mean = float(row["test_acc_mean"])
        low = float(row["test_acc_ci_low"])
        high = float(row["test_acc_ci_high"])
        values = [mean, low, high, float(row["val_acc_mean"]), float(row["test_acc_std"])]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Non-finite metric for {row}")
        if int(row["n"]) != EXPECTED_REPEATS:
            raise RuntimeError(f"Expected n={EXPECTED_REPEATS} for {row}")
        if not (low <= mean <= high):
            raise RuntimeError(f"CI does not contain mean for {row}")


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
        if len(sub) != len(TAO2_SPECS):
            raise RuntimeError(f"{epsilon}: expected {len(TAO2_SPECS)} plot points, found {len(sub)}")
        style = STYLE_CONFIGS[epsilon]
        x = np.asarray([float(row["delta"]) for row in sub], dtype=float)
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

    tick_values = [float(delta) for _, delta, _ in TAO2_SPECS]
    tick_labels = [label for _, _, label in TAO2_SPECS]
    ax.set_xscale("log", base=2)
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.invert_xaxis()
    ax.set_xlabel(
        r"$1-\tau$",
        fontsize=X_LABEL_FONTSIZE,
        fontweight="medium",
        labelpad=0,
    )
    ax.set_ylabel("Test Accuracy", fontsize=FONTSIZE, fontweight="medium")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(True, color="white", linestyle="-", linewidth=1, alpha=1.0)
    ax.tick_params(axis="x", which="major", labelsize=X_TICKLABEL_FONTSIZE)
    ax.tick_params(axis="y", which="major", labelsize=TICKLABEL_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    handles_by_label = dict(zip(labels, handles))
    epsilon_labels = [epsilon_label for _, epsilon_label in EPSILON_SPECS]
    first_labels = epsilon_labels[:3]
    second_labels = [*epsilon_labels[3:], FEATFREE_LABEL]
    first_legend = ax.legend(
        [handles_by_label[label] for label in first_labels],
        first_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.47),
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
        bbox_to_anchor=(0.5, -0.60),
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        shadow=False,
        borderpad=0.2,
        handlelength=1.5,
        columnspacing=0.85,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure6_tao2_curves.pdf"
    png_path = output_dir / "figure6_tao2_curves.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    records = load_long_records(args.figure6_root, args.figure6_add_root, args.figure6_add_again_root)
    plot_rows = build_plot_rows(
        records,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    validate_plot_rows(plot_rows)

    long_csv_path = args.output_dir / "figure6_tao2_long.csv"
    plot_data_path = args.output_dir / "figure6_tao2_plot_data.csv"
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
