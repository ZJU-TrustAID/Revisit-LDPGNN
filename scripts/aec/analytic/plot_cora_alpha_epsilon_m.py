#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


DEFAULT_OUTPUT_DIR = Path("outputs/figure2")

CORA_FEATURE_DIM = 1433
BEST_DIVISOR = 2.18
EPS_MIN = 1e-3
EPS_MAX = 100.0
EPS_PLOT_MAX = 100.0
MARKED_EPS = (0.001, 0.01, 0.1, 1.0, 10.0)
EXPECTED_MARKED_M = (1, 1, 1, 1, 4)

CURVE_COLOR = "#0f6b78"
MARKER_COLOR = "#d9822b"
GRID_COLOR = "#d7dde2"
TEXT_COLOR = "#1f2933"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot alpha_{epsilon,m} for Cora with MBM m=best sampling."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDF/PNG/CSV outputs will be written.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=3000,
        help="Number of dense log-spaced epsilon points.",
    )
    return parser.parse_args()


def resolve_best_m(epsilon: np.ndarray | float) -> np.ndarray:
    values = np.asarray(epsilon, dtype=float)
    m = np.floor(values / BEST_DIVISOR).astype(int)
    return np.clip(m, 1, CORA_FEATURE_DIM)


def alpha_epsilon_m(epsilon: np.ndarray | float, m: np.ndarray | int) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    eps_per_dim = eps / m_arr
    expm1_value = np.expm1(eps_per_dim)
    ratio = (expm1_value + 2.0) / expm1_value
    return (float(CORA_FEATURE_DIM) ** 2) * (ratio**2) / m_arr


def build_epsilon_grid(num_points: int) -> np.ndarray:
    if num_points < 100:
        raise ValueError("--num-points must be at least 100")
    dense = np.logspace(math.log10(EPS_MIN), math.log10(EPS_MAX), num_points)
    thresholds: list[float] = []
    max_m = int(math.floor(EPS_MAX / BEST_DIVISOR))
    for m in range(1, max_m + 1):
        threshold = m * BEST_DIVISOR
        if EPS_MIN <= threshold <= EPS_MAX:
            thresholds.extend(
                [
                    threshold * (1.0 - 1e-8),
                    threshold,
                    threshold * (1.0 + 1e-8),
                ]
            )
    values = np.asarray([*dense, *MARKED_EPS, *thresholds], dtype=float)
    values = values[(values >= EPS_MIN) & (values <= EPS_MAX)]
    return np.unique(np.sort(values))


def format_sci(value: float) -> str:
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"{coefficient}\times10^{{{int(exponent)}}}"


def format_eps(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 0.1:
        return f"{value:.2f}"
    if value < 1:
        return f"{value:.1f}"
    return f"{value:.1f}"


def validate_values(epsilon: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> None:
    if CORA_FEATURE_DIM != 1433:
        raise RuntimeError(f"Expected Cora d=1433, got {CORA_FEATURE_DIM}")
    marked_m = tuple(int(resolve_best_m(np.asarray([eps], dtype=float))[0]) for eps in MARKED_EPS)
    if marked_m != EXPECTED_MARKED_M:
        raise RuntimeError(f"Unexpected marked m values: {marked_m}")
    if not np.all(np.isfinite(epsilon)) or not np.all(epsilon > 0):
        raise RuntimeError("All epsilon values must be finite and positive")
    if not np.all((m >= 1) & (m <= CORA_FEATURE_DIM)):
        raise RuntimeError("All m values must be in [1, d]")
    if not np.all(np.isfinite(alpha)) or not np.all(alpha > 0):
        raise RuntimeError("All alpha values must be finite and positive")


def write_curve_csv(path: Path, epsilon: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epsilon", "m", "alpha"])
        writer.writeheader()
        for eps_value, m_value, alpha_value in zip(epsilon, m, alpha):
            writer.writerow(
                {
                    "epsilon": f"{float(eps_value):.17g}",
                    "m": int(m_value),
                    "alpha": f"{float(alpha_value):.17g}",
                }
            )


def write_marked_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eps_value in MARKED_EPS:
        m_value = int(resolve_best_m(np.asarray([eps_value], dtype=float))[0])
        alpha_value = float(alpha_epsilon_m(np.asarray([eps_value], dtype=float), m_value)[0])
        rows.append(
            {
                "epsilon": f"{eps_value:.17g}",
                "m": m_value,
                "alpha": f"{alpha_value:.17g}",
                "alpha_label": format_sci(alpha_value),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epsilon", "m", "alpha", "alpha_label"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_curve(output_dir: Path, epsilon: np.ndarray, alpha: np.ndarray, marked_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.25, 4.45))
    fig.subplots_adjust(left=0.16, right=0.975, top=0.965, bottom=0.19)

    ax.plot(epsilon, alpha, color=CURVE_COLOR, linewidth=2.6, solid_capstyle="round")

    marked_x = np.asarray([float(row["epsilon"]) for row in marked_rows], dtype=float)
    marked_y = np.asarray([float(row["alpha"]) for row in marked_rows], dtype=float)
    ax.scatter(
        marked_x,
        marked_y,
        s=88,
        color=MARKER_COLOR,
        edgecolor="white",
        linewidth=1.8,
        zorder=4,
        clip_on=False,
    )

    offsets = {
        0.001: (46, -20),
        0.01: (50, -18),
        0.1: (32, -12),
        1.0: (18, 16),
        10.0: (-39, 22),
    }
    for row in marked_rows:
        eps_value = float(row["epsilon"])
        alpha_value = float(row["alpha"])
        label = rf"$({format_eps(eps_value)}, {format_sci(alpha_value)})$"
        ax.annotate(
            label,
            xy=(eps_value, alpha_value),
            xytext=offsets[eps_value],
            textcoords="offset points",
            fontsize=12.8,
            color=TEXT_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            arrowprops={
                "arrowstyle": "-",
                "color": "#8b9aa5",
                "linewidth": 1.1,
                "shrinkA": 4,
                "shrinkB": 5,
            },
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(EPS_MIN, EPS_PLOT_MAX)
    ax.set_ylim(bottom=float(np.min(alpha)) * 0.75, top=float(np.max(alpha)) * 1.8)
    ax.set_xlabel(r"Privacy budget $\epsilon$", fontsize=18, labelpad=8)
    ax.set_ylabel(r"$\alpha_{\epsilon,m}$", fontsize=18, labelpad=9)
    ax.xaxis.set_major_locator(FixedLocator(list(MARKED_EPS)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_eps(value)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.9)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.45, alpha=0.45)
    ax.tick_params(axis="both", which="major", labelsize=14.2, colors=TEXT_COLOR)

    note = r"$d=1433$" + "\n" + r"$m=\max(1,\min(d,\lfloor\epsilon/2.18\rfloor))$"
    ax.text(
        0.035,
        0.065,
        note,
        transform=ax.transAxes,
        fontsize=13.0,
        color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "#f6f8fa", "edgecolor": "#c8d1d9", "alpha": 0.95},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "cora_alpha_epsilon_m.pdf"
    png_path = output_dir / "cora_alpha_epsilon_m.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    epsilon = build_epsilon_grid(args.num_points)
    m = resolve_best_m(epsilon)
    alpha = alpha_epsilon_m(epsilon, m)
    validate_values(epsilon, m, alpha)

    curve_csv = args.output_dir / "cora_alpha_epsilon_m_curve.csv"
    marked_csv = args.output_dir / "cora_alpha_epsilon_m_marked_points.csv"
    write_curve_csv(curve_csv, epsilon, m, alpha)
    marked_rows = write_marked_csv(marked_csv)
    plot_curve(args.output_dir, epsilon, alpha, marked_rows)
    print(f"Saved {curve_csv}")
    print(f"Saved {marked_csv}")
    print(f"Curve rows: {len(epsilon)}")
    print(f"Marked rows: {len(marked_rows)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
