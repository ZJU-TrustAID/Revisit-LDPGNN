#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "rebuttal_figure"

CORA_FEATURE_DIM = 1433
BEST_DIVISOR = 2.5
EPS_MIN = 1e-3
EPS_MAX = 100.0
MARKED_EPS = (0.001, 0.01, 0.1, 1.0, 10.0)

ALPHA_COLOR = "#0f6b78"
LAMBDA_COLOR = "#d9822b"
GRID_COLOR = "#d7dde2"
TEXT_COLOR = "#1f2933"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PM alpha and lambda against epsilon for Cora.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=3000)
    return parser.parse_args()


def resolve_m(epsilon: np.ndarray | float) -> np.ndarray:
    values = np.asarray(epsilon, dtype=float)
    return np.clip(np.floor(values / BEST_DIVISOR).astype(int), 1, CORA_FEATURE_DIM)


def pm_coefficients(
    epsilon: np.ndarray | float,
    m: np.ndarray | int,
) -> tuple[np.ndarray, np.ndarray]:
    eps = np.asarray(epsilon, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    expm1_value = np.expm1(eps / (2.0 * m_arr))
    exp_value = expm1_value + 1.0
    alpha = (
        (float(CORA_FEATURE_DIM) ** 2)
        * (exp_value + 3.0)
        / (3.0 * m_arr * expm1_value**2)
    )
    lambda_value = (
        float(CORA_FEATURE_DIM)
        * exp_value
        / (m_arr * expm1_value)
    )
    return alpha, lambda_value


def build_epsilon_grid(num_points: int) -> np.ndarray:
    if num_points < 100:
        raise ValueError("--num-points must be at least 100")
    dense = np.logspace(math.log10(EPS_MIN), math.log10(EPS_MAX), num_points)
    thresholds: list[float] = []
    for index in range(1, int(math.floor(EPS_MAX / BEST_DIVISOR)) + 1):
        threshold = index * BEST_DIVISOR
        thresholds.extend(
            [threshold * (1.0 - 1e-8), threshold, threshold * (1.0 + 1e-8)]
        )
    values = np.asarray([*dense, *MARKED_EPS, *thresholds], dtype=float)
    values = values[(values >= EPS_MIN) & (values <= EPS_MAX)]
    return np.unique(np.sort(values))


def format_eps(value: float) -> str:
    if value < 0.01:
        return f"{value:.3f}"
    if value < 0.1:
        return f"{value:.2f}"
    return f"{value:.1f}"


def validate_values(
    epsilon: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    lambda_value: np.ndarray,
) -> None:
    if CORA_FEATURE_DIM != 1433:
        raise RuntimeError(f"Expected Cora d=1433, got {CORA_FEATURE_DIM}")
    if not np.all(np.isfinite(epsilon)) or not np.all(epsilon > 0):
        raise RuntimeError("All epsilon values must be finite and positive")
    if not np.all((m >= 1) & (m <= CORA_FEATURE_DIM)):
        raise RuntimeError("All m values must be in [1, d]")
    if not np.all(np.isfinite(alpha)) or not np.all(alpha > 0):
        raise RuntimeError("All alpha values must be finite and positive")
    if not np.all(np.isfinite(lambda_value)) or not np.all(lambda_value > 0):
        raise RuntimeError("All lambda values must be finite and positive")


def write_curve_csv(
    path: Path,
    epsilon: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    lambda_value: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epsilon", "m", "alpha_pm", "lambda_pm"])
        writer.writeheader()
        for eps, m_value, alpha_value, lambda_item in zip(epsilon, m, alpha, lambda_value):
            writer.writerow(
                {
                    "epsilon": f"{float(eps):.17g}",
                    "m": int(m_value),
                    "alpha_pm": f"{float(alpha_value):.17g}",
                    "lambda_pm": f"{float(lambda_item):.17g}",
                }
            )


def plot_curve(
    output_dir: Path,
    epsilon: np.ndarray,
    alpha: np.ndarray,
    lambda_value: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.25, 4.45))
    fig.subplots_adjust(left=0.16, right=0.975, top=0.965, bottom=0.19)

    ax.plot(
        epsilon,
        alpha,
        color=ALPHA_COLOR,
        linewidth=2.6,
        solid_capstyle="round",
        label=r"$\alpha_{\epsilon,m}^{\mathrm{PM}}$",
    )
    ax.plot(
        epsilon,
        lambda_value,
        color=LAMBDA_COLOR,
        linewidth=2.6,
        solid_capstyle="round",
        label=r"$\lambda_{\epsilon,m}^{\mathrm{PM}}$",
    )

    marked_x = np.asarray(MARKED_EPS, dtype=float)
    marked_m = resolve_m(marked_x)
    marked_alpha, marked_lambda = pm_coefficients(marked_x, marked_m)
    ax.scatter(
        marked_x,
        marked_alpha,
        s=88,
        color=ALPHA_COLOR,
        edgecolor="white",
        linewidth=1.8,
        zorder=4,
        clip_on=False,
    )
    ax.scatter(
        marked_x,
        marked_lambda,
        s=88,
        color=LAMBDA_COLOR,
        edgecolor="white",
        linewidth=1.8,
        zorder=4,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(EPS_MIN, EPS_MAX)
    y_min = float(min(np.min(alpha), np.min(lambda_value)))
    y_max = float(max(np.max(alpha), np.max(lambda_value)))
    ax.set_ylim(bottom=y_min * 0.75, top=y_max * 1.8)
    ax.set_xlabel(r"Privacy budget $\epsilon$", fontsize=18, labelpad=8)
    ax.set_ylabel("Coefficient value", fontsize=18, labelpad=9)
    ax.xaxis.set_major_locator(FixedLocator(list(MARKED_EPS)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_eps(value)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.9)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.45, alpha=0.45)
    ax.tick_params(axis="both", which="major", labelsize=14.2, colors=TEXT_COLOR)
    ax.legend(loc="upper right", fontsize=14.2, frameon=True, facecolor="white", edgecolor="#c8d1d9")

    note = r"$d=1433$" + "\n" + r"$m=\max(1,\min(d,\lfloor\epsilon/2.5\rfloor))$"
    ax.text(
        0.035,
        0.065,
        note,
        transform=ax.transAxes,
        fontsize=13.0,
        color=TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "#f6f8fa",
            "edgecolor": "#c8d1d9",
            "alpha": 0.95,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "cora_pm_alpha_lambda_epsilon.pdf"
    png_path = output_dir / "cora_pm_alpha_lambda_epsilon.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    epsilon = build_epsilon_grid(args.num_points)
    m = resolve_m(epsilon)
    alpha, lambda_value = pm_coefficients(epsilon, m)
    validate_values(epsilon, m, alpha, lambda_value)
    curve_csv = args.output_dir / "cora_pm_alpha_lambda_epsilon_curve.csv"
    write_curve_csv(curve_csv, epsilon, m, alpha, lambda_value)
    plot_curve(args.output_dir, epsilon, alpha, lambda_value)
    print(f"Saved {curve_csv}")
    print(f"Curve rows: {len(epsilon)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
