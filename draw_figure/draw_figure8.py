#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_cora_hds_alpha_lambda_epsilon as hds  # noqa: E402
from scripts import plot_cora_pm_alpha_lambda_epsilon as pm  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "rebuttal_figure"

PM_COLOR = hds.ALPHA_COLOR
HDS_COLOR = hds.LAMBDA_COLOR
GRID_COLOR = hds.GRID_COLOR
TEXT_COLOR = hds.TEXT_COLOR

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
BACKGROUND_COLOR = "#f5f5f5"  # LaTeX xcolor: black!4
plt.rcParams["figure.facecolor"] = BACKGROUND_COLOR
plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PM/HDS alpha and lambda curves for Cora.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-points", type=int, default=3000)
    return parser.parse_args()


def build_epsilon_grid(num_points: int) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                pm.build_epsilon_grid(num_points),
                hds.build_epsilon_grid(num_points),
            ]
        )
    )


def write_curve_csv(
    path: Path,
    epsilon: np.ndarray,
    pm_m: np.ndarray,
    hds_m: np.ndarray,
    alpha_pm: np.ndarray,
    lambda_pm: np.ndarray,
    alpha_hds: np.ndarray,
    lambda_hds: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epsilon",
        "m_pm",
        "m_hds",
        "alpha_pm",
        "lambda_pm",
        "alpha_hds",
        "lambda_hds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for values in zip(
            epsilon,
            pm_m,
            hds_m,
            alpha_pm,
            lambda_pm,
            alpha_hds,
            lambda_hds,
        ):
            eps, m_pm, m_hds, a_pm, l_pm, a_hds, l_hds = values
            writer.writerow(
                {
                    "epsilon": f"{float(eps):.17g}",
                    "m_pm": int(m_pm),
                    "m_hds": int(m_hds),
                    "alpha_pm": f"{float(a_pm):.17g}",
                    "lambda_pm": f"{float(l_pm):.17g}",
                    "alpha_hds": f"{float(a_hds):.17g}",
                    "lambda_hds": f"{float(l_hds):.17g}",
                }
            )


def plot_curves(
    output_dir: Path,
    epsilon: np.ndarray,
    alpha_pm: np.ndarray,
    lambda_pm: np.ndarray,
    alpha_hds: np.ndarray,
    lambda_hds: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.19, wspace=0.08)

    panels = (
        (
            axes[0],
            "PM",
            alpha_pm,
            lambda_pm,
            r"$\alpha_{\epsilon,m}^{\mathrm{PM}}$",
            r"$\lambda_{\epsilon,m}^{\mathrm{PM}}$",
        ),
        (
            axes[1],
            "HDS",
            alpha_hds,
            lambda_hds,
            r"$\alpha_{\epsilon,m}^{\mathrm{HDS}}$",
            r"$\lambda_{\epsilon,m}^{\mathrm{HDS}}$",
        ),
    )
    for ax, title, alpha_values, lambda_values, alpha_label, lambda_label in panels:
        ax.plot(
            epsilon,
            alpha_values,
            color=PM_COLOR,
            linestyle="-",
            linewidth=1.8,
            solid_capstyle="round",
            dash_capstyle="round",
            label=alpha_label,
        )
        ax.plot(
            epsilon,
            lambda_values,
            color=HDS_COLOR,
            linestyle="-",
            linewidth=2.0,
            solid_capstyle="round",
            dash_capstyle="round",
            label=lambda_label,
        )
        ax.set_title(title, fontsize=27, color=TEXT_COLOR, pad=10)

    marked_x = np.asarray(hds.MARKED_EPS, dtype=float)
    marked_pm_m = pm.resolve_m(marked_x)
    marked_hds_m = hds.resolve_m(marked_x)
    marked_alpha_pm, marked_lambda_pm = pm.pm_coefficients(marked_x, marked_pm_m)
    marked_alpha_hds, marked_lambda_hds, _, _, _ = hds.hds_terms(marked_x, marked_hds_m)
    marked_panels = (
        (axes[0], marked_alpha_pm, marked_lambda_pm),
        (axes[1], marked_alpha_hds, marked_lambda_hds),
    )
    for ax, marked_alpha, marked_lambda in marked_panels:
        for values, color in ((marked_alpha, PM_COLOR), (marked_lambda, HDS_COLOR)):
            ax.scatter(
                marked_x,
                values,
                s=88,
                color=color,
                edgecolor="white",
                linewidth=1.8,
                zorder=4,
                clip_on=False,
            )

    all_values = np.concatenate([alpha_pm, lambda_pm, alpha_hds, lambda_hds])
    y_limits = (float(np.min(all_values)) * 0.75, float(np.max(all_values)) * 1.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(hds.EPS_MIN, hds.EPS_MAX)
        ax.set_ylim(*y_limits)
        ax.set_xlabel(r"$\epsilon$", fontsize=27, labelpad=6)
        ax.xaxis.set_major_locator(FixedLocator(list(hds.MARKED_EPS)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: hds.format_eps(value)))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.9)
        ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.45, alpha=0.45)
        ax.tick_params(axis="both", which="major", labelsize=22.5, colors=TEXT_COLOR)
        ax.legend(
            loc="upper right",
            fontsize=22.0,
            frameon=True,
            facecolor="white",
            edgecolor="#c8d1d9",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "cora_pm_hds_alpha_lambda_epsilon.pdf"
    png_path = output_dir / "cora_pm_hds_alpha_lambda_epsilon.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    epsilon = build_epsilon_grid(args.num_points)
    pm_m = pm.resolve_m(epsilon)
    hds_m = hds.resolve_m(epsilon)
    alpha_pm, lambda_pm = pm.pm_coefficients(epsilon, pm_m)
    alpha_hds, lambda_hds, _, hds_b, hds_rho = hds.hds_terms(epsilon, hds_m)
    pm.validate_values(epsilon, pm_m, alpha_pm, lambda_pm)
    hds.validate_values(epsilon, hds_m, alpha_hds, lambda_hds, hds_b, hds_rho)

    curve_csv = args.output_dir / "cora_pm_hds_alpha_lambda_epsilon_curve.csv"
    write_curve_csv(
        curve_csv,
        epsilon,
        pm_m,
        hds_m,
        alpha_pm,
        lambda_pm,
        alpha_hds,
        lambda_hds,
    )
    plot_curves(output_dir=args.output_dir, epsilon=epsilon, alpha_pm=alpha_pm,
                lambda_pm=lambda_pm, alpha_hds=alpha_hds, lambda_hds=lambda_hds)
    print(f"Saved {curve_csv}")
    print(f"Curve rows: {len(epsilon)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
