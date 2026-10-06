#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SCRIPTS_DIR = REPO_ROOT / "scripts" / "aec" / "analytic"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "figure2"
AXIS_FONT_SCALE = 1.4
TICK_FONT_SCALE = AXIS_FONT_SCALE * 1.2
ANNOTATION_FONT_SCALE = 1.4 * 1.2

if str(RUNTIME_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SCRIPTS_DIR))

import plot_cora_alpha_epsilon_m as alpha_source  # noqa: E402
import plot_cora_mbm_rectification_coeff as coeff_source  # noqa: E402


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Cora alpha and MBM rectification coefficient as two panels."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the combined PDF and PNG will be written.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=3000,
        help="Number of dense log-spaced epsilon points.",
    )
    return parser.parse_args()


def build_marked_rows(source: ModuleType, value_name: str) -> list[dict[str, Any]]:
    value_function = getattr(source, value_name)
    rows: list[dict[str, Any]] = []
    for eps_value in source.MARKED_EPS:
        m_value = int(
            source.resolve_best_m(np.asarray([eps_value], dtype=float))[0]
        )
        value = float(
            value_function(np.asarray([eps_value], dtype=float), m_value)[0]
        )
        rows.append({"epsilon": eps_value, "value": value})
    return rows


def plot_panel(
    ax: Axes,
    *,
    source: ModuleType,
    epsilon: np.ndarray,
    values: np.ndarray,
    marked_rows: list[dict[str, Any]],
    offsets: dict[float, tuple[int, int]],
    ylabel: str,
) -> None:
    ax.plot(
        epsilon,
        values,
        color=source.CURVE_COLOR,
        linewidth=2.6,
        solid_capstyle="round",
    )

    marked_x = np.asarray([float(row["epsilon"]) for row in marked_rows], dtype=float)
    marked_y = np.asarray([float(row["value"]) for row in marked_rows], dtype=float)
    ax.scatter(
        marked_x,
        marked_y,
        s=88,
        color=source.MARKER_COLOR,
        edgecolor="white",
        linewidth=1.8,
        zorder=4,
        clip_on=False,
    )

    for row in marked_rows:
        eps_value = float(row["epsilon"])
        value = float(row["value"])
        label = rf"${source.format_sci(value)}$"
        ax.annotate(
            label,
            xy=(eps_value, value),
            xytext=offsets[eps_value],
            textcoords="offset points",
            fontsize=12.8 * ANNOTATION_FONT_SCALE,
            color=source.TEXT_COLOR,
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
    ax.set_xlim(source.EPS_MIN, source.EPS_PLOT_MAX)
    ax.set_ylim(bottom=float(np.min(values)) * 0.75, top=float(np.max(values)) * 1.8)
    ax.set_xlabel(r"$\epsilon$", fontsize=18 * AXIS_FONT_SCALE, labelpad=0)
    ax.set_title(
        ylabel,
        fontsize=18 * AXIS_FONT_SCALE,
        color=source.TEXT_COLOR,
        pad=8,
    )
    ax.xaxis.set_major_locator(FixedLocator(list(source.MARKED_EPS)))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: source.format_eps(value))
    )
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=source.GRID_COLOR, linewidth=0.9)
    ax.grid(
        True,
        which="minor",
        color=source.GRID_COLOR,
        linewidth=0.45,
        alpha=0.45,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=14.2 * TICK_FONT_SCALE,
        colors=source.TEXT_COLOR,
    )


def plot_panels(output_dir: Path, num_points: int) -> None:
    alpha_epsilon = alpha_source.build_epsilon_grid(num_points)
    alpha_m = alpha_source.resolve_best_m(alpha_epsilon)
    alpha_values = alpha_source.alpha_epsilon_m(alpha_epsilon, alpha_m)
    alpha_source.validate_values(alpha_epsilon, alpha_m, alpha_values)

    coeff_epsilon = coeff_source.build_epsilon_grid(num_points)
    coeff_m = coeff_source.resolve_best_m(coeff_epsilon)
    coeff_values = coeff_source.rectification_coeff(coeff_epsilon, coeff_m)
    coeff_source.validate_values(coeff_epsilon, coeff_m, coeff_values)

    fig = plt.figure(figsize=(12.5, 4.45))
    axes = (
        fig.add_axes((0.08, 0.19, 0.4075, 0.775)),
        fig.add_axes((0.54, 0.19, 0.4075, 0.775)),
    )

    plot_panel(
        axes[0],
        source=alpha_source,
        epsilon=alpha_epsilon,
        values=alpha_values,
        marked_rows=build_marked_rows(alpha_source, "alpha_epsilon_m"),
        offsets={
            0.001: (46, -20),
            0.01: (50, -18),
            0.1: (32, -12),
            1.0: (33, 1),
            10.0: (-155, -15),
        },
        ylabel=r"$\alpha_{\epsilon,m}$",
    )
    plot_panel(
        axes[1],
        source=coeff_source,
        epsilon=coeff_epsilon,
        values=coeff_values,
        marked_rows=build_marked_rows(coeff_source, "rectification_coeff"),
        offsets={
            0.001: (46, -20),
            0.01: (50, -18),
            0.1: (32, -12),
            1.0: (-125, -28),
            10.0: (-42, 44),
        },
        ylabel=r"$\beta(\varepsilon,d)$",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "cora_alpha_epsilon_m_panels.pdf"
    png_path = output_dir / "cora_alpha_epsilon_m_panels.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    args = parse_args()
    plot_panels(args.output_dir, args.num_points)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
