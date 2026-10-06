from __future__ import annotations

import os
from pathlib import Path

from hparams_search_scripts import gpu_resources

REPO_ROOT = Path(__file__).resolve().parents[2]
AEC_ROOT = REPO_ROOT / "scripts" / "aec"
REFERENCE_ROOT = AEC_ROOT / "reference"
FIXED_ROOT = AEC_ROOT / "fixed_hparams"
SCALED_ROOT = AEC_ROOT / "scaled_search"
WORK_ROOT = REPO_ROOT / "work"
OUTPUT_ROOT = REPO_ROOT / "outputs"


def normalize_mode(mode: str) -> str:
    normalized = mode.removeprefix("search_")
    if normalized not in {"reference", "fixed", "scaled", "full"}:
        raise ValueError(f"Unsupported result mode: {mode!r}")
    return normalized


def result_root(target: int | str, mode: str) -> Path:
    normalized = normalize_mode(mode)
    if normalized == "reference":
        return REFERENCE_ROOT
    name = f"figure{target}" if isinstance(target, int) else target
    return WORK_ROOT / "search" / name / normalized

def ensure_runtime_dirs() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def parse_gpu_ids(value: gpu_resources.GpuIds | None = None) -> list[int]:
    selected = os.environ.get("AEC_GPU_IDS", "default") if value is None else value
    return gpu_resources.parse_gpu_ids(selected)

def max_parallel_per_gpu(gpu_ids: list[int] | None = None) -> int | dict[int, int]:
    selected = parse_gpu_ids() if gpu_ids is None else gpu_ids
    return gpu_resources.resolve_gpu_concurrency(
        selected, os.environ.get("AEC_MAX_PARALLEL_PER_GPU", "default"),
    )

def output_dir(mode: str, figure_id: int) -> Path:
    mode = normalize_mode(mode)
    suffix = "" if mode == "reference" else f"_{mode}"
    path = OUTPUT_ROOT / f"figure{figure_id}{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path
