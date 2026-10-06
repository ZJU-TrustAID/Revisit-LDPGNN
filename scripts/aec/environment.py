from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
import torch
from hparams_search_scripts.gpu_resources import gpu_device_tokens
from .paths import ensure_runtime_dirs, parse_gpu_ids, max_parallel_per_gpu

def check_environment(*, require_gpu: bool = False, configure_gpu: bool = True) -> dict[str, object]:
    ensure_runtime_dirs()
    versions = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "matplotlib", "pyyaml"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count())
    if require_gpu and not cuda_available:
        raise RuntimeError("GPU execution requested but torch.cuda.is_available() is false")
    gpu_ids = parse_gpu_ids() if configure_gpu and cuda_available else []
    gpu_tokens = gpu_device_tokens(gpu_ids) if gpu_ids else {}
    parallel = max_parallel_per_gpu(gpu_ids) if gpu_ids else {}
    info = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "versions": versions,
        "cuda_available": cuda_available,
        "cuda_count": cuda_count,
        "gpu_ids": gpu_ids,
        "gpu_devices": gpu_tokens,
        "gpu_memory_gib": {
            index: torch.cuda.get_device_properties(index).total_memory / 1024**3
            for index in gpu_ids
        },
        "max_parallel_per_gpu": parallel,
    }
    print(info)
    return info
