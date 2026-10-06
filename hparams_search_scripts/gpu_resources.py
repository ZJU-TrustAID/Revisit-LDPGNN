from __future__ import annotations

import json
import os

import torch


GPU_MEMORY_PER_TASK = 6 * 1024**3
GpuIds = str | list[int]
GpuConcurrency = str | int | list[int] | dict[int, int]


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, str) and value.strip().isdecimal():
        value = int(value.strip())
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def parse_gpu_ids(value: GpuIds) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "default":
            count = torch.cuda.device_count()
            if count == 0:
                raise RuntimeError("No CUDA GPUs are visible for gpu_ids=default")
            return list(range(count))
        value = json.loads(text) if text.startswith("[") else text.split(",")
    if not isinstance(value, list) or not value:
        raise ValueError("gpu_ids must be default or a nonempty list of GPU indices")
    ids = [_integer(item, "GPU index", minimum=0) for item in value]
    if len(set(ids)) != len(ids):
        raise ValueError(f"gpu_ids contains duplicate indices: {ids}")
    return ids


def resolve_gpu_concurrency(gpu_ids: list[int], value: GpuConcurrency) -> int | dict[int, int]:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "default":
            limits = {
                index: torch.cuda.get_device_properties(index).total_memory // GPU_MEMORY_PER_TASK
                for index in gpu_ids
            }
            insufficient = [index for index, limit in limits.items() if limit == 0]
            if insufficient:
                raise ValueError(
                    f"GPU indices {insufficient} have less than 6 GiB total memory; "
                    "default concurrency would be zero"
                )
            return limits
        if text.startswith(("[", "{")):
            value = json.loads(text)
        else:
            value = _integer(text, "max_parallel_per_gpu", minimum=1)
    if isinstance(value, list):
        if len(value) != len(gpu_ids):
            raise ValueError("max_parallel_per_gpu list must have one entry for each selected GPU")
        value = dict(zip(gpu_ids, value))
    if isinstance(value, dict):
        limits = {
            _integer(index, "GPU index", minimum=0): _integer(limit, "GPU concurrency", minimum=1)
            for index, limit in value.items()
        }
        if len(limits) != len(value) or set(limits) != set(gpu_ids):
            raise ValueError("GPU concurrency mapping must contain exactly the selected GPU indices")
        return limits
    return _integer(value, "max_parallel_per_gpu", minimum=1)


def gpu_device_tokens(gpu_ids: list[int]) -> dict[int, str]:
    count = torch.cuda.device_count()
    if any(index < 0 or index >= count for index in gpu_ids):
        raise ValueError(f"Selected GPU indices {gpu_ids} exceed the {count} visible CUDA devices")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [str(index) for index in range(count)] if mask is None else [part.strip() for part in mask.split(",")]
    # Child CUDA_VISIBLE_DEVICES uses the original device indices or UUIDs selected by the parent.
    return {index: tokens[index] for index in gpu_ids}
