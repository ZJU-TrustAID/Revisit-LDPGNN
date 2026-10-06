from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.transforms import Compose

from transforms import FeatureTransform, FeaturePerturbation, NFR
from utils import from_args

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


CACHE_KEY_VERSION = 1
CACHE_PAYLOAD_SCHEMA_VERSION = 2
LEGACY_DENSE_PAYLOAD_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = CACHE_KEY_VERSION
CACHEABLE_MECHANISMS = {"mbm", "pm", "hds"}
GRAPH_FINGERPRINT_ATTR = "_pre_smoothing_graph_fingerprint"
RAW_FINGERPRINT_ATTR = "_pre_smoothing_raw_feature_fingerprint"
SPARSE_FLAT_TENSOR_FORMAT = "sparse_flat_v1"
DENSE_TENSOR_FORMAT = "dense_v1"
# Compact storage must provide a clear win. Dense storage remains the fallback
# for other datasets/mechanisms whose outputs are not sufficiently sparse.
SPARSE_STORAGE_MAX_DENSE_FRACTION = 0.5


@dataclass(frozen=True)
class CachePreparationResult:
    status: str
    cache_path: str | None
    bytes_written: int
    mechanism_output_range_before_nfr: float | None
    resolved_m: int | None


def _capture_rng_state(device: torch.device | None) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state(device=device)
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device | None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device is not None and device.type == "cuda" and torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"], device=device)


def _canonical_float_text(value: Any) -> str:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Expected a finite float, got {value!r}")
    return f"{parsed:.12g}"


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def _lock_path_for_cache_file(cache_file: Path) -> Path:
    return cache_file.parent / f".{cache_file.name}.lock"


@contextlib.contextmanager
def _cache_publication_lock(path: str | Path):
    cache_file = Path(path)
    lock_path = _lock_path_for_cache_file(cache_file)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_torch_save(path: str | Path, payload: Any) -> None:
    cache_file = Path(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(cache_file.parent),
        prefix=f".{cache_file.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as handle:
            torch.save(payload, handle, _use_new_zipfile_serialization=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, cache_file)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def estimate_adaptive_tensor_storage_bytes(
    shape: tuple[int, ...] | list[int],
    *,
    element_size: int,
    max_nonzero: int,
) -> int:
    """Return a conservative tensor-payload estimate for the adaptive format."""
    numel = math.prod(int(size) for size in shape)
    dense_bytes = int(numel * int(element_size))
    bounded_nonzero = max(0, min(int(max_nonzero), numel))
    index_size = 4 if numel <= torch.iinfo(torch.int32).max else 8
    sparse_bytes = bounded_nonzero * (index_size + int(element_size))
    if dense_bytes == 0 or sparse_bytes >= dense_bytes * SPARSE_STORAGE_MAX_DENSE_FRACTION:
        return dense_bytes
    return sparse_bytes


def _encode_tensor_for_cache(tensor: torch.Tensor) -> tuple[dict[str, Any], int]:
    """Losslessly encode a tensor, using flat sparse storage only when beneficial."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"cache tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.layout != torch.strided:
        raise ValueError(f"cache tensor must be strided, got layout {tensor.layout}")

    detached = tensor.detach()
    dense_bytes = _tensor_nbytes(detached)
    flat = detached.reshape(-1)
    flat_indices = torch.nonzero(flat, as_tuple=False).flatten()
    index_dtype = torch.int32 if flat.numel() <= torch.iinfo(torch.int32).max else torch.int64
    sparse_bytes = estimate_adaptive_tensor_storage_bytes(
        list(detached.shape),
        element_size=detached.element_size(),
        max_nonzero=int(flat_indices.numel()),
    )

    if sparse_bytes == dense_bytes:
        dense_cpu = detached.to(device="cpu").contiguous()
        return {
            "x_format": DENSE_TENSOR_FORMAT,
            "x": dense_cpu,
        }, _tensor_nbytes(dense_cpu)

    values = flat.index_select(0, flat_indices)
    indices_cpu = flat_indices.to(device="cpu", dtype=index_dtype).contiguous()
    values_cpu = values.to(device="cpu").contiguous()
    return {
        "x_format": SPARSE_FLAT_TENSOR_FORMAT,
        "x_shape": [int(size) for size in detached.shape],
        "x_flat_indices": indices_cpu,
        "x_values": values_cpu,
    }, _tensor_nbytes(indices_cpu) + _tensor_nbytes(values_cpu)


def _validate_encoded_tensor(payload: dict[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version == LEGACY_DENSE_PAYLOAD_SCHEMA_VERSION:
        if not isinstance(payload.get("x"), torch.Tensor):
            raise ValueError("legacy cache payload is missing dense tensor x")
        return

    if schema_version != CACHE_PAYLOAD_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache payload schema version: {schema_version!r}")

    tensor_format = payload.get("x_format")
    if tensor_format == DENSE_TENSOR_FORMAT:
        if not isinstance(payload.get("x"), torch.Tensor):
            raise ValueError("dense cache payload is missing tensor x")
        return
    if tensor_format != SPARSE_FLAT_TENSOR_FORMAT:
        raise ValueError(f"unsupported cached tensor format: {tensor_format!r}")

    shape = payload.get("x_shape")
    indices = payload.get("x_flat_indices")
    values = payload.get("x_values")
    if not isinstance(shape, (tuple, list)) or any(int(size) < 0 for size in shape):
        raise ValueError("sparse cache payload has an invalid x_shape")
    if not isinstance(indices, torch.Tensor) or indices.dim() != 1:
        raise ValueError("sparse cache payload has invalid flat indices")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("sparse cache flat indices must use int32 or int64")
    if not isinstance(values, torch.Tensor) or values.dim() != 1:
        raise ValueError("sparse cache payload has invalid values")
    if indices.numel() != values.numel():
        raise ValueError("sparse cache indices and values have different lengths")

    numel = math.prod(int(size) for size in shape)
    if indices.numel() > 0:
        minimum = int(indices.min().item())
        maximum = int(indices.max().item())
        if minimum < 0 or maximum >= numel:
            raise ValueError("sparse cache flat index is outside x_shape")


def _decode_tensor_from_cache(
    payload: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    _validate_encoded_tensor(payload)
    schema_version = payload.get("schema_version")
    tensor_format = payload.get("x_format")
    if schema_version == LEGACY_DENSE_PAYLOAD_SCHEMA_VERSION or tensor_format == DENSE_TENSOR_FORMAT:
        return payload["x"].to(device=device, dtype=dtype)

    shape = tuple(int(size) for size in payload["x_shape"])
    flat = torch.zeros(math.prod(shape), device=device, dtype=dtype)
    indices = payload["x_flat_indices"].to(device=device, dtype=torch.long)
    if indices.numel() > 0:
        values = payload["x_values"].to(device=device, dtype=dtype)
        flat.index_copy_(0, indices, values)
    return flat.view(shape)


def _payload_tensor_storage_bytes(payload: dict[str, Any]) -> int:
    _validate_encoded_tensor(payload)
    if payload.get("schema_version") == LEGACY_DENSE_PAYLOAD_SCHEMA_VERSION:
        return _tensor_nbytes(payload["x"])
    if payload.get("x_format") == DENSE_TENSOR_FORMAT:
        return _tensor_nbytes(payload["x"])
    return _tensor_nbytes(payload["x_flat_indices"]) + _tensor_nbytes(payload["x_values"])


def compact_legacy_cache_file(path: str | Path) -> tuple[str, int, int]:
    """Atomically compact one legacy dense cache file when sparse storage wins."""
    cache_path = Path(path)
    before_bytes = int(cache_path.stat().st_size)
    with _cache_publication_lock(cache_path):
        payload = _load_payload(cache_path)
        if payload is None:
            return "invalid", before_bytes, 0
        if payload.get("schema_version") == CACHE_PAYLOAD_SCHEMA_VERSION:
            return "already_compact", before_bytes, before_bytes

        encoded_tensor, _ = _encode_tensor_for_cache(payload["x"])
        if encoded_tensor.get("x_format") == DENSE_TENSOR_FORMAT:
            return "dense_fallback", before_bytes, before_bytes

        upgraded = {key: value for key, value in payload.items() if key != "x"}
        upgraded["schema_version"] = CACHE_PAYLOAD_SCHEMA_VERSION
        upgraded.update(encoded_tensor)
        _atomic_torch_save(cache_path, upgraded)
    return "compacted", before_bytes, int(cache_path.stat().st_size)


def _get_edge_index(data) -> torch.Tensor:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is not None:
        if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("data.edge_index must be a [2, num_edges] tensor.")
        return edge_index.long()

    adj_t = getattr(data, "adj_t", None)
    if adj_t is None:
        raise ValueError("data must provide either edge_index or adj_t.")

    if hasattr(adj_t, "coo"):
        row, col, _ = adj_t.coo()
        return torch.stack([row.long(), col.long()], dim=0)

    if isinstance(adj_t, torch.Tensor):
        if adj_t.is_sparse:
            return adj_t.coalesce().indices().long()
        return torch.nonzero(adj_t, as_tuple=False).t().long()

    raise TypeError(f"Unsupported adj_t type: {type(adj_t).__name__}")


def _sorted_unique_edges(data) -> list[tuple[int, int]]:
    edge_index = _get_edge_index(data).cpu()
    edges: set[tuple[int, int]] = set()
    for src, dst in edge_index.t().tolist():
        edges.add((int(src), int(dst)))
    return sorted(edges)


def graph_fingerprint(data) -> str:
    cached = getattr(data, GRAPH_FINGERPRINT_ATTR, None)
    if isinstance(cached, str) and cached:
        return cached

    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is None:
        x = getattr(data, "x", None)
        if x is None:
            raise ValueError("data must define num_nodes or x.")
        num_nodes = int(x.size(0))

    payload = {
        "num_nodes": int(num_nodes),
        "edges": _sorted_unique_edges(data),
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    try:
        setattr(data, GRAPH_FINGERPRINT_ATTR, digest)
    except Exception:
        pass
    return digest


def raw_feature_fingerprint(data) -> str:
    cached = getattr(data, RAW_FINGERPRINT_ATTR, None)
    if isinstance(cached, str) and cached:
        return cached

    x = getattr(data, "x", None)
    if x is None or not isinstance(x, torch.Tensor) or x.dim() != 2:
        raise ValueError("data.x must exist and be a 2D tensor.")

    x_cpu = x.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {
            "shape": list(x_cpu.shape),
            "dtype": str(x_cpu.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = hashlib.sha256()
    payload.update(header)
    payload.update(x_cpu.numpy().tobytes(order="C"))
    digest = payload.hexdigest()[:16]
    try:
        setattr(data, RAW_FINGERPRINT_ATTR, digest)
    except Exception:
        pass
    return digest


def resolve_cache_root(raw_root: str | os.PathLike[str] | None) -> Path | None:
    if raw_root is None:
        return None
    text = str(raw_root).strip()
    if text == "":
        return None
    return Path(text).expanduser().resolve()


def _canonical_x_eps_text(value: Any) -> str:
    parsed = float(value)
    if math.isfinite(parsed):
        return _canonical_float_text(parsed)
    if math.isinf(parsed) and parsed > 0:
        return "inf"
    raise ValueError(f"Expected a positive finite value or inf for x_eps, got {value!r}")


def is_cache_eligible(args) -> bool:
    feature = str(getattr(args, "feature", "")).strip().lower()
    if feature not in {"raw", "sim"}:
        return False
    mechanism = str(getattr(args, "mechanism", "")).strip().lower()
    if mechanism not in CACHEABLE_MECHANISMS:
        return False
    if feature == "raw":
        x_eps = float(getattr(args, "x_eps"))
        if not math.isfinite(x_eps):
            return False
    else:
        sim_reference_eps = getattr(args, "sim_reference_eps", None)
        if sim_reference_eps is None:
            return False
        sim_reference_eps = float(sim_reference_eps)
        if not math.isfinite(sim_reference_eps) or sim_reference_eps <= 0:
            return False
    return resolve_cache_root(getattr(args, "pre_smoothing_feature_cache_root", None)) is not None


def build_cache_key_payload(data, args, rewrite_seed: int | None) -> dict[str, Any]:
    raw_x = getattr(data, "x", None)
    if raw_x is None or not isinstance(raw_x, torch.Tensor) or raw_x.dim() != 2:
        raise ValueError("data.x must exist and be a 2D tensor for pre-smoothing feature caching.")

    raw_shape = [int(raw_x.size(0)), int(raw_x.size(1))]
    raw_dtype = str(raw_x.dtype)
    data_range = getattr(args, "data_range", None)
    if data_range is None or len(data_range) != 2:
        raise ValueError("args.data_range must contain exactly two values.")

    feature = str(getattr(args, "feature")).strip().lower()
    if feature not in {"raw", "sim"}:
        raise ValueError(f"pre-smoothing feature cache only supports raw/sim features, got {feature!r}")

    payload = {
        "version": CACHE_KEY_VERSION,
        "dataset": str(getattr(args, "dataset")),
        "feature": feature,
        "graph": graph_fingerprint(data),
        "raw_feature_fingerprint": raw_feature_fingerprint(data),
        "raw_feature_shape": raw_shape,
        "raw_feature_dtype": raw_dtype,
        "mechanism": str(getattr(args, "mechanism")).strip().lower(),
        "x_eps": _canonical_x_eps_text(getattr(args, "x_eps")),
        "sim_reference_eps": None
        if getattr(args, "sim_reference_eps", None) is None
        else _canonical_float_text(getattr(args, "sim_reference_eps")),
        "m": str(getattr(args, "m")),
        "norm": bool(getattr(args, "norm")),
        "norm_scale": str(getattr(args, "norm_scale")),
        "data_range": [
            _canonical_float_text(data_range[0]),
            _canonical_float_text(data_range[1]),
        ],
        "use_nfr": bool(getattr(args, "use_nfr", False)),
        "tao2": None
        if getattr(args, "tao2", None) is None
        else _canonical_float_text(getattr(args, "tao2")),
        "seed": None if rewrite_seed is None else int(rewrite_seed),
    }
    return payload


def cache_path_for_payload(cache_root: Path, payload: dict[str, Any]) -> Path:
    normalized = _normalize_json_value(payload)
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return cache_root / str(payload["dataset"]) / f"{digest}.pt"


def _apply_nfr_if_enabled(data, args):
    if not bool(getattr(args, "use_nfr", False)):
        return data

    output_range = getattr(data, "output_range", None)
    if output_range is None:
        raise ValueError("NFR requires data.output_range to be available, but it is None in the current configuration.")

    nfr = NFR(B=output_range, tao2=getattr(args, "tao2", None))
    return nfr(data)


def _compute_pre_smoothing_tensor(data, args, rewrite_seed: int | None):
    feature_transform = from_args(FeatureTransform, args).set_rewrite_seed(rewrite_seed)
    feature_perturbation = from_args(FeaturePerturbation, args)
    data = Compose([
        feature_transform,
        feature_perturbation,
    ])(data)

    mechanism_output_range_before_nfr = getattr(data, "output_range", None)
    resolved_m = getattr(data, "feature_mechanism_resolved_m", None)
    data = _apply_nfr_if_enabled(data, args)
    post_rng_state = _capture_rng_state(getattr(getattr(data, "x", None), "device", None))
    return data, mechanism_output_range_before_nfr, resolved_m, post_rng_state


def _load_payload(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("cache payload must be a mapping")
        _validate_encoded_tensor(payload)
    except Exception:
        try:
            cache_path.unlink()
        except FileNotFoundError:
            pass
        return None
    return payload


def _restore_from_payload(data, payload: dict[str, Any], cache_path: Path) -> CachePreparationResult:
    device = getattr(getattr(data, "x", None), "device", torch.device("cpu"))
    current_x = getattr(data, "x", None)
    if isinstance(current_x, torch.Tensor):
        dtype = current_x.dtype
    elif isinstance(payload.get("x"), torch.Tensor):
        dtype = payload["x"].dtype
    else:
        dtype = payload["x_values"].dtype
    data.x = _decode_tensor_from_cache(payload, device=device, dtype=dtype)
    data.output_range = payload.get("mechanism_output_range_before_nfr")
    data.feature_mechanism_resolved_m = payload.get("resolved_m")
    data.feature_mechanism = payload.get("mechanism")
    data.pre_smoothing_feature_cache_status = "hit"
    data.pre_smoothing_feature_cache_path = str(cache_path)
    rng_state = payload.get("post_rng_state")
    if isinstance(rng_state, dict):
        _restore_rng_state(rng_state, getattr(data.x, "device", None))
    return CachePreparationResult(
        status="hit",
        cache_path=str(cache_path),
        bytes_written=_payload_tensor_storage_bytes(payload),
        mechanism_output_range_before_nfr=payload.get("mechanism_output_range_before_nfr"),
        resolved_m=payload.get("resolved_m"),
    )


def prepare_pre_smoothing_input(data, args, rewrite_seed: int | None = None, *, force_rebuild: bool = False):
    cache_root = resolve_cache_root(getattr(args, "pre_smoothing_feature_cache_root", None))
    if not is_cache_eligible(args) or cache_root is None:
        data, mechanism_output_range_before_nfr, resolved_m, _ = _compute_pre_smoothing_tensor(
            data,
            args,
            rewrite_seed,
        )
        data.pre_smoothing_feature_cache_status = "disabled"
        data.pre_smoothing_feature_cache_path = None
        return data, CachePreparationResult(
            status="disabled",
            cache_path=None,
            bytes_written=int(data.x.numel() * data.x.element_size()),
            mechanism_output_range_before_nfr=mechanism_output_range_before_nfr,
            resolved_m=resolved_m,
        )

    key_payload = build_cache_key_payload(data, args, rewrite_seed)
    cache_path = cache_path_for_payload(cache_root, key_payload)

    if not force_rebuild:
        cached = _load_payload(cache_path)
        if cached is not None:
            return data, _restore_from_payload(data, cached, cache_path)

    data, mechanism_output_range_before_nfr, resolved_m, post_rng_state = _compute_pre_smoothing_tensor(
        data,
        args,
        rewrite_seed,
    )
    encoded_tensor, encoded_tensor_bytes = _encode_tensor_for_cache(data.x)
    payload = {
        "schema_version": CACHE_PAYLOAD_SCHEMA_VERSION,
        "key": _normalize_json_value(key_payload),
        "mechanism_output_range_before_nfr": mechanism_output_range_before_nfr,
        "resolved_m": resolved_m,
        "post_rng_state": post_rng_state,
        "use_nfr": bool(getattr(args, "use_nfr", False)),
        "tao2": getattr(args, "tao2", None),
        "mechanism": str(getattr(args, "mechanism")).strip().lower(),
        **encoded_tensor,
    }
    with _cache_publication_lock(cache_path):
        if not force_rebuild:
            cached = _load_payload(cache_path)
            if cached is not None:
                return data, _restore_from_payload(data, cached, cache_path)
        _atomic_torch_save(cache_path, payload)

    data.pre_smoothing_feature_cache_status = "built"
    data.pre_smoothing_feature_cache_path = str(cache_path)
    return data, CachePreparationResult(
        status="built" if not force_rebuild else "rebuilt",
        cache_path=str(cache_path),
        bytes_written=encoded_tensor_bytes,
        mechanism_output_range_before_nfr=mechanism_output_range_before_nfr,
        resolved_m=resolved_m,
    )
