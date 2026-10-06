from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


def _normalize_for_json(value: Any):
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cache_root() -> Path:
    override = os.environ.get("ARTIFICIAL_NODE_FEATURE_CACHE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / ".cache"


def build_cache_key(feature: str, seed: int | None, params: dict[str, Any], graph_key: str) -> str:
    payload = {
        "version": 2,
        "feature": feature,
        "seed": seed,
        "params": _normalize_for_json(params),
        "graph": graph_key,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest[:16]


def feature_cache_path(feature: str, seed: int | None, params: dict[str, Any], graph_key: str) -> Path:
    digest = build_cache_key(feature=feature, seed=seed, params=params, graph_key=graph_key)
    return cache_root() / feature / f"{digest}.pt"


def feature_cache_dir(feature: str, seed: int | None, params: dict[str, Any], graph_key: str) -> Path:
    digest = build_cache_key(feature=feature, seed=seed, params=params, graph_key=graph_key)
    return cache_root() / feature / digest


def _lock_path_for_cache_file(cache_file: Path) -> Path:
    return cache_file.parent / f".{cache_file.name}.lock"


@contextlib.contextmanager
def cache_publication_lock(path: str | Path):
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


def atomic_torch_save(path: str | Path, payload: Any) -> None:
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
            # Large feature tensors have hit inline_container write failures with the
            # default zipfile serializer on shared filesystems, so keep cache writes
            # on the legacy stable format.
            torch.save(payload, handle, _use_new_zipfile_serialization=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, cache_file)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def load_cached_features(path: str | Path):
    cache_file = Path(path)
    if not cache_file.exists():
        return None
    try:
        payload = torch.load(cache_file, map_location="cpu")
        if not isinstance(payload, dict) or "features" not in payload:
            raise RuntimeError(f"Invalid feature cache payload in {cache_file}")
    except Exception:
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass
        return None
    return payload["features"]


def save_cached_features(path: str | Path, *, features: torch.Tensor, params: dict[str, Any], seed: int | None, graph_key: str) -> None:
    cache_file = Path(path)
    atomic_torch_save(
        cache_file,
        {
            "features": features.detach().cpu(),
            "params": _normalize_for_json(params),
            "seed": seed,
            "graph": graph_key,
        },
    )
