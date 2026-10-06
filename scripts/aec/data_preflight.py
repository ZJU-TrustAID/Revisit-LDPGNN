from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import time
import socket

import fsspec
import aiohttp
from datasets import load_dataset

DATASET_NAMES = ("cora", "citeseer", "facebook", "lastfm", "actor", "flickr")


def _enable_env_proxy():
    # Bound urllib/PyG downloads too; otherwise a stalled proxy connection
    # never reaches the dataset-level retry loop.
    socket.setdefaulttimeout(30)
    for protocol in ("http", "https"):
        client = fsspec.config.conf.setdefault(protocol, {}).setdefault("client_kwargs", {})
        client["trust_env"] = True
        client["timeout"] = aiohttp.ClientTimeout(total=None, connect=30, sock_read=30)

def _load_dataset_with_retry(dataset: str, data_dir: str, attempts: int = 3):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(
                dataset=dataset,
                data_dir=data_dir,
                data_range=(0.0, 1.0),
                val_ratio=0.25,
                test_ratio=0.25,
            )
        except Exception as exc:
            last_error = exc
            # PyG's Google Drive downloader may leave a truncated data.zip.
            # Remove it so the next attempt fetches a fresh archive.
            if type(exc).__name__ == "BadZipFile" or "not a zip file" in str(exc).lower():
                for archive in Path(data_dir).glob("**/data.zip"):
                    archive.unlink(missing_ok=True)
            if attempt < attempts:
                delay = 2 ** attempt
                print(
                    f"[dataset-preflight] {dataset}: attempt {attempt}/{attempts} "
                    f"failed ({type(exc).__name__}); retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
    raise last_error


def ensure_datasets(repo_root: Path, names: Iterable[str] = DATASET_NAMES) -> list[dict[str, Any]]:
    """Sequentially download/process all datasets before batch scheduling."""
    data_root = Path(repo_root) / "datasets"
    data_root.mkdir(parents=True, exist_ok=True)
    _enable_env_proxy()
    results: list[dict[str, Any]] = []
    for name in names:
        print(f"[dataset-preflight] {name}: downloading/checking", flush=True)
        data = _load_dataset_with_retry(name, str(data_root))
        record = {
            "dataset": name,
            "nodes": int(getattr(data, "num_nodes", data.x.size(0))),
            "features": int(getattr(data, "num_features", data.x.size(1))),
            "classes": int(getattr(data, "num_classes", int(data.y.max().item()) + 1)),
        }
        results.append(record)
        print(f"[dataset-preflight] {name}: ready {record}", flush=True)
    return results
