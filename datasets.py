import os
import time
import tempfile
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.datasets import (
    Actor,
    AttributedGraphDataset,
    Planetoid,
)
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.io import fs as pyg_filesystem

from transforms import FilterTopClass, Normalize


def _torch_load_with_legacy_pickle(path, *args, **kwargs):
    """Load trusted PyG processed data across PyTorch 2.6+ defaults."""
    kwargs.setdefault("weights_only", False)
    try:
        return torch.load(path, *args, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return torch.load(path, *args, **kwargs)


# PyG's InMemoryDataset.load delegates to this helper. The bundled benchmark
# files are trusted dataset artifacts, not arbitrary model checkpoints.
pyg_filesystem.torch_load = _torch_load_with_legacy_pickle

try:
    from torch_geometric.transforms import AddTrainValTestMask
except ImportError:
    AddTrainValTestMask = None

try:
    from torch_geometric.transforms import RandomNodeSplit
except ImportError:
    RandomNodeSplit = None


def _download_with_retry(
    url: str,
    folder: str,
    attempts: int = 3,
    filename: str | None = None,
) -> str:
    """Download with bounded waits; publish only complete files."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    target = Path(folder) / (filename or url.rsplit("/", 1)[-1].split("?", 1)[0])
    marker = target.with_name(target.name + ".complete")
    if target.exists() and marker.exists():
        if marker.read_text().strip() == str(target.stat().st_size):
            return str(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        temporary = None
        try:
            print(f"Downloading {url} (attempt {attempt}/{attempts}; timeout 30s)", flush=True)
            started = last_report = time.monotonic()
            with requests.get(
                url,
                headers={"Accept-Encoding": "identity"},
                stream=True,
                timeout=(30, 30),
            ) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected is not None else None
                received = 0
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        received += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            total = f" / {expected / 1048576:.1f} MiB" if expected is not None else ""
                            print(f"[download] {target.name}: {received / 1048576:.1f} MiB{total}", flush=True)
                            last_report = now
                        if now - started > 900:
                            raise TimeoutError("download exceeded 15 minutes per attempt")
                if expected is not None and received != expected:
                    raise OSError(f"incomplete download: {received}/{expected} bytes")
                if received == 0:
                    raise OSError("empty download")
            os.replace(temporary, target)
            temporary = None
            marker.write_text(str(received))
            print(f"[download] {target.name}: complete ({received / 1048576:.1f} MiB)", flush=True)
            return str(target)
        except (OSError, requests.RequestException) as exc:
            if attempt == attempts:
                raise
            delay = 2 ** attempt
            print(f"[download] {target.name}: {type(exc).__name__}: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


class KarateClub(InMemoryDataset):
    """Local wrapper around the KarateClub node-classification benchmark format."""

    url = "https://raw.githubusercontent.com/benedekrozemberczki/karateclub/master/dataset/node_level"
    raw_parts = ("edges", "features", "target")

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self._load_processed_data()

    def _load_processed_data(self):
        try:
            return torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            return torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, "raw")

    @property
    def raw_file_names(self):
        return ["edges.csv", "features.csv", "target.csv"]

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        for part in self.raw_parts:
            _download_with_retry(
                f"{self.url}/{self.name}/{part}.csv",
                self.raw_dir,
            )

    def process(self):
        target_file = os.path.join(self.raw_dir, self.raw_file_names[2])
        y = pd.read_csv(target_file)["target"]
        y = torch.from_numpy(y.to_numpy(dtype=int))
        num_nodes = len(y)

        edge_file = os.path.join(self.raw_dir, self.raw_file_names[0])
        edge_frame = pd.read_csv(edge_file)
        edge_index = torch.from_numpy(edge_frame.to_numpy()).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes)

        feature_file = os.path.join(self.raw_dir, self.raw_file_names[1])
        feature_frame = pd.read_csv(feature_file).drop_duplicates()
        feature_frame = feature_frame.pivot(index="node_id", columns="feature_id", values="value").fillna(0)
        feature_frame = feature_frame.reindex(range(num_nodes), fill_value=0)
        x = torch.from_numpy(feature_frame.to_numpy()).float()

        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f"KarateClub-{self.name}()"


class ReliableActor(Actor):
    """Actor dataset with the maintained Geom-GCN source and requests retries."""

    url = "https://raw.githubusercontent.com/bingzhewei/geom-gcn/master"

    @property
    def raw_file_names(self):
        return ["out1_node_feature_label.txt", "out1_graph_edges.txt"]

    def download(self):
        for filename in self.raw_file_names:
            _download_with_retry(
                f"{self.url}/new_data/film/{filename}",
                self.raw_dir,
            )

    def process(self):
        with open(self.raw_paths[0]) as f:
            node_data = [row.split("\t") for row in f.read().split("\n")[1:-1]]
        rows, cols = [], []
        for node_id, feature_ids, _ in node_data:
            indices = [int(value) for value in feature_ids.split(",")]
            rows.extend([int(node_id)] * len(indices))
            cols.extend(indices)
        row, col = torch.tensor(rows), torch.tensor(cols)
        x = torch.zeros(int(row.max()) + 1, int(col.max()) + 1)
        x[row, col] = 1.0
        y = torch.empty(len(node_data), dtype=torch.long)
        for node_id, _, label in node_data:
            y[int(node_id)] = int(label)

        with open(self.raw_paths[1]) as f:
            edge_data = f.read().split("\n")[1:-1]
        edge_index = torch.tensor(
            [[int(value) for value in line.split("\t")] for line in edge_data]
        ).t().contiguous()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


class ReliableFlickr(AttributedGraphDataset):
    """Flickr downloader using the standard environment proxy settings."""

    def download(self):
        archive = _download_with_retry(
            f"https://drive.usercontent.google.com/download?id={self.datasets[self.name]}&confirm=t",
            self.raw_dir,
            filename="data.zip",
        )
        extract_zip(archive, self.raw_dir)
        Path(archive).unlink(missing_ok=True)
        extracted = Path(self.raw_dir) / f"{self.name}.attr"
        for filename in self.raw_file_names:
            (extracted / filename).replace(Path(self.raw_dir) / filename)
        import shutil
        shutil.rmtree(extracted, ignore_errors=True)


DATASET_BUILDERS = {
    "actor": ReliableActor,
    "citeseer": partial(Planetoid, name="citeseer"),
    "cora": partial(Planetoid, name="cora"),
    "facebook": partial(KarateClub, name="facebook"),
    "flickr": partial(ReliableFlickr, name="Flickr"),
    "lastfm": partial(KarateClub, name="lastfm", transform=FilterTopClass(10)),
}


_LOOKUP_TO_CANONICAL = {
    "".join(ch for ch in canonical_name.lower() if ch.isalnum()): canonical_name
    for canonical_name in DATASET_BUILDERS
}


#
# Dataset-name resolution
#
def _require_dataset_input(dataset: Any) -> str:
    if not isinstance(dataset, str):
        raise ValueError(f"dataset must be a string, got {type(dataset).__name__}")

    normalized = dataset.strip()
    if normalized == "":
        raise ValueError("dataset must be non-empty")
    return normalized


def _normalize_lookup_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


#
# Split transform construction
#
def build_split_transform(val_ratio, test_ratio):
    split_kwargs = dict(split="train_rest", num_val=val_ratio, num_test=test_ratio)
    if RandomNodeSplit is not None:
        return RandomNodeSplit(**split_kwargs)
    if AddTrainValTestMask is not None:
        return AddTrainValTestMask(**split_kwargs)
    raise ImportError(
        "Could not find a train/val/test split transform in torch_geometric.transforms. "
        "Expected AddTrainValTestMask (old versions) or RandomNodeSplit (new versions)."
    )


def list_supported_datasets() -> list[str]:
    return sorted(DATASET_BUILDERS, key=str.casefold)


def resolve_dataset_name(dataset: str) -> str:
    requested = _require_dataset_input(dataset)
    try:
        return _LOOKUP_TO_CANONICAL[_normalize_lookup_key(requested)]
    except KeyError as exc:
        raise ValueError(
            f'Unknown dataset "{requested}". Available datasets: {list_supported_datasets()}'
        ) from exc


def is_dataset_supported(dataset: str) -> bool:
    try:
        resolve_dataset_name(dataset)
    except ValueError:
        return False
    return True


def _load_raw_dataset(dataset_name: str, data_dir: str | Path) -> Data:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    dataset = DATASET_BUILDERS[dataset_name](
        root=os.path.join(str(data_dir), dataset_name)
    )
    return dataset[0]


def _validate_random_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must satisfy 0 <= val_ratio < 1, got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must satisfy 0 <= test_ratio < 1, got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be strictly smaller than 1 for random splits")


# graph standardization
def _coerce_feature_matrix(x: torch.Tensor) -> torch.Tensor:
    sparse_layouts = {
        layout
        for layout in (
            getattr(torch, "sparse_coo", None),
            getattr(torch, "sparse_csr", None),
            getattr(torch, "sparse_csc", None),
            getattr(torch, "sparse_bsr", None),
            getattr(torch, "sparse_bsc", None),
        )
        if layout is not None
    }
    if x.is_sparse or x.layout in sparse_layouts:
        x = x.to_dense()
    return x.to(torch.float)


def _looks_like_one_hot(y: torch.Tensor) -> bool:
    if y.ndim != 2 or y.numel() == 0:
        return False
    if y.is_floating_point():
        if not (bool((y >= 0).all().item()) and bool((y <= 1).all().item())):
            return False
        row_sum = y.sum(dim=1)
        return bool(torch.isclose(row_sum, torch.ones_like(row_sum)).all().item())
    if y.dtype == torch.bool:
        row_sum = y.to(torch.int).sum(dim=1)
    else:
        if not bool(((y == 0) | (y == 1)).all().item()):
            return False
        row_sum = y.sum(dim=1)
    return bool((row_sum == 1).all().item())


def _remap_labels_to_contiguous_range(y: torch.Tensor) -> torch.Tensor:
    unique_labels = torch.unique(y, sorted=True)
    expected = torch.arange(unique_labels.numel(), device=unique_labels.device)
    if torch.equal(unique_labels, expected):
        return y

    remapped = torch.empty_like(y)
    for new_label, old_label in enumerate(unique_labels.tolist()):
        remapped[y == old_label] = new_label
    return remapped


def _coerce_label_vector(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unlabeled_mask = torch.zeros(0, dtype=torch.bool)

    if y.ndim == 2 and y.size(-1) == 1:
        y = y.view(-1)
    elif y.ndim == 2 and y.size(-1) > 1:
        if _looks_like_one_hot(y):
            y = y.argmax(dim=1)
        else:
            raise ValueError(
                f"Expected one-dimensional class index labels, got shape {tuple(y.shape)}"
            )

    if y.ndim != 1:
        raise ValueError(f"Expected one-dimensional class index labels, got shape {tuple(y.shape)}")
    if y.numel() == 0:
        raise ValueError("y must not be empty")

    if y.is_floating_point():
        unlabeled_mask = torch.isnan(y)
        if bool(unlabeled_mask.all().item()):
            raise ValueError("y contains no finite class labels")
        if bool(unlabeled_mask.any().item()):
            finite_y = y[~unlabeled_mask]
            if not torch.allclose(finite_y, finite_y.round()):
                raise ValueError("y must contain integer-like class indices when finite")
            y = finite_y.round().to(torch.long)
        else:
            if not torch.allclose(y, y.round()):
                raise ValueError("y must contain integer-like class indices")
            y = y.round().to(torch.long)
    else:
        y = y.to(torch.long)

    if int(y.min().item()) < 0:
        raise ValueError("y must contain non-negative class indices")

    y = _remap_labels_to_contiguous_range(y)

    if bool(unlabeled_mask.any().item()):
        y_with_unlabeled = torch.zeros(unlabeled_mask.size(0), dtype=torch.long, device=y.device)
        y_with_unlabeled[~unlabeled_mask] = y
        return y_with_unlabeled, unlabeled_mask.to(torch.bool)

    return y, torch.zeros(y.size(0), dtype=torch.bool, device=y.device)


def _extract_edge_index(data: Data) -> torch.Tensor:
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        raise ValueError("Raw dataset must provide edge_index before standardization")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    return edge_index.to(torch.long)


def _canonicalize_graph_topology(data: Data, *, num_nodes: int) -> torch.Tensor:
    graph_topology = _extract_edge_index(data)
    graph_topology = to_undirected(graph_topology, num_nodes=num_nodes)
    return coalesce(graph_topology, num_nodes=num_nodes)


def _sample_transductive_split(data: Data, *, val_ratio: float, test_ratio: float) -> Data:
    return build_split_transform(val_ratio, test_ratio)(data)


def _validate_transductive_split_against_unlabeled_targets(
    unlabeled_mask: torch.Tensor,
    data: Data,
) -> None:
    if not bool(unlabeled_mask.any().item()):
        return

    supervised_partition = data.train_mask | data.val_mask | data.test_mask
    if bool((unlabeled_mask.to(supervised_partition.device) & supervised_partition).any().item()):
        raise ValueError("Split masks include unlabeled (NaN) targets, which is unsupported")


def _finalize_benchmark_graph(
    data: Data,
    *,
    dataset_name: str,
    num_nodes: int,
    num_classes: int,
    data_range: tuple[float, float] | None,
) -> Data:
    data = ToSparseTensor()(data)
    if data_range is not None:
        data = Normalize(*data_range)(data)

    data.train_mask = data.train_mask.to(torch.bool)
    data.val_mask = data.val_mask.to(torch.bool)
    data.test_mask = data.test_mask.to(torch.bool)
    data.name = dataset_name
    data.num_classes = num_classes
    data.num_nodes = num_nodes
    return data


def _build_standardized_benchmark_graph(
    dataset_name: str,
    raw_data: Data,
    *,
    data_range: tuple[float, float] | None,
    val_ratio: float,
    test_ratio: float,
) -> Data:
    _validate_random_split_ratios(val_ratio, test_ratio)

    feature_matrix = _coerce_feature_matrix(raw_data.x)
    label_vector, unlabeled_mask = _coerce_label_vector(raw_data.y)

    if feature_matrix.ndim != 2:
        raise ValueError(f"x must have shape [N, F], got {tuple(feature_matrix.shape)}")

    num_nodes = int(feature_matrix.size(0))
    if label_vector.size(0) != num_nodes:
        raise ValueError("x and y must agree on the first dimension")

    graph_topology = _canonicalize_graph_topology(raw_data, num_nodes=num_nodes)
    data = Data(x=feature_matrix, y=label_vector, edge_index=graph_topology, num_nodes=num_nodes)
    data = _sample_transductive_split(data, val_ratio=val_ratio, test_ratio=test_ratio)
    _validate_transductive_split_against_unlabeled_targets(unlabeled_mask, data)

    return _finalize_benchmark_graph(
        data,
        dataset_name=dataset_name,
        num_nodes=num_nodes,
        num_classes=int(label_vector.max().item()) + 1,
        data_range=data_range,
    )


def load_dataset(
    dataset: dict(help="name of the dataset", option="-d") = "cora",
    data_dir: dict(help="directory to store the dataset") = "./datasets",
    data_range: dict(help="min and max feature value", nargs=2, type=float) = (0, 1),
    val_ratio: dict(help="fraction of nodes used for validation") = 0.25,
    test_ratio: dict(help="fraction of nodes used for test") = 0.25,
):
    canonical_name = resolve_dataset_name(dataset)
    raw_data = _load_raw_dataset(canonical_name, data_dir)
    return _build_standardized_benchmark_graph(
        canonical_name,
        raw_data,
        data_range=data_range,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


__all__ = [
    "is_dataset_supported",
    "list_supported_datasets",
    "load_dataset",
    "resolve_dataset_name",
]
