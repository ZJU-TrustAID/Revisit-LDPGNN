from __future__ import annotations

import copy
import hashlib
import json

import networkx as nx
import torch


def clone_data(data):
    if hasattr(data, "clone") and callable(data.clone):
        try:
            return data.clone()
        except Exception:
            pass
    return copy.deepcopy(data)


def get_num_nodes(data) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)
    x = getattr(data, "x", None)
    if x is None:
        raise ValueError("data must define num_nodes or x.")
    return int(x.size(0))


def get_num_features(data) -> int:
    x = getattr(data, "x", None)
    if x is None or x.dim() != 2:
        raise ValueError("data.x must exist and be a 2D tensor.")
    return int(x.size(1))


def get_feature_device(data) -> torch.device:
    x = getattr(data, "x", None)
    if x is None:
        return torch.device("cpu")
    return x.device


def get_feature_dtype(data) -> torch.dtype:
    x = getattr(data, "x", None)
    if x is None:
        return torch.float32
    return x.dtype


def get_edge_index(data) -> torch.Tensor:
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


def get_num_edges(data) -> int:
    return int(get_edge_index(data).size(1))


def build_nx_graph(data) -> nx.Graph:
    num_nodes = get_num_nodes(data)
    edge_index = get_edge_index(data).cpu()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(edge_index.t().tolist())
    return graph


def graph_summary(data) -> dict[str, int]:
    return {
        "num_nodes": get_num_nodes(data),
        "num_edges": get_num_edges(data),
    }


def sorted_unique_edges(data, *, undirected: bool = False) -> list[tuple[int, int]]:
    edge_index = get_edge_index(data).cpu()
    edges: set[tuple[int, int]] = set()
    for src, dst in edge_index.t().tolist():
        src_i = int(src)
        dst_i = int(dst)
        if undirected:
            src_i, dst_i = sorted((src_i, dst_i))
        edges.add((src_i, dst_i))
    return sorted(edges)


def graph_fingerprint(data) -> str:
    payload = {
        "num_nodes": get_num_nodes(data),
        "edges": sorted_unique_edges(data, undirected=False),
    }
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:16]
