from __future__ import annotations

import math

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.utils.extmath import randomized_range_finder
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

from artificial_node_feature_generator.graph import (
    build_nx_graph,
    get_edge_index,
    get_feature_device,
    get_feature_dtype,
    get_num_nodes,
)
from artificial_node_feature_generator.providers.basic import require_feature_dim
from artificial_node_feature_generator.providers.base import BaseFeatureProvider
from artificial_node_feature_generator.types import ProviderOutput

SPARSE_EIGEN_NODE_THRESHOLD = 4096
SPARSE_EIGEN_TOL = 1e-4
RANDOMIZED_EIGEN_NODE_THRESHOLD = 20_000
RANDOMIZED_EIGEN_N_ITER = 2
RANDOMIZED_EIGEN_OVERSAMPLES = 32
RANDOMIZED_EIGEN_RANDOM_STATE = 0


def _build_sparse_adjacency_matrix(data) -> sp.csr_matrix:
    num_nodes = get_num_nodes(data)
    edge_index = get_edge_index(data).cpu().numpy()
    if edge_index.size == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    row = edge_index[0]
    col = edge_index[1]
    values = np.ones(row.shape[0], dtype=np.float64)
    adjacency = sp.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float64).tocsr()
    adjacency.sum_duplicates()
    if adjacency.nnz > 0:
        adjacency.data[:] = 1.0

    # Mirror nx.Graph semantics used elsewhere in the provider set.
    adjacency = adjacency.maximum(adjacency.transpose()).tocsr()
    adjacency.sum_duplicates()
    if adjacency.nnz > 0:
        adjacency.data[:] = 1.0
    return adjacency


def _descending_dense_eigh(matrix: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1][:dim]
    return eigenvalues[order], np.asarray(eigenvectors[:, order], dtype=np.float64)


def _descending_sparse_eigsh(matrix: sp.csr_matrix, dim: int) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = matrix.shape[0]
    if dim >= num_nodes:
        raise ValueError(f"eigen feature dim ({dim}) must be < num_nodes ({num_nodes}) for sparse eigensolvers")

    ncv = min(num_nodes, max(dim + 64, int(math.ceil(dim * 1.25))))
    if ncv <= dim:
        ncv = min(num_nodes, dim + 1)
    v0 = np.linspace(1.0, 2.0, num_nodes, dtype=np.float64)

    try:
        eigenvalues, eigenvectors = eigsh(
            matrix,
            k=dim,
            which="LA",
            tol=SPARSE_EIGEN_TOL,
            ncv=ncv,
            maxiter=max(5 * num_nodes, 10000),
            v0=v0,
        )
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is None or exc.eigenvectors is None or exc.eigenvectors.shape[1] < dim:
            raise RuntimeError(
                f"sparse eigen solver did not converge with enough eigenpairs (wanted {dim})"
            ) from exc
        eigenvalues = exc.eigenvalues
        eigenvectors = exc.eigenvectors

    order = np.argsort(eigenvalues)[::-1][:dim]
    return np.asarray(eigenvalues[order], dtype=np.float64), np.asarray(eigenvectors[:, order], dtype=np.float64)


def _descending_randomized_psd_eigh(
    matrix: sp.csr_matrix,
    dim: int,
    *,
    n_iter: int | None = None,
    n_oversamples: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    num_nodes = matrix.shape[0]
    if dim > num_nodes:
        raise ValueError(f"eigen feature dim ({dim}) cannot exceed num_nodes ({num_nodes})")

    if dim == num_nodes:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix.toarray())
        order = np.argsort(eigenvalues)[::-1][:dim]
        return np.asarray(eigenvalues[order], dtype=np.float64), np.asarray(eigenvectors[:, order], dtype=np.float64)

    if n_iter is None:
        n_iter = RANDOMIZED_EIGEN_N_ITER
    if n_oversamples is None:
        n_oversamples = RANDOMIZED_EIGEN_OVERSAMPLES
    if random_state is None:
        random_state = RANDOMIZED_EIGEN_RANDOM_STATE

    bounded_oversamples = max(1, min(int(n_oversamples), max(1, num_nodes - dim)))
    basis_size = min(num_nodes, dim + bounded_oversamples)
    matrix32 = matrix.astype(np.float32, copy=False)
    basis = randomized_range_finder(
        matrix32,
        size=basis_size,
        n_iter=max(0, int(n_iter)),
        power_iteration_normalizer="QR",
        random_state=int(random_state),
    )
    projected = basis.T @ (matrix32 @ basis)
    projected = np.asarray((projected + projected.T) * 0.5, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    order = np.argsort(eigenvalues)[::-1][:dim]
    selected_vectors = basis @ eigenvectors[:, order]
    return (
        np.asarray(eigenvalues[order], dtype=np.float64),
        np.asarray(selected_vectors, dtype=np.float64),
    )


def _safe_inverse_sqrt(values: np.ndarray) -> np.ndarray:
    inv_sqrt = np.zeros_like(values, dtype=np.float64)
    positive = values > 0.0
    inv_sqrt[positive] = np.power(values[positive], -0.5)
    return inv_sqrt


def _degree_values(data) -> torch.Tensor:
    graph = build_nx_graph(data)
    return torch.tensor([graph.degree(node) for node in range(graph.number_of_nodes())], dtype=torch.long)


def _build_random_frozen_embedding(
    indices: torch.Tensor,
    *,
    num_embeddings: int,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int | None,
) -> torch.Tensor:
    embedding = torch.nn.Embedding(num_embeddings, feature_dim, device=torch.device("cpu"))
    with torch.no_grad():
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed))
        embedding.weight.normal_(mean=0.0, std=1.0, generator=generator)
        embedding.weight.requires_grad_(False)
    features = embedding(indices.cpu()).to(dtype=dtype, device=device)
    return features


def _bucket_indices(degrees: torch.Tensor, boundaries: list[float], num_buckets: int) -> torch.Tensor:
    if not boundaries:
        return torch.zeros_like(degrees, dtype=torch.long)
    idx = torch.bucketize(degrees.float(), torch.tensor(boundaries, dtype=torch.float32), right=False)
    return idx.clamp(max=num_buckets - 1).long()


def _build_normalized_adjacency_dense(data) -> torch.Tensor:
    num_nodes = get_num_nodes(data)
    edge_index = get_edge_index(data).cpu()
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float64)

    adj_t = SparseTensor(
        row=edge_index[0].long(),
        col=edge_index[1].long(),
        value=torch.ones(edge_index.size(1), dtype=torch.float64),
        sparse_sizes=(num_nodes, num_nodes),
    )
    adj_t = gcn_norm(adj_t, add_self_loops=False)
    row, col, value = adj_t.coo()
    dense = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    dense[row.long(), col.long()] = value.to(dtype=dense.dtype)
    return dense


class NodeDegreeFeatureProvider(BaseFeatureProvider):
    name = "node_degree"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        degrees = _degree_values(data)
        feature_dim = require_feature_dim(self.name, params)
        num_embeddings = int(degrees.max().item()) + 1 if degrees.numel() else 1
        features = _build_random_frozen_embedding(
            degrees,
            num_embeddings=num_embeddings,
            feature_dim=feature_dim,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
            seed=seed,
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class DegreeBucketRangeFeatureProvider(BaseFeatureProvider):
    name = "degree_bucket_range"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        degrees = _degree_values(data)
        feature_dim = require_feature_dim(self.name, params)
        num_buckets = int(params.get("num_buckets", 10))
        range_max = int(params.get("range_max", int(degrees.max().item()) if degrees.numel() else num_buckets))
        bucket_size = max(1, math.ceil(range_max / num_buckets))
        boundaries = [bucket_size * (idx + 1) for idx in range(num_buckets - 1)]
        indices = _bucket_indices(degrees, boundaries, num_buckets)
        features = _build_random_frozen_embedding(
            indices,
            num_embeddings=num_buckets,
            feature_dim=feature_dim,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
            seed=seed,
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class DegreeBucketDistributionFeatureProvider(BaseFeatureProvider):
    name = "degree_bucket_distribution"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        degrees = _degree_values(data)
        feature_dim = require_feature_dim(self.name, params)
        num_buckets = int(params.get("num_buckets", 10))
        if degrees.numel() == 0:
            boundaries = []
        else:
            values = degrees.cpu().numpy()
            boundaries = []
            step = max(1, int(100 / num_buckets))
            for percentile in range(step, 100, step):
                try:
                    boundary = np.percentile(values, percentile, method="higher")
                except TypeError:
                    boundary = np.percentile(values, percentile, interpolation="higher")
                boundaries.append(float(boundary))
        indices = _bucket_indices(degrees, boundaries, num_buckets)
        features = _build_random_frozen_embedding(
            indices,
            num_embeddings=num_buckets,
            feature_dim=feature_dim,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
            seed=seed,
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class PageRankFeatureProvider(BaseFeatureProvider):
    name = "pagerank"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        graph = build_nx_graph(data)
        pagerank = nx.pagerank(graph)
        scores = torch.tensor([float(pagerank[idx]) for idx in range(graph.number_of_nodes())], dtype=get_feature_dtype(data))
        features = scores.unsqueeze(1).repeat(1, dim).to(device=get_feature_device(data))
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class OperatorFeatureProvider(BaseFeatureProvider):
    name = "operator"
    cacheable = True

    def cache_seed(self, *, seed: int | None, params: dict) -> int | None:
        _ = seed
        return None

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        _ = seed
        x_steps = int(params.get("x_steps", 0))
        if x_steps < 0:
            raise ValueError(f'feature "{self.name}" requires params["x_steps"] to be >= 0.')

        num_nodes = get_num_nodes(data)
        if num_nodes == 0:
            features = torch.empty((0, 0), dtype=get_feature_dtype(data), device=get_feature_device(data))
            return ProviderOutput(features=features, source=self.source, cacheable=True)

        # Existing operator caches created before the HOA-aligned x_steps semantics
        # change must be cleared manually because the cache key is intentionally unchanged.
        if x_steps == 0:
            features = torch.eye(
                num_nodes,
                dtype=get_feature_dtype(data),
                device=get_feature_device(data),
            )
            return ProviderOutput(features=features, source=self.source, cacheable=True)

        normalized_adjacency = _build_normalized_adjacency_dense(data)
        power = normalized_adjacency.clone()
        accumulated = torch.zeros_like(normalized_adjacency)

        for _ in range(x_steps):
            accumulated = accumulated + power
            power = power @ normalized_adjacency

        features = (accumulated / float(x_steps)).to(
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class EigenFeatureProvider(BaseFeatureProvider):
    name = "eigen"
    cacheable = True

    def cache_seed(self, *, seed: int | None, params: dict) -> int | None:
        return None

    def _should_use_sparse_solver(self, *, num_nodes: int, dim: int) -> bool:
        return num_nodes > SPARSE_EIGEN_NODE_THRESHOLD and dim < num_nodes - 1

    def _dense_feature_matrix(self, adjacency: sp.csr_matrix, dim: int) -> np.ndarray:
        dense_matrix = self._dense_eigen_matrix(adjacency)
        _, eigenvectors = _descending_dense_eigh(dense_matrix, dim)
        return eigenvectors

    def _sparse_feature_matrix(self, adjacency: sp.csr_matrix, dim: int) -> np.ndarray:
        sparse_matrix = self._sparse_eigen_matrix(adjacency)
        _, eigenvectors = _descending_sparse_eigsh(sparse_matrix, dim)
        return eigenvectors

    def _dense_eigen_matrix(self, adjacency: sp.csr_matrix) -> np.ndarray:
        return adjacency.toarray()

    def _sparse_eigen_matrix(self, adjacency: sp.csr_matrix) -> sp.csr_matrix:
        return adjacency

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        adjacency = _build_sparse_adjacency_matrix(data)
        num_nodes = adjacency.shape[0]
        if dim > num_nodes:
            raise ValueError(f"eigen feature dim ({dim}) cannot exceed num_nodes ({num_nodes})")

        if self._should_use_sparse_solver(num_nodes=num_nodes, dim=dim):
            selected = self._sparse_feature_matrix(adjacency, dim)
        else:
            selected = self._dense_feature_matrix(adjacency, dim)

        features = torch.as_tensor(
            np.asarray(selected, dtype=np.float32),
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class EigenNormFeatureProvider(EigenFeatureProvider):
    name = "eigen_norm"

    def _should_use_randomized_solver(self, *, num_nodes: int, dim: int) -> bool:
        return num_nodes >= RANDOMIZED_EIGEN_NODE_THRESHOLD and dim < num_nodes

    def _dense_feature_matrix(self, adjacency: sp.csr_matrix, dim: int) -> np.ndarray:
        dense_adjacency = adjacency.toarray()
        degrees = dense_adjacency.sum(axis=1)
        inv_sqrt_degree = _safe_inverse_sqrt(np.asarray(degrees, dtype=np.float64))
        normalized = (inv_sqrt_degree[:, None] * dense_adjacency) * inv_sqrt_degree[None, :]
        _, eigenvectors = _descending_dense_eigh(normalized, dim)
        return inv_sqrt_degree[:, None] * eigenvectors

    def _sparse_feature_matrix(self, adjacency: sp.csr_matrix, dim: int) -> np.ndarray:
        degrees = np.asarray(adjacency.sum(axis=1), dtype=np.float64).reshape(-1)
        inv_sqrt_degree = _safe_inverse_sqrt(degrees)
        normalizer = sp.diags(inv_sqrt_degree, format="csr")
        normalized = (normalizer @ adjacency @ normalizer).tocsr()
        if self._should_use_randomized_solver(num_nodes=adjacency.shape[0], dim=dim):
            # Shift the symmetric normalized adjacency into [0, 1] so randomized SVD
            # targets the same top eigenspace as the original largest-algebraic solve.
            shifted = (normalized.astype(np.float32, copy=False) * 0.5) + (
                sp.eye(normalized.shape[0], format="csr", dtype=np.float32) * 0.5
            )
            _, eigenvectors = _descending_randomized_psd_eigh(shifted.tocsr(), dim)
        else:
            _, eigenvectors = _descending_sparse_eigsh(normalized, dim)
        return inv_sqrt_degree[:, None] * eigenvectors
