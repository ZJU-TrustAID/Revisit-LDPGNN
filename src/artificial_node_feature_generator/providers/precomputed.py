from __future__ import annotations

import collections
import collections.abc
import contextlib
import importlib
import os
import random
import sys
import tempfile
from pathlib import Path

import torch

from artificial_node_feature_generator.cache import atomic_torch_save, feature_cache_dir, repo_root
from artificial_node_feature_generator.graph import get_feature_device, get_feature_dtype, get_num_nodes, graph_fingerprint, sorted_unique_edges
from artificial_node_feature_generator.providers.basic import require_feature_dim
from artificial_node_feature_generator.providers.base import BaseFeatureProvider
from artificial_node_feature_generator.types import ProviderOutput

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


class DeepWalkFeatureProvider(BaseFeatureProvider):
    name = "deepwalk"
    source = "generated"
    cacheable = False
    _MAX_MEMORY_DATA_SIZE = 1_000_000_000

    def _compat_patch(self) -> None:
        if not hasattr(collections, "Iterable"):
            collections.Iterable = collections.abc.Iterable
        if not hasattr(collections, "Mapping"):
            collections.Mapping = collections.abc.Mapping

    def _load_deepwalk_modules(self):
        submodule_root = repo_root() / "deepwalk"
        package_root = submodule_root / "deepwalk"
        if not package_root.exists():
            raise RuntimeError("DeepWalk submodule is not initialized. Run `git submodule update --init --recursive`.")
        self._compat_patch()
        if str(submodule_root) not in sys.path:
            sys.path.insert(0, str(submodule_root))
        try:
            graph_module = importlib.import_module("deepwalk.graph")
            walks_module = importlib.import_module("deepwalk.walks")
        except Exception as exc:
            raise RuntimeError("DeepWalk executable dependencies are missing or incompatible.") from exc
        return graph_module, walks_module

    def _load_word2vec(self):
        try:
            from gensim.models import Word2Vec
        except Exception as exc:
            raise RuntimeError("DeepWalk executable dependencies are missing or incompatible.") from exc
        return Word2Vec

    def _export_edgelist(self, data, path: Path, *, undirected: bool) -> None:
        edges = sorted_unique_edges(data, undirected=undirected)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._allocate_tmp_path(path)
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                for src, dst in edges:
                    handle.write(f"{src} {dst}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            self._remove_file(tmp_path)

    def _load_generated_embeddings(self, path: Path, *, num_nodes: int, feature_dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        rows: dict[str, list[float]] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline().strip().split()
                if len(header) != 2:
                    raise RuntimeError(f"DeepWalk generation failed: invalid embeddings header in {path}")
                for line in handle:
                    tokens = line.strip().split()
                    if not tokens:
                        continue
                    rows[tokens[0]] = [float(value) for value in tokens[1:]]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"DeepWalk generation failed: invalid embeddings file {path}: {exc}") from exc
        missing = [str(index) for index in range(num_nodes) if str(index) not in rows]
        if missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(f"DeepWalk generation failed: missing node ids {preview}")
        try:
            features = torch.tensor([rows[str(index)] for index in range(num_nodes)], dtype=dtype, device=device)
        except Exception as exc:
            raise RuntimeError(f"DeepWalk generation failed: invalid embeddings rows in {path}: {exc}") from exc
        if features.size(1) != feature_dim:
            raise RuntimeError(f"DeepWalk generation failed: expected dim {feature_dim}, got {features.size(1)}")
        return features

    def _cleanup_walk_files(self, walk_files: list[str]) -> None:
        for walk_file in walk_files:
            try:
                Path(walk_file).unlink()
            except FileNotFoundError:
                pass

    def _allocate_tmp_path(self, path: Path) -> Path:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        return Path(tmp_name)

    def _remove_file(self, path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    @contextlib.contextmanager
    def _generation_lock(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        lock_path = run_dir / ".lock"
        with lock_path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_cached_features(
        self,
        path: Path,
        *,
        num_nodes: int,
        feature_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        cleanup_invalid: bool,
    ) -> torch.Tensor | None:
        if not path.exists():
            return None
        try:
            features = torch.load(path, map_location="cpu")
            if not isinstance(features, torch.Tensor):
                raise RuntimeError(f"DeepWalk generation failed: invalid feature cache payload in {path}")
            if features.dim() != 2 or features.size(0) != num_nodes or features.size(1) != feature_dim:
                raise RuntimeError(
                    f"DeepWalk generation failed: invalid feature cache shape {tuple(features.shape)} in {path}"
                )
        except Exception:
            if cleanup_invalid:
                self._remove_file(path)
            return None
        return features.to(dtype=dtype, device=device)

    def _save_word2vec_format(self, keyed_vectors, path: Path) -> None:
        tmp_path = self._allocate_tmp_path(path)
        try:
            keyed_vectors.save_word2vec_format(str(tmp_path))
            os.replace(tmp_path, path)
        finally:
            self._remove_file(tmp_path)

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        feature_dim = require_feature_dim(self.name, params)
        walk_length = int(params.get("walk_length", 40))
        number_walks = int(params.get("number_walks", 10))
        window_size = int(params.get("window_size", 5))
        workers = int(params.get("workers", 1))
        undirected = bool(params.get("undirected", True))

        effective_params = {
            "feature_dim": feature_dim,
            "walk_length": walk_length,
            "number_walks": number_walks,
            "window_size": window_size,
            "workers": workers,
            "undirected": undirected,
        }
        graph_key = graph_fingerprint(data)
        run_dir = feature_cache_dir(self.name, seed=seed, params=effective_params, graph_key=graph_key)
        features_path = run_dir / "features.pt"
        embeddings_path = run_dir / "embeddings.word2vec"
        edgelist_path = run_dir / "graph.edgelist"
        num_nodes = get_num_nodes(data)
        feature_dtype = get_feature_dtype(data)
        feature_device = get_feature_device(data)

        features = self._load_cached_features(
            features_path,
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            dtype=feature_dtype,
            device=feature_device,
            cleanup_invalid=False,
        )
        if features is not None:
            return ProviderOutput(features=features, source=self.source, cacheable=False)

        # Many search candidates share the same DeepWalk cache directory, so
        # cache publication must be serialized and written atomically.
        with self._generation_lock(run_dir):
            features = self._load_cached_features(
                features_path,
                num_nodes=num_nodes,
                feature_dim=feature_dim,
                dtype=feature_dtype,
                device=feature_device,
                cleanup_invalid=True,
            )
            if features is not None:
                return ProviderOutput(features=features, source=self.source, cacheable=False)

            if embeddings_path.exists():
                try:
                    features = self._load_generated_embeddings(
                        embeddings_path,
                        num_nodes=num_nodes,
                        feature_dim=feature_dim,
                        dtype=feature_dtype,
                        device=feature_device,
                    )
                except RuntimeError:
                    self._remove_file(embeddings_path)
                else:
                    atomic_torch_save(features_path, features.detach().cpu())
                    return ProviderOutput(features=features, source=self.source, cacheable=False)

            graph_module, walks_module = self._load_deepwalk_modules()
            Word2Vec = self._load_word2vec()
            self._export_edgelist(data, edgelist_path, undirected=undirected)
            seed_value = 0 if seed is None else int(seed)
            try:
                graph_obj = graph_module.load_edgelist(str(edgelist_path), undirected=undirected)
                for node in range(num_nodes):
                    graph_obj[node]
                total_walk_steps = len(graph_obj.nodes()) * number_walks * walk_length
                if total_walk_steps < self._MAX_MEMORY_DATA_SIZE:
                    walks = graph_module.build_deepwalk_corpus(
                        graph_obj,
                        num_paths=number_walks,
                        path_length=walk_length,
                        alpha=0,
                        rand=random.Random(seed_value),
                    )
                    model = Word2Vec(
                        sentences=walks,
                        vector_size=feature_dim,
                        window=window_size,
                        min_count=0,
                        sg=1,
                        hs=1,
                        workers=workers,
                        seed=seed_value,
                    )
                else:
                    walk_files = walks_module.write_walks_to_disk(
                        graph_obj,
                        str(run_dir / "walks"),
                        num_paths=number_walks,
                        path_length=walk_length,
                        alpha=0,
                        rand=random.Random(seed_value),
                        num_workers=workers,
                    )
                    try:
                        model = Word2Vec(
                            sentences=walks_module.WalksCorpus(walk_files),
                            vector_size=feature_dim,
                            window=window_size,
                            min_count=0,
                            sg=1,
                            hs=1,
                            workers=workers,
                            seed=seed_value,
                        )
                    finally:
                        self._cleanup_walk_files(walk_files)
                self._save_word2vec_format(model.wv, embeddings_path)
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"DeepWalk generation failed: {exc}") from exc

            features = self._load_generated_embeddings(
                embeddings_path,
                num_nodes=num_nodes,
                feature_dim=feature_dim,
                dtype=feature_dtype,
                device=feature_device,
            )
            atomic_torch_save(features_path, features.detach().cpu())
        return ProviderOutput(
            features=features,
            source=self.source,
            cacheable=False,
        )
