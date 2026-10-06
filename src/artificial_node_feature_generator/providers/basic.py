from __future__ import annotations

import torch

from artificial_node_feature_generator.graph import get_feature_device, get_feature_dtype, get_num_nodes
from artificial_node_feature_generator.providers.base import BaseFeatureProvider
from artificial_node_feature_generator.types import ProviderOutput


def require_feature_dim(feature_name: str, params: dict) -> int:
    if "feature_dim" not in params or params["feature_dim"] is None:
        raise ValueError(f'feature "{feature_name}" requires params["feature_dim"].')
    return int(params["feature_dim"])


def _cpu_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


class RawFeatureProvider(BaseFeatureProvider):
    name = "raw"
    source = "passthrough"

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        return ProviderOutput(features=data.x.clone(), source=self.source, cacheable=False)


class RandomNormalFeatureProvider(BaseFeatureProvider):
    name = "random_normal"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        generator = _cpu_generator(seed)
        features = torch.normal(
            mean=mean,
            std=std,
            size=(get_num_nodes(data), dim),
            generator=generator,
            device=torch.device("cpu"),
        ).to(dtype=get_feature_dtype(data), device=get_feature_device(data))
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class RandomSignedOnehotFeatureProvider(BaseFeatureProvider):
    name = "random_signed_onehot"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        num_nodes = get_num_nodes(data)
        generator = _cpu_generator(seed)

        features = torch.zeros(
            (num_nodes, dim),
            dtype=get_feature_dtype(data),
            device=torch.device("cpu"),
        )
        row_indices = torch.arange(num_nodes, device=torch.device("cpu"))
        sampled_indices = torch.randint(
            low=0,
            high=dim,
            size=(num_nodes,),
            generator=generator,
            device=torch.device("cpu"),
        )
        sampled_signs = torch.randint(
            low=0,
            high=2,
            size=(num_nodes,),
            generator=generator,
            device=torch.device("cpu"),
        ).to(dtype=features.dtype)
        sampled_signs = sampled_signs * 2 - 1
        features[row_indices, sampled_indices] = sampled_signs
        features = features.to(dtype=get_feature_dtype(data), device=get_feature_device(data))
        return ProviderOutput(features=features, source=self.source, cacheable=True)


class SharedFeatureProvider(BaseFeatureProvider):
    name = "shared"
    cacheable = True

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        dim = require_feature_dim(self.name, params)
        value = float(params.get("value", 1.0))
        features = torch.full(
            (get_num_nodes(data), dim),
            fill_value=value,
            dtype=get_feature_dtype(data),
            device=get_feature_device(data),
        )
        return ProviderOutput(features=features, source=self.source, cacheable=True)
