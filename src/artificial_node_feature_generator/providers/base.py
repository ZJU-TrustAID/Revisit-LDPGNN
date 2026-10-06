from __future__ import annotations

from artificial_node_feature_generator.types import ProviderOutput


class BaseFeatureProvider:
    name = ""
    source = "generated"
    cacheable = False

    def build(self, data, *, params: dict, seed: int | None = None) -> ProviderOutput:
        raise NotImplementedError

    def cache_seed(self, *, seed: int | None, params: dict) -> int | None:
        return seed

    def provider_output_from_cache(self, features):
        return ProviderOutput(
            features=features,
            source=self.source,
            cacheable=self.cacheable,
        )
