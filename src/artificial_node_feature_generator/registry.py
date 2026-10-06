from __future__ import annotations

from artificial_node_feature_generator.providers import (
    BaseFeatureProvider,
    DeepWalkFeatureProvider,
    DegreeBucketDistributionFeatureProvider,
    DegreeBucketRangeFeatureProvider,
    EigenFeatureProvider,
    EigenNormFeatureProvider,
    NodeDegreeFeatureProvider,
    OperatorFeatureProvider,
    PageRankFeatureProvider,
    RandomNormalFeatureProvider,
    RandomSignedOnehotFeatureProvider,
    RawFeatureProvider,
    SharedFeatureProvider,
)


_PROVIDERS: dict[str, type[BaseFeatureProvider] | BaseFeatureProvider] = {}


def register_provider(feature_name: str, provider_cls_or_instance) -> None:
    if not isinstance(feature_name, str) or not feature_name.strip():
        raise ValueError("feature_name must be a non-empty string.")
    _PROVIDERS[feature_name] = provider_cls_or_instance


def get_provider(feature_name: str) -> BaseFeatureProvider:
    try:
        provider = _PROVIDERS[feature_name]
    except KeyError as exc:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f'Unsupported feature "{feature_name}". Available: {available}') from exc
    if isinstance(provider, BaseFeatureProvider):
        return provider
    return provider()


register_provider("raw", RawFeatureProvider)
register_provider("random_normal", RandomNormalFeatureProvider)
register_provider("random_signed_onehot", RandomSignedOnehotFeatureProvider)
register_provider("shared", SharedFeatureProvider)
register_provider("node_degree", NodeDegreeFeatureProvider)
register_provider("degree_bucket_range", DegreeBucketRangeFeatureProvider)
register_provider("degree_bucket_distribution", DegreeBucketDistributionFeatureProvider)
register_provider("pagerank", PageRankFeatureProvider)
register_provider("operator", OperatorFeatureProvider)
register_provider("eigen", EigenFeatureProvider)
register_provider("eigen_norm", EigenNormFeatureProvider)
register_provider("deepwalk", DeepWalkFeatureProvider)
