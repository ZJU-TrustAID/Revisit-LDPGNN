from __future__ import annotations

from artificial_node_feature_generator.cache import (
    cache_publication_lock,
    feature_cache_path,
    load_cached_features,
    save_cached_features,
)
from artificial_node_feature_generator.graph import clone_data, get_feature_device, get_feature_dtype, graph_fingerprint
from artificial_node_feature_generator.registry import get_provider

def rewrite_features(
    data,
    feature: str,
    params: dict | None = None,
    seed: int | None = None,
):
    params = {} if params is None else dict(params)
    target = clone_data(data)
    provider = get_provider(feature)
    graph_key = graph_fingerprint(target)
    cache_seed = None
    cache_file = None

    output = None
    if provider.cacheable:
        cache_seed = provider.cache_seed(seed=seed, params=params)
        cache_file = feature_cache_path(feature=feature, seed=cache_seed, params=params, graph_key=graph_key)
        cached = load_cached_features(cache_file)
        if cached is not None:
            output = provider.provider_output_from_cache(cached)

    if output is None:
        if provider.cacheable:
            with cache_publication_lock(cache_file):
                cached = load_cached_features(cache_file)
                if cached is not None:
                    output = provider.provider_output_from_cache(cached)
                else:
                    output = provider.build(
                        target,
                        params=params,
                        seed=seed,
                    )
                    save_cached_features(
                        cache_file,
                        features=output.features,
                        params=params,
                        seed=cache_seed,
                        graph_key=graph_key,
                    )
        else:
            output = provider.build(
                target,
                params=params,
                seed=seed,
            )

    features = output.features.to(
        dtype=get_feature_dtype(target),
        device=get_feature_device(target),
    )
    target.x = features
    return target
