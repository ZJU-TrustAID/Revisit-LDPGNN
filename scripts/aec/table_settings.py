from __future__ import annotations


TABLE6_FEATURE_DIM = 1600


def table6_feature_dims(value: int | list[int]) -> tuple[int, ...]:
    values = value if isinstance(value, list) else [value]
    if not values or any(type(dim) is not int or dim <= 0 for dim in values):
        raise ValueError("Table 6 feature_dim must be a positive integer or a nonempty list of positive integers")
    if len(set(values)) != len(values):
        raise ValueError("Table 6 feature_dim must not contain duplicate dimensions")
    return tuple(sorted(values))
