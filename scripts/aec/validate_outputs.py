from __future__ import annotations

import argparse
import json
from .paths import normalize_mode
from .search_results import TARGETS, validate_search_output
from .table_settings import TABLE6_FEATURE_DIM


def validate_outputs(
    mode: str = "scaled", *, table6_feature_dim: int | list[int] = TABLE6_FEATURE_DIM,
) -> list[dict]:
    mode = normalize_mode(mode)
    if mode not in {"scaled", "full"}:
        raise ValueError("Search validation requires scaled or full mode")
    cache = {}
    return [
        validate_search_output(target, mode, cache=cache, feature_dim=table6_feature_dim)
        for target in TARGETS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate search results, seed coverage and figure/table statistics.")
    parser.add_argument("--mode", choices=("scaled", "full"), default="scaled")
    parser.add_argument("--table6-feature-dim", type=int, nargs="+", default=[TABLE6_FEATURE_DIM])
    args = parser.parse_args()
    print(json.dumps(validate_outputs(args.mode, table6_feature_dim=args.table6_feature_dim), indent=2))


if __name__ == "__main__":
    main()
