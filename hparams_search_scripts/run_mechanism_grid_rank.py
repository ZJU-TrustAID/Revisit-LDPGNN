#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from hparams_search_scripts import mechanism_stage_utils
except ModuleNotFoundError:
    import mechanism_stage_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate grid CSVs and emit the fixed Top-5 verify set.")
    parser.add_argument("job_dir", type=str, help="job directory containing grid outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_dir = Path(args.job_dir).resolve()
    job_spec = mechanism_stage_utils.load_job_spec(job_dir)

    expected_candidates = len(mechanism_stage_utils.job_candidates(job_spec))
    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None
    ranking_rows = []
    while True:
        try:
            ranking_rows = mechanism_stage_utils.aggregate_grid_results(job_spec, job_dir)
            if len(ranking_rows) >= expected_candidates:
                break
            last_error = RuntimeError(
                f"Grid results are still incomplete: expected {expected_candidates} candidates, "
                f"found {len(ranking_rows)}"
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for grouped grid outputs under {mechanism_stage_utils.grid_stage_dir(job_dir)}"
            ) from last_error
        time.sleep(1.0)

    mechanism_stage_utils.write_grid_ranking(job_dir, ranking_rows)
    mechanism_stage_utils.write_verify_topk(
        job_dir,
        ranking_rows,
        topk=int(job_spec.get("verify_topk", mechanism_stage_utils.VERIFY_TOPK)),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
