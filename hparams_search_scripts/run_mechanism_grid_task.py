#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from hparams_search_scripts.mechanism_stage_context import build_stage_command, resolve_job_context
    from hparams_search_scripts import mechanism_stage_utils
except ModuleNotFoundError:
    from mechanism_stage_context import build_stage_command, resolve_job_context  # type: ignore
    import mechanism_stage_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one grid candidate for a YAML-driven search job.")
    parser.add_argument("job_dir", type=str, help="job directory containing job_spec.yaml")
    parser.add_argument("--candidate_id", type=int, required=True, help="candidate id from job_spec.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = resolve_job_context(args.job_dir)
    candidate = mechanism_stage_utils.job_candidate_by_id(ctx.job_spec, args.candidate_id)

    job_dir = Path(args.job_dir).resolve()
    output_dir = mechanism_stage_utils.grid_candidate_output_dir(job_dir, candidate)
    if mechanism_stage_utils.candidate_result_exists(
        output_dir,
        candidate,
        expected_seed=ctx.base_seed,
    ):
        return

    grid_defaults = ctx.defaults["stage"]["grid"]
    command = build_stage_command(
        ctx,
        candidate,
        stage_name="grid",
        output_dir=output_dir,
        seed=ctx.base_seed,
        max_epochs=int(grid_defaults["max_epochs"]),
        patience=int(grid_defaults["patience"]),
        repeats=int(grid_defaults["repeats"]),
    )

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(command, cwd=str(repo_root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Grid task failed with exit code {completed.returncode}")
    if not mechanism_stage_utils.candidate_result_exists(
        output_dir,
        candidate,
        expected_seed=ctx.base_seed,
    ):
        raise RuntimeError(f"Grid task finished but produced no matching CSV under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

