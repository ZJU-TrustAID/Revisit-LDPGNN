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
    parser = argparse.ArgumentParser(description="Run one verify repeat for a ranked candidate.")
    parser.add_argument("job_dir", type=str, help="job directory containing verify_topk.csv")
    parser.add_argument("--rank", type=int, required=True, help="rank from verify_topk.csv")
    parser.add_argument("--repeat_id", type=int, required=True, help="1-based verify repeat id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rank < 1:
        raise RuntimeError("--rank must be >= 1")
    if args.repeat_id < 1:
        raise RuntimeError("--repeat_id must be >= 1")

    ctx = resolve_job_context(args.job_dir)
    job_dir = Path(args.job_dir).resolve()
    ranked = mechanism_stage_utils.ranked_candidate_by_rank(
        mechanism_stage_utils.verify_topk_path(job_dir),
        args.rank,
    )
    verify_defaults = ctx.defaults["stage"]["verify"]
    seed = ctx.base_seed + args.repeat_id - 1
    output_dir = mechanism_stage_utils.verify_candidate_output_dir(job_dir, ranked, args.repeat_id)
    if mechanism_stage_utils.candidate_result_exists(
        output_dir,
        ranked.candidate,
        expected_seed=seed,
    ):
        return

    command = build_stage_command(
        ctx,
        ranked.candidate,
        stage_name="verify",
        output_dir=output_dir,
        seed=seed,
        max_epochs=int(verify_defaults["max_epochs"]),
        patience=int(verify_defaults["patience"]),
        repeats=1,
    )

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(command, cwd=str(repo_root), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Verify task failed with exit code {completed.returncode}")
    if not mechanism_stage_utils.candidate_result_exists(
        output_dir,
        ranked.candidate,
        expected_seed=seed,
    ):
        raise RuntimeError(f"Verify task finished but produced no matching CSV under {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

