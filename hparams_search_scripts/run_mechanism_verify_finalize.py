#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time

try:
    from hparams_search_scripts.mechanism_stage_context import build_recommended_command_parts, resolve_job_context
    from hparams_search_scripts import mechanism_stage_utils
except ModuleNotFoundError:
    from mechanism_stage_context import build_recommended_command_parts, resolve_job_context  # type: ignore
    import mechanism_stage_utils  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize verify outputs and emit best artifacts.")
    parser.add_argument("job_dir", type=str, help="job directory containing verify outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = resolve_job_context(args.job_dir)
    job_dir = ctx.job_dir

    expected_candidates = min(
        int(ctx.job_spec.get("verify_topk", mechanism_stage_utils.VERIFY_TOPK)),
        len(mechanism_stage_utils.job_candidates(ctx.job_spec)),
    )
    expected_repeats = int(ctx.defaults["stage"]["verify"]["repeats"])
    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None
    summary_rows = []
    while True:
        try:
            summary_rows = mechanism_stage_utils.aggregate_verify_results(ctx.job_spec, job_dir)
            if len(summary_rows) >= expected_candidates and all(
                int(row.get("n", 0)) >= expected_repeats for row in summary_rows
            ):
                break
            last_error = RuntimeError(
                f"Verify results are still incomplete: expected {expected_candidates} candidates "
                f"with {expected_repeats} repeats, found {len(summary_rows)} summaries"
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for verify outputs under "
                f"{mechanism_stage_utils.verify_stage_dir(job_dir)}"
            ) from last_error
        time.sleep(1.0)

    mechanism_stage_utils.write_verify_summary(job_dir, summary_rows)
    winner = summary_rows[0]
    candidate = mechanism_stage_utils.job_candidate_by_id(ctx.job_spec, int(winner["candidate_id"]))
    recommended_parts = build_recommended_command_parts(ctx, candidate)
    recommended_command = mechanism_stage_utils.shell_join(recommended_parts)
    mechanism_stage_utils.recommended_command_path(job_dir).write_text(
        recommended_command + "\n",
        encoding="utf-8",
    )

    best_config = {
        "schema_version": mechanism_stage_utils.SCHEMA_VERSION,
        "job_id": ctx.job_spec["job_id"],
        "best_rank": 1,
        "best_candidate": candidate.to_dict(),
        "fixed_params": ctx.fixed_params,
        "defaults": ctx.defaults,
        "verify_metrics": {
            "val_acc": {
                "mean": float(winner["val_acc_mean"]),
                "std": float(winner["val_acc_std"]),
                "min": float(winner["val_acc_min"]),
                "max": float(winner["val_acc_max"]),
                "n": int(winner["n"]),
            },
            "test_acc": {
                "mean": float(winner["test_acc_mean"]),
                "std": float(winner["test_acc_std"]),
                "min": float(winner["test_acc_min"]),
                "max": float(winner["test_acc_max"]),
                "n": int(winner["n"]),
            },
        },
        "artifacts": {
            "verify_summary_csv": str(mechanism_stage_utils.verify_summary_path(job_dir)),
            "recommended_command_txt": str(mechanism_stage_utils.recommended_command_path(job_dir)),
        },
        "recommended_command": recommended_command,
    }
    mechanism_stage_utils.write_yaml_file(
        mechanism_stage_utils.best_config_path(job_dir),
        best_config,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
