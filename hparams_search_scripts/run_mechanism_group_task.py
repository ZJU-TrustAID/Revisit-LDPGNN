#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

try:
    from datasets import load_dataset
    from hparams_search_scripts.mechanism_stage_context import build_stage_command, resolve_job_context
    from hparams_search_scripts import mechanism_stage_utils
    from main import build_parser, finalize_parsed_args, repeat_seed, seed_everything, to_scalar_metrics
    from models import HOA, KProp, NodeClassifier
    from scripts.pre_smoothing_feature_cache import _capture_rng_state, _restore_rng_state, prepare_pre_smoothing_input
    from trainer import Trainer
    from utils import from_args
except ModuleNotFoundError:
    from datasets import load_dataset  # type: ignore
    from mechanism_stage_context import build_stage_command, resolve_job_context  # type: ignore
    import mechanism_stage_utils  # type: ignore
    from main import build_parser, finalize_parsed_args, repeat_seed, seed_everything, to_scalar_metrics  # type: ignore
    from models import HOA, KProp, NodeClassifier  # type: ignore
    from scripts.pre_smoothing_feature_cache import _capture_rng_state, _restore_rng_state, prepare_pre_smoothing_input  # type: ignore
    from trainer import Trainer  # type: ignore
    from utils import from_args  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a grouped state task for a YAML-driven search job.')
    parser.add_argument('job_dir', type=str, help='job directory containing job_spec.yaml')
    parser.add_argument('--stage', choices=['grid', 'verify'], required=True, help='search stage to execute')
    parser.add_argument('--candidate_id', type=int, dest='candidate_ids', action='append', required=True,
                        help='candidate id to execute inside this grouped state task (repeat for multiple candidates)')
    parser.add_argument('--repeat_id', type=int, default=None,
                        help='verify repeat id; required when --stage verify is selected')
    return parser.parse_args()


def _stage_seed(ctx, stage_name: str, repeat_id: int | None) -> int:
    if stage_name == 'grid':
        return int(ctx.base_seed)
    if repeat_id is None:
        raise ValueError('repeat_id is required for verify stage')
    return int(ctx.base_seed) + int(repeat_id) - 1


def _stage_defaults(ctx, stage_name: str) -> tuple[int, int, int]:
    defaults = ctx.defaults['stage'][stage_name]
    repeats = 1 if stage_name == 'verify' else int(defaults['repeats'])
    return int(defaults['max_epochs']), int(defaults['patience']), repeats


def _candidate_state_key(candidate: mechanism_stage_utils.CandidateSpec) -> tuple[int, str]:
    return int(candidate.x_steps), mechanism_stage_utils.canonical_optional_float_text(candidate.tao2)


def _validate_grouped_state(candidates: list[mechanism_stage_utils.CandidateSpec]) -> tuple[int, str]:
    if len(candidates) == 0:
        raise ValueError('candidate_ids must be non-empty')
    reference = _candidate_state_key(candidates[0])
    for candidate in candidates[1:]:
        if _candidate_state_key(candidate) != reference:
            raise ValueError('all grouped candidates must share the same x_steps and tao2 state')
    return reference


def _build_stage_namespace(
    parser,
    ctx,
    candidate: mechanism_stage_utils.CandidateSpec,
    *,
    stage_name: str,
    output_dir: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    repeats: int,
):
    command = build_stage_command(
        ctx,
        candidate,
        stage_name=stage_name,
        output_dir=output_dir,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        repeats=repeats,
    )
    args = parser.parse_args(command[2:])
    finalize_parsed_args(parser, args)
    args.cmd = mechanism_stage_utils.shell_join(command)
    return args


def _build_group_smoother(name: str, x_steps: int):
    smoother_name = str(name).strip().lower()
    if smoother_name == 'hoa':
        smoother_cls = HOA
    elif smoother_name == 'kprop':
        smoother_cls = KProp
    else:
        raise ValueError(f'unsupported smoother {name!r} for grouped state execution')
    return smoother_cls(
        steps=int(x_steps),
        aggregator='add',
        add_self_loops=False,
        normalize=False,
        cached=False,
    )


@torch.no_grad()
def _compute_smoothed_x(prepared_data, *, smoother_name: str, x_steps: int):
    if int(x_steps) <= 0:
        return prepared_data.x
    normalized_adj_t = gcn_norm(prepared_data.adj_t, add_self_loops=False)
    smoother = _build_group_smoother(smoother_name, x_steps=int(x_steps))
    return smoother(prepared_data.x, normalized_adj_t)


def _run_single_candidate(args, data, *, run_id: str):
    input_dim = int(getattr(data, 'operator_num_features', data.num_features))
    model = from_args(
        NodeClassifier,
        args,
        input_dim=input_dim,
        num_classes=data.num_classes,
    )
    trainer = from_args(Trainer, args, logger=None)
    best_metrics = trainer.fit(model, data)
    return to_scalar_metrics(best_metrics)


def _write_single_result_csv(args, metrics: dict[str, float], *, run_id: str) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    row = dict(metrics)
    if 'test/acc' not in row and 'test/acc' in metrics:
        row['test/acc'] = metrics['test/acc']
    df = pd.DataFrame([row]).rename_axis('version').reset_index()
    df['Name'] = run_id
    for arg_name, arg_val in vars(args).items():
        if arg_name == 'input_already_smoothed':
            continue
        df[arg_name] = [arg_val]
    df.to_csv(os.path.join(args.output_dir, f'{run_id}.csv'), index=False)


def _candidate_output_dir(job_dir: Path, stage_name: str, ranked_lookup: dict[int, mechanism_stage_utils.RankedCandidate],
                          candidate: mechanism_stage_utils.CandidateSpec, repeat_id: int | None) -> Path:
    if stage_name == 'grid':
        return mechanism_stage_utils.grid_candidate_output_dir(job_dir, candidate)
    if repeat_id is None:
        raise ValueError('repeat_id is required for verify stage')
    ranked = ranked_lookup.get(int(candidate.candidate_id))
    if ranked is None:
        raise ValueError(f'candidate_id {candidate.candidate_id} is not present in verify_topk.csv')
    return mechanism_stage_utils.verify_candidate_output_dir(job_dir, ranked, int(repeat_id))


def main() -> None:
    cli_args = parse_args()
    ctx = resolve_job_context(cli_args.job_dir)
    job_dir = Path(cli_args.job_dir).resolve()

    if cli_args.stage == 'verify' and cli_args.repeat_id is None:
        raise ValueError('--repeat_id is required when --stage verify is selected')

    candidates = [
        mechanism_stage_utils.job_candidate_by_id(ctx.job_spec, int(candidate_id))
        for candidate_id in cli_args.candidate_ids
    ]
    candidates = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    x_steps, tao2 = _validate_grouped_state(candidates)

    ranked_lookup: dict[int, mechanism_stage_utils.RankedCandidate] = {}
    if cli_args.stage == 'verify':
        ranked_lookup = {
            ranked.candidate_id: ranked
            for ranked in mechanism_stage_utils.load_ranked_candidates(
                mechanism_stage_utils.verify_topk_path(job_dir)
            )
        }

    max_epochs, patience, repeats = _stage_defaults(ctx, cli_args.stage)
    if repeats != 1:
        raise ValueError(f'grouped stage runner expects repeats=1, got repeats={repeats}')
    seed = _stage_seed(ctx, cli_args.stage, cli_args.repeat_id)

    parser = build_parser()
    reference_output_dir = _candidate_output_dir(job_dir, cli_args.stage, ranked_lookup, candidates[0], cli_args.repeat_id)
    reference_args = _build_stage_namespace(
        parser,
        ctx,
        candidates[0],
        stage_name=cli_args.stage,
        output_dir=reference_output_dir,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        repeats=repeats,
    )
    if str(getattr(reference_args, 'feature', '')).strip().lower() == 'operator':
        raise ValueError('grouped state runner is only intended for non-operator figure3 feature paths')

    current_seed = repeat_seed(reference_args.seed, 0)
    if current_seed is not None:
        seed_everything(current_seed)

    dataset = from_args(load_dataset, reference_args)
    prepared_data = dataset.clone().to(reference_args.device)
    prepared_data, _ = prepare_pre_smoothing_input(prepared_data, reference_args, rewrite_seed=current_seed)
    post_prepare_rng_state = _capture_rng_state(getattr(prepared_data.x, 'device', None))
    smoothed_x = _compute_smoothed_x(prepared_data, smoother_name=str(reference_args.smoother), x_steps=int(x_steps))

    for candidate in candidates:
        output_dir = _candidate_output_dir(job_dir, cli_args.stage, ranked_lookup, candidate, cli_args.repeat_id)
        if mechanism_stage_utils.candidate_result_exists(output_dir, candidate, expected_seed=seed):
            continue

        candidate_args = _build_stage_namespace(
            parser,
            ctx,
            candidate,
            stage_name=cli_args.stage,
            output_dir=output_dir,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            repeats=repeats,
        )
        candidate_args.input_already_smoothed = True

        _restore_rng_state(post_prepare_rng_state, getattr(prepared_data.x, 'device', None))
        run_id = str(uuid.uuid1())
        data = prepared_data.clone()
        data.x = smoothed_x
        metrics = _run_single_candidate(candidate_args, data, run_id=run_id)
        _write_single_result_csv(candidate_args, metrics, run_id=run_id)

        if not mechanism_stage_utils.candidate_result_exists(output_dir, candidate, expected_seed=seed):
            raise RuntimeError(
                f'Grouped {cli_args.stage} task finished but produced no matching CSV under {output_dir} '
                f'for candidate_id={candidate.candidate_id}, x_steps={x_steps}, tao2={tao2}'
            )


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
