import os
import sys
import traceback
import uuid
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import random
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
from models import NodeClassifier
from scripts.pre_smoothing_feature_cache import prepare_pre_smoothing_input
from trainer import Trainer
from transforms import FeatureTransform, FeaturePerturbation, NFR
from utils import print_args, WandbLogger, add_parameters_as_argument, \
    measure_runtime, from_args, str2bool, Enum, EnumAction, bootstrap


class LogMode(Enum):
    INDIVIDUAL = 'individual'
    COLLECTIVE = 'collective'


def validate_gradient_clip_args(parser, args):
    if args.gradient_clip_max_norm <= 0:
        parser.error('--gradient-clip-max-norm must be > 0')

    if not args.gradient_clip and args.gradient_clip_max_norm != 1.0:
        parser.error('--gradient-clip-max-norm can only be used with --gradient-clip')


def validate_sim_args(parser, args):
    feature = str(getattr(args, 'feature', '')).strip().lower()
    sim_epoch_refresh = bool(getattr(args, 'sim_epoch_refresh', False))
    sim_reference_eps = getattr(args, 'sim_reference_eps', None)

    if sim_epoch_refresh:
        if feature != 'sim':
            parser.error('--sim-epoch-refresh requires --feature sim')

    if feature != 'sim':
        if sim_reference_eps is not None:
            parser.error('--sim-reference-eps requires --feature sim')
        return

    if sim_reference_eps is not None:
        if not np.isfinite(sim_reference_eps) or sim_reference_eps <= 0:
            parser.error('--sim-reference-eps must be finite and > 0')

    if sim_reference_eps is None:
        parser.error('--feature sim requires --sim-reference-eps to be provided')

    mechanism = str(getattr(args, 'mechanism', '')).strip().lower()
    supported_sim_mechanisms = set(FeatureTransform.supported_sim_mechanisms)
    if mechanism not in supported_sim_mechanisms:
        parser.error(
            '--feature sim currently supports --mechanism in '
            f'{sorted(supported_sim_mechanisms)}, got {mechanism or "<missing>"}'
        )


def _cli_flag(arg_name):
    return f'--{arg_name.replace("_", "-")}'


def validate_feature_rewrite_args(parser, args):
    feature = str(args.feature).strip().lower()
    rewrite_features = set(FeatureTransform.rewrite_supported_features)
    rewrite_arg_names = FeatureTransform.all_rewrite_arg_names

    provided_rewrite_args = [
        arg_name for arg_name in rewrite_arg_names
        if getattr(args, arg_name, None) is not None
    ]

    if feature not in rewrite_features:
        if provided_rewrite_args:
            parser.error(
                f'{_cli_flag(provided_rewrite_args[0])} is only available for artificial rewrite features, '
                f'got --feature {feature}'
            )
    elif feature != 'operator':
        if args.feature_dim is None:
            parser.error(f'--feature-dim is required when --feature {feature} is selected')
        if int(args.feature_dim) <= 0:
            parser.error('--feature-dim must be > 0')

    if feature in rewrite_features:
        allowed_args = set(FeatureTransform.rewrite_common_arg_names) | set(
            FeatureTransform.rewrite_feature_arg_names[feature]
        )
        disallowed = [
            arg_name for arg_name in provided_rewrite_args
            if arg_name not in allowed_args
        ]
        if disallowed:
            parser.error(
                f'{_cli_flag(disallowed[0])} is not valid when --feature {feature} is selected'
            )

    if args.scale is not None:
        if not np.isfinite(args.scale) or args.scale <= 0:
            parser.error('--scale must be finite and > 0')
        args.scale = float(args.scale)
    else:
        args.scale = 1.0

    if args.feature_preprojection:
        if feature != 'operator':
            parser.error('--feature-preprojection is only available when --feature operator is selected')
        if args.preprojection_output_dim is None:
            parser.error('--preprojection-output-dim is required when --feature-preprojection is enabled')
    elif args.preprojection_output_dim is not None:
        parser.error('--preprojection-output-dim requires --feature-preprojection')

    if feature == 'operator':
        if args.feature_dim is not None:
            parser.error('--feature-dim is not valid when --feature operator is selected')
        if args.preprojection_output_dim is not None and int(args.preprojection_output_dim) <= 0:
            parser.error('--preprojection-output-dim must be > 0')
    else:
        if args.preprojection_output_dim is not None and int(args.preprojection_output_dim) <= 0:
            parser.error('--preprojection-output-dim must be > 0')

    if feature == 'random_normal' and args.random_normal_std is not None and args.random_normal_std <= 0:
        parser.error('--random-normal-std must be > 0')

    if feature in {'degree_bucket_range', 'degree_bucket_distribution'}:
        if args.degree_bucket_num_buckets is not None and args.degree_bucket_num_buckets <= 0:
            parser.error('--degree-bucket-num-buckets must be > 0')

    if feature == 'degree_bucket_range':
        if args.degree_bucket_range_max is not None and args.degree_bucket_range_max <= 0:
            parser.error('--degree-bucket-range-max must be > 0')

    if feature == 'deepwalk':
        for arg_name in (
            'deepwalk_walk_length',
            'deepwalk_number_walks',
            'deepwalk_window_size',
            'deepwalk_workers',
        ):
            value = getattr(args, arg_name)
            if value is not None and value <= 0:
                parser.error(f'{_cli_flag(arg_name)} must be > 0')


def configure_determinism():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def confidence_interval(data, func=np.mean, size=1000, ci=95, seed=12345):
    bs_replicates = bootstrap(data, func=func, n_boot=size, seed=seed)
    p = 50 - ci / 2, 50 + ci / 2
    bounds = np.nanpercentile(bs_replicates, p)
    return (bounds[1] - bounds[0]) / 2


def preprocess_data(data, args, rewrite_seed=None):
    feature_transform = from_args(FeatureTransform, args).set_rewrite_seed(rewrite_seed)
    feature_perturbation = from_args(FeaturePerturbation, args)
    from torch_geometric.transforms import Compose

    return Compose([
        feature_transform,
        feature_perturbation,
    ])(data)


def apply_nfr_if_enabled(data, args):
    if not bool(getattr(args, 'use_nfr', False)):
        return data

    output_range = getattr(data, 'output_range', None)
    if output_range is None:
        raise ValueError('NFR requires data.output_range to be available, but it is None in the current configuration.')

    nfr = NFR(B=output_range, tao2=getattr(args, 'tao2', None))
    return nfr(data)


def build_sim_epoch_refresh_callback(args):
    if not args.sim_epoch_refresh:
        return None

    feature_transform = from_args(FeatureTransform, args)

    def refresh_callback(epoch, data, eval_data=None):
        _ = epoch
        feature_transform.refresh_sim_features(data)
        data = apply_nfr_if_enabled(data, args)
        if eval_data is not None and eval_data is not data:
            eval_data.x = data.x.clone()
            eval_data.output_range = data.output_range

    return refresh_callback


def to_scalar(value):
    if torch.is_tensor(value):
        return value.item()
    return float(value)


def to_scalar_metrics(metrics):
    return {metric: to_scalar(value) for metric, value in metrics.items()}


def repeat_seed(seed, repeat_id):
    if seed is None:
        return None
    return seed + repeat_id


def run_single_repeat(args, repeat_id, run_id, logger=None):
    current_seed = repeat_seed(args.seed, repeat_id)
    if current_seed is not None:
        seed_everything(current_seed)

    dataset = from_args(load_dataset, args)     # Load the dataset.
    data = dataset.clone().to(args.device)      # Move training data to the selected device.
    data, _ = prepare_pre_smoothing_input(data, args, rewrite_seed=current_seed)
    input_dim = int(getattr(data, 'operator_num_features', data.num_features))

    model = from_args(
        NodeClassifier,
        args,
        input_dim=input_dim,
        num_classes=data.num_classes,
    )
    trainer = from_args(Trainer, args, logger=logger if args.log_mode == LogMode.INDIVIDUAL else None)
    sim_epoch_refresh_fn = build_sim_epoch_refresh_callback(args)
    best_metrics = trainer.fit(
        model,
        data,
        epoch_end_data_refresh_fn=sim_epoch_refresh_fn,
    )
    metrics = to_scalar_metrics(best_metrics)
    return metrics, metrics['test/acc']


@measure_runtime
def run(args):
    if args.seed is not None:
        seed_everything(args.seed)

    test_acc = []
    run_metrics = {}
    run_id = str(uuid.uuid1())

    logger = None
    if args.log and args.log_mode == LogMode.COLLECTIVE:
        logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, reinit=False, group=run_id)

    repeat_iterator = range(args.repeats)
    progbar = None
    if args.show_progress:
        progbar = tqdm(repeat_iterator, file=sys.stdout)
        repeat_iterator = progbar

    for version in repeat_iterator:
        repeat_logger = None
        if args.log and args.log_mode == LogMode.INDIVIDUAL:
            args.version = version
            repeat_logger = WandbLogger(project=args.project_name, config=args, enabled=args.log, group=run_id)

        try:
            metrics, current_test_acc = run_single_repeat(
                args=args,
                repeat_id=version,
                run_id=run_id,
                logger=repeat_logger,
            )

            for metric, value in metrics.items():
                run_metrics[metric] = run_metrics.get(metric, []) + [to_scalar(value)]

            test_acc.append(to_scalar(current_test_acc))
            if progbar is not None:
                progbar.set_postfix({'last_test_acc': current_test_acc, 'avg_test_acc': np.mean(test_acc)})

        except Exception as e:
            error = ''.join(traceback.format_exception(Exception, e, e.__traceback__))
            if repeat_logger is not None:
                repeat_logger.log_summary({'error': error})
            raise e
        finally:
            if repeat_logger is not None:
                repeat_logger.finish()

    if args.log and args.log_mode == LogMode.COLLECTIVE:
        ci_seed = args.seed if args.seed is not None else 12345
        summary = {}
        for metric, values in run_metrics.items():
            summary[metric + '_mean'] = np.mean(values)
            summary[metric + '_ci'] = confidence_interval(values, size=1000, ci=95, seed=ci_seed)
        logger.log_summary(summary)

    if not args.log:
        os.makedirs(args.output_dir, exist_ok=True)
        df_results = pd.DataFrame(run_metrics).rename_axis('version').reset_index()
        if 'test/acc' not in df_results.columns:
            df_results['test/acc'] = test_acc
        df_results['Name'] = run_id
        for arg_name, arg_val in vars(args).items():
            df_results[arg_name] = [arg_val] * len(test_acc)
        df_results.to_csv(os.path.join(args.output_dir, f'{run_id}.csv'), index=False)


def build_parser():
    init_parser = ArgumentParser(add_help=False, conflict_handler='resolve')

    feature_transform_args = (
        'feature',
        'sim_reference_eps',
        'feature_dim',
        'scale',
        'feature_preprojection',
        'preprojection_output_dim',
        'random_normal_mean',
        'random_normal_std',
        'shared_value',
        'degree_bucket_num_buckets',
        'degree_bucket_range_max',
        'deepwalk_walk_length',
        'deepwalk_number_walks',
        'deepwalk_window_size',
        'deepwalk_workers',
        'deepwalk_undirected',
    )
    feature_perturbation_args = ('mechanism', 'x_eps', 'm')
    calibrator_perturbation_args = ('norm', 'norm_scale')
    calibrator_model_args = ('x_steps', 'smoother')
    model_args = ('model', 'hidden_dim', 'dropout')

    group_dataset = init_parser.add_argument_group('dataset arguments')
    add_parameters_as_argument(load_dataset, group_dataset)
    group_dataset.add_argument(
        '--inf_eps_unit_map', '--inf-eps-unit-map',
        dest='inf_eps_unit_map',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help=(
            'when x_eps=inf for four-stage mechanisms, deterministically map '
            'features to the mechanism unit domain [-1,1] for strict '
            'finite-vs-inf comparisons'
        ),
    )

    group_feature_transform = init_parser.add_argument_group('feature transformation arguments')
    add_parameters_as_argument(FeatureTransform, group_feature_transform, include=feature_transform_args)

    group_feature_perturbation = init_parser.add_argument_group('feature perturbation arguments')
    add_parameters_as_argument(FeaturePerturbation, group_feature_perturbation, include=feature_perturbation_args)

    group_calibrator = init_parser.add_argument_group('calibrator arguments')
    add_parameters_as_argument(FeaturePerturbation, group_calibrator, include=calibrator_perturbation_args)
    add_parameters_as_argument(NodeClassifier, group_calibrator, include=calibrator_model_args)

    group_model = init_parser.add_argument_group('model arguments')
    add_parameters_as_argument(NodeClassifier, group_model, include=model_args)

    group_nfr = init_parser.add_argument_group('nfr arguments')
    group_nfr.add_argument(
        '--use_nfr', '--use-nfr',
        action='store_true',
        default=False,
        help='apply NFR soft-threshold feature rewriting after preprocessing',
    )
    group_nfr.add_argument(
        '--tao2',
        type=float,
        default=None,
        help='NFR soft-threshold hyperparameter',
    )

    group_trainer = init_parser.add_argument_group('trainer arguments')
    add_parameters_as_argument(Trainer, group_trainer)
    group_trainer.add_argument('--device', help='desired device for training', choices=['cpu', 'cuda'], default='cuda')

    group_expr = init_parser.add_argument_group('experiment arguments')
    group_expr.add_argument('-s', '--seed', type=int, default=None, help='initial random seed')
    group_expr.add_argument('-r', '--repeats', type=int, default=1, help='number of times the experiment is repeated')
    group_expr.add_argument('-o', '--output-dir', type=str, default='./output', help='directory to store the results')
    group_expr.add_argument('--log', type=str2bool, nargs='?', const=True, default=False, help='enable wandb logging')
    group_expr.add_argument('--log-mode', type=LogMode, action=EnumAction, default=LogMode.INDIVIDUAL,
                            help='wandb logging mode')
    group_expr.add_argument('--project-name', type=str, default='LPGNN', help='wandb project name')
    group_expr.add_argument(
        '--pre_smoothing_feature_cache_root', '--pre-smoothing-feature-cache-root',
        dest='pre_smoothing_feature_cache_root',
        type=str,
        default=None,
        help='optional cache root for raw->perturbation->optional NFR tensors loaded before HOA/KProp',
    )

    return ArgumentParser(parents=[init_parser], formatter_class=ArgumentDefaultsHelpFormatter)


def finalize_parsed_args(parser, args):
    configure_determinism()

    validate_gradient_clip_args(parser, args)
    validate_sim_args(parser, args)
    validate_feature_rewrite_args(parser, args)
    if args.pre_smoothing_feature_cache_root is not None and str(args.pre_smoothing_feature_cache_root).strip() == '':
        args.pre_smoothing_feature_cache_root = None
    if args.use_nfr and args.tao2 is None:
        parser.error('--tao2 must be provided when --use_nfr is enabled')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is required but not available in the current environment')

    return args


def parse_cli_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    finalize_parsed_args(parser, args)
    return parser, args


def main():
    parser, args = parse_cli_args()

    print_args(args)
    args.cmd = ' '.join(sys.argv)  # store calling command

    if args.seed is not None:
        seed_everything(args.seed)

    try:
        run(args=args)
    except KeyboardInterrupt:
        print('Graceful Shutdown...')


if __name__ == '__main__':
    main()
