#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hparams_search_scripts import mechanism_stage_utils
from hparams_search_scripts import table_search_suite_core as core

from datasets import list_supported_datasets, resolve_dataset_name
from mechanisms import supported_feature_mechanisms
from transforms import FeatureTransform
from hparams_search_scripts.gpu_resources import parse_gpu_ids, resolve_gpu_concurrency


TOP_LEVEL_KEYS = {"device", "defaults", "search_space", "seed"}
DEVICE_KEYS = {
    "device",
    "cpu_worker_count",
    "gpu_ids",
    "max_parallel_per_gpu",
    "gpu_launch_interval_sec",
}
DEFAULTS_KEYS = {"dataset", "model", "trainer", "stage"}
DEFAULT_DATASET_KEYS = {"data_range", "val_ratio", "test_ratio"}
DEFAULT_MODEL_KEYS = {"hidden_dim"}
DEFAULT_TRAINER_KEYS = {
    "optimizer",
    "gradient_clip",
    "gradient_clip_max_norm",
    "sim_epoch_refresh",
    "show_progress",
    "log_every_epoch",
}
DEFAULT_STAGE_KEYS = {"grid", "verify"}
DEFAULT_STAGE_INNER_KEYS = {"patience", "repeats"}
SEARCH_SPACE_KEYS = {
    "dataset",
    "feature_transformation",
    "feature_perturbation",
    "calibrator",
    "model",
    "trainer",
    "nfr",
}
SEARCH_DATASET_KEYS = {"datasets"}
SEARCH_FEATURE_KEYS = {
    "features",
    "sim_reference_eps",
    "feature_dim",
    "scale",
    "feature_preprojection",
    "preprojection_output_dim",
    "random_normal_mean",
    "random_normal_std",
    "shared_value",
    "degree_bucket_num_buckets",
    "degree_bucket_range_max",
    "deepwalk_walk_length",
    "deepwalk_number_walks",
    "deepwalk_window_size",
    "deepwalk_workers",
    "deepwalk_undirected",
}
SEARCH_PERTURBATION_KEYS = {"mechanisms", "x_eps", "m"}
SEARCH_CALIBRATOR_KEYS = {"norm", "norm_scale", "x_steps", "smoother"}
SEARCH_CALIBRATOR_REQUIRED_KEYS = {"x_steps"}
SEARCH_MODEL_KEYS = {"backbones", "dropout"}
SEARCH_TRAINER_KEYS = {"learning_rate", "weight_decay"}
SEARCH_NFR_KEYS = {"use_nfr", "tao2"}

SUPPORTED_FEATURES = tuple(FeatureTransform.supported_features)
SUPPORTED_REWRITE_FEATURES = tuple(FeatureTransform.rewrite_supported_features)
SUPPORTED_REWRITE_FEATURE_SET = set(SUPPORTED_REWRITE_FEATURES)
SUPPORTED_SIM_MECHANISMS = set(FeatureTransform.supported_sim_mechanisms)
SUPPORTED_MECHANISMS = set(supported_feature_mechanisms)
SUPPORTED_OPTIMIZERS = {"sgd", "adam"}
SUPPORTED_BACKBONES = {"gcn", "sage", "gat"}
SUPPORTED_SMOOTHERS = {"kprop", "hoa"}
DEFAULT_REWRITE_PERTURBATION = {
    "mechanisms": ["mbm"],
    "x_eps": ["inf"],
    "m": ["best"],
}
DEFAULT_SIM_PERTURBATION = {
    "x_eps": ["inf"],
}
DEFAULT_REWRITE_CALIBRATOR = {
    "norm": [False],
    "norm_scale": [],
}
DEFAULT_ZERO_STEP_SMOOTHER = ["kprop"]
class SearchError(RuntimeError):
    pass


def _supported_dataset_names() -> list[str]:
    return list_supported_datasets()


def _expect_mapping(raw: Any, path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise SearchError(f"{path} must be a mapping")
    return raw


def _expect_exact_keys(mapping: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(mapping.keys())
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise SearchError(f"{path} has unknown fields: {unknown}")
    if missing:
        raise SearchError(f"{path} is missing required fields: {missing}")


def _expect_allowed_keys(
    mapping: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str],
    path: str,
) -> None:
    actual = set(mapping.keys())
    unknown = sorted(actual - allowed)
    missing = sorted(required - actual)
    if unknown:
        raise SearchError(f"{path} has unknown fields: {unknown}")
    if missing:
        raise SearchError(f"{path} is missing required fields: {missing}")


def _parse_string(raw: Any, path: str) -> str:
    if not isinstance(raw, str):
        raise SearchError(f"{path} must be a string")
    value = raw.strip()
    if value == "":
        raise SearchError(f"{path} must be non-empty")
    return value


def _parse_bool(raw: Any, path: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    raise SearchError(f"{path} must be a boolean")


def _parse_positive_int(raw: Any, path: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SearchError(f"{path} must be a positive integer") from exc
    if value < 1:
        raise SearchError(f"{path} must be a positive integer")
    return value


def _parse_nonnegative_int(raw: Any, path: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise SearchError(f"{path} must be a non-negative integer") from exc
    if value < 0:
        raise SearchError(f"{path} must be a non-negative integer")
    return value


def _parse_positive_float(raw: Any, path: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SearchError(f"{path} must be a positive finite number") from exc
    if not math.isfinite(value) or value <= 0:
        raise SearchError(f"{path} must be a positive finite number")
    return value


def _parse_nonnegative_float(raw: Any, path: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SearchError(f"{path} must be a non-negative finite number") from exc
    if not math.isfinite(value) or value < 0:
        raise SearchError(f"{path} must be a non-negative finite number")
    return value


def _parse_probability(raw: Any, path: str) -> float:
    value = _parse_nonnegative_float(raw, path)
    if value > 1:
        raise SearchError(f"{path} must be between 0 and 1")
    return value


def _parse_positive_probability(raw: Any, path: str) -> float:
    value = _parse_positive_float(raw, path)
    if value > 1:
        raise SearchError(f"{path} must be between 0 and 1")
    return value


def _parse_float_text(raw: Any, path: str) -> str:
    return mechanism_stage_utils.canonical_float_text(_parse_nonnegative_float(raw, path))


def _parse_positive_float_text(raw: Any, path: str) -> str:
    return mechanism_stage_utils.canonical_float_text(_parse_positive_float(raw, path))


def _parse_scale_text(raw: Any, path: str) -> str:
    return _parse_positive_float_text(raw, path)


def _parse_any_float_text(raw: Any, path: str) -> str:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise SearchError(f"{path} must be a finite number") from exc
    if not math.isfinite(value):
        raise SearchError(f"{path} must be a finite number")
    return mechanism_stage_utils.canonical_float_text(value)


def _parse_x_eps_text(raw: Any, path: str) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "inf":
        return "inf"
    value = _parse_positive_float(raw, path)
    return mechanism_stage_utils.canonical_float_text(value)


def _parse_norm_scale_text(raw: Any, path: str) -> str:
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.lower() == "none":
            return "none"
    return mechanism_stage_utils.canonical_float_text(_parse_positive_float(raw, path))


def _parse_m_text(raw: Any, path: str) -> str:
    if isinstance(raw, str):
        stripped = raw.strip().lower()
        if stripped in {"best", "max"}:
            return stripped
        raw = stripped
    value = _parse_positive_int(raw, path)
    return str(value)


def _parse_choice(raw: Any, path: str, *, choices: set[str]) -> str:
    value = _parse_string(raw, path)
    lowered = value.lower()
    if lowered not in choices:
        raise SearchError(f"{path} must be one of {sorted(choices)}, got {value!r}")
    return lowered


def _parse_dataset_name(raw: Any, path: str, *, supported_datasets: set[str]) -> str:
    value = _parse_string(raw, path)
    try:
        return resolve_dataset_name(value)
    except ValueError as exc:
        raise SearchError(
            f"{path}={value!r} is unsupported. Supported datasets include: {sorted(supported_datasets)}"
        ) from exc


def _parse_list(
    raw: Any,
    path: str,
    *,
    item_parser: Callable[[Any, str], Any],
    allow_empty: bool,
) -> list[Any]:
    if not isinstance(raw, list):
        raise SearchError(f"{path} must be a list")

    values: list[Any] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        parsed = item_parser(item, f"{path}[{index}]")
        marker = mechanism_stage_utils.canonical_search_value(parsed)
        if marker in seen:
            raise SearchError(f"{path} contains duplicate value {parsed!r}")
        seen.add(marker)
        values.append(parsed)

    if not allow_empty and len(values) == 0:
        raise SearchError(f"{path} must be non-empty")
    return values


def _validate_device_section(raw: Any) -> mechanism_stage_utils.ExecutionSettings:
    mapping = _expect_mapping(raw, "device")
    _expect_exact_keys(mapping, DEVICE_KEYS, "device")

    device = _parse_choice(mapping["device"], "device.device", choices={"cpu", "gpu"})
    if device == "cpu":
        cpu_worker_count = _parse_positive_int(mapping["cpu_worker_count"], "device.cpu_worker_count")
        for key in ("gpu_ids", "max_parallel_per_gpu", "gpu_launch_interval_sec"):
            if mapping[key] not in (None, [], ""):
                raise SearchError(f"device.{key} must not be set when device.device=cpu")
        return mechanism_stage_utils.ExecutionSettings(
            device="cpu",
            worker_ids=list(range(cpu_worker_count)),
            max_parallel_per_worker=1,
            launch_interval_sec=0.0,
        )

    if mapping["cpu_worker_count"] not in (None, ""):
        raise SearchError("device.cpu_worker_count must not be set when device.device=gpu")
    gpu_ids = parse_gpu_ids(mapping["gpu_ids"])
    max_parallel_per_gpu = resolve_gpu_concurrency(gpu_ids, mapping["max_parallel_per_gpu"])
    launch_interval_sec = _parse_nonnegative_float(
        mapping["gpu_launch_interval_sec"],
        "device.gpu_launch_interval_sec",
    )
    return mechanism_stage_utils.ExecutionSettings(
        device="gpu",
        worker_ids=gpu_ids,
        max_parallel_per_worker=max_parallel_per_gpu,
        launch_interval_sec=launch_interval_sec,
    )


def _validate_defaults_section(raw: Any) -> dict[str, Any]:
    mapping = _expect_mapping(raw, "defaults")
    _expect_exact_keys(mapping, DEFAULTS_KEYS, "defaults")

    dataset = _expect_mapping(mapping["dataset"], "defaults.dataset")
    _expect_exact_keys(dataset, DEFAULT_DATASET_KEYS, "defaults.dataset")
    data_range_raw = dataset["data_range"]
    if not isinstance(data_range_raw, list) or len(data_range_raw) != 2:
        raise SearchError("defaults.dataset.data_range must be a list with exactly 2 numbers")
    data_range = [
        _parse_any_float_text(data_range_raw[0], "defaults.dataset.data_range[0]"),
        _parse_any_float_text(data_range_raw[1], "defaults.dataset.data_range[1]"),
    ]
    if float(data_range[0]) >= float(data_range[1]):
        raise SearchError("defaults.dataset.data_range must have min < max")

    model = _expect_mapping(mapping["model"], "defaults.model")
    _expect_exact_keys(model, DEFAULT_MODEL_KEYS, "defaults.model")

    trainer = _expect_mapping(mapping["trainer"], "defaults.trainer")
    _expect_exact_keys(trainer, DEFAULT_TRAINER_KEYS, "defaults.trainer")

    stage = _expect_mapping(mapping["stage"], "defaults.stage")
    _expect_exact_keys(stage, DEFAULT_STAGE_KEYS, "defaults.stage")
    grid = _expect_mapping(stage["grid"], "defaults.stage.grid")
    verify = _expect_mapping(stage["verify"], "defaults.stage.verify")
    _expect_exact_keys(grid, DEFAULT_STAGE_INNER_KEYS, "defaults.stage.grid")
    _expect_exact_keys(verify, DEFAULT_STAGE_INNER_KEYS, "defaults.stage.verify")

    grid_repeats = _parse_positive_int(grid["repeats"], "defaults.stage.grid.repeats")
    verify_repeats = _parse_positive_int(verify["repeats"], "defaults.stage.verify.repeats")
    if grid_repeats != 1:
        raise SearchError("defaults.stage.grid.repeats must be exactly 1")

    return {
        "dataset": {
            "data_range": [float(data_range[0]), float(data_range[1])],
            "val_ratio": _parse_probability(dataset["val_ratio"], "defaults.dataset.val_ratio"),
            "test_ratio": _parse_probability(dataset["test_ratio"], "defaults.dataset.test_ratio"),
        },
        "model": {
            "hidden_dim": _parse_positive_int(model["hidden_dim"], "defaults.model.hidden_dim"),
        },
        "trainer": {
            "optimizer": _parse_choice(
                trainer["optimizer"],
                "defaults.trainer.optimizer",
                choices=SUPPORTED_OPTIMIZERS,
            ),
            "gradient_clip": _parse_bool(
                trainer["gradient_clip"],
                "defaults.trainer.gradient_clip",
            ),
            "gradient_clip_max_norm": _parse_positive_float(
                trainer["gradient_clip_max_norm"],
                "defaults.trainer.gradient_clip_max_norm",
            ),
            "sim_epoch_refresh": _parse_bool(
                trainer["sim_epoch_refresh"],
                "defaults.trainer.sim_epoch_refresh",
            ),
            "show_progress": _parse_bool(
                trainer["show_progress"],
                "defaults.trainer.show_progress",
            ),
            "log_every_epoch": _parse_bool(
                trainer["log_every_epoch"],
                "defaults.trainer.log_every_epoch",
            ),
        },
        "stage": {
            "grid": {
                "patience": _parse_positive_int(grid["patience"], "defaults.stage.grid.patience"),
                "repeats": grid_repeats,
                "max_epochs": mechanism_stage_utils.GRID_MAX_EPOCHS,
            },
            "verify": {
                "patience": _parse_positive_int(verify["patience"], "defaults.stage.verify.patience"),
                "repeats": verify_repeats,
                "max_epochs": mechanism_stage_utils.VERIFY_MAX_EPOCHS,
            },
        },
    }


def _validate_search_space_section(raw: Any) -> dict[str, Any]:
    supported_datasets = set(_supported_dataset_names())
    mapping = dict(_expect_mapping(raw, "search_space"))
    _expect_exact_keys(mapping, SEARCH_SPACE_KEYS, "search_space")

    dataset = _expect_mapping(mapping["dataset"], "search_space.dataset")
    _expect_exact_keys(dataset, SEARCH_DATASET_KEYS, "search_space.dataset")

    feature = _expect_mapping(mapping["feature_transformation"], "search_space.feature_transformation")
    _expect_allowed_keys(
        feature,
        allowed=SEARCH_FEATURE_KEYS,
        required=SEARCH_FEATURE_KEYS - {"scale", "feature_preprojection", "preprojection_output_dim"},
        path="search_space.feature_transformation",
    )

    calibrator = _expect_mapping(mapping["calibrator"], "search_space.calibrator")
    _expect_allowed_keys(
        calibrator,
        allowed=SEARCH_CALIBRATOR_KEYS,
        required=SEARCH_CALIBRATOR_REQUIRED_KEYS,
        path="search_space.calibrator",
    )

    model = _expect_mapping(mapping["model"], "search_space.model")
    _expect_exact_keys(model, SEARCH_MODEL_KEYS, "search_space.model")

    trainer = _expect_mapping(mapping["trainer"], "search_space.trainer")
    _expect_exact_keys(trainer, SEARCH_TRAINER_KEYS, "search_space.trainer")

    nfr = _expect_mapping(mapping["nfr"], "search_space.nfr")
    _expect_exact_keys(nfr, SEARCH_NFR_KEYS, "search_space.nfr")

    normalized_feature = {
        "dataset": {
            "datasets": _parse_list(
                dataset["datasets"],
                "search_space.dataset.datasets",
                item_parser=lambda value, path: _parse_dataset_name(
                    value,
                    path,
                    supported_datasets=supported_datasets,
                ),
                allow_empty=False,
            ),
        },
        "feature_transformation": {
            "features": _parse_list(
                feature["features"],
                "search_space.feature_transformation.features",
                item_parser=lambda value, path: _parse_choice(value, path, choices=set(SUPPORTED_FEATURES)),
                allow_empty=False,
            ),
            "sim_reference_eps": _parse_list(
                feature["sim_reference_eps"],
                "search_space.feature_transformation.sim_reference_eps",
                item_parser=_parse_positive_float_text,
                allow_empty=True,
            ),
            "feature_dim": _parse_list(
                feature["feature_dim"],
                "search_space.feature_transformation.feature_dim",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "scale": _parse_list(
                feature.get("scale", []),
                "search_space.feature_transformation.scale",
                item_parser=_parse_scale_text,
                allow_empty=True,
            ),
            "feature_preprojection": _parse_list(
                feature.get("feature_preprojection", []),
                "search_space.feature_transformation.feature_preprojection",
                item_parser=_parse_bool,
                allow_empty=True,
            ),
            "preprojection_output_dim": _parse_list(
                feature.get("preprojection_output_dim", []),
                "search_space.feature_transformation.preprojection_output_dim",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "random_normal_mean": _parse_list(
                feature["random_normal_mean"],
                "search_space.feature_transformation.random_normal_mean",
                item_parser=_parse_any_float_text,
                allow_empty=True,
            ),
            "random_normal_std": _parse_list(
                feature["random_normal_std"],
                "search_space.feature_transformation.random_normal_std",
                item_parser=_parse_positive_float_text,
                allow_empty=True,
            ),
            "shared_value": _parse_list(
                feature["shared_value"],
                "search_space.feature_transformation.shared_value",
                item_parser=_parse_any_float_text,
                allow_empty=True,
            ),
            "degree_bucket_num_buckets": _parse_list(
                feature["degree_bucket_num_buckets"],
                "search_space.feature_transformation.degree_bucket_num_buckets",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "degree_bucket_range_max": _parse_list(
                feature["degree_bucket_range_max"],
                "search_space.feature_transformation.degree_bucket_range_max",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "deepwalk_walk_length": _parse_list(
                feature["deepwalk_walk_length"],
                "search_space.feature_transformation.deepwalk_walk_length",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "deepwalk_number_walks": _parse_list(
                feature["deepwalk_number_walks"],
                "search_space.feature_transformation.deepwalk_number_walks",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "deepwalk_window_size": _parse_list(
                feature["deepwalk_window_size"],
                "search_space.feature_transformation.deepwalk_window_size",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "deepwalk_workers": _parse_list(
                feature["deepwalk_workers"],
                "search_space.feature_transformation.deepwalk_workers",
                item_parser=_parse_positive_int,
                allow_empty=True,
            ),
            "deepwalk_undirected": _parse_list(
                feature["deepwalk_undirected"],
                "search_space.feature_transformation.deepwalk_undirected",
                item_parser=_parse_bool,
                allow_empty=True,
            ),
            "_scale_was_specified": "scale" in feature,
        },
    }

    rewrite_only_requested = (
        len(normalized_feature["feature_transformation"]["features"]) > 0
        and set(normalized_feature["feature_transformation"]["features"]) <= SUPPORTED_REWRITE_FEATURE_SET
    )
    sim_only_requested = set(normalized_feature["feature_transformation"]["features"]) == {"sim"}
    operator_only_requested = set(normalized_feature["feature_transformation"]["features"]) == {"operator"}
    perturbation_raw = mapping["feature_perturbation"]
    if isinstance(perturbation_raw, list):
        if len(perturbation_raw) != 0:
            raise SearchError(
                "search_space.feature_perturbation may only use the [] shortcut when it is empty"
            )
        if not rewrite_only_requested:
            raise SearchError(
                "search_space.feature_perturbation may be [] only when all selected features bypass perturbation"
            )
        normalized_perturbation = {
            "mechanisms": [],
            "x_eps": [],
            "m": [],
        }
    else:
        perturbation = _expect_mapping(perturbation_raw, "search_space.feature_perturbation")
        required_perturbation_keys = SEARCH_PERTURBATION_KEYS - {"x_eps"} if sim_only_requested else SEARCH_PERTURBATION_KEYS
        _expect_allowed_keys(
            perturbation,
            allowed=SEARCH_PERTURBATION_KEYS,
            required=required_perturbation_keys,
            path="search_space.feature_perturbation",
        )
        normalized_perturbation = {
            "mechanisms": _parse_list(
                perturbation["mechanisms"],
                "search_space.feature_perturbation.mechanisms",
                item_parser=lambda value, path: _parse_choice(value, path, choices=SUPPORTED_MECHANISMS),
                allow_empty=True,
            ),
            "x_eps": (
                _parse_list(
                    perturbation["x_eps"],
                    "search_space.feature_perturbation.x_eps",
                    item_parser=_parse_x_eps_text,
                    allow_empty=True,
                )
                if "x_eps" in perturbation
                else []
            ),
            "m": _parse_list(
                perturbation["m"],
                "search_space.feature_perturbation.m",
                item_parser=_parse_m_text,
                allow_empty=True,
            ),
        }

    if "norm" in calibrator:
        normalized_norm = _parse_list(
            calibrator["norm"],
            "search_space.calibrator.norm",
            item_parser=_parse_bool,
            allow_empty=False,
        )
    elif rewrite_only_requested:
        normalized_norm = list(DEFAULT_REWRITE_CALIBRATOR["norm"])
    else:
        raise SearchError("search_space.calibrator is missing required fields: ['norm']")

    if "norm_scale" in calibrator:
        normalized_norm_scale = _parse_list(
            calibrator["norm_scale"],
            "search_space.calibrator.norm_scale",
            item_parser=_parse_norm_scale_text,
            allow_empty=True,
        )
    elif rewrite_only_requested:
        normalized_norm_scale = list(DEFAULT_REWRITE_CALIBRATOR["norm_scale"])
    else:
        raise SearchError("search_space.calibrator is missing required fields: ['norm_scale']")

    normalized_x_steps = _parse_list(
        calibrator["x_steps"],
        "search_space.calibrator.x_steps",
        item_parser=_parse_nonnegative_int,
        allow_empty=False,
    )

    if "smoother" in calibrator:
        normalized_smoother = _parse_list(
            calibrator["smoother"],
            "search_space.calibrator.smoother",
            item_parser=lambda value, path: _parse_choice(value, path, choices=SUPPORTED_SMOOTHERS),
            allow_empty=False,
        )
    elif all(step == 0 for step in normalized_x_steps) or operator_only_requested:
        normalized_smoother = list(DEFAULT_ZERO_STEP_SMOOTHER)
    else:
        raise SearchError("search_space.calibrator is missing required fields: ['smoother']")

    normalized = {
        **normalized_feature,
        "feature_perturbation": {
            **normalized_perturbation,
        },
        "calibrator": {
            "norm": normalized_norm,
            "norm_scale": normalized_norm_scale,
            "x_steps": normalized_x_steps,
            "smoother": normalized_smoother,
        },
        "model": {
            "backbones": _parse_list(
                model["backbones"],
                "search_space.model.backbones",
                item_parser=lambda value, path: _parse_choice(value, path, choices=SUPPORTED_BACKBONES),
                allow_empty=False,
            ),
            "dropout": _parse_list(
                model["dropout"],
                "search_space.model.dropout",
                item_parser=lambda value, path: mechanism_stage_utils.canonical_float_text(
                    _parse_probability(value, path)
                ),
                allow_empty=False,
            ),
        },
        "trainer": {
            "learning_rate": _parse_list(
                trainer["learning_rate"],
                "search_space.trainer.learning_rate",
                item_parser=_parse_positive_float_text,
                allow_empty=False,
            ),
            "weight_decay": _parse_list(
                trainer["weight_decay"],
                "search_space.trainer.weight_decay",
                item_parser=_parse_float_text,
                allow_empty=False,
            ),
        },
        "nfr": {
            "use_nfr": _parse_list(
                nfr["use_nfr"],
                "search_space.nfr.use_nfr",
                item_parser=_parse_bool,
                allow_empty=False,
            ),
            "tao2": _parse_list(
                nfr["tao2"],
                "search_space.nfr.tao2",
                item_parser=_parse_positive_float_text,
                allow_empty=True,
            ),
        },
    }
    _validate_cross_constraints(normalized)
    return normalized


def _require_empty(values: list[Any], path: str) -> None:
    if len(values) > 0:
        raise SearchError(f"{path} must be an empty list in the current configuration")


def _require_non_empty(values: list[Any], path: str) -> None:
    if len(values) == 0:
        raise SearchError(f"{path} must be non-empty in the current configuration")


def _validate_cross_constraints(search_space: dict[str, Any]) -> None:
    feature_cfg = search_space["feature_transformation"]
    perturb_cfg = search_space["feature_perturbation"]
    calibrator_cfg = search_space["calibrator"]
    nfr_cfg = search_space["nfr"]
    features = set(feature_cfg["features"])
    scale_values = feature_cfg["scale"]
    feature_preprojection_values = set(feature_cfg["feature_preprojection"])
    norm_values = set(calibrator_cfg["norm"])
    use_nfr_values = set(nfr_cfg["use_nfr"])
    rewrite_only = len(features) > 0 and features <= SUPPORTED_REWRITE_FEATURE_SET
    sim_only = features == {"sim"}
    rewrite_features = features & SUPPORTED_REWRITE_FEATURE_SET
    rewrite_features_requiring_dim = rewrite_features - {"operator"}

    if rewrite_only:
        for field_name, default_values in DEFAULT_REWRITE_PERTURBATION.items():
            if len(perturb_cfg[field_name]) == 0:
                perturb_cfg[field_name] = list(default_values)
    else:
        _require_non_empty(
            perturb_cfg["mechanisms"],
            "search_space.feature_perturbation.mechanisms",
        )
        if sim_only and len(perturb_cfg["x_eps"]) == 0:
            perturb_cfg["x_eps"] = list(DEFAULT_SIM_PERTURBATION["x_eps"])
        _require_non_empty(
            perturb_cfg["x_eps"],
            "search_space.feature_perturbation.x_eps",
        )
        _require_non_empty(
            perturb_cfg["m"],
            "search_space.feature_perturbation.m",
        )

    mechanisms = set(perturb_cfg["mechanisms"])
    x_eps_values = set(perturb_cfg["x_eps"])

    if "sim" in features:
        _require_non_empty(
            feature_cfg["sim_reference_eps"],
            "search_space.feature_transformation.sim_reference_eps",
        )
        invalid_mechanisms = sorted(mechanisms - SUPPORTED_SIM_MECHANISMS)
        if invalid_mechanisms:
            raise SearchError(
                "feature=sim only supports mechanisms "
                f"{sorted(SUPPORTED_SIM_MECHANISMS)}, got {invalid_mechanisms}"
            )
    else:
        _require_empty(
            feature_cfg["sim_reference_eps"],
            "search_space.feature_transformation.sim_reference_eps",
        )

    if len(scale_values) == 0:
        feature_cfg["scale"] = [mechanism_stage_utils.canonical_float_text(1.0)]

    if rewrite_features_requiring_dim:
        _require_non_empty(
            feature_cfg["feature_dim"],
            "search_space.feature_transformation.feature_dim",
        )
    else:
        _require_empty(
            feature_cfg["feature_dim"],
            "search_space.feature_transformation.feature_dim",
        )

    if "operator" in features:
        if len(feature_cfg["feature_preprojection"]) == 0:
            feature_cfg["feature_preprojection"] = [False]
    else:
        _require_empty(
            feature_cfg["feature_preprojection"],
            "search_space.feature_transformation.feature_preprojection",
        )

    if True in feature_preprojection_values or (
        "operator" in features and True in set(feature_cfg["feature_preprojection"])
    ):
        _require_non_empty(
            feature_cfg["preprojection_output_dim"],
            "search_space.feature_transformation.preprojection_output_dim",
        )
    else:
        _require_empty(
            feature_cfg["preprojection_output_dim"],
            "search_space.feature_transformation.preprojection_output_dim",
        )

    if "random_normal" in features:
        _require_non_empty(
            feature_cfg["random_normal_mean"],
            "search_space.feature_transformation.random_normal_mean",
        )
        _require_non_empty(
            feature_cfg["random_normal_std"],
            "search_space.feature_transformation.random_normal_std",
        )
    else:
        _require_empty(
            feature_cfg["random_normal_mean"],
            "search_space.feature_transformation.random_normal_mean",
        )
        _require_empty(
            feature_cfg["random_normal_std"],
            "search_space.feature_transformation.random_normal_std",
        )

    if "shared" in features:
        _require_non_empty(
            feature_cfg["shared_value"],
            "search_space.feature_transformation.shared_value",
        )
    else:
        _require_empty(
            feature_cfg["shared_value"],
            "search_space.feature_transformation.shared_value",
        )

    if "degree_bucket_distribution" in features or "degree_bucket_range" in features:
        _require_non_empty(
            feature_cfg["degree_bucket_num_buckets"],
            "search_space.feature_transformation.degree_bucket_num_buckets",
        )
    else:
        _require_empty(
            feature_cfg["degree_bucket_num_buckets"],
            "search_space.feature_transformation.degree_bucket_num_buckets",
        )

    if "degree_bucket_range" in features:
        _require_non_empty(
            feature_cfg["degree_bucket_range_max"],
            "search_space.feature_transformation.degree_bucket_range_max",
        )
    else:
        _require_empty(
            feature_cfg["degree_bucket_range_max"],
            "search_space.feature_transformation.degree_bucket_range_max",
        )

    if "deepwalk" in features:
        for field_name in (
            "deepwalk_walk_length",
            "deepwalk_number_walks",
            "deepwalk_window_size",
            "deepwalk_workers",
            "deepwalk_undirected",
        ):
            _require_non_empty(
                feature_cfg[field_name],
                f"search_space.feature_transformation.{field_name}",
            )
    else:
        for field_name in (
            "deepwalk_walk_length",
            "deepwalk_number_walks",
            "deepwalk_window_size",
            "deepwalk_workers",
            "deepwalk_undirected",
        ):
            _require_empty(
                feature_cfg[field_name],
                f"search_space.feature_transformation.{field_name}",
            )

    if True in norm_values:
        _require_non_empty(
            calibrator_cfg["norm_scale"],
            "search_space.calibrator.norm_scale",
        )
    else:
        _require_empty(
            calibrator_cfg["norm_scale"],
            "search_space.calibrator.norm_scale",
        )

    if True in use_nfr_values:
        _require_non_empty(
            nfr_cfg["tao2"],
            "search_space.nfr.tao2",
        )
        if len(features & SUPPORTED_REWRITE_FEATURE_SET) > 0:
            raise SearchError("use_nfr=true is currently unsupported for rewrite features")
        if "raw" in features and "inf" in x_eps_values:
            raise SearchError("use_nfr=true cannot be combined with feature=raw and x_eps=inf")
    else:
        _require_empty(
            nfr_cfg["tao2"],
            "search_space.nfr.tao2",
        )

def load_search_config(config_path: Path) -> dict[str, Any]:
    raw = mechanism_stage_utils.read_yaml_file(config_path)
    return parse_search_config(raw, source=str(config_path))


def parse_search_config(raw: Any, *, source: str) -> dict[str, Any]:
    config_path = source
    mapping = _expect_mapping(raw, str(config_path))
    if "seed" not in mapping:
        mapping = dict(mapping)
        mapping["seed"] = mechanism_stage_utils.DEFAULT_BASE_SEED
    _expect_exact_keys(mapping, TOP_LEVEL_KEYS, str(config_path))
    base_seed = _parse_nonnegative_int(mapping["seed"], f"{config_path}.seed")

    defaults = _validate_defaults_section(mapping["defaults"])
    search_space = _validate_search_space_section(mapping["search_space"])
    if defaults["trainer"]["sim_epoch_refresh"] and set(search_space["feature_transformation"]["features"]) != {"sim"}:
        raise SearchError(
            "defaults.trainer.sim_epoch_refresh=true requires search_space.feature_transformation.features to contain only 'sim'"
        )

    return {
        "device": _validate_device_section(mapping["device"]),
        "defaults": defaults,
        "search_space": search_space,
        "meta": {
            "rank_metric": "val/acc",
            "verify_topk": mechanism_stage_utils.VERIFY_TOPK,
            "base_seed": base_seed,
        },
    }


def _feature_outer_variants(search_space: dict[str, Any]) -> list[dict[str, Any]]:
    feature_cfg = search_space["feature_transformation"]
    scale_values = (
        feature_cfg["scale"]
        if len(feature_cfg["scale"]) > 0
        else [mechanism_stage_utils.canonical_float_text(1.0)]
    )
    preprojection_values = (
        feature_cfg["feature_preprojection"]
        if len(feature_cfg["feature_preprojection"]) > 0
        else [False]
    )
    variants: list[dict[str, Any]] = []

    for feature in feature_cfg["features"]:
        if feature == "raw":
            for scale in scale_values:
                variants.append(
                    {
                        "feature": feature,
                        "sim_reference_eps": None,
                        "feature_dim": None,
                        "scale": scale,
                        "feature_preprojection": None,
                        "preprojection_output_dim": None,
                        "random_normal_mean": None,
                        "random_normal_std": None,
                        "shared_value": None,
                        "degree_bucket_num_buckets": None,
                        "degree_bucket_range_max": None,
                        "deepwalk_walk_length": None,
                        "deepwalk_number_walks": None,
                        "deepwalk_window_size": None,
                        "deepwalk_workers": None,
                        "deepwalk_undirected": None,
                    }
                )
            continue

        if feature == "sim":
            for sim_reference_eps in feature_cfg["sim_reference_eps"]:
                for scale in scale_values:
                    variants.append(
                        {
                            "feature": feature,
                            "sim_reference_eps": sim_reference_eps,
                            "feature_dim": None,
                            "scale": scale,
                            "feature_preprojection": None,
                            "preprojection_output_dim": None,
                            "random_normal_mean": None,
                            "random_normal_std": None,
                            "shared_value": None,
                            "degree_bucket_num_buckets": None,
                            "degree_bucket_range_max": None,
                            "deepwalk_walk_length": None,
                            "deepwalk_number_walks": None,
                            "deepwalk_window_size": None,
                            "deepwalk_workers": None,
                            "deepwalk_undirected": None,
                        }
                    )
            continue

        if feature == "operator":
            for scale in scale_values:
                for feature_preprojection in preprojection_values:
                    if feature_preprojection:
                        output_dim_values = feature_cfg["preprojection_output_dim"]
                    else:
                        output_dim_values = [None]
                    for output_dim in output_dim_values:
                        variants.append(
                            {
                                "feature": feature,
                                "sim_reference_eps": None,
                                "feature_dim": None,
                                "scale": scale,
                                "feature_preprojection": bool(feature_preprojection),
                                "preprojection_output_dim": output_dim,
                                "random_normal_mean": None,
                                "random_normal_std": None,
                                "shared_value": None,
                                "degree_bucket_num_buckets": None,
                                "degree_bucket_range_max": None,
                                "deepwalk_walk_length": None,
                                "deepwalk_number_walks": None,
                                "deepwalk_window_size": None,
                                "deepwalk_workers": None,
                                "deepwalk_undirected": None,
                            }
                        )
            continue

        if feature == "random_normal":
            for feature_dim in feature_cfg["feature_dim"]:
                for scale in scale_values:
                    for mean_value in feature_cfg["random_normal_mean"]:
                        for std_value in feature_cfg["random_normal_std"]:
                            variants.append(
                                {
                                    "feature": feature,
                                    "sim_reference_eps": None,
                                    "feature_dim": feature_dim,
                                    "scale": scale,
                                    "feature_preprojection": None,
                                    "preprojection_output_dim": None,
                                    "random_normal_mean": mean_value,
                                    "random_normal_std": std_value,
                                    "shared_value": None,
                                    "degree_bucket_num_buckets": None,
                                    "degree_bucket_range_max": None,
                                    "deepwalk_walk_length": None,
                                    "deepwalk_number_walks": None,
                                    "deepwalk_window_size": None,
                                    "deepwalk_workers": None,
                                    "deepwalk_undirected": None,
                                }
                            )
            continue

        if feature == "shared":
            for feature_dim in feature_cfg["feature_dim"]:
                for scale in scale_values:
                    for shared_value in feature_cfg["shared_value"]:
                        variants.append(
                            {
                                "feature": feature,
                                "sim_reference_eps": None,
                                "feature_dim": feature_dim,
                                "scale": scale,
                                "feature_preprojection": None,
                                "preprojection_output_dim": None,
                                "random_normal_mean": None,
                                "random_normal_std": None,
                                "shared_value": shared_value,
                                "degree_bucket_num_buckets": None,
                                "degree_bucket_range_max": None,
                                "deepwalk_walk_length": None,
                                "deepwalk_number_walks": None,
                                "deepwalk_window_size": None,
                                "deepwalk_workers": None,
                                "deepwalk_undirected": None,
                            }
                        )
            continue

        if feature in {"random_signed_onehot", "node_degree", "pagerank", "eigen", "eigen_norm"}:
            for feature_dim in feature_cfg["feature_dim"]:
                for scale in scale_values:
                    variants.append(
                        {
                            "feature": feature,
                            "sim_reference_eps": None,
                            "feature_dim": feature_dim,
                            "scale": scale,
                            "feature_preprojection": None,
                            "preprojection_output_dim": None,
                            "random_normal_mean": None,
                            "random_normal_std": None,
                            "shared_value": None,
                            "degree_bucket_num_buckets": None,
                            "degree_bucket_range_max": None,
                            "deepwalk_walk_length": None,
                            "deepwalk_number_walks": None,
                            "deepwalk_window_size": None,
                            "deepwalk_workers": None,
                            "deepwalk_undirected": None,
                        }
                    )
            continue

        if feature == "degree_bucket_distribution":
            for feature_dim in feature_cfg["feature_dim"]:
                for scale in scale_values:
                    for num_buckets in feature_cfg["degree_bucket_num_buckets"]:
                        variants.append(
                            {
                                "feature": feature,
                                "sim_reference_eps": None,
                                "feature_dim": feature_dim,
                                "scale": scale,
                                "feature_preprojection": None,
                                "preprojection_output_dim": None,
                                "random_normal_mean": None,
                                "random_normal_std": None,
                                "shared_value": None,
                                "degree_bucket_num_buckets": num_buckets,
                                "degree_bucket_range_max": None,
                                "deepwalk_walk_length": None,
                                "deepwalk_number_walks": None,
                                "deepwalk_window_size": None,
                                "deepwalk_workers": None,
                                "deepwalk_undirected": None,
                            }
                        )
            continue

        if feature == "degree_bucket_range":
            for feature_dim in feature_cfg["feature_dim"]:
                for scale in scale_values:
                    for num_buckets in feature_cfg["degree_bucket_num_buckets"]:
                        for range_max in feature_cfg["degree_bucket_range_max"]:
                            variants.append(
                                {
                                    "feature": feature,
                                    "sim_reference_eps": None,
                                    "feature_dim": feature_dim,
                                    "scale": scale,
                                    "feature_preprojection": None,
                                    "preprojection_output_dim": None,
                                    "random_normal_mean": None,
                                    "random_normal_std": None,
                                    "shared_value": None,
                                    "degree_bucket_num_buckets": num_buckets,
                                    "degree_bucket_range_max": range_max,
                                    "deepwalk_walk_length": None,
                                    "deepwalk_number_walks": None,
                                    "deepwalk_window_size": None,
                                    "deepwalk_workers": None,
                                    "deepwalk_undirected": None,
                                }
                            )
            continue

        if feature == "deepwalk":
            for feature_dim in feature_cfg["feature_dim"]:
                for scale in scale_values:
                    for walk_length in feature_cfg["deepwalk_walk_length"]:
                        for number_walks in feature_cfg["deepwalk_number_walks"]:
                            for window_size in feature_cfg["deepwalk_window_size"]:
                                for workers in feature_cfg["deepwalk_workers"]:
                                    for undirected in feature_cfg["deepwalk_undirected"]:
                                        variants.append(
                                            {
                                                "feature": feature,
                                                "sim_reference_eps": None,
                                                "feature_dim": feature_dim,
                                                "scale": scale,
                                                "feature_preprojection": None,
                                                "preprojection_output_dim": None,
                                                "random_normal_mean": None,
                                                "random_normal_std": None,
                                                "shared_value": None,
                                                "degree_bucket_num_buckets": None,
                                                "degree_bucket_range_max": None,
                                                "deepwalk_walk_length": walk_length,
                                                "deepwalk_number_walks": number_walks,
                                                "deepwalk_window_size": window_size,
                                                "deepwalk_workers": workers,
                                                "deepwalk_undirected": undirected,
                                            }
                                        )
            continue

        raise SearchError(f"Unsupported feature variant builder for {feature!r}")

    return variants


def _active_path_parts(fixed_params: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for axis_name, value in mechanism_stage_utils.normalized_outer_fixed_params(fixed_params).items():
        if value is None:
            continue
        parts.append(f"{axis_name}={mechanism_stage_utils.path_token(value)}")
    return parts


def build_batch_spec(
    *,
    search_config: dict[str, Any],
    output_root: Path,
    config_copy_source: Path,
    pre_smoothing_feature_cache_root: str | None = None,
) -> core.BatchSpec:
    defaults = search_config["defaults"]
    search_space = search_config["search_space"]
    meta = search_config["meta"]

    training_device = "cpu" if search_config["device"].device == "cpu" else "cuda"
    jobs: list[core.BatchJob] = []

    feature_variants = _feature_outer_variants(search_space)
    for dataset_name in search_space["dataset"]["datasets"]:
        for feature_variant in feature_variants:
            for mechanism in search_space["feature_perturbation"]["mechanisms"]:
                for x_eps in search_space["feature_perturbation"]["x_eps"]:
                    for m_value in search_space["feature_perturbation"]["m"]:
                        for norm_enabled in search_space["calibrator"]["norm"]:
                            norm_scale_values = (
                                search_space["calibrator"]["norm_scale"] if norm_enabled else ["none"]
                            )
                            for norm_scale in norm_scale_values:
                                configured_smoother_values = list(search_space["calibrator"]["smoother"])
                                # Operator configs with multiple smoothers use one direct HOA job.
                                # into one HOA-style direct job. A single smoother is now explicit.
                                smoother_values = (
                                    [None]
                                    if feature_variant["feature"] == "operator" and len(configured_smoother_values) != 1
                                    else configured_smoother_values
                                )
                                for smoother in smoother_values:
                                    for backbone in search_space["model"]["backbones"]:
                                        for use_nfr in search_space["nfr"]["use_nfr"]:
                                            fixed_params = {
                                                "dataset": dataset_name,
                                                **feature_variant,
                                                "mechanism": mechanism,
                                                "x_eps": x_eps,
                                                "m": m_value,
                                                "norm": bool(norm_enabled),
                                                "norm_scale": norm_scale,
                                                "smoother": smoother,
                                                "backbone": backbone,
                                                "use_nfr": bool(use_nfr),
                                            }
                                            tao2_values = (
                                                search_space["nfr"]["tao2"] if use_nfr else ["none"]
                                            )
                                            candidates = mechanism_stage_utils.build_candidate_specs(
                                                x_steps_values=search_space["calibrator"]["x_steps"],
                                                learning_rate_values=search_space["trainer"]["learning_rate"],
                                                weight_decay_values=search_space["trainer"]["weight_decay"],
                                                dropout_values=search_space["model"]["dropout"],
                                                tao2_values=tao2_values,
                                            )
                                            candidate_space = {
                                                "x_steps": list(search_space["calibrator"]["x_steps"]),
                                                "learning_rate": list(search_space["trainer"]["learning_rate"]),
                                                "weight_decay": list(search_space["trainer"]["weight_decay"]),
                                                "dropout": list(search_space["model"]["dropout"]),
                                                "tao2": list(tao2_values),
                                            }
                                            job_id = mechanism_stage_utils.stable_job_id(fixed_params)
                                            path_parts = _active_path_parts(fixed_params)
                                            display_name = ", ".join(path_parts)
                                            job_spec = {
                                                "schema_version": mechanism_stage_utils.SCHEMA_VERSION,
                                                "job_id": job_id,
                                                "display_name": display_name,
                                                "python_bin": sys.executable,
                                                "training_device": training_device,
                                                "pre_smoothing_feature_cache_root": (
                                                    None
                                                    if pre_smoothing_feature_cache_root is None
                                                    else str(pre_smoothing_feature_cache_root).strip() or None
                                                ),
                                                "base_seed": meta["base_seed"],
                                                "rank_metric": meta["rank_metric"],
                                                "verify_topk": meta["verify_topk"],
                                                "defaults": defaults,
                                                "fixed_params": fixed_params,
                                                "candidate_space": candidate_space,
                                                "candidates": [candidate.to_dict() for candidate in candidates],
                                            }
                                            jobs.append(
                                                core.BatchJob(
                                                    job_id=job_id,
                                                    job_dir=output_root.joinpath(*path_parts),
                                                    display_name=display_name,
                                                    job_spec=job_spec,
                                                )
                                            )

    return core.BatchSpec(
        output_root=output_root,
        execution=search_config["device"],
        jobs=jobs,
        config_copy_source=config_copy_source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YAML-driven batch hyperparameter search.")
    parser.add_argument("--config", required=True, type=str, help="search configuration YAML")
    parser.add_argument(
        "--output_root_dir",
        required=True,
        type=str,
        help="root directory for all batch outputs",
    )
    parser.add_argument(
        "--pre_smoothing_feature_cache_root",
        "--pre-smoothing-feature-cache-root",
        dest="pre_smoothing_feature_cache_root",
        required=False,
        type=str,
        default=None,
        help="optional cache root for raw->perturbation->optional NFR tensors loaded before HOA/KProp",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise SearchError(f"Config file not found: {config_path}")

    search_config = load_search_config(config_path)
    batch_spec = build_batch_spec(
        search_config=search_config,
        output_root=Path(args.output_root_dir).resolve(),
        config_copy_source=config_path,
        pre_smoothing_feature_cache_root=args.pre_smoothing_feature_cache_root,
    )
    _, _, failed_count, _ = core.run_batch_search(batch_spec, repo_root=repo_root)
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SearchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except core.SuiteError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
