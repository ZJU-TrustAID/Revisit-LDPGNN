#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from hparams_search_scripts import mechanism_stage_utils as stage_utils
except ModuleNotFoundError:
    import mechanism_stage_utils as stage_utils  # type: ignore


@dataclass(frozen=True)
class JobContext:
    job_dir: Path
    job_spec: dict[str, Any]
    defaults: dict[str, Any]
    fixed_params: dict[str, Any]
    python_bin: str
    training_device: str
    base_seed: int
    verify_topk: int
    pre_smoothing_feature_cache_root: str | None


def resolve_job_context(job_dir: str | Path) -> JobContext:
    job_dir_path = Path(job_dir).resolve()
    job_spec = stage_utils.load_job_spec(job_dir_path)

    defaults = job_spec.get("defaults")
    if not isinstance(defaults, dict):
        raise stage_utils.StageError("job_spec.defaults must be a mapping")

    fixed_params = job_spec.get("fixed_params")
    if not isinstance(fixed_params, dict):
        raise stage_utils.StageError("job_spec.fixed_params must be a mapping")

    python_bin = str(job_spec.get("python_bin", "python")).strip() or "python"
    training_device = str(job_spec.get("training_device", "cpu")).strip().lower()
    pre_smoothing_feature_cache_root = job_spec.get("pre_smoothing_feature_cache_root")
    if pre_smoothing_feature_cache_root is not None:
        pre_smoothing_feature_cache_root = str(pre_smoothing_feature_cache_root).strip() or None
    if training_device not in {"cpu", "cuda"}:
        raise stage_utils.StageError(
            f"Unsupported training_device={training_device!r} in job spec"
        )

    try:
        base_seed = int(job_spec.get("base_seed", stage_utils.DEFAULT_BASE_SEED))
        verify_topk = int(job_spec.get("verify_topk", stage_utils.VERIFY_TOPK))
    except (TypeError, ValueError) as exc:
        raise stage_utils.StageError("job_spec base_seed/verify_topk must be integers") from exc

    return JobContext(
        job_dir=job_dir_path,
        job_spec=job_spec,
        defaults=defaults,
        fixed_params=fixed_params,
        python_bin=python_bin,
        training_device=training_device,
        base_seed=base_seed,
        verify_topk=verify_topk,
        pre_smoothing_feature_cache_root=pre_smoothing_feature_cache_root,
    )


def _append_flag(parts: list[str], name: str, value: Any) -> None:
    parts.extend([f"--{name}", str(value)])


def _append_optional(parts: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and value.strip() == "":
        return
    _append_flag(parts, name, value)


def _append_bool(parts: list[str], name: str, value: bool) -> None:
    parts.extend([f"--{name}", "true" if value else "false"])


def _append_feature_specific_args(parts: list[str], fixed_params: dict[str, Any]) -> None:
    feature = str(fixed_params["feature"])
    scale = fixed_params.get("scale")
    if scale is not None and not stage_utils.is_default_rewrite_scale_value(scale):
        _append_flag(parts, "scale", scale)

    if feature == "sim":
        _append_flag(parts, "sim_reference_eps", fixed_params["sim_reference_eps"])
        return

    if feature == "operator":
        if bool(fixed_params.get("feature_preprojection")):
            _append_bool(parts, "feature_preprojection", True)
            _append_flag(parts, "preprojection_output_dim", fixed_params["preprojection_output_dim"])
        return

    if feature == "raw":
        return

    for field_name in (
        "feature_dim",
        "random_normal_mean",
        "random_normal_std",
        "shared_value",
        "degree_bucket_num_buckets",
        "degree_bucket_range_max",
        "deepwalk_walk_length",
        "deepwalk_number_walks",
        "deepwalk_window_size",
        "deepwalk_workers",
    ):
        value = fixed_params.get(field_name)
        if value is not None:
            _append_flag(parts, field_name, value)

    if fixed_params.get("deepwalk_undirected") is not None:
        _append_bool(parts, "deepwalk_undirected", bool(fixed_params["deepwalk_undirected"]))


def build_main_base_args(ctx: JobContext) -> list[str]:
    parts = [
        "main.py",
        "--dataset",
        str(ctx.fixed_params["dataset"]),
        "--feature",
        str(ctx.fixed_params["feature"]),
        "--mechanism",
        str(ctx.fixed_params["mechanism"]),
        "--x_eps",
        str(ctx.fixed_params["x_eps"]),
        "--m",
        str(ctx.fixed_params["m"]),
        "--model",
        str(ctx.fixed_params["backbone"]),
        "--hidden_dim",
        str(ctx.defaults["model"]["hidden_dim"]),
        "--optimizer",
        str(ctx.defaults["trainer"]["optimizer"]),
        "--device",
        ctx.training_device,
        "--val_ratio",
        str(ctx.defaults["dataset"]["val_ratio"]),
        "--test_ratio",
        str(ctx.defaults["dataset"]["test_ratio"]),
    ]

    data_range = ctx.defaults["dataset"]["data_range"]
    parts.extend(["--data_range", str(data_range[0]), str(data_range[1])])

    _append_feature_specific_args(parts, ctx.fixed_params)
    if ctx.fixed_params.get("smoother") is not None:
        parts.extend(["--smoother", str(ctx.fixed_params["smoother"])])

    _append_bool(parts, "norm", bool(ctx.fixed_params["norm"]))
    if bool(ctx.fixed_params["norm"]) and str(ctx.fixed_params["norm_scale"]) != "none":
        _append_flag(parts, "norm_scale", ctx.fixed_params["norm_scale"])

    _append_bool(
        parts,
        "gradient_clip",
        bool(ctx.defaults["trainer"]["gradient_clip"]),
    )
    _append_flag(
        parts,
        "gradient_clip_max_norm",
        ctx.defaults["trainer"]["gradient_clip_max_norm"],
    )
    _append_bool(
        parts,
        "sim_epoch_refresh",
        bool(ctx.defaults["trainer"]["sim_epoch_refresh"]),
    )
    _append_bool(
        parts,
        "show_progress",
        bool(ctx.defaults["trainer"]["show_progress"]),
    )
    _append_bool(
        parts,
        "log_every_epoch",
        bool(ctx.defaults["trainer"]["log_every_epoch"]),
    )

    if bool(ctx.fixed_params["use_nfr"]):
        parts.append("--use_nfr")
    _append_optional(
        parts,
        "pre_smoothing_feature_cache_root",
        ctx.pre_smoothing_feature_cache_root,
    )

    return parts


def build_stage_command(
    ctx: JobContext,
    candidate: stage_utils.CandidateSpec,
    *,
    stage_name: str,
    output_dir: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    repeats: int,
) -> list[str]:
    parts = [
        ctx.python_bin,
        *build_main_base_args(ctx),
        "--x_steps",
        str(candidate.x_steps),
        "--learning_rate",
        candidate.learning_rate,
        "--weight_decay",
        candidate.weight_decay,
        "--dropout",
        candidate.dropout,
        "--max_epochs",
        str(max_epochs),
        "--patience",
        str(patience),
        "-s",
        str(seed),
        "-r",
        str(repeats),
        "-o",
        str(output_dir),
    ]

    if bool(ctx.fixed_params["use_nfr"]) and candidate.tao2 != "none":
        _append_flag(parts, "tao2", candidate.tao2)

    return parts


def build_recommended_command_parts(
    ctx: JobContext,
    candidate: stage_utils.CandidateSpec,
) -> list[str]:
    verify_defaults = ctx.defaults["stage"]["verify"]
    output_dir = ctx.job_dir / "final"
    return build_stage_command(
        ctx,
        candidate,
        stage_name="final",
        output_dir=output_dir,
        seed=ctx.base_seed,
        max_epochs=int(verify_defaults["max_epochs"]),
        patience=int(verify_defaults["patience"]),
        repeats=int(verify_defaults["repeats"]),
    )
