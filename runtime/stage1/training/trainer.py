import logging
import torch
import os
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)
import numpy as np
import random
import functools
from functools import partial
import math
import json
from pathlib import Path
import time
from tqdm import tqdm
from enum import Enum
import io
import gzip
import pandas as pd
import glob
import signal
from pandas import Series
from scipy.io.wavfile import write
import traceback
from types import SimpleNamespace
from torch.utils.data import Subset
from torch.nn import functional as F
import distributed
from training import log
from training.alignkd import (
    attention_kd_per_sample,
    globally_normalized_masked_loss,
    gram_all_kd_per_sample,
    gram_soft_kd_per_sample,
    gram_topk_kd_per_sample,
    loss_gradient_l2_norm,
    parameter_gradient_l2_norm,
    sparse_topk_tail_kd_per_sample,
    temporal_cosine_gram,
)
from models.model import get_model_class
from data.sampler import CustomDistributedSampler
from utils.utils import retry, numparams, group_weight_decay_params
from utils.utils import GradNormTracker, LossTrackingLRScheduler, LazyConversionDict
from metrics.get_metrics import Metric
from models.generate import generate_greedy, generate_greedy_batch

class TrainerMode(Enum):
    Train = "train"
    EvaluateCheckpoint = "evaluate_checkpoint"

def worker_init_fn(logging_initializer, worker_id):
    # Initialize logging for this worker
    # This prevents "I/O operation on closed file" errors
    try:
        logging_initializer()
    except Exception as e:
        # If logging initialization fails, continue without it
        # to avoid breaking the data loading
        pass  # Silently fail - don't print to avoid spam
    
    # Suppress ALL logging from workers to avoid spam
    # Workers should not log - only main process should log
    logging.getLogger().setLevel(logging.CRITICAL)  # Only critical errors
    
    # Suppress transformers and other library warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set random seed for reproducibility
    seed = torch.utils.data.get_worker_info().seed
    sync_random_seed(seed)

def sync_random_seed(seed):
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def output_control_weighted_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    region_ids: torch.Tensor,
    *,
    structure_weight: float,
    reasoning_weight: float,
    answer_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Token-normalized CE emphasizing strict structure and copied choices."""

    if logits.shape[:2] != targets.shape or targets.shape != region_ids.shape:
        raise ValueError("output-control logits/targets/regions shape mismatch")
    weights = (structure_weight, reasoning_weight, answer_weight)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in weights
    ):
        raise ValueError("output-control region weights must be finite and positive")
    if bool(((region_ids < 0) | (region_ids > 3)).any()):
        raise ValueError("output-control region ids must be in {0,1,2,3}")
    active = region_ids.gt(0)
    if not bool(active.any()):
        raise ValueError("output-control batch has no active target tokens")
    if any(not bool(region_ids.eq(value).any()) for value in (1, 2, 3)):
        raise ValueError("output-control batch must contain all three token regions")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    token_weights = torch.zeros_like(per_token)
    token_weights = torch.where(region_ids.eq(1), float(structure_weight), token_weights)
    token_weights = torch.where(region_ids.eq(2), float(reasoning_weight), token_weights)
    token_weights = torch.where(region_ids.eq(3), float(answer_weight), token_weights)
    loss = (per_token * token_weights).sum() / token_weights.sum().clamp_min(1e-12)
    diagnostics = {
        name: per_token.masked_select(region_ids.eq(value)).mean()
        for name, value in (("structure", 1), ("reasoning", 2), ("answer", 3))
    }
    return loss, diagnostics


def validate_output_control_weight_schedule(
    output_control_config: dict,
    max_optimizer_updates: int,
) -> tuple[dict[str, Any], ...]:
    """Validate and normalize the output-control region-weight schedule."""

    if not isinstance(output_control_config, dict):
        raise ValueError("train.output_control_warmup must be a mapping")

    raw_phases = output_control_config.get("phases")
    if raw_phases is None:
        raw_phases = [
            {
                "start_optimizer_step": 1,
                "end_optimizer_step": max_optimizer_updates or None,
                "structure_weight": output_control_config.get(
                    "structure_weight", 1.0
                ),
                "reasoning_weight": output_control_config.get(
                    "reasoning_weight", 0.25
                ),
                "answer_weight": output_control_config.get("answer_weight", 4.0),
            }
        ]
    elif not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError(
            "train.output_control_warmup.phases must be a non-empty list"
        )

    phases = []
    expected_start = 1
    for index, raw_phase in enumerate(raw_phases):
        if not isinstance(raw_phase, dict):
            raise ValueError(f"output-control phase {index} must be a mapping")
        start = raw_phase.get("start_optimizer_step")
        end = raw_phase.get("end_optimizer_step")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start != expected_start
        ):
            raise ValueError(
                "output-control phases must be ordered, contiguous, and start at "
                f"optimizer step 1; phase {index} starts at {start!r}, expected "
                f"{expected_start}"
            )
        if (
            isinstance(end, bool)
            or (end is not None and not isinstance(end, int))
            or (end is not None and end < start)
        ):
            raise ValueError(
                f"output-control phase {index} has invalid end_optimizer_step={end!r}"
            )
        if end is None and index != len(raw_phases) - 1:
            raise ValueError("only the final output-control phase may have no end step")

        weights = {}
        for key, default in (
            ("structure_weight", 1.0),
            ("reasoning_weight", 0.25),
            ("answer_weight", 4.0),
        ):
            value = raw_phase.get(key, default)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"output-control phase {index} {key} must be finite and positive"
                )
            weights[key] = float(value)

        phases.append(
            {
                "start_optimizer_step": start,
                "end_optimizer_step": end,
                "label": str(raw_phase.get("label", f"phase_{index + 1}")),
                **weights,
            }
        )
        if end is not None:
            expected_start = end + 1

    if max_optimizer_updates > 0:
        final_end = phases[-1]["end_optimizer_step"]
        if final_end != max_optimizer_updates:
            raise ValueError(
                "output-control phases must cover the complete formal horizon: "
                f"final end={final_end!r}, max_optimizer_updates={max_optimizer_updates}"
            )
    return tuple(phases)


def output_control_weights_for_optimizer_step(
    phases: Sequence[dict[str, Any]],
    optimizer_step: int,
) -> tuple[int, float, float, float]:
    """Resolve the prevalidated phase and weights for one optimizer update."""

    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step <= 0
    ):
        raise ValueError(
            f"optimizer_step must be a positive integer, got {optimizer_step!r}"
        )
    for index, phase in enumerate(phases):
        end = phase["end_optimizer_step"]
        if optimizer_step >= phase["start_optimizer_step"] and (
            end is None or optimizer_step <= end
        ):
            return (
                index,
                phase["structure_weight"],
                phase["reasoning_weight"],
                phase["answer_weight"],
            )
    raise ValueError(
        f"no output-control phase covers optimizer step {optimizer_step}"
    )


def resolve_step_lr_schedule_total_steps(
    configured_steps: int | None,
    inferred_steps: int,
    allow_beyond_run: bool = False,
) -> int:
    """Resolve an explicit LR horizon, optionally longer than this run cutoff."""

    if (
        isinstance(inferred_steps, bool)
        or not isinstance(inferred_steps, int)
        or inferred_steps <= 0
    ):
        raise ValueError(
            f"inferred LR schedule steps must be positive, got {inferred_steps!r}"
        )
    if configured_steps is None:
        return inferred_steps
    if (
        isinstance(configured_steps, bool)
        or not isinstance(configured_steps, int)
        or configured_steps <= 0
        or (configured_steps > inferred_steps and not allow_beyond_run)
    ):
        raise ValueError(
            "train.optimizer.schedule_total_optimizer_steps must be an integer in "
            f"[1, {inferred_steps}], got {configured_steps!r}"
        )
    return configured_steps


def validate_train_drop_last(data_config: dict) -> bool:
    """Return the train-loader drop policy, preserving the legacy default."""
    value = data_config.get("drop_last", True)
    if not isinstance(value, bool):
        raise ValueError(
            f"data.drop_last must be true or false, got {value!r}"
        )
    return value


def validate_train_sampler_shuffle(order_config: dict) -> bool:
    """Resolve sampler shuffling while preserving the legacy default."""
    value = order_config.get("sampler_shuffle", True)
    if not isinstance(value, bool):
        raise ValueError(
            f"data.order.sampler_shuffle must be true or false, got {value!r}"
        )
    return value


def validate_train_loader_world_size(
    configured_world_size: int, actual_world_size: int, drop_last: bool
) -> None:
    """Fail before sampler construction when exposure identity could drift."""
    if int(configured_world_size) != int(actual_world_size):
        raise RuntimeError(
            "Configured/runtime world-size mismatch: "
            f"{configured_world_size} != {actual_world_size}"
        )
    if not drop_last and int(actual_world_size) != 1:
        raise RuntimeError(
            "data.drop_last=false is supported only for world_size=1; "
            "distributed padding would duplicate exposures"
        )


_TRAINING_STATE_CONFIG_KEYS = (
    "myconfig",
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "max_optimizer_updates",
    "checkpoint_optimizer_steps",
    "naive_kd",
    "retention_kd",
    "abc_replay_objective",
    "sequence_retention_kd",
    "align_kd",
    "output_control_warmup",
    "optimizer",
    "resume_branch",
    "world_size",
    "data",
)


def training_state_config_contract(config: dict) -> dict:
    """Return the exact config subset covered by full-state resume."""
    train_config = config["train"]
    data_config = config.get("data", {}) or {}
    return {
        "myconfig": config.get("myconfig"),
        "num_epochs": train_config.get("num_epochs"),
        "batch_size": train_config.get("batch_size"),
        "gradient_accumulation_steps": train_config.get(
            "gradient_accumulation_steps"
        ),
        "max_optimizer_updates": train_config.get("max_optimizer_updates"),
        "checkpoint_optimizer_steps": train_config.get(
            "checkpoint_optimizer_steps"
        ),
        "naive_kd": train_config.get("naive_kd"),
        "retention_kd": train_config.get("retention_kd"),
        "abc_replay_objective": train_config.get("abc_replay_objective"),
        "sequence_retention_kd": train_config.get("sequence_retention_kd"),
        "align_kd": train_config.get("align_kd"),
        "output_control_warmup": train_config.get("output_control_warmup"),
        "optimizer": train_config.get("optimizer"),
        "resume_branch": train_config.get("resume_branch"),
        "world_size": train_config.get("world_size"),
        "data": {
            "datafiles": data_config.get("datafiles"),
            "manifest_sha256": data_config.get("manifest_sha256"),
            "drop_last": data_config.get("drop_last", True),
            "order": data_config.get("order"),
            "ced_hidden_cache": data_config.get("ced_hidden_cache"),
            "sequence_retention_cache": data_config.get(
                "sequence_retention_cache"
            ),
        },
    }


def strict_resume_config_mismatches(saved_config: dict, current_config: dict) -> dict:
    """Describe every strict resume-contract mismatch without an escape hatch."""
    if not isinstance(saved_config, dict):
        return {"config": {"saved": type(saved_config).__name__, "current": "dict"}}

    missing = "<MISSING>"
    mismatches = {}
    expected_keys = set(_TRAINING_STATE_CONFIG_KEYS)
    for key in sorted(expected_keys):
        saved_value = saved_config[key] if key in saved_config else missing
        current_value = current_config[key] if key in current_config else missing
        # Checkpoints written before resume lineage existed have no key.  This is
        # semantically identical only for a non-branching current run.
        if (
            key == "resume_branch"
            and saved_value == missing
            and current_value is None
        ):
            continue
        if saved_value != current_value:
            mismatches[key] = {
                "saved": saved_value,
                "current": current_value,
            }

    unexpected_keys = sorted(set(saved_config) - expected_keys)
    if unexpected_keys:
        mismatches["unexpected_saved_config_keys"] = {
            "saved": unexpected_keys,
            "current": [],
        }
    return mismatches


def validate_naive_kd_horizon(value):
    """Validate an inclusive optimizer-step KD horizon (None means unbounded)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "train.naive_kd.active_until_optimizer_step_inclusive must be "
            f"an integer >= 0 or null, got {value!r}"
        )
    if value < 0:
        raise ValueError(
            "train.naive_kd.active_until_optimizer_step_inclusive must be >= 0, "
            f"got {value}"
        )
    return value


def validate_naive_kd_scores_key(value) -> str:
    """Validate the per-row four-option score field consumed by naive KD."""
    if value is None:
        return "teacher_scores_abcd"
    if (
        not isinstance(value, str)
        or not value.isidentifier()
        or not value.endswith("_scores_abcd")
    ):
        raise ValueError(
            "train.naive_kd.target_scores_key must be an identifier ending "
            f"in _scores_abcd, got {value!r}"
        )
    return value


def validate_replay_target_scores_key(value) -> str:
    """Validate the routed four-option score field consumed by replay KD."""
    if value is None:
        return "ref_scores_abcd"
    if (
        not isinstance(value, str)
        or not value.isidentifier()
        or not value.endswith("_scores_abcd")
    ):
        raise ValueError(
            "train.abc_replay_objective.target_scores_key must be an identifier "
            f"ending in _scores_abcd, got {value!r}"
        )
    return value


def validate_replay_first_token_gold_ce_weights(
    value, group_count: int
) -> tuple[float, ...]:
    """Return per-group first-token gold-choice CE weights; omitted means off."""
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, int)
        or group_count < 1
    ):
        raise ValueError(
            f"replay group_count must be a positive integer, got {group_count!r}"
        )
    if value is None:
        return (0.0,) * group_count
    if not isinstance(value, (list, tuple)) or len(value) != group_count:
        raise ValueError(
            "train.abc_replay_objective.first_token_gold_ce_weights must contain "
            f"one value per replay group ({group_count}), got {value!r}"
        )
    weights = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item < 0.0 for item in weights):
        raise ValueError(
            "train.abc_replay_objective.first_token_gold_ce_weights must be "
            "finite and non-negative"
        )
    return weights


def validate_replay_gradient_projection(value, group_names):
    """Validate optional one-sided correction-vs-retention gradient projection."""
    group_names = tuple(str(name) for name in group_names)
    config = dict(value or {})
    enabled = bool(config.get("enabled", False))
    if not enabled:
        return {
            "enabled": False,
            "correction_groups": tuple(),
            "retention_groups": tuple(),
            "eps": 1e-12,
        }
    correction = tuple(str(name) for name in config.get("correction_groups", ()))
    retention = tuple(str(name) for name in config.get("retention_groups", ()))
    if not correction or not retention:
        raise ValueError(
            "abc_replay_objective.gradient_projection requires non-empty "
            "correction_groups and retention_groups"
        )
    if len(set(correction)) != len(correction) or len(set(retention)) != len(retention):
        raise ValueError("gradient-projection group lists must be unique")
    overlap = set(correction).intersection(retention)
    unknown = set(correction).union(retention).difference(group_names)
    missing = set(group_names).difference(correction).difference(retention)
    if overlap or unknown or missing:
        raise ValueError(
            "gradient-projection groups must form an exact disjoint partition of "
            f"replay groups; overlap={sorted(overlap)} unknown={sorted(unknown)} "
            f"missing={sorted(missing)}"
        )
    eps = float(config.get("eps", 1e-12))
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("gradient-projection eps must be finite and positive")
    return {
        "enabled": True,
        "correction_groups": correction,
        "retention_groups": retention,
        "eps": eps,
    }


def project_correction_gradients_(
    named_parameters,
    correction_gradients,
    *,
    eps: float = 1e-12,
):
    """Project correction gradients away from a conflicting retention gradient.

    ``parameter.grad`` must contain the retention gradient.  Correction
    gradients are supplied separately.  The function replaces each active
    ``parameter.grad`` with ``g_ret + g_corr_projected`` and returns scalar
    diagnostics.  It deliberately leaves retention untouched.
    """
    named_parameters = tuple(named_parameters)
    correction_gradients = tuple(correction_gradients)
    if len(named_parameters) != len(correction_gradients):
        raise ValueError("correction gradient/parameter length mismatch")
    dot = None
    correction_norm_sq = None
    retention_norm_sq = None
    for (_, parameter), correction in zip(named_parameters, correction_gradients):
        retention = parameter.grad
        if correction is None or retention is None:
            continue
        correction_f = correction.detach().float()
        retention_f = retention.detach().float()
        item_dot = torch.sum(correction_f * retention_f)
        item_corr = torch.sum(correction_f * correction_f)
        item_ret = torch.sum(retention_f * retention_f)
        dot = item_dot if dot is None else dot + item_dot
        correction_norm_sq = item_corr if correction_norm_sq is None else correction_norm_sq + item_corr
        retention_norm_sq = item_ret if retention_norm_sq is None else retention_norm_sq + item_ret
    if dot is None:
        raise ValueError("gradient projection found no shared active parameters")
    eps_tensor = dot.new_tensor(float(eps))
    conflict = bool((dot < 0.0).item()) and bool((retention_norm_sq > eps_tensor).item())
    coefficient = dot / retention_norm_sq.clamp_min(eps_tensor) if conflict else dot.new_zeros(())
    for (_, parameter), correction in zip(named_parameters, correction_gradients):
        retention = parameter.grad
        if correction is None:
            continue
        projected = correction
        if conflict and retention is not None:
            projected = correction - coefficient.to(dtype=correction.dtype) * retention
        if retention is None:
            parameter.grad = projected.clone()
        else:
            parameter.grad.add_(projected)
    denom = torch.sqrt(correction_norm_sq.clamp_min(eps_tensor)) * torch.sqrt(
        retention_norm_sq.clamp_min(eps_tensor)
    )
    cosine = (dot / denom).clamp(min=-1.0, max=1.0)
    return {
        "conflict": conflict,
        "cosine_before": float(cosine.item()),
        "correction_removed_fraction": float(max(0.0, -cosine.item())) if conflict else 0.0,
        "dot": float(dot.item()),
        "correction_norm": float(torch.sqrt(correction_norm_sq).item()),
        "retention_norm": float(torch.sqrt(retention_norm_sq).item()),
    }


def resolve_replay_target_scores(
    batch_data_dict: Mapping[str, Any], target_scores_key: str, device
) -> torch.Tensor:
    """Load the configured routed replay-KD target without assuming incumbent scores."""
    target_scores_key = validate_replay_target_scores_key(target_scores_key)
    if target_scores_key not in batch_data_dict:
        raise ValueError(
            f"abc_replay_objective batch is missing routed target {target_scores_key!r}"
        )
    scores = batch_data_dict[target_scores_key]
    if not isinstance(scores, torch.Tensor):
        scores = torch.as_tensor(scores)
    return scores.to(device, non_blocking=True).float()


def abcd_kd_per_sample(
    student_scores: torch.Tensor,
    target_scores: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Temperature-scaled four-option KL shared by correction and retention KD."""
    if student_scores.ndim != 2 or tuple(student_scores.shape) != tuple(
        target_scores.shape
    ) or student_scores.shape[-1] != 4:
        raise ValueError(
            "ABCD KD scores must have matching [B,4] shapes, got "
            f"student={tuple(student_scores.shape)} target={tuple(target_scores.shape)}"
        )
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"ABCD KD tau must be finite and positive, got {tau!r}")
    if not bool(torch.isfinite(student_scores).all()) or not bool(
        torch.isfinite(target_scores).all()
    ):
        raise FloatingPointError("ABCD KD scores contain NaN or Inf")
    target_probs = F.softmax(target_scores.float() / tau, dim=-1)
    student_log_probs = F.log_softmax(student_scores.float() / tau, dim=-1)
    return F.kl_div(
        student_log_probs,
        target_probs,
        reduction="none",
    ).sum(dim=-1) * (tau ** 2)


def masked_abcd_kd_per_sample(
    student_scores: torch.Tensor,
    target_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Temperature-scaled ABCD KL restricted to each row's valid options."""
    if (
        student_scores.ndim != 2
        or tuple(student_scores.shape) != tuple(target_scores.shape)
        or tuple(student_scores.shape) != tuple(valid_mask.shape)
        or student_scores.shape[-1] != 4
    ):
        raise ValueError(
            "masked ABCD KD tensors must have matching [B,4] shapes, got "
            f"student={tuple(student_scores.shape)} "
            f"target={tuple(target_scores.shape)} mask={tuple(valid_mask.shape)}"
        )
    tau = float(tau)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"masked ABCD KD tau must be finite and positive, got {tau!r}")
    if not bool(torch.isfinite(student_scores).all()) or not bool(
        torch.isfinite(target_scores).all()
    ):
        raise FloatingPointError("masked ABCD KD scores contain NaN or Inf")
    valid_mask = valid_mask.bool()
    if bool(valid_mask.sum(dim=-1).lt(2).any()):
        raise ValueError("masked ABCD KD requires at least two valid options per row")
    floor = torch.finfo(torch.float32).min
    target_logits = (target_scores.float() / tau).masked_fill(~valid_mask, floor)
    student_logits = (student_scores.float() / tau).masked_fill(~valid_mask, floor)
    target_probs = F.softmax(target_logits, dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    per_coordinate = F.kl_div(student_log_probs, target_probs, reduction="none")
    return per_coordinate.masked_fill(~valid_mask, 0.0).sum(dim=-1) * (tau ** 2)


def replay_group_weighted_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    group_weights: tuple[float, ...],
    *,
    ignore_index: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Compute a token-normalized CE mean per replay stream, then weight streams."""
    if logits.shape[:2] != targets.shape or targets.shape[0] != group_ids.numel():
        raise ValueError("replay CE logits/targets/group shape mismatch")
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape_as(targets)
    active_tokens = targets.ne(ignore_index)
    means = []
    for group_id in range(len(group_weights)):
        selected = group_ids.eq(group_id).unsqueeze(-1) & active_tokens
        if not bool(selected.any()):
            raise ValueError(f"replay CE group {group_id} has no active target tokens")
        means.append(per_token.masked_select(selected).mean())
    loss = sum(float(weight) * value for weight, value in zip(group_weights, means))
    return loss, tuple(means)


def replay_group_first_token_gold_ce(
    student_scores: torch.Tensor,
    gold_option_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    group_ids: torch.Tensor,
    group_weights: tuple[float, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Compute valid-option gold CE at the first answer token per replay group."""
    if (
        student_scores.ndim != 2
        or student_scores.shape[-1] != 4
        or tuple(student_scores.shape) != tuple(valid_mask.shape)
        or gold_option_indices.ndim != 1
        or group_ids.ndim != 1
        or student_scores.shape[0] != gold_option_indices.numel()
        or student_scores.shape[0] != group_ids.numel()
    ):
        raise ValueError(
            "replay first-token gold CE requires scores/mask [B,4] and gold/group [B]"
        )
    if not group_weights:
        raise ValueError("replay first-token gold CE requires at least one group")
    if any(
        not math.isfinite(float(weight)) or float(weight) < 0.0
        for weight in group_weights
    ):
        raise ValueError(
            "replay first-token gold CE weights must be finite and non-negative"
        )
    if not bool(torch.isfinite(student_scores).all()):
        raise FloatingPointError("replay first-token gold CE scores contain NaN or Inf")

    valid_mask = valid_mask.bool()
    gold_option_indices = gold_option_indices.long()
    if bool(valid_mask.sum(dim=-1).lt(2).any()):
        raise ValueError("replay first-token gold CE requires at least two valid options per row")
    invalid_gold = gold_option_indices.lt(0) | gold_option_indices.ge(4)
    safe_gold = gold_option_indices.clamp(min=0, max=3)
    invalid_gold = invalid_gold | ~valid_mask.gather(1, safe_gold[:, None]).squeeze(1)
    if bool(invalid_gold.any()):
        raise ValueError("replay first-token gold CE gold index is invalid or masked")

    floor = torch.finfo(torch.float32).min
    masked_scores = student_scores.float().masked_fill(~valid_mask, floor)
    per_sample = F.cross_entropy(masked_scores, gold_option_indices, reduction="none")
    means = []
    for group_id in range(len(group_weights)):
        selected = group_ids.eq(group_id)
        if not bool(selected.any()):
            raise ValueError(f"replay first-token gold CE group {group_id} has no samples")
        means.append(per_sample.masked_select(selected).mean())
    loss = sum(float(weight) * value for weight, value in zip(group_weights, means))
    return loss, tuple(means)


def slice_model_input(input_dict: Any, row_slice: slice) -> dict[str, Any]:
    """Materialize and slice every batch-first model input for one replay stream."""
    result: dict[str, Any] = {}
    for key in input_dict.keys():
        value = input_dict[key]
        if isinstance(value, dict):
            result[key] = {inner_key: inner_value[row_slice] for inner_key, inner_value in value.items()}
        else:
            result[key] = value[row_slice]
    return result


def replay_group_slices(
    group_ids: torch.Tensor,
    configured_counts: tuple[int, ...],
    *,
    dynamic_counts: bool,
) -> tuple[tuple[slice, ...], tuple[int, ...]]:
    """Validate contiguous replay groups and return their batch slices.

    The historical replay contract uses one fixed count per group.  Full-pool
    natural sampling needs the counts to vary slightly from batch to batch,
    while retaining one contiguous forward per group for memory safety.  This
    helper keeps the historical path unchanged and makes the dynamic path
    fail closed if a group is absent, reordered, or outside the configured
    group inventory.
    """
    if group_ids.ndim != 1:
        raise ValueError("abc_replay_objective group ids must be one-dimensional")
    group_count = len(configured_counts)
    if group_count < 1:
        raise ValueError("abc_replay_objective requires at least one group")
    if dynamic_counts:
        if group_ids.numel() < group_count:
            raise ValueError("dynamic replay batch is smaller than its group inventory")
        if bool(group_ids.lt(0).any()) or bool(group_ids.ge(group_count).any()):
            raise ValueError("dynamic replay group id is outside the configured inventory")
        counts_tensor = torch.bincount(group_ids, minlength=group_count)
        counts = tuple(int(value) for value in counts_tensor.detach().cpu().tolist())
        if any(value <= 0 for value in counts):
            raise ValueError("dynamic replay batches must contain every configured group")
        expected = torch.repeat_interleave(
            torch.arange(group_count, device=group_ids.device, dtype=torch.long),
            counts_tensor,
        )
    else:
        counts = tuple(int(value) for value in configured_counts)
        expected = torch.repeat_interleave(
            torch.arange(group_count, device=group_ids.device, dtype=torch.long),
            torch.tensor(counts, device=group_ids.device),
        )
    if not torch.equal(group_ids, expected):
        raise ValueError(
            "abc_replay_objective requires the configured contiguous replay group order"
        )
    slices = []
    start = 0
    for count in counts:
        stop = start + count
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices), counts


def naive_kd_is_active(
    enabled: bool,
    total_step: int,
    active_until_optimizer_step_inclusive: int | None,
) -> bool:
    """Return whether KD applies to the pending optimizer update."""
    horizon = validate_naive_kd_horizon(active_until_optimizer_step_inclusive)
    return bool(enabled) and (
        horizon is None or int(total_step) + 1 <= horizon
    )


def prepare_resumed_data_iterator(
    data_loader,
    resume_batch_index: int,
    restore_rng_state,
):
    """Position an epoch iterator before restoring checkpoint RNG state.

    A mid-epoch resume must replay already-consumed dataset batches to restore
    the iterator position, then restore the checkpoint RNG *before* fetching
    the first resumed batch.  Restoring inside a ``for`` body is one batch too
    late because ``enumerate`` has already fetched that batch.
    """
    resume_batch_index = int(resume_batch_index)
    if resume_batch_index < 0:
        raise ValueError(
            f"resume_batch_index must be non-negative, got {resume_batch_index}"
        )
    if resume_batch_index == 0:
        restore_rng_state()
        return iter(data_loader)

    num_workers = int(getattr(data_loader, "num_workers", 0) or 0)
    if num_workers != 0:
        raise RuntimeError(
            "Exact mid-epoch full-state resume requires num_workers=0; "
            f"got num_workers={num_workers}"
        )

    data_iterator = iter(data_loader)
    for skipped_batch_index in range(resume_batch_index):
        try:
            next(data_iterator)
        except StopIteration as exc:
            raise RuntimeError(
                "Resume batch index exceeds the available epoch batches: "
                f"resume_batch_index={resume_batch_index} "
                f"available_before_exhaustion={skipped_batch_index}"
            ) from exc
    restore_rng_state()
    return data_iterator

class Trainer:

    def __init__(self, config, distributed_ctx: distributed.IDistributedContext = distributed.get_local_context()):
        self.distributed = distributed_ctx
        # noinspection PyPackageRequirements
        self.logger = logging.getLogger(__name__)
        self.config = config

        # init device
        self.device = None
        self.device_type = None
        if config["gpu"] and torch.cuda.is_available():
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.device = torch.device(self.device_type)

        self.use_mixed_precision = config["train"]["mixed_precision"]["use_mixed_precision"]
        if self.use_mixed_precision:
            amp_dtype = config.get("mixed_precision_dtype", "float16")
            if amp_dtype == "float16":
                dtype = torch.float16
            elif amp_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown mixed precision dtype: {amp_dtype}")
        else:
            if self.device_type == "cuda":
                dtype = torch.float16
            elif self.device_type == "cpu":
                dtype = torch.bfloat16
        self.fast_dtype = dtype

        log_file_name = self.config.get('log_file_name')
        if log_file_name is not None:
            # select log file name by local rank
            log_file_name = log_file_name.split(os.path.pathsep)
            if len(log_file_name) == 0:
                log_file_name = None
            else:
                log_file_name = log_file_name[distributed_ctx.local_rank() % len(log_file_name)]

        # Fix level for all installed handlers
        logger = logging.getLogger()
        if len(logger.handlers) == 0:
            from log import configure_logging
            configure_logging(file_name=log_file_name)
        elif log_file_name is not None:
            # initializing separate file logging
            for handler in logger.handlers:
                level = logger.getEffectiveLevel()
                if level > handler.level:
                    handler.setLevel(level)

            # And enable INFO to separate log file
            logger.setLevel(logging.INFO)

            os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(next(iter(logger.handlers)).formatter)
            logger.addHandler(file_handler)

        self._worker_log_sink = None

        seed = config["train"]["random_seed"]
        if seed is None:
            seed = torch.seed()
            logging.info(f"Random seed is {seed}")
        else:
            if isinstance(seed, Sequence):
                seed = seed[self.distributed.rank()]
            else:
                seed += self.distributed.rank()
            logging.info(f"Setting random seed to {seed}")
            torch.manual_seed(seed)
        sync_random_seed(seed)

        self._is_cleanup_enabled = 0
        self._cleanup_hook_list = list()

        self._parallel_pipeline_host = None
        self._fpie_inference_test = None
        self._fpie_temp_dir = None

    def get_model(self):
        # model
        model_type = self.config['model']['model_type']
        Model = get_model_class(model_type=model_type)

        model = Model(
            audioenc_name = self.config['model']['encoder']['audioenc_name'],
            d_in = self.config['model']['encoder']['out_emb'],
            text_decoder = self.config['model']['decoder']['text_decoder'],
            prefix_length = self.config['model']['decoder']['prefix_length'],
            freeze_text_decoder_weights = self.config['model']['decoder']['freeze_gpt_weights'],
            d_out = self.config['model']['encoder']['d_proj'],
            use_pretrained_audioencoder = self.config['model']['encoder']['use_pretrained_audioencoder'],
            freeze_audio_encoder_weights= self.config['model']['encoder']['freeze_audio_encoder_weights'],
            pretrained_audioencoder_path = self.config['model']['encoder']['pretrained_audioencoder_path'],
            model_variant = self.config['model'].get('model_variant', 'legacy'),
            adapter_config = self.config['model'].get('adapter', {}),
            decoder_config = self.config['model'].get('decoder', {}),
            encoder_config = self.config['model'].get('encoder', {}),
        )
        
        return model

    def _unwrap_model(self, model):
        return model.module if hasattr(model, "module") else model

    def _extract_checkpoint_state(self, checkpoint):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
            return checkpoint["state_dict"]
        return checkpoint

    def _checkpoint_key_candidates(self, key):
        if not key.startswith("caption_decoder.lm."):
            return []

        suffix = key[len("caption_decoder.lm."):]
        peft_key = f"caption_decoder.lm.base_model.model.{suffix}"
        candidates = [peft_key]

        if peft_key.endswith(".weight"):
            base_layer_key = peft_key[:-len(".weight")] + ".base_layer.weight"
            candidates.append(base_layer_key)

        return candidates

    def _is_allowed_missing_checkpoint_key(self, key):
        optional_prefixes = (
            "caption_decoder.prefix_adapter.listen_memory_ln.",
            "caption_decoder.prefix_adapter.listen_query_ln.",
            "caption_decoder.prefix_adapter.listen_memory_attn.",
            "caption_decoder.prefix_adapter.listen_token_to_gate.",
        )
        return key.startswith(optional_prefixes)

    def _prepare_checkpoint_state_for_model(self, model, checkpoint_state):
        model_state = self._unwrap_model(model).state_dict()
        prepared_state = {}
        remapped_keys = []

        for key, value in checkpoint_state.items():
            load_key = key

            if key not in model_state:
                for candidate in self._checkpoint_key_candidates(key):
                    candidate_tensor = model_state.get(candidate)
                    if candidate_tensor is None:
                        continue
                    if tuple(candidate_tensor.shape) != tuple(value.shape):
                        continue
                    load_key = candidate
                    remapped_keys.append((key, candidate))
                    break

            prepared_state[load_key] = value

        return prepared_state, remapped_keys

    def _load_checkpoint_into_model(self, model, checkpoint_path, strict):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_state = self._extract_checkpoint_state(checkpoint)
        checkpoint_state, remapped_keys = self._prepare_checkpoint_state_for_model(
            model, checkpoint_state
        )
        incompatible = model.load_state_dict(checkpoint_state, strict=False)
        allowed_missing_keys = [
            key for key in incompatible.missing_keys
            if self._is_allowed_missing_checkpoint_key(key)
        ]
        disallowed_missing_keys = [
            key for key in incompatible.missing_keys
            if not self._is_allowed_missing_checkpoint_key(key)
        ]
        if strict and (disallowed_missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                "Error(s) in loading state_dict for Mellow:\n"
                f"\tMissing key(s): {disallowed_missing_keys}\n"
                f"\tUnexpected key(s): {incompatible.unexpected_keys}"
            )

        if self.distributed.rank() == 0:
            self.logger.info(
                "Loaded checkpoint %s strict=%s remapped_lm_keys=%d allowed_missing_keys=%d",
                checkpoint_path,
                strict,
                len(remapped_keys),
                len(allowed_missing_keys),
            )
            if remapped_keys:
                self.logger.info("Sample remapped checkpoint keys: %s", remapped_keys[:8])
            if disallowed_missing_keys:
                self.logger.warning(
                    "Missing checkpoint keys (%d): %s",
                    len(disallowed_missing_keys),
                    disallowed_missing_keys[:20],
                )
            if allowed_missing_keys:
                self.logger.info(
                    "Allowed missing checkpoint keys (%d): %s",
                    len(allowed_missing_keys),
                    allowed_missing_keys[:20],
                )
            if incompatible.unexpected_keys:
                self.logger.warning(
                    "Unexpected checkpoint keys (%d): %s",
                    len(incompatible.unexpected_keys),
                    incompatible.unexpected_keys[:20],
                )

        return checkpoint

    def _to_float(self, value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _get_adapter_alpha(self, model):
        return self._to_float(self._unwrap_model(model).get_adapter_alpha())

    def _get_adapter_stats(self, model):
        stats = self._unwrap_model(model).get_adapter_stats()
        if stats is None:
            return None
        return {k: self._to_float(v) for k, v in stats.items()}

    def _format_adapter_stats(self, stats):
        priority_keys = [
            "alpha",
            "impact_ratio",
            "scaled_delta_norm",
            "z0_norm",
            "token_gate_mean",
            "token_gate_std",
            "token_gate_max",
            "listen_state_norm",
            "listen_state_std",
            "listen_tokens_norm",
            "listen_tokens_std",
            "pass2_minus_pass1_audio_norm",
            "pass2_minus_pass1_audio_ratio",
        ]
        ordered_keys = [key for key in priority_keys if key in stats]
        ordered_keys.extend(sorted(key for key in stats if key not in set(priority_keys)))
        return " ".join(f"{key}={stats[key]:.8e}" for key in ordered_keys)

    def _write_trainable_params(self, model):
        trainable = [
            (name, int(param.numel()))
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        fpath = os.path.join(self.config["save_dir"], "trainable_params.txt")
        with open(fpath, "w") as f:
            total = sum(count for _, count in trainable)
            f.write(f"total_trainable_params: {total}\n")
            for name, count in trainable:
                f.write(f"{name}\t{count}\n")
        self.logger.info("Wrote trainable parameter list to %s", fpath)

    def _build_variant_optimizer_params(self, model, weight_decay):
        optimizer_cfg = self.config["train"]["optimizer"]
        base_lr = float(optimizer_cfg["learning_rate"])
        lr_adapter = float(optimizer_cfg.get("lr_adapter", base_lr))
        lr_map = float(optimizer_cfg.get("lr_map", base_lr))
        lr_lora = float(optimizer_cfg.get("lr_lora", base_lr))
        lr_ced = float(optimizer_cfg.get("lr_ced", base_lr))
        exclude_bias_bn = self.config.get("exclude_bias_bn_from_weight_decay", False)

        groups = {}
        group_names = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_name = name[len("module."):] if name.startswith("module.") else name

            if "prefix_adapter" in param_name:
                role = "adapter"
                lr = lr_adapter
            elif "lora_" in param_name:
                role = "lora"
                lr = lr_lora
            elif param_name.startswith("audio_encoder.base.ced."):
                role = "ced"
                lr = lr_ced
            elif (
                "audio_encoder.projection" in param_name
                or "audio_encoder.base.c2l" in param_name
                or "audio_encoder.base.mapper" in param_name
                or "single20_" in param_name
            ):
                role = "map"
                lr = lr_map
            else:
                role = "default"
                lr = base_lr

            param_weight_decay = weight_decay
            if exclude_bias_bn and (param.ndim <= 1 or name.endswith(".bias")):
                param_weight_decay = 0.0

            key = (role, lr, param_weight_decay)
            groups.setdefault(key, []).append(param)
            group_names.setdefault(key, []).append(name)

        if not groups:
            raise ValueError(
                "No trainable parameters found. Check model.model_variant and freeze settings."
            )

        params = []
        for (role, lr, param_weight_decay), group_params in sorted(groups.items()):
            names = group_names[(role, lr, param_weight_decay)]
            self.logger.info(
                "optimizer group role=%s lr=%s weight_decay=%s params=%s",
                role, lr, param_weight_decay, names
            )
            params.append({
                "params": group_params,
                "lr": lr,
                "weight_decay": param_weight_decay,
            })
        return params

    @staticmethod
    def _is_step_lr_scheduler(name):
        return name in {"warmup_cosine", "two_phase_decay"}

    def _build_step_lr_scheduler(self, optimizer, total_steps: int):
        optimizer_cfg = self.config["train"]["optimizer"]
        schedule = optimizer_cfg["scheduler"]
        if not self._is_step_lr_scheduler(schedule):
            return None

        inferred_total_steps = max(1, int(total_steps))
        total_steps = resolve_step_lr_schedule_total_steps(
            optimizer_cfg.get("schedule_total_optimizer_steps"),
            inferred_total_steps,
            bool(optimizer_cfg.get("allow_schedule_horizon_beyond_run", False)),
        )
        base_lr = float(optimizer_cfg["learning_rate"])
        min_lr = float(optimizer_cfg.get("min_lr", 0.0))
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        if schedule == "warmup_cosine":
            warmup_steps = int(optimizer_cfg.get("warmup_steps", 0) or 0)
            if warmup_steps <= 0:
                warmup_ratio = float(optimizer_cfg.get("warmup_ratio", 0.0) or 0.0)
                warmup_steps = int(round(total_steps * warmup_ratio))
            warmup_steps = max(0, min(warmup_steps, total_steps - 1))

            def lr_lambda(step: int):
                if warmup_steps > 0 and step < warmup_steps:
                    return max(min_factor, float(step + 1) / float(warmup_steps))
                denom = max(1, total_steps - warmup_steps)
                progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

        elif schedule == "two_phase_decay":
            decay_start_ratio = float(optimizer_cfg.get("decay_start_ratio", 0.8))
            decay_start = int(round(total_steps * decay_start_ratio))
            decay_start = max(0, min(decay_start, total_steps - 1))

            def lr_lambda(step: int):
                if step < decay_start:
                    return 1.0
                denom = max(1, total_steps - decay_start)
                progress = min(1.0, max(0.0, float(step - decay_start) / float(denom)))
                return 1.0 + (min_factor - 1.0) * progress

        else:
            raise ValueError(f"No such step lr schedule: {schedule}")

        if self.distributed.rank() == 0:
            self.logger.info(
                "Using step-level LR scheduler=%s total_steps=%d inferred_training_steps=%d "
                "base_lr=%g min_lr=%g",
                schedule,
                total_steps,
                inferred_total_steps,
                base_lr,
                min_lr,
            )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _get_training_checkpoint_name(
        self,
        epoch: int,
        next_batch_index: int,
        completed_epoch: bool = False,
        optimizer_step: int | None = None,
    ):
        prefix = self.config.get("myconfig")
        prefix = prefix + '-' if prefix else ''
        if completed_epoch:
            return f"{prefix}-epo-{epoch + 1}-complete.ckpt"
        if optimizer_step is not None and self.config["train"].get("checkpoint_name_by_optimizer_step", False):
            return f"{prefix}-optstep-{optimizer_step:06d}.ckpt"
        return f"{prefix}-epo-{epoch + 1}-batch-{next_batch_index:06d}.ckpt"

    def _get_checkpoint_batch_indices(self, num_batches_per_epoch: int):
        fraction = float(self.config["train"].get("checkpoint_interval_fraction", 0.5) or 0.0)
        if fraction <= 0.0:
            return set()
        if fraction >= 1.0:
            return set()

        checkpoint_batches = set()
        multiplier = 1
        while True:
            next_batch_index = int(math.ceil(num_batches_per_epoch * fraction * multiplier))
            if next_batch_index >= num_batches_per_epoch:
                break
            if next_batch_index > 0:
                checkpoint_batches.add(next_batch_index)
            multiplier += 1
        return checkpoint_batches

    def _get_checkpoint_optimizer_steps(self):
        raw_steps = self.config["train"].get("checkpoint_optimizer_steps", [])
        if raw_steps is None:
            return set()
        if isinstance(raw_steps, str):
            raw_steps = [part.strip() for part in raw_steps.split(",") if part.strip()]
        return {int(step) for step in raw_steps if int(step) > 0}

    def _write_latest_training_state_pointer(self, training_state_path: str):
        training_state_path = str(Path(training_state_path).resolve())
        save_dir = Path(self.config["save_dir"]).resolve()
        pointer_dirs = [save_dir]
        if save_dir.parent != save_dir:
            pointer_dirs.append(save_dir.parent)

        for pointer_dir in pointer_dirs:
            try:
                pointer_dir.mkdir(parents=True, exist_ok=True)
                pointer_path = pointer_dir / "latest_training_state.txt"
                tmp_path = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
                tmp_path.write_text(training_state_path + "\n")
                os.replace(tmp_path, pointer_path)
            except Exception:
                self.logger.warning(
                    "Failed to update latest training-state pointer in %s",
                    pointer_dir,
                    exc_info=True,
                )

    @retry
    def _save_training_state(
        self,
        training_state_fpath,
        model,
        optimizer,
        lr_scheduler,
        grad_scaler,
        grad_norm_tracker,
        *,
        epoch: int,
        next_batch_index: int,
        total_step: int,
        num_batches_per_epoch: int,
        completed_epoch: bool,
    ):
        payload = {
            "state_dict": self.distributed.get_distributed_model_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": grad_scaler.state_dict(),
            "grad_norm_tracker_state_dict": grad_norm_tracker.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "training_state": {
                "resume_contract_version": 1,
                "epoch": int(epoch),
                "next_batch_index": int(next_batch_index),
                "total_step": int(total_step),
                "num_batches_per_epoch": int(num_batches_per_epoch),
                "completed_epoch": bool(completed_epoch),
                "checkpoint_kind": "epoch_complete" if completed_epoch else "mid_epoch",
            },
            "config": training_state_config_contract(self.config),
        }
        if lr_scheduler is not None:
            payload["lr_scheduler_state_dict"] = lr_scheduler.state_dict()

        tmp_training_state_fpath = f"{training_state_fpath}.tmp"
        with open(tmp_training_state_fpath, "wb") as f:
            torch.save(payload, f)
            f.flush()
        os.replace(tmp_training_state_fpath, training_state_fpath)
        self._write_latest_training_state_pointer(training_state_fpath)

    def _restore_training_state_from_checkpoint(
        self,
        checkpoint,
        optimizer,
        lr_scheduler,
        grad_scaler,
        grad_norm_tracker,
    ):
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("training_state"), dict):
            if self.distributed.rank() == 0:
                self.logger.info(
                    "Resume checkpoint is model-only; optimizer/scheduler/scaler were not restored."
                )
            return 0, 0, 0

        saved_config = checkpoint.get("config")
        current_config = training_state_config_contract(self.config)
        mismatches = strict_resume_config_mismatches(saved_config, current_config)
        if mismatches:
            raise RuntimeError(
                "Refusing non-identical full-state resume; checkpoint/config mismatches: "
                + json.dumps(mismatches, sort_keys=True, default=str)
            )

        required_full_state = {
            "optimizer_state_dict",
            "grad_scaler_state_dict",
            "grad_norm_tracker_state_dict",
            "rng_state",
        }
        if lr_scheduler is not None:
            required_full_state.add("lr_scheduler_state_dict")
        missing_full_state = sorted(required_full_state - set(checkpoint))
        if missing_full_state:
            raise RuntimeError(
                "Full-state resume checkpoint lacks required fields: "
                f"{missing_full_state}"
            )

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        grad_norm_tracker.load_state_dict(checkpoint["grad_norm_tracker_state_dict"])
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, dict):
            raise RuntimeError("Full-state resume checkpoint lacks rng_state")
        required_rng = {"python", "numpy", "torch_cpu", "torch_cuda_all"}
        missing_rng = sorted(required_rng - set(rng_state))
        if missing_rng:
            raise RuntimeError(f"Full-state resume checkpoint lacks RNG fields: {missing_rng}")
        self._pending_resume_rng_state = rng_state

        training_state = checkpoint["training_state"]
        if int(training_state.get("resume_contract_version", -1)) != 1:
            raise RuntimeError(
                "Unsupported full-state resume contract version: "
                f"{training_state.get('resume_contract_version')}"
            )
        epoch = int(training_state.get("epoch", 0))
        next_batch_index = int(training_state.get("next_batch_index", 0))
        total_step = int(training_state.get("total_step", 0))
        completed_epoch = bool(training_state.get("completed_epoch", False))
        if completed_epoch:
            start_epoch = epoch + 1
            resume_batch_index = 0
        else:
            start_epoch = epoch
            resume_batch_index = max(0, next_batch_index)

        if self.distributed.rank() == 0:
            self.logger.info(
                "Restored full training state: start_epoch=%d resume_batch_index=%d total_step=%d completed_epoch=%s",
                start_epoch + 1,
                resume_batch_index,
                total_step,
                completed_epoch,
            )

        return start_epoch, resume_batch_index, total_step

    def _restore_pending_resume_rng_state(self) -> bool:
        pending_rng_state = getattr(self, "_pending_resume_rng_state", None)
        if pending_rng_state is None:
            return False

        random.setstate(pending_rng_state["python"])
        np.random.set_state(pending_rng_state["numpy"])
        # Full checkpoints are loaded with ``map_location=self.device`` so the
        # CPU RNG tensor is moved to CUDA together with model/optimizer state.
        # PyTorch's CPU generator accepts only a CPU ByteTensor.  Normalize the
        # saved RNG payload explicitly instead of depending on load placement.
        cpu_rng_state = pending_rng_state["torch_cpu"]
        if not isinstance(cpu_rng_state, torch.Tensor):
            raise RuntimeError("Saved torch_cpu RNG state is not a tensor")
        cpu_rng_state = cpu_rng_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            cuda_states = pending_rng_state["torch_cuda_all"]
            if len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError(
                    "CUDA RNG state count does not match visible devices: "
                    f"saved={len(cuda_states)} current={torch.cuda.device_count()}"
                )
            normalized_cuda_states = []
            for index, cuda_state in enumerate(cuda_states):
                if not isinstance(cuda_state, torch.Tensor):
                    raise RuntimeError(
                        f"Saved CUDA RNG state {index} is not a tensor"
                    )
                normalized_cuda_states.append(
                    cuda_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
                )
            torch.cuda.set_rng_state_all(normalized_cuda_states)
        self._pending_resume_rng_state = None
        if self.distributed.rank() == 0:
            self.logger.info(
                "Restored Python/NumPy/Torch RNG states before the first resumed batch"
            )
        return True

    def _install_training_signal_handlers(self):
        stop_requested = {"value": False}
        previous_handlers = {}
        enabled = os.environ.get("MELLOW_ENABLE_SIGNAL_CHECKPOINTS", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            if self.distributed.rank() == 0:
                self.logger.info("Signal-triggered training-state checkpoints are disabled.")
            return stop_requested, previous_handlers

        def _handle_stop_signal(signum, _frame):
            stop_requested["value"] = True
            if self.distributed.rank() == 0:
                self.logger.warning(
                    "Received signal %s; will save training state after the current batch.",
                    signum,
                )

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_stop_signal)

        return stop_requested, previous_handlers

    def _restore_signal_handlers(self, previous_handlers):
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    def _distributed_stop_requested(self, stop_requested):
        value = torch.tensor(
            1.0 if stop_requested["value"] else 0.0,
            device=self.device,
            dtype=torch.float32,
        )
        reduced = self.distributed.all_reduce(value, average=False)
        return bool(reduced.item() > 0)

    def get_num_data_workers(self):
        if self.config["train"]["num_workers"] is not None:
            # backward compatibility
            return self.config["train"]["num_workers"]

        num_workers = os.cpu_count() * self.config["num_data_workers_per_cpu"]
        num_workers /= self.distributed.world_size()
        num_workers = math.ceil(num_workers - 0.05)
        return num_workers

    def _cleanup_worker_log_sink(self):
        sink = self._worker_log_sink
        if sink is None:
            return

        self._worker_log_sink = None
        sink.close()

    def get_worker_logging_initializer(self):
        sink = self._worker_log_sink
        if sink is None:
            from .log import WorkerLogSink
            self._worker_log_sink = sink = WorkerLogSink()
            self._cleanup_hook_list.append(self._cleanup_worker_log_sink)

        sink.start()
        return sink.init_worker

    def _get_data_worker_init_fn(self):
        return functools.partial(worker_init_fn, self.get_worker_logging_initializer())

    def _cleanup_parallel_pipeline_host(self):
        parallel_pipeline_host = self._parallel_pipeline_host
        if parallel_pipeline_host is None:
            return

        self._parallel_pipeline_host = None
        parallel_pipeline_host.close()

    def get_parallel_pipeline_host(self):
        assert self._is_cleanup_enabled > 0
        host = self._parallel_pipeline_host
        if host is None:
            from parallel_pipeline import create_parallel_pipeline_host
            host = create_parallel_pipeline_host(self.get_num_data_workers(),
                                                 initializer=self.get_worker_logging_initializer())

            host.configure_thread_pool("cognitive", self.config['wer_eval']['cognitive']['threads'])
            self._cleanup_hook_list.append(self._cleanup_parallel_pipeline_host)
            self._parallel_pipeline_host = host

        return host

    def __enter__(self):
        self._is_cleanup_enabled += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._is_cleanup_enabled > 0
        self._is_cleanup_enabled -= 1
        if self._is_cleanup_enabled > 0:
            return

        cleanup_hook_list = self._cleanup_hook_list
        self._cleanup_hook_list = list()
        for hook in reversed(cleanup_hook_list):
            # noinspection PyBroadException
            try:
                hook()
            except Exception:
                logging.warning("Exception while runinning cleanup hook", exc_info=True, stack_info=True)
                
    def get_data(self, key):
        # creating and return dataset + sampler
        sampling_rate = self.config['data']['sampling_rate']
        segment_seconds = self.config['data']['segment_seconds']
        datafiles = self.config['data'][key]
        data_path = self.config['data']['datapath']
        sampling_rate = self.config['data']['sampling_rate']
        tokenizer_type = self.config['data']['tokenizer_type']
        ip_text_len = self.config['data']['ip_text_len']
        op_text_len = self.config['data']['op_text_len']
        audio_mode = self.config["data"].get("audio_mode", "normal")
        audio_length_policy = self.config["data"].get("audio_length_policy", "fixed")
        ced_hidden_cache = self.config["data"].get("ced_hidden_cache")
        beats_hidden_cache = self.config["data"].get("beats_hidden_cache")
        alignkd_teacher_cache = self.config["data"].get("alignkd_teacher_cache")
        sequence_retention_cache = self.config["data"].get(
            "sequence_retention_cache"
        )
        num_workers = self.get_num_data_workers()

        if self.config["mode"] is TrainerMode.Train:
            from data.audiotext_dataset import AudioTextDataset, collate_fn

            naive_kd_config = self.config["train"].get("naive_kd", {}) or {}
            naive_kd_target_scores_key = validate_naive_kd_scores_key(
                naive_kd_config.get("target_scores_key")
            )
            dataset = AudioTextDataset(
                data_path=data_path,
                datafiles=datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                tokenizer_type=tokenizer_type,
                ip_text_len=ip_text_len,
                op_text_len=op_text_len,
                audio_mode=audio_mode,
                audio_length_policy=audio_length_policy,
                ced_hidden_cache=ced_hidden_cache,
                beats_hidden_cache=beats_hidden_cache,
                alignkd_teacher_cache=alignkd_teacher_cache,
                sequence_retention_cache=sequence_retention_cache,
                naive_kd_target_scores_key=naive_kd_target_scores_key,
                output_control_regions=bool(
                    (self.config["train"].get("output_control_warmup", {}) or {}).get(
                        "enabled", False
                    )
                ),
                mixture_config=self.config["data"].get("mixture", {}),
                order_config=self.config["data"].get("order", {}),
            )

            order_config = self.config["data"].get("order", {}) or {}
            sampler_seed = int(order_config.get("sampler_seed", 0))
            sampler_shuffle = validate_train_sampler_shuffle(order_config)
            train_drop_last = validate_train_drop_last(self.config["data"])
            configured_world_size = int(
                self.config["train"].get(
                    "world_size", self.distributed.world_size()
                )
            )
            actual_world_size = int(self.distributed.world_size())
            validate_train_loader_world_size(
                configured_world_size, actual_world_size, train_drop_last
            )
            data_sampler = CustomDistributedSampler(
                        dataset, shuffle=sampler_shuffle,
                        num_replicas=self.distributed.world_size(), 
                        rank=self.distributed.rank(), 
                        drop_last=train_drop_last,
                        seed=sampler_seed,
                        )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config["train"]["batch_size"], num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True, sampler=data_sampler,
                drop_last=train_drop_last, worker_init_fn=self._get_data_worker_init_fn(),
                persistent_workers=self.config["train"]["persistent_data_workers"] if num_workers > 0 else False,
            )
        elif self.config["mode"] is TrainerMode.EvaluateCheckpoint:
            from data.audiotext_eval_dataset import AudioTextEvalDataset, collate_fn
            dataset = AudioTextEvalDataset(
                data_path=data_path,
                datafiles=datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                tokenizer_type=tokenizer_type,
                ip_text_len=ip_text_len,
                op_text_len=op_text_len,
                audio_mode=audio_mode,
                audio_length_policy=audio_length_policy,
                ced_hidden_cache=ced_hidden_cache,
                beats_hidden_cache=beats_hidden_cache,
            )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config["train"]["batch_size"], num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True,
                drop_last=False, worker_init_fn=self._get_data_worker_init_fn(),
                persistent_workers=self.config["train"]["persistent_data_workers"] if num_workers > 0 else False,
            )

            data_sampler = None
        else:
            mode = self.config["mode"]
            raise ValueError(f"{mode} dataloader mode not supported'")
        
        return dataset, data_sampler, data_loader

    def get_masked_kd_side_data(
        self,
        masked_kd_config,
        *,
        datafiles_key="side_datafiles",
        cache_root_key="real_cache_root",
        cache_policy_key="real_cache_policy",
        cache_point_key="real_cache_point",
        view_name="real",
    ):
        side_datafiles = masked_kd_config.get(datafiles_key) or []
        if isinstance(side_datafiles, str):
            side_datafiles = [side_datafiles]
        if not side_datafiles:
            raise ValueError(f"train.masked_kd.{datafiles_key} must contain at least one JSON file")

        data_config = self.config["data"]
        sampling_rate = data_config["sampling_rate"]
        segment_seconds = data_config["segment_seconds"]
        data_path = data_config["datapath"]
        tokenizer_type = data_config["tokenizer_type"]
        ip_text_len = data_config["ip_text_len"]
        op_text_len = data_config["op_text_len"]
        audio_mode = data_config.get("audio_mode", "normal")
        audio_length_policy = data_config.get("audio_length_policy", "fixed")
        ced_hidden_cache = data_config.get("ced_hidden_cache")
        if cache_root_key and masked_kd_config.get(cache_root_key):
            ced_hidden_cache = {
                "root": masked_kd_config[cache_root_key],
                "policy": masked_kd_config.get(cache_policy_key, "fixed20"),
                "cache_point": masked_kd_config.get(cache_point_key, "frozen_ced_hidden_before_mapper"),
            }
        if masked_kd_config.get("beats_cache_root"):
            raise ValueError("train.masked_kd.beats_cache_root is not supported for side-batch MaskedKD yet")

        rho = float(masked_kd_config.get("rho", 0.2) or 0.2)
        default_side_batch = max(1, int(round(float(self.config["train"]["batch_size"]) * rho)))
        side_batch_size = int(masked_kd_config.get("side_batch_size", default_side_batch) or default_side_batch)
        if side_batch_size <= 0:
            raise ValueError(f"train.masked_kd.side_batch_size must be > 0, got {side_batch_size}")

        from data.audiotext_dataset import AudioTextDataset, collate_fn

        order_config = dict(data_config.get("order", {}) or {})
        order_config.update(masked_kd_config.get("side_order", {}) or {})
        order_config.setdefault("shuffle", True)
        order_config.setdefault("seed", int(masked_kd_config.get("side_seed", 20260617)))
        order_config.setdefault("sampler_seed", int(masked_kd_config.get("side_sampler_seed", 20260617)))
        order_config.setdefault("rule", f"maskedkd_side_batch_{view_name}")

        dataset = AudioTextDataset(
            data_path=data_path,
            datafiles=side_datafiles,
            sampling_rate=sampling_rate,
            max_clip_len=segment_seconds,
            tokenizer_type=tokenizer_type,
            ip_text_len=ip_text_len,
            op_text_len=op_text_len,
            audio_mode=audio_mode,
            audio_length_policy=audio_length_policy,
            ced_hidden_cache=ced_hidden_cache,
            beats_hidden_cache=None,
            alignkd_teacher_cache=None,
            mixture_config={},
            order_config=order_config,
        )
        data_sampler = CustomDistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=self.distributed.world_size(),
            rank=self.distributed.rank(),
            drop_last=True,
            seed=int(order_config.get("sampler_seed", 0)),
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=side_batch_size,
            num_workers=self.get_num_data_workers(),
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=data_sampler,
            drop_last=True,
            worker_init_fn=self._get_data_worker_init_fn(),
            persistent_workers=self.config["train"]["persistent_data_workers"] if self.get_num_data_workers() > 0 else False,
        )
        if len(data_loader) <= 0:
            raise ValueError(
                f"MaskedKD side DataLoader has no batches: rows={len(dataset)} "
                f"view={view_name} world_size={self.distributed.world_size()} batch_size={side_batch_size}"
            )
        return dataset, data_sampler, data_loader

    def _make_train_input_dict_from_batch(self, batch_data_dict):
        batch_audio1 = batch_data_dict["waveform1"]
        batch_audio2 = batch_data_dict["waveform2"]
        batch_input = batch_data_dict["input"]
        batch_answer = dict(batch_data_dict["answer"])
        batch_answer["attention_mask"] = torch.stack([
            torch.cat((torch.ones(self.config["model"]["decoder"]["total_prefix_length"]), text), dim=0)
            for text in batch_answer["attention_mask"]
        ])

        input_dict = {
            "audio1": batch_audio1,
            "audio2": batch_audio2,
            "input": batch_input,
            "answer": batch_answer,
        }
        if "waveform1_lengths" in batch_data_dict:
            input_dict["audio1_lengths"] = batch_data_dict["waveform1_lengths"]
        if "waveform2_lengths" in batch_data_dict:
            input_dict["audio2_lengths"] = batch_data_dict["waveform2_lengths"]
        if "ced_hidden" in batch_data_dict:
            input_dict["ced_hidden"] = batch_data_dict["ced_hidden"]
        if "ced_hidden_segment_lengths" in batch_data_dict:
            input_dict["ced_hidden_segment_lengths"] = batch_data_dict["ced_hidden_segment_lengths"]
        if "ced_hidden_lengths" in batch_data_dict:
            input_dict["ced_hidden_lengths"] = batch_data_dict["ced_hidden_lengths"]
        if "beats_hidden" in batch_data_dict:
            input_dict["beats_hidden"] = batch_data_dict["beats_hidden"]
        if "beats_hidden_segment_lengths" in batch_data_dict:
            input_dict["beats_hidden_segment_lengths"] = batch_data_dict["beats_hidden_segment_lengths"]
        if "beats_hidden_lengths" in batch_data_dict:
            input_dict["beats_hidden_lengths"] = batch_data_dict["beats_hidden_lengths"]
        return LazyConversionDict(input_dict, lambda x: x.to(self.device))

    def train(self):
        self.logger.info("Training Mellow with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        # creating dataset
        dataset, data_sampler, data_loader = self.get_data("datafiles")   
        masked_kd_config = self.config["train"].get("masked_kd", {}) or {}
        masked_kd_enabled = bool(masked_kd_config.get("enabled", False))
        masked_kd_mode = str(masked_kd_config.get("mode", "side_batch"))
        if masked_kd_enabled and masked_kd_mode != "side_batch":
            raise ValueError(f"Only train.masked_kd.mode=side_batch is supported, got {masked_kd_mode}")
        masked_kd_side_dataset = None
        masked_kd_side_sampler = None
        masked_kd_side_loader = None
        masked_kd_mask_dataset = None
        masked_kd_mask_sampler = None
        masked_kd_mask_loader = None
        if masked_kd_enabled:
            masked_kd_side_dataset, masked_kd_side_sampler, masked_kd_side_loader = self.get_masked_kd_side_data(
                masked_kd_config,
                datafiles_key="side_datafiles",
                cache_root_key="real_cache_root",
                cache_policy_key="real_cache_policy",
                cache_point_key="real_cache_point",
                view_name="real",
            )
            if masked_kd_config.get("masked_side_datafiles"):
                masked_kd_mask_dataset, masked_kd_mask_sampler, masked_kd_mask_loader = self.get_masked_kd_side_data(
                    masked_kd_config,
                    datafiles_key="masked_side_datafiles",
                    cache_root_key="masked_cache_root",
                    cache_policy_key="masked_cache_policy",
                    cache_point_key="masked_cache_point",
                    view_name="masked",
                )
                if len(masked_kd_mask_dataset) != len(masked_kd_side_dataset):
                    raise ValueError(
                        "MaskedKD real/masked side datasets must have the same length, "
                        f"got real={len(masked_kd_side_dataset)} masked={len(masked_kd_mask_dataset)}"
                    )
            if self.distributed.rank() == 0:
                self.logger.info(
                    "MaskedKD real side-batch enabled: rows=%d batches_per_rank=%d rho=%s side_batch_size=%s files=%s",
                    len(masked_kd_side_dataset),
                    len(masked_kd_side_loader),
                    masked_kd_config.get("rho", 0.2),
                    masked_kd_side_loader.batch_size,
                    masked_kd_config.get("side_datafiles"),
                )
                if masked_kd_mask_loader is not None:
                    self.logger.info(
                        "MaskedKD masked side-batch enabled: rows=%d batches_per_rank=%d side_batch_size=%s files=%s",
                        len(masked_kd_mask_dataset),
                        len(masked_kd_mask_loader),
                        masked_kd_mask_loader.batch_size,
                        masked_kd_config.get("masked_side_datafiles"),
                    )
        start_epoch = 0
        if self.distributed.rank() == 0 and hasattr(dataset, "mixture_summary"):
            os.makedirs(self.config["save_dir"], exist_ok=True)
            mixture_fpath = os.path.join(self.config["save_dir"], "data_mixture_summary.json")
            with open(mixture_fpath, "w") as f:
                json.dump(dataset.mixture_summary, f, indent=2, sort_keys=True)
            self.logger.info("Wrote data mixture summary to %s", mixture_fpath)
        if self.distributed.rank() == 0 and hasattr(dataset, "data_order_summary"):
            os.makedirs(self.config["save_dir"], exist_ok=True)
            order_fpath = os.path.join(self.config["save_dir"], "data_order_summary.json")
            with open(order_fpath, "w") as f:
                json.dump(dataset.data_order_summary, f, indent=2, sort_keys=True)
            self.logger.info("Wrote data order summary to %s", order_fpath)

        # Construct NN model
        model = self.get_model()
        model = model.to(self.device)
        resume_checkpoint_payload = None
        if self.config["resume_checkpoint"] and self.config["resume_checkpoint"] != "":
            resume_checkpoint_payload = self._load_checkpoint_into_model(
                model,
                self.config["resume_checkpoint"],
                strict=False,
            )

        model = self.distributed.create_distributed_model(model)
        model.train()
        os.makedirs(self.config["save_dir"], exist_ok=True)

        if self.distributed.rank() == 0:
            self.logger.info("Mellow has %d parameters of which %d are trainable" % numparams(model))
            self.logger.info("%s", model)
            self._write_trainable_params(model)
            adapter_alpha = self._get_adapter_alpha(model)
            if adapter_alpha is not None:
                self.logger.info("adapter.alpha initial value: %.8f", adapter_alpha)

        # add weight decay to appropriate layers
        weight_decay = self.config["train"]["optimizer"]["weight_decay"]
        model_variant = self.config["model"].get("model_variant", "legacy")
        use_discriminative_lrs = bool(
            self.config["train"]["optimizer"].get("discriminative_lrs", False)
        )
        if model_variant == "legacy" and not use_discriminative_lrs:
            parameters = group_weight_decay_params(
                model,
                weight_decay=weight_decay,
                rnn_weight_decay=None,
                exclude_bias_bn_from_weight_decay=self.config.get("exclude_bias_bn_from_weight_decay", False)
            )
        else:
            parameters = self._build_variant_optimizer_params(model, weight_decay)

        optimizer_type = self.config["train"]["optimizer"]["optimizer_type"]
        optimizer = getattr(torch.optim, optimizer_type)(
            parameters,
            lr=float(self.config["train"]["optimizer"]["learning_rate"]), # * self.distributed.world_size(), # disabled multiplying the world size with the learning rate, this creates hardship for multi-node training
            weight_decay=weight_decay,
        )
        del parameters

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        # broadcast optimizer state to all other processes
        self.distributed.broadcast_optimizer_state(optimizer)
        # Train the model
        num_batches_per_epoch = len(data_loader)
        max_train_fraction = float(self.config["train"].get("max_train_fraction", 0.0) or 0.0)

        t0 = time.time()
        lowest_accerr_epo = 1000.0
        max_grad_norm = self.config["train"]["max_grad_norm"]
        grad_norm_tracker = GradNormTracker(initial_l2_norm=max_grad_norm, initial_max_norm=10 * max_grad_norm)
        listen_shuffle_margin_config = self.config["train"].get("listen_shuffle_margin", {}) or {}
        listen_shuffle_margin_enabled = bool(listen_shuffle_margin_config.get("enabled", False))
        listen_shuffle_margin_weight = float(listen_shuffle_margin_config.get("weight", 0.1))
        listen_shuffle_margin = float(listen_shuffle_margin_config.get("margin", 0.05))
        output_control_config = self.config["train"].get("output_control_warmup", {}) or {}
        output_control_enabled = bool(output_control_config.get("enabled", False))
        if output_control_enabled and listen_shuffle_margin_enabled:
            raise ValueError("output-control warmup cannot be mixed with listen-shuffle loss")
        gradient_accumulation_steps = int(self.config["train"].get("gradient_accumulation_steps", 1) or 1)
        if gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}")
        max_optimizer_updates = int(self.config["train"].get("max_optimizer_updates", 0) or 0)
        output_control_phases = (
            validate_output_control_weight_schedule(
                output_control_config,
                max_optimizer_updates,
            )
            if output_control_enabled
            else ()
        )
        checkpoint_optimizer_steps = self._get_checkpoint_optimizer_steps()
        naive_kd_config = self.config["train"].get("naive_kd", {}) or {}
        naive_kd_enabled = bool(naive_kd_config.get("enabled", False))
        naive_kd_lambda = float(naive_kd_config.get("lambda_kd", 0.0) or 0.0)
        naive_kd_tau = float(naive_kd_config.get("tau", 1.0) or 1.0)
        naive_kd_scores_key = validate_naive_kd_scores_key(
            naive_kd_config.get("target_scores_key")
        )
        naive_kd_active_until_optimizer_step_inclusive = validate_naive_kd_horizon(
            naive_kd_config.get("active_until_optimizer_step_inclusive")
        )
        naive_kd_enabled = naive_kd_enabled and naive_kd_lambda > 0.0
        naive_kd_option_token_ids = list(naive_kd_config.get("option_token_ids", []) or [])
        if naive_kd_enabled:
            if len(naive_kd_option_token_ids) != 4:
                raise ValueError("train.naive_kd.option_token_ids must contain four token ids for a/b/c/d")
            if naive_kd_tau <= 0:
                raise ValueError(f"train.naive_kd.tau must be > 0, got {naive_kd_tau}")
        naive_kd_option_token_ids_tensor = (
            torch.tensor(naive_kd_option_token_ids, device=self.device, dtype=torch.long)
            if naive_kd_enabled else None
        )
        retention_kd_config = self.config["train"].get("retention_kd", {}) or {}
        retention_kd_enabled = bool(retention_kd_config.get("enabled", False))
        retention_kd_lambda = float(
            retention_kd_config.get("lambda_kd", 0.0) or 0.0
        )
        retention_kd_tau = float(retention_kd_config.get("tau", 2.0) or 2.0)
        retention_kd_scores_key = validate_naive_kd_scores_key(
            retention_kd_config.get("target_scores_key", "ref_scores_abcd")
        )
        retention_kd_mask_key = str(
            retention_kd_config.get(
                "mask_key", "is_retention_kd_sample"
            )
        )
        retention_kd_option_token_ids = list(
            retention_kd_config.get(
                "option_token_ids", naive_kd_option_token_ids
            )
            or []
        )
        retention_kd_enabled = retention_kd_enabled and retention_kd_lambda > 0.0
        if retention_kd_enabled:
            if retention_kd_mask_key != "is_retention_kd_sample":
                raise ValueError(
                    "Round-4 retention KD requires "
                    "mask_key=is_retention_kd_sample"
                )
            if retention_kd_scores_key != "ref_scores_abcd":
                raise ValueError(
                    "Round-4 retention KD requires target_scores_key=ref_scores_abcd"
                )
            if len(retention_kd_option_token_ids) != 4:
                raise ValueError(
                    "train.retention_kd.option_token_ids must contain four token ids"
                )
            if retention_kd_tau <= 0.0:
                raise ValueError(
                    f"train.retention_kd.tau must be > 0, got {retention_kd_tau}"
                )
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "Retention KD global sample normalization requires "
                    "gradient_accumulation_steps=1"
                )
        retention_kd_option_token_ids_tensor = (
            torch.tensor(
                retention_kd_option_token_ids,
                device=self.device,
                dtype=torch.long,
            )
            if retention_kd_enabled
            else None
        )
        replay_config = self.config["train"].get("abc_replay_objective", {}) or {}
        replay_enabled = bool(replay_config.get("enabled", False))
        replay_dynamic_group_counts = bool(
            replay_config.get("dynamic_group_counts", False)
        )
        replay_tau = float(replay_config.get("tau", 2.0) or 2.0)
        replay_group_counts = tuple(
            int(value) for value in replay_config.get("group_counts", [32, 8, 8])
        )
        replay_group_names = tuple(
            str(value) for value in replay_config.get("group_names", ["a", "b", "c"])
        )
        replay_gradient_projection = validate_replay_gradient_projection(
            replay_config.get("gradient_projection"), replay_group_names
        )
        replay_gradient_projection_enabled = bool(
            replay_gradient_projection["enabled"]
        )
        replay_ce_weights = tuple(
            float(value) for value in replay_config.get("ce_weights", [1.0, 0.25, 0.25])
        )
        replay_kd_weights = tuple(
            float(value) for value in replay_config.get("kd_weights", [0.0, 0.5, 1.0])
        )
        replay_kd_target_roles = tuple(
            str(value)
            for value in replay_config.get(
                "kd_target_roles", ["none", "incumbent", "incumbent"]
            )
        )
        replay_target_scores_key = validate_replay_target_scores_key(
            replay_config.get("target_scores_key", "ref_scores_abcd")
        )
        replay_first_token_gold_ce_weights = (
            validate_replay_first_token_gold_ce_weights(
                replay_config.get("first_token_gold_ce_weights"),
                len(replay_group_counts),
            )
        )
        replay_first_token_gold_ce_enabled = any(
            weight > 0.0 for weight in replay_first_token_gold_ce_weights
        )
        replay_option_token_ids = tuple(
            int(value) for value in replay_config.get("option_token_ids", [81, 82, 83, 84])
        )
        if replay_gradient_projection_enabled and not replay_enabled:
            raise ValueError(
                "abc_replay_objective.gradient_projection requires abc_replay_objective.enabled=true"
            )
        if replay_enabled:
            if retention_kd_enabled or naive_kd_enabled or output_control_enabled:
                raise ValueError(
                    "abc_replay_objective cannot be combined with naive/retention KD or output-control CE"
                )
            if gradient_accumulation_steps != 1:
                raise ValueError("abc_replay_objective requires gradient_accumulation_steps=1")
            group_count = len(replay_group_counts)
            if group_count < 1:
                raise ValueError("abc_replay_objective requires at least one group")
            if (
                not replay_dynamic_group_counts
                and sum(replay_group_counts) != int(self.config["train"]["batch_size"])
            ):
                raise ValueError("abc_replay_objective group_counts must sum to batch_size")
            if any(value <= 0 for value in replay_group_counts):
                raise ValueError("abc_replay_objective group counts must be positive")
            if not (
                len(replay_group_names)
                == len(replay_ce_weights)
                == len(replay_kd_weights)
                == len(replay_kd_target_roles)
                == len(replay_first_token_gold_ce_weights)
                == group_count
            ):
                raise ValueError(
                    "abc_replay_objective group names/counts, CE/KD/first-token weights, "
                    "and KD target roles must align"
                )
            if (
                len(set(replay_group_names)) != group_count
                or any(not name.isidentifier() or name.lower() != name for name in replay_group_names)
            ):
                raise ValueError("abc_replay_objective group names must be unique lowercase identifiers")
            if any(not math.isfinite(value) or value < 0.0 for value in replay_ce_weights + replay_kd_weights):
                raise ValueError("abc_replay_objective weights must be finite and non-negative")
            allowed_target_roles = {"none", "teacher", "incumbent"}
            if any(role not in allowed_target_roles for role in replay_kd_target_roles):
                raise ValueError("abc_replay_objective has an unsupported KD target role")
            if any(
                (weight > 0.0) != (role != "none")
                for weight, role in zip(replay_kd_weights, replay_kd_target_roles)
            ):
                raise ValueError("every replay KD weight must match a non-none target role exactly")
            if not math.isfinite(replay_tau) or replay_tau <= 0.0:
                raise ValueError("abc_replay_objective tau must be finite and positive")
            if len(replay_option_token_ids) != 4:
                raise ValueError("abc_replay_objective requires four option token ids")
            if (
                replay_gradient_projection_enabled
                and self.distributed.world_size() != 1
            ):
                raise ValueError(
                    "abc_replay_objective.gradient_projection currently requires world_size=1"
                )
        replay_option_token_ids_tensor = (
            torch.tensor(replay_option_token_ids, device=self.device, dtype=torch.long)
            if replay_enabled
            else None
        )
        sequence_retention_config = (
            self.config["train"].get("sequence_retention_kd", {}) or {}
        )
        sequence_retention_enabled = bool(
            sequence_retention_config.get("enabled", False)
        )
        sequence_retention_lambda = float(
            sequence_retention_config.get("lambda_kd", 0.0) or 0.0
        )
        sequence_retention_tau = float(
            sequence_retention_config.get("tau", 2.0) or 2.0
        )
        sequence_retention_top_k = int(
            sequence_retention_config.get("top_k", 32) or 32
        )
        sequence_retention_enabled = (
            sequence_retention_enabled and sequence_retention_lambda > 0.0
        )
        if sequence_retention_enabled:
            if sequence_retention_tau <= 0.0:
                raise ValueError("train.sequence_retention_kd.tau must be > 0")
            if sequence_retention_top_k != 32:
                raise ValueError(
                    "Round-6 sequence retention requires top_k=32"
                )
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "Sequence-retention global sample normalization requires "
                    "gradient_accumulation_steps=1"
                )
            cache_config = self.config["data"].get(
                "sequence_retention_cache", {}
            ) or {}
            if not cache_config:
                raise ValueError(
                    "train.sequence_retention_kd.enabled=true requires "
                    "data.sequence_retention_cache"
                )
            if (
                int(cache_config.get("top_k", 32)) != sequence_retention_top_k
                or float(cache_config.get("tau", 2.0)) != sequence_retention_tau
            ):
                raise ValueError(
                    "Sequence-retention train/cache top_k and tau must match"
                )
        alignkd_config = self.config["train"].get("align_kd", {}) or {}
        alignkd_enabled = bool(alignkd_config.get("enabled", False))
        alignkd_attention_enabled = bool(
            alignkd_config.get("attention_enabled", False)
        )
        alignkd_attention_group_selection = str(
            alignkd_config.get("attention_group_selection", "all") or "all"
        ).lower()
        alignkd_feature_all_enabled = bool(
            alignkd_config.get("feature_all_enabled", False)
        )
        alignkd_feature_soft_enabled = bool(
            alignkd_config.get("feature_soft_enabled", False)
        )
        alignkd_feature_top16_enabled = bool(
            alignkd_config.get("feature_top16_enabled", False)
        )
        alignkd_lambda_attention = float(
            alignkd_config.get("lambda_attention", 0.05) or 0.0
        )
        alignkd_lambda_feature_all = float(
            alignkd_config.get("lambda_feature_all", 0.02) or 0.0
        )
        alignkd_lambda_feature_soft = float(
            alignkd_config.get(
                "lambda_feature_soft",
                alignkd_lambda_feature_all
                * float(alignkd_config.get("feature_soft_relative_weight", 0.1)),
            )
            or 0.0
        )
        alignkd_lambda_feature_top16 = float(
            alignkd_config.get(
                "lambda_feature_top16",
                alignkd_lambda_feature_all
                * float(alignkd_config.get("feature_top16_relative_weight", 0.1)),
            )
            or 0.0
        )
        alignkd_audio_mass_weight = float(
            alignkd_config.get("audio_mass_weight", 0.25)
        )
        alignkd_target_audio_tokens = int(
            alignkd_config.get("target_audio_tokens", 126)
        )
        if alignkd_enabled:
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "Audio-AlignKD global sample normalization currently requires "
                    "gradient_accumulation_steps=1"
                )
            if alignkd_target_audio_tokens != 126:
                raise ValueError(
                    "Audio-AlignKD v1 cache requires target_audio_tokens=126"
                )
            if not self.config["data"].get("alignkd_teacher_cache"):
                raise ValueError(
                    "train.align_kd.enabled=true requires data.alignkd_teacher_cache"
                )
            decoder_alignkd = (
                self.config["model"].get("decoder", {}).get("align_kd", {}) or {}
            )
            if not decoder_alignkd.get("enabled", False):
                raise ValueError(
                    "train.align_kd.enabled=true requires model.decoder.align_kd.enabled=true"
                )
            if bool(decoder_alignkd.get("attention_enabled", False)) != alignkd_attention_enabled:
                raise ValueError(
                    "train/model decoder AlignKD attention flags must match"
                )
            if self.config["model"]["decoder"].get("prefix_layout") != "single_audio":
                raise ValueError("Audio-AlignKD requires decoder.prefix_layout=single_audio")
            if alignkd_feature_soft_enabled and alignkd_feature_top16_enabled:
                raise ValueError("A4 soft focus and A5 hard top16 are mutually exclusive")
            if (
                alignkd_feature_soft_enabled or alignkd_feature_top16_enabled
            ) and not alignkd_feature_all_enabled:
                raise ValueError("focus feature KD requires feature_all_enabled=true")
            if alignkd_attention_enabled and alignkd_lambda_attention <= 0:
                raise ValueError("attention_enabled requires lambda_attention > 0")
            if alignkd_attention_group_selection not in {
                "all",
                "question_and_gold_option",
            }:
                raise ValueError(
                    "attention_group_selection must be all or "
                    "question_and_gold_option"
                )
            if alignkd_feature_all_enabled and alignkd_lambda_feature_all <= 0:
                raise ValueError("feature_all_enabled requires lambda_feature_all > 0")
            if alignkd_feature_soft_enabled and alignkd_lambda_feature_soft <= 0:
                raise ValueError("feature_soft_enabled requires positive focus weight")
            if alignkd_feature_top16_enabled and alignkd_lambda_feature_top16 <= 0:
                raise ValueError("feature_top16_enabled requires positive focus weight")
        elif any(
            (
                alignkd_attention_enabled,
                alignkd_feature_all_enabled,
                alignkd_feature_soft_enabled,
                alignkd_feature_top16_enabled,
            )
        ):
            raise ValueError("AlignKD loss flags require train.align_kd.enabled=true")
        gradient_probe_config = self.config["train"].get("kd_gradient_probe", {}) or {}
        gradient_probe_enabled = bool(gradient_probe_config.get("enabled", False))
        gradient_probe_interval = int(gradient_probe_config.get("interval", 10) or 0)
        gradient_probe_filename = str(
            gradient_probe_config.get("filename", "audio_alignkd_gradient_probes.jsonl")
        )
        gradient_probe_path = os.path.join(self.config["save_dir"], gradient_probe_filename)
        gradient_probe_mapper_params = []
        gradient_probe_layer0_qk_params = []
        if gradient_probe_enabled:
            if gradient_accumulation_steps != 1:
                raise ValueError("KD gradient probes require gradient_accumulation_steps=1")
            if self.distributed.world_size() != 1:
                raise ValueError(
                    "KD gradient probes are a canonical single-GPU smoke diagnostic"
                )
            if gradient_probe_interval <= 0:
                raise ValueError("train.kd_gradient_probe.interval must be positive")
            for parameter_name, parameter in model.named_parameters():
                normalized_name = (
                    parameter_name[len("module."):]
                    if parameter_name.startswith("module.")
                    else parameter_name
                )
                if "audio_encoder.base.mapper." in normalized_name:
                    gradient_probe_mapper_params.append(parameter)
                if (
                    "caption_decoder.lm.model.layers.0.self_attn.q_proj." in normalized_name
                    or "caption_decoder.lm.model.layers.0.self_attn.k_proj." in normalized_name
                ):
                    gradient_probe_layer0_qk_params.append(parameter)
            if not gradient_probe_mapper_params:
                raise ValueError(
                    "KD gradient probe could not find trainable CED mapper parameters"
                )
            if alignkd_attention_enabled and not gradient_probe_layer0_qk_params:
                raise ValueError(
                    "KD gradient probe could not find trainable layer-0 Q/K parameters"
                )
            if self.distributed.rank() == 0:
                Path(gradient_probe_path).parent.mkdir(parents=True, exist_ok=True)
                Path(gradient_probe_path).write_text("", encoding="utf-8")
                self.logger.info(
                    "KD gradient probe enabled: interval=%d mapper_params=%d "
                    "layer0_qk_params=%d output=%s",
                    gradient_probe_interval,
                    len(gradient_probe_mapper_params),
                    len(gradient_probe_layer0_qk_params),
                    gradient_probe_path,
                )
        masked_kd_rho = float(masked_kd_config.get("rho", 0.2) or 0.2) if masked_kd_enabled else 0.0
        masked_kd_lambda_real = float(masked_kd_config.get("lambda_real", 0.0) or 0.0) if masked_kd_enabled else 0.0
        masked_kd_lambda_mask = float(masked_kd_config.get("lambda_mask", 0.0) or 0.0) if masked_kd_enabled else 0.0
        masked_kd_lambda_drop = float(masked_kd_config.get("lambda_drop", 0.0) or 0.0) if masked_kd_enabled else 0.0
        masked_kd_tau = float(masked_kd_config.get("tau", 1.0) or 1.0) if masked_kd_enabled else 1.0
        masked_kd_normalize = bool(masked_kd_config.get("normalize_by_1_plus_rho", True))
        masked_kd_drop_positive_only = bool(masked_kd_config.get("drop_positive_only", True))
        masked_kd_drop_cap = float(masked_kd_config.get("drop_cap", 5.0) or 0.0) if masked_kd_enabled else 0.0
        masked_kd_drop_target_space = str(masked_kd_config.get("drop_target_space", "margin") or "margin").lower()
        masked_kd_drop_loss_type = str(masked_kd_config.get("drop_loss_type", "mse") or "mse").lower()
        masked_kd_drop_response_loss = str(
            masked_kd_config.get("drop_response_loss", "smooth_l1") or "smooth_l1"
        ).lower()
        masked_kd_drop_prob_tau = float(masked_kd_config.get("drop_prob_tau", masked_kd_tau) or masked_kd_tau)
        masked_kd_drop_response_tau = float(masked_kd_config.get("drop_response_tau", 2.0) or 2.0)
        masked_kd_drop_huber_beta = float(masked_kd_config.get("drop_huber_beta", 1.0) or 1.0)
        masked_kd_drop_hinge_margin = float(masked_kd_config.get("drop_hinge_margin", 0.0) or 0.0)
        masked_kd_drop_rank_margin_scale = float(masked_kd_config.get("drop_rank_margin_scale", 1.0) or 1.0)
        masked_kd_drop_order_min_gap = float(masked_kd_config.get("drop_order_min_gap", 0.0) or 0.0)
        masked_kd_drop_order_temperature = float(
            masked_kd_config.get("drop_order_temperature", 1.0) or 1.0
        )
        masked_kd_drop_min_teacher_drop = float(masked_kd_config.get("drop_min_teacher_drop", 0.0) or 0.0)
        masked_kd_drop_weight_power = float(masked_kd_config.get("drop_weight_power", 0.0) or 0.0)
        masked_kd_drop_teacher_drop_key = str(masked_kd_config.get("drop_teacher_drop_key", "") or "")
        masked_kd_drop_teacher_correct_key = str(masked_kd_config.get("drop_teacher_correct_key", "") or "")
        masked_kd_drop_teacher_real_scores_key = str(
            masked_kd_config.get("drop_teacher_real_scores_key", "") or ""
        )
        masked_kd_drop_teacher_masked_scores_key = str(
            masked_kd_config.get("drop_teacher_masked_scores_key", "") or ""
        )
        masked_kd_mask_teacher_scores_key = str(
            masked_kd_config.get("mask_teacher_scores_key", "") or ""
        )
        masked_kd_drop_real_teacher_must_be_correct = bool(
            masked_kd_config.get("drop_real_teacher_must_be_correct", False)
        )
        masked_kd_drop_gradient_cosine_interval = int(
            masked_kd_config.get("drop_gradient_cosine_interval", 0) or 0
        )
        masked_kd_require_paired_side_batches = bool(
            masked_kd_config.get("require_paired_side_batches", True)
        )
        masked_kd_option_token_ids = list(masked_kd_config.get("option_token_ids", []) or [])
        masked_kd_needs_option_scores = (
            masked_kd_lambda_real > 0.0 or masked_kd_lambda_mask > 0.0 or masked_kd_lambda_drop > 0.0
        )
        masked_kd_needs_masked_view = masked_kd_lambda_mask > 0.0 or masked_kd_lambda_drop > 0.0
        if masked_kd_enabled:
            if masked_kd_rho <= 0:
                raise ValueError(f"train.masked_kd.rho must be > 0, got {masked_kd_rho}")
            if masked_kd_tau <= 0:
                raise ValueError(f"train.masked_kd.tau must be > 0, got {masked_kd_tau}")
            if masked_kd_needs_option_scores and len(masked_kd_option_token_ids) != 4:
                raise ValueError("train.masked_kd.option_token_ids must contain four token ids for a/b/c/d")
            if masked_kd_needs_masked_view and masked_kd_mask_loader is None:
                raise ValueError(
                    "train.masked_kd.lambda_mask/lambda_drop > 0 requires masked_side_datafiles and masked_cache_root"
                )
            if masked_kd_drop_target_space == "probability":
                masked_kd_drop_target_space = "prob"
            masked_kd_drop_loss_type = {
                "probability_shift": "prob_shift",
                "probability_shift_distillation": "prob_shift",
                "delta_prob": "prob_shift",
                "delta_p": "prob_shift",
                "centered_logit": "centered_logit_response",
                "centered_logit_response_distillation": "centered_logit_response",
                "clr": "centered_logit_response",
                "pairwise_preference": "pairwise_preference_shift",
                "pairwise_preference_shift_distillation": "pairwise_preference_shift",
                "pps": "pairwise_preference_shift",
                "order_contrastive": "drop_order_contrastive",
                "order_logistic": "drop_order_contrastive",
                "pairwise_order": "drop_order_contrastive",
                "pairwise_order_logistic": "drop_order_contrastive",
                "listwise_contrastive": "drop_listwise_kl",
                "listwise_kl": "drop_listwise_kl",
                "drop_distribution": "drop_listwise_kl",
            }.get(masked_kd_drop_loss_type, masked_kd_drop_loss_type)
            masked_kd_drop_response_loss = {
                "smoothl1": "smooth_l1",
                "huber": "smooth_l1",
            }.get(masked_kd_drop_response_loss, masked_kd_drop_response_loss)
            if masked_kd_drop_target_space not in {"margin", "log_odds", "prob"}:
                raise ValueError(
                    "train.masked_kd.drop_target_space must be one of "
                    f"margin/log_odds/prob, got {masked_kd_drop_target_space}"
                )
            if masked_kd_drop_loss_type not in {
                "mse",
                "smooth_l1",
                "huber",
                "lower_bound_hinge",
                "pairwise_logistic",
                "adaptive_rank",
                "prob_shift",
                "centered_logit_response",
                "pairwise_preference_shift",
                "drop_order_contrastive",
                "drop_listwise_kl",
            }:
                raise ValueError(
                    "train.masked_kd.drop_loss_type must be one of "
                    "mse/smooth_l1/huber/lower_bound_hinge/"
                    "pairwise_logistic/adaptive_rank/prob_shift/"
                    "centered_logit_response/pairwise_preference_shift/"
                    "drop_order_contrastive/drop_listwise_kl, "
                    f"got {masked_kd_drop_loss_type}"
                )
            if masked_kd_drop_response_loss not in {"mse", "smooth_l1"}:
                raise ValueError(
                    "train.masked_kd.drop_response_loss must be one of "
                    f"mse/smooth_l1, got {masked_kd_drop_response_loss}"
                )
            if masked_kd_drop_prob_tau <= 0:
                raise ValueError(
                    f"train.masked_kd.drop_prob_tau must be > 0, got {masked_kd_drop_prob_tau}"
                )
            if masked_kd_drop_response_tau <= 0:
                raise ValueError(
                    "train.masked_kd.drop_response_tau must be > 0, "
                    f"got {masked_kd_drop_response_tau}"
                )
            if masked_kd_drop_huber_beta <= 0:
                raise ValueError(
                    f"train.masked_kd.drop_huber_beta must be > 0, got {masked_kd_drop_huber_beta}"
                )
            if masked_kd_drop_hinge_margin < 0:
                raise ValueError(
                    "train.masked_kd.drop_hinge_margin must be >= 0, "
                    f"got {masked_kd_drop_hinge_margin}"
                )
            if masked_kd_drop_rank_margin_scale < 0:
                raise ValueError(
                    "train.masked_kd.drop_rank_margin_scale must be >= 0, "
                    f"got {masked_kd_drop_rank_margin_scale}"
                )
            if masked_kd_drop_order_min_gap < 0:
                raise ValueError(
                    "train.masked_kd.drop_order_min_gap must be >= 0, "
                    f"got {masked_kd_drop_order_min_gap}"
                )
            if masked_kd_drop_order_temperature <= 0:
                raise ValueError(
                    "train.masked_kd.drop_order_temperature must be > 0, "
                    f"got {masked_kd_drop_order_temperature}"
                )
            if masked_kd_drop_min_teacher_drop < 0:
                raise ValueError(
                    "train.masked_kd.drop_min_teacher_drop must be >= 0, "
                    f"got {masked_kd_drop_min_teacher_drop}"
                )
            if masked_kd_drop_weight_power < 0:
                raise ValueError(
                    "train.masked_kd.drop_weight_power must be >= 0, "
                    f"got {masked_kd_drop_weight_power}"
                )
            if masked_kd_drop_gradient_cosine_interval < 0:
                raise ValueError(
                    "train.masked_kd.drop_gradient_cosine_interval must be >= 0, "
                    f"got {masked_kd_drop_gradient_cosine_interval}"
                )
        masked_kd_option_token_ids_tensor = (
            torch.tensor(masked_kd_option_token_ids, device=self.device, dtype=torch.long)
            if masked_kd_enabled and masked_kd_needs_option_scores else None
        )
        masked_kd_grad_cosine_params = (
            tuple(p for p in model.parameters() if p.requires_grad)
            if masked_kd_enabled and masked_kd_drop_gradient_cosine_interval > 0 else tuple()
        )

        def _loss_gradient_cosine(loss_a, loss_b):
            grads_a = torch.autograd.grad(
                loss_a,
                masked_kd_grad_cosine_params,
                retain_graph=True,
                allow_unused=True,
            )
            grads_b = torch.autograd.grad(
                loss_b,
                masked_kd_grad_cosine_params,
                retain_graph=True,
                allow_unused=True,
            )
            dot = torch.zeros((), device=self.device, dtype=torch.float32)
            norm_a_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            norm_b_sq = torch.zeros((), device=self.device, dtype=torch.float32)
            for grad_a, grad_b in zip(grads_a, grads_b):
                if grad_a is None or grad_b is None:
                    continue
                grad_a = grad_a.detach().float()
                grad_b = grad_b.detach().float()
                dot = dot + torch.sum(grad_a * grad_b)
                norm_a_sq = norm_a_sq + torch.sum(grad_a * grad_a)
                norm_b_sq = norm_b_sq + torch.sum(grad_b * grad_b)
            stats = torch.stack([dot, norm_a_sq, norm_b_sq])
            stats = self.distributed.all_reduce(stats, average=False)
            denom = torch.sqrt(stats[1]).clamp_min(1e-12) * torch.sqrt(stats[2]).clamp_min(1e-12)
            if bool(((stats[1] <= 0.0) | (stats[2] <= 0.0)).item()):
                return None
            return torch.clamp(stats[0] / denom, min=-1.0, max=1.0).item()

        def _all_reduce_max_scalar(value):
            if self.distributed.world_size() <= 1:
                return value
            value = value.detach().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
                return value
            return self.distributed.all_reduce(value)

        def _option_margin_abcd(scores, gold_option_index):
            gold_option_index = gold_option_index.to(device=scores.device, dtype=torch.long)
            gold_scores = scores.gather(1, gold_option_index[:, None]).squeeze(1)
            gold_mask = F.one_hot(gold_option_index, num_classes=scores.shape[-1]).bool()
            other_scores = scores.masked_fill(gold_mask, torch.finfo(scores.dtype).min)
            return gold_scores - other_scores.max(dim=-1).values

        def _option_log_odds_abcd(scores, gold_option_index):
            gold_option_index = gold_option_index.to(device=scores.device, dtype=torch.long)
            log_probs = F.log_softmax(scores, dim=-1)
            gold_log_probs = log_probs.gather(1, gold_option_index[:, None]).squeeze(1)
            gold_mask = F.one_hot(gold_option_index, num_classes=scores.shape[-1]).bool()
            other_log_probs = log_probs.masked_fill(gold_mask, torch.finfo(log_probs.dtype).min)
            wrong_log_prob = torch.logsumexp(other_log_probs, dim=-1)
            return gold_log_probs - wrong_log_prob

        def _option_prob_abcd(scores, gold_option_index):
            gold_option_index = gold_option_index.to(device=scores.device, dtype=torch.long)
            probs = F.softmax(scores, dim=-1)
            return probs.gather(1, gold_option_index[:, None]).squeeze(1)

        def _option_evidence_abcd(scores, gold_option_index):
            if masked_kd_drop_target_space == "margin":
                return _option_margin_abcd(scores, gold_option_index)
            if masked_kd_drop_target_space == "log_odds":
                return _option_log_odds_abcd(scores, gold_option_index)
            if masked_kd_drop_target_space == "prob":
                return _option_prob_abcd(scores, gold_option_index)
            raise AssertionError(f"unexpected MaskedKD drop target space: {masked_kd_drop_target_space}")

        def _centered_abcd(values):
            return values - values.mean(dim=-1, keepdim=True)

        def _pairwise_preferences_abcd(real_scores, masked_scores):
            pair_i = torch.tensor([0, 0, 0, 1, 1, 2], device=real_scores.device, dtype=torch.long)
            pair_j = torch.tensor([1, 2, 3, 2, 3, 3], device=real_scores.device, dtype=torch.long)
            real_margin = real_scores[:, pair_i] - real_scores[:, pair_j]
            masked_margin = masked_scores[:, pair_i] - masked_scores[:, pair_j]
            return (real_margin - masked_margin) / masked_kd_drop_response_tau

        def _response_loss_values(student_response, teacher_response):
            teacher_response = teacher_response.detach()
            if masked_kd_drop_response_loss == "mse":
                return (student_response - teacher_response).pow(2).mean(dim=-1)
            if masked_kd_drop_response_loss == "smooth_l1":
                return F.smooth_l1_loss(
                    student_response,
                    teacher_response,
                    beta=masked_kd_drop_huber_beta,
                    reduction="none",
                ).mean(dim=-1)
            raise AssertionError(f"unexpected MaskedKD response loss: {masked_kd_drop_response_loss}")

        def _evidence_response_loss_values(
            real_student_scores,
            masked_student_scores,
            real_teacher_scores,
            masked_teacher_scores,
        ):
            if masked_kd_drop_loss_type == "prob_shift":
                student_response = (
                    F.softmax(real_student_scores / masked_kd_drop_prob_tau, dim=-1)
                    - F.softmax(masked_student_scores / masked_kd_drop_prob_tau, dim=-1)
                )
                teacher_response = (
                    F.softmax(real_teacher_scores / masked_kd_drop_prob_tau, dim=-1)
                    - F.softmax(masked_teacher_scores / masked_kd_drop_prob_tau, dim=-1)
                )
                return _response_loss_values(student_response, teacher_response)
            if masked_kd_drop_loss_type == "centered_logit_response":
                student_response = _centered_abcd(
                    (real_student_scores - masked_student_scores) / masked_kd_drop_response_tau
                )
                teacher_response = _centered_abcd(
                    (real_teacher_scores - masked_teacher_scores) / masked_kd_drop_response_tau
                )
                return _response_loss_values(student_response, teacher_response)
            if masked_kd_drop_loss_type == "pairwise_preference_shift":
                student_response = _pairwise_preferences_abcd(real_student_scores, masked_student_scores)
                teacher_response = _pairwise_preferences_abcd(real_teacher_scores, masked_teacher_scores)
                return _response_loss_values(student_response, teacher_response)
            raise AssertionError(f"unexpected MaskedKD evidence-response loss: {masked_kd_drop_loss_type}")

        def _drop_order_contrastive_loss(student_drop, teacher_drop, weights=None):
            if student_drop.numel() < 2:
                return student_drop.sum() * 0.0
            student_diff = student_drop[:, None] - student_drop[None, :]
            teacher_diff = (teacher_drop[:, None] - teacher_drop[None, :]).detach()
            pair_mask = teacher_diff > masked_kd_drop_order_min_gap
            if not bool(pair_mask.any().detach().item()):
                return student_drop.sum() * 0.0
            target_margin = (
                masked_kd_drop_hinge_margin
                + masked_kd_drop_rank_margin_scale * teacher_diff.clamp_min(0.0)
            )
            pair_losses = F.softplus(
                (target_margin - student_diff) / masked_kd_drop_order_temperature
            ) * masked_kd_drop_order_temperature
            pair_losses = pair_losses[pair_mask]
            if weights is not None:
                pair_weights = (weights[:, None] * weights[None, :]).clamp_min(0.0).sqrt()
                pair_weights = pair_weights[pair_mask]
                return (pair_losses * pair_weights).sum() / pair_weights.sum().clamp_min(1e-12)
            return pair_losses.mean()

        def _drop_listwise_kl_loss(student_drop, teacher_drop, weights=None):
            if student_drop.numel() < 2:
                return student_drop.sum() * 0.0
            teacher_logits = teacher_drop.detach() / masked_kd_drop_order_temperature
            if weights is not None:
                teacher_logits = teacher_logits + weights.detach().clamp_min(1e-12).log()
            teacher_probs = F.softmax(teacher_logits, dim=0)
            student_log_probs = F.log_softmax(student_drop / masked_kd_drop_order_temperature, dim=0)
            return F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="sum",
            ) * (masked_kd_drop_order_temperature ** 2)

        def _check_masked_kd_pairing(real_batch, masked_batch):
            if not masked_kd_require_paired_side_batches:
                return
            real_ids = real_batch.get("id")
            masked_real_ids = masked_batch.get("maskedkd_real_id")
            if real_ids is not None and masked_real_ids is not None:
                if list(real_ids) != list(masked_real_ids):
                    raise ValueError(
                        "MaskedKD real/masked side batches are not paired: "
                        f"real_ids[:3]={list(real_ids)[:3]} "
                        f"masked_real_ids[:3]={list(masked_real_ids)[:3]}"
                    )
                return
            real_source = real_batch.get("source_index")
            masked_source = masked_batch.get("source_index")
            if real_source is not None and masked_source is not None:
                if list(real_source) != list(masked_source):
                    raise ValueError(
                        "MaskedKD real/masked side batches have mismatched source_index: "
                        f"real_source[:3]={list(real_source)[:3]} "
                        f"masked_source[:3]={list(masked_source)[:3]}"
                    )
                return
            raise ValueError(
                "MaskedKD pairing check requires real id + masked maskedkd_real_id "
                "or matching source_index metadata"
            )

        max_train_batches = int(self.config["train"].get("max_train_batches", 0) or 0)
        if max_train_batches <= 0 and max_train_fraction > 0:
            max_train_batches = max(1, int(math.ceil(num_batches_per_epoch * max_train_fraction)))
            self.logger.info(
                "max_train_fraction=%.6f -> max_train_batches=%d of %d",
                max_train_fraction,
                max_train_batches,
                num_batches_per_epoch,
            )
        elif max_train_batches > 0:
            self.logger.info(
                "Using explicit max_train_batches=%d of %d",
                max_train_batches,
                num_batches_per_epoch,
            )
        if self.distributed.rank() == 0:
            self.logger.info(
                "Optimizer update policy: grad_accum=%d max_optimizer_updates=%d checkpoint_optimizer_steps=%s naive_kd_enabled=%s lambda=%g tau=%g target_scores_key=%s active_until_optimizer_step_inclusive=%s retention_kd_enabled=%s retention_lambda=%g retention_tau=%g sequence_retention_enabled=%s sequence_retention_lambda=%g sequence_retention_tau=%g masked_kd_enabled=%s rho=%g lambda_real=%g lambda_mask=%g lambda_drop=%g tau=%g",
                gradient_accumulation_steps,
                max_optimizer_updates,
                sorted(checkpoint_optimizer_steps),
                naive_kd_enabled,
                naive_kd_lambda,
                naive_kd_tau,
                naive_kd_scores_key,
                naive_kd_active_until_optimizer_step_inclusive,
                retention_kd_enabled,
                retention_kd_lambda,
                retention_kd_tau,
                sequence_retention_enabled,
                sequence_retention_lambda,
                sequence_retention_tau,
                masked_kd_enabled,
                masked_kd_rho,
                masked_kd_lambda_real,
                masked_kd_lambda_mask,
                masked_kd_lambda_drop,
                masked_kd_tau,
            )
            self.logger.info(
                "Audio-AlignKD: enabled=%s attention=%s lambda_attn=%g "
                "feature_all=%s lambda_all=%g feature_soft=%s lambda_soft=%g "
                "feature_top16=%s lambda_top16=%g target_audio_tokens=%d "
                "attention_group_selection=%s",
                alignkd_enabled,
                alignkd_attention_enabled,
                alignkd_lambda_attention,
                alignkd_feature_all_enabled,
                alignkd_lambda_feature_all,
                alignkd_feature_soft_enabled,
                alignkd_lambda_feature_soft,
                alignkd_feature_top16_enabled,
                alignkd_lambda_feature_top16,
                alignkd_target_audio_tokens,
                alignkd_attention_group_selection,
            )
            if output_control_enabled:
                self.logger.info(
                    "OUTPUT_CONTROL_WEIGHT_SCHEDULE %s",
                    json.dumps(
                        output_control_phases,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )

        lr_scheduler = None
        loss_tracker = None
        step_lr_scheduler = False
        if self.config["train"]["optimizer"]["scheduler"] is not None:
            lr_schedule = self.config["train"]["optimizer"]["scheduler"]
            if self._is_step_lr_scheduler(lr_schedule):
                if max_optimizer_updates > 0:
                    total_optimizer_steps = max_optimizer_updates
                elif max_train_batches > 0:
                    total_optimizer_steps = int(math.ceil(max_train_batches / gradient_accumulation_steps))
                else:
                    total_optimizer_steps = int(math.ceil(
                        num_batches_per_epoch * self.config["train"]["num_epochs"] / gradient_accumulation_steps
                    ))
                lr_scheduler = self._build_step_lr_scheduler(optimizer, total_optimizer_steps)
                step_lr_scheduler = True
            elif lr_schedule == "step":
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.5)
            elif lr_schedule == "cosine":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config["train"]["num_epochs"])
            elif lr_schedule == "cosine_restarts":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
            elif lr_schedule == "loss_tracking":
                lr_scheduler = loss_tracker = LossTrackingLRScheduler(
                    optimizer, lr_min=self.config.get("lr_min", 0.))
            else:
                raise ValueError(f"No such lr schedule: {lr_schedule}")

        resume_batch_index = 0
        total_step = 0
        resume_mid_epoch = False
        if resume_checkpoint_payload is not None:
            start_epoch, resume_batch_index, total_step = self._restore_training_state_from_checkpoint(
                resume_checkpoint_payload,
                optimizer,
                lr_scheduler,
                grad_scaler,
                grad_norm_tracker,
            )
            if isinstance(resume_checkpoint_payload, dict):
                resume_training_state = resume_checkpoint_payload.get("training_state")
                resume_mid_epoch = (
                    isinstance(resume_training_state, dict)
                    and not bool(resume_training_state.get("completed_epoch", False))
                )
            self.distributed.broadcast_optimizer_state(optimizer)
    
        loss_history = dict(
            optimizer_step=[],
            loss=[],
            lm_loss=[],
            naive_kd_active=[],
            naive_kd_effective_lambda=[],
            naive_kd_loss=[],
            naive_kd_count=[],
            retention_kd_loss=[],
            retention_kd_count=[],
            replay_ce_a=[],
            replay_ce_b=[],
            replay_ce_c=[],
            replay_kl_b=[],
            replay_kl_c=[],
            replay_projection_conflict=[],
            replay_projection_cosine_before=[],
            replay_projection_correction_removed_fraction=[],
            replay_projection_correction_norm=[],
            replay_projection_retention_norm=[],
            sequence_retention_kd_loss=[],
            sequence_retention_sample_count=[],
            sequence_retention_token_count=[],
            alignkd_attention_loss=[],
            alignkd_feature_all_loss=[],
            alignkd_feature_soft_loss=[],
            alignkd_feature_top16_loss=[],
            alignkd_count=[],
            masked_kd_side_loss=[],
            masked_kd_real_ce=[],
            masked_kd_real_kd=[],
            masked_kd_mask_kd=[],
            masked_kd_drop_loss=[],
            masked_kd_teacher_drop_mean=[],
            masked_kd_teacher_drop_max=[],
            masked_kd_drop_active_frac=[],
            masked_kd_grad_cosine=[],
            masked_kd_side_count=[],
            listen_shuffle_loss=[],
            listen_shuffle_margin_loss=[],
            total_grad_norm=[],
            grad_scale=[],
            lr=[],
            adapter_alpha=[],
            adapter_impact_ratio=[],
            output_control_phase=[],
            output_control_structure_weight=[],
            output_control_reasoning_weight=[],
            output_control_answer_weight=[],
        )

        stop_training = False
        stop_due_to_signal = False
        output_control_last_phase_index = None
        mid_epoch_checkpoint_batches = self._get_checkpoint_batch_indices(num_batches_per_epoch)
        if self.distributed.rank() == 0:
            self.logger.info(
                "Training resume position: epoch=%d batch=%d total_step=%d",
                start_epoch + 1,
                resume_batch_index,
                total_step,
            )
            if mid_epoch_checkpoint_batches:
                self.logger.info(
                    "Mid-epoch training-state checkpoints at batch indices: %s",
                    sorted(mid_epoch_checkpoint_batches),
                )
        ignore_index = dataset.tokenizer.encode(dataset.tokenizer.pad_token)[0]
        stop_requested, previous_signal_handlers = self._install_training_signal_handlers()
        signal_checkpointing_enabled = bool(previous_signal_handlers)
        for epoch in range(start_epoch, self.config["train"]["num_epochs"]):
            # set epoch to use different seeds for different epochs during sampling
            data_sampler.set_epoch(epoch)
            if masked_kd_side_sampler is not None:
                masked_kd_side_sampler.set_epoch(epoch)
            if masked_kd_mask_sampler is not None:
                masked_kd_mask_sampler.set_epoch(epoch)
            masked_kd_side_iter = iter(masked_kd_side_loader) if masked_kd_side_loader is not None else None
            masked_kd_mask_iter = iter(masked_kd_mask_loader) if masked_kd_mask_loader is not None else None
            masked_kd_side_cycle = 0
            masked_kd_mask_cycle = 0
            tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)
            metrics_train = {"epoch": epoch}
            accerr_epo = 0  # accumulated error per epoch
            epoch_resume_batch_index = resume_batch_index if epoch == start_epoch else 0
            last_batch_index = epoch_resume_batch_index
            if epoch_resume_batch_index > 0:
                if self.distributed.rank() == 0:
                    self.logger.info(
                        "Resuming epoch %d from batch index %d/%d",
                        epoch + 1,
                        epoch_resume_batch_index,
                        num_batches_per_epoch,
                    )
                tqdm_handler.update(min(epoch_resume_batch_index, num_batches_per_epoch))

            if epoch > 0 and lr_scheduler is not None and not step_lr_scheduler:
                if resume_mid_epoch and epoch == start_epoch:
                    lr = lr_scheduler.get_last_lr()[0]
                else:
                    lr_scheduler.step()
                    lr = lr_scheduler.get_last_lr()[0]
            elif lr_scheduler is None:
                lr = optimizer.param_groups[0]["lr"]
                if epoch == 0:
                    print("Starting the training with a learning rate of {}".format(lr))

            optimizer.zero_grad(set_to_none=True)
            micro_step_in_accum = 0
            data_iterator = prepare_resumed_data_iterator(
                data_loader,
                epoch_resume_batch_index,
                self._restore_pending_resume_rng_state,
            )
            for ii, batch_data_dict in enumerate(
                data_iterator, start=epoch_resume_batch_index
            ):
                batch_audio1 = batch_data_dict['waveform1']
                batch_audio2 = batch_data_dict['waveform2']
                batch_input = batch_data_dict['input']
                batch_answer = batch_data_dict['answer']

                batch_answer['attention_mask'] = torch.stack([torch.cat((torch.ones(self.config["model"]["decoder"]["total_prefix_length"]), text), dim=0) for text in batch_answer['attention_mask']])

                input_dict = {
                    "audio1":batch_audio1,
                    "audio2": batch_audio2,
                    "input":batch_input,
                    "answer":batch_answer,
                }
                if "waveform1_lengths" in batch_data_dict:
                    input_dict["audio1_lengths"] = batch_data_dict["waveform1_lengths"]
                if "waveform2_lengths" in batch_data_dict:
                    input_dict["audio2_lengths"] = batch_data_dict["waveform2_lengths"]
                if "ced_hidden" in batch_data_dict:
                    input_dict["ced_hidden"] = batch_data_dict["ced_hidden"]
                if "ced_hidden_segment_lengths" in batch_data_dict:
                    input_dict["ced_hidden_segment_lengths"] = batch_data_dict["ced_hidden_segment_lengths"]
                if "ced_hidden_lengths" in batch_data_dict:
                    input_dict["ced_hidden_lengths"] = batch_data_dict["ced_hidden_lengths"]
                if "beats_hidden" in batch_data_dict:
                    input_dict["beats_hidden"] = batch_data_dict["beats_hidden"]
                if "beats_hidden_segment_lengths" in batch_data_dict:
                    input_dict["beats_hidden_segment_lengths"] = batch_data_dict["beats_hidden_segment_lengths"]
                if "beats_hidden_lengths" in batch_data_dict:
                    input_dict["beats_hidden_lengths"] = batch_data_dict["beats_hidden_lengths"]
                if alignkd_enabled:
                    if "alignkd_query_group_mask" not in batch_data_dict:
                        raise ValueError(
                            "Audio-AlignKD batch is missing alignkd_query_group_mask"
                        )
                    input_dict["alignkd_query_group_mask"] = batch_data_dict[
                        "alignkd_query_group_mask"
                    ]
                input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                
                replay_group_ids = None
                if replay_enabled:
                    required_replay_fields = [
                        "replay_group_id",
                        replay_target_scores_key,
                        "option_valid_mask_abcd",
                    ]
                    if replay_first_token_gold_ce_enabled:
                        required_replay_fields.append("gold_option_index")
                    missing_replay_fields = [
                        key for key in required_replay_fields if key not in batch_data_dict
                    ]
                    if missing_replay_fields:
                        raise ValueError(
                            f"abc_replay_objective batch is missing {missing_replay_fields}"
                        )
                    replay_group_ids = batch_data_dict["replay_group_id"].to(
                        self.device, non_blocking=True
                    ).long()
                    replay_slices, _active_replay_group_counts = replay_group_slices(
                        replay_group_ids,
                        replay_group_counts,
                        dynamic_counts=replay_dynamic_group_counts,
                    )
                    group_outputs = []
                    for row_slice in replay_slices:
                        group_outputs.append(model(slice_model_input(input_dict, row_slice)))
                    model_outputs = SimpleNamespace(
                        logits=torch.cat([output.logits for output in group_outputs], dim=0)
                    )
                else:
                    model_outputs = model(input_dict)
                adapter_stats = self._get_adapter_stats(model)
                answer_len = input_dict['answer']['input_ids'].shape[1]
                prefix_len = model_outputs.logits.shape[1] - answer_len
                logits = model_outputs.logits[:, prefix_len - 1: -1]
                current_optimizer_step = total_step + 1
                output_control_phase_index = None
                output_control_structure_weight = None
                output_control_reasoning_weight = None
                output_control_answer_weight = None
                if output_control_enabled:
                    if "output_control_region_ids" not in batch_data_dict:
                        raise ValueError("output-control batch is missing token regions")
                    (
                        output_control_phase_index,
                        output_control_structure_weight,
                        output_control_reasoning_weight,
                        output_control_answer_weight,
                    ) = output_control_weights_for_optimizer_step(
                        output_control_phases,
                        current_optimizer_step,
                    )
                    if (
                        output_control_phase_index != output_control_last_phase_index
                        and self.distributed.rank() == 0
                    ):
                        self.logger.info(
                            "OUTPUT_CONTROL_PHASE %s",
                            json.dumps(
                                {
                                    **output_control_phases[output_control_phase_index],
                                    "optimizer_step": current_optimizer_step,
                                    "phase_index": output_control_phase_index,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    output_control_last_phase_index = output_control_phase_index
                    output_control_regions = batch_data_dict[
                        "output_control_region_ids"
                    ].to(self.device, non_blocking=True)
                    lm_loss, _output_control_diagnostics = output_control_weighted_ce(
                        logits,
                        input_dict["answer"]["input_ids"],
                        output_control_regions,
                        structure_weight=output_control_structure_weight,
                        reasoning_weight=output_control_reasoning_weight,
                        answer_weight=output_control_answer_weight,
                    )
                elif replay_enabled:
                    lm_loss, replay_ce_values = replay_group_weighted_ce(
                        logits,
                        input_dict["answer"]["input_ids"],
                        replay_group_ids,
                        replay_ce_weights,
                        ignore_index=ignore_index,
                    )
                else:
                    lm_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_dict['answer']['input_ids'].flatten(), ignore_index=ignore_index)
                loss = lm_loss
                replay_first_token_gold_ce_values = None
                if replay_enabled and replay_first_token_gold_ce_enabled:
                    valid_options = batch_data_dict["option_valid_mask_abcd"].to(
                        self.device, non_blocking=True
                    ).bool()
                    student_scores = logits[:, 0, replay_option_token_ids_tensor].float()
                    (
                        replay_first_token_gold_ce_loss,
                        replay_first_token_gold_ce_values,
                    ) = replay_group_first_token_gold_ce(
                        student_scores,
                        batch_data_dict["gold_option_index"].to(
                            self.device, non_blocking=True
                        ).long(),
                        valid_options,
                        replay_group_ids,
                        replay_first_token_gold_ce_weights,
                    )
                    loss = loss + replay_first_token_gold_ce_loss
                replay_kd_per_sample = None
                replay_kl_values = None
                replay_correction_loss = None
                replay_retention_loss = None
                if replay_enabled:
                    replay_target_scores = resolve_replay_target_scores(
                        batch_data_dict, replay_target_scores_key, self.device
                    )
                    valid_options = batch_data_dict["option_valid_mask_abcd"].to(
                        self.device, non_blocking=True
                    ).bool()
                    student_scores = logits[:, 0, replay_option_token_ids_tensor].float()
                    replay_kd_per_sample = masked_abcd_kd_per_sample(
                        student_scores, replay_target_scores, valid_options, replay_tau
                    )
                    replay_kl_values = tuple(
                        replay_kd_per_sample.masked_select(replay_group_ids.eq(group_id)).mean()
                        for group_id in range(len(replay_group_counts))
                    )
                    replay_kd_loss = sum(
                        float(weight) * value
                        for weight, value in zip(replay_kd_weights, replay_kl_values)
                    )
                    loss = loss + replay_kd_loss
                    if replay_gradient_projection_enabled:
                        group_loss_values = []
                        for group_id, group_name in enumerate(replay_group_names):
                            group_loss = (
                                float(replay_ce_weights[group_id])
                                * replay_ce_values[group_id]
                                + float(replay_kd_weights[group_id])
                                * replay_kl_values[group_id]
                            )
                            if replay_first_token_gold_ce_values is not None:
                                group_loss = group_loss + (
                                    float(replay_first_token_gold_ce_weights[group_id])
                                    * replay_first_token_gold_ce_values[group_id]
                                )
                            group_loss_values.append(group_loss)
                        correction_names = set(
                            replay_gradient_projection["correction_groups"]
                        )
                        retention_names = set(
                            replay_gradient_projection["retention_groups"]
                        )
                        replay_correction_loss = sum(
                            value
                            for name, value in zip(replay_group_names, group_loss_values)
                            if name in correction_names
                        )
                        replay_retention_loss = sum(
                            value
                            for name, value in zip(replay_group_names, group_loss_values)
                            if name in retention_names
                        )
                        loss = replay_correction_loss + replay_retention_loss
                listen_shuffle_loss = None
                listen_shuffle_margin_loss = None
                if listen_shuffle_margin_enabled and input_dict['answer']['input_ids'].shape[0] > 1:
                    shuffled_outputs = model(input_dict, force_shuffle_listen_state=True)
                    shuffled_prefix_len = shuffled_outputs.logits.shape[1] - answer_len
                    shuffled_logits = shuffled_outputs.logits[:, shuffled_prefix_len - 1: -1]
                    listen_shuffle_loss = F.cross_entropy(
                        shuffled_logits.reshape(-1, shuffled_logits.shape[-1]),
                        input_dict['answer']['input_ids'].flatten(),
                        ignore_index=ignore_index,
                    )
                    listen_shuffle_margin_loss = F.relu(lm_loss + listen_shuffle_margin - listen_shuffle_loss)
                    loss = lm_loss + listen_shuffle_margin_weight * listen_shuffle_margin_loss

                naive_kd_active = naive_kd_is_active(
                    naive_kd_enabled,
                    total_step,
                    naive_kd_active_until_optimizer_step_inclusive,
                )
                naive_kd_effective_lambda = (
                    naive_kd_lambda if naive_kd_active else 0.0
                )

                kd_mask = None
                if naive_kd_active or alignkd_enabled:
                    if "is_kd_sample" not in batch_data_dict:
                        raise ValueError("KD-enabled batch is missing is_kd_sample")
                    kd_mask = batch_data_dict["is_kd_sample"].to(
                        self.device, non_blocking=True
                    ).bool()

                naive_kd_loss = None
                naive_kd_count = None
                naive_kd_loss_sum = None
                if naive_kd_active:
                    if naive_kd_scores_key not in batch_data_dict:
                        raise ValueError(
                            "Naive KD batch is missing configured target scores: "
                            f"{naive_kd_scores_key}"
                        )
                    naive_kd_count = kd_mask.to(dtype=torch.float32).sum()
                    teacher_scores = batch_data_dict[naive_kd_scores_key].to(
                        self.device, non_blocking=True
                    ).float()
                    if teacher_scores.ndim != 2 or teacher_scores.shape[-1] != 4:
                        raise ValueError(
                            f"Naive KD target scores must have shape [B,4], got "
                            f"{tuple(teacher_scores.shape)}"
                        )
                    if not bool(torch.isfinite(teacher_scores).all()):
                        raise FloatingPointError(
                            f"Naive KD target scores are non-finite: {naive_kd_scores_key}"
                        )
                    student_scores = logits[:, 0, naive_kd_option_token_ids_tensor].float()
                    naive_kd_per_sample = abcd_kd_per_sample(
                        student_scores,
                        teacher_scores,
                        naive_kd_tau,
                    )
                    (
                        naive_kd_loss,
                        naive_kd_loss_sum,
                        _naive_kd_global_count,
                    ) = globally_normalized_masked_loss(
                        naive_kd_per_sample, kd_mask, self.distributed
                    )
                    loss = loss + naive_kd_effective_lambda * naive_kd_loss

                retention_kd_loss = None
                retention_kd_loss_sum = None
                retention_kd_count = None
                if retention_kd_enabled:
                    if retention_kd_mask_key not in batch_data_dict:
                        raise ValueError(
                            "Retention KD batch is missing configured mask: "
                            f"{retention_kd_mask_key}"
                        )
                    if retention_kd_scores_key not in batch_data_dict:
                        raise ValueError(
                            "Retention KD batch is missing configured scores: "
                            f"{retention_kd_scores_key}"
                        )
                    retention_mask = batch_data_dict[retention_kd_mask_key].to(
                        self.device, non_blocking=True
                    ).bool()
                    retention_kd_count = retention_mask.to(
                        dtype=torch.float32
                    ).sum()
                    ref_scores = batch_data_dict[retention_kd_scores_key].to(
                        self.device, non_blocking=True
                    ).float()
                    student_retention_scores = logits[
                        :, 0, retention_kd_option_token_ids_tensor
                    ].float()
                    retention_kd_per_sample = abcd_kd_per_sample(
                        student_retention_scores,
                        ref_scores,
                        retention_kd_tau,
                    )
                    (
                        retention_kd_loss,
                        retention_kd_loss_sum,
                        _retention_kd_global_count,
                    ) = globally_normalized_masked_loss(
                        retention_kd_per_sample,
                        retention_mask,
                        self.distributed,
                    )
                    loss = loss + retention_kd_lambda * retention_kd_loss

                sequence_retention_kd_loss = None
                sequence_retention_kd_loss_sum = None
                sequence_retention_sample_count = None
                sequence_retention_token_count = None
                sequence_retention_active_mask = None
                if sequence_retention_enabled:
                    required_sequence_fields = (
                        "sequence_retention_topk_ids",
                        "sequence_retention_topk_probs",
                        "sequence_retention_tail_probs",
                        "sequence_retention_active_mask",
                        "sequence_retention_cache_valid",
                    )
                    missing_sequence_fields = [
                        key
                        for key in required_sequence_fields
                        if key not in batch_data_dict
                    ]
                    if missing_sequence_fields:
                        raise ValueError(
                            "Sequence-retention batch is missing fields: "
                            f"{missing_sequence_fields}"
                        )
                    sequence_retention_active_mask = batch_data_dict[
                        "sequence_retention_active_mask"
                    ].to(self.device, non_blocking=True).bool()
                    if sequence_retention_active_mask.shape != logits.shape[:2]:
                        raise ValueError(
                            "Sequence-retention active mask/logit shape mismatch: "
                            f"{tuple(sequence_retention_active_mask.shape)} != "
                            f"{tuple(logits.shape[:2])}"
                        )
                    answer_nonpad = input_dict["answer"]["input_ids"] != ignore_index
                    if bool((sequence_retention_active_mask & ~answer_nonpad).any()):
                        raise ValueError(
                            "Sequence-retention mask includes a padded answer token"
                        )
                    sequence_cache_valid = batch_data_dict[
                        "sequence_retention_cache_valid"
                    ].to(self.device, non_blocking=True).bool()
                    sequence_sample_mask = sequence_retention_active_mask.any(dim=-1)
                    if bool((sequence_sample_mask & ~sequence_cache_valid).any()):
                        raise ValueError(
                            "Sequence-retention active sample lacks a valid REF cache row"
                        )
                    sequence_topk_ids = batch_data_dict[
                        "sequence_retention_topk_ids"
                    ].to(self.device, non_blocking=True).long()
                    if sequence_topk_ids.shape != (
                        logits.shape[0],
                        logits.shape[1],
                        sequence_retention_top_k,
                    ):
                        raise ValueError(
                            "Sequence-retention top-k shape mismatch: "
                            f"{tuple(sequence_topk_ids.shape)}"
                        )
                    active_gold = input_dict["answer"]["input_ids"].masked_select(
                        sequence_retention_active_mask
                    )
                    active_teacher_top1 = sequence_topk_ids[..., 0].masked_select(
                        sequence_retention_active_mask
                    )
                    if not torch.equal(active_gold, active_teacher_top1):
                        raise ValueError(
                            "Sequence-retention active mask violates REF-top1 == gold"
                        )
                    sequence_per_sample = sparse_topk_tail_kd_per_sample(
                        logits,
                        sequence_topk_ids,
                        batch_data_dict["sequence_retention_topk_probs"].to(
                            self.device, non_blocking=True
                        ),
                        batch_data_dict["sequence_retention_tail_probs"].to(
                            self.device, non_blocking=True
                        ),
                        sequence_retention_active_mask,
                        tau=sequence_retention_tau,
                    )
                    sequence_retention_sample_count = sequence_sample_mask.to(
                        dtype=torch.float32
                    ).sum()
                    sequence_retention_token_count = (
                        sequence_retention_active_mask.to(dtype=torch.float32).sum()
                    )
                    (
                        sequence_retention_kd_loss,
                        sequence_retention_kd_loss_sum,
                        _sequence_retention_global_count,
                    ) = globally_normalized_masked_loss(
                        sequence_per_sample,
                        sequence_sample_mask,
                        self.distributed,
                    )
                    loss = (
                        loss
                        + sequence_retention_lambda * sequence_retention_kd_loss
                    )

                alignkd_attention_loss = None
                alignkd_attention_loss_sum = None
                alignkd_feature_all_loss = None
                alignkd_feature_all_loss_sum = None
                alignkd_feature_soft_loss = None
                alignkd_feature_soft_loss_sum = None
                alignkd_feature_top16_loss = None
                alignkd_feature_top16_loss_sum = None
                alignkd_count = None
                if alignkd_enabled:
                    alignkd_count = kd_mask.to(dtype=torch.float32).sum()
                    if "teacher_cache_valid" not in batch_data_dict:
                        raise ValueError("Audio-AlignKD batch is missing teacher_cache_valid")
                    cache_valid = batch_data_dict["teacher_cache_valid"].to(
                        self.device, non_blocking=True
                    ).bool()
                    if not torch.equal(cache_valid, kd_mask):
                        sample_keys = batch_data_dict.get("sample_key", [])
                        raise ValueError(
                            "Audio-AlignKD cache_valid must exactly equal is_kd_sample; "
                            f"sample_keys={list(sample_keys)[:8]}"
                        )
                    query_group_mask = input_dict["alignkd_query_group_mask"].bool()
                    if bool((query_group_mask[kd_mask].sum(dim=-1) <= 0).any()):
                        raise ValueError("An Align sample has an empty Q/A/B/C/D query group")
                    alignkd_aux = getattr(model_outputs, "alignkd_aux", None)
                    if not isinstance(alignkd_aux, dict):
                        raise ValueError("Model did not return alignkd_aux")

                    if alignkd_attention_enabled:
                        required = {
                            "student_attn_qabcd_126",
                            "student_audio_mass_qabcd",
                        }
                        missing = required - set(alignkd_aux)
                        if missing:
                            raise ValueError(
                                f"Model AlignKD attention output is missing {sorted(missing)}"
                            )
                        for key in (
                            "teacher_attn_qabcd_126",
                            "teacher_audio_mass_qabcd",
                        ):
                            if key not in batch_data_dict:
                                raise ValueError(f"Audio-AlignKD batch is missing {key}")
                        attention_group_mask = None
                        if (
                            alignkd_attention_group_selection
                            == "question_and_gold_option"
                        ):
                            if "gold_option_index" not in batch_data_dict:
                                raise ValueError(
                                    "question_and_gold_option attention requires "
                                    "gold_option_index"
                                )
                            gold_option_index = batch_data_dict[
                                "gold_option_index"
                            ].to(self.device, non_blocking=True).long()
                            invalid_gold = (gold_option_index < 0) | (
                                gold_option_index >= 4
                            )
                            if bool((invalid_gold & kd_mask).any().detach().item()):
                                raise ValueError(
                                    "An Align sample has an invalid gold_option_index"
                                )
                            attention_group_mask = torch.zeros(
                                (gold_option_index.shape[0], 5),
                                dtype=torch.bool,
                                device=self.device,
                            )
                            attention_group_mask[:, 0] = True
                            attention_group_mask.scatter_(
                                1,
                                gold_option_index.clamp(min=0, max=3)[:, None] + 1,
                                True,
                            )
                        attention_per_sample = attention_kd_per_sample(
                            alignkd_aux["student_attn_qabcd_126"],
                            alignkd_aux["student_audio_mass_qabcd"],
                            batch_data_dict["teacher_attn_qabcd_126"].to(
                                self.device, non_blocking=True
                            ),
                            batch_data_dict["teacher_audio_mass_qabcd"].to(
                                self.device, non_blocking=True
                            ),
                            group_mask=attention_group_mask,
                            audio_mass_weight=alignkd_audio_mass_weight,
                        )
                        (
                            alignkd_attention_loss,
                            alignkd_attention_loss_sum,
                            _alignkd_attention_global_count,
                        ) = globally_normalized_masked_loss(
                            attention_per_sample, kd_mask, self.distributed
                        )
                        loss = loss + alignkd_lambda_attention * alignkd_attention_loss

                    student_gram = None
                    teacher_gram = None
                    if alignkd_feature_all_enabled:
                        if "student_audio_features" not in alignkd_aux:
                            raise ValueError(
                                "Model AlignKD output is missing student_audio_features"
                            )
                        if "teacher_feature_gram_126" not in batch_data_dict:
                            raise ValueError(
                                "Audio-AlignKD batch is missing teacher_feature_gram_126"
                            )
                        student_gram = temporal_cosine_gram(
                            alignkd_aux["student_audio_features"]
                        )
                        teacher_gram = batch_data_dict[
                            "teacher_feature_gram_126"
                        ].to(self.device, non_blocking=True)
                        feature_all_per_sample = gram_all_kd_per_sample(
                            student_gram, teacher_gram
                        )
                        (
                            alignkd_feature_all_loss,
                            alignkd_feature_all_loss_sum,
                            _alignkd_feature_all_global_count,
                        ) = globally_normalized_masked_loss(
                            feature_all_per_sample, kd_mask, self.distributed
                        )
                        loss = loss + (
                            alignkd_lambda_feature_all * alignkd_feature_all_loss
                        )

                    if alignkd_feature_soft_enabled:
                        if "teacher_focus_126" not in batch_data_dict:
                            raise ValueError(
                                "Audio-AlignKD batch is missing teacher_focus_126"
                            )
                        feature_soft_per_sample = gram_soft_kd_per_sample(
                            student_gram,
                            teacher_gram,
                            batch_data_dict["teacher_focus_126"].to(
                                self.device, non_blocking=True
                            ),
                        )
                        (
                            alignkd_feature_soft_loss,
                            alignkd_feature_soft_loss_sum,
                            _alignkd_feature_soft_global_count,
                        ) = globally_normalized_masked_loss(
                            feature_soft_per_sample, kd_mask, self.distributed
                        )
                        loss = loss + (
                            alignkd_lambda_feature_soft * alignkd_feature_soft_loss
                        )

                    if alignkd_feature_top16_enabled:
                        if "teacher_top16_indices" not in batch_data_dict:
                            raise ValueError(
                                "Audio-AlignKD batch is missing teacher_top16_indices"
                            )
                        feature_top16_per_sample = gram_topk_kd_per_sample(
                            student_gram,
                            teacher_gram,
                            batch_data_dict["teacher_top16_indices"].to(
                                self.device, non_blocking=True
                            ),
                        )
                        (
                            alignkd_feature_top16_loss,
                            alignkd_feature_top16_loss_sum,
                            _alignkd_feature_top16_global_count,
                        ) = globally_normalized_masked_loss(
                            feature_top16_per_sample, kd_mask, self.distributed
                        )
                        loss = loss + (
                            alignkd_lambda_feature_top16
                            * alignkd_feature_top16_loss
                        )

                masked_kd_main_loss_for_grad_cosine = loss
                masked_kd_side_loss = None
                masked_kd_real_ce = None
                masked_kd_real_kd = None
                masked_kd_mask_kd = None
                masked_kd_drop_loss = None
                masked_kd_grad_cosine = None
                masked_kd_side_count = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_side_loss_sum = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_real_ce_sum = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_real_kd_sum = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_mask_kd_sum = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_drop_loss_sum = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_teacher_drop_sum = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_teacher_drop_count = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_teacher_drop_max = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_drop_active_count = (
                    torch.zeros((), device=self.device, dtype=torch.float32)
                    if masked_kd_enabled else None
                )
                masked_kd_side_input_dict = None
                masked_kd_side_outputs = None
                masked_kd_mask_input_dict = None
                masked_kd_mask_outputs = None
                masked_kd_mask_batch = None
                if masked_kd_enabled:
                    try:
                        masked_kd_side_batch = next(masked_kd_side_iter)
                    except StopIteration:
                        masked_kd_side_cycle += 1
                        masked_kd_side_sampler.set_epoch(epoch * 100000 + masked_kd_side_cycle)
                        masked_kd_side_iter = iter(masked_kd_side_loader)
                        masked_kd_side_batch = next(masked_kd_side_iter)

                    masked_kd_side_input_dict = self._make_train_input_dict_from_batch(masked_kd_side_batch)
                    masked_kd_side_outputs = model(masked_kd_side_input_dict)
                    side_answer_len = masked_kd_side_input_dict["answer"]["input_ids"].shape[1]
                    side_prefix_len = masked_kd_side_outputs.logits.shape[1] - side_answer_len
                    side_logits = masked_kd_side_outputs.logits[:, side_prefix_len - 1: -1]
                    masked_kd_real_ce = F.cross_entropy(
                        side_logits.reshape(-1, side_logits.shape[-1]),
                        masked_kd_side_input_dict["answer"]["input_ids"].flatten(),
                        ignore_index=ignore_index,
                    )
                    masked_kd_side_loss = masked_kd_real_ce
                    masked_kd_side_count = torch.tensor(
                        float(side_logits.shape[0]), device=self.device, dtype=torch.float32
                    )
                    side_teacher_scores = None
                    side_student_scores = None
                    if masked_kd_lambda_real > 0.0 or masked_kd_lambda_drop > 0.0:
                        if "teacher_scores_abcd" not in masked_kd_side_batch:
                            raise ValueError("MaskedKD side batch is missing teacher_scores_abcd")
                        side_teacher_scores = masked_kd_side_batch["teacher_scores_abcd"].to(
                            self.device, non_blocking=True
                        ).float()
                        side_student_scores = side_logits[:, 0, masked_kd_option_token_ids_tensor].float()
                    if masked_kd_lambda_real > 0.0:
                        teacher_probs = F.softmax(side_teacher_scores / masked_kd_tau, dim=-1)
                        student_log_probs = F.log_softmax(side_student_scores / masked_kd_tau, dim=-1)
                        masked_kd_real_kd = F.kl_div(
                            student_log_probs,
                            teacher_probs,
                            reduction="batchmean",
                        ) * (masked_kd_tau ** 2)
                        masked_kd_side_loss = masked_kd_side_loss + masked_kd_lambda_real * masked_kd_real_kd
                    if masked_kd_needs_masked_view:
                        try:
                            masked_kd_mask_batch = next(masked_kd_mask_iter)
                        except StopIteration:
                            masked_kd_mask_cycle += 1
                            masked_kd_mask_sampler.set_epoch(epoch * 100000 + masked_kd_mask_cycle)
                            masked_kd_mask_iter = iter(masked_kd_mask_loader)
                            masked_kd_mask_batch = next(masked_kd_mask_iter)

                        if masked_kd_lambda_drop > 0.0:
                            _check_masked_kd_pairing(masked_kd_side_batch, masked_kd_mask_batch)
                        masked_teacher_scores_key = masked_kd_mask_teacher_scores_key or "teacher_scores_abcd"
                        if masked_teacher_scores_key in masked_kd_mask_batch:
                            masked_teacher_scores_source = masked_kd_mask_batch
                        elif masked_teacher_scores_key in masked_kd_side_batch:
                            masked_teacher_scores_source = masked_kd_side_batch
                        else:
                            raise ValueError(
                                "MaskedKD masked teacher scores key "
                                f"{masked_teacher_scores_key!r} was not found in side or masked batch"
                            )
                        masked_kd_mask_input_dict = self._make_train_input_dict_from_batch(masked_kd_mask_batch)
                        masked_kd_mask_outputs = model(masked_kd_mask_input_dict)
                        mask_answer_len = masked_kd_mask_input_dict["answer"]["input_ids"].shape[1]
                        mask_prefix_len = masked_kd_mask_outputs.logits.shape[1] - mask_answer_len
                        mask_logits = masked_kd_mask_outputs.logits[:, mask_prefix_len - 1: -1]
                        masked_teacher_scores = masked_teacher_scores_source[masked_teacher_scores_key].to(
                            self.device, non_blocking=True
                        ).float()
                        masked_student_scores = mask_logits[:, 0, masked_kd_option_token_ids_tensor].float()
                        if masked_kd_lambda_mask > 0.0:
                            masked_teacher_probs = F.softmax(masked_teacher_scores / masked_kd_tau, dim=-1)
                            masked_student_log_probs = F.log_softmax(masked_student_scores / masked_kd_tau, dim=-1)
                            masked_kd_mask_kd = F.kl_div(
                                masked_student_log_probs,
                                masked_teacher_probs,
                                reduction="batchmean",
                            ) * (masked_kd_tau ** 2)
                            masked_kd_side_loss = masked_kd_side_loss + masked_kd_lambda_mask * masked_kd_mask_kd
                        if masked_kd_lambda_drop > 0.0:
                            if "gold_option_index" not in masked_kd_side_batch:
                                raise ValueError("MaskedKD side batch is missing gold_option_index")
                            gold_option_index = masked_kd_side_batch["gold_option_index"].to(
                                self.device, non_blocking=True
                            )
                            if masked_kd_drop_teacher_drop_key:
                                if masked_kd_drop_teacher_drop_key in masked_kd_side_batch:
                                    teacher_drop = masked_kd_side_batch[masked_kd_drop_teacher_drop_key].to(
                                        self.device, non_blocking=True
                                    ).float().detach()
                                elif masked_kd_drop_teacher_drop_key in masked_kd_mask_batch:
                                    teacher_drop = masked_kd_mask_batch[masked_kd_drop_teacher_drop_key].to(
                                        self.device, non_blocking=True
                                    ).float().detach()
                                else:
                                    raise ValueError(
                                        "MaskedKD drop_teacher_drop_key="
                                        f"{masked_kd_drop_teacher_drop_key!r} was not found in side or masked batch"
                                    )
                            else:
                                teacher_drop = (
                                    _option_evidence_abcd(side_teacher_scores, gold_option_index)
                                    - _option_evidence_abcd(masked_teacher_scores, gold_option_index)
                                ).detach()
                            if masked_kd_drop_positive_only:
                                teacher_drop = F.relu(teacher_drop)
                            if masked_kd_drop_cap > 0.0:
                                teacher_drop = teacher_drop.clamp(max=masked_kd_drop_cap)
                            student_drop = (
                                _option_evidence_abcd(side_student_scores, gold_option_index)
                                - _option_evidence_abcd(masked_student_scores, gold_option_index)
                            )
                            drop_loss_mask = torch.ones_like(teacher_drop, dtype=torch.bool)
                            if masked_kd_drop_min_teacher_drop > 0.0:
                                drop_loss_mask = drop_loss_mask & (
                                    teacher_drop >= masked_kd_drop_min_teacher_drop
                                )
                            if masked_kd_drop_real_teacher_must_be_correct:
                                if masked_kd_drop_teacher_correct_key:
                                    if masked_kd_drop_teacher_correct_key in masked_kd_side_batch:
                                        teacher_correct_mask = masked_kd_side_batch[masked_kd_drop_teacher_correct_key].to(
                                            self.device, non_blocking=True
                                        ).bool()
                                    elif masked_kd_drop_teacher_correct_key in masked_kd_mask_batch:
                                        teacher_correct_mask = masked_kd_mask_batch[masked_kd_drop_teacher_correct_key].to(
                                            self.device, non_blocking=True
                                        ).bool()
                                    else:
                                        raise ValueError(
                                            "MaskedKD drop_teacher_correct_key="
                                            f"{masked_kd_drop_teacher_correct_key!r} was not found in side or masked batch"
                                        )
                                    drop_loss_mask = drop_loss_mask & teacher_correct_mask
                                else:
                                    drop_loss_mask = drop_loss_mask & (
                                        side_teacher_scores.argmax(dim=-1) == gold_option_index
                                    )
                            drop_loss_weights = None
                            if masked_kd_drop_weight_power > 0.0:
                                drop_loss_weights = teacher_drop.detach().float().clamp_min(0.0).pow(
                                    masked_kd_drop_weight_power
                                )
                                drop_loss_mask = drop_loss_mask & (drop_loss_weights > 0.0)
                            teacher_drop_for_stats = teacher_drop.detach().float()
                            masked_kd_teacher_drop_sum = teacher_drop_for_stats.sum()
                            masked_kd_teacher_drop_count = torch.tensor(
                                float(teacher_drop_for_stats.numel()), device=self.device, dtype=torch.float32
                            )
                            if teacher_drop_for_stats.numel() > 0:
                                masked_kd_teacher_drop_max = teacher_drop_for_stats.max()
                            masked_kd_drop_active_count = drop_loss_mask.to(dtype=torch.float32).sum()
                            if bool(drop_loss_mask.any().detach().item()):
                                selected_student_drop = student_drop[drop_loss_mask]
                                selected_teacher_drop = teacher_drop[drop_loss_mask]
                                selected_weights = (
                                    drop_loss_weights[drop_loss_mask]
                                    if drop_loss_weights is not None else None
                                )
                                if masked_kd_drop_loss_type == "drop_order_contrastive":
                                    masked_kd_drop_loss = _drop_order_contrastive_loss(
                                        selected_student_drop,
                                        selected_teacher_drop,
                                        selected_weights,
                                    )
                                elif masked_kd_drop_loss_type == "drop_listwise_kl":
                                    masked_kd_drop_loss = _drop_listwise_kl_loss(
                                        selected_student_drop,
                                        selected_teacher_drop,
                                        selected_weights,
                                    )
                                elif masked_kd_drop_loss_type in {
                                    "prob_shift",
                                    "centered_logit_response",
                                    "pairwise_preference_shift",
                                }:
                                    response_real_teacher_scores = side_teacher_scores
                                    response_masked_teacher_scores = masked_teacher_scores
                                    if masked_kd_drop_teacher_real_scores_key:
                                        if masked_kd_drop_teacher_real_scores_key in masked_kd_side_batch:
                                            response_real_teacher_scores = masked_kd_side_batch[
                                                masked_kd_drop_teacher_real_scores_key
                                            ].to(self.device, non_blocking=True).float()
                                        elif masked_kd_drop_teacher_real_scores_key in masked_kd_mask_batch:
                                            response_real_teacher_scores = masked_kd_mask_batch[
                                                masked_kd_drop_teacher_real_scores_key
                                            ].to(self.device, non_blocking=True).float()
                                        else:
                                            raise ValueError(
                                                "MaskedKD drop_teacher_real_scores_key="
                                                f"{masked_kd_drop_teacher_real_scores_key!r} was not found "
                                                "in side or masked batch"
                                            )
                                    if masked_kd_drop_teacher_masked_scores_key:
                                        if masked_kd_drop_teacher_masked_scores_key in masked_kd_mask_batch:
                                            response_masked_teacher_scores = masked_kd_mask_batch[
                                                masked_kd_drop_teacher_masked_scores_key
                                            ].to(self.device, non_blocking=True).float()
                                        elif masked_kd_drop_teacher_masked_scores_key in masked_kd_side_batch:
                                            response_masked_teacher_scores = masked_kd_side_batch[
                                                masked_kd_drop_teacher_masked_scores_key
                                            ].to(self.device, non_blocking=True).float()
                                        else:
                                            raise ValueError(
                                                "MaskedKD drop_teacher_masked_scores_key="
                                                f"{masked_kd_drop_teacher_masked_scores_key!r} was not found "
                                                "in side or masked batch"
                                            )
                                    drop_loss_values = _evidence_response_loss_values(
                                        side_student_scores,
                                        masked_student_scores,
                                        response_real_teacher_scores,
                                        response_masked_teacher_scores,
                                    )[drop_loss_mask]
                                elif masked_kd_drop_loss_type == "mse":
                                    drop_loss_values = (selected_student_drop - selected_teacher_drop).pow(2)
                                elif masked_kd_drop_loss_type in {"smooth_l1", "huber"}:
                                    drop_loss_values = F.smooth_l1_loss(
                                        selected_student_drop,
                                        selected_teacher_drop,
                                        beta=masked_kd_drop_huber_beta,
                                        reduction="none",
                                    )
                                elif masked_kd_drop_loss_type == "lower_bound_hinge":
                                    drop_loss_values = F.relu(
                                        selected_teacher_drop - selected_student_drop - masked_kd_drop_hinge_margin
                                    )
                                elif masked_kd_drop_loss_type in {"pairwise_logistic", "adaptive_rank"}:
                                    adaptive_margin = (
                                        masked_kd_drop_hinge_margin
                                        + masked_kd_drop_rank_margin_scale * selected_teacher_drop
                                    )
                                    drop_loss_values = F.softplus(adaptive_margin - selected_student_drop)
                                else:
                                    raise AssertionError(
                                        f"unexpected MaskedKD drop_loss_type: {masked_kd_drop_loss_type}"
                                    )
                                if masked_kd_drop_loss_type not in {"drop_order_contrastive", "drop_listwise_kl"}:
                                    if selected_weights is not None:
                                        weight_sum = selected_weights.sum().clamp_min(1e-12)
                                        masked_kd_drop_loss = (drop_loss_values * selected_weights).sum() / weight_sum
                                    else:
                                        masked_kd_drop_loss = drop_loss_values.mean()
                            else:
                                masked_kd_drop_loss = student_drop.sum() * 0.0
                            masked_kd_side_loss = masked_kd_side_loss + masked_kd_lambda_drop * masked_kd_drop_loss
                    masked_kd_side_loss_sum = masked_kd_side_loss.detach() * masked_kd_side_count.detach()
                    masked_kd_real_ce_sum = masked_kd_real_ce.detach() * masked_kd_side_count.detach()
                    if masked_kd_real_kd is not None:
                        masked_kd_real_kd_sum = masked_kd_real_kd.detach() * masked_kd_side_count.detach()
                    if masked_kd_mask_kd is not None:
                        masked_kd_mask_kd_sum = masked_kd_mask_kd.detach() * masked_kd_side_count.detach()
                    if masked_kd_drop_loss is not None:
                        masked_kd_drop_loss_sum = masked_kd_drop_loss.detach() * masked_kd_side_count.detach()
                    if masked_kd_normalize:
                        loss = (loss + masked_kd_rho * masked_kd_side_loss) / (1.0 + masked_kd_rho)
                    else:
                        loss = loss + masked_kd_rho * masked_kd_side_loss

                gradient_probe_row = None
                if (
                    gradient_probe_enabled
                    and (total_step + 1) % gradient_probe_interval == 0
                ):
                    gradient_probe_row = {
                        "schema_version": "audio_alignkd_gradient_probe_v1",
                        "experiment": self.config.get("myconfig"),
                        "optimizer_update": int(total_step + 1),
                        "batch_index": int(ii),
                        "align_count_local": int(
                            kd_mask.sum().detach().item() if kd_mask is not None else 0
                        ),
                    }
                    ce_mapper_norm = loss_gradient_l2_norm(
                        lm_loss, gradient_probe_mapper_params
                    )
                    gradient_probe_row["ce_mapper_norm"] = float(ce_mapper_norm.item())

                    auxiliary_losses = []
                    if naive_kd_loss is not None:
                        auxiliary_losses.append(
                            ("abcd", naive_kd_effective_lambda * naive_kd_loss)
                        )
                    if retention_kd_loss is not None:
                        auxiliary_losses.append(
                            (
                                "retention",
                                retention_kd_lambda * retention_kd_loss,
                            )
                        )
                    if sequence_retention_kd_loss is not None:
                        auxiliary_losses.append(
                            (
                                "sequence_retention",
                                sequence_retention_lambda
                                * sequence_retention_kd_loss,
                            )
                        )
                    if alignkd_attention_loss is not None:
                        auxiliary_losses.append(
                            (
                                "attention",
                                alignkd_lambda_attention * alignkd_attention_loss,
                            )
                        )
                    if alignkd_feature_all_loss is not None:
                        auxiliary_losses.append(
                            (
                                "feature_all",
                                alignkd_lambda_feature_all * alignkd_feature_all_loss,
                            )
                        )
                    if alignkd_feature_soft_loss is not None:
                        auxiliary_losses.append(
                            (
                                "feature_soft",
                                alignkd_lambda_feature_soft * alignkd_feature_soft_loss,
                            )
                        )
                    if alignkd_feature_top16_loss is not None:
                        auxiliary_losses.append(
                            (
                                "feature_top16",
                                alignkd_lambda_feature_top16
                                * alignkd_feature_top16_loss,
                            )
                        )

                    has_align = (
                        kd_mask is None
                        or bool(kd_mask.any().detach().item())
                        or (
                            sequence_retention_active_mask is not None
                            and bool(
                                sequence_retention_active_mask.any().detach().item()
                            )
                        )
                    )
                    if has_align:
                        ce_mapper_value = float(ce_mapper_norm.item())
                        for auxiliary_name, auxiliary_loss in auxiliary_losses:
                            auxiliary_mapper_norm = loss_gradient_l2_norm(
                                auxiliary_loss, gradient_probe_mapper_params
                            )
                            auxiliary_mapper_value = float(
                                auxiliary_mapper_norm.item()
                            )
                            gradient_probe_row[
                                f"{auxiliary_name}_mapper_norm"
                            ] = auxiliary_mapper_value
                            gradient_probe_row[
                                f"{auxiliary_name}_to_ce_mapper_ratio"
                            ] = (
                                auxiliary_mapper_value / ce_mapper_value
                                if ce_mapper_value > 0.0
                                else None
                            )

                        if alignkd_attention_loss is not None:
                            ce_qk_norm = loss_gradient_l2_norm(
                                lm_loss, gradient_probe_layer0_qk_params
                            )
                            attention_qk_norm = loss_gradient_l2_norm(
                                alignkd_lambda_attention
                                * alignkd_attention_loss,
                                gradient_probe_layer0_qk_params,
                            )
                            ce_qk_value = float(ce_qk_norm.item())
                            attention_qk_value = float(attention_qk_norm.item())
                            gradient_probe_row["ce_layer0_qk_norm"] = ce_qk_value
                            gradient_probe_row[
                                "attention_layer0_qk_norm"
                            ] = attention_qk_value
                            gradient_probe_row[
                                "attention_to_ce_layer0_qk_ratio"
                            ] = (
                                attention_qk_value / ce_qk_value
                                if ce_qk_value > 0.0
                                else None
                            )

                finite_loss = torch.isfinite(loss.detach()).to(dtype=torch.float32)
                finite_loss_count = self.distributed.all_reduce(finite_loss, average=False).item()
                if finite_loss_count < self.distributed.world_size():
                    if self.distributed.rank() == 0:
                        self.logger.warning(
                            "Skipping non-finite loss at epoch=%d step=%d finite_ranks=%d/%d",
                            epoch + 1, ii + 1, int(finite_loss_count), self.distributed.world_size(),
                        )
                    optimizer.zero_grad(set_to_none=True)
                    micro_step_in_accum = 0
                    self.distributed.barrier()
                    del batch_audio1, batch_audio2, batch_input, batch_answer
                    del input_dict, model_outputs, finite_loss
                    if masked_kd_side_input_dict is not None:
                        del masked_kd_side_input_dict, masked_kd_side_outputs
                    if masked_kd_mask_input_dict is not None:
                        del masked_kd_mask_input_dict, masked_kd_mask_outputs
                    if listen_shuffle_loss is not None:
                        del shuffled_outputs
                    tqdm_handler.update(1)
                    last_batch_index = ii + 1
                    if max_train_batches > 0 and ii + 1 >= max_train_batches:
                        stop_training = True
                        self.logger.info("max_train_batches=%d reached; stopping training", max_train_batches)
                        break
                    if max_optimizer_updates > 0 and total_step >= max_optimizer_updates:
                        stop_training = True
                        self.logger.info(
                            "max_optimizer_updates=%d reached; stopping training",
                            max_optimizer_updates,
                        )
                        break
                    continue

                micro_step_in_accum += 1
                should_optimizer_step = micro_step_in_accum >= gradient_accumulation_steps
                scaled_loss = loss / gradient_accumulation_steps
                replay_gradient_projection_stats = None
                if (
                    should_optimizer_step
                    and masked_kd_drop_gradient_cosine_interval > 0
                    and masked_kd_drop_loss is not None
                    and masked_kd_grad_cosine_params
                    and (total_step + 1) % masked_kd_drop_gradient_cosine_interval == 0
                ):
                    masked_kd_grad_cosine = _loss_gradient_cosine(
                        masked_kd_main_loss_for_grad_cosine,
                        masked_kd_drop_loss,
                    )
                if replay_gradient_projection_enabled:
                    if not should_optimizer_step or gradient_accumulation_steps != 1:
                        raise RuntimeError(
                            "replay gradient projection requires one microbatch per optimizer step"
                        )
                    if replay_correction_loss is None or replay_retention_loss is None:
                        raise RuntimeError("replay gradient-projection losses were not constructed")
                    projection_parameters = tuple(
                        (name, parameter)
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    )
                    grad_scaler.scale(replay_correction_loss).backward(retain_graph=True)
                    correction_gradients = tuple(
                        parameter.grad.detach().clone()
                        if parameter.grad is not None
                        else None
                        for _, parameter in projection_parameters
                    )
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(replay_retention_loss).backward()
                    replay_gradient_projection_stats = project_correction_gradients_(
                        projection_parameters,
                        correction_gradients,
                        eps=float(replay_gradient_projection["eps"]),
                    )
                    del correction_gradients, projection_parameters
                else:
                    grad_scaler.scale(scaled_loss).backward()

                total_norm = None
                grad_scale = float(grad_scaler.get_scale())
                if should_optimizer_step:
                    grad_scaler.unscale_(optimizer)  # to use the same max_grad_norm value for gradient clipping

                    if gradient_probe_row is not None:
                        gradient_probe_row["total_mapper_grad_norm"] = float(
                            parameter_gradient_l2_norm(
                                gradient_probe_mapper_params
                            ).item()
                        )
                        if gradient_probe_layer0_qk_params:
                            gradient_probe_row["total_layer0_qk_grad_norm"] = float(
                                parameter_gradient_l2_norm(
                                    gradient_probe_layer0_qk_params
                                ).item()
                            )
                        with open(gradient_probe_path, "a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(gradient_probe_row, sort_keys=True) + "\n"
                            )
                            handle.flush()
                            os.fsync(handle.fileno())
                        self.logger.info(
                            "Audio-AlignKD gradient probe: %s",
                            json.dumps(gradient_probe_row, sort_keys=True),
                        )

                    total_norm, grad_scale = grad_norm_tracker.track_and_clip_(list(model.named_parameters()))

                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    micro_step_in_accum = 0
                    if step_lr_scheduler and lr_scheduler is not None:
                        lr_scheduler.step()

                if not should_optimizer_step:
                    del batch_audio1, batch_audio2, batch_input, batch_answer
                    del input_dict, model_outputs
                    if masked_kd_side_input_dict is not None:
                        del masked_kd_side_input_dict, masked_kd_side_outputs
                    if masked_kd_mask_input_dict is not None:
                        del masked_kd_mask_input_dict, masked_kd_mask_outputs
                    if listen_shuffle_loss is not None:
                        del shuffled_outputs
                    next_batch_index = ii + 1
                    last_batch_index = next_batch_index
                    tqdm_handler.update(1)
                    if max_train_batches > 0 and ii + 1 >= max_train_batches:
                        stop_training = True
                        self.logger.info("max_train_batches=%d reached; stopping training", max_train_batches)
                        break
                    continue

                loss = self.distributed.all_reduce(loss.detach()).item()
                lm_loss_value = self.distributed.all_reduce(lm_loss.detach()).item()
                naive_kd_loss_value = None
                naive_kd_count_value = None
                if naive_kd_count is not None:
                    naive_kd_loss_sum_value = self.distributed.all_reduce(
                        naive_kd_loss_sum.detach().clone(),
                        average=False,
                    ).item()
                    naive_kd_count_value = self.distributed.all_reduce(
                        naive_kd_count.detach().clone(),
                        average=False,
                    ).item()
                    naive_kd_loss_value = (
                        naive_kd_loss_sum_value / naive_kd_count_value
                        if naive_kd_count_value > 0 else 0.0
                    )
                retention_kd_loss_value = None
                retention_kd_count_value = None
                if retention_kd_count is not None:
                    retention_kd_loss_sum_value = self.distributed.all_reduce(
                        retention_kd_loss_sum.detach().clone(),
                        average=False,
                    ).item()
                    retention_kd_count_value = self.distributed.all_reduce(
                        retention_kd_count.detach().clone(),
                        average=False,
                    ).item()
                    retention_kd_loss_value = (
                        retention_kd_loss_sum_value / retention_kd_count_value
                        if retention_kd_count_value > 0
                        else 0.0
                    )
                replay_ce_value_tuple = None
                replay_first_token_gold_ce_value_tuple = None
                replay_kl_value_tuple = None
                if replay_enabled:
                    replay_ce_value_tuple = tuple(
                        self.distributed.all_reduce(value.detach()).item()
                        for value in replay_ce_values
                    )
                    replay_kl_value_tuple = tuple(
                        self.distributed.all_reduce(value.detach()).item()
                        for value in replay_kl_values
                    )
                    if replay_first_token_gold_ce_values is not None:
                        replay_first_token_gold_ce_value_tuple = tuple(
                            self.distributed.all_reduce(value.detach()).item()
                            for value in replay_first_token_gold_ce_values
                        )
                sequence_retention_kd_loss_value = None
                sequence_retention_sample_count_value = None
                sequence_retention_token_count_value = None
                if sequence_retention_sample_count is not None:
                    sequence_retention_kd_loss_sum_value = self.distributed.all_reduce(
                        sequence_retention_kd_loss_sum.detach().clone(),
                        average=False,
                    ).item()
                    sequence_retention_sample_count_value = self.distributed.all_reduce(
                        sequence_retention_sample_count.detach().clone(),
                        average=False,
                    ).item()
                    sequence_retention_token_count_value = self.distributed.all_reduce(
                        sequence_retention_token_count.detach().clone(),
                        average=False,
                    ).item()
                    sequence_retention_kd_loss_value = (
                        sequence_retention_kd_loss_sum_value
                        / sequence_retention_sample_count_value
                        if sequence_retention_sample_count_value > 0
                        else 0.0
                    )
                alignkd_attention_loss_value = None
                alignkd_feature_all_loss_value = None
                alignkd_feature_soft_loss_value = None
                alignkd_feature_top16_loss_value = None
                alignkd_count_value = None
                if alignkd_count is not None:
                    alignkd_count_value = self.distributed.all_reduce(
                        alignkd_count.detach().clone(), average=False
                    ).item()

                    def _reduce_alignkd_sum(local_sum):
                        if local_sum is None:
                            return None
                        global_sum = self.distributed.all_reduce(
                            local_sum.detach().clone(), average=False
                        ).item()
                        return (
                            global_sum / alignkd_count_value
                            if alignkd_count_value > 0
                            else 0.0
                        )

                    alignkd_attention_loss_value = _reduce_alignkd_sum(
                        alignkd_attention_loss_sum
                    )
                    alignkd_feature_all_loss_value = _reduce_alignkd_sum(
                        alignkd_feature_all_loss_sum
                    )
                    alignkd_feature_soft_loss_value = _reduce_alignkd_sum(
                        alignkd_feature_soft_loss_sum
                    )
                    alignkd_feature_top16_loss_value = _reduce_alignkd_sum(
                        alignkd_feature_top16_loss_sum
                    )
                masked_kd_side_loss_value = None
                masked_kd_real_ce_value = None
                masked_kd_real_kd_value = None
                masked_kd_mask_kd_value = None
                masked_kd_drop_loss_value = None
                masked_kd_teacher_drop_mean_value = None
                masked_kd_teacher_drop_max_value = None
                masked_kd_drop_active_frac_value = None
                masked_kd_side_count_value = None
                if masked_kd_side_count is not None:
                    masked_kd_side_loss_sum_value = self.distributed.all_reduce(
                        masked_kd_side_loss_sum.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_real_ce_sum_value = self.distributed.all_reduce(
                        masked_kd_real_ce_sum.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_real_kd_sum_value = self.distributed.all_reduce(
                        masked_kd_real_kd_sum.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_mask_kd_sum_value = self.distributed.all_reduce(
                        masked_kd_mask_kd_sum.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_drop_loss_sum_value = self.distributed.all_reduce(
                        masked_kd_drop_loss_sum.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_teacher_drop_sum_value = self.distributed.all_reduce(
                        masked_kd_teacher_drop_sum.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_teacher_drop_count_value = self.distributed.all_reduce(
                        masked_kd_teacher_drop_count.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_teacher_drop_max_value = _all_reduce_max_scalar(
                        masked_kd_teacher_drop_max.detach().clone(),
                    ).item()
                    masked_kd_drop_active_count_value = self.distributed.all_reduce(
                        masked_kd_drop_active_count.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_side_count_value = self.distributed.all_reduce(
                        masked_kd_side_count.detach().clone(),
                        average=False,
                    ).item()
                    masked_kd_side_loss_value = (
                        masked_kd_side_loss_sum_value / masked_kd_side_count_value
                        if masked_kd_side_count_value > 0 else 0.0
                    )
                    masked_kd_real_ce_value = (
                        masked_kd_real_ce_sum_value / masked_kd_side_count_value
                        if masked_kd_side_count_value > 0 else 0.0
                    )
                    masked_kd_real_kd_value = (
                        masked_kd_real_kd_sum_value / masked_kd_side_count_value
                        if masked_kd_side_count_value > 0 else 0.0
                    )
                    masked_kd_mask_kd_value = (
                        masked_kd_mask_kd_sum_value / masked_kd_side_count_value
                        if masked_kd_side_count_value > 0 else 0.0
                    )
                    masked_kd_drop_loss_value = (
                        masked_kd_drop_loss_sum_value / masked_kd_side_count_value
                        if masked_kd_side_count_value > 0 else 0.0
                    )
                    masked_kd_teacher_drop_mean_value = (
                        masked_kd_teacher_drop_sum_value / masked_kd_teacher_drop_count_value
                        if masked_kd_teacher_drop_count_value > 0 else 0.0
                    )
                    masked_kd_drop_active_frac_value = (
                        masked_kd_drop_active_count_value / masked_kd_teacher_drop_count_value
                        if masked_kd_teacher_drop_count_value > 0 else 0.0
                    )
                listen_shuffle_loss_value = None
                listen_shuffle_margin_loss_value = None
                if listen_shuffle_loss is not None:
                    listen_shuffle_loss_value = self.distributed.all_reduce(listen_shuffle_loss.detach()).item()
                if listen_shuffle_margin_loss is not None:
                    listen_shuffle_margin_loss_value = self.distributed.all_reduce(listen_shuffle_margin_loss.detach()).item()
                accerr_epo += loss

                if loss_tracker is not None:
                    loss_tracker.track_loss(loss)

                loss_history["optimizer_step"].append(current_optimizer_step)
                loss_history["loss"].append(loss)
                loss_history["lm_loss"].append(lm_loss_value)
                loss_history["naive_kd_active"].append(naive_kd_active)
                loss_history["naive_kd_effective_lambda"].append(
                    naive_kd_effective_lambda
                )
                if naive_kd_loss_value is not None:
                    loss_history["naive_kd_loss"].append(naive_kd_loss_value)
                if naive_kd_count_value is not None:
                    loss_history["naive_kd_count"].append(naive_kd_count_value)
                if retention_kd_loss_value is not None:
                    loss_history["retention_kd_loss"].append(
                        retention_kd_loss_value
                    )
                if retention_kd_count_value is not None:
                    loss_history["retention_kd_count"].append(
                        retention_kd_count_value
                    )
                if replay_ce_value_tuple is not None:
                    for group_name, ce_value, kl_value, kd_weight in zip(
                        replay_group_names,
                        replay_ce_value_tuple,
                        replay_kl_value_tuple,
                        replay_kd_weights,
                    ):
                        loss_history.setdefault(f"replay_ce_{group_name}", []).append(ce_value)
                        if kd_weight > 0.0:
                            loss_history.setdefault(f"replay_kl_{group_name}", []).append(kl_value)
                if replay_first_token_gold_ce_value_tuple is not None:
                    for group_name, choice_ce_value, choice_ce_weight in zip(
                        replay_group_names,
                        replay_first_token_gold_ce_value_tuple,
                        replay_first_token_gold_ce_weights,
                    ):
                        if choice_ce_weight > 0.0:
                            loss_history.setdefault(
                                f"replay_first_token_gold_ce_{group_name}", []
                            ).append(choice_ce_value)
                if replay_gradient_projection_stats is not None:
                    loss_history["replay_projection_conflict"].append(
                        bool(replay_gradient_projection_stats["conflict"])
                    )
                    loss_history["replay_projection_cosine_before"].append(
                        replay_gradient_projection_stats["cosine_before"]
                    )
                    loss_history[
                        "replay_projection_correction_removed_fraction"
                    ].append(
                        replay_gradient_projection_stats[
                            "correction_removed_fraction"
                        ]
                    )
                    loss_history["replay_projection_correction_norm"].append(
                        replay_gradient_projection_stats["correction_norm"]
                    )
                    loss_history["replay_projection_retention_norm"].append(
                        replay_gradient_projection_stats["retention_norm"]
                    )
                if sequence_retention_kd_loss_value is not None:
                    loss_history["sequence_retention_kd_loss"].append(
                        sequence_retention_kd_loss_value
                    )
                if sequence_retention_sample_count_value is not None:
                    loss_history["sequence_retention_sample_count"].append(
                        sequence_retention_sample_count_value
                    )
                if sequence_retention_token_count_value is not None:
                    loss_history["sequence_retention_token_count"].append(
                        sequence_retention_token_count_value
                    )
                if alignkd_attention_loss_value is not None:
                    loss_history["alignkd_attention_loss"].append(
                        alignkd_attention_loss_value
                    )
                if alignkd_feature_all_loss_value is not None:
                    loss_history["alignkd_feature_all_loss"].append(
                        alignkd_feature_all_loss_value
                    )
                if alignkd_feature_soft_loss_value is not None:
                    loss_history["alignkd_feature_soft_loss"].append(
                        alignkd_feature_soft_loss_value
                    )
                if alignkd_feature_top16_loss_value is not None:
                    loss_history["alignkd_feature_top16_loss"].append(
                        alignkd_feature_top16_loss_value
                    )
                if alignkd_count_value is not None:
                    loss_history["alignkd_count"].append(alignkd_count_value)
                if masked_kd_side_loss_value is not None:
                    loss_history["masked_kd_side_loss"].append(masked_kd_side_loss_value)
                if masked_kd_real_ce_value is not None:
                    loss_history["masked_kd_real_ce"].append(masked_kd_real_ce_value)
                if masked_kd_real_kd_value is not None:
                    loss_history["masked_kd_real_kd"].append(masked_kd_real_kd_value)
                if masked_kd_mask_kd_value is not None:
                    loss_history["masked_kd_mask_kd"].append(masked_kd_mask_kd_value)
                if masked_kd_drop_loss_value is not None:
                    loss_history["masked_kd_drop_loss"].append(masked_kd_drop_loss_value)
                if masked_kd_teacher_drop_mean_value is not None:
                    loss_history["masked_kd_teacher_drop_mean"].append(masked_kd_teacher_drop_mean_value)
                if masked_kd_teacher_drop_max_value is not None:
                    loss_history["masked_kd_teacher_drop_max"].append(masked_kd_teacher_drop_max_value)
                if masked_kd_drop_active_frac_value is not None:
                    loss_history["masked_kd_drop_active_frac"].append(masked_kd_drop_active_frac_value)
                if masked_kd_grad_cosine is not None:
                    loss_history["masked_kd_grad_cosine"].append(masked_kd_grad_cosine)
                if masked_kd_side_count_value is not None:
                    loss_history["masked_kd_side_count"].append(masked_kd_side_count_value)
                if listen_shuffle_loss_value is not None:
                    loss_history["listen_shuffle_loss"].append(listen_shuffle_loss_value)
                if listen_shuffle_margin_loss_value is not None:
                    loss_history["listen_shuffle_margin_loss"].append(listen_shuffle_margin_loss_value)
                loss_history["total_grad_norm"].append(total_norm)
                loss_history["grad_scale"].append(grad_scale)
                loss_history["lr"].append(optimizer.param_groups[0]["lr"])
                if adapter_stats is not None:
                    loss_history["adapter_alpha"].append(adapter_stats["alpha"])
                    loss_history["adapter_impact_ratio"].append(adapter_stats["impact_ratio"])
                if output_control_enabled:
                    loss_history["output_control_phase"].append(
                        output_control_phase_index
                    )
                    loss_history["output_control_structure_weight"].append(
                        output_control_structure_weight
                    )
                    loss_history["output_control_reasoning_weight"].append(
                        output_control_reasoning_weight
                    )
                    loss_history["output_control_answer_weight"].append(
                        output_control_answer_weight
                    )

                # Print log for current step
                total_step += 1
                if self.distributed.rank() == 0:
                    self.logger.info(
                        "AUDIO_ALIGNKD_KD_SCHEDULE %s",
                        json.dumps(
                            {
                                "optimizer_step": total_step,
                                "naive_kd_active": naive_kd_active,
                                "naive_kd_effective_lambda": naive_kd_effective_lambda,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                if total_step % self.config["train"]["log_step"] == 0 and self.distributed.rank() == 0:
                    errdict = {
                        "accerr_epo": accerr_epo,
                        "loss": loss,
                        "lm_loss": lm_loss_value,
                    }
                    if naive_kd_loss_value is not None:
                        errdict["naive_kd_loss"] = naive_kd_loss_value
                    if naive_kd_count_value is not None:
                        errdict["naive_kd_count"] = naive_kd_count_value
                    if retention_kd_loss_value is not None:
                        errdict["retention_kd_loss"] = retention_kd_loss_value
                    if retention_kd_count_value is not None:
                        errdict["retention_kd_count"] = retention_kd_count_value
                    if replay_ce_value_tuple is not None:
                        for group_name, ce_value, kl_value, kd_weight in zip(
                            replay_group_names,
                            replay_ce_value_tuple,
                            replay_kl_value_tuple,
                            replay_kd_weights,
                        ):
                            errdict[f"replay_ce_{group_name}"] = ce_value
                            if kd_weight > 0.0:
                                errdict[f"replay_kl_{group_name}"] = kl_value
                    if replay_first_token_gold_ce_value_tuple is not None:
                        for group_name, choice_ce_value, choice_ce_weight in zip(
                            replay_group_names,
                            replay_first_token_gold_ce_value_tuple,
                            replay_first_token_gold_ce_weights,
                        ):
                            if choice_ce_weight > 0.0:
                                errdict[
                                    f"replay_first_token_gold_ce_{group_name}"
                                ] = choice_ce_value
                    if replay_gradient_projection_stats is not None:
                        errdict.update(
                            {
                                "replay_projection_conflict": bool(
                                    replay_gradient_projection_stats["conflict"]
                                ),
                                "replay_projection_cosine_before": replay_gradient_projection_stats[
                                    "cosine_before"
                                ],
                                "replay_projection_correction_removed_fraction": replay_gradient_projection_stats[
                                    "correction_removed_fraction"
                                ],
                            }
                        )
                    if sequence_retention_kd_loss_value is not None:
                        errdict["sequence_retention_kd_loss"] = (
                            sequence_retention_kd_loss_value
                        )
                    if sequence_retention_sample_count_value is not None:
                        errdict["sequence_retention_sample_count"] = (
                            sequence_retention_sample_count_value
                        )
                    if sequence_retention_token_count_value is not None:
                        errdict["sequence_retention_token_count"] = (
                            sequence_retention_token_count_value
                        )
                    if alignkd_attention_loss_value is not None:
                        errdict["alignkd_attention_loss"] = alignkd_attention_loss_value
                    if alignkd_feature_all_loss_value is not None:
                        errdict["alignkd_feature_all_loss"] = alignkd_feature_all_loss_value
                    if alignkd_feature_soft_loss_value is not None:
                        errdict["alignkd_feature_soft_loss"] = alignkd_feature_soft_loss_value
                    if alignkd_feature_top16_loss_value is not None:
                        errdict["alignkd_feature_top16_loss"] = (
                            alignkd_feature_top16_loss_value
                        )
                    if alignkd_count_value is not None:
                        errdict["alignkd_count"] = alignkd_count_value
                    if masked_kd_side_loss_value is not None:
                        errdict["masked_kd_side_loss"] = masked_kd_side_loss_value
                    if masked_kd_real_ce_value is not None:
                        errdict["masked_kd_real_ce"] = masked_kd_real_ce_value
                    if masked_kd_real_kd_value is not None:
                        errdict["masked_kd_real_kd"] = masked_kd_real_kd_value
                    if masked_kd_mask_kd_value is not None:
                        errdict["masked_kd_mask_kd"] = masked_kd_mask_kd_value
                    if masked_kd_drop_loss_value is not None:
                        errdict["masked_kd_drop_loss"] = masked_kd_drop_loss_value
                    if masked_kd_side_count_value is not None:
                        errdict["masked_kd_side_count"] = masked_kd_side_count_value
                    if listen_shuffle_loss_value is not None:
                        errdict["listen_shuffle_loss"] = listen_shuffle_loss_value
                    if listen_shuffle_margin_loss_value is not None:
                        errdict["listen_shuffle_margin"] = listen_shuffle_margin_loss_value

                    errstr = ", ".join(
                        "{}: {:6.3f}(e-6)".format(k, v * 1e6) for k, v in errdict.items()
                    )
                    self.logger.info(
                        "Epoch [%3d/%3d], Step [%3d/%3d], %s",
                        epoch + 1, self.config["train"]["num_epochs"], ii + 1,
                        num_batches_per_epoch, errstr
                    )
                    diagdict = {}
                    if masked_kd_teacher_drop_mean_value is not None:
                        diagdict["teacher_drop_mean"] = masked_kd_teacher_drop_mean_value
                    if masked_kd_teacher_drop_max_value is not None:
                        diagdict["teacher_drop_max"] = masked_kd_teacher_drop_max_value
                    if masked_kd_drop_active_frac_value is not None:
                        diagdict["drop_active_frac"] = masked_kd_drop_active_frac_value
                    if masked_kd_grad_cosine is not None:
                        diagdict["grad_cosine"] = masked_kd_grad_cosine
                    if diagdict:
                        self.logger.info(
                            "MaskedKD diagnostics: %s",
                            ", ".join("{}: {:.6f}".format(k, v) for k, v in diagdict.items()),
                        )
                    if adapter_stats is not None:
                        self.logger.info(
                            "Adapter stats: %s",
                            self._format_adapter_stats(adapter_stats),
                        )
                    tqdm_handler.update(1)

                del batch_audio1, batch_audio2, batch_input, batch_answer
                del input_dict, model_outputs
                if masked_kd_side_input_dict is not None:
                    del masked_kd_side_input_dict, masked_kd_side_outputs
                if masked_kd_mask_input_dict is not None:
                    del masked_kd_mask_input_dict, masked_kd_mask_outputs
                if listen_shuffle_loss is not None:
                    del shuffled_outputs

                next_batch_index = ii + 1
                last_batch_index = next_batch_index
                signal_stop_requested = (
                    self._distributed_stop_requested(stop_requested)
                    if signal_checkpointing_enabled else False
                )
                is_mid_epoch_checkpoint = (
                    next_batch_index in mid_epoch_checkpoint_batches
                    or total_step in checkpoint_optimizer_steps
                )
                if is_mid_epoch_checkpoint or signal_stop_requested:
                    training_state_fname = self._get_training_checkpoint_name(
                        epoch,
                        next_batch_index,
                        completed_epoch=False,
                        optimizer_step=total_step,
                    )
                    training_state_fpath = os.path.join(
                        self.config["save_dir"],
                        "training-state-" + training_state_fname,
                    )
                    if self.distributed.rank() == 0:
                        self.logger.info(
                            "Saving training state to: %s Total training time: %f hours",
                            training_state_fpath,
                            (time.time() - t0) / 3600.0,
                        )
                        self._save_training_state(
                            training_state_fpath,
                            model,
                            optimizer,
                            lr_scheduler,
                            grad_scaler,
                            grad_norm_tracker,
                            epoch=epoch,
                            next_batch_index=next_batch_index,
                            total_step=total_step,
                            num_batches_per_epoch=num_batches_per_epoch,
                            completed_epoch=False,
                        )
                    self.distributed.barrier()
                if signal_stop_requested:
                    stop_due_to_signal = True
                    stop_training = True
                    self.logger.info("Stop signal handled; saved training state and will exit training loop")
                    break

                if self.config.get("functiontest", False):
                    stop_training = True
                    self.logger.info("functiontest enabled; stopping after one training batch")
                    break
                if max_optimizer_updates > 0 and total_step >= max_optimizer_updates:
                    stop_training = True
                    self.logger.info(
                        "max_optimizer_updates=%d reached; stopping training",
                        max_optimizer_updates,
                    )
                    break
                if max_train_batches > 0 and ii + 1 >= max_train_batches:
                    stop_training = True
                    self.logger.info("max_train_batches=%d reached; stopping training", max_train_batches)
                    break


            metrics_train["accerr"] = float(accerr_epo)
            if accerr_epo < lowest_accerr_epo and self.distributed.rank() == 0:
                lowest_accerr_epo = accerr_epo
                self.logger.info("The lowest accumulated error so far is {%f}", accerr_epo)

            # Save the MODEL checkpoint
            is_save_epoch = (epoch + 1) % self.config["train"]["sav_per_num_epochs"] == 0
            epoch_completed = last_batch_index >= num_batches_per_epoch
            if is_save_epoch:
                if stop_due_to_signal:
                    if self.distributed.rank() == 0:
                        self.logger.info(
                            "Skipping model checkpoint save because stop signal was handled; "
                            "latest training-state checkpoint is resumable."
                        )
                    self.distributed.barrier()
                    if stop_training:
                        break
                    continue

                fname = self._get_checkpoint_name(epoch)
                save_dir = self.config["save_dir"]

                model_fpath = os.path.join(save_dir, "model-" + fname)
                metrics_train["checkpoint"] = model_fpath

                if self.distributed.rank() == 0:
                    os.makedirs(save_dir, exist_ok=True)

                    self.logger.info("Saving model to: %s Total training time: %f hours",
                                        model_fpath, (time.time() - t0) / 3600.0)

                    self._save_model_state(model_fpath, model)

                    training_state_fname = self._get_training_checkpoint_name(
                        epoch,
                        num_batches_per_epoch if epoch_completed else last_batch_index,
                        completed_epoch=epoch_completed,
                        optimizer_step=total_step,
                    )
                    training_state_fpath = os.path.join(
                        save_dir,
                        "training-state-" + training_state_fname,
                    )
                    self.logger.info("Saving epoch training state to: %s", training_state_fpath)
                    self._save_training_state(
                        training_state_fpath,
                        model,
                        optimizer,
                        lr_scheduler,
                        grad_scaler,
                        grad_norm_tracker,
                        epoch=epoch,
                        next_batch_index=num_batches_per_epoch if epoch_completed else last_batch_index,
                        total_step=total_step,
                        num_batches_per_epoch=num_batches_per_epoch,
                        completed_epoch=epoch_completed,
                    )

            # distributed: broadcast parameters to ensure that models do not diverge
            self.distributed.broadcast_parameters(model.state_dict())
            self.distributed.broadcast_optimizer_state(optimizer)
            if stop_training:
                break

        self._restore_signal_handlers(previous_signal_handlers)

        if self.distributed.rank() == 0:
            final_alpha = self._get_adapter_alpha(model)
            if final_alpha is not None:
                self.logger.info("adapter.alpha final value: %.8f", final_alpha)
            last_forward_adapter_stats = self._get_adapter_stats(model)
            metrics_fpath = os.path.join(self.config["save_dir"], "metrics.json")
            with open(metrics_fpath, "w") as f:
                json.dump({
                    "model_variant": self.config["model"].get("model_variant", "legacy"),
                    "loss_history": loss_history,
                    "adapter_alpha_final": final_alpha,
                    "last_forward_adapter_stats": last_forward_adapter_stats,
                }, f, indent=2)
            self.logger.info("Wrote training metrics to %s", metrics_fpath)

        if stop_due_to_signal:
            if self.distributed.rank() == 0:
                self.logger.info(
                    "Exiting with code 130 after signal-triggered training-state save; "
                    "resubmit the same EXP_NAME to resume."
                )
            raise SystemExit(130)

    # pylint: disable=too-many-locals
    def evaluate_checkpoint(self):
        self.logger.info("Validate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        model = self.get_model()
        model = model.to(self.device)
        self._load_checkpoint_into_model(
            model,
            self.config["checkpoint_path"],
            strict=True,
        )
        model.eval()

        tasks = self.config["data"]["datafiles"]
        val_score = 0
        eval_results = {}
        for task in tasks:
            self.logger.info("Evaluating task %s", task)
            self.config["data"]["datafiles"] = [task]

            metric = Metric(task, self.config["data"]["sampling_rate"])
            dataset, _, data_loader = self.get_data("datafiles")
            if self.distributed.world_size() > 1:
                shard_indices = list(
                    range(self.distributed.rank(), len(dataset), self.distributed.world_size())
                )
                sharded_dataset = Subset(dataset, shard_indices)
                data_loader = torch.utils.data.DataLoader(
                    sharded_dataset,
                    batch_size=self.config["train"]["batch_size"],
                    num_workers=self.get_num_data_workers(),
                    collate_fn=data_loader.collate_fn,
                    pin_memory=True,
                    drop_last=False,
                    worker_init_fn=self._get_data_worker_init_fn(),
                    persistent_workers=(
                        self.config["train"]["persistent_data_workers"]
                        if self.get_num_data_workers() > 0 else False
                    ),
                )
                self.logger.info(
                    "Distributed eval rank %d/%d processing %d/%d samples",
                    self.distributed.rank(),
                    self.distributed.world_size(),
                    len(sharded_dataset),
                    len(dataset),
                )
            num_batches_per_epoch = len(data_loader)
            tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)

            generations, answers, filepaths, inputs, indices = [], [], [], [], []
            with torch.no_grad():
                for batch_data_dict in tqdm(data_loader):
                    batch_indices = batch_data_dict['index']
                    batch_audio1 = batch_data_dict['waveform1']
                    batch_audio2 = batch_data_dict['waveform2']
                    batch_input = batch_data_dict['input']
                    batch_answer = batch_data_dict['answer']
                    batch_answer_text = batch_data_dict['answer_text']
                    batch_input_text = batch_data_dict['input_text']
                    batch_file_paths = batch_data_dict['file_path1']

                    input_dict = {
                        "audio1":batch_audio1,
                        "audio2": batch_audio2,
                        "input":batch_input,
                        "answer":batch_answer,
                    }
                    if "waveform1_lengths" in batch_data_dict:
                        input_dict["audio1_lengths"] = batch_data_dict["waveform1_lengths"]
                    if "waveform2_lengths" in batch_data_dict:
                        input_dict["audio2_lengths"] = batch_data_dict["waveform2_lengths"]
                    if "ced_hidden" in batch_data_dict:
                        input_dict["ced_hidden"] = batch_data_dict["ced_hidden"]
                    if "ced_hidden_segment_lengths" in batch_data_dict:
                        input_dict["ced_hidden_segment_lengths"] = batch_data_dict["ced_hidden_segment_lengths"]
                    if "ced_hidden_lengths" in batch_data_dict:
                        input_dict["ced_hidden_lengths"] = batch_data_dict["ced_hidden_lengths"]
                    if "beats_hidden" in batch_data_dict:
                        input_dict["beats_hidden"] = batch_data_dict["beats_hidden"]
                    if "beats_hidden_segment_lengths" in batch_data_dict:
                        input_dict["beats_hidden_segment_lengths"] = batch_data_dict["beats_hidden_segment_lengths"]
                    if "beats_hidden_lengths" in batch_data_dict:
                        input_dict["beats_hidden_lengths"] = batch_data_dict["beats_hidden_lengths"]
                    input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                
                    prefix, _, _ = model.generate_prefix_inference(input_dict)
                    generated_text = generate_greedy_batch(
                        model,
                        dataset.tokenizer,
                        embed=prefix,
                        entry_length=int(self.config["model"].get("generation_max_length", 300)),
                    )
                    generations += generated_text
                    answers += batch_answer_text
                    inputs += batch_input_text
                    filepaths += batch_file_paths
                    indices += batch_indices
                    #break

            if self.distributed.world_size() > 1:
                gathered = self.distributed.all_gather_object({
                    "indices": indices,
                    "generations": generations,
                    "answers": answers,
                    "filepaths": filepaths,
                    "inputs": inputs,
                })
                if self.distributed.rank() == 0:
                    rows = []
                    for shard in gathered:
                        rows.extend(zip(
                            shard["indices"],
                            shard["generations"],
                            shard["answers"],
                            shard["filepaths"],
                            shard["inputs"],
                        ))
                    rows.sort(key=lambda item: item[0])
                    if rows:
                        indices, generations, answers, filepaths, inputs = map(list, zip(*rows))
                    else:
                        indices, generations, answers, filepaths, inputs = [], [], [], [], []

                self.distributed.barrier()
                if self.distributed.rank() != 0:
                    continue

            metric.get_metrics(generations, answers, filepaths)
            if self.distributed.rank() == 0:
                self.logger.info("Task %s results", task)
                for key in metric.metrics.keys():
                    self.logger.info("%s: %f", key, metric.metrics[key]["score"])

                taskname = task.split(os.path.sep)[-1].split(".json")[0]
                eval_results[taskname] = {
                    "task": task,
                    "checkpoint_path": self.config["checkpoint_path"],
                    "model_variant": self.config["model"].get("model_variant", "legacy"),
                    "metrics": metric.metrics,
                }

                predictions_fpath = os.path.join(
                    self.config["save_dir"], f"eval_{taskname}_predictions.jsonl"
                )
                with open(predictions_fpath, "w") as f:
                    for index, generation, answer, filepath, input_text in zip(
                        indices, generations, answers, filepaths, inputs
                    ):
                        f.write(json.dumps({
                            "index": index,
                            "input": input_text,
                            "prediction": generation,
                            "answer": answer,
                            "file_path": filepath,
                        }) + "\n")
                self.logger.info("Wrote predictions to %s", predictions_fpath)
            
            # azure logging
            val_score += metric.metrics["main"]["score"]

        if self.distributed.rank() == 0:
            metrics_fpath = os.path.join(self.config["save_dir"], "eval_metrics.json")
            with open(metrics_fpath, "w") as f:
                json.dump({
                    "checkpoint_path": self.config["checkpoint_path"],
                    "model_variant": self.config["model"].get("model_variant", "legacy"),
                    "val_score": val_score,
                    "tasks": eval_results,
                }, f, indent=2)
            self.logger.info("Wrote evaluation metrics to %s", metrics_fpath)
            
    # pylint: disable=too-many-locals
    def evaluate_experiment(self):
        self.logger.info("Evaluate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        model = self.get_model()
        model = model.to(self.device)

        foldername = f"{os.path.sep}".join(self.config["checkpoint_path"].split(os.path.sep)[:-1])
        max_epochs = max([int(f.split("-epo-")[-1].split(".ckpt")[0]) for f in glob.glob(os.path.join(foldername,"*.ckpt"))])

        tasks = self.config["data"]["datafiles"]
        for e in range(1, max_epochs+1):
            checkpoint_path = self.config["checkpoint_path"].replace("-epo-1",f"-epo-{e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=True)
            model.eval()

            val_score = 0
            for task in tasks:
                self.logger.info("Evaluating task %s", task)
                self.config["data"]["datafiles"] = [task]

                metric = Metric(task, self.config["data"]["sampling_rate"])
                dataset, _, data_loader = self.get_data("datafiles")
                num_batches_per_epoch = len(data_loader)
                # tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)

                generations, answers, filepaths, inputs = [], [], [], []
                with torch.no_grad():
                    for batch_data_dict in tqdm(data_loader):
                        batch_audio1 = batch_data_dict['waveform1']
                        batch_audio2 = batch_data_dict['waveform2']
                        batch_input = batch_data_dict['input']
                        batch_answer = batch_data_dict['answer']
                        batch_answer_text = batch_data_dict['answer_text']
                        batch_input_text = batch_data_dict['input_text']
                        batch_file_paths = batch_data_dict['file_path1']

                        input_dict = {
                            "audio1":batch_audio1,
                            "audio2": batch_audio2,
                            "input":batch_input,
                            "answer":batch_answer,
                        }
                        if "waveform1_lengths" in batch_data_dict:
                            input_dict["audio1_lengths"] = batch_data_dict["waveform1_lengths"]
                        if "waveform2_lengths" in batch_data_dict:
                            input_dict["audio2_lengths"] = batch_data_dict["waveform2_lengths"]
                        if "ced_hidden" in batch_data_dict:
                            input_dict["ced_hidden"] = batch_data_dict["ced_hidden"]
                        if "ced_hidden_segment_lengths" in batch_data_dict:
                            input_dict["ced_hidden_segment_lengths"] = batch_data_dict["ced_hidden_segment_lengths"]
                        if "ced_hidden_lengths" in batch_data_dict:
                            input_dict["ced_hidden_lengths"] = batch_data_dict["ced_hidden_lengths"]
                        if "beats_hidden" in batch_data_dict:
                            input_dict["beats_hidden"] = batch_data_dict["beats_hidden"]
                        if "beats_hidden_segment_lengths" in batch_data_dict:
                            input_dict["beats_hidden_segment_lengths"] = batch_data_dict["beats_hidden_segment_lengths"]
                        if "beats_hidden_lengths" in batch_data_dict:
                            input_dict["beats_hidden_lengths"] = batch_data_dict["beats_hidden_lengths"]
                        input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                    
                        prefix, _, _ = model.generate_prefix_inference(input_dict)
                        generated_text = generate_greedy_batch(model, data_loader.dataset.tokenizer, embed=prefix)
                        generations += generated_text
                        answers += batch_answer_text
                        inputs += batch_input_text
                        filepaths += batch_file_paths

                metric.get_metrics(generations, answers, filepaths)
                if self.distributed.rank() == 0:
                    self.logger.info(f"Epoch {e}, Task %s results", task)
                    for key in metric.metrics.keys():
                        self.logger.info("%s: %f", key, metric.metrics[key]["score"])
                
                # azure logging
                val_score += metric.metrics["main"]["score"]
                taskname = task.split(os.path.sep)[-1].split(".json")[0]
                if self.distributed.rank() == 0:
                    self.log_step_metric(taskname, metric.metrics["main"]["score"])
                    
            if self.distributed.rank() == 0:
                self.log_step_metric(f"val_score", val_score)

    def _get_checkpoint_name(self, epoch: int):
        prefix = self.config.get("myconfig")
        prefix = prefix + '-' if prefix else ''
        return f"{prefix}-epo-{epoch + 1}.ckpt"

    @retry
    def _save_model_state(self, model_fpath, model):
        with open(model_fpath, "wb") as f:
            torch.save(self.distributed.get_distributed_model_state(model), f)
            # make sure data is sent to blobstorage
            f.flush()
            f.close()
