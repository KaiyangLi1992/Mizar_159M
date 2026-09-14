"""Losses and compact student-side probes for heterogeneous Audio-AlignKD.

The teacher and student do not share a tokenizer, hidden dimension, head
count, or audio-token rate.  This module therefore exposes only two common
representations:

* five semantic-group distributions over 126 audio positions plus audio mass;
* a parameter-free 126 x 126 temporal cosine Gram matrix.

No teacher model is loaded here.  All teacher tensors are expected to come
from the versioned offline cache.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F


NUM_SEMANTIC_GROUPS = 5
TARGET_AUDIO_TOKENS = 126


def _require_shape(name: str, tensor: torch.Tensor, suffix: tuple[int, ...]) -> None:
    if tensor.ndim != len(suffix) + 1 or tuple(tensor.shape[1:]) != suffix:
        raise ValueError(
            f"{name} must have shape [batch,{','.join(map(str, suffix))}], "
            f"got {tuple(tensor.shape)}"
        )


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().detach().item()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # HF Llama rotary embeddings are [batch, sequence, head_dim].
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


def _maybe_qk_norm(module: Any, name: str, value: torch.Tensor) -> torch.Tensor:
    norm = getattr(module, name, None)
    return norm(value) if norm is not None else value


def llama_layer0_grouped_audio_attention(
    lm: torch.nn.Module,
    prefix_embeddings: torch.Tensor,
    query_group_mask: torch.Tensor,
    *,
    audio_tokens: int = TARGET_AUDIO_TOKENS,
    separator_tokens: int = 1,
    epsilon: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute only layer-0 prompt-query attention to audio keys.

    ``prefix_embeddings`` must use the causal layout
    ``[audio, separator, fixed-length prompt]``.  The returned conditional
    distribution is ``[B,5,126]`` and audio mass is ``[B,5]``.  The function
    never asks the language model to materialize attentions for all layers.
    """

    if prefix_embeddings.ndim != 3:
        raise ValueError(
            f"prefix_embeddings must be [B,S,D], got {tuple(prefix_embeddings.shape)}"
        )
    _require_shape(
        "query_group_mask",
        query_group_mask,
        (NUM_SEMANTIC_GROUPS, query_group_mask.shape[-1]),
    )
    if prefix_embeddings.shape[0] != query_group_mask.shape[0]:
        raise ValueError("prefix_embeddings/query_group_mask batch mismatch")
    if audio_tokens <= 0 or separator_tokens < 0:
        raise ValueError("audio_tokens must be positive and separator_tokens non-negative")

    prompt_tokens = int(query_group_mask.shape[-1])
    prompt_start = audio_tokens + separator_tokens
    expected_prefix = prompt_start + prompt_tokens
    if prefix_embeddings.shape[1] != expected_prefix:
        raise ValueError(
            "Audio-AlignKD requires prefix_layout=single_audio with exact layout "
            f"[audio={audio_tokens},sep={separator_tokens},prompt={prompt_tokens}]; "
            f"got prefix length {prefix_embeddings.shape[1]}"
        )

    base_model = getattr(lm, "model", None)
    if base_model is None or not hasattr(base_model, "layers") or not base_model.layers:
        raise TypeError("Expected a Llama-family causal LM exposing model.layers")
    layer0 = base_model.layers[0]
    attention = layer0.self_attn

    hidden = layer0.input_layernorm(prefix_embeddings)
    batch, sequence, _ = hidden.shape
    num_heads = int(attention.num_heads)
    num_kv_heads = int(attention.num_key_value_heads)
    head_dim = int(attention.head_dim)
    if num_heads % num_kv_heads:
        raise ValueError(f"GQA head mismatch: q={num_heads}, kv={num_kv_heads}")

    query = attention.q_proj(hidden).view(batch, sequence, num_heads, head_dim).transpose(1, 2)
    key = attention.k_proj(hidden).view(batch, sequence, num_kv_heads, head_dim).transpose(1, 2)
    query = _maybe_qk_norm(attention, "q_norm", query)
    key = _maybe_qk_norm(attention, "k_norm", key)

    positions = torch.arange(sequence, device=hidden.device, dtype=torch.long).unsqueeze(0)
    positions = positions.expand(batch, -1)
    cos, sin = base_model.rotary_emb(hidden, positions)
    query, key = _apply_rotary(query, key, cos, sin)
    key = _repeat_kv(key, num_heads // num_kv_heads)

    # Only prompt queries are needed.  Keys end at the end of the prefix; a
    # small causal mask makes this identical to layer-0 self attention.
    query = query[:, :, prompt_start:expected_prefix, :]
    scaling = float(getattr(attention, "scaling", head_dim**-0.5))
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scaling
    query_positions = torch.arange(prompt_start, expected_prefix, device=hidden.device)
    key_positions = torch.arange(sequence, device=hidden.device)
    causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    scores = scores.masked_fill(~causal.view(1, 1, prompt_tokens, sequence), -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)

    # Average raw audio probability over heads and the tokenizer-specific
    # tokens inside each semantic group.  Normalization within audio is done
    # only after preserving total audio mass.
    audio_probability = probabilities[..., :audio_tokens].mean(dim=1)
    group_mask = query_group_mask.to(device=hidden.device, dtype=audio_probability.dtype)
    group_count = group_mask.sum(dim=-1)
    grouped_raw = torch.einsum("bgq,bqt->bgt", group_mask, audio_probability)
    grouped_raw = grouped_raw / group_count.clamp_min(1.0).unsqueeze(-1)
    audio_mass = grouped_raw.sum(dim=-1)
    conditional = grouped_raw / audio_mass.clamp_min(epsilon).unsqueeze(-1)

    # Replay rows intentionally have no semantic groups.  Keep their tensors
    # finite and zero; the trainer excludes them with is_kd_sample.
    empty = group_count <= 0
    conditional = conditional.masked_fill(empty.unsqueeze(-1), 0.0)
    audio_mass = audio_mass.masked_fill(empty, 0.0)
    _require_finite("student_attention", conditional)
    _require_finite("student_audio_mass", audio_mass)
    return conditional, audio_mass


def normalize_temporal_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"features must be [B,T,D], got {tuple(features.shape)}")
    normalized = F.layer_norm(features.float(), (features.shape[-1],))
    return F.normalize(normalized, p=2, dim=-1, eps=1.0e-8)


def temporal_cosine_gram(features: torch.Tensor) -> torch.Tensor:
    normalized = normalize_temporal_features(features)
    gram = torch.matmul(normalized, normalized.transpose(-1, -2))
    _require_finite("student_feature_gram", gram)
    return gram


def attention_kd_per_sample(
    student_distribution: torch.Tensor,
    student_mass: torch.Tensor,
    teacher_distribution: torch.Tensor,
    teacher_mass: torch.Tensor,
    *,
    group_mask: torch.Tensor | None = None,
    audio_mass_weight: float = 0.25,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    _require_shape(
        "student_distribution",
        student_distribution,
        (NUM_SEMANTIC_GROUPS, TARGET_AUDIO_TOKENS),
    )
    _require_shape(
        "teacher_distribution",
        teacher_distribution,
        (NUM_SEMANTIC_GROUPS, TARGET_AUDIO_TOKENS),
    )
    _require_shape("student_mass", student_mass, (NUM_SEMANTIC_GROUPS,))
    _require_shape("teacher_mass", teacher_mass, (NUM_SEMANTIC_GROUPS,))

    student = student_distribution.float().clamp_min(epsilon)
    teacher = teacher_distribution.float().clamp_min(epsilon)
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    mixture = 0.5 * (student + teacher)
    js = 0.5 * (
        (teacher * (teacher.log() - mixture.log())).sum(dim=-1)
        + (student * (student.log() - mixture.log())).sum(dim=-1)
    )
    mass = F.smooth_l1_loss(
        student_mass.float(), teacher_mass.float(), reduction="none"
    )
    per_group = js + float(audio_mass_weight) * mass
    if group_mask is None:
        per_sample = per_group.mean(dim=-1)
    else:
        _require_shape("group_mask", group_mask, (NUM_SEMANTIC_GROUPS,))
        selected = group_mask.to(device=per_group.device, dtype=per_group.dtype)
        if bool(((selected < 0.0) | (selected > 1.0)).any().detach().item()):
            raise ValueError("group_mask values must be in [0,1]")
        selected_count = selected.sum(dim=-1)
        if bool((selected_count <= 0).any().detach().item()):
            raise ValueError("group_mask must select at least one group per sample")
        per_sample = (per_group * selected).sum(dim=-1) / selected_count
    _require_finite("attention_kd_per_sample", per_sample)
    return per_sample


def gram_all_kd_per_sample(
    student_gram: torch.Tensor,
    teacher_gram: torch.Tensor,
) -> torch.Tensor:
    _require_shape(
        "student_gram", student_gram, (TARGET_AUDIO_TOKENS, TARGET_AUDIO_TOKENS)
    )
    _require_shape(
        "teacher_gram", teacher_gram, (TARGET_AUDIO_TOKENS, TARGET_AUDIO_TOKENS)
    )
    error = F.smooth_l1_loss(student_gram.float(), teacher_gram.float(), reduction="none")
    per_sample = error.mean(dim=(-2, -1))
    _require_finite("gram_all_kd_per_sample", per_sample)
    return per_sample


def gram_soft_kd_per_sample(
    student_gram: torch.Tensor,
    teacher_gram: torch.Tensor,
    teacher_focus: torch.Tensor,
) -> torch.Tensor:
    _require_shape(
        "student_gram", student_gram, (TARGET_AUDIO_TOKENS, TARGET_AUDIO_TOKENS)
    )
    _require_shape(
        "teacher_gram", teacher_gram, (TARGET_AUDIO_TOKENS, TARGET_AUDIO_TOKENS)
    )
    _require_shape("teacher_focus", teacher_focus, (TARGET_AUDIO_TOKENS,))
    focus = teacher_focus.float().clamp_min(0.0)
    focus = focus / focus.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    pair_weight = focus.unsqueeze(-1) * focus.unsqueeze(-2)
    error = F.smooth_l1_loss(student_gram.float(), teacher_gram.float(), reduction="none")
    per_sample = (pair_weight * error).sum(dim=(-2, -1))
    _require_finite("gram_soft_kd_per_sample", per_sample)
    return per_sample


def gram_topk_kd_per_sample(
    student_gram: torch.Tensor,
    teacher_gram: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    expected_k: int = 16,
) -> torch.Tensor:
    _require_shape(
        "student_gram", student_gram, (TARGET_AUDIO_TOKENS, TARGET_AUDIO_TOKENS)
    )
    _require_shape(
        "teacher_gram", teacher_gram, (TARGET_AUDIO_TOKENS, TARGET_AUDIO_TOKENS)
    )
    _require_shape("topk_indices", topk_indices, (expected_k,))
    indices = topk_indices.long()
    if bool(((indices < 0) | (indices >= TARGET_AUDIO_TOKENS)).any().detach().item()):
        raise ValueError("topk_indices contains an out-of-range index")
    if bool((indices.sort(dim=-1).values.diff(dim=-1) == 0).any().detach().item()):
        raise ValueError("topk_indices must be unique per sample")

    batch_index = torch.arange(student_gram.shape[0], device=student_gram.device)
    student_sub = student_gram[
        batch_index[:, None, None], indices[:, :, None], indices[:, None, :]
    ]
    teacher_sub = teacher_gram[
        batch_index[:, None, None], indices[:, :, None], indices[:, None, :]
    ]
    error = F.smooth_l1_loss(student_sub.float(), teacher_sub.float(), reduction="none")
    per_sample = error.mean(dim=(-2, -1))
    _require_finite("gram_topk_kd_per_sample", per_sample)
    return per_sample


def sparse_topk_tail_kd_per_sample(
    student_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    teacher_tail_prob: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    tau: float = 2.0,
    probability_tolerance: float = 1.0e-5,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """KL over a sparse teacher top-k distribution plus one tail bucket.

    This representation is intended for same-family, teacher-forced
    correct-token retention.  The explicit top-k categories retain their
    identities; all remaining vocabulary probability is matched as one
    aggregate category.  Token losses are averaged inside each sample and
    samples without an active token return an exact differentiable zero.
    """

    if student_logits.ndim != 3:
        raise ValueError(
            "student_logits must be [B,L,V], got "
            f"{tuple(student_logits.shape)}"
        )
    batch, length, vocab = student_logits.shape
    if teacher_topk_ids.ndim != 3 or teacher_topk_ids.shape[:2] != (batch, length):
        raise ValueError(
            "teacher_topk_ids must be [B,L,K] with matching B/L, got "
            f"{tuple(teacher_topk_ids.shape)}"
        )
    if teacher_topk_probs.shape != teacher_topk_ids.shape:
        raise ValueError("teacher_topk_probs must match teacher_topk_ids")
    if teacher_tail_prob.shape != (batch, length):
        raise ValueError("teacher_tail_prob must be [B,L]")
    if active_mask.shape != (batch, length):
        raise ValueError("active_mask must be [B,L]")
    if float(tau) <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if teacher_topk_ids.shape[-1] <= 0 or teacher_topk_ids.shape[-1] >= vocab:
        raise ValueError("top-k must be positive and smaller than the vocabulary")

    active = active_mask.to(device=student_logits.device, dtype=torch.bool)
    ids = teacher_topk_ids.to(device=student_logits.device, dtype=torch.long)
    probabilities = teacher_topk_probs.to(
        device=student_logits.device, dtype=torch.float32
    )
    tail = teacher_tail_prob.to(device=student_logits.device, dtype=torch.float32)
    _require_finite("student_logits", student_logits)
    _require_finite("teacher_topk_probs", probabilities)
    _require_finite("teacher_tail_prob", tail)

    expanded_active = active.unsqueeze(-1)
    if bool((((ids < 0) | (ids >= vocab)) & expanded_active).any().detach().item()):
        raise ValueError("teacher_topk_ids contains an active out-of-range id")
    sorted_ids = ids.sort(dim=-1).values
    duplicates = sorted_ids.diff(dim=-1) == 0
    if bool((duplicates & expanded_active).any().detach().item()):
        raise ValueError("teacher_topk_ids must be unique at active tokens")
    if bool(
        (
            ((probabilities < 0.0) | (probabilities > 1.0))
            & expanded_active
        ).any().detach().item()
    ) or bool((((tail < 0.0) | (tail > 1.0)) & active).any().detach().item()):
        raise ValueError("teacher sparse probabilities must lie in [0,1]")
    teacher_mass = probabilities.sum(dim=-1) + tail
    if active.any() and not torch.allclose(
        teacher_mass.masked_select(active),
        torch.ones_like(teacher_mass.masked_select(active)),
        atol=float(probability_tolerance),
        rtol=0.0,
    ):
        maximum_error = (
            teacher_mass.masked_select(active) - 1.0
        ).abs().max().item()
        raise ValueError(
            "teacher top-k plus tail probability does not sum to one; "
            f"max_abs_error={maximum_error:.9g}"
        )

    # Sentinels outside the active mask need not contain valid vocabulary ids.
    safe_ids = ids.masked_fill(~expanded_active, 0)
    student_log_probs = F.log_softmax(
        student_logits.float() / float(tau), dim=-1
    )
    selected_log_probs = student_log_probs.gather(dim=-1, index=safe_ids)
    selected_probs = selected_log_probs.exp()
    student_tail = (1.0 - selected_probs.sum(dim=-1)).clamp_min(float(epsilon))

    explicit_kl = torch.where(
        probabilities > 0.0,
        probabilities
        * (probabilities.clamp_min(float(epsilon)).log() - selected_log_probs),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    tail_kl = torch.where(
        tail > 0.0,
        tail * (tail.clamp_min(float(epsilon)).log() - student_tail.log()),
        torch.zeros_like(tail),
    )
    per_token = (explicit_kl + tail_kl) * float(tau) ** 2
    per_token = per_token.masked_fill(~active, 0.0)
    token_count = active.to(dtype=torch.float32).sum(dim=-1)
    per_sample = per_token.sum(dim=-1) / token_count.clamp_min(1.0)
    per_sample = torch.where(token_count > 0.0, per_sample, per_sample * 0.0)
    _require_finite("sparse_topk_tail_kd_per_sample", per_sample)
    return per_sample


def globally_normalized_masked_loss(
    per_sample: torch.Tensor,
    mask: torch.Tensor,
    distributed: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a DDP-correct global sample mean while retaining local gradients.

    DDP averages gradients across ranks.  Multiplying the local numerator by
    ``world_size / global_count`` therefore yields the gradient of the global
    Align-sample mean even when ranks contain different Align counts.
    """

    if per_sample.ndim != 1 or mask.ndim != 1 or per_sample.shape != mask.shape:
        raise ValueError(
            f"per_sample/mask must both be [B], got {tuple(per_sample.shape)} and {tuple(mask.shape)}"
        )
    selected = mask.to(device=per_sample.device, dtype=torch.bool)
    local_sum = per_sample.masked_select(selected).sum()
    local_count = selected.to(dtype=torch.float32).sum()
    global_count = distributed.all_reduce(local_count.detach().clone(), average=False)
    if float(global_count.detach().item()) <= 0.0:
        return local_sum * 0.0, local_sum.detach() * 0.0, global_count
    scale = float(distributed.world_size()) / global_count
    loss = local_sum * scale
    return loss, local_sum.detach(), global_count


def loss_gradient_l2_norm(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Measure a loss's gradient norm on a small, explicit parameter probe.

    This is intended for short smoke runs.  It deliberately uses
    ``autograd.grad`` instead of touching ``parameter.grad``, so the real
    optimizer backward remains unchanged.
    """

    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not params:
        raise ValueError("loss_gradient_l2_norm requires trainable parameters")
    gradients = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device, dtype=torch.float32)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    norm = squared.sqrt()
    _require_finite("loss_gradient_l2_norm", norm)
    return norm


def parameter_gradient_l2_norm(
    parameters: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    """Return the current accumulated ``parameter.grad`` L2 norm."""

    params = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not params:
        raise ValueError("parameter_gradient_l2_norm requires trainable parameters")
    device = params[0].device
    squared = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in params:
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    norm = squared.sqrt()
    _require_finite("parameter_gradient_l2_norm", norm)
    return norm
