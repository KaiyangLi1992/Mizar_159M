import torch
import torch.nn as nn


def masked_mean(x, mask=None, dim=1):
    if mask is None:
        return x.mean(dim=dim)

    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return (x * mask).sum(dim=dim) / denom


class QuestionConditionedPrefixAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        alpha_init: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.q_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, z0, q_emb=None, q_tokens=None, q_mask=None, audio_mask=None, return_stats=False):
        if q_emb is None:
            if q_tokens is None:
                raise ValueError("QuestionConditionedPrefixAdapter requires q_emb or q_tokens")
            q_emb = masked_mean(q_tokens, q_mask)

        q_bias = self.q_proj(q_emb).unsqueeze(1)
        z = z0 + q_bias

        h = self.ln1(z)
        attn_out, _ = self.self_attn(
            h,
            h,
            h,
            key_padding_mask=audio_mask,
            need_weights=False,
        )
        z = z + attn_out

        h = self.ln2(z)
        delta = self.ffn(h)
        scaled_delta = self.alpha * delta
        z_final = z0 + scaled_delta

        if not return_stats:
            return z_final

        with torch.no_grad():
            z0_norm = torch.linalg.vector_norm(z0.detach())
            scaled_delta_norm = torch.linalg.vector_norm(scaled_delta.detach())
            impact_ratio = scaled_delta_norm / z0_norm.clamp_min(1e-12)
            stats = {
                "alpha": self.alpha.detach(),
                "z0_norm": z0_norm,
                "scaled_delta_norm": scaled_delta_norm,
                "impact_ratio": impact_ratio,
            }

        return z_final, stats


class QuestionFiLMAdapter(nn.Module):
    """Question-conditioned FiLM and token gating on audio prefix tokens."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        alpha_init: float = 0.0,
        gate_bias_init: float = 0.0,
        gate_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del n_heads, kwargs

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.audio_ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.film = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3 * d_model),
        )
        self.token_audio_proj = nn.Linear(d_model, d_model)
        self.token_context_proj = nn.Linear(d_model, d_model)
        self.token_gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.token_gate.bias, float(gate_bias_init))
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z0, q_emb=None, q_tokens=None, q_mask=None, audio_mask=None, return_stats=False):
        del audio_mask
        if q_emb is None:
            if q_tokens is None:
                raise ValueError("QuestionFiLMAdapter requires q_emb or q_tokens")
            q_emb = masked_mean(q_tokens, q_mask)

        audio_h = self.audio_ln(z0)
        evidence = self.q_proj(q_emb)
        gamma, beta, channel_gate_logits = self.film(evidence).chunk(3, dim=-1)
        gamma = torch.tanh(gamma)
        channel_gate = torch.sigmoid(channel_gate_logits)

        film_delta = audio_h * gamma.unsqueeze(1) + beta.unsqueeze(1)
        token_hidden = torch.tanh(
            self.token_audio_proj(audio_h) + self.token_context_proj(evidence).unsqueeze(1)
        )
        token_gate = torch.sigmoid(self.token_gate(token_hidden) / self.gate_temperature)

        h = z0 + token_gate * film_delta
        delta = self.delta_ffn(h + evidence.unsqueeze(1))
        delta = delta * token_gate * channel_gate.unsqueeze(1)
        scaled_delta = self.alpha * delta
        z_final = z0 + scaled_delta

        if not return_stats:
            return z_final

        with torch.no_grad():
            z0_norm = torch.linalg.vector_norm(z0.detach())
            scaled_delta_norm = torch.linalg.vector_norm(scaled_delta.detach())
            impact_ratio = scaled_delta_norm / z0_norm.clamp_min(1e-12)
            stats = {
                "alpha": self.alpha.detach(),
                "z0_norm": z0_norm,
                "scaled_delta_norm": scaled_delta_norm,
                "impact_ratio": impact_ratio,
                "token_gate_mean": token_gate.detach().mean(),
                "token_gate_std": token_gate.detach().std(unbiased=False),
                "channel_gate_mean": channel_gate.detach().mean(),
                "film_delta_norm": torch.linalg.vector_norm(film_delta.detach()),
                "evidence_norm": torch.linalg.vector_norm(evidence.detach()),
            }

        return z_final, stats


class QuestionAudioCrossFiLMAdapter(nn.Module):
    """Question-first audio adapter.

    The prompt tokens query the audio tokens to collect question-relevant audio
    evidence. That evidence then FiLM-modulates and gates the audio prefix tokens
    before the final prefix is passed to the frozen language model.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        alpha_init: float = 0.0,
        gate_bias_init: float = 0.0,
        gate_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gate_temperature = max(float(gate_temperature), 1e-6)

        self.audio_ln = nn.LayerNorm(d_model)
        self.question_ln = nn.LayerNorm(d_model)
        self.q_to_audio_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.evidence_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.film = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3 * d_model),
        )

        self.token_audio_proj = nn.Linear(d_model, d_model)
        self.token_context_proj = nn.Linear(d_model, d_model)
        self.token_gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.token_gate.bias, float(gate_bias_init))

        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z0, q_emb=None, q_tokens=None, q_mask=None, audio_mask=None, return_stats=False):
        if q_tokens is None:
            if q_emb is None:
                raise ValueError("QuestionAudioCrossFiLMAdapter requires q_tokens or q_emb")
            q_tokens = q_emb.unsqueeze(1)
            q_mask = None

        audio_h = self.audio_ln(z0)
        question_h = self.question_ln(q_tokens)

        q_audio, _ = self.q_to_audio_attn(
            question_h,
            audio_h,
            audio_h,
            key_padding_mask=audio_mask,
            need_weights=False,
        )
        evidence = self.evidence_proj(masked_mean(q_audio, q_mask))

        gamma, beta, channel_gate_logits = self.film(evidence).chunk(3, dim=-1)
        gamma = torch.tanh(gamma)
        channel_gate = torch.sigmoid(channel_gate_logits)

        film_delta = audio_h * gamma.unsqueeze(1) + beta.unsqueeze(1)
        token_hidden = torch.tanh(
            self.token_audio_proj(audio_h) + self.token_context_proj(evidence).unsqueeze(1)
        )
        token_gate = torch.sigmoid(self.token_gate(token_hidden) / self.gate_temperature)

        h = z0 + token_gate * film_delta
        delta = self.delta_ffn(h + evidence.unsqueeze(1))
        delta = delta * token_gate * channel_gate.unsqueeze(1)
        scaled_delta = self.alpha * delta
        z_final = z0 + scaled_delta

        if not return_stats:
            return z_final

        with torch.no_grad():
            z0_norm = torch.linalg.vector_norm(z0.detach())
            scaled_delta_norm = torch.linalg.vector_norm(scaled_delta.detach())
            impact_ratio = scaled_delta_norm / z0_norm.clamp_min(1e-12)
            stats = {
                "alpha": self.alpha.detach(),
                "z0_norm": z0_norm,
                "scaled_delta_norm": scaled_delta_norm,
                "impact_ratio": impact_ratio,
                "token_gate_mean": token_gate.detach().mean(),
                "token_gate_std": token_gate.detach().std(unbiased=False),
                "channel_gate_mean": channel_gate.detach().mean(),
                "film_delta_norm": torch.linalg.vector_norm(film_delta.detach()),
                "evidence_norm": torch.linalg.vector_norm(evidence.detach()),
            }

        return z_final, stats


class QFormerLiteLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.query_self_ln = nn.LayerNorm(d_model)
        self.query_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_question_ln = nn.LayerNorm(d_model)
        self.query_question_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_audio_ln = nn.LayerNorm(d_model)
        self.query_audio_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, question_h, audio_h, question_key_padding=None, audio_key_padding=None):
        h = self.query_self_ln(queries)
        queries = queries + self.query_self_attn(h, h, h, need_weights=False)[0]

        h = self.query_question_ln(queries)
        queries = queries + self.query_question_attn(
            h,
            question_h,
            question_h,
            key_padding_mask=question_key_padding,
            need_weights=False,
        )[0]

        h = self.query_audio_ln(queries)
        queries = queries + self.query_audio_attn(
            h,
            audio_h,
            audio_h,
            key_padding_mask=audio_key_padding,
            need_weights=False,
        )[0]

        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class QuestionAudioQFormerLiteAdapter(nn.Module):
    """Small QFormer-style question-conditioned audio token selector."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
        alpha_init: float = 0.0,
        num_queries: int = 8,
        num_layers: int = 1,
        gate_bias_init: float = -1.0,
        gate_temperature: float = 1.0,
        gate_mode: str = "sigmoid",
        select_scale: float = 8.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.gate_mode = str(gate_mode)
        self.select_scale = float(select_scale)
        self.query_tokens = nn.Parameter(torch.empty(int(num_queries), d_model))
        nn.init.normal_(self.query_tokens, std=0.02)

        self.audio_ln = nn.LayerNorm(d_model)
        self.question_ln = nn.LayerNorm(d_model)
        self.question_to_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            QFormerLiteLayer(d_model, n_heads, mlp_ratio, dropout)
            for _ in range(int(num_layers))
        ])

        self.audio_read_ln = nn.LayerNorm(d_model)
        self.audio_to_query_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.token_audio_proj = nn.Linear(d_model, d_model)
        self.token_evidence_proj = nn.Linear(d_model, d_model)
        self.token_gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.token_gate.bias, float(gate_bias_init))

        self.channel_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.delta_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z0, q_emb=None, q_tokens=None, q_mask=None, audio_mask=None, return_stats=False):
        if q_tokens is None:
            if q_emb is None:
                raise ValueError("QuestionAudioQFormerLiteAdapter requires q_tokens or q_emb")
            q_tokens = q_emb.unsqueeze(1)
            q_mask = None
        elif q_emb is None:
            q_emb = masked_mean(q_tokens, q_mask)

        audio_h = self.audio_ln(z0)
        question_h = self.question_ln(q_tokens)
        question_key_padding = None
        if q_mask is not None:
            question_key_padding = q_mask.to(device=q_tokens.device) == 0

        queries = self.query_tokens.unsqueeze(0).expand(z0.shape[0], -1, -1)
        queries = queries + self.question_to_query(q_emb).unsqueeze(1)
        for layer in self.layers:
            queries = layer(
                queries,
                question_h,
                audio_h,
                question_key_padding=question_key_padding,
                audio_key_padding=audio_mask,
            )

        audio_query, _ = self.audio_to_query_attn(
            self.audio_read_ln(z0),
            queries,
            queries,
            need_weights=False,
        )
        token_hidden = torch.tanh(
            self.token_audio_proj(audio_h) + self.token_evidence_proj(audio_query)
        )
        token_logits = self.token_gate(token_hidden) / self.gate_temperature
        if self.gate_mode == "softmax":
            token_gate = torch.softmax(token_logits, dim=1) * self.select_scale
        elif self.gate_mode == "sigmoid":
            token_gate = torch.sigmoid(token_logits)
        else:
            raise ValueError(f"Unknown QFormer-lite gate_mode: {self.gate_mode}")

        query_summary = queries.mean(dim=1)
        channel_gate = torch.sigmoid(self.channel_gate(query_summary))
        delta = self.delta_ffn(z0 + token_gate * audio_query)
        delta = delta * token_gate * channel_gate.unsqueeze(1)
        scaled_delta = self.alpha * delta
        z_final = z0 + scaled_delta

        if not return_stats:
            return z_final

        with torch.no_grad():
            z0_norm = torch.linalg.vector_norm(z0.detach())
            scaled_delta_norm = torch.linalg.vector_norm(scaled_delta.detach())
            impact_ratio = scaled_delta_norm / z0_norm.clamp_min(1e-12)
            stats = {
                "alpha": self.alpha.detach(),
                "z0_norm": z0_norm,
                "scaled_delta_norm": scaled_delta_norm,
                "impact_ratio": impact_ratio,
                "token_gate_mean": token_gate.detach().mean(),
                "token_gate_std": token_gate.detach().std(unbiased=False),
                "token_gate_max": token_gate.detach().max(),
                "channel_gate_mean": channel_gate.detach().mean(),
                "audio_query_norm": torch.linalg.vector_norm(audio_query.detach()),
                "query_norm": torch.linalg.vector_norm(queries.detach()),
            }

        return z_final, stats


class ListenTwiceQFormerLiteAdapter(QuestionAudioQFormerLiteAdapter):
    """QFormer-lite adapter conditioned on a first-pass SLM listen state.

    The listen projections are zero-initialized so loading a one-pass
    QFormer-lite checkpoint starts with identical adapter behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.query_tokens.shape[-1]
        self.listen_to_query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.listen_to_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.listen_memory_ln = nn.LayerNorm(d_model)
        self.listen_query_ln = nn.LayerNorm(d_model)
        self.listen_memory_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=kwargs.get("n_heads", 4) if kwargs else 4,
            dropout=kwargs.get("dropout", 0.0) if kwargs else 0.0,
            batch_first=True,
        )
        self.listen_token_to_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.listen_to_query[-1].weight)
        nn.init.zeros_(self.listen_to_query[-1].bias)
        nn.init.zeros_(self.listen_to_gate[-1].weight)
        nn.init.zeros_(self.listen_to_gate[-1].bias)
        nn.init.zeros_(self.listen_memory_attn.out_proj.weight)
        nn.init.zeros_(self.listen_memory_attn.out_proj.bias)
        nn.init.zeros_(self.listen_token_to_gate[-1].weight)
        nn.init.zeros_(self.listen_token_to_gate[-1].bias)

    def forward(
        self,
        z0,
        q_emb=None,
        q_tokens=None,
        q_mask=None,
        audio_mask=None,
        listen_state=None,
        listen_tokens=None,
        return_stats=False,
    ):
        if q_tokens is None:
            if q_emb is None:
                raise ValueError("ListenTwiceQFormerLiteAdapter requires q_tokens or q_emb")
            q_tokens = q_emb.unsqueeze(1)
            q_mask = None
        elif q_emb is None:
            q_emb = masked_mean(q_tokens, q_mask)

        audio_h = self.audio_ln(z0)
        question_h = self.question_ln(q_tokens)
        question_key_padding = None
        if q_mask is not None:
            question_key_padding = q_mask.to(device=q_tokens.device) == 0

        listen_query_bias = 0.0
        listen_gate_bias = 0.0
        listen_token_gate_bias = 0.0
        if listen_state is not None:
            listen_query_bias = self.listen_to_query(listen_state).unsqueeze(1)
            listen_gate_bias = self.listen_to_gate(listen_state).unsqueeze(1)

        queries = self.query_tokens.unsqueeze(0).expand(z0.shape[0], -1, -1)
        queries = queries + self.question_to_query(q_emb).unsqueeze(1) + listen_query_bias
        if listen_tokens is not None:
            if listen_tokens.shape[1] != z0.shape[1]:
                listen_tokens = listen_tokens.mean(dim=1, keepdim=True).expand(-1, z0.shape[1], -1)
            listen_h = self.listen_memory_ln(listen_tokens)
            listen_query_delta, _ = self.listen_memory_attn(
                self.listen_query_ln(queries),
                listen_h,
                listen_h,
                need_weights=False,
            )
            queries = queries + listen_query_delta
            listen_token_gate_bias = self.listen_token_to_gate(listen_h)
        for layer in self.layers:
            queries = layer(
                queries,
                question_h,
                audio_h,
                question_key_padding=question_key_padding,
                audio_key_padding=audio_mask,
            )

        audio_query, _ = self.audio_to_query_attn(
            self.audio_read_ln(z0),
            queries,
            queries,
            need_weights=False,
        )
        token_hidden = torch.tanh(
            self.token_audio_proj(audio_h)
            + self.token_evidence_proj(audio_query)
            + listen_gate_bias
            + listen_token_gate_bias
        )
        token_logits = self.token_gate(token_hidden) / self.gate_temperature
        if self.gate_mode == "softmax":
            token_gate = torch.softmax(token_logits, dim=1) * self.select_scale
        elif self.gate_mode == "sigmoid":
            token_gate = torch.sigmoid(token_logits)
        else:
            raise ValueError(f"Unknown QFormer-lite gate_mode: {self.gate_mode}")

        query_summary = queries.mean(dim=1)
        channel_gate = torch.sigmoid(self.channel_gate(query_summary))
        delta = self.delta_ffn(z0 + token_gate * audio_query)
        delta = delta * token_gate * channel_gate.unsqueeze(1)
        scaled_delta = self.alpha * delta
        z_final = z0 + scaled_delta

        if not return_stats:
            return z_final

        with torch.no_grad():
            z0_norm = torch.linalg.vector_norm(z0.detach())
            scaled_delta_norm = torch.linalg.vector_norm(scaled_delta.detach())
            impact_ratio = scaled_delta_norm / z0_norm.clamp_min(1e-12)
            stats = {
                "alpha": self.alpha.detach(),
                "z0_norm": z0_norm,
                "scaled_delta_norm": scaled_delta_norm,
                "impact_ratio": impact_ratio,
                "token_gate_mean": token_gate.detach().mean(),
                "token_gate_std": token_gate.detach().std(unbiased=False),
                "token_gate_max": token_gate.detach().max(),
                "channel_gate_mean": channel_gate.detach().mean(),
                "audio_query_norm": torch.linalg.vector_norm(audio_query.detach()),
                "query_norm": torch.linalg.vector_norm(queries.detach()),
            }
            if listen_state is not None:
                listen_detached = listen_state.detach()
                stats["listen_state_norm"] = torch.linalg.vector_norm(listen_detached)
                stats["listen_state_std"] = listen_detached.std(unbiased=False)
            if listen_tokens is not None:
                listen_tokens_detached = listen_tokens.detach()
                stats["listen_tokens_norm"] = torch.linalg.vector_norm(listen_tokens_detached)
                stats["listen_tokens_std"] = listen_tokens_detached.std(unbiased=False)

        return z_final, stats
