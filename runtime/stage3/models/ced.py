from contextlib import nullcontext
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModelForAudioClassification


class CEDFrequencyMerger(nn.Module):
    def __init__(self, hidden_dim: int, freq_bins: int, d_out: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * freq_bins),
            nn.Linear(hidden_dim * freq_bins, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, freq_bins, hidden_dim = x.shape
        return self.net(x.reshape(bsz, time_steps, freq_bins * hidden_dim))


class CEDDirectMapper(nn.Module):
    def __init__(self, hidden_dim: int, d_out: int, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )
        self.audio_type = nn.Parameter(torch.empty(1, 1, d_out))
        nn.init.normal_(self.audio_type, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden) + self.audio_type.to(device=hidden.device, dtype=hidden.dtype)


class CEDTemporalMapper(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        freq_bins: int,
        d_out: int,
        output_tokens: int = 258,
        resample: bool = True,
        max_temporal_tokens: int = 512,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.resample = bool(resample)
        self.freq_merger = CEDFrequencyMerger(hidden_dim, freq_bins, d_out, dropout=dropout)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out),
        )
        self.pos_in = nn.Parameter(torch.empty(1, max_temporal_tokens, d_out))
        self.pos_out = nn.Parameter(torch.empty(1, max(output_tokens, max_temporal_tokens), d_out))
        self.audio_type = nn.Parameter(torch.empty(1, 1, d_out))
        self.refine_ln = nn.LayerNorm(d_out)
        self.refine_mlp = nn.Sequential(
            nn.Linear(d_out, d_out * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out * 2, d_out),
        )
        nn.init.normal_(self.pos_in, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_out, mean=0.0, std=0.02)
        nn.init.normal_(self.audio_type, mean=0.0, std=0.02)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        bsz, time_steps, _freq_bins, _hidden_dim = grid.shape
        h = self.freq_merger(grid)
        global_h = grid.mean(dim=(1, 2))
        h = h + self.global_proj(global_h).unsqueeze(1)
        if time_steps > self.pos_in.shape[1]:
            raise ValueError(f"CED temporal length {time_steps} exceeds max positional length {self.pos_in.shape[1]}")
        h = h + self.pos_in[:, :time_steps, :].to(device=h.device, dtype=h.dtype)

        if self.resample and time_steps != self.output_tokens:
            h = F.interpolate(
                h.transpose(1, 2),
                size=self.output_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()

        out_tokens = h.shape[1]
        if out_tokens > self.pos_out.shape[1]:
            raise ValueError(f"CED output length {out_tokens} exceeds max positional length {self.pos_out.shape[1]}")
        h = h + self.pos_out[:, :out_tokens, :].to(device=h.device, dtype=h.dtype)
        h = h + self.audio_type.to(device=h.device, dtype=h.dtype)
        h = h + self.refine_mlp(self.refine_ln(h))
        return h


class CEDCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.05):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_ln = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        kv = self.kv_ln(keys)
        attn, _ = self.cross_attn(
            self.q_ln(queries),
            kv,
            kv,
            need_weights=False,
        )
        queries = queries + attn
        queries = queries + self.mlp(self.mlp_ln(queries))
        return queries


class CEDCrossAttentionResampler(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        d_out: int,
        output_tokens: int = 258,
        num_layers: int = 2,
        num_heads: int = 8,
        max_input_tokens: int = 1024,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
            nn.LayerNorm(d_out),
        )
        self.input_pos = nn.Parameter(torch.empty(1, max_input_tokens, d_out))
        self.queries = nn.Parameter(torch.empty(1, self.output_tokens, d_out))
        self.audio_type = nn.Parameter(torch.empty(1, 1, d_out))
        self.blocks = nn.ModuleList(
            CEDCrossAttentionBlock(d_out, num_heads=num_heads, dropout=dropout)
            for _ in range(int(num_layers))
        )
        self.out_ln = nn.LayerNorm(d_out)
        nn.init.normal_(self.input_pos, mean=0.0, std=0.02)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)
        nn.init.normal_(self.audio_type, mean=0.0, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > self.input_pos.shape[1]:
            raise ValueError(
                f"CED hidden length {hidden.shape[1]} exceeds max positional length {self.input_pos.shape[1]}"
            )
        keys = self.input_proj(hidden)
        keys = keys + self.input_pos[:, : hidden.shape[1], :].to(device=hidden.device, dtype=hidden.dtype)
        queries = self.queries.to(device=hidden.device, dtype=hidden.dtype).expand(hidden.shape[0], -1, -1)
        queries = queries + self.audio_type.to(device=hidden.device, dtype=hidden.dtype)
        queries = queries + self.global_proj(hidden.mean(dim=1)).unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, keys)
        return self.out_ln(queries)


class CEDSmallWrapper(nn.Module):
    """CED-Small encoder plus audio-prefix mapper for Experiment C.

    The wrapper accepts a 20s waveform from the existing Mellow dataloader,
    extracts CED hidden patch tokens, and returns SLM-dim audio prefix tokens.
    """

    single_audio_mode = True

    def __init__(self, encoder_config: dict = None, d_out: int = 576):
        super().__init__()
        encoder_config = dict(encoder_config or {})
        model_name = encoder_config.get("ced_model_name") or encoder_config.get("pretrained_audioencoder_path") or "mispeech/ced-small"
        if model_name == "":
            model_name = "mispeech/ced-small"
        cache_dir = encoder_config.get("hf_cache_dir") or os.environ.get("HF_HOME")
        model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        self.ced = model.encoder
        self.config = self.ced.config
        self.hidden_dim = int(self.config.embed_dim)
        self.freq_bins = int(self.config.n_mels // self.config.patch_stride)
        self.time_bins_per_split = int(self.config.target_length // self.config.patch_stride)
        self.input_sample_rate = int(encoder_config.get("input_sampling_rate", 32000))
        self.ced_sample_rate = int(encoder_config.get("ced_sampling_rate", 16000))

        self.resample = None
        if self.input_sample_rate != self.ced_sample_rate:
            self.resample = torchaudio.transforms.Resample(self.input_sample_rate, self.ced_sample_rate)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.ced_sample_rate,
            win_length=int(self.config.win_size),
            center=bool(self.config.center),
            n_fft=int(self.config.n_fft),
            f_min=float(self.config.f_min),
            f_max=float(self.config.f_max),
            hop_length=int(self.config.hop_size),
            n_mels=int(self.config.n_mels),
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=120)

        mapper_type = str(encoder_config.get("ced_mapper", "freq_merge_resample_258")).lower()
        dropout = float(encoder_config.get("ced_mapper_dropout", 0.05))
        output_tokens = int(encoder_config.get("ced_output_tokens", 258))
        if mapper_type in {"direct_504", "direct", "direct_variable", "scheme_e1"}:
            self.mapper = CEDDirectMapper(self.hidden_dim, d_out, dropout=dropout)
            self.output_tokens = None
        elif mapper_type in {"freq_merge_126", "frequency_merge_126", "freq_merge_variable", "frequency_merge_variable", "scheme_e2"}:
            self.mapper = CEDTemporalMapper(
                self.hidden_dim,
                self.freq_bins,
                d_out,
                output_tokens=output_tokens,
                resample=False,
                dropout=dropout,
            )
            self.output_tokens = None
        elif mapper_type in {"freq_merge_resample_258", "frequency_merge_resample_258", "scheme_c"}:
            self.mapper = CEDTemporalMapper(
                self.hidden_dim,
                self.freq_bins,
                d_out,
                output_tokens=output_tokens,
                resample=True,
                dropout=dropout,
            )
            self.output_tokens = output_tokens
        elif mapper_type in {"cross_attn_resample_258", "cross_attention_resample_258", "scheme_d"}:
            self.mapper = CEDCrossAttentionResampler(
                self.hidden_dim,
                d_out,
                output_tokens=output_tokens,
                num_layers=int(encoder_config.get("ced_resampler_layers", 2)),
                num_heads=int(encoder_config.get("ced_resampler_heads", 8)),
                dropout=dropout,
            )
            self.output_tokens = output_tokens
        else:
            raise ValueError(f"Unknown CED mapper type: {mapper_type}")
        self.mapper_type = mapper_type

        self.freeze_encoder = bool(encoder_config.get("freeze_audio_encoder_weights", True))
        self.finetune_last_n_layers = int(encoder_config.get("ced_finetune_last_n_layers", 0) or 0)
        self.hidden_cache_point = str(
            encoder_config.get("ced_hidden_cache_point", "final_hidden")
        ).lower()
        self.variable_length = bool(encoder_config.get("ced_variable_length", False))
        self.encoder_forward_requires_grad = not self.freeze_encoder
        if self.freeze_encoder:
            for param in self.ced.parameters():
                param.requires_grad = False
            if self.finetune_last_n_layers > 0:
                blocks = getattr(self.ced, "blocks", None)
                if blocks is None or len(blocks) < self.finetune_last_n_layers:
                    raise ValueError(
                        "ced_finetune_last_n_layers requires CED encoder.blocks "
                        f"with at least {self.finetune_last_n_layers} layers"
                    )
                for block in blocks[-self.finetune_last_n_layers:]:
                    for param in block.parameters():
                        param.requires_grad = True
                final_norm = getattr(self.ced, "norm", None)
                if final_norm is not None:
                    for param in final_norm.parameters():
                        param.requires_grad = True
                self.encoder_forward_requires_grad = True
        if self.hidden_cache_point in {"last5_out", "pre_last4", "block7_out_before_last4"}:
            if self.finetune_last_n_layers <= 0:
                raise ValueError(
                    "last5_out CED hidden cache requires ced_finetune_last_n_layers > 0"
                )

    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.dim() != 2:
            raise ValueError(f"CEDSmallWrapper expects [B, samples], got {tuple(waveform.shape)}")
        waveform = waveform.float()
        if self.resample is not None:
            waveform = self.resample(waveform)
        mel = self.mel_spectrogram(waveform)
        return self.amplitude_to_db(mel)

    def _prepare_ced_input(self, input_values: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(input_values, 1)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.ced.init_bn(x)
        return torch.permute(x, (0, 2, 1, 3))

    def _forward_hidden_segments(self, input_values: torch.Tensor) -> list[torch.Tensor]:
        x = self._prepare_ced_input(input_values)

        if x.shape[-1] <= self.ced.maximal_allowed_length:
            return [self.ced.forward_features(x)]

        splits = list(x.split(self.ced.maximal_allowed_length, -1))
        if self.variable_length:
            return [
                self.ced.forward_features(split)
                for split in splits
                if split.shape[-1] > 0
            ]

        if splits[-1].shape[-1] < self.ced.maximal_allowed_length:
            if self.ced.config.pad_last:
                pad = torch.zeros(*x.shape[:-1], self.ced.maximal_allowed_length, device=x.device, dtype=x.dtype)
                pad[..., : splits[-1].shape[-1]] = splits[-1]
                splits[-1] = pad
            else:
                splits = splits[:-1]
        if not splits:
            raise ValueError("CED long-input split produced no chunks.")
        stacked = torch.stack(splits, dim=1)
        bsz, n_splits = stacked.shape[:2]
        flat = stacked.reshape(bsz * n_splits, *stacked.shape[2:])
        hidden = self.ced.forward_features(flat)
        return [hidden.reshape(bsz, n_splits * hidden.shape[1], hidden.shape[2])]

    def _forward_hidden(self, input_values: torch.Tensor) -> torch.Tensor:
        return torch.cat(self._forward_hidden_segments(input_values), dim=1)

    def _forward_features_until_last_n(self, x: torch.Tensor, last_n: int) -> torch.Tensor:
        x = self.ced.patch_embed(x)
        _, _, _, time_steps = x.shape
        x = x + self.ced.time_pos_embed[:, :, :, :time_steps]
        x = x + self.ced.freq_pos_embed[:, :, :, :]
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))
        if self.ced.config.pooling == "token":
            cls_token = self.ced.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.ced.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.ced.pos_drop(x)
        blocks = list(self.ced.blocks)
        for block in blocks[:-last_n]:
            x = block(x)
        return x

    def _forward_pre_last_n_segments(
        self,
        input_values: torch.Tensor,
        last_n: int,
    ) -> list[torch.Tensor]:
        x = self._prepare_ced_input(input_values)

        if x.shape[-1] <= self.ced.maximal_allowed_length:
            return [self._forward_features_until_last_n(x, last_n)]

        splits = list(x.split(self.ced.maximal_allowed_length, -1))
        if self.variable_length:
            return [
                self._forward_features_until_last_n(split, last_n)
                for split in splits
                if split.shape[-1] > 0
            ]

        if splits[-1].shape[-1] < self.ced.maximal_allowed_length:
            if self.ced.config.pad_last:
                pad = torch.zeros(*x.shape[:-1], self.ced.maximal_allowed_length, device=x.device, dtype=x.dtype)
                pad[..., : splits[-1].shape[-1]] = splits[-1]
                splits[-1] = pad
            else:
                splits = splits[:-1]
        if not splits:
            raise ValueError("CED long-input split produced no chunks.")
        stacked = torch.stack(splits, dim=1)
        bsz, n_splits = stacked.shape[:2]
        flat = stacked.reshape(bsz * n_splits, *stacked.shape[2:])
        hidden = self._forward_features_until_last_n(flat, last_n)
        return [hidden.reshape(bsz, n_splits * hidden.shape[1], hidden.shape[2])]

    def _continue_from_pre_last_n(self, hidden: torch.Tensor) -> torch.Tensor:
        param = next(self.ced.parameters(), None)
        if param is not None and hidden.is_floating_point():
            hidden = hidden.to(device=param.device, dtype=param.dtype)
        x = hidden
        for block in list(self.ced.blocks)[-self.finetune_last_n_layers:]:
            x = block(x)
        return self.ced.norm(x)

    def _continue_cached_last5_out(
        self,
        hidden: torch.Tensor,
        hidden_segment_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_segment_lengths is None:
            return self._continue_from_pre_last_n(hidden)

        outputs = []
        max_tokens = 0
        for batch_idx in range(hidden.shape[0]):
            start = 0
            segments = []
            lengths = hidden_segment_lengths[batch_idx].detach().cpu().tolist()
            for length in lengths:
                length = int(length)
                if length <= 0:
                    continue
                segment = hidden[batch_idx : batch_idx + 1, start : start + length, :]
                segments.append(self._continue_from_pre_last_n(segment).squeeze(0))
                start += length
            if not segments:
                raise ValueError("last5_out cache entry has no valid CED segments")
            merged = torch.cat(segments, dim=0)
            outputs.append(merged)
            max_tokens = max(max_tokens, int(merged.shape[0]))

        padded = []
        for merged in outputs:
            if merged.shape[0] < max_tokens:
                pad = merged.new_zeros(max_tokens - merged.shape[0], merged.shape[-1])
                merged = torch.cat((merged, pad), dim=0)
            padded.append(merged)
        return torch.stack(padded, dim=0)

    def _hidden_segments_to_temporal_grid(self, hidden_segments: list[torch.Tensor]) -> torch.Tensor:
        grids = []
        for hidden in hidden_segments:
            usable = (hidden.shape[1] // self.freq_bins) * self.freq_bins
            hidden = hidden[:, :usable, :]
            time_steps = usable // self.freq_bins
            grid = hidden.reshape(
                hidden.shape[0],
                self.freq_bins,
                time_steps,
                hidden.shape[-1],
            )
            grids.append(grid.permute(0, 2, 1, 3))
        if not grids:
            raise ValueError("CED variable-length hidden produced no temporal grid.")
        return torch.cat(grids, dim=1)

    def _hidden_to_temporal_grid(self, hidden: torch.Tensor) -> torch.Tensor:
        tokens_per_split = self.freq_bins * self.time_bins_per_split
        n_splits = max(1, hidden.shape[1] // tokens_per_split)
        usable = n_splits * tokens_per_split
        hidden = hidden[:, :usable, :]
        grid = hidden.reshape(
            hidden.shape[0],
            n_splits,
            self.freq_bins,
            self.time_bins_per_split,
            hidden.shape[-1],
        )
        return grid.permute(0, 1, 3, 2, 4).reshape(
            hidden.shape[0],
            n_splits * self.time_bins_per_split,
            self.freq_bins,
            hidden.shape[-1],
        )

    def _embedding_from_hidden(
        self,
        hidden: torch.Tensor,
        hidden_segment_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mapper_param = next(self.mapper.parameters(), None)
        if mapper_param is not None and hidden.is_floating_point():
            hidden = hidden.to(device=mapper_param.device, dtype=mapper_param.dtype)

        if isinstance(self.mapper, (CEDDirectMapper, CEDCrossAttentionResampler)):
            return self.mapper(hidden)

        if hidden_segment_lengths is not None and self.variable_length:
            segments = []
            start = 0
            lengths = hidden_segment_lengths[0].detach().cpu().tolist()
            for length in lengths:
                length = int(length)
                if length <= 0:
                    continue
                segments.append(hidden[:, start : start + length, :])
                start += length
            return self.mapper(self._hidden_segments_to_temporal_grid(segments))

        if self.variable_length:
            return self.mapper(self._hidden_segments_to_temporal_grid([hidden]))
        return self.mapper(self._hidden_to_temporal_grid(hidden))

    def forward(self, x: torch.Tensor | dict) -> dict:
        cached_segment_lengths = None
        if isinstance(x, dict):
            hidden = x["ced_hidden"]
            cached_segment_lengths = x.get("ced_hidden_segment_lengths")
            if self.hidden_cache_point in {"last5_out", "pre_last4", "block7_out_before_last4"}:
                hidden = self._continue_cached_last5_out(hidden, cached_segment_lengths)
            embedding = self._embedding_from_hidden(hidden, cached_segment_lengths)
            clipwise_output = embedding.new_zeros(embedding.shape[0], 527)
            return {
                "embedding": embedding,
                "clipwise_output": clipwise_output,
                "ced_hidden": hidden,
            }

        if self.variable_length and x.shape[0] != 1:
            raise ValueError(
                "CED variable-length mode requires per-device batch_size=1 to avoid "
                "batch padding being interpreted as real audio."
            )
        mel = self._waveform_to_mel(x)
        context = nullcontext() if self.encoder_forward_requires_grad else torch.no_grad()
        with context:
            hidden_segments = self._forward_hidden_segments(mel)
        hidden = torch.cat(hidden_segments, dim=1)
        if self.variable_length:
            segment_lengths = torch.tensor(
                [[seg.shape[1] for seg in hidden_segments]],
                device=hidden.device,
                dtype=torch.long,
            )
            embedding = self._embedding_from_hidden(hidden, segment_lengths)
        else:
            embedding = self._embedding_from_hidden(hidden)
        clipwise_output = embedding.new_zeros(embedding.shape[0], 527)
        return {
            "embedding": embedding,
            "clipwise_output": clipwise_output,
            "ced_hidden": hidden,
        }
