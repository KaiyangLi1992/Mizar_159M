from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models.beats_official.BEATs import BEATs, BEATsConfig


class BEATsTemporalMapper(nn.Module):
    """BEATs token mapper aligned with the CED-B temporal mapper family."""

    def __init__(
        self,
        hidden_dim: int,
        d_out: int,
        output_tokens: int = 126,
        max_temporal_tokens: int = 2048,
        dropout: float = 0.05,
        resample_before_projection: bool = True,
        resample_mode: str = "linear",
    ):
        super().__init__()
        self.output_tokens = int(output_tokens)
        self.resample_before_projection = bool(resample_before_projection)
        self.resample_mode = str(resample_mode)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )
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

    def _length_mean(self, hidden: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        if lengths is None:
            return hidden.mean(dim=1)
        lengths = lengths.to(device=hidden.device, dtype=torch.long).clamp(min=1, max=hidden.shape[1])
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        denom = lengths.to(dtype=hidden.dtype).unsqueeze(1)
        return (hidden * mask.unsqueeze(-1).to(dtype=hidden.dtype)).sum(dim=1) / denom

    def _resample_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] == self.output_tokens:
            return hidden
        if self.resample_mode in {"avg", "avg_pool", "adaptive_avg_pool"}:
            return F.adaptive_avg_pool1d(
                hidden.transpose(1, 2),
                self.output_tokens,
            ).transpose(1, 2).contiguous()
        if self.resample_mode not in {"linear", "interpolate"}:
            raise ValueError(f"Unsupported BEATs temporal resample mode: {self.resample_mode}")
        return F.interpolate(
            hidden.transpose(1, 2),
            size=self.output_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).contiguous()

    def forward(self, hidden: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.resample_before_projection:
            hidden = self._resample_hidden(hidden)
            lengths = None

        time_steps = hidden.shape[1]
        if time_steps > self.pos_in.shape[1]:
            raise ValueError(f"BEATs temporal length {time_steps} exceeds max positional length {self.pos_in.shape[1]}")

        h = self.token_proj(hidden)
        h = h + self.global_proj(self._length_mean(hidden, lengths)).unsqueeze(1)
        h = h + self.pos_in[:, :time_steps, :].to(device=h.device, dtype=h.dtype)

        if not self.resample_before_projection and time_steps != self.output_tokens:
            if self.resample_mode in {"avg", "avg_pool", "adaptive_avg_pool"}:
                h = F.adaptive_avg_pool1d(h.transpose(1, 2), self.output_tokens).transpose(1, 2).contiguous()
            elif self.resample_mode in {"linear", "interpolate"}:
                h = F.interpolate(
                    h.transpose(1, 2),
                    size=self.output_tokens,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2).contiguous()
            else:
                raise ValueError(f"Unsupported BEATs temporal resample mode: {self.resample_mode}")

        out_tokens = h.shape[1]
        if out_tokens > self.pos_out.shape[1]:
            raise ValueError(f"BEATs output length {out_tokens} exceeds max positional length {self.pos_out.shape[1]}")
        h = h + self.pos_out[:, :out_tokens, :].to(device=h.device, dtype=h.dtype)
        h = h + self.audio_type.to(device=h.device, dtype=h.dtype)
        h = h + self.refine_mlp(self.refine_ln(h))
        return h


class BEATsBaseWrapper(nn.Module):
    """Frozen BEATs base plus trainable audio-prefix mapper.

    The BEATs encoder is loaded lazily. Cached-hidden training only constructs
    the mapper; raw-audio eval loads the checkpoint when first needed.
    """

    single_audio_mode = True

    def __init__(self, encoder_config: dict = None, d_out: int = 576):
        super().__init__()
        encoder_config = dict(encoder_config or {})
        self.encoder_config = encoder_config
        self.hidden_dim = int(encoder_config.get("beats_hidden_dim", 768))
        self.input_sample_rate = int(encoder_config.get("input_sampling_rate", 32000))
        self.beats_sample_rate = int(encoder_config.get("beats_sampling_rate", 16000))
        self.output_tokens = int(encoder_config.get("beats_output_tokens", 126))
        dropout = float(encoder_config.get("beats_mapper_dropout", 0.05))
        max_temporal_tokens = int(encoder_config.get("beats_max_temporal_tokens", 2048))
        self.mapper = BEATsTemporalMapper(
            self.hidden_dim,
            d_out,
            output_tokens=self.output_tokens,
            max_temporal_tokens=max_temporal_tokens,
            dropout=dropout,
            resample_before_projection=bool(encoder_config.get("beats_resample_before_projection", True)),
            resample_mode=str(
                encoder_config.get(
                    "beats_temporal_resample_mode",
                    encoder_config.get("beats_temporal_compress", "linear"),
                )
            ),
        )
        self.freeze_encoder = bool(encoder_config.get("freeze_audio_encoder_weights", True))
        self.encoder_forward_requires_grad = not self.freeze_encoder
        self._beats: BEATs | None = None
        self._beats_device: torch.device | None = None
        self.resample = None
        if self.input_sample_rate != self.beats_sample_rate:
            self.resample = torchaudio.transforms.Resample(self.input_sample_rate, self.beats_sample_rate)

    def _resolve_checkpoint_path(self) -> str:
        explicit = (
            self.encoder_config.get("beats_checkpoint_path")
            or self.encoder_config.get("pretrained_audioencoder_path")
            or ""
        )
        if explicit and Path(str(explicit)).exists():
            return str(explicit)

        repo_id = str(self.encoder_config.get("beats_hf_repo_id", "Bencr/beats-checkpoints"))
        filename = str(self.encoder_config.get("beats_hf_filename", "BEATs_iter3_plus_AS2M.pt"))
        cache_dir = self.encoder_config.get("hf_cache_dir") or os.environ.get("HF_HOME")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "BEATs checkpoint is not local and huggingface_hub is unavailable. "
                "Set model.encoder.beats_checkpoint_path to a local .pt file."
            ) from exc
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

    def _ensure_beats(self, device: torch.device) -> BEATs:
        if self._beats is not None and self._beats_device == device:
            return self._beats

        checkpoint = torch.load(self._resolve_checkpoint_path(), map_location="cpu")
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if self.freeze_encoder:
            for param in model.parameters():
                param.requires_grad = False
        model.to(device)

        if int(cfg.encoder_embed_dim) != self.hidden_dim:
            raise ValueError(
                f"BEATs checkpoint hidden dim {cfg.encoder_embed_dim} does not match mapper hidden dim {self.hidden_dim}"
            )
        self._beats = model
        self._beats_device = device
        return model

    def _waveform_to_beats_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 3 and waveform.shape[1] == 1:
            waveform = waveform[:, 0, :]
        if waveform.dim() != 2:
            raise ValueError(f"BEATsBaseWrapper expects [B, samples], got {tuple(waveform.shape)}")
        waveform = waveform.float()
        if self.resample is not None:
            self.resample = self.resample.to(waveform.device)
            waveform = self.resample(waveform)
        return waveform

    def _extract_hidden_from_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        beats = self._ensure_beats(device)
        audio_16k = self._waveform_to_beats_audio(waveform).float()

        # Kaldi fbank is kept on CPU for compatibility with torchaudio builds;
        # transformer layers still run on the waveform/model device.
        with torch.cuda.amp.autocast(enabled=False):
            fbank = beats.preprocess(audio_16k.detach().cpu().float()).to(device=device)
        fbank = fbank.unsqueeze(1)
        features = beats.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = beats.layer_norm(features)
        if beats.post_extract_proj is not None:
            features = beats.post_extract_proj(features)
        x = beats.dropout_input(features)
        x, _layer_results = beats.encoder(x, padding_mask=None)
        return x

    def _embedding_from_hidden(
        self,
        hidden: torch.Tensor,
        hidden_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mapper_param = next(self.mapper.parameters(), None)
        if mapper_param is not None and hidden.is_floating_point():
            hidden = hidden.to(device=mapper_param.device, dtype=mapper_param.dtype)
        return self.mapper(hidden, hidden_lengths)

    def forward(self, x: torch.Tensor | dict) -> dict:
        hidden_lengths = None
        if isinstance(x, dict):
            hidden = x["beats_hidden"]
            hidden_lengths = x.get("beats_hidden_lengths")
            embedding = self._embedding_from_hidden(hidden, hidden_lengths)
            clipwise_output = embedding.new_zeros(embedding.shape[0], 527)
            return {
                "embedding": embedding,
                "clipwise_output": clipwise_output,
                "beats_hidden": hidden,
            }

        context = nullcontext() if self.encoder_forward_requires_grad else torch.no_grad()
        with context:
            hidden = self._extract_hidden_from_audio(x)
        embedding = self._embedding_from_hidden(hidden, hidden_lengths)
        clipwise_output = embedding.new_zeros(embedding.shape[0], 527)
        return {
            "embedding": embedding,
            "clipwise_output": clipwise_output,
            "beats_hidden": hidden,
        }
