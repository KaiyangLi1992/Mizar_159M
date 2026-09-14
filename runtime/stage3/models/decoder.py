
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from enum import Enum
# from transformers import GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from typing import Tuple, Optional, Union
from models.adapter import (
    ListenTwiceQFormerLiteAdapter,
    QuestionAudioCrossFiLMAdapter,
    QuestionAudioQFormerLiteAdapter,
    QuestionConditionedPrefixAdapter,
    QuestionFiLMAdapter,
)
from training.alignkd import llama_layer0_grouped_audio_attention
try:
    import torch.distributed.tensor
except (ImportError, AttributeError):
    pass
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


SINGLE_AUDIO_PREFIX_LAYOUTS = {"single_audio", "text_first_single_audio"}

def get_decoder(name: str):
    if name == "Decoder":
        return DecoderModel
    else:
        raise Exception('The decoder model {} is incorrect or not supported'.format(name))
    
def downsample(x):
    if x.shape[1] == 32:
        return x
    clip_latent = x[:,0,:].unsqueeze(1)
    pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
    x = torch.concat((clip_latent,pooled),axis=1)
    return x

class Downsampler(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        # self.fc = nn.Linear(din, dout)

    def forward(self, x):
        # x = self.fc(x)
        clip_latent = x[:,0,:].unsqueeze(1)
        # downsample the timesteps by 4. Downsample only the frame-level info
        pooled = nnf.avg_pool2d(x[:,1:,:], kernel_size=(8,1))
        # add clip-level latent back to audio
        x = torch.concat((clip_latent,pooled),axis=1)
        return x

class DecoderModel(nn.Module):
    def __init__(
        self,
        text_decoder: str,
        prefix_length: int,
        freeze_decoder_weights: bool = True,
        adapter_config: Optional[dict] = None,
        decoder_config: Optional[dict] = None,
    ):
        super(DecoderModel, self).__init__()
        self.prefix_length = prefix_length
        self.text_decoder = text_decoder.lower()
        self._last_adapter_stats = None
        decoder_config = decoder_config or {}
        listen_twice_config = decoder_config.get("listen_twice", {}) or {}
        self.listen_twice = bool(listen_twice_config.get("enabled", False))
        self.listen_twice_pass1_mode = str(listen_twice_config.get("pass1_mode", "qformer_soft"))
        self.detach_listen_state = bool(listen_twice_config.get("detach_listen_state", True))
        self.shuffle_listen_state = bool(listen_twice_config.get("shuffle_listen_state", False))
        self.listen_state_pooling = str(listen_twice_config.get("listen_state_pooling", "last")).lower()
        self.listen_memory = str(listen_twice_config.get("listen_memory", "none")).lower()
        self.append_listen_tokens = bool(listen_twice_config.get("append_listen_tokens", False))
        self.prefix_layout = str(decoder_config.get("prefix_layout", "two_audio")).lower()
        alignkd_config = dict(decoder_config.get("align_kd", {}) or {})
        self.alignkd_enabled = bool(alignkd_config.get("enabled", False))
        self.alignkd_attention_enabled = bool(
            alignkd_config.get("attention_enabled", False)
        )
        self.alignkd_target_audio_tokens = int(
            alignkd_config.get("target_audio_tokens", 126)
        )
        if self.alignkd_enabled and self.prefix_layout != "single_audio":
            raise ValueError(
                "Audio-AlignKD requires decoder.prefix_layout=single_audio so prompt "
                "queries can causally attend to the complete audio prefix"
            )
        if self.alignkd_attention_enabled and not self.alignkd_enabled:
            raise ValueError("align_kd.attention_enabled requires align_kd.enabled=true")
        self.use_chunk_position_embeddings = bool(
            decoder_config.get("use_chunk_position_embeddings", self.prefix_layout == "single20")
        )
        if self.append_listen_tokens:
            raise NotImplementedError("Listen Twice Stage A does not append listen tokens")
        # self.gpt = GPT2LMHeadModel.from_pretrained(text_decoder)
        self.lm = AutoModelForCausalLM.from_pretrained(text_decoder)
        if not ("gpt2" in self.text_decoder or "smollm2" in self.text_decoder):
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        self.gradient_checkpointing = bool(
            decoder_config.get("gradient_checkpointing", False)
        )
        if self.gradient_checkpointing:
            if not hasattr(self.lm, "gradient_checkpointing_enable"):
                raise RuntimeError(
                    "configured decoder does not support gradient checkpointing"
                )
            self.lm.gradient_checkpointing_enable()
            self.lm.config.use_cache = False
        self.lm_embedding_size = self.lm.get_input_embeddings().weight.shape[1]

        if freeze_decoder_weights:
            for p in self.lm.parameters():
                p.requires_grad = False

        lora_config = decoder_config.get("lora", {}) or {}
        self.lora = bool(lora_config.get("enabled", False))
        if self.lora:
            if LoraConfig is None or get_peft_model is None:
                raise ImportError("peft is required when model.decoder.lora.enabled=true")
            lora_config = LoraConfig(
                r=int(lora_config.get("r", 8)),
                lora_alpha=int(lora_config.get("alpha", 16)),
                target_modules=list(lora_config.get("target_modules", ["q_proj", "v_proj"])),
                lora_dropout=float(lora_config.get("dropout", 0.05)),
                bias=str(lora_config.get("bias", "none")),
                task_type="CAUSAL_LM"
            )
            self.lm = get_peft_model(self.lm, lora_config)

        adapter_config = adapter_config or {}
        self.prefix_adapter = None
        if adapter_config.get("enabled", False):
            adapter_type = str(adapter_config.get("type", "one_pass_residual"))
            adapter_cls = {
                "one_pass_residual": QuestionConditionedPrefixAdapter,
                "question_film": QuestionFiLMAdapter,
                "question_audio_cross_film": QuestionAudioCrossFiLMAdapter,
                "question_audio_qformer_lite": QuestionAudioQFormerLiteAdapter,
                "listen_twice_qformer_lite": ListenTwiceQFormerLiteAdapter,
            }.get(adapter_type)
            if adapter_cls is None:
                raise ValueError(f"Unknown model.adapter.type: {adapter_type}")
            self.prefix_adapter = adapter_cls(
                d_model=self.lm_embedding_size,
                n_heads=int(adapter_config.get("n_heads", 4)),
                mlp_ratio=int(adapter_config.get("mlp_ratio", 4)),
                dropout=float(adapter_config.get("dropout", 0.1)),
                alpha_init=float(adapter_config.get("alpha_init", 0.0)),
                num_queries=int(adapter_config.get("num_queries", 8)),
                num_layers=int(adapter_config.get("num_layers", 1)),
                gate_bias_init=float(adapter_config.get("gate_bias_init", 0.0)),
                gate_temperature=float(adapter_config.get("gate_temperature", 1.0)),
                gate_mode=str(adapter_config.get("gate_mode", "sigmoid")),
                select_scale=float(adapter_config.get("select_scale", 8.0)),
            )

        if self.prefix_layout in SINGLE_AUDIO_PREFIX_LAYOUTS:
            self.single20_time_boundary = None
            self.single20_chunk_position = None
        elif self.prefix_layout == "single20":
            self.single20_time_boundary = nn.Parameter(
                torch.empty(1, 1, self.lm_embedding_size)
            )
            nn.init.normal_(self.single20_time_boundary, mean=0.0, std=0.02)
            if self.use_chunk_position_embeddings:
                self.single20_chunk_position = nn.Parameter(
                    torch.empty(2, 1, self.lm_embedding_size)
                )
                nn.init.normal_(self.single20_chunk_position, mean=0.0, std=0.02)
            else:
                self.single20_chunk_position = None
        elif self.prefix_layout in {"two_audio", "mellow", "legacy"}:
            self.single20_time_boundary = None
            self.single20_chunk_position = None
        else:
            raise ValueError(f"Unknown decoder prefix_layout: {self.prefix_layout}")

    def embed_token_ids(self, input_ids):
        return self.lm.get_input_embeddings()(input_ids)

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def _embed_prompt(self, texts_enc):
        dtext = self.embed_token_ids(texts_enc["input_ids"])
        return dtext.contiguous()

    def _embed_answer(self, tokens):
        return self.embed_token_ids(tokens["input_ids"])

    def _embed_separator(self, batch_size, device):
        if "gpt" in self.text_decoder:
            sep_token = torch.tensor([50256], device=device)
        elif "smollm2" in self.text_decoder:
            sep_token = torch.tensor([0], device=device)
        else:
            raise ValueError(f"text decoder {self.text_decoder} not supported")
        sep_embed = self.embed_token_ids(sep_token)
        return sep_embed.unsqueeze(0).repeat(batch_size, 1, 1)

    def _mean_pool_prompt(self, dtext, texts_enc):
        # q_emb is intentionally computed only from input prompt tokens.
        # Never include target answer tokens here, or training will leak labels.
        attention_mask = texts_enc.get("attention_mask")
        if attention_mask is None:
            return dtext.mean(dim=1)

        mask = attention_mask.to(device=dtext.device, dtype=dtext.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (dtext * mask).sum(dim=1) / denom

    def _average_adapter_stats(self, stats1, stats2):
        stats = {}
        for key in sorted(set(stats1) | set(stats2)):
            if key in stats1 and key in stats2:
                stats[key] = (stats1[key] + stats2[key]) / 2
        return stats

    def _make_prefix(self, audio_projections1, audio_projections2, dtext, sep_embed=None):
        if sep_embed is None:
            sep_embed = self._embed_separator(dtext.shape[0], dtext.device)
        if self.prefix_layout in SINGLE_AUDIO_PREFIX_LAYOUTS:
            if audio_projections1.shape[-1] != self.lm_embedding_size:
                raise ValueError(
                    f"{self.prefix_layout} prefix requires audio projection dim to match decoder embedding dim: "
                    f"{audio_projections1.shape[-1]} vs {self.lm_embedding_size}"
                )
            if self.prefix_layout == "text_first_single_audio":
                return torch.cat((dtext, sep_embed, audio_projections1), dim=1)
            return torch.cat((audio_projections1, sep_embed, dtext), dim=1)
        if self.prefix_layout == "single20":
            if audio_projections1.shape[-1] != self.lm_embedding_size:
                raise ValueError(
                    "single20 prefix requires audio projection dim to match decoder embedding dim: "
                    f"{audio_projections1.shape[-1]} vs {self.lm_embedding_size}"
                )
            if self.single20_chunk_position is not None:
                pos = self.single20_chunk_position.to(
                    device=audio_projections1.device,
                    dtype=audio_projections1.dtype,
                )
                audio_projections1 = audio_projections1 + pos[0].unsqueeze(0)
                audio_projections2 = audio_projections2 + pos[1].unsqueeze(0)
            time_boundary = self.single20_time_boundary.to(
                device=audio_projections1.device,
                dtype=audio_projections1.dtype,
            ).repeat(dtext.shape[0], 1, 1)
            return torch.cat((audio_projections1, time_boundary, audio_projections2, sep_embed, dtext), dim=1)
        return torch.cat((audio_projections1, sep_embed, audio_projections2, sep_embed, dtext), dim=1)

    def _adapt_single_audio_prefix(
        self,
        audio_projections,
        q_emb,
        q_tokens=None,
        q_mask=None,
        listen_state=None,
        listen_tokens=None,
        return_stats=True,
    ):
        adapter_kwargs = {
            "q_emb": q_emb,
            "q_tokens": q_tokens,
            "q_mask": q_mask,
            "return_stats": return_stats,
        }
        if listen_state is not None:
            adapter_kwargs["listen_state"] = listen_state
        if listen_tokens is not None:
            adapter_kwargs["listen_tokens"] = listen_tokens

        return self.prefix_adapter(audio_projections, **adapter_kwargs)

    def _adapt_audio_prefixes(
        self,
        audio_projections1,
        audio_projections2,
        q_emb,
        q_tokens=None,
        q_mask=None,
        listen_state=None,
        listen_tokens1=None,
        listen_tokens2=None,
        update_stats=True,
    ):
        if self.prefix_adapter is None:
            self._last_adapter_stats = None
            return audio_projections1, audio_projections2

        if not update_stats:
            audio_projections1 = self._adapt_single_audio_prefix(
                audio_projections1,
                q_emb=q_emb,
                q_tokens=q_tokens,
                q_mask=q_mask,
                listen_state=listen_state,
                listen_tokens=listen_tokens1,
                return_stats=False,
            )
            audio_projections2 = self._adapt_single_audio_prefix(
                audio_projections2,
                q_emb=q_emb,
                q_tokens=q_tokens,
                q_mask=q_mask,
                listen_state=listen_state,
                listen_tokens=listen_tokens2,
                return_stats=False,
            )
            return audio_projections1, audio_projections2

        audio_projections1, stats1 = self._adapt_single_audio_prefix(
            audio_projections1,
            q_emb=q_emb,
            q_tokens=q_tokens,
            q_mask=q_mask,
            listen_state=listen_state,
            listen_tokens=listen_tokens1,
            return_stats=True,
        )
        audio_projections2, stats2 = self._adapt_single_audio_prefix(
            audio_projections2,
            q_emb=q_emb,
            q_tokens=q_tokens,
            q_mask=q_mask,
            listen_state=listen_state,
            listen_tokens=listen_tokens2,
            return_stats=True,
        )

        self._last_adapter_stats = self._average_adapter_stats(stats1, stats2)
        return audio_projections1, audio_projections2

    def _build_pass1_prefix(self, audio_projections1, audio_projections2, dtext, q_emb, q_mask, return_audio=False):
        mode = self.listen_twice_pass1_mode
        if mode in {"raw", "none"}:
            pass1_audio1, pass1_audio2 = audio_projections1, audio_projections2
        elif mode in {"qformer_soft", "adapter", "adapted"}:
            pass1_audio1, pass1_audio2 = self._adapt_audio_prefixes(
                audio_projections1,
                audio_projections2,
                q_emb,
                q_tokens=dtext,
                q_mask=q_mask,
                listen_state=None,
                update_stats=False,
            )
        else:
            raise ValueError(f"Unknown Listen Twice pass1_mode: {mode}")

        prefix = self._make_prefix(pass1_audio1, pass1_audio2, dtext)
        if return_audio:
            return prefix, pass1_audio1, pass1_audio2
        return prefix

    def _pool_listen_state(self, pass1_hidden, audio_projections1, audio_projections2, dtext, q_mask):
        mode = self.listen_state_pooling
        if mode in {"none", "off", "false", "0", "token_only"}:
            return None
        if mode in {"last", "prefix_last"}:
            return pass1_hidden[:, -1, :]

        prompt_start = audio_projections1.shape[1] + 1 + audio_projections2.shape[1] + 1
        prompt_hidden = pass1_hidden[:, prompt_start:prompt_start + dtext.shape[1], :]

        if mode in {"prompt_last_valid", "last_valid_prompt"}:
            if q_mask is None:
                return prompt_hidden[:, -1, :]
            prompt_lengths = q_mask.to(device=prompt_hidden.device).sum(dim=1).long().clamp_min(1)
            batch_idx = torch.arange(prompt_hidden.shape[0], device=prompt_hidden.device)
            return prompt_hidden[batch_idx, prompt_lengths - 1, :]

        if mode in {"prompt_mean", "mean_prompt"}:
            if q_mask is None:
                return prompt_hidden.mean(dim=1)
            mask = q_mask.to(device=prompt_hidden.device, dtype=prompt_hidden.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (prompt_hidden * mask).sum(dim=1) / denom

        raise ValueError(f"Unknown Listen Twice listen_state_pooling: {mode}")

    def _slice_listen_memory(self, pass1_hidden, pass1_audio1, pass1_audio2):
        if self.listen_memory in {"none", "off", "false", "0"}:
            return None, None
        if self.listen_memory not in {"audio", "audio_tokens", "prefix_audio"}:
            raise ValueError(f"Unknown Listen Twice listen_memory: {self.listen_memory}")

        audio1_len = pass1_audio1.shape[1]
        audio2_len = pass1_audio2.shape[1]
        audio1_hidden = pass1_hidden[:, :audio1_len, :]
        audio2_start = audio1_len + 1
        audio2_hidden = pass1_hidden[:, audio2_start:audio2_start + audio2_len, :]
        return audio1_hidden, audio2_hidden

    def _compute_listen_state(self, audio_projections1, audio_projections2, dtext, q_emb, q_mask, force_shuffle_listen_state=None):
        grad_enabled = not self.detach_listen_state
        with torch.set_grad_enabled(grad_enabled):
            prefix0, pass1_audio1, pass1_audio2 = self._build_pass1_prefix(
                audio_projections1,
                audio_projections2,
                dtext,
                q_emb,
                q_mask,
                return_audio=True,
            )
            pass1_outputs = self.lm(
                inputs_embeds=prefix0,
                output_hidden_states=True,
                use_cache=False,
            )
            pass1_hidden = pass1_outputs.hidden_states[-1]
            listen_state = self._pool_listen_state(
                pass1_hidden,
                audio_projections1,
                audio_projections2,
                dtext,
                q_mask,
            )
            listen_tokens1, listen_tokens2 = self._slice_listen_memory(
                pass1_hidden,
                pass1_audio1,
                pass1_audio2,
            )

        if self.detach_listen_state:
            if listen_state is not None:
                listen_state = listen_state.detach()
            if listen_tokens1 is not None:
                listen_tokens1 = listen_tokens1.detach()
            if listen_tokens2 is not None:
                listen_tokens2 = listen_tokens2.detach()
        shuffle_listen_state = self.shuffle_listen_state if force_shuffle_listen_state is None else bool(force_shuffle_listen_state)
        if shuffle_listen_state and pass1_hidden.shape[0] > 1:
            if listen_state is not None:
                listen_state = torch.roll(listen_state, shifts=1, dims=0)
            if listen_tokens1 is not None:
                listen_tokens1 = torch.roll(listen_tokens1, shifts=1, dims=0)
            if listen_tokens2 is not None:
                listen_tokens2 = torch.roll(listen_tokens2, shifts=1, dims=0)
        return listen_state, listen_tokens1, listen_tokens2, pass1_audio1, pass1_audio2

    def _add_listen_twice_prefix_delta_stats(self, pass1_audio1, pass1_audio2, pass2_audio1, pass2_audio2):
        if self._last_adapter_stats is None:
            return

        with torch.no_grad():
            pass1_audio1 = pass1_audio1.detach()
            pass1_audio2 = pass1_audio2.detach()
            pass2_audio1 = pass2_audio1.detach()
            pass2_audio2 = pass2_audio2.detach()
            pass1_norm = (
                torch.linalg.vector_norm(pass1_audio1)
                + torch.linalg.vector_norm(pass1_audio2)
            ) / 2
            pass2_norm = (
                torch.linalg.vector_norm(pass2_audio1)
                + torch.linalg.vector_norm(pass2_audio2)
            ) / 2
            delta_norm = (
                torch.linalg.vector_norm(pass2_audio1 - pass1_audio1)
                + torch.linalg.vector_norm(pass2_audio2 - pass1_audio2)
            ) / 2
            self._last_adapter_stats["pass1_audio_norm"] = pass1_norm
            self._last_adapter_stats["pass2_audio_norm"] = pass2_norm
            self._last_adapter_stats["pass2_minus_pass1_audio_norm"] = delta_norm
            self._last_adapter_stats["pass2_minus_pass1_audio_ratio"] = (
                delta_norm / pass1_norm.clamp_min(1e-12)
            )

    def _build_answer_prefix(self, audio_projections1, audio_projections2, dtext, q_emb, q_mask, force_shuffle_listen_state=None):
        if self.prefix_layout in SINGLE_AUDIO_PREFIX_LAYOUTS:
            if self.listen_twice:
                raise NotImplementedError(
                    f"{self.prefix_layout} prefix_layout does not support Listen Twice yet"
                )
            if self.prefix_adapter is not None:
                audio_projections1, stats = self._adapt_single_audio_prefix(
                    audio_projections1,
                    q_emb=q_emb,
                    q_tokens=dtext,
                    q_mask=q_mask,
                    return_stats=True,
                )
                self._last_adapter_stats = stats
            else:
                self._last_adapter_stats = None
            return self._make_prefix(audio_projections1, None, dtext)

        if self.listen_twice:
            listen_state, listen_tokens1, listen_tokens2, pass1_audio1, pass1_audio2 = self._compute_listen_state(
                audio_projections1,
                audio_projections2,
                dtext,
                q_emb,
                q_mask,
                force_shuffle_listen_state=force_shuffle_listen_state,
            )
            audio_projections1, audio_projections2 = self._adapt_audio_prefixes(
                audio_projections1,
                audio_projections2,
                q_emb,
                q_tokens=dtext,
                q_mask=q_mask,
                listen_state=listen_state,
                listen_tokens1=listen_tokens1,
                listen_tokens2=listen_tokens2,
            )
            self._add_listen_twice_prefix_delta_stats(
                pass1_audio1,
                pass1_audio2,
                audio_projections1,
                audio_projections2,
            )
        else:
            audio_projections1, audio_projections2 = self._adapt_audio_prefixes(
                audio_projections1,
                audio_projections2,
                q_emb,
                q_tokens=dtext,
                q_mask=q_mask,
            )

        return self._make_prefix(audio_projections1, audio_projections2, dtext)

    def get_adapter_alpha(self):
        if self.prefix_adapter is None:
            return None
        return self.prefix_adapter.alpha.detach()

    def get_adapter_stats(self):
        return self._last_adapter_stats
    
    def generate_prefix_inference(self, daudio1, daudio2, texts_enc):
        if self.prefix_layout in SINGLE_AUDIO_PREFIX_LAYOUTS:
            audio_projections1 = daudio1.contiguous()
            audio_projections2 = None
        else:
            audio_projections1 = downsample(daudio1).contiguous()
            audio_projections2 = downsample(daudio2).contiguous()
        dtext = self._embed_prompt(texts_enc)
        q_emb = self._mean_pool_prompt(dtext, texts_enc)
        q_mask = texts_enc.get("attention_mask")
        return self._build_answer_prefix(audio_projections1, audio_projections2, dtext, q_emb, q_mask)

    def forward(
        self,
        daudio1: torch.Tensor,
        daudio2: torch.Tensor,
        texts_enc: torch.Tensor,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        force_shuffle_listen_state=None,
        alignkd_query_group_mask: Optional[torch.Tensor] = None,
    ):

        dtext = self._embed_prompt(texts_enc)
        embedding_text = self._embed_answer(tokens)
        sep_embed = self._embed_separator(dtext.shape[0], dtext.device)
        
        if self.prefix_layout in SINGLE_AUDIO_PREFIX_LAYOUTS:
            audio_projections1 = daudio1.contiguous()
            audio_projections2 = None
        else:
            audio_projections1 = downsample(daudio1).contiguous()
            audio_projections2 = downsample(daudio2).contiguous()
        q_emb = self._mean_pool_prompt(dtext, texts_enc)
        q_mask = texts_enc.get("attention_mask")
        prefix = self._build_answer_prefix(
            audio_projections1,
            audio_projections2,
            dtext,
            q_emb,
            q_mask,
            force_shuffle_listen_state=force_shuffle_listen_state,
        )
        alignkd_aux = None
        if self.alignkd_enabled:
            if audio_projections1.shape[1] != self.alignkd_target_audio_tokens:
                raise ValueError(
                    "Audio-AlignKD student mapper output length mismatch: "
                    f"expected={self.alignkd_target_audio_tokens} "
                    f"got={audio_projections1.shape[1]}"
                )
            alignkd_aux = {
                # This is the exact mapper output used by the LM in this
                # forward.  Do not recompute it: mapper dropout would differ.
                "student_audio_features": audio_projections1,
            }
            if self.alignkd_attention_enabled:
                if alignkd_query_group_mask is None:
                    raise ValueError(
                        "align_kd attention is enabled but alignkd_query_group_mask is missing"
                    )
                student_attention, student_audio_mass = (
                    llama_layer0_grouped_audio_attention(
                        self.lm,
                        prefix,
                        alignkd_query_group_mask,
                        audio_tokens=self.alignkd_target_audio_tokens,
                    )
                )
                alignkd_aux["student_attn_qabcd_126"] = student_attention
                alignkd_aux["student_audio_mass_qabcd"] = student_audio_mass
        embedding_cat = torch.cat((prefix, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens['input_ids'].shape[0], tokens['input_ids'].device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.lm(
            inputs_embeds=embedding_cat,
            labels=labels,
            attention_mask=mask,
            use_cache=False,
        )
        if alignkd_aux is not None:
            # Hugging Face ModelOutput dataclasses permit auxiliary attributes;
            # existing callers still consume the unchanged .loss/.logits API.
            out.alignkd_aux = alignkd_aux
        return out
