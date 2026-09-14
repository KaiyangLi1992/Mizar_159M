import sys
sys.path.append('')
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
import os

from models.audio import get_audio_encoder
from models.decoder import get_decoder


ADAPTER_VARIANTS = {
    "one_pass_adapter",
    "one_pass_adapter_train_map",
    "one_pass_adapter_lora",
    "one_pass_adapter_train_map_lora",
    "q_audio_film_adapter",
    "q_audio_film_adapter_train_map",
    "q_audio_cross_adapter",
    "q_audio_cross_adapter_train_map",
    "q_audio_qformer_adapter",
    "q_audio_qformer_adapter_train_map",
    "q_audio_qformer_adapter_lora",
    "q_audio_qformer_adapter_train_map_lora",
    "listen_twice_qformer_adapter",
    "listen_twice_qformer_adapter_train_map",
    "listen_twice_qformer_adapter_lora",
    "listen_twice_qformer_adapter_train_map_lora",
}

MAP_TRAIN_VARIANTS = {
    "train_map_only",
    "train_map_lora",
    "one_pass_adapter_train_map",
    "one_pass_adapter_train_map_lora",
    "q_audio_film_adapter_train_map",
    "q_audio_cross_adapter_train_map",
    "q_audio_qformer_adapter_train_map",
    "q_audio_qformer_adapter_train_map_lora",
    "listen_twice_qformer_adapter_train_map",
    "listen_twice_qformer_adapter_train_map_lora",
}

LORA_VARIANTS = {
    "lora_only",
    "train_map_lora",
    "one_pass_adapter_lora",
    "one_pass_adapter_train_map_lora",
    "q_audio_qformer_adapter_lora",
    "q_audio_qformer_adapter_train_map_lora",
    "listen_twice_qformer_adapter_lora",
    "listen_twice_qformer_adapter_train_map_lora",
}

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(0.)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        """Initialize a Batchnorm layer. """
        m.bias.data.fill_(0.)
        m.weight.data.fill_(1.)

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.linear1)
        init_layer(self.linear2)
        init_bn(self.layer_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds

class AudioEncoder(nn.Module):
    def __init__(self, 
                 audioenc_name:str, 
                 d_in: int, d_out: int, 
                 use_pretrained_audioencoder: bool, 
                 freeze_audio_encoder_weights: bool,
                 pretrained_audioencoder_path: str,
                 encoder_config: dict = None) -> None:
        super().__init__()

        audio_encoder, pretrained_emb_size = get_audio_encoder(audioenc_name)
        encoder_config = dict(encoder_config or {})

        if use_pretrained_audioencoder:
            d_in = pretrained_emb_size

        if audioenc_name in {
            "CEDSmall", "CED-Small", "CED_SMALL", "CED",
            "BEATsBase", "BEATs", "BEATSBase", "BEATS", "BEATs-Base",
            "HTSATSingleAudio126", "HTSAT-126", "HTSAT126",
        }:
            encoder_config.setdefault("freeze_audio_encoder_weights", freeze_audio_encoder_weights)
            encoder_config.setdefault("pretrained_audioencoder_path", pretrained_audioencoder_path)
            encoder_config.setdefault("use_pretrained_audioencoder", use_pretrained_audioencoder)
            self.base = audio_encoder(encoder_config=encoder_config, d_out=d_out)
            self.projection = nn.Identity()
            self.single_audio_mode = True
            return

        self.base = audio_encoder()
        self.single_audio_mode = False
        
        if use_pretrained_audioencoder:
            if not pretrained_audioencoder_path:
                raise ValueError("use_pretrained_audioencoder=true requires pretrained_audioencoder_path")
            if audioenc_name == 'HTSAT':
                # Load pretrained weights for HTSAT. Stage D may pass a full
                # Mellow checkpoint and intentionally extract only the raw
                # frozen HTSAT weights, leaving the map/SLM freshly initialized.
                pretrained_model_path = pretrained_audioencoder_path
                if os.path.isdir(pretrained_model_path):
                    pretrained_model_path = os.path.join(pretrained_model_path, 'HTSAT_AudioSet_Saved_1.ckpt')
                ckpt = torch.load(pretrained_model_path, map_location="cpu")
                state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                new_ckpt = {}
                for key, value in state.items():
                    if key.startswith("audio_encoder.base.htsat."):
                        new_ckpt[key[len("audio_encoder.base.htsat."):]] = value
                    elif key.startswith("htsat."):
                        new_ckpt[key[len("htsat."):]] = value
                    elif key.startswith("sed_model."):
                        new_ckpt[key[len("sed_model."):]] = value
                    elif key.startswith("model."):
                        new_ckpt[key[len("model."):]] = value
                if not new_ckpt:
                    for key, value in state.items():
                        # Original HTSAT checkpoints in this codebase used a
                        # 10-character prefix before the actual module names.
                        new_ckpt[key[10:]] = value
                self.base.htsat.load_state_dict(new_ckpt)
            elif audioenc_name == 'Cnn14':
                # Load pretrained weights for Cnn14
                pretrained_model_path = os.path.join(pretrained_audioencoder_path, 'Cnn14_mAP=0.431.pth')
                ckpt = torch.load(pretrained_model_path, map_location="cpu")["model"]
                self.base.cnn14.load_state_dict(ckpt)
            else:
                raise NotImplementedError('Add loading audio encoder weights code for {}'.format(audioenc_name))

        self.projection = Projection(pretrained_emb_size if use_pretrained_audioencoder else d_in, d_out)

        if freeze_audio_encoder_weights:
            if audioenc_name == 'HTSAT':
                # Match the official Mellow training branch: freeze the HTSAT
                # core, while leaving HTSATWrapper.c2l trainable.
                for p in self.base.htsat.parameters():
                    p.requires_grad = False
            elif audioenc_name == 'Cnn14':
                for p in self.base.cnn14.parameters():
                    p.requires_grad = False

    def forward(self, x):
        out_dict = self.base(x)
        audio_features, audio_classification_output = out_dict['embedding'], out_dict['clipwise_output']
        projected_vec = self.projection(audio_features)
        return projected_vec, audio_classification_output, out_dict

class Mellow(nn.Module):
    def __init__(self,
                # audio
                audioenc_name: str,
                d_in: int,
                # text decoder
                text_decoder: str,
                prefix_length: int,
                freeze_text_decoder_weights: bool,
                # common
                d_out: int,
                use_pretrained_audioencoder: bool,
                freeze_audio_encoder_weights: bool,
                pretrained_audioencoder_path: str = None,
                model_variant: str = "legacy",
                adapter_config: dict = None,
                decoder_config: dict = None,
                encoder_config: dict = None,
                ):
        super().__init__()
        self.model_variant = model_variant or "legacy"
        adapter_config = dict(adapter_config or {})
        decoder_config = dict(decoder_config or {})
        self.adapter_config = adapter_config
        if self.model_variant in ADAPTER_VARIANTS:
            adapter_config["enabled"] = True
        elif self.model_variant in {"legacy", "original", "train_map_only", "lora_only", "train_map_lora"}:
            adapter_config["enabled"] = bool(adapter_config.get("enabled", False)) and self.model_variant == "legacy"
        if self.model_variant in LORA_VARIANTS:
            decoder_config.setdefault("lora", {})
            decoder_config["lora"]["enabled"] = True

        self.audio_encoder = AudioEncoder(
            audioenc_name, d_in, d_out,
            use_pretrained_audioencoder, freeze_audio_encoder_weights,
            pretrained_audioencoder_path,
            encoder_config=encoder_config)

        self.caption_decoder = get_decoder('Decoder')(
            text_decoder, prefix_length, freeze_text_decoder_weights, adapter_config, decoder_config,
        )
        self.configure_trainable_parameters()

    def _set_requires_grad(self, module, requires_grad):
        for p in module.parameters():
            p.requires_grad = requires_grad

    def configure_trainable_parameters(self):
        if self.model_variant == "legacy":
            return

        self._set_requires_grad(self, False)

        if self.model_variant == "original":
            return

        # Keep the raw audio base and dense SLM weights frozen; LoRA variants
        # re-enable only the injected low-rank SLM adapter weights below.
        self._set_requires_grad(self.audio_encoder.base, False)
        self._set_requires_grad(self.caption_decoder.lm, False)

        if self.model_variant in MAP_TRAIN_VARIANTS:
            self._set_requires_grad(self.audio_encoder.projection, True)

        if self.model_variant in ADAPTER_VARIANTS:
            if self.caption_decoder.prefix_adapter is None:
                raise ValueError(f"{self.model_variant} requires model.adapter.enabled=true")
            self._set_requires_grad(self.caption_decoder.prefix_adapter, True)
            listen_memory = str(getattr(self.caption_decoder, "listen_memory", "none")).lower()
            train_listen_memory = listen_memory not in {"none", "off", "false", "0"}
            listen_pooling = str(getattr(self.caption_decoder, "listen_state_pooling", "last")).lower()
            train_listen_state = listen_pooling not in {"none", "off", "false", "0", "token_only"}
            memory_prefixes = (
                "listen_memory_ln",
                "listen_query_ln",
                "listen_memory_attn",
                "listen_token_to_gate",
            )
            if not train_listen_memory:
                for name, param in self.caption_decoder.prefix_adapter.named_parameters():
                    if name.startswith(memory_prefixes):
                        param.requires_grad = False
            if not train_listen_state:
                for name, param in self.caption_decoder.prefix_adapter.named_parameters():
                    if name.startswith(("listen_to_query", "listen_to_gate")):
                        param.requires_grad = False
            if (
                self.model_variant == "listen_twice_qformer_adapter"
                and self.adapter_config.get("train_listen_only", False)
            ):
                self._set_requires_grad(self.caption_decoder.prefix_adapter, False)
                for name, param in self.caption_decoder.prefix_adapter.named_parameters():
                    if train_listen_state and name.startswith(("listen_to_query", "listen_to_gate")):
                        param.requires_grad = True
                    elif train_listen_memory and name.startswith(memory_prefixes):
                        param.requires_grad = True

        if getattr(self.caption_decoder, "lora", False):
            for name, param in self.caption_decoder.lm.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def get_adapter_alpha(self):
        return self.caption_decoder.get_adapter_alpha()

    def get_adapter_stats(self):
        return self.caption_decoder.get_adapter_stats()

    def forward(self, input_dict, force_shuffle_listen_state=None):
        audio1 = input_dict['audio1']
        audio2 = input_dict['audio2']
        texts_enc = input_dict['input']
        texts_dec = input_dict['answer']

        if "ced_hidden" in input_dict:
            audio_input = {
                "ced_hidden": input_dict["ced_hidden"],
            }
            if "ced_hidden_segment_lengths" in input_dict:
                audio_input["ced_hidden_segment_lengths"] = input_dict["ced_hidden_segment_lengths"]
            audio_embed1, _, _ = self.audio_encoder(audio_input)
            audio_embed2 = None
        elif "beats_hidden" in input_dict:
            audio_input = {
                "beats_hidden": input_dict["beats_hidden"],
            }
            if "beats_hidden_lengths" in input_dict:
                audio_input["beats_hidden_lengths"] = input_dict["beats_hidden_lengths"]
            audio_embed1, _, _ = self.audio_encoder(audio_input)
            audio_embed2 = None
        elif getattr(self.audio_encoder, "single_audio_mode", False):
            audio_20s = torch.cat((audio1, audio2), dim=-1)
            audio_embed1, _, _ = self.audio_encoder(audio_20s)
            audio_embed2 = None
        else:
            audio_embed1, _, _ = self.audio_encoder(audio1)
            audio_embed2, _, _ = self.audio_encoder(audio2)
        out = self.caption_decoder(
            audio_embed1,
            audio_embed2,
            texts_enc,
            texts_dec,
            force_shuffle_listen_state=force_shuffle_listen_state,
            alignkd_query_group_mask=input_dict.get("alignkd_query_group_mask"),
        )
        return out
    
    def generate_prefix_inference(self, input_dict):
        audio1 = input_dict['audio1']
        audio2 = input_dict['audio2']
        texts_enc = input_dict['input']

        if "ced_hidden" in input_dict:
            audio_input = {
                "ced_hidden": input_dict["ced_hidden"],
            }
            if "ced_hidden_segment_lengths" in input_dict:
                audio_input["ced_hidden_segment_lengths"] = input_dict["ced_hidden_segment_lengths"]
            audio_embed1, _, od1 = self.audio_encoder(audio_input)
            audio_embed2, od2 = None, None
        elif "beats_hidden" in input_dict:
            audio_input = {
                "beats_hidden": input_dict["beats_hidden"],
            }
            if "beats_hidden_lengths" in input_dict:
                audio_input["beats_hidden_lengths"] = input_dict["beats_hidden_lengths"]
            audio_embed1, _, od1 = self.audio_encoder(audio_input)
            audio_embed2, od2 = None, None
        elif getattr(self.audio_encoder, "single_audio_mode", False):
            audio_20s = torch.cat((audio1, audio2), dim=-1)
            audio_embed1, _, od1 = self.audio_encoder(audio_20s)
            audio_embed2, od2 = None, None
        else:
            audio_embed1, _, od1 = self.audio_encoder(audio1)
            audio_embed2, _, od2 = self.audio_encoder(audio2)
        prefix = self.caption_decoder.generate_prefix_inference(audio_embed1, audio_embed2, texts_enc)
        return prefix, od1, od2
