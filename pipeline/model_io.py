"""Load the released architecture and a plain tensor checkpoint strictly."""
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/stage3'))
from models.mellow import Mellow

def load_model(checkpoint,device='cpu'):
    base=ROOT/'assets'
    enc={'audioenc_name':'CEDSmall','out_emb':384,'d_proj':576,
         'pretrained_audioencoder_path':str(base/'ced-small'),'input_sampling_rate':32000,
         'ced_sampling_rate':16000,'ced_mapper':'freq_merge_126','ced_output_tokens':126,
         'ced_mapper_dropout':0.05,'ced_variable_length':False,
         'freeze_audio_encoder_weights':True,'use_pretrained_audioencoder':True}
    dec={'text_decoder':str(base/'smollm2-135m'),'prefix_length':40,
         'total_prefix_length':383,'freeze_gpt_weights':False,
         'prefix_layout':'single_audio','use_chunk_position_embeddings':False}
    model=Mellow(audioenc_name='CEDSmall',d_in=384,text_decoder=dec['text_decoder'],
                 prefix_length=40,freeze_text_decoder_weights=False,d_out=576,
                 use_pretrained_audioencoder=True,freeze_audio_encoder_weights=True,
                 pretrained_audioencoder_path=enc['pretrained_audioencoder_path'],
                 model_variant='legacy',adapter_config={'enabled':False},
                 encoder_config=enc,decoder_config=dec)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model=model.float().eval().to(device)
    tokenizer=AutoTokenizer.from_pretrained(base/'smollm2-135m',local_files_only=True)
    tokenizer.add_special_tokens({'pad_token':'!'})
    if tokenizer.pad_token_id!=17 or tokenizer.eos_token_id!=0:raise ValueError('Tokenizer contract changed')
    return model,tokenizer
