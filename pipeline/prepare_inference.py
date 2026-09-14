"""Encode audio/text prefixes and export the trained decoder for vLLM."""
import argparse,hashlib,json,os
from pathlib import Path
from common import sha256_file, require
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True,help='JSON array: id, audio, question; optional choices')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();require(not a.output.exists(),f'Refuse overwrite: {a.output}')
    import numpy as np
    import torch,torchaudio
    from model_io import load_model
    from utils.audio_paths import resolve_audio_path
    torch.set_num_threads(a.threads);torch.manual_seed(1234)
    rows=json.loads(a.input.read_text());require(isinstance(rows,list) and rows,'Nonempty JSON array required')
    require(len({str(r['id']) for r in rows})==len(rows),'Duplicate IDs')
    model,tok=load_model(a.checkpoint,a.device)
    a.output.mkdir(parents=True)
    decoder=a.output/'decoder';model.caption_decoder.lm.save_pretrained(decoder,safe_serialization=True);tok.save_pretrained(decoder)
    records=[]
    for i,r in enumerate(rows):
        require(not r.get('filepath2') or r.get('filepath2') == r.get('filepath1'), 'Inference requires one original recording per question')
        path=Path(resolve_audio_path(r.get('audio',r.get('filepath1',''))))
        if not path.is_absolute():path=a.input.resolve().parent/path
        require(path.is_file(),f'Missing audio: {path}')
        wav,sr=torchaudio.load(path)
        require(wav.numel()>0 and bool(torch.isfinite(wav).all()),f'Invalid waveform: {path}')
        if sr!=32000:wav=torchaudio.transforms.Resample(sr,32000)(wav)
        wav=wav.mean(dim=0)[:640000]
        wav=torch.nn.functional.pad(wav,(0,640000-wav.numel())).float()
        if 'input' in r:prompt=r['input']
        else:
            prompt=r['question']
            if 'choices' in r:
                require(len(r['choices'])==4,'Expected four choice texts')
                prompt+=' '+' '.join(f'{c}) {s}' for c,s in zip('abcd',r['choices']))
        prompt=prompt.lower();require('<|endoftext|>' not in prompt,'Question-side EOT is prohibited')
        text=tok([prompt],return_tensors='pt',padding='max_length',truncation=True,max_length=256)
        text={k:v.to(a.device) for k,v in text.items()}
        with torch.inference_mode():
            prefix,_,_=model.generate_prefix_inference({'audio1':wav.unsqueeze(0).to(a.device),'audio2':torch.empty(1,0,device=a.device),'input':text})
        require(tuple(prefix.shape)==(1,383,576),f'Unexpected prefix shape: {prefix.shape}')
        require(bool(torch.isfinite(prefix).all()),'Nonfinite prefix')
        dest=a.output/f'prefix_{i:06d}.npy';np.save(dest,prefix[0].cpu().numpy().astype('<f4'))
        records.append({'id':r['id'],'prompt':prompt,'input_ids':text['input_ids'][0].cpu().tolist(),
                        'prefix':dest.name,'prefix_sha256':sha256_file(dest),
                        'audio_file_sha256':sha256_file(path),
                        'waveform_sha256':hashlib.sha256(wav.numpy().astype('<f4').tobytes()).hexdigest()})
    (a.output/'manifest.json').write_text(json.dumps({'checkpoint_sha256':sha256_file(a.checkpoint),
        'input_sha256':sha256_file(a.input),'audio':'mono, resample32k, first20s, zero-pad',
        'torch_version':torch.__version__,'prefix_dtype':'float32',
        'decoder_files':{str(f.relative_to(a.output)):sha256_file(f) for f in sorted(decoder.rglob('*')) if f.is_file()},
        'records':records},indent=2)+'\n')
    print(a.output/'manifest.json')
if __name__=='__main__':main()
