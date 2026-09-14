"""Unrestricted greedy answer generation; preserves raw text and token IDs."""
import argparse,json,os
from pathlib import Path
from common import require,sha256_file

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--gpu-memory-utilization',type=float,default=0.8);a=p.parse_args()
    require(not a.output.exists(),f'Refuse overwrite: {a.output}');require(a.batch_size>0,'Batch size must be positive')
    import numpy as np
    import torch
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM,SamplingParams,__version__
    from transformers import AutoTokenizer
    manifest=json.loads((a.bundle/'manifest.json').read_text());rows=manifest['records']
    require(manifest.get('decoder_files'), 'Missing trained-decoder file inventory')
    for name,digest in manifest['decoder_files'].items():
        f=(a.bundle/name).resolve();require(f.is_relative_to(a.bundle.resolve()),'Decoder path escapes bundle')
        require(sha256_file(f)==digest,f'Decoder integrity failure: {name}')
    require(rows and len({str(r['id']) for r in rows})==len(rows),'Invalid input IDs')
    tok=AutoTokenizer.from_pretrained(a.bundle/'decoder',local_files_only=True)
    require(tok.eos_token_id==0,'Unexpected EOS token')
    llm=LLM(model=str(a.bundle/'decoder'),skip_tokenizer_init=True,dtype='float32',
             tensor_parallel_size=1,seed=1234,enable_prompt_embeds=True,max_model_len=1024,
             enforce_eager=True,gpu_memory_utilization=a.gpu_memory_utilization)
    sampling=SamplingParams(temperature=0,top_p=1,max_tokens=300,stop_token_ids=[0],detokenize=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    partial=a.output.with_suffix(a.output.suffix+'.partial.jsonl')
    require(not partial.exists(),f'Incomplete previous run: {partial}')
    result=[]
    with partial.open('x') as journal:
        for start in range(0,len(rows),a.batch_size):
            batch=rows[start:start+a.batch_size];prompts=[]
            for r in batch:
                f=a.bundle/r['prefix'];require(sha256_file(f)==r['prefix_sha256'],'Prefix integrity failure')
                x=np.load(f,allow_pickle=False);require(x.shape==(383,576) and np.isfinite(x).all(),'Invalid prefix')
                prompts.append({'prompt_embeds':torch.from_numpy(x.copy())})
            outputs=llm.generate(prompts,sampling,use_tqdm=False);require(len(outputs)==len(batch),'Missing generation')
            for r,o in zip(batch,outputs):
                g=o.outputs[0];require(g.finish_reason in ['stop','length'],'Generation did not finish')
                value={**r,'model_output':tok.decode(list(g.token_ids),skip_special_tokens=True),
                       'generated_token_ids':list(g.token_ids),'finish_reason':g.finish_reason,
                       'engine':'vllm','engine_version':__version__,'checkpoint_sha256':manifest['checkpoint_sha256']}
                result.append(value);journal.write(json.dumps(value,ensure_ascii=False)+'\n');journal.flush()
    require(len(result)==len(rows),'Incomplete output')
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(a.output)
if __name__=='__main__':main()
