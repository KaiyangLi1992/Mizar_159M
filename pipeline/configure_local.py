"""Configure bundled assets with absolute paths on the reviewer's machine."""
import argparse,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'configs/assets.local.yaml');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    assets=yaml.safe_load((ROOT/'configs/assets.example.yaml').read_text())
    assets.update(hf_cache=str(ROOT/'cache/huggingface'),smollm2_135m=str(ROOT/'assets/smollm2-135m'),ced_small=str(ROOT/'assets/ced-small'))
    for s in ['s1','s2']:
        assets[s+'_manifest']=str(ROOT/f'data/prepared/{s}.json')
        assets[s+'_checkpoint']=str(ROOT/f'checkpoints/intermediate/stage{s[-1]}.ckpt')
    assets['s3_candidates']=str(ROOT/'data/prepared/s3_candidates.json')
    assets['s1_ced_cache']=str(ROOT/'cache/s1')
    assets['s2_ced_cache_roots']=[str(ROOT/f'cache/s2/shard{i}') for i in range(4)]
    assets['s3_q4_manifests']={s:str(ROOT/f'data/prepared/s3_q4_seed{s}.json') for s in [20260905,20260906,20260907,20260908,20260909]}
    for k,v in assets['evaluation'].items():
        if k=='adqa_model_input':n='adqa_clean.json'
        elif k=='adqa_ground_truth':n='adqa_clean_ground_truth.jsonl'
        elif k=='adqa_clean_ids':n='ADQA_DEV_CLEAN_V1.json'
        else:n=k+('.py' if 'evaluator' in k else '.json')
        assets['evaluation'][k]=str(ROOT/'data/evaluation'/n)
    (ROOT/'cache/huggingface').mkdir(parents=True,exist_ok=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(yaml.safe_dump(assets,sort_keys=False))
    print(a.output)
if __name__=='__main__':main()
