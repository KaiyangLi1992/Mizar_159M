"""Verify packaged data, base assets and checkpoints without loading models."""
import argparse,json
from pathlib import Path
from common import require,sha256_file
ROOT=Path(__file__).resolve().parents[1]
def records(v):
    if isinstance(v,dict):
        if all(k in v for k in ['path','bytes','sha256']):yield v
        else:
            for x in v.values():yield from records(x)
    elif isinstance(v,list):
        for x in v:yield from records(x)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quick',action='store_true');a=p.parse_args()
    b=json.loads((ROOT/'provenance/local_bundle.json').read_text())
    require(b['recipe']=='strongAC_plus_AVQA_joint_pool_Q4','Wrong release lineage')
    expected = 1 if b.get('release_scope') == 'primary_model_and_training_data' else 7
    require(len(b['checkpoints'])==expected,'Unexpected checkpoint inventory')
    n=0
    for r in records(b):
        path=(ROOT/r['path']).resolve();require(path.is_relative_to(ROOT),'Path escapes bundle')
        require(path.is_file() and path.stat().st_size==r['bytes'],f'Missing or truncated: {path}')
        if not a.quick:require(sha256_file(path)==r['sha256'],f'Checksum mismatch: {path}')
        n+=1
    print(f'PASS: {n} files; quick={a.quick}')
if __name__=='__main__':main()
