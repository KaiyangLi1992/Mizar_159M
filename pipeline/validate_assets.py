"""Validate stage-specific assets against source or verified release identities."""
import argparse,json
from pathlib import Path
import yaml
from common import require,sha256_file
ROOT=Path(__file__).resolve().parents[1]
def lookup(obj,key):
    for part in key.split('.'):
        obj=obj[part if part in obj else int(part)]
    return obj

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--stage',choices=['s1','s2','s3-q4','all'],default='all')
    p.add_argument('--quick',action='store_true',help='Check existence and sizes only; not an integrity verification')
    a=p.parse_args();assets=yaml.safe_load(a.assets.read_text());ledger=json.loads((ROOT/'provenance/artifacts.json').read_text())
    bundle_path=ROOT/'provenance/local_bundle.json';bundle=json.loads(bundle_path.read_text()) if bundle_path.exists() else {}
    records=[];stages=['s1','s2','s3-q4'] if a.stage=='all' else [a.stage]
    for stage in stages:
        if stage=='s1':records.append(('s1_manifest',ledger['s1']['manifest']))
        elif stage=='s2':records.extend([('s2_manifest',ledger['s2']['manifest']),('s1_checkpoint',ledger['s1']['adopted_checkpoint'])])
        else:
            records.append(('s2_checkpoint',ledger['s2']['adopted_checkpoint']))
            records.extend((f's3_q4_manifests.{s}',r) for s,r in ledger['s3_q4']['manifests'].items())
    for key,ref in records:
        path=Path(lookup(assets,key));require(path.is_file(),f'Missing {key}: {path}')
        accepted=[ref]
        if key.endswith('_checkpoint'):
            export=bundle.get('checkpoints',{}).get('stage'+key[1],{})
            if export and export.get('source_sha256')==ref['sha256']:accepted.append(export)
        require(path.stat().st_size in [r['bytes'] for r in accepted],f'Size mismatch: {key}')
        if not a.quick:require(sha256_file(path) in [r['sha256'] for r in accepted],f'SHA mismatch: {key}')
        print('OK',key)
    for key in ['smollm2_135m','ced_small']:
        model=Path(assets[key]);require((model/'config.json').is_file(),f'Missing local model config: {model}')
        require(any(model.glob('*.safetensors')) or any(model.glob('*.bin')),f'Missing base weights: {model}')
    caches=[]
    if 's1' in stages:caches.append(assets['s1_ced_cache'])
    if 's2' in stages:caches.extend(assets['s2_ced_cache_roots'])
    for value in caches:
        root=Path(value);require(root.is_dir(),f'Missing CED cache: {root}; see docs/DATA.md')
        metadata=[root/'metadata.json'] if (root/'metadata.json').exists() else sorted(root.glob('*/metadata.json'))
        require(bool(metadata),f'No cache metadata: {root}')
        for f in metadata:
            m=json.loads(f.read_text());require(m.get('policy')=='fixed20' and m.get('cache_point')=='final_hidden',f'Wrong cache protocol: {f}')
            n=1
            for d in m['shape']:n*=d
            require((f.parent/'hidden.f16.dat').stat().st_size==n*2,f'Cache size mismatch: {f}')
            for name in ['keys.jsonl','valid_tokens.npy','segment_lengths.npy']:require((f.parent/name).is_file(),f'Missing {f.parent/name}')
    print('PASS',a.stage,'(quick check)' if a.quick else '(SHA-verified files)')
if __name__=='__main__':main()
