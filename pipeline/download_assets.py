"""Download the primary HF model, with optional full training metadata."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-data',action='store_true')
    p.add_argument('--repo-id',default='kaiyangli1992/Mizar-159M');p.add_argument('--revision',default='main')
    a=p.parse_args()
    from huggingface_hub import snapshot_download
    patterns=['assets/**','checkpoints/final/stage3_q4_seed20260905.ckpt','provenance/local_bundle.json']
    if a.training_data:patterns.append('data/manifests/**')
    snapshot_download(repo_id=a.repo_id,revision=a.revision,local_dir=ROOT,allow_patterns=patterns)
    receipt=ROOT/'provenance/local_bundle.json';b=json.loads(receipt.read_text())
    if not a.training_data:
        b['datasets']={k:r for k,r in b['datasets'].items() if (ROOT/r['path']).exists()}
    b['download_repository']=a.repo_id;b['download_revision']=a.revision
    receipt.write_text(json.dumps(b,indent=2)+'\n')
    print('Downloaded assets. Run python pipeline/verify_bundle.py before use.')
if __name__=='__main__':main()
