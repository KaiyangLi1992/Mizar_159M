"""Unpack exact released manifests, verifying compressed and original SHA-256."""
import argparse, gzip, json, shutil
from pathlib import Path
from common import require, sha256_file
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'data/prepared');a=p.parse_args()
    bundle=json.loads((ROOT/'provenance/local_bundle.json').read_text())
    a.output.mkdir(parents=True,exist_ok=True)
    for key,ref in bundle['datasets'].items():
        if 'uncompressed_sha256' not in ref:continue
        src=ROOT/ref['path'];require(sha256_file(src)==ref['sha256'],f'Corrupt archive: {src}')
        dest=a.output/(key+'.json')
        if not dest.exists():
            temp=dest.with_suffix('.partial')
            with gzip.open(src,'rb') as f,temp.open('wb') as out:shutil.copyfileobj(f,out,8<<20)
            require(sha256_file(temp)==ref['uncompressed_sha256'],f'Corrupt decoded file: {src}')
            temp.rename(dest)
        require(sha256_file(dest)==ref['uncompressed_sha256'],f'Existing output differs: {dest}')
        print(dest)
if __name__=='__main__':main()
