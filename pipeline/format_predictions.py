"""Validate full ID coverage and format raw answers for official evaluators."""
import argparse,json
from pathlib import Path
from common import require

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',choices=['mmar','adqa','mmau-full'],required=True)
    p.add_argument('--references',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    refs=json.loads(a.references.read_text());preds=json.loads(a.predictions.read_text())
    by={str(r['id']):r for r in preds};ids=[str(r['id']) for r in refs]
    require(len(by)==len(preds) and len(set(ids))==len(ids),'Duplicate IDs')
    require(set(by)==set(ids),'Missing or unexpected prediction IDs')
    expected={'mmar':1000,'adqa':1577,'mmau-full':9000}[a.benchmark]
    require(len(refs)==expected,f'Expected exactly {expected} references')
    if a.benchmark=='adqa':
        mask=json.loads((Path(__file__).resolve().parents[1]/'data/evaluation/ADQA_DEV_CLEAN_V1.json').read_text())
        require(set(ids)==set(map(str,mask['kept_ids'])),'Wrong ADQA mask')
    output=[]
    for ref in refs:
        pred=by[str(ref['id'])];raw=pred['model_output']
        require(isinstance(raw,str),'Raw generation must be text')
        require(pred.get('finish_reason') in ['stop','length'],'Invalid generation completion')
        output.append({**ref,'answer_prediction':raw,'model_output':raw})
    require(not a.output.exists(),f'Refuse overwrite: {a.output}')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(a.output)
if __name__=='__main__':main()
