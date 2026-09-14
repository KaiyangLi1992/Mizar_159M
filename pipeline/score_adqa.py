"""Run the frozen official ADQA parser on exactly the clean 1,577 IDs."""
import argparse,importlib.util,json
from pathlib import Path
from common import require,sha256_file
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),f'Refuse overwrite: {a.output}')
    evaluator=ROOT/'data/evaluation/adqa_official_evaluator.py'
    require(sha256_file(evaluator)=='bd3e690ed5f727a38ef636b58340481b88f04d4171ccf5707074a78ac0dd9fd4','Official evaluator changed')
    spec=importlib.util.spec_from_file_location('official_adqa',evaluator);official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    gold=official.load_ground_truth(ROOT/'data/evaluation/adqa_clean_ground_truth.jsonl')
    mask=json.loads((ROOT/'data/evaluation/ADQA_DEV_CLEAN_V1.json').read_text())
    keep=set(map(str,mask['kept_ids']));require(len(keep)==1577 and set(gold)==keep,'Wrong clean gold IDs')
    rows=json.loads(a.predictions.read_text());preds={str(r['id']):r['model_output'] for r in rows}
    require(len(preds)==len(rows)==1577 and set(preds)==keep,'Prediction coverage mismatch')
    require(all(isinstance(v,str) for v in preds.values()),'Predictions must preserve raw text')
    require(all(r.get('finish_reason') in ['stop','length'] for r in rows),'Unfinished generation')
    result=official.evaluate(gold,preds)
    require(result['total_scored']==1577 and result['missing_predictions']==0,'Incomplete score')
    result.update(protocol='official_ADQA_clean1577',evaluator_sha256=sha256_file(evaluator))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
