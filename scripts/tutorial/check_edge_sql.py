"""Score one native Qwen ExecuTorch SQL smoke after timing; no model execution."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.data.db import execute_readonly_sql
from clinical_trial_mvp.evaluation.scorers import results_equivalent
from clinical_trial_mvp.serving.benchmark import snapshot_fingerprint


def check(path,source):
    record=json.loads(path.read_text())
    if record.get('model_id')!='Qwen/Qwen3-4B' or record.get('stage')!='run' or record.get('evidence')!='hardware' or record.get('status')!='completed':
        raise ValueError('Require a completed native Qwen3-4B run, not an eager/failed/substitute record')
    raw=record['generation']['text']
    sql=raw.strip().removesuffix('<|im_end|>').strip()
    case=next(c for c in json.loads((ROOT/'evals/evaluations.json').read_text()) if c['id']=='eval_001')
    settings=replace(Settings.from_env(),project_root=ROOT,db_path=ROOT/'data/db/clinical_trials.duckdb')
    actual=execute_readonly_sql(sql,settings)
    reference=execute_readonly_sql(case['reference_sql'],settings)
    fingerprint=snapshot_fingerprint(source)
    # Pinned source-artifact identity, not a historical performance record.
    dell='9d76380658964726b0ebbfd59e23fd141cee771d9a116197b9ea4caaa737b616'
    return dict(case_id='eval_001',question=case['question'],source=path.relative_to(ROOT).as_posix(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw_generated_text=raw,
        normalization='Removed only the terminal Qwen <|im_end|> token before SQL execution; raw output preserved.',
        sql=sql,sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),reference_sql=case['reference_sql'],
        generated_query=asdict(actual),reference_query=asdict(reference),execution_success=actual.ok,
        result_equivalence=actual.ok and reference.ok and results_equivalent(actual.rows,reference.rows),
        database_sha256=hashlib.sha256(settings.db_path.read_bytes()).hexdigest(),base_snapshot_sha256=fingerprint,
        recorded_dell_base_snapshot_sha256=dell,matches_dell_base_snapshot=fingerprint==dell,
        scope='One direct short-schema SQL smoke; no plan, repair, full workload or human semantic review. SQL checks occur after native timing.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'models/edge-qwen3-4b/source')
    args=parser.parse_args()
    result=check((ROOT/args.record).resolve(),(ROOT/args.source).resolve())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=ROOT/f'experiment-tracker/tutorial/edge/qwen3-4b-sql-smoke-{stamp}.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('SQL executed:',result['execution_success'],'reference match:',result['result_equivalence'])
    print('Saved:',output.relative_to(ROOT))
