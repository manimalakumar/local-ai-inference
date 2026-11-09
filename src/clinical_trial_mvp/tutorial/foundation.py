"""Read-only adaptation of the prior serving tutorial; reuse schema/harness/evidence."""
import hashlib
import json
from pathlib import Path

from clinical_trial_mvp.data.schema import RELATIONSHIPS
from clinical_trial_mvp.serving.benchmark import Workload, prepare
from clinical_trial_mvp.tutorial.harness import TUTORIAL_CASES


def checked(root, relative):
    manifest = json.loads((Path(root) / 'tutorial/serving_foundation.json').read_text())
    data = (Path(root) / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest['files_sha256'][relative]:
        raise ValueError('Foundation evidence/data checksum mismatch: ' + relative)
    return data


def inventory(root):
    import duckdb
    path = Path(root) / 'data/db/clinical_trials.duckdb'
    checked(root, 'data/db/clinical_trials.duckdb')
    checked(root, 'evals/evaluations.json')
    with duckdb.connect(str(path), read_only=True) as db:
        tables = [row[0] for row in db.execute('SHOW TABLES').fetchall()]
        sample = db.execute('SELECT phase, indication, target_enrollment FROM trials LIMIT 3').fetchall()
    return dict(origin='Frozen synthetic database; read-only inspection, no model run',
                database_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), tables=tables,
                relationships=RELATIONSHIPS, sample_columns=['phase', 'indication', 'target_enrollment'], sample=sample)


def contexts(root):
    checked(root, 'experiment-tracker/mvp3/retrieval-cache/current/evaluations.json')
    full = prepare(Path(root), Workload(profile='full'), TUTORIAL_CASES)
    selected = prepare(Path(root), Workload(profile='retrieved'), TUTORIAL_CASES)
    rows = []
    for case in full[0]:
        question = case['question']
        complete, retrieved = full[1][question], selected[1][question]
        rows.append(dict(case_id=case['id'], question=question,
            full_schema_characters=len(complete[0]), retrieved_schema_characters=len(retrieved[0]),
            selected_tables=retrieved[1].get('tables'),
            full_prompt=full[2][case['id']], retrieved_prompt=selected[2][case['id']]))
    return dict(note='Character counts are not token counts. Frozen retrieval is reused; no embedding model is loaded.',
                case_ids=full[3]['case_ids'], rows=rows)


def mac_history(root):
    manifest = json.loads((Path(root) / 'tutorial/serving_foundation.json').read_text())
    for path in manifest['report_sources']:
        checked(root, path)
    return dict(origin='Historical Mac report transcription, not a fresh benchmark',
                context=manifest['context_report_rows'], serving=manifest['mac_serving_report_rows'],
                limitations='Context report uses 30 cases / 23 reference queries. Serving screen uses three direct requests per prompt/runtime, not the full agent; no general runtime ranking.')


def gpu_history(root):
    root = Path(root)
    manifest = json.loads((root / 'tutorial/serving_foundation.json').read_text())
    rows = []
    for path in manifest['gpu_summaries']:
        result = json.loads(checked(root, path))
        archive = 'baseline/mvp3/' + result['source']['archive']
        raw = checked(root, archive)
        if hashlib.sha256(raw).hexdigest() != result['source']['archived_sha256']:
            raise ValueError('Imported engine notebook checksum mismatch')
        rows.append(dict(runtime=result['runtime'], clients=result['configuration']['concurrency'],
                         serving=result['serving'], quality=result['quality'],
                         evidence='historical notebook summary — rounded; pairing unverified', source=path))
    return rows


def history_table(rows):
    lines = ['| Runtime | Clients | TTFT p50 s ≈ | Request E2E p50 s ≈ | Aggregate output tok/s ≈ | Application result match | Evidence |',
             '|---|---|---|---|---|---|---|']
    for r in rows:
        s, q = r['serving'], r['quality']
        lines.append(f'| {r["runtime"]} | {r["clients"]} | {s.get("ttft_p50_seconds")} | '
                     f'{s.get("e2e_p50_seconds")} | {s.get("output_tokens_per_second")} | '
                     f'{q["result_equivalent"]}/{q["reference_scored"]} | {r["evidence"]} |')
    return '\n'.join(lines)
