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
    return dict(origin='Loading results.',
                context=manifest['context_report_rows'], serving=manifest['mac_serving_report_rows'],
                limitations='Context report uses 30 cases / 23 reference queries. Serving screen uses three direct requests per prompt/runtime, not the full agent; no general runtime ranking.')


def context_results_table(rows):
    """Present the existing schema comparison without changing its measurements."""
    model_names = {'Nemotron Q4': 'Nemotron 3 Nano 4B (Q4_K_M)',
                   'Qwen Q4': 'Qwen3-4B (Q4_K_M)'}
    lines = ['| Model | Schema context | Median prompt tokens | Median application latency (s) | Reference-result matches |',
             '|---|---|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {model_names.get(row["model"], row["model"])} | {row["schema"]} | '
                     f'{row["prompt_tokens_per_invocation_p50"]:g} | '
                     f'{row["application_e2e_p50_s"]:.3f} | {row["result_match"]} |')
    return '\n'.join(lines)


def local_serving_results_table(rows):
    """Show direct-serving timing separately from application-quality scoring."""
    lines = ['| Runtime | Schema context | Measured requests | Median first-token time (s) | Median request completion (s) | Output tokens |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["runtime"]} | {row["schema"]} | {row["requests"]} | '
                     f'{row["ttft_p50_s"]:.3f} | {row["request_e2e_p50_s"]:.3f} | {row["output_tokens"]} |')
    return '\n'.join(lines)


def local_answer_example(root):
    """Load an actual Nemotron GGUF/llama.cpp answer from the MVP3 source tree."""
    root = Path(root)
    folder = root / 'baseline/mvp1/experiment-tracker/local-answer-example'
    name = 'eval_20260926T222549_800940Z'
    raw = (folder / (name + '.jsonl')).read_bytes()
    manifest = json.loads((folder / (name + '_manifest.json')).read_text())
    if (hashlib.sha256(raw).hexdigest() != manifest['detail']['sha256'] or
        manifest['detail']['sha256'] != 'a1b7d1edc3a85afc16cd33755b81b0bf56fc73ef5728bf4f4ec6b107af7139cb' or
        manifest['runtime'] != 'llama_cpp' or manifest['quantization'] != 'Q4_K_M' or
        manifest['model_sha256'] != 'be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2' or
        hashlib.sha256((root / 'data/db/clinical_trials.duckdb').read_bytes()).hexdigest() != manifest['database_sha256']):
        raise ValueError('Local answer source or database fingerprint mismatch')
    record = next(json.loads(line) for line in raw.splitlines() if json.loads(line)['id'] == 'eval_001')
    return dict(origin='Nemotron 3 Nano 4B Q4_K_M / llama.cpp',
                model=manifest['model'], model_sha256=manifest['model_sha256'],
                database_sha256=manifest['database_sha256'], application_record=record)


def structured_answer_markdown(example):
    """Display a question, generated SQL and aggregate answer without transport JSON."""
    record = example.get('application_record')
    if not record:
        return '**Structured answer:** run the full-schema baseline to view its generated answer.'
    attempt = (record.get('attempts') or [{}])[-1]
    result = attempt.get('query_result') or {}
    columns, rows = result.get('columns', []), result.get('rows', [])
    lines = ['### Question', record.get('question', record.get('case_id', '')), '',
             '### Generated SQL', '```sql', attempt.get('sql') or '(No SQL generated)', '```', '',
             '### Structured answer']
    if columns:
        lines += ['| ' + ' | '.join(c.replace('_', ' ').capitalize() for c in columns) + ' |',
                  '|' + '|'.join('---' for _ in columns) + '|']
        for row in rows:
            values = [row.get(c) for c in columns] if isinstance(row, dict) else row
            lines.append('| ' + ' | '.join(str(v).replace('|', '\\|') for v in values) + ' |')
    else:
        lines.append('No result rows available for this attempt.')
    scores = record.get('scores') or {}
    def status(value):
        return 'Passed' if value is True else 'Failed' if value is False else 'Unavailable'
    lines += ['', '### Quality check', '| Check | Result |', '|---|---|',
              f'| SQL executed | {status(attempt.get("execution_success"))} |',
              f'| Reference-result match | {status(scores.get("result_equivalence"))} |']
    return '\n'.join(lines)


def baseline_results_summary(study, history, answer):
    """Close chapter 00 with its measured results and a brief interpretation."""
    if study.mode == 'plan':
        return '**Baseline plan ready.** Run the configured local comparisons in Live mode to collect results.'
    lines = ['| Baseline | Schema context | Reference-result matches | Median application latency (s) | Median first-token time (s) | Median request completion (s) |',
             '|---|---|---:|---:|---:|---:|']
    def number(value):
        return '—' if value is None else f'{value:.3f}'
    if study.mode == 'replay':
        names = {'Nemotron Q4': 'Nemotron 3 Nano 4B (Q4_K_M)', 'Qwen Q4': 'Qwen3-4B (Q4_K_M)'}
        for row in history['context']:
            lines.append(f'| {names.get(row["model"], row["model"])} | {row["schema"]} | '
                         f'{row["result_match"]} | {number(row["application_e2e_p50_s"])} | — | — |')
        for row in history['serving']:
            lines.append(f'| {row["runtime"]} / Nemotron 3 Nano 4B (Q4_K_M) | {row["schema"]} | — | — | '
                         f'{number(row["ttft_p50_s"])} | {number(row["request_e2e_p50_s"])} |')
        interpretation = ('Retrieved schema reduced median application latency for both models. '
            'Reference-result matches decreased for Nemotron and increased slightly for Qwen. '
            'llama.cpp and Ollama produced matching outputs with similar request timings for the same context.')
    else:
        for key, row in study.results.items():
            quality, serving = row.get('quality') or {}, row.get('serving') or {}
            matches = (f'{quality["result_equivalent"]}/{quality["reference_scored"]}'
                       if 'reference_scored' in quality else '—')
            lines.append(f'| {key} | {row["configuration"]["profile"]} | {matches} | '
                         f'{number(quality.get("latency_p50_seconds"))} | '
                         f'{number(serving.get("ttft_p50_seconds"))} | {number(serving.get("e2e_p50_seconds"))} |')
        interpretation = ('The table brings together the context and local-serving results from this run. '
            'Read reference-result matches alongside response times to assess each configuration.')
    record = answer.get('application_record') or {}
    attempt = (record.get('attempts') or [{}])[-1]
    matched = (record.get('scores') or {}).get('result_equivalence')
    if record:
        check = 'Matched (one question)' if matched is True else 'Not matched' if matched is False else '—'
        context = (record.get('schema_selection') or {}).get('mode', '—')
        lines.append(f'| Structured-answer example | {context} | {check} | '
                     f'{number(record.get("latency_seconds"))} | — | — |')
    if attempt.get('execution_success') is True and matched is True:
        interpretation += ' The worked structured answer executed successfully and matched its reference result.'
    elif record:
        interpretation += ' The worked answer\'s execution and reference-result checks are shown in section 4.'
    lines += ['', 'Application latency includes planning and SQL checks; request timing covers the model-generation call.',
              'A dash indicates a metric not measured in that comparison.', '', '**Result summary**', '', interpretation]
    return '\n'.join(lines)


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
