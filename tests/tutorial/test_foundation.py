"""The serving foundation reuses frozen evidence and adds no live experiment."""
import hashlib
import json
import sys
import types

import pytest

from clinical_trial_mvp.tutorial import foundation
from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.replay import RecordedStudy
from clinical_trial_mvp.tutorial.serving_notebook import notebook_document
from clinical_trial_mvp.tutorial.token_analysis import analyze, table


def test_foundation_manifest_preserves_frozen_inputs_and_historical_sources():
    manifest = json.loads((ROOT / 'tutorial/serving_foundation.json').read_text())
    for path in manifest['files_sha256']:
        assert foundation.checked(ROOT, path)
    database_hash = manifest['files_sha256']['data/db/clinical_trials.duckdb']
    recorded = RecordedStudy(ROOT).result('vllm', 'qwen', 'fp16_cache_off')
    assert database_hash == recorded['database_sha256']
    assert manifest['database']['synthetic'] is True


def test_inventory_is_read_only_and_contexts_reuse_existing_cache():
    path = ROOT / 'data/db/clinical_trials.duckdb'
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    data = foundation.inventory(ROOT)
    assert len(data['tables']) == 10 and data['database_sha256'] == before
    context = foundation.contexts(ROOT)
    assert context['case_ids'] == ['eval_001', 'eval_002', 'eval_005', 'eval_007']
    assert len(context['rows']) == 4
    assert all(row['full_prompt'] and row['retrieved_prompt'] for row in context['rows'])
    assert 'not token counts' in context['note']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_imported_engine_history_stays_summary_evidence():
    rows = foundation.gpu_history(ROOT)
    assert len(rows) == 6
    assert {r['runtime'] for r in rows} == {'vllm', 'sglang'}
    assert all('pairing unverified' in r['evidence'] for r in rows)
    assert 'historical notebook summary' in foundation.history_table(rows)
    history = foundation.mac_history(ROOT)
    assert history['context'][0]['result_match'] == '10/23'
    assert history['serving'][0]['request_e2e_p50_s'] == 1.393


def test_all_foundation_cells_run_offline_without_new_runtime(monkeypatch):
    monkeypatch.setenv('LOCAL_AI_LAB_MODE', 'replay')
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('foundation launched a process'))
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: pytest.fail('foundation started server'))
    display = types.ModuleType('IPython.display')
    display.JSON = lambda value: value
    display.Markdown = lambda value: value
    display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'IPython.display', display)
    monkeypatch.chdir(ROOT / 'notebooks')
    document = notebook_document()
    text = '\n'.join(''.join(c['source']) for c in document['cells'])
    body = '\n'.join(''.join(c['source']) for c in document['cells'][1:])
    assert 'ExecuTorch' not in body and 'DLSIM' not in text  # Navigation may link to optional chapter 05.
    assert 'llama.cpp' in text and 'Ollama' in text
    code = '\n'.join(''.join(c['source']) for c in document['cells'] if c['cell_type']=='code')
    assert 'gpu_history(' not in code and 'gpu_rows(' not in code and "Study('vllm'" not in code
    namespace = {}
    for cell in document['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'foundation-cell', 'exec'), namespace)
    assert namespace['study'].mode == 'replay'
    assert namespace['answer']['origin'] == 'Nemotron 3 Nano 4B Q4_K_M / llama.cpp'


def test_local_worked_answer_uses_original_gguf_record_and_matching_database():
    example=foundation.local_answer_example(ROOT)
    assert example['model_sha256']=='be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2'
    record=example['application_record']
    assert record['id']=='eval_001'
    assert record['attempts'][-1]['execution_success'] is True
    assert record['scores']['result_equivalence'] is True
    rows=record['attempts'][-1]['query_result']['rows']
    assert sum(r['enrolled_subjects'] for r in rows)==170
    assert 'Structured answer' in foundation.structured_answer_markdown(example)


def test_token_audit_uses_paired_targets_and_keeps_missing_counts_unknown():
    replay = RecordedStudy(ROOT)
    results = {name: replay.result('vllm', 'qwen', name)
               for name in ('fp16_cache_on', 'awq_w4a16_cache_on')}
    audit = analyze(results)
    saved = json.loads((ROOT / 'tutorial/output_token_analysis.json').read_text())
    saved.pop('label')
    assert audit == saved
    assert len(audit['per_case']) == 16
    assert [r['output_tokens_mean'] for r in audit['rows']] == [152, 150.75, 151.5, 151.5]
    assert all(r['requests'] == 20 for r in audit['rows'])
    assert '152.00' in table(audit)
    unknown = analyze({'fp16_cache_on': {'cache_study': {'records': [
        {'phase': 'cold', 'case_id': 'fixture', 'output_tokens': None}]}}})
    assert unknown['rows'][0]['output_tokens_mean'] is None
    assert 'unavailable' in table(unknown)


def test_all_learner_chapters_show_the_shared_workflow_and_current_location():
    from clinical_trial_mvp.tutorial.workflow_notebook import notebook_document as chapter, FILENAMES
    from clinical_trial_mvp.tutorial.edge_notebook import notebook_document as edge, FILENAME as EDGE
    from clinical_trial_mvp.tutorial.nemotron_edge_notebook import notebook_document as nemotron_edge, FILENAME as NEMOTRON_EDGE
    from clinical_trial_mvp.tutorial.serving_notebook import FILENAME as FOUNDATION
    from clinical_trial_mvp.tutorial.navigation import CHAPTERS, workflow_table
    documents = {FOUNDATION: notebook_document(), EDGE: edge(), NEMOTRON_EDGE: nemotron_edge()}
    documents.update({name: chapter(*key) for key, name in FILENAMES.items()})
    assert workflow_table() in (ROOT / 'README.md').read_text()
    for filename, document in documents.items():
        source = ''.join(document['cells'][0]['source'])
        assert source.count('**You are here →') == 1
        assert source.count('(current)**') == 1
        assert workflow_table(filename, prefix='') in source
        row = next(row for row in CHAPTERS if row[1] == filename)
        assert f'**You are here → {row[2]}.**' in source
        assert document['cells'][0]['cell_type'] == 'markdown'
        assert '../SETUP.md' in source
