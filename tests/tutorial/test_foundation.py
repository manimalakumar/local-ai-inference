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
    assert 'ExecuTorch' not in text and 'DLSIM' not in text
    assert 'llama.cpp' in text and 'Ollama' in text and 'SGLang' in text
    namespace = {}
    for cell in document['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'foundation-cell', 'exec'), namespace)
    assert namespace['example'].mode == 'replay'


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


def test_all_learner_chapters_begin_with_learning_requirements_and_outputs():
    from clinical_trial_mvp.tutorial.workflow_notebook import notebook_document as chapter
    for runtime in ('vllm', 'nim'):
        for model in ('qwen', 'nemotron'):
            source = ''.join(chapter(runtime, model)['cells'][0]['source'])
            assert all(word in source for word in ('**Learn:', '**Requirements:', '**Expected outputs:'))
