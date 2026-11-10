"""Historical CPU replay and review coverage must not fabricate new evidence."""
import hashlib
import importlib.util
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial import edge_notebook


def test_cpu_portability_replays_exact_hardware_record_and_preserves_unknowns(monkeypatch):
    monkeypatch.setenv('LOCAL_AI_LAB_MODE','replay')
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('unexpected new experiment'))
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: pytest.fail('unexpected edge process'))
    before = set(sys.modules)
    display = types.ModuleType('IPython.display')
    display.JSON = lambda value: value
    display.Markdown = lambda value: value
    display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'IPython.display', display)
    monkeypatch.chdir(ROOT / 'notebooks')
    namespace = {}
    for cell in edge_notebook.notebook_document()['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'edge-cell', 'exec'), namespace)
    record = namespace['edge']
    assert record['model_name'] == 'Qwen/Qwen3-4B'
    assert record['sql_smoke']['matches_dell_base_snapshot'] and record['sql_smoke']['result_equivalence']
    assert record['operator_fallback_count'] is None and record['quality'] is None
    assert record['single_request_seconds'] == 0.876 and 'INT4' in record['dtype']
    history = edge_notebook.recorded_edge(ROOT)
    assert history['model_name'] == 'HuggingFaceTB/SmolLM2-135M' and history['single_request_seconds'] == 0.199
    assert not {'torch','executorch'} & (set(sys.modules)-before)


def test_edge_corrupt_record_is_rejected(tmp_path):
    p=tmp_path/edge_notebook.RECORD
    p.parent.mkdir(parents=True)
    p.write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        edge_notebook.recorded_edge(tmp_path)


def test_review_packet_covers_all_distinct_sources_without_entering_judgments():
    path=ROOT/'scripts/tutorial/build_sql_review_packet.py'
    spec=importlib.util.spec_from_file_location('review_packet',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    packet=module.build()
    assert packet == json.loads((ROOT/'tutorial/sql_review_distinct.json').read_text())
    assert packet['distinct_outputs'] == 61 and len(packet['no_sql_outputs']) == 5
    assert len(packet['case_ids']) == 4
    from clinical_trial_mvp.tutorial.reviews import validate_reviews
    allowed = {(e['case_id'], e['sql_sha256']) for e in packet['outputs']}
    reviews = validate_reviews(json.loads((ROOT/'tutorial/semantic_reviews.json').read_text())['reviews'], allowed=allowed)
    assert packet['reviewed_outputs'] == len(reviews)
    scopes=set()
    phases=set()
    for e in packet['outputs']:
        assert hashlib.sha256(e['sql'].encode()).hexdigest() == e['sql_sha256']
        if (e['case_id'], e['sql_sha256']) not in reviews:
            assert all(e[k] is None for k in ('reviewer','reviewed_at','correct','rationale'))
        for occurrence in e['occurrences']:
            scopes.add(occurrence['study'])
            if 'phase' in occurrence:
                phases.add(occurrence['phase'])
            record=json.loads((ROOT/occurrence['source']).read_text())
            for key in occurrence['pointer'].strip('/').split('/'):
                record=record[int(key)] if isinstance(record,list) else record[key]
            assert record['case_id'] == e['case_id']
    assert scopes == {'vllm-qwen','vllm-nemotron','nim-qwen','nim-nemotron'}
    assert phases == {'cold','warm','cache_off'}
    validate_reviews(json.loads((ROOT/'tutorial/semantic_reviews.json').read_text())['reviews'], allowed=allowed)


def test_qwen_replay_rejects_corrupt_new_evidence_instead_of_using_smollm2(tmp_path):
    path=tmp_path/edge_notebook.QWEN_RECORD
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError,match='checksum mismatch'):
        edge_notebook.recorded_qwen_edge(tmp_path)


def test_qwen_export_recipe_keeps_4b_architecture_and_qk_normalization():
    path=ROOT/'scripts/tutorial/run_qwen_edge.py'
    spec=importlib.util.spec_from_file_location('qwen_export_recipe',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config=dict(model_type='qwen3',num_hidden_layers=36,hidden_size=2560,num_attention_heads=32,
                num_key_value_heads=8,head_dim=128,vocab_size=151936,intermediate_size=9728,
                rms_norm_eps=1e-6,rope_theta=1000000,bos_token_id=151643,eos_token_id=151645)
    params=module.params(config)
    assert params['head_dim']==128 and params['n_heads']==32
    assert params['dim']!=params['head_dim']*params['n_heads']  # Qwen head_dim must not be inferred from dim/heads.
    assert params['use_qk_norm'] and params['qk_norm_before_rope'] and params['use_hf_rope']
    with pytest.raises(ValueError,match='4B'):
        module.params({**config,'hidden_size':1024})
    with pytest.raises(ValueError,match='unquantized'):
        module.params({**config,'quantization_config':{'quant_method':'awq'}})
    lineage=json.loads((ROOT/'experiment-tracker/mvp4/runs/20261106T224700_600112Z.json').read_text())['checkpoint_pair']
    assert module.REVISION in lineage['base_revision_attestation']
