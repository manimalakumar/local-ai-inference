"""Protect the distinction between compatibility evidence and measured inference."""
import json

import pytest

from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.nemotron_edge_notebook import RECORD, recorded_probe, notebook_document


def test_probe_is_pinned_to_dell_model_and_does_not_claim_inference():
    result = recorded_probe(ROOT)
    baseline = json.loads((ROOT / 'experiment-tracker/mvp4/nemotron/20261107T031845_031809Z.json').read_text())
    assert result['model_revision'] == baseline['checkpoint']['base_revision']
    assert result['status'] == 'blocked_compatibility'
    assert all(not attempt['accepted'] for attempt in result['attempts'])
    assert 'qwen3_4b' in result['installed_model_types']
    assert result['config']['model_type'] == 'nemotron_h'
    for field in ('weights_downloaded', 'graph_capture_attempted', 'pte_produced', 'native_inference_run'):
        assert result[field] is False
    for field in ('sql_correctness', 'ttft_seconds', 'end_to_end_seconds', 'peak_memory_bytes'):
        assert result[field] is None


def test_changed_probe_evidence_is_rejected(tmp_path):
    path = tmp_path / RECORD
    path.parent.mkdir(parents=True)
    raw = json.loads((ROOT / RECORD).read_text())
    raw['native_inference_run'] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='checksum'):
        recorded_probe(tmp_path)


def test_run_all_reads_evidence_without_starting_processes(monkeypatch):
    monkeypatch.setenv('LOCAL_AI_LAB_MODE','replay')
    import subprocess
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('launched process'))
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('launched process'))
    monkeypatch.chdir(ROOT / 'notebooks')
    namespace = {}
    for cell in notebook_document()['cells']:
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), 'nemotron-edge-cell', 'exec'), namespace)
    assert namespace['probe']['native_inference_run'] is False
    assert namespace['MODE']=='replay'
    assert namespace['probe']['status']=='blocked_compatibility'
