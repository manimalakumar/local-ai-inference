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


def test_display_hides_workspace_paths_without_changing_runtime_configuration(monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace
    import IPython.display
    from clinical_trial_mvp.tutorial import edge_workflow
    path = '/Users/example/Documents/Codex/workspace/.venv-edge/bin/python'
    config = {'edge_python': path}
    probe = {'status': 'blocked_compatibility', 'native_inference_run': False,
             'command': [path, '/Users/example/Documents/Codex/workspace/probe.py'],
             'ttft_seconds': None}
    original = deepcopy(probe)
    fake = SimpleNamespace(mode='live', session_id='display-check', results={}, config=config,
                           preflight=lambda: {'runtime_python': path}, probe=lambda: probe)
    monkeypatch.setattr(edge_workflow, 'EdgeStudy', lambda *a, **k: fake)
    displayed = []
    monkeypatch.setattr(IPython.display, 'display', lambda item: displayed.append(item.data))
    monkeypatch.chdir(ROOT)
    namespace = {}
    selected = {'code-edge-configure', 'code-edge-preflight-check', 'code-edge-probe'}
    for cell in notebook_document()['cells']:
        if cell.get('id') in selected:
            exec(compile(''.join(cell['source']), 'nemotron-display', 'exec'), namespace)
    assert len(displayed) == 3
    assert 'Codex' not in json.dumps(displayed)
    assert 'blocked_compatibility' in json.dumps(displayed)
    assert fake.config['edge_python'] == path
    assert probe == original  # Actual executable and raw evidence retain their full paths.
