"""Recorded data integrity, explicit modes and visible learner mechanisms."""
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from clinical_trial_mvp.tutorial import workflow as w
from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.replay import RecordedStudy, LABEL
from clinical_trial_mvp.tutorial.teaching import server_example, request_example, response_example, interpret


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'scripts/tutorial/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('runtime,model', [('vllm','qwen'),('vllm','nemotron'),('nim','qwen'),('nim','nemotron')])
def test_replay_never_launches_downloads_or_calls_network(monkeypatch, runtime, model):
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **k: pytest.fail('replay subprocess'))
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: pytest.fail('replay launch'))
    monkeypatch.setattr(w.nim_study, 'get_json', lambda *a, **k: pytest.fail('replay HTTP'))
    study = w.Study(runtime, model, root=ROOT, mode='replay')
    assert study.mode == 'replay'
    study.prepare()
    study.smoke()
    for name in study.configs:
        result = study.run(name)
        if name == 'awq_stretch':
            assert result['status'] == 'unavailable' and not result.get('serving')
        else:
            assert result['_recorded']['label'] == LABEL and result['status'] == 'completed'
    assert LABEL in study.report()
    assert (LABEL in study.cache_report()) if runtime == 'vllm' else ('No controlled' in study.cache_report())
    study.close()


def test_checksum_mismatch_does_not_silently_replay_other_record(tmp_path):
    archive = RecordedStudy(ROOT)
    rel = archive.index['studies']['vllm-qwen']['fp16_cache_on']
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    archive.root = tmp_path
    with pytest.raises(ValueError, match='checksum mismatch'):
        archive.result('vllm', 'qwen', 'fp16_cache_on')


def test_live_failure_never_uses_recorded_fallback(monkeypatch, tmp_path):
    config = w.template('nim', 'qwen')
    config['mode'] = 'live'
    study = w.Study('nim', 'qwen', root=tmp_path, config=config)
    study.prepared = True
    monkeypatch.setattr(study, '_nim_start', lambda: (_ for _ in ()).throw(RuntimeError('live startup failed')))
    monkeypatch.setattr(study, 'recorded', lambda *a: pytest.fail('silent replay fallback'))
    with pytest.raises(RuntimeError, match='live startup failed'):
        study.run('smoke')
    assert study.results['smoke']['status'] == 'failed'
    assert study.results['smoke']['session_id'] == study.session_id


@pytest.mark.parametrize('runtime,model,name', [('vllm','qwen','fp16_cache_off'),('nim','qwen','baseline'),('nim','nemotron','baseline')])
def test_three_visible_paths_match_saved_contract(runtime, model, name):
    study = w.Study(runtime, model, root=ROOT, mode='replay')
    study.run(name)
    server = server_example(study, name)
    assert server['origin'] == LABEL and server['dtype'] == 'float16'
    if runtime == 'nim':
        assert server['scheduler_max_num_seqs'] == 4
        assert 'reconstructed' in server['command_origin']
    request = request_example(study, name)
    assert request['method'] == 'POST'
    assert request['json']['max_tokens'] == 1024
    assert 'Question:' in request['json']['messages'][-1]['content']
    assert ('response_format' in request['json']) == (runtime == 'vllm' or model == 'qwen')
    response = response_example(study, name)
    assert response['attempts'] and response['reference_sql']
    assert response['semantic_review'].startswith('unreviewed')
    assert all(word in interpret(study, name) for word in ('Prediction:', 'Observation:', 'Explanation:', 'Limitation:', 'Decision:'))


def test_guided_cache_interpretation_uses_paired_phase_not_warmed_summary():
    study = w.Study('vllm', 'qwen', root=ROOT, mode='replay')
    study.run('fp16_cache_on')
    prose = interpret(study, 'fp16_cache_on')
    assert '2.019 → 0.083' in prose and '6.038 → 4.039' in prose
    assert '10/20 cold' in prose and '10/20 warm' in prose
    assert 'Four distinct questions' in prose


def test_two_figure_numeric_inputs_are_reproducible():
    module = load_script('render_findings')
    data = module.figure_data()
    assert data == json.loads((ROOT / 'tutorial/figures/figure_data.json').read_text())
    assert len(data['caching']) == 4 and len(data['concurrency']) == 3
    assert data['distinct_questions'] == 4
    assert all(r['scheduler_capacity'] == 4 for r in data['concurrency'])


def test_review_queue_does_not_invent_human_judgments():
    queue = json.loads((ROOT / 'tutorial/sql_review_queue.json').read_text())
    assert queue['reviewed_cases'] == 0 and len(queue['cases']) == 4
    assert all(c['reviewer'] is None and c['correct'] is None for c in queue['cases'])
    from clinical_trial_mvp.tutorial.reviews import validate_reviews
    validate_reviews(json.loads((ROOT / 'tutorial/semantic_reviews.json').read_text())['reviews'])


def test_front_door_has_no_misplaced_hybrid_or_nim_cleanup_instructions():
    from clinical_trial_mvp.tutorial.workflow_notebook import notebook_document
    text = '\n'.join(''.join(c['source']) for c in notebook_document('vllm', 'qwen')['cells'])
    assert 'FP32 SSM' not in text
    assert 'pausing between successful live NIM' not in text
    assert 'MODE = "live"' in text and 'request_example' in text and 'response_example' in text


def test_clean_export_contains_only_committed_source_not_worktree_caches(tmp_path):
    module = load_script('export_tracked')
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'README.md').write_text('source')
    subprocess.run(['git','add','README.md'],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=repo,check=True)
    (repo / '__pycache__').mkdir()
    (repo / '__pycache__/bad.pyc').write_bytes(b'not source')
    output = tmp_path / 'export.zip'
    module.export(repo, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ['local-ai-inference/', 'local-ai-inference/README.md']
    assert module.forbidden('.git/config') and module.forbidden('x/__pycache__/bad.pyc')
