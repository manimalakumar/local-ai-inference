"""Exercise the fresh baseline boundary without pretending to run a GPU."""
import importlib.util
import json
from pathlib import Path

import pytest

from clinical_trial_mvp.tutorial import foundation_study as module
from clinical_trial_mvp.tutorial.harness import ROOT


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCAL_AI_LAB_MODE', raising=False)
    cfg = module.template()
    cfg['mode'] = 'live'
    path = tmp_path / 'server.json'
    metadata = dict(runtime='vllm', model='clinical-trial-mvp3', engine_version='test fixture',
        hardware='TEST FIXTURE — not hardware evidence', model_sha256='a'*64,
        evidence='hardware', launch_settings=dict(context=8192, dtype='float16',
        prefix_cache=False, quantization=None))
    path.write_text(json.dumps(metadata))
    cfg['servers']['qwen']['provenance'] = str(path)
    monkeypatch.setattr(module.foundation, 'inventory', lambda root: {})
    monkeypatch.setattr('clinical_trial_mvp.serving.benchmark.prepare', lambda *a: None)
    return tmp_path, cfg


def test_replay_needs_neither_server_metadata_nor_requests(tmp_path, monkeypatch):
    monkeypatch.delenv('LOCAL_AI_LAB_MODE', raising=False)
    monkeypatch.setattr(module, 'serve_and_benchmark', lambda *a, **k: pytest.fail('model call'))
    cfg = module.local_template()
    cfg['mode'] = 'replay'
    study = module.FoundationStudy(tmp_path, cfg)
    assert study.mode == 'replay'
    assert not tmp_path.joinpath('experiment-tracker').exists()
    with pytest.raises(ValueError, match='Replay'):
        study.context_rows()


def test_live_missing_metadata_fails_before_inference(configured, monkeypatch):
    root, cfg = configured
    Path(cfg['servers']['qwen']['provenance']).unlink()
    monkeypatch.setattr(module, 'serve_and_benchmark', lambda *a, **k: pytest.fail('model call'))
    with pytest.raises(ValueError, match='NEW launch metadata'):
        module.FoundationStudy(root, cfg)


def test_context_concurrency_and_answer_use_new_rows_only(configured, monkeypatch):
    root, cfg = configured
    calls = []
    def fake(spec, **kw):
        calls.append((spec, kw))
        return dict(status='completed', evidence='hardware', run_id='TEST FIXTURE',
            experiment=spec['experiment'], configuration=spec,
            records={'quality':[dict(case_id='eval_001', plan_prompt='new fixture prompt',
                                     attempts=[dict(sql='SELECT 1')])]}, quality={}, serving={})
    monkeypatch.setattr(module, 'serve_and_benchmark', fake)
    study = module.FoundationStudy(root, cfg)
    rows = study.context_rows()
    assert [r['configuration']['profile'] for r in rows] == ['full', 'retrieved']
    gpu = study.gpu_rows()
    assert [r['configuration']['concurrency'] for r in gpu] == [1, 2, 4]
    assert len(calls) == 5
    answer = study.answer()
    assert answer['application_record']['plan_prompt'] == 'new fixture prompt'
    assert len(calls) == 5  # The worked example reuses this session's result.
    assert study.local_rows() == [] and 'Not selected' in study.table([])
    assert 'Fresh measurements' in study.table(rows)
    assert len(list(root.glob('experiment-tracker/tutorial/foundation/*/*.json'))) == 5
    assert all(kw['execute'] is True for _, kw in calls)


def test_failed_live_row_is_saved_and_never_replayed(configured, monkeypatch):
    root, cfg = configured
    monkeypatch.setattr(module, 'serve_and_benchmark', lambda spec, **kw:
        dict(status='failed', evidence='none', reason='fixture server unavailable', configuration=spec))
    study = module.FoundationStudy(root, cfg)
    with pytest.raises(RuntimeError, match='Fresh baseline failed'):
        study.context_rows()
    saved = list(root.glob('experiment-tracker/tutorial/foundation/*/*.json'))
    assert len(saved) == 1 and json.loads(saved[0].read_text())['status'] == 'failed'
    with pytest.raises(RuntimeError, match='Previous row failed'):
        study.answer()


def test_plan_needs_no_metadata_and_does_not_measure(configured, monkeypatch):
    root, cfg = configured
    cfg['mode'] = 'plan'
    Path(cfg['servers']['qwen']['provenance']).unlink()
    monkeypatch.setattr(module, 'serve_and_benchmark', lambda spec, **kw:
        dict(status='plan_only', evidence='none', configuration=spec, experiment=spec['experiment'])
        if kw['execute'] is False else pytest.fail('Plan enabled inference'))
    study = module.FoundationStudy(root, cfg)
    assert all(r['status'] == 'plan_only' for r in study.context_rows())
    assert 'Plan only' in study.table(study.gpu_rows())
    assert not list(root.glob('experiment-tracker/tutorial/foundation/*/*.json'))


@pytest.mark.parametrize('change', [dict(concurrency=[0]), dict(local_servers=['qwen']),
    dict(context_server='unknown'), dict(case_ids=[]), dict(repeats=True)])
def test_invalid_config_is_rejected_before_a_request(configured, change):
    root, cfg = configured
    cfg.update(change)
    with pytest.raises(ValueError):
        module.FoundationStudy(root, cfg)


def test_environment_can_force_safe_replay(configured, monkeypatch):
    root, cfg = configured
    Path(cfg['servers']['qwen']['provenance']).unlink()
    monkeypatch.setenv('LOCAL_AI_LAB_MODE', 'replay')
    assert module.FoundationStudy(root, cfg).mode == 'replay'


def test_full_baseline_quality_keeps_policy_cases_while_serving_filters_them(monkeypatch):
    from clinical_trial_mvp.tutorial import harness
    from clinical_trial_mvp.tutorial.memory import PeakMemory
    counts={}
    def quality(client, settings, cases, contexts, workload, emit):
        counts['quality']=len(cases)
        assert workload.concurrency==1
        return dict(cases=len(cases))
    def serving(client, cases, prompts, workload, emit):
        counts['serving']=len(cases)
        emit(dict(prompt_tokens=1))
        return {}
    monkeypatch.setattr(harness,'run_quality',quality)
    monkeypatch.setattr(harness,'run_serving',serving)
    spec=dict(module.template()['servers']['qwen'], experiment='TEST FIXTURE',hardware='TEST',
              profile='full',concurrency=4)
    server=dict(runtime='vllm',model=spec['model'],engine_version='TEST',hardware='TEST',
                model_sha256='a'*64, evidence='test',launch_settings=dict(context=8192,
                dtype='float16',prefix_cache=False,quantization=None))
    ids=[c['id'] for c in json.loads((ROOT/'evals/evaluations.json').read_text())]
    result=harness.serve_and_benchmark(spec,execute=True,root=ROOT,case_ids=ids,client=object(),
        server=server,memory=PeakMemory(reader=lambda: 0),include_policy_cases=True)
    assert result['status']=='completed'
    assert result['evidence']=='test_or_unverified'
    assert counts==dict(quality=30,serving=26)
    assert len(result['quality_inputs']['case_ids'])==30
    assert len(result['inputs']['case_ids'])==26


def test_setup_helper_does_not_overwrite_config_or_require_weights_for_plan(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location('configure_foundation', ROOT / 'scripts/tutorial/configure_foundation.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    path = tmp_path / 'baseline.local.json'
    args = ['--mode','plan','--config',str(path),'--snapshot',str(tmp_path/'no-weights'),
            '--server-python','/test/gpu/python']
    assert helper.main(args) == 0
    config = json.loads(path.read_text())
    assert config['mode'] == 'plan'
    import shlex
    command=next(line for line in capsys.readouterr().out.splitlines() if line.startswith('/test/gpu/python'))
    argv=shlex.split(command)
    assert argv[1].endswith('/launch_foundation.py')
    assert argv[2:4]==['--runtime','vllm']
    with pytest.raises(SystemExit):
        helper.main(args)


@pytest.mark.parametrize('occupied,existing', [(True,False), (False,True), (False,False)])
def test_launch_guard_does_not_attach_to_an_old_server_or_overwrite_metadata(tmp_path, monkeypatch, occupied, existing):
    spec = importlib.util.spec_from_file_location('launch_foundation', ROOT / 'scripts/tutorial/launch_foundation.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    from contextlib import nullcontext
    def connect(*a, **k):
        if occupied:
            return nullcontext()
        raise ConnectionRefusedError
    monkeypatch.setattr(helper.socket, 'create_connection', connect)
    metadata = tmp_path / 'server.json'
    if existing:
        metadata.write_text('keep')
    calls=[]
    args=['--runtime','vllm','--snapshot',str(tmp_path/'weights'),'--metadata',str(metadata)]
    if occupied or existing:
        with pytest.raises(SystemExit):
            helper.main(args, launcher=lambda *a, **k: pytest.fail('launched through guard'))
    else:
        assert helper.main(args, launcher=lambda *a, **k: calls.append((a,k)) or 0) == 0
        assert len(calls)==1 and calls[0][1]['prefix_cache'] is False
