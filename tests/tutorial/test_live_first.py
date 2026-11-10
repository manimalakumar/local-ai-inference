"""Live session authority, evidence isolation and explicit edge execution."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clinical_trial_mvp.tutorial import workflow as w
from clinical_trial_mvp.tutorial.edge_workflow import EdgeStudy
from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.teaching import interpret, request_example, response_example, server_example


def live(tmp_path, model='qwen', runtime='vllm'):
    obj=w.Study(runtime,model,root=tmp_path,mode='live')
    obj.prepared=True
    obj.smoke_result={'status':'completed','smoke_passed':True}
    return obj


def test_explicit_mode_overrides_stale_local_mode_and_legacy_execute(tmp_path):
    path=tmp_path/'tutorial/config/vllm-qwen.local.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(mode='replay',execute=False)))
    study=w.Study('vllm','qwen',root=tmp_path,mode='live')
    assert study.live and study.results=={} and 'execute' not in study.config
    with pytest.raises(ValueError,match='Contradictory'):
        w.Study('vllm','qwen',root=tmp_path,config=dict(w.template('vllm','qwen'),mode='live',execute=False))


def test_live_rerun_calls_inference_and_saves_unique_current_evidence(tmp_path,monkeypatch):
    study=live(tmp_path);calls=[]
    @contextmanager
    def server(*args):yield
    monkeypatch.setattr(study,'_native_server',server)
    monkeypatch.setattr(study,'recorded',lambda *a:pytest.fail('read archived measurement'))
    def inference(spec,*args,execute=False,**kwargs):
        calls.append(execute)
        return dict(configuration=spec,experiment=spec['experiment'],status='completed',
                    evidence='hardware',quality={'result_equivalent':1,'reference_scored':4})
    monkeypatch.setattr(w.optimization,'run_variant',inference)
    first=study.run('fp16_cache_off');second=study.run('fp16_cache_off')
    assert calls==[True,True]
    assert first['session_evidence_path']!=second['session_evidence_path']
    assert first['session_id']==second['session_id']==study.session_id
    assert second['request_started_at']>=first['request_started_at']
    assert all((tmp_path/r['session_evidence_path']).exists() for r in (first,second))
    assert study.current_results()=={'fp16_cache_off':second}


def test_failed_live_start_is_saved_and_never_filled_by_archive(tmp_path,monkeypatch):
    study=live(tmp_path)
    @contextmanager
    def broken(*args):
        raise RuntimeError('startup failed')
        yield
    monkeypatch.setattr(study,'_native_server',broken)
    monkeypatch.setattr(study,'recorded',lambda *a:pytest.fail('archive fallback'))
    with pytest.raises(RuntimeError,match='startup failed'):study.run('fp16_cache_off')
    result=study.results['fp16_cache_off']
    assert result['status']=='failed' and result['session_id']==study.session_id
    assert (tmp_path/result['session_evidence_path']).is_file()
    assert 'failed' in study.report()


def test_every_live_display_excludes_old_and_archived_rows(monkeypatch):
    study=w.Study('vllm','qwen',root=ROOT,mode='live')
    old=dict(status='completed',experiment='OLD-MEASUREMENT',session_id='previous',
             quality={'result_equivalent':999},serving={'e2e_p50_seconds':999},
             configuration=study.configs['fp16_cache_off'])
    study.results['fp16_cache_off']=old
    study.results['fp16_cache_on']=dict(old,session_id=study.session_id,_recorded={'source':'old.json'})
    monkeypatch.setattr(study,'recorded',lambda *a:pytest.fail('implicit archive'))
    assert study.current_results()=={}
    assert '999' not in study.report() and 'OLD-MEASUREMENT' not in study.report()
    assert 'No controlled' in study.cache_report()
    assert '999' not in interpret(study,'fp16_cache_off')
    assert 'No response' in response_example(study,'fp16_cache_off')['status']
    assert 'Constructed from selected launch settings' in server_example(study,'fp16_cache_off')['command_origin']
    assert 'Constructed' in request_example(study,'fp16_cache_off')['note']
    from clinical_trial_mvp.tutorial.token_analysis import analyze
    from clinical_trial_mvp.tutorial.session_figures import figure
    assert analyze(study.current_results())['rows']==[]
    assert figure(study) is None


def test_plan_never_probes_hardware_or_executes_inference(tmp_path,monkeypatch):
    study=w.Study('vllm','qwen',root=ROOT,mode='plan')
    monkeypatch.setattr(w.subprocess,'run',lambda *a,**k:pytest.fail('plan subprocess'))
    monkeypatch.setattr(w.subprocess,'check_output',lambda *a,**k:pytest.fail('plan hardware'))
    monkeypatch.setattr(study,'recorded',lambda *a:pytest.fail('plan archive'))
    assert study.preflight()['status']=='not_required'
    assert 'Offline plan' in study.prepare()
    assert study.run('fp16_cache_off')['status']=='plan_only'
    assert study.current_results()['fp16_cache_off']['session_id']==study.session_id
    assert not (tmp_path/'experiment-tracker').exists()


def test_preflight_checks_cuda_in_selected_environment_without_launch(tmp_path,monkeypatch):
    study=live(tmp_path)
    for path in (study.base,study.awq):path.mkdir(parents=True);(path/'config.json').write_text('{}')
    python=tmp_path/'cuda-python';python.write_text('fixture')
    study.config['vllm_python']=str(python)
    monkeypatch.setattr(w.platform,'system',lambda:'Linux')
    monkeypatch.setattr(w,'endpoint_busy',lambda:False)
    monkeypatch.setattr(study,'_revisions',lambda **k:('a'*40,'b'*40))
    monkeypatch.setattr(w.subprocess,'Popen',lambda *a,**k:pytest.fail('preflight server launch'))
    calls=[]
    def check(command,**kwargs):
        calls.append(command)
        return 'fixture GPU' if command[0]=='nvidia-smi' else json.dumps(dict(cuda=True,vllm='fixture',python=str(python)))
    monkeypatch.setattr(w.subprocess,'check_output',check)
    assert study.preflight()['runtime']['cuda'] is True
    assert calls[1][0]==str(python)


def test_nim_preflight_checks_local_image_gpu_and_profile(tmp_path,monkeypatch):
    study=live(tmp_path,runtime='nim');study.base.mkdir(parents=True);(study.base/'config.json').write_text('{}')
    study.config['nim_profile_id']='chosen-profile'
    monkeypatch.setattr(w.platform,'system',lambda:'Linux')
    monkeypatch.setattr(w,'endpoint_busy',lambda:False)
    monkeypatch.setattr(study,'_revisions',lambda **k:('a'*40,None))
    calls=[]
    monkeypatch.setattr(w.subprocess,'check_output',lambda command,**k:calls.append(command) or 'fixture GPU')
    assert study.preflight()['selected_profile']=='chosen-profile'
    assert any('--pull=never' in command and '--gpus' in command for command in calls)
    assert not any('pull' in command for command in calls)


def test_edge_live_generation_runs_afresh_without_reading_recorded_results(tmp_path,monkeypatch):
    obj=EdgeStudy('qwen',root=tmp_path,mode='live')
    from clinical_trial_mvp.tutorial import edge_workflow as e,edge_notebook
    monkeypatch.setattr(edge_notebook,'recorded_qwen_edge',lambda *a:pytest.fail('archived generation'))
    calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        output=Path(command[command.index('--output')+1])
        output.write_text(json.dumps(dict(status='completed',stage='run',evidence='hardware',generation={'text':'SELECT 1'})))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(e.subprocess,'run',execute)
    a=obj.stage('run');b=obj.stage('run')
    assert len(calls)==2 and all('run' in c for c in calls)
    assert a['session_evidence_path']!=b['session_evidence_path']
    assert a['generation']['text']=='SELECT 1'
    with pytest.raises(ValueError,match='this session'):
        EdgeStudy('qwen',root=tmp_path,mode='live').stage('sql_check')


def test_reused_export_checks_identity_and_hashes_without_export(tmp_path,monkeypatch):
    obj=EdgeStudy('qwen',root=tmp_path,mode='live');obj.folder.mkdir(parents=True)
    identity=dict(model_id='Qwen/Qwen3-4B',revision='1cfa9a7208912126459214e8b04321603b3df60c')
    (obj.folder/'source-identity.json').write_text(json.dumps(identity))
    (obj.folder/'validation.json').write_text(json.dumps(dict(passed=True)))
    pte=obj.folder/'model.pte';pte.write_bytes(b'fixture')
    lineage=dict(model_id=identity['model_id'],model_revision=identity['revision'],files_sha256={'model.pte':hashlib.sha256(pte.read_bytes()).hexdigest()})
    (obj.folder/'artifact-lineage.json').write_text(json.dumps(lineage))
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('export should be reused'))
    assert obj.stage('export')['status']=='reused_artifact'
    obj._reuse_checked=False;pte.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):obj.stage('export')
    assert obj.results['export']['status']=='failed'


def test_nemotron_live_probe_is_fresh_and_never_claims_inference(tmp_path,monkeypatch):
    obj=EdgeStudy('nemotron',root=tmp_path,mode='live',config={'download_models':True})
    from clinical_trial_mvp.tutorial import nemotron_edge_notebook as n
    monkeypatch.setattr(n,'recorded_probe',lambda *a:pytest.fail('archived probe'))
    def execute(command,**kwargs):
        output=Path(command[command.index('--output')+1])
        output.write_text(json.dumps(dict(status='blocked_compatibility',native_inference_run=False)))
        return SimpleNamespace(returncode=0,stderr='')
    monkeypatch.setattr('subprocess.run',execute)
    result=obj.probe()
    assert result['status']=='blocked_compatibility' and not result['native_inference_run']
    assert result['session_id']==obj.session_id


def test_edge_plan_never_starts_a_process(tmp_path,monkeypatch):
    monkeypatch.setattr('subprocess.run',lambda *a,**k:pytest.fail('plan process'))
    for model in ('qwen','nemotron'):
        obj=EdgeStudy(model,root=tmp_path,mode='plan')
        assert obj.preflight()['status']=='not_required'
        result=obj.stage('run') if model=='qwen' else obj.probe()
        assert result['status']=='plan_only'
    assert not (tmp_path/'experiment-tracker').exists()


def test_failed_rerun_drops_earlier_success_records(tmp_path,monkeypatch):
    study=live(tmp_path)
    study.results['fp16_cache_off']=dict(status='completed',session_id=study.session_id,
        records={'quality':[{'case_id':'old'}]},quality={'result_equivalent':99})
    @contextmanager
    def broken(*args):
        raise RuntimeError('rerun failed')
        yield
    monkeypatch.setattr(study,'_native_server',broken)
    with pytest.raises(RuntimeError,match='rerun failed'):study.run('fp16_cache_off')
    result=study.results['fp16_cache_off']
    assert result['status']=='failed' and not result.get('records') and not result.get('quality')
