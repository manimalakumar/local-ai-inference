import builtins
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

from clinical_trial_mvp.tutorial import nim_study as n
from clinical_trial_mvp.tutorial.harness import ROOT, TUTORIAL_CASES
from clinical_trial_mvp.serving.benchmark import digest, Workload, prepare
from clinical_trial_mvp.serving.client import Sample, Generation
from clinical_trial_mvp.tutorial.memory import PeakMemory

PROFILE_ID = '1' * 64
DISCOVERY = f'''MODEL PROFILES
- Compatible with system and runnable:
  - {PROFILE_ID} (vllm-fp16-tp1-pp1)
- Compatible with system but low memory:
  - {'2' * 64} (vllm-bf16-tp1-pp1)
- Incompatible with system:
  - {'3' * 64} (vllm-fp16-tp4-pp1)
'''


def configured(root, key='qwen'):
    profile = n.template(key)
    profile.update(image='nvcr.io/nim/nvidia/vllm-model-free-nim@sha256:' + 'a' * 64,
                   model_revision='test-base-revision')
    folder = root / profile['snapshot']
    folder.mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'model_type': n.MODELS[key]['model_type']}))
    (folder / 'model.safetensors').write_bytes(b'weights-fixture')
    return profile


def builder():
    path = ROOT / 'scripts/tutorial/build_nim_notebooks.py'
    spec = importlib.util.spec_from_file_location('nim_notebook_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('key', ['qwen', 'nemotron'])
def test_templates_and_commands_keep_exact_model_and_credentials_out_of_payload(tmp_path, monkeypatch, key):
    profile = configured(tmp_path, key)
    monkeypatch.setenv('NGC_API_KEY', 'test-private-ngc-secret')
    monkeypatch.setenv('HF_TOKEN', 'test-private-hf-secret')
    commands = n.commands(profile, root=tmp_path)
    assert profile['model_id'] == n.MODELS[key]['model_id']
    assert 'NIM_MODEL_PATH=/model' in commands['discover']
    assert commands['discover'][-1] == 'list-model-profiles'
    assert 'readonly' in next(s for s in commands['deploy'] if s.startswith('type=bind'))
    assert '127.0.0.1:8000:8000' in commands['deploy']
    assert '--no-enable-prefix-caching' in commands['deploy']
    assert '--max-num-seqs' in commands['deploy']
    assert not any('secret' in s for cmd in commands.values() for s in cmd)
    if key == 'nemotron':
        assert '--mamba-ssm-cache-dtype' in commands['deploy'] and '--trust-remote-code' in commands['deploy']


@pytest.mark.parametrize('mutation', [
    {'image':'nvcr.io/nim/nvidia/vllm-model-free-nim:latest'},
    {'model_revision':'SET_REVISION'}, {'model_id':'nvidia/llama3.1-nemotron-nano-4b-v1.1'},
    {'backend':'sglang'}, {'gpu_device':'all'},
])
def test_invalid_or_substituted_deployment_rejected(tmp_path, mutation):
    profile = configured(tmp_path)
    with pytest.raises(ValueError):
        n.validate({**profile, **mutation}, root=tmp_path)


def test_discovery_keeps_explicit_selection_and_only_runnable_ids(tmp_path, monkeypatch):
    profile = configured(tmp_path)
    monkeypatch.setattr(n.platform,'system',lambda:'Linux')
    monkeypatch.setattr(n,'gpu_identity',lambda p:('GPU-test','Test GPU'))
    monkeypatch.setattr(n.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout=DISCOVERY,stderr=''))
    result = n.discover(profile,root=tmp_path)
    assert result['runnable_profile_ids'] == [PROFILE_ID]
    assert result['profile_id'] == ''  # no implicit precision/backend selection
    assert result['gpu_uuid'] == 'GPU-test' and result['discovery_succeeded']
    profile['profile_id'] = PROFILE_ID
    assert n.discover(profile,root=tmp_path)['profile_id'] == PROFILE_ID


def test_failed_discovery_retains_sanitized_error_without_runnable_profiles(tmp_path,monkeypatch):
    profile = configured(tmp_path)
    monkeypatch.setattr(n.platform,'system',lambda:'Linux')
    monkeypatch.setattr(n,'gpu_identity',lambda p:('GPU-test','Test GPU'))
    monkeypatch.setenv('NGC_API_KEY','test-ngc-secret')
    monkeypatch.setattr(n.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(
        returncode=1,stdout=DISCOVERY,stderr='Access failed: test-ngc-secret'))
    result = n.discover(profile,root=tmp_path)
    assert not result['discovery_succeeded'] and result['runnable_profile_ids'] == []
    assert 'test-ngc-secret' not in json.dumps(result) and '[REDACTED]' in result['discovery_error']


def deployment_fixture(root,monkeypatch):
    profile = configured(root)
    profile.update(hardware='Test GPU',gpu_uuid='GPU-test',profile_id=PROFILE_ID,
                   discovery_succeeded=True,discovery_output=DISCOVERY)
    profile['discovery_contract'] = n.contract(profile,root=root)
    monkeypatch.setattr(n.platform,'system',lambda:'Linux')
    monkeypatch.setattr(n,'gpu_identity',lambda p:('GPU-test','Test GPU'))
    return profile


def test_deployment_waits_for_health_and_checks_model_alias_before_hardware_evidence(tmp_path,monkeypatch):
    profile = deployment_fixture(tmp_path,monkeypatch)
    calls=[]
    monkeypatch.setattr(n.subprocess,'run',lambda command,**kw:
        calls.append(command) or types.SimpleNamespace(returncode=0,stdout='container-id\n',stderr=''))
    routes=[]
    def get(client,route):
        routes.append(route)
        return {'data':[{'id':profile['model']}]} if route.endswith('models') else None
    monkeypatch.setattr(n,'get_json',get)
    ticks=iter([10,11,12])
    path=tmp_path/'metadata.json'
    result=n.deploy(profile,path,root=tmp_path,clock=lambda:next(ticks))
    assert result['status']=='ready' and result['evidence']=='hardware'
    assert routes==['/v1/health/ready','/v1/models']
    assert result['readiness_seconds']==2
    assert result['model_sha256']==profile['discovery_contract']['model_sha256']
    assert json.loads(path.read_text())['profile_id']==PROFILE_ID
    with pytest.raises(FileExistsError):
        n.deploy(profile,path,root=tmp_path)
    assert len(calls)==1


def test_deployment_rejects_changed_discovery_contract_and_nonrunnable_profile(tmp_path,monkeypatch):
    profile=deployment_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(n.subprocess,'run',lambda *a,**kw:pytest.fail('must reject before Docker'))
    changed=copy.deepcopy(profile)
    changed['launch_settings']['context']=4096
    with pytest.raises(ValueError,match='discovery again'):
        n.deploy(changed,tmp_path/'changed.json',root=tmp_path)
    profile['profile_id']='2'*64
    with pytest.raises(ValueError,match='runnable'):
        n.deploy(profile,tmp_path/'low-memory.json',root=tmp_path)


def test_deployment_failure_is_saved_without_hardware_claim(tmp_path,monkeypatch):
    profile=deployment_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(n.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=1,stdout='',stderr='unsupported architecture'))
    path=tmp_path/'failure.json'
    result=n.deploy(profile,path,root=tmp_path)
    assert result['status']=='failed' and result['evidence']=='none'
    assert 'unsupported architecture' in result['reason']
    assert json.loads(path.read_text())['status']=='failed'


def test_ready_wrong_model_never_counts_as_success(tmp_path,monkeypatch):
    profile=deployment_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(n.subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0,stdout='container',stderr=''))
    monkeypatch.setattr(n,'get_json',lambda client,route:{'data':[{'id':'wrong-model'}]})
    result=n.deploy(profile,tmp_path/'wrong.json',root=tmp_path)
    assert result['status']=='failed' and 'different model' in result['reason']


@pytest.fixture
def run_root(tmp_path):
    for name in ('data/db/clinical_trials.duckdb','evals/evaluations.json','tutorial/semantic_reviews.json'):
        target=tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    (tmp_path/'src').symlink_to(ROOT/'src',target_is_directory=True)
    return tmp_path


def run_fixture(root,key='qwen',name='baseline'):
    spec=n.configurations(key)[name]
    spec['repeats']=1
    profile=n.template(key)
    profile.update(hardware='Test GPU',gpu_uuid='GPU-test',model_revision='test-revision',
                   image='nvcr.io/nim/nvidia/vllm-model-free-nim@sha256:'+'a'*64,
                   profile_id=PROFILE_ID,discovery_output=DISCOVERY,backend='vllm',
                   discovery_contract={'model_sha256':'b'*64})
    server={**{k:profile[k] for k in ('model','model_id','model_revision','image','profile_id','gpu_uuid',
                                     'hardware','launch_settings','discovery_contract')},
            'runtime':'nim','engine_version':profile['image'],'model_sha256':'b'*64,
            'status':'ready','evidence':'hardware','discovery_sha256':digest(DISCOVERY)}
    for path,value in ((root/spec['nim_profile'],profile),(root/spec['provenance'],server)):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    return spec,profile,server


PLAN={'population':'Enrolled or Randomized subjects','measure':'count subjects','grouping':'treatment arm',
      'time_window':'none','filters':'none','ranking':'none','needs_clarification':False,'clarification_question':''}


class Fake:
    generation=Generation()
    def complete(self,messages,schema=None):
        if schema and 'population' in schema.get('properties',{}):
            content=PLAN
        else:
            cases=prepare(ROOT,Workload(profile='full'),TUTORIAL_CASES)[0]
            case=next((c for c in cases if c['question'] in messages[-1]['content']),cases[0])
            content={'sql':case['reference_sql'],'rationale':''}
        return Sample(success=True,content=json.dumps(content),prompt_tokens=100,output_tokens=10,
                      ttft_seconds=.1,e2e_seconds=.2)


@pytest.mark.parametrize('key',['qwen','nemotron'])
def test_run_reuses_clinical_scoring_and_fake_client_is_not_hardware(run_root,monkeypatch,key):
    spec,_,_=run_fixture(run_root,key)
    monkeypatch.setattr(n.platform,'system',lambda:'Linux')
    result=n.run(spec,execute=True,root=run_root,client=Fake(),memory=PeakMemory(reader=lambda:100))
    assert result['status']=='completed',result.get('reason')
    assert result['evidence']=='test_or_unverified'
    assert result['quality']['cases']==4 and result['quality']['result_equivalent']>0
    assert result['quality']['semantic']['reviewed']==0
    assert not (run_root/'experiment-tracker/mvp4/nim/runs').exists()


def test_live_smoke_failure_is_not_a_pass(run_root,monkeypatch):
    spec,_,_=run_fixture(run_root,name='smoke')
    monkeypatch.setattr(n.platform,'system',lambda:'Linux')
    class Broken:
        generation=Generation()
        def complete(self,*a,**kw):
            return Sample(success=False,error='unsupported kernel')
    result=n.run(spec,execute=True,root=run_root,client=Broken(),memory=PeakMemory(reader=lambda:100))
    assert result['status']=='failed'
    assert not result.get('smoke_passed',False)


@pytest.mark.parametrize('field,value',[('model_id','different-model'),('profile_id','2'*64),('gpu_uuid','GPU-other')])
def test_model_profile_gpu_mismatch_fails_before_benchmark(run_root,monkeypatch,field,value):
    spec,_,server=run_fixture(run_root)
    server[field]=value
    (run_root/spec['provenance']).write_text(json.dumps(server))
    monkeypatch.setattr(n.platform,'system',lambda:'Linux')
    monkeypatch.setattr(n,'serve_and_benchmark',lambda *a,**kw:pytest.fail('mismatched deployment'))
    result=n.run(spec,execute=True,root=run_root,client=Fake())
    assert result['status']=='failed' and 'mismatch' in result['reason']


@pytest.mark.parametrize('key',['qwen','nemotron'])
def test_every_nim_notebook_cell_runs_offline_without_docker_gpu_or_network(monkeypatch,key):
    build=builder()
    document=json.loads((ROOT/'notebooks'/build.FILENAMES[key]).read_text())
    assert document==build.notebook_document(key)
    monkeypatch.setattr(n.subprocess,'run',lambda *a,**kw:pytest.fail('offline Docker call'))
    monkeypatch.setattr(n,'get_json',lambda *a,**kw:pytest.fail('offline network call'))
    monkeypatch.setattr(n,'gpu_identity',lambda *a,**kw:pytest.fail('offline GPU call'))
    original=builtins.__import__
    def guarded(name,*a,**kw):
        assert name.split('.')[0] not in {'torch','vllm','sglang','transformers'}
        return original(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    display=types.ModuleType('IPython.display')
    display.Markdown=lambda text:text
    display.display=lambda *a,**kw:None
    monkeypatch.setitem(sys.modules,'IPython.display',display)
    monkeypatch.chdir(ROOT/'notebooks')
    namespace={}
    for cell in document['cells']:
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            exec(compile(''.join(cell['source']),'nim-cell','exec'),namespace)
    assert namespace['MODEL_ID']==n.MODELS[key]['model_id']
    assert set(namespace['results'])=={'smoke','baseline','schema_retrieved','concurrency_2','concurrency_4'}
    assert all(r['status']=='plan_only' and r['evidence']=='none' for r in namespace['results'].values())


def test_cli_init_is_explicit_and_never_overwrites(tmp_path):
    spec=importlib.util.spec_from_file_location('nim_cli',ROOT/'scripts/tutorial/nim_deploy.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path=tmp_path/'nim.local.json'
    assert module.main(['init','--model','qwen','--profile',str(path)])==0
    assert json.loads(path.read_text())['model_id']=='Qwen/Qwen3-4B'
    with pytest.raises(SystemExit):
        module.main(['init','--model','nemotron','--profile',str(path)])
