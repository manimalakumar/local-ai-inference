"""Local nb00 configuration and launch rejection paths; no native engine is run."""
import importlib.util
import json
from pathlib import Path

import pytest

from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.foundation_study import FoundationStudy, local_template


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/tutorial'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_default_local_plan_selects_only_gguf_runtime(monkeypatch):
    monkeypatch.setenv('LOCAL_AI_LAB_MODE','plan')
    cfg=local_template()
    assert cfg['mode']=='live'
    assert 'gpu_servers' not in cfg and 'concurrency' not in cfg
    assert cfg['servers']['nemotron']['runtime']=='llama_cpp'
    study=FoundationStudy(ROOT,cfg)
    assert all(row['status']=='plan_only' for row in study.context_rows()+study.local_rows())
    assert study.gpu_rows()==[]


def test_local_setup_is_explicit_and_does_not_overwrite_config(tmp_path):
    module=script('configure_local_foundation')
    config=tmp_path/'baseline.local.json'
    args=['--mode','plan','--gguf',str(tmp_path/'model.gguf'),
          '--server-python','/test/local/python','--config',str(config)]
    assert module.main(args)==0
    saved=json.loads(config.read_text())
    assert saved['servers']['nemotron']['runtime']=='llama_cpp'
    with pytest.raises(SystemExit):module.main(args)


@pytest.mark.parametrize('reason',['metadata','occupied','fingerprint'])
def test_local_launcher_rejects_invalid_state_before_starting_an_engine(tmp_path,monkeypatch,reason):
    from contextlib import nullcontext
    module=script('launch_local_foundation')
    gguf=tmp_path/'model.gguf';gguf.write_bytes(b'test fixture: not a model')
    metadata=tmp_path/'metadata.json'
    if reason=='metadata':metadata.write_text('keep')
    def connect(*a,**k):
        if reason=='occupied':return nullcontext()
        raise ConnectionRefusedError
    monkeypatch.setattr(module.socket,'create_connection',connect)
    monkeypatch.setattr(module.subprocess,'Popen',lambda *a,**k:pytest.fail('started a native engine'))
    with pytest.raises(SystemExit):module.main(['--gguf',str(gguf),'--metadata',str(metadata)])
    if reason=='metadata':assert metadata.read_text()=='keep'


@pytest.mark.parametrize('http_failure', [False, True])
def test_visible_notebook_api_cell_sends_prompt_and_propagates_server_errors(monkeypatch, http_failure):
    import io
    import urllib.error
    from types import SimpleNamespace
    from clinical_trial_mvp.serving import client
    from clinical_trial_mvp.tutorial.serving_notebook import notebook_document

    calls=[]
    def open_request(request, timeout):
        calls.append((request, timeout))
        if http_failure:
            raise urllib.error.HTTPError(request.full_url, 503, 'Unavailable', {}, None)
        return io.BytesIO(json.dumps({'choices': [{
            'finish_reason': 'stop', 'message': {'content': '{"population":"subjects"}'}}],
            'usage': {'completion_tokens': 5}}).encode())

    original=client.OpenAIClient
    monkeypatch.setattr(client, 'OpenAIClient',
        lambda *args, **kwargs: original(*args, **kwargs, opener=open_request))
    namespace={'ROOT': ROOT, 'study': SimpleNamespace(mode='live', config=local_template()),
               'display': lambda *args: None, 'JSON': lambda x: x, 'Markdown': lambda x: x}
    cell=next(c for c in notebook_document()['cells'] if c['id']=='first-model-api-call')
    source=''.join(cell['source'])
    if http_failure:
        with pytest.raises(urllib.error.HTTPError):exec(source,namespace)
    else:
        exec(source,namespace)
        assert namespace['model_reply']=='{"population":"subjects"}'
    assert len(calls)==1
    request, timeout=calls[0]
    assert request.full_url=='http://127.0.0.1:8080/v1/chat/completions'
    assert request.get_method()=='POST' and timeout==180
    payload=json.loads(request.data)
    assert payload['model']=='clinical-trial-local' and payload['stream'] is False
    assert 'stream_options' not in payload
    assert payload['response_format']['type']=='json_schema'
    assert 'subjects' in payload['messages'][1]['content']
    assert namespace['first_question'] in payload['messages'][1]['content']


def test_visible_embedding_cell_encodes_passages_and_question_locally(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from clinical_trial_mvp.tutorial import foundation
    from clinical_trial_mvp.tutorial.serving_notebook import notebook_document

    calls=[]
    class FakeModel:
        def __init__(self, path, **options):
            assert path==str(tmp_path)
            assert options==dict(device='cpu',local_files_only=True,trust_remote_code=False)
        def encode(self,texts,**options):
            calls.append((texts,options))
            return SimpleNamespace(tolist=lambda: [[1.0,0.0] for _ in texts])
    fake=ModuleType('sentence_transformers');fake.SentenceTransformer=FakeModel
    monkeypatch.setitem(sys.modules,'sentence_transformers',fake)
    cfg=local_template();cfg['embedding_model_path']=str(tmp_path)
    namespace={'ROOT':ROOT, 'Path':Path, 'context':foundation.contexts(ROOT),
        'study':SimpleNamespace(mode='live',config=cfg),
        'display':lambda *args:None,'JSON':lambda value:value,'Markdown':lambda value:value}
    cell=next(c for c in notebook_document()['cells'] if c['id']=='embedding-model-call')
    exec(''.join(cell['source']),namespace)
    assert len(calls)==2
    assert all(text.startswith('passage: ') for text in calls[0][0])
    assert calls[1][0]==['query: '+namespace['embedding_question']]
    assert all(options['normalize_embeddings'] is True for _,options in calls)
    assert namespace['selected_context'].prompt.startswith('DuckDB clinical trial schema')
