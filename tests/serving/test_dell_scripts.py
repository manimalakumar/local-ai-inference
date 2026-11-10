"""Manual launcher contracts and notebook-interpreter selection; no GPU/server starts."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[2]


def launcher():
    spec=importlib.util.spec_from_file_location('dell_serve',ROOT/'scripts/dell/serve.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.mark.parametrize('runtime,cache,awq',[('vllm',False,False),('vllm',True,True),('sglang',False,False),('sglang',True,False)])
def test_manual_preview_has_explicit_runtime_precision_and_cache(runtime,cache,awq,tmp_path,monkeypatch,capsys):
    mod=launcher()
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**k:pytest.fail('preview started a server'))
    args=[runtime,'--python',str(tmp_path/'correct-python'),'--dry-run']
    if cache:args+=['--prefix-cache']
    if awq:args+=['--awq']
    assert mod.main(args)==0
    text=capsys.readouterr().out
    assert str(tmp_path/'correct-python') in text and '--dtype float16' in text and '--port 8001' in text
    if runtime=='vllm':
        assert 'VLLM_SERVER_DEV_MODE=1' in text
        assert ('--enable-prefix-caching' if cache else '--no-enable-prefix-caching') in text
    else:assert ('--disable-radix-cache' in text) is not cache
    assert ('--quantization awq' in text) is awq


def test_manual_metadata_fingerprint_matches_benchmark_contract(tmp_path):
    from clinical_trial_mvp.serving.benchmark import snapshot_fingerprint
    (tmp_path/'config.json').write_text('{"model_type":"qwen3"}')
    (tmp_path/'model.safetensors').write_bytes(b'fixture weights')
    (tmp_path/'.cache').mkdir();(tmp_path/'.cache/ignored.json').write_text('{}')
    assert launcher().snapshot_fingerprint(tmp_path)==snapshot_fingerprint(tmp_path)


def test_occupied_port_is_not_stopped_or_reused(tmp_path,monkeypatch):
    mod=launcher();runtime=tmp_path/'runtime-python';runtime.write_text('fixture')
    snapshot=tmp_path/'model';snapshot.mkdir();(snapshot/'config.json').write_text('{}')
    class Socket:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def connect_ex(self,*args):return 0
    monkeypatch.setattr(mod.platform,'system',lambda:'Linux')
    monkeypatch.setattr(mod.socket,'socket',lambda *a:Socket())
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**k:pytest.fail('external server must not be replaced'))
    monkeypatch.setattr(mod.os,'killpg',lambda *a:pytest.fail('external server must not be killed'))
    with pytest.raises(SystemExit):mod.main(['vllm','--python',str(runtime),'--snapshot',str(snapshot)])


def test_cleanup_targets_only_owned_process_group(monkeypatch):
    mod=launcher();killed=[];waited=[]
    monkeypatch.setattr(mod.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    mod.stop_owned(SimpleNamespace(pid=12345,wait=lambda **k:waited.append(k)))
    assert killed==[(12345,signal.SIGTERM)] and waited==[dict(timeout=30)]


def test_jupyter_uses_explicit_notebook_python_despite_wrong_active_environment(tmp_path):
    fake=tmp_path/'notebook-python';log=tmp_path/'calls.jsonl'
    fake.write_text('''#!/usr/bin/env python3
import json,os,sys
if sys.argv[1:2]==['-']:sys.stdin.read()
with open(os.environ['TEST_CALLS'],'a') as f:
 f.write(json.dumps({'args':sys.argv[1:],'mode':os.getenv('LOCAL_AI_LAB_MODE')})+'\\n')
''')
    fake.chmod(0o755)
    env=dict(os.environ,NOTEBOOK_PYTHON=str(fake),VIRTUAL_ENV='/wrong/sglang/env',
             TEST_CALLS=str(log),LOCAL_AI_LAB_MODE='replay')
    result=subprocess.run(['bash',str(ROOT/'scripts/dell/start_jupyter.sh')],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert any(c['args'][:3]==['-m','ipykernel','install'] for c in calls)
    final=calls[-1]
    assert final['args'][:3]==['-m','jupyter','lab']
    assert final['mode'] is None
    assert '--ip=127.0.0.1' in final['args'] and '--ServerApp.port_retries=0' in final['args']
    assert not any('vllm.entrypoints' in ' '.join(c['args']) for c in calls)
