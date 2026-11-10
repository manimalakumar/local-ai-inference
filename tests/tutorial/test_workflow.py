"""Meaningful lifecycle and regression tests without Docker, weights or a GPU."""
import hashlib
import json
import signal
import types

import pytest

from clinical_trial_mvp.tutorial import workflow as w
from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.memory import PeakMemory


def study(tmp_path, runtime='vllm', model='qwen'):
    config = w.template(runtime, model)
    config['execute'] = True
    config['mode'] = 'live'
    obj = w.Study(runtime, model, root=tmp_path, config=config)
    obj.prepared = True
    obj.python = '/gpu/env/bin/python'
    return obj


def test_archive_manifest_matches_all_preserved_bytes():
    manifest = json.loads((ROOT / 'notebooks/history/dell-2026-11-07/MANIFEST.json').read_text())
    assert len(manifest['files_sha256']) >= 65
    for path, expected in manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path


def test_native_memory_uses_nvidia_smi_uuid_format(monkeypatch):
    raw = '3cef7f0f-cdcd-b01a-ea3d-94c897e1b848'
    commands = []
    def output(command, **kwargs):
        commands.append(command)
        return '1024\n'
    monkeypatch.setattr(w.subprocess, 'check_output', output)
    assert PeakMemory(gpu_uuid=raw)._read() == 1024 * 1024**2
    assert commands[0][commands[0].index('--id') + 1] == 'GPU-' + raw
    assert PeakMemory(gpu_uuid='GPU-' + raw).gpu_uuid == 'GPU-' + raw
    assert PeakMemory(gpu_uuid=None).gpu_uuid is None


def test_configuration_is_offline_and_model_specific_by_default(tmp_path):
    for runtime in ('vllm', 'nim'):
        for model in ('qwen', 'nemotron'):
            config = w.read_config(runtime, model, root=tmp_path)
            assert 'execute' not in config and not config['download_models']
            assert config['mode'] == 'live'
            assert config['run_awq'] == (runtime == 'vllm' and model == 'qwen')
            assert config['structured_outputs'] == (model == 'qwen')
            assert config['nim_profile_id'] is None


@pytest.mark.parametrize('delta', [{'execute': 'true'}, {'repeats': 0},
    {'run_tag': '../escape'}, {'typo': True}, {'readiness_timeout_seconds': float('nan')},
    {'base_snapshot': None}, {'base_revision': True}])
def test_config_rejects_mistyped_or_invalid_controls(tmp_path, delta):
    path = tmp_path / 'tutorial/config/vllm-qwen.local.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(delta))
    with pytest.raises(ValueError):
        w.read_config('vllm', 'qwen', root=tmp_path)


def test_offline_prepare_never_calls_platform_downloads_or_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(w.platform, 'system', lambda: pytest.fail('offline platform probe'))
    monkeypatch.setattr(w.subprocess, 'run', lambda *a, **k: pytest.fail('offline Docker'))
    obj = w.Study('nim', 'qwen', root=tmp_path, mode='replay')
    assert 'Replay' in obj.prepare()
    assert not (tmp_path / 'experiment-tracker').exists()


def test_nemotron_core_reads_only_base_revision_without_awq(tmp_path):
    obj = study(tmp_path, model='nemotron')
    path = tmp_path / 'models/nemotron-revision.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'model_id': w.nemotron.MODEL_ID, 'revision': 'exact'}))
    assert obj._revisions() == ('exact', None)
    obj.config['run_awq'] = True
    assert obj._revisions() == ('exact', None)
    assert 'Core Nemotron' in obj.smoke()


def test_qwen_schema_setting_flows_into_every_nim_request_spec(tmp_path):
    obj = study(tmp_path, runtime='nim')
    assert all(s['structured_outputs'] is True for s in obj.configs.values())
    assert len({s['provenance'] for s in obj.configs.values()}) == 1
    other = study(tmp_path, runtime='nim')
    assert obj.configs['baseline']['provenance'] != other.configs['baseline']['provenance']


class Process:
    pid = 12345
    def poll(self):
        return None
    def wait(self, **kwargs):
        return 0


def test_native_server_uses_explicit_python_dev_mode_and_stops_process_group(tmp_path, monkeypatch):
    obj = study(tmp_path)
    spec = obj.configs['fp16_cache_on']
    monkeypatch.setattr(w, 'endpoint_busy', lambda: False)
    calls, killed = [], []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        (tmp_path / spec['provenance']).write_text(json.dumps({'runtime': 'vllm'}))
        return Process()
    monkeypatch.setattr(w.subprocess, 'Popen', popen)
    monkeypatch.setattr(w.nim_study, 'get_json', lambda c, route:
        {'data': [{'id': spec['model']}]} if route.endswith('/models') else None)
    monkeypatch.setattr(w.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    with pytest.raises(ValueError, match='measurement failed'):
        with obj._native_server(spec, tmp_path / 'weights'):
            assert obj.server['status'] == 'ready'
            assert obj.server['readiness_seconds'] >= 0
            raise ValueError('measurement failed')
    command, options = calls[0]
    assert command[0] == '/gpu/env/bin/python'
    assert options['env']['VLLM_SERVER_DEV_MODE'] == '1'
    assert options['start_new_session'] is True
    assert killed == [(Process.pid, signal.SIGTERM)]
    assert obj.process is None and obj.log_file is None
    assert json.loads((tmp_path / spec['provenance']).read_text())['server_dev_mode']


def test_external_server_is_never_stopped_or_reused(tmp_path, monkeypatch):
    obj = study(tmp_path)
    monkeypatch.setattr(w, 'endpoint_busy', lambda: True)
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **kw: pytest.fail('must not start'))
    monkeypatch.setattr(w.os, 'killpg', lambda *a: pytest.fail('must not kill'))
    with pytest.raises(RuntimeError, match='already occupied'):
        with obj._native_server(obj.configs['fp16_cache_off'], tmp_path):
            pytest.fail('must not yield')


def test_owned_native_server_is_cleaned_on_startup_failure(tmp_path, monkeypatch):
    obj = study(tmp_path)
    monkeypatch.setattr(w, 'endpoint_busy', lambda: False)
    process = Process()
    process.poll = lambda: 1
    monkeypatch.setattr(w.subprocess, 'Popen', lambda *a, **k: process)
    killed = []
    monkeypatch.setattr(w.os, 'killpg', lambda *args: killed.append(args))
    with pytest.raises(RuntimeError, match='exited during startup'):
        with obj._native_server(obj.configs['fp16_cache_off'], tmp_path):
            pytest.fail('failed server must not yield')
    assert killed and obj.log_file is None


def test_failed_nim_smoke_preserves_failure_and_cleans_container(tmp_path, monkeypatch):
    obj = study(tmp_path, runtime='nim')
    def start():
        obj.container = 'owned-container'
    monkeypatch.setattr(obj, '_nim_start', start)
    monkeypatch.setattr(w.nim_study, 'run', lambda *a, **k: dict(status='failed', reason='schema failure'))
    calls = []
    monkeypatch.setattr(w.subprocess, 'run', lambda command, **k:
        calls.append(command) or types.SimpleNamespace(stdout='log', stderr='', returncode=0))
    with pytest.raises(RuntimeError, match='schema failure'):
        obj.run('smoke')
    assert obj.results['smoke']['status'] == 'failed'
    assert ['docker', 'stop', '--time', '30', 'owned-container'] in calls
    assert obj.container is None
    obj.close()
    assert len(calls) == 2


def test_nim_benchmarks_cannot_bypass_failed_smoke(tmp_path, monkeypatch):
    obj = study(tmp_path, runtime='nim')
    obj.smoke_result = {'smoke_passed': False}
    monkeypatch.setattr(obj, '_nim_start', lambda: pytest.fail('no server start'))
    with pytest.raises(RuntimeError, match='successful NIM smoke'):
        obj.run('baseline')


def test_nim_reuses_one_owned_deployment_until_cleanup(tmp_path, monkeypatch):
    obj = study(tmp_path, runtime='nim')
    obj.profile = {}
    monkeypatch.setattr(w, 'endpoint_busy', lambda: False)
    deployed, calls = [], []
    def deploy(profile, metadata, **kwargs):
        deployed.append((dict(profile), metadata))
        return {'status': 'ready'}
    monkeypatch.setattr(w.nim_study, 'deploy', deploy)
    monkeypatch.setattr(w.nim_study, 'run', lambda spec, **kwargs:
        dict(status='completed', smoke_passed=True, configuration=spec))
    def docker(command, **kwargs):
        calls.append(command)
        return types.SimpleNamespace(returncode=1 if command[1] == 'container' else 0, stdout='', stderr='')
    monkeypatch.setattr(w.subprocess, 'run', docker)
    for name in ('smoke', 'baseline', 'schema_retrieved', 'concurrency_2', 'concurrency_4'):
        assert obj.run(name)['status'] == 'completed'
    assert len(deployed) == 1
    name = deployed[0][0]['container_name']
    assert obj.session_id.lower() in name
    assert name != 'clinical-trial-nim-qwen'
    assert not any(c[1] == 'stop' for c in calls)
    obj.close()
    assert ['docker', 'stop', '--time', '30', name] in calls


def test_failed_docker_stop_can_be_retried_without_losing_ownership(tmp_path, monkeypatch):
    obj = study(tmp_path, runtime='nim')
    obj.container = 'owned'
    def fail_stop(command, **kwargs):
        return types.SimpleNamespace(returncode=1, stdout='', stderr='daemon unavailable')
    monkeypatch.setattr(w.subprocess, 'run', fail_stop)
    with pytest.raises(RuntimeError, match='Docker cleanup failed'):
        obj.close()
    assert obj.container == 'owned'


def test_nemotron_optional_experiments_skip_without_server(tmp_path, monkeypatch):
    obj = study(tmp_path, model='nemotron')
    monkeypatch.setattr(obj, '_native_server', lambda *a: pytest.fail('no server'))
    assert obj.run('prefix_cache')['status'] == 'skipped'
    assert obj.run('awq_stretch')['status'] == 'skipped'
    assert len(list((tmp_path / 'experiment-tracker/mvp4/workflow/skipped').glob('*.json'))) == 2


@pytest.mark.parametrize('name', ['schema_retrieved', 'concurrency_2', 'concurrency_4',
                                'batch_capacity_1', 'batch_capacity_2', 'reasoning_on',
                                'output_256', 'output_2048'])
def test_qwen_new_cells_dispatch_to_control_runner_with_smoke_and_pair(tmp_path, monkeypatch, name):
    from contextlib import contextmanager
    obj = study(tmp_path)
    obj.smoke_result = {'status':'completed'}
    obj.pair = {'fp16_sha256':'base'}
    @contextmanager
    def server(spec, snapshot):
        assert snapshot == obj.base
        yield
    monkeypatch.setattr(obj, '_native_server', server)
    calls = []
    def run(spec, pair, **kwargs):
        calls.append((spec, pair, kwargs))
        return {'status':'completed', 'configuration':spec}
    monkeypatch.setattr(w.optimization, 'run_control', run)
    monkeypatch.setattr(w.optimization, 'run_variant', lambda *a, **k: pytest.fail('factorial cache path used for concurrent controls'))
    assert obj.run(name)['status'] == 'completed'
    assert calls[0][1] == obj.pair and calls[0][2]['smoke'] == obj.smoke_result
    assert calls[0][2]['execute'] is True


def test_nemotron_failed_awq_smoke_is_retried_before_benchmark(tmp_path, monkeypatch):
    obj = study(tmp_path, model='nemotron')
    obj.config['run_awq'] = True
    obj.results['fp16_baseline'] = {'status':'completed'}
    obj.smoke_result = {'status':'failed'}
    def retry():
        raise RuntimeError('compatibility smoke retried')
    monkeypatch.setattr(obj, '_stretch_smoke', retry)
    monkeypatch.setattr(obj, '_native_server', lambda *a: pytest.fail('must not benchmark failed smoke'))
    with pytest.raises(RuntimeError, match='smoke retried'):
        obj.run('awq_stretch')
    assert obj.results['awq_stretch']['status'] == 'failed'
    assert obj.results['fp16_baseline']['status'] == 'completed'


def test_nemotron_awq_rejects_unpinned_source_before_download(tmp_path, monkeypatch):
    obj = study(tmp_path, model='nemotron')
    obj.config.update(run_awq=True, awq_source='example/checkpoint', awq_revision='main', awq_derivation='recipe')
    monkeypatch.setattr(obj, '_native_server', lambda *a: pytest.fail('must validate before startup'))
    with pytest.raises(ValueError, match='exact compatible source'):
        obj._stretch_smoke()


@pytest.mark.parametrize('smoke_status', ['completed', 'failed'])
def test_nemotron_awq_validates_pair_and_smokes_before_benchmark(tmp_path, monkeypatch, smoke_status):
    from contextlib import contextmanager
    obj = study(tmp_path, model='nemotron')
    obj.config.update(run_awq=True, awq_source='example/validated-awq',
                      awq_revision='b'*40, awq_derivation='documented conversion from base')
    obj.base_revision = 'a'*40
    obj.results['fp16_baseline'] = {'status':'completed'}
    config = {'model_type':'nemotron_h', 'architectures':['NemotronHForCausalLM'],
              'hidden_size':16, 'num_hidden_layers':2, 'vocab_size':100}
    for folder in (obj.base, obj.awq):
        folder.mkdir(parents=True)
        (folder/'config.json').write_text(json.dumps(config))
        (folder/'tokenizer.json').write_text('{"same":true}')
        (folder/'model.safetensors').write_bytes(folder.name.encode())
    config['quantization_config'] = {'quant_method':'awq','bits':4}
    (obj.awq/'config.json').write_text(json.dumps(config))
    events = []
    @contextmanager
    def server(spec, snapshot):
        assert snapshot == obj.awq and spec['mamba_ssm_cache_dtype'] == 'float32'
        events.append('start')
        yield
        events.append('stop')
    monkeypatch.setattr(obj, '_native_server', server)
    def smoke(spec, pair, **kwargs):
        assert pair['awq_config']['quant_method'] == 'awq'
        assert pair['fp16_sha256'] != pair['awq_sha256']
        assert kwargs['execute'] is True
        events.append('smoke')
        return {'status':smoke_status,'configuration':spec}
    monkeypatch.setattr(w.optimization, 'awq_smoke', smoke)
    def benchmark(spec, checkpoint, **kwargs):
        assert kwargs['smoke']['status'] == 'completed'
        assert kwargs['pair'] == obj.pair
        events.append('benchmark')
        return {'status':'completed','configuration':spec}
    monkeypatch.setattr(w.nemotron, 'run', benchmark)
    if smoke_status == 'completed':
        assert obj.run('awq_stretch')['status'] == 'completed'
        assert events == ['start','smoke','stop','start','benchmark','stop']
    else:
        with pytest.raises(RuntimeError, match='AWQ smoke failed'):
            obj.run('awq_stretch')
        assert events == ['start','smoke','stop']
        assert obj.results['awq_stretch']['status'] == 'failed'
    assert obj.results['fp16_baseline']['status'] == 'completed'
    assert obj.results['smoke']['session_id'] == obj.session_id
