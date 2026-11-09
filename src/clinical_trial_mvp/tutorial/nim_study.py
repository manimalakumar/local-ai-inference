"""Model-specific NIM 2.x study using the inherited clinical workload and adapter.

No NIM image/model/GPU compatibility is inferred from a model-family name. These
helpers deploy a digest-pinned vLLM-backed model-free image with staged local HF
artifacts, then bind discovery and benchmark evidence to that exact deployment.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import os
import platform
import re
import subprocess
import time

from clinical_trial_mvp.serving.benchmark import digest, snapshot_fingerprint
from clinical_trial_mvp.serving.client import OpenAIClient
from clinical_trial_mvp.serving.portable import portable_text, portable_value
from clinical_trial_mvp.tutorial.harness import ROOT, serve_and_benchmark
from clinical_trial_mvp.tutorial.nim import nim_gate, runnable_profiles
from clinical_trial_mvp.tutorial.optimization import save_record

MODELS = {
    'qwen': {'model_id': 'Qwen/Qwen3-4B', 'model_type': 'qwen3', 'label': 'Qwen3-4B'},
    'nemotron': {'model_id': 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
                 'model_type': 'nemotron_h', 'label': 'Nemotron 3 Nano 4B'},
}


def template(key):
    model = MODELS[key]
    folder = 'qwen3-4b-frozen' if key == 'qwen' else 'nemotron-3-nano-4b-frozen'
    return {'model_key': key, 'model_id': model['model_id'],
            'model': f'clinical-trial-mvp4-nim-{key}', 'model_revision': 'SET_EXACT_BASE_REVISION',
            'image': 'nvcr.io/nim/nvidia/vllm-model-free-nim@sha256:SET_EXACT_DIGEST',
            'backend': 'vllm', 'snapshot': 'models/' + folder, 'gpu_device': '0',
            'hardware': '', 'gpu_uuid': '', 'profile_id': '', 'discovery_output': '',
            'launch_settings': {'context': 8192, 'dtype': 'float16', 'prefix_cache': False,
                                'quantization': None, 'max_num_seqs': 4,
                                **({'mamba_ssm_cache_dtype': 'float32', 'trust_remote_code': True}
                                   if key == 'nemotron' else {})}}


def safe_text(value):
    text = portable_text(value)
    for name in ('NGC_API_KEY', 'HF_TOKEN'):
        secret = os.environ.get(name)
        if secret:
            text = text.replace(secret, '[REDACTED]')
    return text


def validate(profile, *, root=ROOT):
    key = profile.get('model_key')
    if key not in MODELS or profile.get('model_id') != MODELS[key]['model_id']:
        raise ValueError('Exact model identity must match the Qwen3-4B or Nemotron 3 Nano 4B notebook')
    if profile.get('backend') != 'vllm':
        raise ValueError('This NIM 2.x deployment recipe requires a vLLM-backed model-free image')
    if not re.fullmatch(r'nvcr\.io/nim/[A-Za-z0-9_./-]+@sha256:[0-9a-f]{64}', profile.get('image', '')):
        raise ValueError('Select an accessible NIM model-free image pinned by SHA-256 digest')
    if not profile.get('model_revision') or profile['model_revision'].startswith('SET_'):
        raise ValueError('Record the exact local model revision')
    if not re.fullmatch(r'[A-Za-z0-9._/-]+', profile.get('model', '')):
        raise ValueError('Use a simple served model alias')
    if not re.fullmatch(r'(?:[0-9]+|GPU-[A-Za-z0-9-]+)', str(profile.get('gpu_device', ''))):
        raise ValueError('Choose one GPU index or UUID')
    settings = profile.get('launch_settings', {})
    for key in ('context', 'max_num_seqs'):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError('Positive integer launch setting required: ' + key)
    if settings.get('dtype') != 'float16' or settings.get('quantization') is not None:
        raise ValueError('The initial NIM study uses unquantized weights loaded as FP16')
    if type(settings.get('prefix_cache')) is not bool:
        raise ValueError('Set prefix_cache explicitly')
    snapshot = Path(root) / profile['snapshot']
    if not snapshot.is_dir():
        raise ValueError('Stage the complete local Hugging Face snapshot first')
    config = json.loads((snapshot / 'config.json').read_text())
    if config.get('model_type') != MODELS[profile['model_key']]['model_type'] or config.get('quantization_config'):
        raise ValueError('Snapshot architecture/precision differs from the requested base model')
    if profile['model_key'] == 'nemotron' and (
        settings.get('mamba_ssm_cache_dtype') != 'float32' or settings.get('trust_remote_code') is not True
    ):
        raise ValueError('Nemotron requires the recorded local-code and FP32 SSM settings in this recipe')
    return snapshot.resolve()


def commands(profile, *, root=ROOT):
    snapshot = validate(profile, root=root)
    settings = profile['launch_settings']
    common = ['docker', 'run', '--rm', '--gpus', 'device=' + str(profile['gpu_device']),
              '-e', 'NGC_API_KEY', '-e', 'HF_TOKEN', '-e', 'NIM_MODEL_PATH=/model',
              '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
              '--mount', f'type=bind,source={snapshot},target=/model,readonly']
    args = ['--served-model-name', profile['model'], '--dtype', 'float16',
            '--max-model-len', str(settings['context']), '--max-num-seqs', str(settings['max_num_seqs']),
            '--enable-prefix-caching' if settings['prefix_cache'] else '--no-enable-prefix-caching']
    if profile['model_key'] == 'nemotron':
        args += ['--trust-remote-code', '--mamba-ssm-cache-dtype', 'float32']
    return {'discover': [*common, profile['image'], 'list-model-profiles'],
            'deploy': [*common, '--detach', '--name', profile.get('container_name', 'clinical-trial-nim-' + profile['model_key']),
                       '--shm-size=16g', '-p', '127.0.0.1:8000:8000',
                       '-e', 'NIM_MODEL_PROFILE=' + profile.get('profile_id', ''),
                       profile['image'], *args]}


def gpu_identity(profile):
    raw = subprocess.check_output(['nvidia-smi', '--id', str(profile['gpu_device']),
        '--query-gpu=uuid,name', '--format=csv,noheader'], text=True, timeout=10).strip().splitlines()
    if len(raw) != 1 or ',' not in raw[0]:
        raise ValueError('Select exactly one GPU for this study')
    uuid, name = raw[0].split(',', 1)
    return uuid.strip(), name.strip()


def contract(profile, *, root=ROOT):
    snapshot = validate(profile, root=root)
    return {'model_id': profile['model_id'], 'model_revision': profile['model_revision'],
            'model_sha256': snapshot_fingerprint(snapshot), 'image': profile['image'],
            'model': profile['model'], 'backend': profile['backend'],
            'launch_settings': dict(profile['launch_settings']), 'gpu_device': str(profile['gpu_device'])}


def discover(profile, *, root=ROOT):
    if platform.system() != 'Linux':
        raise ValueError('Run NIM discovery on the Linux GPU host')
    requested = contract(profile, root=root)
    uuid, hardware = gpu_identity(profile)
    completed = subprocess.run(commands(profile, root=root)['discover'], capture_output=True, text=True)
    updated = dict(profile, hardware=hardware, gpu_uuid=uuid, discovery_contract=requested,
                   discovery_output=safe_text(completed.stdout), discovery_error=safe_text(completed.stderr),
                   discovery_succeeded=completed.returncode == 0,
                   discovered_at=datetime.now(timezone.utc).isoformat())
    ids = runnable_profiles(updated['discovery_output']) if completed.returncode == 0 else []
    # Profile selection is explicit; never silently pick a different precision/backend.
    if updated.get('profile_id') not in ids:
        updated['profile_id'] = ''
    updated['runnable_profile_ids'] = ids
    return updated


def get_json(client, route):
    with client.opener(client.base_url.removesuffix('/v1') + route, timeout=5) as response:
        if response.status != 200:
            raise ValueError('NIM HTTP endpoint unavailable: ' + route)
        body = response.read()
    return json.loads(body) if body else None


def deploy(profile, metadata_path, *, root=ROOT, timeout=1800, clock=time.perf_counter, sleep=time.sleep):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Use a positive finite readiness timeout')
    if platform.system() != 'Linux':
        raise ValueError('Run NIM deployment on the Linux GPU host')
    requested = contract(profile, root=root)
    uuid, hardware = gpu_identity(profile)
    if not profile.get('discovery_succeeded') or profile.get('discovery_contract') != requested:
        raise ValueError('Run discovery again after changing model, image, revision or launch settings')
    if profile.get('gpu_uuid') != uuid:
        raise ValueError('Discovery belongs to a different GPU')
    allowed, reason = nim_gate(profile, model=profile['model'], hardware=hardware, enabled=True, system='Linux')
    if not allowed:
        raise ValueError(reason)
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {'runtime': 'nim', 'status': 'starting', 'evidence': 'none',
              'model': profile['model'], 'model_id': profile['model_id'],
              'model_revision': profile['model_revision'], 'model_sha256': requested['model_sha256'],
              'image': profile['image'], 'profile_id': profile['profile_id'], 'hardware': hardware,
              'gpu_uuid': uuid, 'backend': 'vllm', 'engine_version': profile['image'],
              'launch_settings': profile['launch_settings'],
              'discovery_contract': requested, 'discovery_sha256': digest(profile['discovery_output']),
              'deployment_sha256': digest({'contract': requested, 'profile_id': profile['profile_id']}),
              'configuration_scope': 'Requested explicit CLI settings; retain/review resolved startup logs',
              'reasoning_control': 'common /no_think prompt only; backend template mode unverified'}
    # Reserve metadata before Docker starts; never overwrite a previous deployment.
    with path.open('x') as output:
        json.dump(result, output, indent=2)
    started = clock()
    try:
        completed = subprocess.run(commands(profile, root=root)['deploy'], capture_output=True, text=True)
        if completed.returncode:
            raise ValueError('NIM startup failed: ' + safe_text(completed.stderr))
        result['container_id'] = completed.stdout.strip()
        client = OpenAIClient('http://127.0.0.1:8000', profile['model'])
        while clock() - started < timeout:
            try:
                get_json(client, '/v1/health/ready')
                models = get_json(client, '/v1/models') or {}
                if profile['model'] not in {m.get('id') for m in models.get('data', [])}:
                    raise ValueError('Ready endpoint serves a different model alias')
                result.update(status='ready', evidence='hardware', readiness_seconds=clock() - started)
                break
            except OSError:
                sleep(2)
        else:
            raise ValueError('NIM readiness timed out; inspect and stop this container before retrying')
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result.update(status='failed', evidence='none', reason=safe_text(exc))
    path.write_text(json.dumps(portable_value(result), indent=2) + '\n')
    return result


def configurations(key, *, run_tag='run01'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('Use a simple unique run tag')
    model = MODELS[key]
    baseline = {'runtime': 'nim', 'host': 'cuda', 'hardware': 'GPU pending profile discovery',
                'model': f'clinical-trial-mvp4-nim-{key}', 'model_id': model['model_id'],
                'base_url': 'http://127.0.0.1:8000', 'context': 8192, 'dtype': 'float16',
                'quantization': None, 'prefix_cache': False, 'profile': 'full', 'concurrency': 1,
                'max_tokens': 1024, 'repeats': 5, 'warmups': 1, 'structured_outputs': False,
                'nim_profile': f'tutorial/nim_{key}.local.json',
                'provenance': f'experiment-tracker/mvp4/nim/provenance/{key}-{run_tag}.json'}
    return {name: {**baseline, **delta, 'experiment': f'MVP4/NIM/{key}/{name}'}
            for name, delta in {'smoke': {'repeats': 1}, 'baseline': {},
                                'schema_retrieved': {'profile': 'retrieved'},
                                'concurrency_2': {'concurrency': 2},
                                'concurrency_4': {'concurrency': 4}}.items()}


def run(spec, *, execute=False, root=ROOT, client=None, memory=None):
    """Profile + deployment binding before the common clinical quality/serving harness."""
    root = Path(root)
    if not execute:
        return serve_and_benchmark(spec, root=root, execute=False)
    injected = client is not None
    result = dict(run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'),
                  experiment=spec['experiment'], configuration=spec, runtime='nim',
                  hardware=spec['hardware'], status='skipped', evidence='none', quality=None,
                  serving=None, memory={'peak_bytes': None, 'scope': 'unmeasured'}, records={})
    try:
        profile = json.loads((root / spec['nim_profile']).read_text())
        allowed, reason = nim_gate(profile, model=spec['model'], hardware=profile.get('hardware'),
                                   enabled=True, system=platform.system())
        if not allowed:
            result['reason'] = reason
        else:
            if profile.get('model_id') != spec['model_id']:
                raise ValueError('Profile belongs to a different model; no silent substitution')
            server = json.loads((root / spec['provenance']).read_text())
            if server.get('status') != 'ready' or server.get('evidence') != 'hardware':
                raise ValueError('A successful recorded NIM deployment is required')
            for key in ('model', 'model_id', 'image', 'profile_id', 'gpu_uuid', 'model_revision'):
                if server.get(key) != profile.get(key):
                    raise ValueError('Profile/deployment binding mismatch: ' + key)
            if server.get('discovery_contract') != profile.get('discovery_contract'):
                raise ValueError('Discovery evidence differs from deployment')
            if server.get('model_sha256') != profile['discovery_contract'].get('model_sha256'):
                raise ValueError('Model fingerprint differs from discovery')
            if server.get('discovery_sha256') != digest(profile['discovery_output']):
                raise ValueError('Profile discovery output changed after deployment')
            if server.get('launch_settings') != profile.get('launch_settings'):
                raise ValueError('Launch settings changed after deployment')
            if not injected:
                probe = OpenAIClient(spec['base_url'], spec['model'])
                get_json(probe, '/v1/health/ready')
                models = get_json(probe, '/v1/models') or {}
                if spec['model'] not in {m.get('id') for m in models.get('data', [])}:
                    raise ValueError('Endpoint alias differs from recorded deployment')
            live_spec = {**spec, 'hardware': profile['hardware']}
            result = serve_and_benchmark(live_spec, root=root, execute=True,
                client=client, memory=memory, server=server, nim_profile=profile)
            result['nim_profile_identity'] = {'model_id': profile['model_id'], 'image': profile['image'],
                'profile_id': profile['profile_id'], 'backend': profile['backend']}
            result['pairing'] = 'Deployment comparison only: backend version, effective settings and thinking parity require separate verification'
            if spec['experiment'].endswith('/smoke') and result.get('status') == 'completed':
                quality, serving = result['quality'], result['serving']
                result['smoke_passed'] = quality['sql_executed'] > 0 and serving['failed'] == 0
                if not result['smoke_passed']:
                    result.update(status='failed', reason='NIM smoke produced no executable SQL or serving requests failed')
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result.update(status='failed', reason=safe_text(exc))
    if not injected:
        save_record(root, 'nim/runs', result)
    return portable_value(result)
