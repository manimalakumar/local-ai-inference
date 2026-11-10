"""Notebook orchestration. Explicit live sessions; own only servers we start."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import socket
import subprocess
import time
import uuid
import sys
from clinical_trial_mvp.serving.portable import portable_value

from clinical_trial_mvp.serving.client import OpenAIClient
from clinical_trial_mvp.tutorial import nemotron, nim_study, optimization
from clinical_trial_mvp.tutorial.harness import ROOT, measurements_markdown
from clinical_trial_mvp.tutorial.cache_study import phase_table
from clinical_trial_mvp.tutorial.replay import RecordedStudy, LABEL, display_result


def template(runtime, model):
    if runtime not in ('vllm', 'nim') or model not in ('qwen', 'nemotron'):
        raise ValueError('Choose vllm/nim and qwen/nemotron')
    return dict(runtime=runtime, model=model, mode='live', download_models=False,
                base_snapshot='models/qwen3-4b-frozen' if model == 'qwen' else 'models/nemotron-3-nano-4b-frozen',
                awq_snapshot='models/qwen3-4b-awq-4bit-frozen' if model == 'qwen' else 'models/nemotron-3-nano-4b-awq-4bit-frozen',
                vllm_python='~/.venv-vllm/bin/python', run_tag=None,
                base_revision=None, awq_revision=None,
                awq_source='Qwen/Qwen3-4B-AWQ' if model == 'qwen' else None,
                awq_derivation=None, run_awq=model == 'qwen' and runtime == 'vllm',
                prefix_supported=False, repeats=5, readiness_timeout_seconds=1800,
                structured_outputs=model == 'qwen',
                nim_image=None, nim_image_tag='nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.5',
                nim_profile_id=None, nim_profile_path=f'tutorial/nim_{model}.local.json')


def read_config(runtime, model, *, root=ROOT, mode=None):
    path = Path(root) / f'tutorial/config/{runtime}-{model}.local.json'
    config = template(runtime, model)
    if path.exists():
        supplied = json.loads(path.read_text())
        unknown = supplied.keys() - config.keys() - {'execute'}
        if unknown:
            raise ValueError('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
        config.update(supplied)
        if 'execute' in supplied and type(supplied['execute']) is not bool:
            raise ValueError('execute must be a boolean; remove it and use MODE')
        if mode is None and 'execute' in supplied:
            expected = config['mode'] == 'live'
            if supplied['execute'] != expected:
                raise ValueError('Contradictory execute/mode settings; remove execute and use MODE')
        config.pop('execute', None)
    if (config['runtime'], config['model']) != (runtime, model):
        raise ValueError('Configuration belongs to another notebook')
    # Notebook validation explicitly forces a non-live mode in its child kernel.
    override = mode if mode is not None else os.environ.get('LOCAL_AI_LAB_MODE')
    if override:
        config['mode'] = override
    if config['mode'] not in ('replay', 'plan', 'live'):
        raise ValueError('mode must be replay, plan or live')
    for key in ('download_models', 'run_awq', 'prefix_supported', 'structured_outputs'):
        if type(config[key]) is not bool:
            raise ValueError(key + ' must be a JSON boolean')
    for key in ('vllm_python', 'base_snapshot', 'awq_snapshot', 'nim_image_tag', 'nim_profile_path'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(key + ' must be a nonempty path/string')
    for key in ('run_tag', 'base_revision', 'awq_revision', 'awq_source', 'awq_derivation', 'nim_image', 'nim_profile_id'):
        if config[key] is not None and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(key + ' must be a nonempty string or null')
    if type(config['repeats']) is not int or config['repeats'] < 1:
        raise ValueError('repeats must be a positive integer')
    timeout = config['readiness_timeout_seconds']
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 7200:
        raise ValueError('readiness timeout must be between 0 and 7200 seconds')
    # Reuse the study validation for user-supplied tags; every session still has a UUID.
    if config['run_tag'] is not None:
        optimization.configurations(run_tag=config['run_tag'])
    return config


def endpoint_busy():
    try:
        with socket.create_connection(('127.0.0.1', 8000), timeout=1):
            return True
    except ConnectionRefusedError:
        return False


class Study:
    def __init__(self, runtime, model, *, root=ROOT, config=None, mode=None):
        self.root = Path(root)
        self.config = read_config(runtime, model, root=root, mode=mode) if config is None else dict(config)
        if mode is not None:
            self.config['mode'] = mode
            self.config.pop('execute', None)
        elif 'execute' in self.config:
            if type(self.config['execute']) is not bool or self.config['execute'] != (self.config.get('mode') == 'live'):
                raise ValueError('Contradictory execute/mode settings; remove execute and use MODE')
            self.config.pop('execute')
        if self.config.get('mode') not in ('live', 'plan', 'replay'):
            raise ValueError('MODE must be live, plan or replay')
        self.runtime, self.model = runtime, model
        self.session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:10]
        self.tag = (self.config['run_tag'] or f'{runtime}-{model}') + '-' + self.session_id
        self.configs = (nim_study.configurations(model, run_tag=self.tag) if runtime == 'nim' else
                        nemotron.configurations(self.tag, repeats=self.config['repeats']) if model == 'nemotron' else
                        optimization.configurations(run_tag=self.tag, repeats=self.config['repeats']))
        if runtime == 'nim':
            for spec in self.configs.values():
                spec.update(nim_profile=self.config['nim_profile_path'],
                            structured_outputs=self.config['structured_outputs'])
                if not spec['experiment'].endswith('/smoke'):
                    spec['repeats'] = self.config['repeats']
        self.base = self.root / Path(self.config['base_snapshot']).expanduser()
        self.awq = self.root / Path(self.config['awq_snapshot']).expanduser()
        self.results = {}
        self.checkpoint = self.pair = self.smoke_result = self.profile = None
        self.prepared = False
        self.process = self.log_file = self.container = self.server = None
        self.log_path = None

    @property
    def live(self):
        return self.mode == 'live'

    @property
    def mode(self):
        return self.config['mode']

    def recorded(self, name):
        if self.mode != 'replay':
            raise RuntimeError('Archived results require explicit Replay mode')
        return RecordedStudy(self.root).result(self.runtime, self.model, name)

    def _revisions(self, *, allow_download=True):
        path = self.root / f'models/{"qwen-revisions" if self.model == "qwen" else "nemotron-revision"}.json'
        saved = json.loads(path.read_text()) if path.exists() else {}
        base = saved.get('base', {}) if self.model == 'qwen' else saved
        awq = saved.get('awq', {})
        base_id = nim_study.MODELS[self.model]['model_id']
        if base and base.get('model_id') != base_id:
            raise ValueError('Saved base revision belongs to a different model')
        if awq and awq.get('model_id') != self.config['awq_source']:
            raise ValueError('Saved AWQ revision belongs to a different source')
        base_revision = self.config['base_revision'] or base.get('revision')
        awq_revision = self.config['awq_revision'] or awq.get('revision')
        # Nemotron stretch artifacts are deliberately not consulted by core preparation.
        needs_awq = self.runtime == 'vllm' and self.model == 'qwen' and self.config['run_awq']
        if self.config['download_models'] and allow_download:
            from huggingface_hub import HfApi, snapshot_download
            base_revision = base_revision or HfApi().model_info(base_id).sha
            snapshot_download(repo_id=base_id, revision=base_revision, local_dir=str(self.base))
            base = dict(model_id=base_id, revision=base_revision)
            if needs_awq:
                source = self.config['awq_source']
                if not source:
                    raise ValueError('AWQ requires an explicit compatible checkpoint source')
                awq_revision = awq_revision or HfApi().model_info(source).sha
                snapshot_download(repo_id=source, revision=awq_revision, local_dir=str(self.awq))
                awq = dict(model_id=source, revision=awq_revision)
            saved = dict(base=base, awq=awq) if self.model == 'qwen' else base
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(saved, indent=2) + '\n')
        if not base_revision or (needs_awq and not awq_revision):
            raise ValueError('Set exact base/AWQ revisions in the configuration, restore the models revision JSON, or enable download_models')
        return base_revision, awq_revision

    def preflight(self, *, require_profile=True):
        """Validate local prerequisites; never install, download, pull or start a model."""
        if not self.live:
            return dict(mode=self.mode, session_id=self.session_id, status='not_required',
                        reason='Plan/Replay performs no hardware probes')
        if platform.system() != 'Linux':
            raise ValueError('Live serving requires the Linux NVIDIA GPU host; select MODE="plan" or "replay" here')
        self._free_endpoint()
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,uuid',
                '--format=csv,noheader'], text=True, timeout=20).strip()
            if not gpu:
                raise ValueError('nvidia-smi found no GPU')
            self.hardware_target = gpu
            if not self.config['download_models']:
                base_revision, awq_revision = self._revisions(allow_download=False)
                for revision in (base_revision, awq_revision):
                    if revision is not None and not re.fullmatch(r'[0-9a-f]{40}', revision):
                        raise ValueError('Use exact 40-character checkpoint commit SHAs; see tutorial/RUN_ALL.md')
            snapshots = [self.base]
            if self.runtime == 'vllm' and self.model == 'qwen':
                snapshots.append(self.awq)
            for snapshot in snapshots:
                if not (snapshot / 'config.json').is_file() and not self.config['download_models']:
                    raise ValueError('Checkpoint missing: ' + str(snapshot) + '; stage weights or opt in to download_models')
            details = dict(mode=self.mode, session_id=self.session_id, hardware=gpu,
                           checkpoints=[str(p) for p in snapshots])
            if self.runtime == 'vllm':
                python = Path(self.config['vllm_python']).expanduser()
                if not python.is_absolute():
                    python = self.root / python
                if not python.is_file():
                    raise ValueError('Set vllm_python to the existing CUDA environment Python')
                probe = subprocess.check_output([str(python), '-c',
                    'import json,sys,torch,vllm; print(json.dumps(dict(python=sys.executable,vllm=vllm.__version__,cuda=torch.cuda.is_available())))'],
                    text=True, timeout=120)
                info = json.loads(probe.strip().splitlines()[-1])
                if not info['cuda']:
                    raise ValueError('CUDA is unavailable in vllm_python')
                details['runtime'] = info
            else:
                subprocess.check_output(['docker', 'info', '--format', '{{json .ServerVersion}}'], text=True, timeout=30)
                profile_path = self.root / self.config['nim_profile_path']
                profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
                saved_image = profile.get('image') or ''
                image = self.config['nim_image'] or (saved_image if 'SET_' not in saved_image and saved_image else self.config['nim_image_tag'])
                subprocess.check_output(['docker', 'image', 'inspect', image], text=True, timeout=30)
                subprocess.check_output(['docker', 'run', '--rm', '--pull=never', '--gpus', 'device='+str(profile.get('gpu_device', '0')),
                    '--entrypoint', 'nvidia-smi', image, '-L'], text=True, timeout=60)
                selected = self.config['nim_profile_id'] or profile.get('profile_id')
                if not selected and require_profile:
                    raise ValueError('Choose nim_profile_id after discovery; see tutorial/NIM_MVP4.md')
                if selected and profile.get('runnable_profile_ids') and selected not in profile['runnable_profile_ids']:
                    raise ValueError('Selected NIM profile is not listed as runnable; repeat discovery for this GPU/image')
                details.update(image=image, selected_profile=selected)
            self.preflight_result = details
            return details
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError('Preflight failed; check runtime, GPU, Docker access and pre-staged image/checkpoints. No automatic installation or image pull: ' + str(exc)) from exc

    def current_results(self):
        if self.mode == 'replay':
            return dict(self.results)
        return {name: result for name, result in self.results.items()
                if result.get('session_id') == self.session_id and not result.get('_recorded')}

    def _capture(self, name, result):
        result = dict(result, session_id=self.session_id, mode=self.mode,
                      request_started_at=self.request_started_at,
                      recorded_at=datetime.now(timezone.utc).isoformat())
        result.setdefault('experiment', result.get('configuration', {}).get('experiment', name))
        if self.live:
            path = self._path(name + '-' + uuid.uuid4().hex[:8], '.json')
            result['session_evidence_path'] = str(path.relative_to(self.root))
            with path.open('x') as output:
                json.dump(portable_value(result), output, indent=2, default=str)
                output.write('\n')
        self.results[name] = result
        return result

    def prepare(self):
        if self.mode == 'replay':
            return 'Replay: recorded Dell measurements; no model download, server, GPU or network calls.'
        if self.prepared or not self.live:
            return 'Offline plan: no downloads, server, Docker or GPU calls.'
        if platform.system() != 'Linux':
            raise ValueError('Live experiments require the Linux GPU host')
        if not hasattr(self, 'preflight_result'):
            self.preflight()
        base_revision, awq_revision = self._revisions()
        self.base_revision = base_revision
        if self.runtime == 'vllm':
            python = Path(self.config['vllm_python']).expanduser()
            if not python.is_absolute():
                python = self.root / python
            if not python.is_file():
                raise ValueError('Set vllm_python to the working GPU environment Python; no project symlink is needed')
            self.python = str(python)
            if self.model == 'qwen' and not self.config['run_awq']:
                raise ValueError('The Qwen four-variant study requires its AWQ compatibility smoke first')
            if self.model == 'nemotron':
                self.checkpoint = nemotron.baseline_checkpoint(self.base, revision=base_revision)
            if self.model == 'qwen' and self.config['run_awq']:
                if not self.config['awq_derivation']:
                    raise ValueError('Record awq_derivation: an explicit attestation linking the AWQ checkpoint to the exact base revision')
                self.pair = optimization.checkpoint_pair(self.base, self.awq,
                    base_revision=f'{nim_study.MODELS[self.model]["model_id"]}@{base_revision}; '
                                  f'{self.config["awq_source"]}@{awq_revision}; {self.config["awq_derivation"]}')
        else:
            path = self.root / self.config['nim_profile_path']
            profile = json.loads(path.read_text()) if path.exists() else nim_study.template(self.model)
            profile.update(snapshot=str(self.base), model_revision=base_revision)
            image = self.config['nim_image']
            if not image and 'SET_' not in profile['image']:
                image = profile['image']
            if not image:
                image = subprocess.check_output(['docker', 'image', 'inspect', self.config['nim_image_tag'],
                    '--format', '{{index .RepoDigests 0}}'], text=True, timeout=30).strip()
            profile['image'] = image
            selected = self.config['nim_profile_id'] or profile.get('profile_id')
            profile = nim_study.discover(profile, root=self.root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(profile, indent=2) + '\n')
            if selected not in profile['runnable_profile_ids']:
                raise ValueError('Set nim_profile_id to a runnable ID from ' + str(path) +
                                 ': ' + str(profile['runnable_profile_ids']))
            profile['profile_id'] = selected
            path.write_text(json.dumps(profile, indent=2) + '\n')
            self.profile = profile
        self.prepared = True
        return 'Validated local checkpoint/configuration; ready to run.'

    def _path(self, name, suffix):
        path = self.root / f'experiment-tracker/mvp4/workflow/{self.tag}-{name}{suffix}'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _free_endpoint(self):
        if endpoint_busy():
            raise RuntimeError('Port 8000 is already occupied. Stop that server explicitly; this notebook will not stop an external server.')

    @contextmanager
    def _native_server(self, spec, snapshot):
        self._free_endpoint()
        spec['provenance'] = str(self._path(uuid.uuid4().hex[:8], '.json').relative_to(self.root))
        preview = nemotron.launch_preview if self.model == 'nemotron' else optimization.launch_preview
        command = shlex.split(preview(spec, snapshot, python=self.python))
        self.log_path = self._path(uuid.uuid4().hex[:8], '.log')
        self.log_file = self.log_path.open('x')
        started = time.monotonic()
        print('Starting vLLM; waiting for readiness. Log:', self.log_path, flush=True)
        try:
            self.process = subprocess.Popen(command, cwd=self.root, env={**os.environ,
                'VLLM_SERVER_DEV_MODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)
            client = OpenAIClient(spec['base_url'], spec['model'])
            path = self.root / spec['provenance']
            while time.monotonic() - started < self.config['readiness_timeout_seconds']:
                if self.process.poll() is not None:
                    raise RuntimeError('vLLM exited during startup; inspect ' + str(self.log_path))
                try:
                    metadata = json.loads(path.read_text())
                    nim_study.get_json(client, '/health')
                    models = nim_study.get_json(client, '/v1/models') or {}
                    if spec['model'] not in {m.get('id') for m in models.get('data', [])}:
                        raise ValueError('Server model alias differs from the requested deployment')
                    metadata.update(status='ready', server_pid=self.process.pid,
                        readiness_seconds=time.monotonic() - started,
                        server_dev_mode=True, startup_log=str(self.log_path.relative_to(self.root)))
                    path.write_text(json.dumps(metadata, indent=2) + '\n')
                    self.server = metadata
                    break
                except (OSError, json.JSONDecodeError):
                    time.sleep(2)
            else:
                raise RuntimeError('vLLM readiness timeout; inspect ' + str(self.log_path))
            yield
        except BaseException as exc:
            path = self.root / spec['provenance']
            # Preserve partial startup evidence, including failures before the launcher writes metadata.
            if self.server is None or self.server.get('status') != 'ready':
                try:
                    record = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    record = {}
                record.update(status='failed', evidence='none', reason=str(exc),
                              startup_log=str(self.log_path.relative_to(self.root)))
                path.write_text(json.dumps(record, indent=2) + '\n')
            raise
        finally:
            self.close()

    def _nim_start(self):
        if self.container:
            return
        self._free_endpoint()
        name = f'clinical-trial-mvp4-{self.model}-{self.session_id}'.lower()
        probe = subprocess.run(['docker', 'container', 'inspect', name], capture_output=True, text=True, timeout=30)
        if probe.returncode == 0:
            raise RuntimeError('Container already exists; refusing to take ownership: ' + name)
        self.profile['container_name'] = name
        self.container = name
        metadata = self.root / self.configs['baseline']['provenance']
        print('Starting NIM; waiting for readiness. Container:', name, flush=True)
        self.server = nim_study.deploy(self.profile, metadata, root=self.root,
                                      timeout=self.config['readiness_timeout_seconds'])
        if self.server['status'] != 'ready':
            raise RuntimeError(self.server.get('reason', 'NIM failed readiness'))

    def smoke(self):
        if not self.live or self.runtime == 'nim' or self.model == 'nemotron':
            return self._smoke()
        self.request_started_at = datetime.now(timezone.utc).isoformat()
        try:
            return self._smoke()
        except BaseException as exc:
            self.smoke_result = self._capture('smoke', dict(status='failed', evidence='none', reason=str(exc),
                configuration=self.configs['awq_w4a16_cache_off']))
            raise

    def _smoke(self):
        if self.mode == 'replay':
            self.smoke_result = self.recorded('smoke')
            if self.runtime == 'nim':
                self.results['smoke'] = self.smoke_result
            return self.smoke_result
        if self.runtime == 'nim':
            return self.run('smoke')
        if self.model == 'nemotron':
            return 'Core Nemotron has no AWQ dependency; enabled AWQ smoke runs only at the final stretch row.'
        if not self.live:
            self.smoke_result = optimization.awq_smoke({}, execute=False, root=self.root)
            return self.smoke_result
        self.request_started_at = datetime.now(timezone.utc).isoformat()
        self.prepare()
        source = self.configs['awq_stretch' if self.model == 'nemotron' else 'awq_w4a16_cache_off']
        spec = {**source, 'prefix_cache': False}
        with self._native_server(spec, self.awq):
            self.smoke_result = optimization.awq_smoke(spec, self.pair, execute=True, root=self.root)
        self.smoke_result = self._capture('smoke', self.smoke_result)
        if self.smoke_result['status'] != 'completed':
            raise RuntimeError('AWQ smoke failed; result saved. All dependent variants remain blocked.')
        return self.smoke_result

    def _stretch_smoke(self):
        source, revision = self.config['awq_source'], self.config['awq_revision']
        if not source or not revision or not self.config['awq_derivation']:
            raise ValueError('Nemotron AWQ stretch needs an exact compatible source, revision and derivation attestation; completed core records are retained')
        if self.config['download_models']:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=source, revision=revision, local_dir=str(self.awq))
        self.pair = optimization.checkpoint_pair(self.base, self.awq,
            base_revision=f'{nemotron.MODEL_ID}@{self.base_revision}; {source}@{revision}; {self.config["awq_derivation"]}')
        spec = dict(self.configs['awq_stretch'])
        with self._native_server(spec, self.awq):
            self.smoke_result = optimization.awq_smoke(spec, self.pair, execute=True, root=self.root)
        if self.smoke_result['status'] != 'completed':
            raise RuntimeError('Nemotron AWQ smoke failed; completed core records are retained')

    def run(self, name):
        if not self.live:
            return self._run(name)
        self.request_started_at = datetime.now(timezone.utc).isoformat()
        self.results.pop(name, None)  # A failed rerun cannot retain an earlier success's records.
        print(f'Starting {name} | session {self.session_id}', flush=True)
        try:
            result = self._run(name)
            return self._capture(name, result)
        except BaseException as exc:
            failed = dict(self.results.get(name) or {}, status='failed', reason=str(exc),
                          configuration=self.configs[name], experiment=self.configs[name]['experiment'])
            self._capture(name, failed)
            self.close()
            raise

    def _run(self, name):
        spec = self.configs[name]
        if self.mode == 'replay':
            result = self.recorded(name)
            self.results[name] = result
            return result
        if not self.live:
            result = (nim_study.run(spec, root=self.root) if self.runtime == 'nim' else
                      nemotron.run(spec, root=self.root) if self.model == 'nemotron' else
                      optimization.run_variant(spec, root=self.root))
            self.request_started_at = datetime.now(timezone.utc).isoformat()
            return self._capture(name, result)
        self.prepare()
        if self.model == 'nemotron' and self.runtime == 'vllm' and (
            (name == 'awq_stretch' and not self.config['run_awq']) or
            (name == 'prefix_cache' and not self.config['prefix_supported'])):
            result = dict(run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'),
                experiment=spec['experiment'], configuration=spec, status='skipped', evidence='none',
                reason='Optional AWQ disabled or hybrid prefix support not verified in configuration')
            optimization.save_record(self.root, 'workflow/skipped', result)
            self.results[name] = result
            return result
        try:
            if self.runtime == 'nim':
                if name != 'smoke' and not (self.smoke_result or {}).get('smoke_passed'):
                    raise RuntimeError('Complete a successful NIM smoke before benchmark cells')
                self._nim_start()
                result = nim_study.run(spec, execute=True, root=self.root)
                if name == 'smoke':
                    self.smoke_result = result
            else:
                if self.model == 'qwen' and not (self.smoke_result or {}).get('status') == 'completed':
                    raise RuntimeError('Complete AWQ smoke before the four Qwen variants')
                if self.model == 'nemotron' and name != 'fp16_baseline' and self.results.get('fp16_baseline', {}).get('status') != 'completed':
                    raise RuntimeError('Complete the Nemotron FP16 baseline before core interventions')
                if name == 'awq_stretch' and self.smoke_result is None:
                    self._stretch_smoke()
                with self._native_server(spec, self.awq if spec['quantization'] else self.base):
                    result = (nemotron.run(spec, self.checkpoint, execute=True, root=self.root,
                        prefix_supported=self.config['prefix_supported'], pair=self.pair, smoke=self.smoke_result)
                        if self.model == 'nemotron' else optimization.run_variant(spec, self.pair,
                            smoke=self.smoke_result, execute=True, root=self.root))
            self.results[name] = result
            if result['status'] != 'completed':
                raise RuntimeError(result.get('reason', 'Experiment failed; inspect saved result'))
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        """Idempotent: never stop a PID/container not started by this Study."""
        if self.process is not None:
            process, self.process = self.process, None
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        self.server = None
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.container is not None:
            container = self.container
            try:
                logs = subprocess.run(['docker', 'logs', container], capture_output=True, text=True, timeout=30)
                self._path('nim-container', '.log').write_text(nim_study.safe_text(logs.stdout + logs.stderr))
            finally:
                stopped = subprocess.run(['docker', 'stop', '--time', '30', container], capture_output=True, text=True, timeout=60)
                if stopped.returncode and 'No such container' not in stopped.stderr:
                    raise RuntimeError('Docker cleanup failed: ' + nim_study.safe_text(stopped.stderr))
                self.container = None
        return 'Study servers stopped. Saved records and notebook outputs are retained.'

    def report(self):
        text = measurements_markdown([display_result(r) for r in self.current_results().values()])
        rows = ['| Experiment | Application E2E p50/p95 s | Request TTFT p95 s |', '|---|---|---|']
        def fmt(value):
            return '—' if value is None else f'{value:.3f}'
        for name, result in self.current_results().items():
            quality, serving = result.get('quality') or {}, result.get('serving') or {}
            label = name + (' · ' + LABEL if result.get('_recorded') else '')
            rows.append(f'| {label} | {fmt(quality.get("latency_p50_seconds"))}/{fmt(quality.get("latency_p95_seconds"))} | {fmt(serving.get("ttft_p95_seconds"))} |')
        return f'**Mode: {self.mode} · Session: {self.session_id}**\n\n' + text + '\n\n' + '\n'.join(rows) + '\n\nEvidence: ' + ', '.join(r.get('session_evidence_path', r.get('_recorded', {}).get('source', 'unavailable')) for r in self.current_results().values())

    def cache_report(self):
        selected = {name + (' · ' + LABEL if result.get('_recorded') else ''): result
                    for name, result in self.current_results().items()
                    if result.get('cache_study') or result.get('configuration', {}).get('prefix_cache')}
        return phase_table(selected) if selected else 'No controlled prefix-cache phase measurements in these results.'
