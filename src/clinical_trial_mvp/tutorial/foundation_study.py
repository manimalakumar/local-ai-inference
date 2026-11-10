"""Fresh chapter-00 baselines through the inherited clinical benchmark.

The caller starts/stops its server. Replay never contacts that server. Live uses
only newly collected rows, with failures saved rather than replaced by history.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from clinical_trial_mvp.serving.client import loopback_url
from clinical_trial_mvp.serving.portable import portable_value
from clinical_trial_mvp.tutorial import foundation
from clinical_trial_mvp.tutorial.harness import (
    TUTORIAL_CASES, measurements_markdown, serve_and_benchmark, validate_provenance,
)


def template():
    return dict(mode='replay', case_ids=list(TUTORIAL_CASES), repeats=3, warmups=1,
        max_tokens=1024, concurrency=[1, 2, 4], context_server='qwen',
        local_servers=[], gpu_servers=['qwen'],
        servers={'qwen':dict(runtime='vllm', host='cuda',
            model='clinical-trial-mvp3', base_url='http://127.0.0.1:8000',
            provenance='REPLACE_WITH_NEW_SERVER_METADATA.json',
            context=8192, dtype='float16', prefix_cache=False, quantization=None)})


def local_template():
    """Chapter 00 starts with a local GGUF endpoint; GPU studies are separate."""
    cfg = template()
    cfg.update(mode='live', context_server='nemotron', local_servers=['nemotron'], gpu_servers=[])
    cfg.pop('gpu_servers')
    cfg.pop('concurrency')
    cfg['servers'] = {'nemotron': dict(runtime='llama_cpp', model='clinical-trial-local',
        base_url='http://127.0.0.1:8080', provenance='REPLACE_WITH_NEW_LOCAL_SERVER_METADATA.json',
        context=8192, dtype='Q4_K_M')}
    return cfg


class FoundationStudy:
    def __init__(self, root, config=None):
        self.root = Path(root)
        path = self.root / 'tutorial/config/local-serving.local.json'
        self.config = dict(config if config is not None else
                           json.loads(path.read_text()) if path.exists() else local_template())
        self.mode = os.environ.get('LOCAL_AI_LAB_MODE', self.config.get('mode', 'live'))
        if self.mode not in {'replay', 'plan', 'live'}:
            raise ValueError('Chapter 00 mode must be replay, plan or live')
        self.results = {}
        self.session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.servers = {}
        if self.mode != 'replay':
            self._validate_config()

    def _validate_config(self):
        cfg = self.config
        for name in ('repeats', 'max_tokens'):
            if type(cfg.get(name)) is not int or cfg[name] < 1:
                raise ValueError(name + ' must be a positive integer')
        if type(cfg.get('warmups')) is not int or cfg['warmups'] < 0:
            raise ValueError('warmups must be a nonnegative integer')
        concurrency = cfg.get('concurrency', [1])
        if not isinstance(concurrency, list) or not concurrency or any(
                type(x) is not int or x < 1 for x in concurrency):
            raise ValueError('concurrency must be a list of positive integers')
        if not isinstance(cfg.get('case_ids'), list) or not cfg['case_ids']:
            raise ValueError('Select explicit case_ids; use all IDs for the full evaluation')
        for group in ('local_servers', 'gpu_servers'):
            if not isinstance(cfg.get(group, []), list):
                raise ValueError(group + ' must be a list of server names')
        selected = {cfg.get('context_server'), *cfg.get('local_servers', []), *cfg.get('gpu_servers', [])} - {None}
        if not selected:
            raise ValueError('Select at least one server for the fresh baseline')
        for name in selected:
            spec = cfg.get('servers', {}).get(name)
            if not isinstance(spec, dict):
                raise ValueError('Unknown configured server: ' + str(name))
            if spec.get('runtime') not in {'llama_cpp', 'ollama', 'vllm', 'sglang'}:
                raise ValueError('Unsupported baseline runtime')
            loopback_url(spec['base_url'])
            if name in cfg.get('local_servers', []) and spec['runtime'] not in {'llama_cpp', 'ollama'}:
                raise ValueError('local_servers selects llama_cpp/Ollama adapters')
            if name in cfg.get('gpu_servers', []) and spec['runtime'] not in {'vllm', 'sglang'}:
                raise ValueError('gpu_servers selects vLLM/SGLang adapters')
            if self.mode == 'live':
                path = Path(spec['provenance']).expanduser()
                path = path if path.is_absolute() else self.root / path
                if not path.is_file():
                    raise ValueError('Start the server and supply its NEW launch metadata: ' + str(path))
                server = json.loads(path.read_text())
                validate_provenance(spec, server)
                if server.get('evidence') != 'hardware':
                    raise ValueError('Live baseline requires actual hardware launch metadata')
                self.servers[name] = server
        # Validate frozen data and selected questions before sending any model request.
        foundation.inventory(self.root)
        from clinical_trial_mvp.serving.benchmark import Workload, prepare
        for profile in ('full', 'retrieved'):
            prepare(self.root, Workload(profile=profile), cfg['case_ids'])

    def _run(self, group, name, profile='full', concurrency=1):
        if self.mode == 'replay':
            raise ValueError('Replay does not execute fresh baseline rows')
        key = f'{group}/{name}/{profile}/c{concurrency}'
        if key in self.results:
            old = self.results[key]
            if self.mode == 'live' and old['status'] != 'completed':
                raise RuntimeError('Previous row failed; fix the server/config and restart the kernel: ' + old['output_path'])
            return old
        spec = dict(self.config['servers'][name], experiment='Baseline/' + key,
            profile=profile, concurrency=concurrency, repeats=self.config['repeats'],
            warmups=self.config['warmups'], max_tokens=self.config['max_tokens'],
            hardware=self.servers.get(name, {}).get('hardware', 'pending local launch'))
        result = serve_and_benchmark(spec, execute=self.mode == 'live', root=self.root,
            case_ids=self.config['case_ids'], server=self.servers.get(name), include_policy_cases=True)
        result['baseline_mode'] = self.mode
        if self.mode == 'live':
            path = Path('experiment-tracker/tutorial/foundation') / self.session_id / (
                hashlib.sha256(key.encode()).hexdigest()[:16] + '.json')
            result['output_path'] = path.as_posix()
            destination = self.root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('x') as output:
                json.dump(portable_value(result), output, indent=2, default=str)
                output.write('\n')
        self.results[key] = result
        if self.mode == 'live' and (result['status'] != 'completed' or result.get('evidence') != 'hardware'):
            raise RuntimeError('Fresh baseline failed; saved ' + result['output_path'] + ': ' + result.get('reason', result['status']))
        return result

    def context_rows(self):
        name = self.config['context_server']
        return [self._run('context', name, profile) for profile in ('full', 'retrieved')] if name else []

    def local_rows(self):
        return [self._run('local-serving', name) for name in self.config['local_servers']]

    def gpu_rows(self):
        return [self._run('concurrency', name, concurrency=c)
                for name in self.config.get('gpu_servers', []) for c in self.config.get('concurrency', [1])]

    def table(self, rows):
        if not rows:
            return ('**Not selected for this run.** Add compatible running endpoints to the local '
                    'configuration to explore this comparison.')
        return '**' + ('Fresh measurements' if self.mode == 'live' else 'Plan only — no inference') + '**\n\n' + measurements_markdown(rows)

    def answer(self):
        name = self.config['context_server']
        if not name:
            return dict(status='not_selected', reason='Select context_server for the full-schema SQL example')
        result = self._run('context', name)
        records = result.get('records', {}).get('quality', [])
        return dict(origin='fresh baseline' if self.mode == 'live' else 'plan only',
            run_id=result.get('run_id'), output_path=result.get('output_path'),
            configuration=result['configuration'],
            application_record=records[0] if records else None,
            note='Full-schema example from this run: inspect the prompt, plan, SQL attempts and result scores.')
