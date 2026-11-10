"""Nemotron core interventions; AWQ is independently gated stretch work."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import re
import shlex

from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.evaluation.dataset import sha256_file
from clinical_trial_mvp.serving.benchmark import Workload, prepare, run_quality, run_serving, snapshot_fingerprint
from clinical_trial_mvp.serving.client import Generation, OpenAIClient
from clinical_trial_mvp.serving.portable import portable_value
from clinical_trial_mvp.tutorial.cache_study import measure_requests, ready, sql_score
from clinical_trial_mvp.tutorial.harness import ROOT, TUTORIAL_CASES, semantic_scores, validate_provenance
from clinical_trial_mvp.tutorial.memory import PeakMemory
from clinical_trial_mvp.tutorial.optimization import checkpoint_pair, check_smoke, save_record

MODEL_ID = 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'


def configurations(run_tag='nemotron-run01', *, repeats=5):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_tag):
        raise ValueError('Use a simple unique run tag')
    baseline = dict(runtime='vllm', host='cuda', hardware='GPU host (verify SKU)',
                    model='clinical-trial-mvp3', model_id=MODEL_ID,
                    base_url='http://127.0.0.1:8000', context=8192, dtype='float16',
                    profile='full', quantization=None, prefix_cache=False,
                    concurrency=1, max_num_seqs=4, max_tokens=1024, reasoning=False,
                    repeats=repeats, warmups=1, mamba_ssm_cache_dtype='float32', trust_remote_code=True)
    interventions = {
        'fp16_baseline': {},
        'prefix_cache': {'prefix_cache': True},
        'schema_retrieved': {'profile': 'retrieved'},
        'concurrency_2': {'concurrency': 2},
        'concurrency_4': {'concurrency': 4},
        'batch_capacity_1': {'concurrency': 4, 'max_num_seqs': 1},
        'batch_capacity_2': {'concurrency': 4, 'max_num_seqs': 2},
        'reasoning_on': {'reasoning': True},
        'output_256': {'max_tokens': 256},
        'output_2048': {'max_tokens': 2048},
        'awq_stretch': {'quantization': 'awq', 'priority': 'stretch'},
    }
    return {name: {**baseline, 'priority': 'core', **delta,
                   'experiment': 'MVP4/Nemotron/' + name,
                   'provenance': f'experiment-tracker/mvp4/provenance/{run_tag}-{name}.json'}
            for name, delta in interventions.items()}


def baseline_checkpoint(snapshot, *, revision):
    """Core needs only an unquantized Nemotron snapshot; never consult AWQ files."""
    if not revision or revision.startswith('SET_'):
        raise ValueError('Set the exact Nemotron base revision')
    snapshot = Path(snapshot)
    config = json.loads((snapshot / 'config.json').read_text())
    if config.get('quantization_config'):
        raise ValueError('Core baseline must be unquantized')
    if config.get('model_type') != 'nemotron_h':
        raise ValueError('Expected the Nemotron hybrid architecture (nemotron_h)')
    return {'model_id': MODEL_ID, 'base_revision': revision,
            'model_sha256': snapshot_fingerprint(snapshot)}


def launch_preview(spec, snapshot, *, python='.venv-vllm/bin/python'):
    cmd = [python, 'scripts/serving/launch_nemotron.py', '--snapshot', str(snapshot),
           '--metadata', spec['provenance'], '--context', str(spec['context']),
           '--max-num-seqs', str(spec['max_num_seqs']), '--execute']
    if spec['prefix_cache']:
        cmd += ['--prefix-cache']
    if spec['quantization']:
        cmd += ['--quantization', spec['quantization']]
    return shlex.join(cmd)


class NemotronClient(OpenAIClient):
    """Use native chat-template thinking control and remove conflicting /no_think."""
    def __init__(self, *args, reasoning=False, **kwargs):
        super().__init__(*args, enable_thinking=reasoning, **kwargs)
        self.reasoning = reasoning

    def complete(self, messages, schema=None):
        if self.reasoning:
            messages = [{**m, 'content': m['content'].replace('/no_think', '/think')}
                        for m in messages]
            # Keep thought + final JSON in content; no version-specific reasoning
            # parser plugin is assumed. Parse/scoring happen after the timed call.
            schema = None
        return super().complete(messages, schema)


def final_answer(sample, *, reasoning=False):
    if not sample.success:
        return sample
    if reasoning and '</think>' not in sample.content:
        return replace(sample, success=False, error='Reasoning did not finish with a closing think marker')
    # Some templates put the opening marker in the prompt, so the completion
    # contains only the closing marker. Discard the complete reasoning prefix.
    text = sample.content.rsplit('</think>', 1)[-1].strip() if '</think>' in sample.content else sample.content.strip()
    if '<think>' in text:
        return replace(sample, success=False, error='Reasoning did not finish within output limit')
    return replace(sample, content=text)


class AnswerClient:
    """Normalize final JSON only for the inherited clinical planning/repair phase."""
    def __init__(self, client):
        self.client, self.generation = client, client.generation
    def complete(self, messages, schema=None):
        return final_answer(self.client.complete(messages, schema), reasoning=getattr(self.client, 'reasoning', False))


def validate_server(spec, checkpoint, server):
    if not checkpoint:
        raise ValueError('Validate the baseline checkpoint first')
    validate_provenance(spec, server)
    if server['model_sha256'] != checkpoint['model_sha256']:
        raise ValueError('Server fingerprint differs from the validated checkpoint')
    for key in ('max_num_seqs', 'mamba_ssm_cache_dtype', 'trust_remote_code'):
        if key not in spec:
            continue
        if server['launch_settings'].get(key) != spec[key]:
            raise ValueError('Server setting mismatch: ' + key)
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)', server['engine_version'])
    if spec.get('model_id') == MODEL_ID and (not match or tuple(map(int, match.groups())) < (0, 15, 1)):
        raise ValueError('Nemotron model card requires vLLM >= 0.15.1')
    if not server.get('gpu_uuid'):
        raise ValueError('Record the actual GPU UUID for memory sampling')


def run(spec, checkpoint=None, *, execute=False, root=ROOT, prefix_supported=False,
        pair=None, smoke=None, client=None, server=None, memory_factory=None):
    # Shared clinical scoring also serves Qwen's single-control interventions.
    category = 'nemotron' if spec.get('model_id') == MODEL_ID else 'qwen-controls'
    root = Path(root)
    workload = Workload(profile=spec['profile'], concurrency=spec['concurrency'],
                        repeats=spec['repeats'], warmups=spec['warmups'])
    cases, contexts, prompts, inputs = prepare(root, workload, TUTORIAL_CASES)
    result = dict(run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'),
                  experiment=spec['experiment'], configuration=spec, runtime='vllm',
                  hardware=spec['hardware'], inputs=inputs, status='plan_only', evidence='none',
                  quality=None, serving=None, memory={'peak_bytes': None, 'scope': 'unmeasured'}, records={})
    if not execute:
        return result
    # An unsupported cache experiment never blocks the remaining core runs.
    if spec['prefix_cache'] and not prefix_supported:
        result.update(status='skipped', reason='Verify hybrid prefix-cache support on the installed build/backend first')
        save_record(root, category, result)
        return result
    if spec['quantization'] and (not pair or not smoke):
        result.update(status='skipped', reason='AWQ stretch needs a validated compatible checkpoint pair and successful smoke')
        save_record(root, category, result)
        return result
    injected = client is not None
    try:
        server = server or json.loads((root / spec['provenance']).read_text())
        active_checkpoint = checkpoint
        if spec['quantization']:
            if not checkpoint or pair.get('fp16_sha256') != checkpoint['model_sha256']:
                raise ValueError('AWQ pair differs from the validated core baseline')
            active_checkpoint = {**checkpoint, 'model_sha256': pair['awq_sha256']}
            check_smoke(smoke, pair, server)
        validate_server(spec, active_checkpoint, server)
        if not injected and platform.system() != 'Linux':
            raise ValueError('Live runs require the Linux GPU host')
        result.update(server=server, checkpoint=active_checkpoint,
                      database_sha256=sha256_file(root / 'data/db/clinical_trials.duckdb'))
        import duckdb
        with duckdb.connect() as connection:
            result['evaluation_date'] = str(connection.execute('SELECT CURRENT_DATE').fetchone()[0])
        client = client or NemotronClient(spec['base_url'], spec['model'],
            Generation(max_tokens=spec['max_tokens']), reasoning=spec['reasoning'])
        if not injected:
            ready(client)
        factory = memory_factory or (lambda: PeakMemory(gpu_uuid=server['gpu_uuid']))
        settings = replace(Settings.from_env(), project_root=root,
                           db_path=root / 'data/db/clinical_trials.duckdb')
        quality_records, serving_records = [], []
        if spec['prefix_cache']:
            result['cache_study'] = measure_requests(spec, client, root=root,
                gpu_uuid=server['gpu_uuid'], memory_factory=factory)
        with factory() as memory:
            result['serving'] = run_serving(client, cases, prompts, workload, serving_records.append)
            result['quality'] = run_quality(AnswerClient(client), settings, cases, contexts,
                replace(workload, mode='quality', concurrency=1, repeats=1, warmups=0), quality_records.append)
        result['memory'] = memory.result()
        # Score each timed direct completion outside its transport clock. Records
        # retain raw thoughts; final JSON is scored after removing complete tags.
        from clinical_trial_mvp.data.db import execute_readonly_sql
        references = {}
        for case in cases:
            ref = execute_readonly_sql(case['reference_sql'], settings)
            if not ref.ok:
                raise ValueError('Reference SQL failed')
            references[case['id']] = asdict(ref)
        from clinical_trial_mvp.serving.client import Sample
        by_id = {c['id']: c for c in cases}
        for record in serving_records:
            sample = Sample(**{k: record[k] for k in Sample.__dataclass_fields__ if k in record})
            record.update(sql_score(final_answer(sample, reasoning=spec['reasoning']), by_id[record['case_id']], references[record['case_id']], settings))
        reviews_path = root / 'tutorial/semantic_reviews.json'
        reviews = json.loads(reviews_path.read_text()).get('reviews', []) if reviews_path.exists() else []
        result['serving']['result_equivalent'] = sum(r['scores']['result_equivalence'] is True for r in serving_records)
        result['serving']['reference_scored'] = len(serving_records)
        result['serving']['semantic'] = semantic_scores(serving_records, reviews)
        result['quality']['semantic'] = semantic_scores(quality_records, reviews)
        result['quality']['sql_executed'] = sum(bool(r['attempts'] and r['attempts'][-1].get('execution_success')) for r in quality_records)
        result['records'] = {'quality': quality_records, 'serving': serving_records}
        result.update(status='completed', evidence='hardware' if not injected and server.get('evidence') == 'hardware' else 'test_or_unverified')
        result['ttft_definition'] = 'First generated content, including thought text when reasoning is on; not final-answer TTFT'
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        result.update(status='failed', reason=str(exc))
    if not injected:
        save_record(root, category, result)
    return portable_value(result)
