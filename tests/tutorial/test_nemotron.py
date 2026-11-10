import copy
import json
from pathlib import Path

import pytest

from clinical_trial_mvp.tutorial import nemotron as n
from clinical_trial_mvp.serving.client import Sample, Generation, OpenAIClient
from clinical_trial_mvp.serving.benchmark import Workload, prepare
from clinical_trial_mvp.tutorial.harness import ROOT, TUTORIAL_CASES
from clinical_trial_mvp.tutorial.memory import PeakMemory


def server(spec):
    return {'runtime': 'vllm', 'model': spec['model'], 'engine_version': '0.31.0',
            'model_sha256': 'a' * 64, 'hardware': 'test GPU', 'gpu_uuid': 'GPU-test',
            'evidence': 'hardware', 'launch_settings': {k: spec[k] for k in
                ('context', 'dtype', 'prefix_cache', 'quantization', 'max_num_seqs',
                 'trust_remote_code', 'mamba_ssm_cache_dtype')}}


PLAN = {'population': 'Enrolled or Randomized subjects', 'measure': 'count subjects',
        'grouping': 'treatment arm', 'time_window': 'none', 'filters': 'none', 'ranking': 'none',
        'needs_clarification': False, 'clarification_question': ''}


class Fake:
    generation = Generation()
    reasoning = False
    def __init__(self):
        self.calls = []
        self.cases = prepare(ROOT, Workload(profile='full'), TUTORIAL_CASES)[0]
    def complete(self, messages, schema=None):
        self.calls.append(messages)
        if schema and 'population' in schema.get('properties', {}):
            content = json.dumps(PLAN)
        else:
            prompt = messages[-1]['content']
            case = next((c for c in self.cases if c['question'] in prompt), self.cases[0])
            content = json.dumps({'sql': case['reference_sql'], 'rationale': ''})
        return Sample(success=True, content=content, prompt_tokens=120, output_tokens=12,
                      ttft_seconds=.1, e2e_seconds=.2)


def test_core_configurations_isolate_interventions_and_awq_is_stretch():
    configs = n.configurations()
    baseline = configs['fp16_baseline']
    assert baseline['quantization'] is None and not baseline['prefix_cache']
    assert baseline['dtype'] == 'float16' and baseline['mamba_ssm_cache_dtype'] == 'float32'
    assert configs['schema_retrieved']['profile'] == 'retrieved'
    assert [configs[k]['concurrency'] for k in ('fp16_baseline','concurrency_2','concurrency_4')] == [1,2,4]
    a, b = configs['concurrency_4'], configs['batch_capacity_1']
    ignored = {'experiment', 'provenance', 'max_num_seqs'}
    assert {k:v for k,v in a.items() if k not in ignored} == {k:v for k,v in b.items() if k not in ignored}
    assert (a['max_num_seqs'], b['max_num_seqs']) == (4, 1)
    assert configs['reasoning_on']['reasoning']
    assert [configs[k]['max_tokens'] for k in ('output_256','fp16_baseline','output_2048')] == [256,1024,2048]
    assert [k for k,s in configs.items() if s['priority'] == 'stretch'] == ['awq_stretch']
    assert all(s['quantization'] is None for s in configs.values() if s['priority'] == 'core')


def test_unquantized_baseline_validation_does_not_require_awq(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type':'nemotron_h'}))
    (tmp_path / 'model.safetensors').write_bytes(b'fixture')
    checked = n.baseline_checkpoint(tmp_path, revision='exact-base-revision')
    assert checked['model_id'] == n.MODEL_ID and len(checked['model_sha256']) == 64
    with pytest.raises(ValueError, match='exact'):
        n.baseline_checkpoint(tmp_path, revision='SET_REVISION')
    (tmp_path / 'config.json').write_text(json.dumps({'model_type':'qwen3'}))
    with pytest.raises(ValueError, match='hybrid'):
        n.baseline_checkpoint(tmp_path, revision='exact')


@pytest.mark.parametrize('name', list(n.configurations()))
def test_every_configuration_plans_without_awq_or_network(name, monkeypatch):
    monkeypatch.setattr(n, 'OpenAIClient', lambda *a, **kw: pytest.fail('offline called server'))
    result = n.run(n.configurations()[name], root=ROOT)
    assert result['status'] == 'plan_only' and result['evidence'] == 'none'
    assert result['quality'] is None and result['memory']['peak_bytes'] is None
    assert result['inputs']['case_ids'] == list(TUTORIAL_CASES)


@pytest.mark.parametrize('name', ['fp16_baseline','schema_retrieved','concurrency_2','concurrency_4','batch_capacity_1','output_256','output_2048'])
def test_core_runs_real_clinical_scoring_without_an_awq_checkpoint(name, monkeypatch):
    spec = n.configurations(repeats=1)[name]
    monkeypatch.setattr(n, 'checkpoint_pair', lambda *a, **kw: pytest.fail('Core must not check AWQ'))
    fake = Fake()
    result = n.run(spec, {'model_sha256':'a'*64}, execute=True, root=ROOT, client=fake,
                   server=server(spec), memory_factory=lambda: PeakMemory(reader=lambda: 100))
    assert result['status'] == 'completed', result.get('reason')
    assert result['evidence'] == 'test_or_unverified'
    assert result['serving']['result_equivalent'] == 4
    assert result['quality']['cases'] == 4 and result['records']['quality']
    assert result['memory']['peak_bytes'] == 100
    assert result['serving']['semantic']['total'] == 4


def test_unsupported_cache_and_missing_awq_skip_without_affecting_core(monkeypatch):
    saved = []
    monkeypatch.setattr(n, 'save_record', lambda *args: saved.append(args[-1]))
    for name in ('prefix_cache','awq_stretch'):
        result = n.run(n.configurations()[name], execute=True, root=ROOT)
        assert result['status'] == 'skipped' and result['quality'] is None
    assert len(saved) == 2
    assert n.run(n.configurations()['fp16_baseline'])['status'] == 'plan_only'


def test_supported_cache_collects_separate_phases(monkeypatch):
    spec = n.configurations(repeats=1)['prefix_cache']
    monkeypatch.setattr(n, 'measure_requests', lambda *a, **kw: {'phases': {'cold': {}, 'warm': {}}})
    result = n.run(spec, {'model_sha256':'a'*64}, execute=True, root=ROOT, prefix_supported=True,
                   client=Fake(), server=server(spec), memory_factory=lambda: PeakMemory(reader=lambda: 100))
    assert result['status'] == 'completed', result.get('reason')
    assert set(result['cache_study']['phases']) == {'cold', 'warm'}


@pytest.mark.parametrize('change', [
    {'engine_version':'0.15.0'}, {'model_sha256':'b'*64}, {'gpu_uuid':None},
    {'launch_settings': {'dtype':'float16','context':8192,'prefix_cache':False,'quantization':None}},
])
def test_provenance_checks_actual_hybrid_settings_version_and_weights(change):
    spec = n.configurations()['fp16_baseline']
    with pytest.raises(ValueError):
        n.validate_server(spec, {'model_sha256':'a'*64}, {**server(spec), **change})


def test_reasoning_changes_template_and_conflicting_prompt_directive(monkeypatch):
    captured = []
    def complete(self, messages, schema=None):
        captured.append((self.enable_thinking,messages,schema))
        return Sample(success=True,content='thought</think>{"sql":"SELECT 1"}')
    monkeypatch.setattr(OpenAIClient,'complete',complete)
    client = n.NemotronClient('http://127.0.0.1:8000','alias',reasoning=True)
    sample = client.complete([{'role':'system','content':'/no_think rules'}], {'json':'schema'})
    assert captured == [(True,[{'role':'system','content':'/think rules'}],None)]
    assert n.final_answer(sample, reasoning=True).content == '{"sql":"SELECT 1"}'
    assert sample.content.startswith('thought')  # Raw completion retained.
    assert not n.final_answer(Sample(success=True,content='unfinished thought'),reasoning=True).success


def test_reasoning_wire_payload_uses_requested_template_control():
    sent = []
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __iter__(self):
            event = {'choices':[{'index':0,'delta':{'content':'thought</think>{"sql":"SELECT 1"}'},
                                 'finish_reason':'stop'}], 'usage':{'prompt_tokens':3,'completion_tokens':8}}
            yield ('data: '+json.dumps(event)+'\n').encode()
            yield b'\n'
            yield b'data: [DONE]\n'
            yield b'\n'
    def opener(request, **kwargs):
        sent.append(json.loads(request.data))
        return Response()
    client = n.NemotronClient('http://127.0.0.1:8000','alias',reasoning=True,opener=opener)
    sample = client.complete([{'role':'system','content':'/no_think rules'}], {'type':'object'})
    assert sample.success
    assert sent[0]['chat_template_kwargs'] == {'enable_thinking':True}
    assert sent[0]['messages'][0]['content'] == '/think rules'
    assert 'response_format' not in sent[0]
    assert sample.output_tokens == 8


def test_reasoning_experiment_retains_raw_output_and_scores_final_json():
    class ReasoningFake(Fake):
        reasoning = True
        def complete(self, *args, **kwargs):
            sample = super().complete(*args, **kwargs)
            sample.content = 'internal thought</think>' + sample.content
            return sample
    spec = n.configurations(repeats=1)['reasoning_on']
    result = n.run(spec, {'model_sha256':'a'*64}, execute=True, root=ROOT, client=ReasoningFake(),
                   server=server(spec), memory_factory=lambda: PeakMemory(reader=lambda: 100))
    assert result['status'] == 'completed', result.get('reason')
    assert result['serving']['result_equivalent'] == 4
    assert all(r['content'].startswith('internal thought') for r in result['records']['serving'])
    assert all(r['attempts'][0]['sql'].startswith(('SELECT','WITH')) for r in result['records']['serving'])


def test_launcher_records_distinct_scheduler_capacity_and_fp32_ssm():
    from clinical_trial_mvp.serving.launch import launch_command
    baseline = n.configurations()['fp16_baseline']
    preview = n.launch_preview(baseline,Path('models/snapshot with spaces'))
    assert '--max-num-seqs 4' in preview and 'launch_nemotron.py' in preview
    cmd = launch_command('vllm', 'snapshot', dtype='float16', max_num_seqs=1,
                         trust_remote_code=True, mamba_ssm_cache_dtype='float32')
    assert cmd[cmd.index('--max-num-seqs')+1] == '1'
    assert '--trust-remote-code' in cmd
    assert cmd[cmd.index('--mamba-ssm-cache-dtype')+1] == 'float32'


@pytest.mark.parametrize('model', ['qwen', 'nemotron'])
def test_capacity_series_changes_only_scheduler_limit_at_four_clients(model):
    from clinical_trial_mvp.tutorial import optimization
    configs = optimization.study_configurations() if model == 'qwen' else n.configurations()
    reference = configs['concurrency_4']
    ignored = {'experiment', 'provenance', 'max_num_seqs'}
    for name, capacity in [('batch_capacity_1', 1), ('batch_capacity_2', 2), ('concurrency_4', 4)]:
        spec = configs[name]
        assert spec['concurrency'] == 4 and spec['max_num_seqs'] == capacity
        assert {k:v for k,v in spec.items() if k not in ignored} == {k:v for k,v in reference.items() if k not in ignored}


@pytest.mark.parametrize('name', ['schema_retrieved', 'concurrency_2', 'concurrency_4',
                                'batch_capacity_1', 'batch_capacity_2', 'reasoning_on',
                                'output_256', 'output_2048'])
def test_qwen_controls_score_real_clinical_workload_without_hybrid_requirements(name):
    from clinical_trial_mvp.tutorial import optimization
    spec = optimization.study_configurations(repeats=1)[name]
    class ThinkingFake(Fake):
        reasoning = spec['reasoning']
        def complete(self, *args, **kwargs):
            sample = super().complete(*args, **kwargs)
            if self.reasoning:
                sample.content = 'thought</think>' + sample.content
            return sample
    metadata = {'runtime': 'vllm', 'model': spec['model'], 'engine_version': '0.32.0',
                'model_sha256': 'a'*64, 'hardware': 'test GPU', 'gpu_uuid': 'GPU-test',
                'evidence': 'hardware', 'launch_settings': {k: spec[k] for k in
                    ('context','dtype','prefix_cache','quantization','max_num_seqs')}}
    result = n.run(spec, {'model_sha256':'a'*64}, execute=True, root=ROOT,
                   client=ThinkingFake(), server=metadata,
                   memory_factory=lambda: PeakMemory(reader=lambda: 100))
    assert result['status'] == 'completed', result.get('reason')
    assert result['serving']['result_equivalent'] == 4
    assert result['quality']['result_equivalent'] == 4
    assert result['memory']['peak_bytes'] == 100
    if spec['reasoning']:
        assert all(r['content'].startswith('thought') for r in result['records']['serving'])
