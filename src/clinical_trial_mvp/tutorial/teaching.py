"""Three visible mechanisms and guided interpretation for the inference lab."""
import json
from pathlib import Path
import shlex

from clinical_trial_mvp.agent.analyst import PLAN_CHAT_SYSTEM_MESSAGE, PLAN_JSON_SCHEMA, build_plan_prompt
from clinical_trial_mvp.serving.client import Generation, OpenAIClient
from clinical_trial_mvp.serving.launch import launch_command
from clinical_trial_mvp.tutorial import nim_study
from clinical_trial_mvp.tutorial.replay import LABEL, evidence_links


def _record(study, name):
    return study.current_results().get(name) or (study.recorded(name) if study.mode == 'replay' else {})


def server_example(study, name):
    result = _record(study, name)
    spec = result.get('configuration') or study.configs[name]
    server = result.get('server') or {}
    settings = server.get('launch_settings') or spec
    if study.runtime == 'nim' and not server:
        settings = nim_study.template(study.model)['launch_settings']
    details = dict(origin=LABEL if result.get('_recorded') else study.mode,
                   model=spec.get('model_id', nim_study.MODELS[study.model]['model_id']),
                   revision=(result.get('checkpoint_pair') or {}).get('base_revision_attestation') or
                            (result.get('checkpoint') or {}).get('base_revision') or
                            server.get('model_revision') or study.config.get('base_revision') or 'not set locally',
                   dtype=settings.get('dtype'), prefix_cache=settings.get('prefix_cache'),
                   context=settings.get('context'), scheduler_max_num_seqs=settings.get('max_num_seqs',
                       'runtime default; not explicitly recorded in this native Qwen run'))
    if server.get('command'):
        details.update(command=shlex.join(server['command']),
                       command_origin='recorded launch argv; original recorder made paths portable')
    elif study.runtime == 'nim':
        image = server.get('image') or study.config['nim_image'] or '<digest pending local Docker inspection>'
        profile = server.get('profile_id') or study.config['nim_profile_id'] or '<choose after discovery>'
        details.update(image=image, profile_id=profile, command_origin='reconstructed deployment shape; raw Docker argv and original host snapshot path were not saved',
            command=f'docker run --gpus device=0 --mount type=bind,source=<local-checkpoint>,target=/model,readonly '
                    f'-e NIM_MODEL_PATH=/model -e NIM_MODEL_PROFILE={profile} {image} '
                    f'--dtype float16 --max-model-len {settings["context"]} --max-num-seqs {settings["max_num_seqs"]} --no-enable-prefix-caching')
        if study.model == 'nemotron':
            details['command'] += ' --trust-remote-code --mamba-ssm-cache-dtype float32'
    else:
        snapshot = study.awq if spec.get('quantization') else study.base
        options = {key: spec[key] for key in ('context', 'dtype', 'prefix_cache', 'quantization',
                   'max_num_seqs', 'trust_remote_code', 'mamba_ssm_cache_dtype') if key in spec}
        argv = launch_command('vllm', snapshot, **options)
        argv[0] = str(Path(study.config['vllm_python']).expanduser())
        details.update(command='VLLM_SERVER_DEV_MODE=1 ' + shlex.join(argv), command_origin='planned argv; not executed')
    return details


def request_example(study, name):
    result = _record(study, name)
    spec = result.get('configuration') or study.configs[name]
    quality = result.get('records', {}).get('quality', [])
    prompt = quality[0].get('plan_prompt') if quality else None
    cases = json.loads((study.root / 'evals/evaluations.json').read_text())
    if not prompt:
        from clinical_trial_mvp.serving.benchmark import Workload, prepare
        case_id = cases[0]['id']
        selected, contexts, _, _ = prepare(study.root, Workload(profile=spec.get('profile', 'full')), [case_id])
        question = selected[0]['question']
        schema_text, _, guidance = contexts[question]
        prompt = build_plan_prompt(question, schema_text, guidance)
    system = PLAN_CHAT_SYSTEM_MESSAGE
    schema = PLAN_JSON_SCHEMA if study.runtime == 'vllm' or spec.get('structured_outputs') else None
    if spec.get('reasoning'):
        prompt, system, schema = prompt.replace('/no_think', '/think'), system.replace('/no_think', '/think'), None
    client = OpenAIClient(spec['base_url'], spec['model'], Generation(max_tokens=spec['max_tokens']),
                         request_profile='nim' if study.runtime == 'nim' else 'vllm',
                         enable_thinking=spec.get('reasoning', False))
    return dict(origin=LABEL if result.get('_recorded') else study.mode,
                session_id=study.session_id, note='Reconstructed with the current request builder, selected schema profile and generation settings; not a captured wire payload. Replay explicitly uses its saved prompt and settings.',
                method='POST', url=client.base_url + '/chat/completions',
                headers={'Content-Type': 'application/json'},
                json=client.request_payload([{'role': 'system', 'content': system},
                                            {'role': 'user', 'content': prompt}], schema))


def response_example(study, name, case_id='eval_001'):
    result = _record(study, name)
    records = result.get('records', {}).get('quality', [])
    record = next((r for r in records if r['case_id'] == case_id), None)
    if record is None:
        return dict(origin=study.mode, status='No response: this plan has not executed. No SQL or scores fabricated.')
    case = next(c for c in json.loads((study.root / 'evals/evaluations.json').read_text()) if c['id'] == case_id)
    import hashlib
    from clinical_trial_mvp.tutorial.reviews import validate_reviews
    path = study.root / 'tutorial/semantic_reviews.json'
    reviews = validate_reviews(json.loads(path.read_text())['reviews']) if path.exists() else {}
    sql = ((record.get('attempts') or [{}])[-1].get('sql') or '').strip()
    review = reviews.get((case_id, hashlib.sha256(sql.encode()).hexdigest())) if sql else None
    status = ('human review: ' + ('correct' if review['correct'] else 'incorrect') +
              ' — ' + review['reviewer'] + ', ' + review['reviewed_at']) if review else (
              'unreviewed: reference-result match is not human semantic validation')
    return dict(origin=LABEL if result.get('_recorded') else study.mode, case_id=case_id, question=case['question'],
                plan=record.get('plan'), plan_error=record.get('plan_error'), attempts=record.get('attempts'),
                reference_sql=case['reference_sql'], recorded_comparison=record.get('scores'),
                semantic_review=status, human_review=review)


def interpret(study, name):
    result = _record(study, name)
    spec = result.get('configuration') or study.configs[name]
    quality, serving = result.get('quality') or {}, result.get('serving') or {}
    base_name = 'baseline' if study.runtime == 'nim' else 'fp16_baseline' if study.model == 'nemotron' else 'fp16_cache_off'
    baseline = _record(study, base_name)
    baseline_quality = baseline.get('quality') or {}
    def num(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    phase = (result.get('cache_study') or {}).get('phases', {})
    prediction = 'Establish executable SQL and separate application quality from request speed.'
    explanation = 'Transport timing and application correctness measure different stages.'
    decision = 'Use this baseline to judge useful completed tasks, not latency alone.'
    observation = (f'Application result match: {quality.get("result_equivalent", "—")}/{quality.get("reference_scored", "—")}; '
                   f'SQL executed: {quality.get("sql_executed", "—")}/{quality.get("cases", "—")}. '
                   f'Request E2E p50/p95: {num(serving.get("e2e_p50_seconds"))}/{num(serving.get("e2e_p95_seconds"))} s; '
                   f'aggregate output: {num(serving.get("output_tokens_per_second"))} tok/s.')
    if 'cold' in phase and 'warm' in phase:
        cold, warm = phase['cold'], phase['warm']
        prediction = 'Shared-prefix reuse should reduce warm prefill latency more than total generation time.'
        observation = (f'Cold → warm TTFT p50: {num(cold["ttft_p50_seconds"])} → {num(warm["ttft_p50_seconds"])} s. '
                       f'Request E2E p50: {num(cold["e2e_p50_seconds"])} → {num(warm["e2e_p50_seconds"])} s. '
                       f'Direct SQL result match: {cold["result_equivalent"]}/{cold["reference_scored"]} cold; '
                       f'{warm["result_equivalent"]}/{warm["reference_scored"]} warm.')
        explanation = 'The paired phase records are consistent with avoiding repeated prefill; generation still consumes time.'
        decision = 'Consider caching for repeated schema/system prefixes after checking both phase quality counts.'
    elif name == 'schema_retrieved':
        prediction = 'Less context may reduce input work but can remove definitions required for correct SQL.'
        observation += f' Baseline application result match: {baseline_quality.get("result_equivalent", "—")}/{baseline_quality.get("reference_scored", "—")}.'
        explanation = 'Context reduction changes the information available to the model, not just compute cost.'
        decision = 'Retain needed clinical definitions; accept reduced context only with adequate task quality.'
    elif name.startswith('concurrency') or name == 'batch_capacity_1':
        prediction = 'More overlapping clients may improve aggregate throughput while increasing per-request latency.'
        explanation = 'Client overlap and scheduler capacity are separate controls; these records do not show achieved batch occupancy.'
        decision = 'Choose concurrency using both latency limits and correctly completed work.'
    elif name.startswith('awq'):
        prediction = 'Four-bit weights reduce weight storage; latency and SQL quality may change in either direction.'
        explanation = 'Quantization changes weight representation and kernel behavior; caching addresses prefix reuse.'
        decision = 'Benchmark compatible AWQ when weight capacity matters; no native VRAM reduction is established here.'
    elif name == 'reasoning_on' or name.startswith('output_'):
        prediction = 'Reasoning/output budgets change generation work and may affect task completion.'
        explanation = 'Thoughts consume output tokens; TTFT includes thoughts, and truncated answers remain failures.'
        decision = 'Use the smallest budget that preserves required task quality; do not optimize output tok/s alone.'
    if result.get('status') not in ('completed',):
        observation = result.get('reason', 'No measurement has been executed in plan mode.')
    boundary = ('Four distinct questions, not 20 independent clinical tasks. Repeated requests give illustrative latency summaries; '
                'Human review coverage is separate from machine reference matches; consult the review packet. Memory conclusions require valid telemetry. These observations do not generalize to other domains or stacks.')
    return '\n\n'.join(filter(None, [evidence_links(result), '**Prediction:** ' + prediction,
        '**Observation:** ' + observation, '**Explanation:** ' + explanation,
        '**Limitation:** ' + boundary, '**Decision:** ' + decision]))
