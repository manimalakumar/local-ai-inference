"""Presentation-only tables; retain complete source data in expandable details."""
from html import escape
import json


def _text(value):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    return str(value).replace('|', r'\|').replace('\n', '<br>')


def _number(value):
    return '—' if value is None else f'{value:.3f}'


def _pair(row, first, second):
    return _number(row.get(first)) + ' / ' + _number(row.get(second))


def _count(row, numerator, denominator):
    a, b = row.get(numerator), row.get(denominator)
    return '—' if a is None or b is None else f'{a}/{b}'


def _table(columns, rows):
    return '\n'.join(['| ' + ' | '.join(columns) + ' |',
                      '|' + '|'.join('---' for _ in columns) + '|',
                      *['| ' + ' | '.join(_text(value) for value in row) + ' |' for row in rows]])


def result_table(result):
    """Render a supplied result only; never read evidence or execute any work."""
    if not isinstance(result, dict):
        return str(result)
    quality, serving = result.get('quality') or {}, result.get('serving') or {}
    phases = (result.get('cache_study') or {}).get('phases') or {}
    memory = result.get('memory') or {}
    if quality or serving or phases:
        columns = ['Result / phase', 'Status', 'SQL executed', 'Reference matches',
                   'TTFT p50/p95 (s)', 'Request E2E p50/p95 (s)',
                   'Application p50/p95 (s)', 'Output tok/s', 'Peak memory (GiB)']
        rows = [[result.get('experiment', 'Application / serving'), result.get('status'),
                 _count(quality, 'sql_executed', 'cases'),
                 _count(quality, 'result_equivalent', 'reference_scored'),
                 _pair(serving, 'ttft_p50_seconds', 'ttft_p95_seconds'),
                 _pair(serving, 'e2e_p50_seconds', 'e2e_p95_seconds'),
                 _pair(quality, 'latency_p50_seconds', 'latency_p95_seconds'),
                 _number(serving.get('output_tokens_per_second')),
                 _number(memory['peak_bytes'] / 2**30 if memory.get('peak_bytes') is not None else None)]]
        for phase in ('cold', 'warm'):
            stats = phases.get(phase)
            if stats is None:
                continue
            rows.append([phase, result.get('status'), _count(stats, 'sql_executed', 'requests'),
                         _count(stats, 'result_equivalent', 'reference_scored'),
                         _pair(stats, 'ttft_p50_seconds', 'ttft_p95_seconds'),
                         _pair(stats, 'e2e_p50_seconds', 'e2e_p95_seconds'), '—',
                         _number(stats.get('output_tokens_per_second')),
                         _number(stats['peak_gpu_bytes'] / 2**30 if stats.get('peak_gpu_bytes') is not None else None)])
        text = _table(columns, rows)
    else:
        observer = result.get('native_observer') or {}
        sql = result.get('sql_check') or result.get('sql_smoke') or result
        sample = result.get('sample') or {}
        attempts = result.get('attempts') or []
        last_attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
        executed = sql.get('execution_success', last_attempt.get('execution_success'))
        matched = sql.get('result_equivalence', (result.get('scores') or {}).get('result_equivalence'))
        ttft = result.get('ttft_seconds', result.get('native_ttft_seconds', sample.get('ttft_seconds')))
        if ttft is None and 'first_token_ms' in observer and 'inference_start_ms' in observer:
            ttft = (observer['first_token_ms'] - observer['inference_start_ms']) / 1000
        columns = ['Result', 'Status', 'SQL executed', 'Reference match', 'TTFT (s)',
                   'Request / inference (s)', 'Model load (s)', 'Peak memory (GiB)', 'Artifacts (MiB)']
        rows = [[result.get('stage', result.get('experiment', result.get('model_type', (result.get('configuration') or {}).get('experiment', 'Result')))),
                 result.get('status'), executed, matched,
                 _number(ttft), _number(result.get('single_request_seconds', result.get('inference_seconds', sample.get('e2e_seconds')))),
                 _number(result.get('startup_seconds', result.get('model_load_seconds'))),
                 _number(memory['peak_bytes'] / 2**30 if memory.get('peak_bytes') is not None else None),
                 _number(result['footprint_bytes'] / 2**20 if result.get('footprint_bytes') is not None else None)]]
        text = _table(columns, rows)
        if isinstance(result.get('attempts'), list) and any('model_class' in a for a in result['attempts']):
            attempts = result['attempts']
            text += '\n\n' + _table(['Model configuration', 'Accepted', 'Reason'],
                [[a.get('model_class'), a.get('accepted'), a.get('reason')] for a in attempts])
        if 'native_inference_run' in result:
            text += '\n\nNative inference run: **' + _text(result['native_inference_run']) + '**.'
    if memory.get('scope'):
        text += '\n\nMemory scope: ' + _text(memory['scope']) + '.'
    provenance = [(key, result[key]) for key in ('session_id', 'session', 'session_evidence_path', 'evidence_path', 'reason') if result.get(key) is not None]
    if provenance:
        text += '\n\n' + _table([key.replace('_', ' ').capitalize() for key, _ in provenance],
                                [[value for _, value in provenance]])
    text += ('\n\n<details><summary>Full result details</summary>\n\n<pre>' +
             escape(json.dumps(result, indent=2, ensure_ascii=False, default=str)) + '</pre>\n</details>')
    return text
