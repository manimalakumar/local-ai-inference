"""Inspect recorded output work before interpreting quantization latency."""
from statistics import mean, median


def analyze(results):
    rows, cases = [], []
    for name, precision in [('fp16_cache_on', 'FP16'), ('awq_w4a16_cache_on', 'AWQ W4A16')]:
        result = results.get(name, {})
        study = result.get('cache_study') or {}
        for phase in ('cold', 'warm'):
            records = [r for r in study.get('records', []) if r.get('phase') == phase]
            if not records:
                continue
            def stats(group):
                tokens = [r['output_tokens'] for r in group if r.get('output_tokens') is not None]
                decode = [r['decode_tokens_per_second'] for r in group if r.get('decode_tokens_per_second') is not None]
                latency = [r['e2e_seconds'] for r in group if r.get('e2e_seconds') is not None]
                return dict(requests=len(group), token_samples=len(tokens),
                    output_tokens_mean=mean(tokens) if tokens else None,
                    output_tokens_median=median(tokens) if tokens else None,
                    output_tokens_min=min(tokens) if tokens else None,
                    output_tokens_max=max(tokens) if tokens else None,
                    estimated_decode_tok_s_median=median(decode) if decode else None,
                    e2e_median=median(latency) if latency else None)
            source = result.get('_recorded', {}).get('source', result.get('run_id'))
            rows.append(dict(precision=precision, phase=phase, source=source, **stats(records)))
            for case_id in sorted({r['case_id'] for r in records}):
                cases.append(dict(precision=precision, phase=phase, case_id=case_id,
                                  **stats([r for r in records if r['case_id'] == case_id])))
    return dict(scope='Per-target cache-on records, separated into cold/warm. Not supplemental warmed serving summaries.',
                rows=rows, per_case=cases,
                limitation='The decode rate is a client estimate from usage and E2E-minus-TTFT, not GPU kernel profiling. Different generated SQL/text is not identical computation.')


def table(analysis):
    def number(value):
        return 'unavailable' if value is None else f'{value:.2f}'
    lines = ['| Configuration | Phase | Requests | Mean output tokens | Median output tokens | Estimated decode tok/s p50 |',
             '|---|---|---|---|---|---|']
    for row in analysis['rows']:
        lines.append(f'| {row["precision"]} | {row["phase"]} | {row["requests"]} | '
                     f'{number(row["output_tokens_mean"])} | {number(row["output_tokens_median"])} | '
                     f'{number(row["estimated_decode_tok_s_median"])} |')
    return '\n'.join(lines) if analysis['rows'] else 'No output-token measurements in these results. Plan mode fabricates no counts.'
