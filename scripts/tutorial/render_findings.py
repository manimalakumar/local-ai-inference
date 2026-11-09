"""Reproduce exactly two figures from checksum-verified recorded Dell measurements."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.replay import RecordedStudy, LABEL


def figure_data(root=ROOT):
    replay = RecordedStudy(root)
    caching, concurrency = [], []
    for name, precision in [('fp16_cache_on', 'FP16'), ('awq_w4a16_cache_on', 'AWQ W4A16')]:
        record = replay.result('vllm', 'qwen', name)
        for phase in ('cold', 'warm'):
            stats = record['cache_study']['phases'][phase]
            caching.append(dict(precision=precision, phase=phase, ttft=stats['ttft_p50_seconds'],
                e2e=stats['e2e_p50_seconds'], matched=stats['result_equivalent'], scored=stats['reference_scored'],
                source=record['_recorded']['source']))
    for name in ('fp16_baseline', 'concurrency_2', 'concurrency_4'):
        record = replay.result('vllm', 'nemotron', name)
        stats = record['serving']
        concurrency.append(dict(clients=record['configuration']['concurrency'],
            scheduler_capacity=record['configuration']['max_num_seqs'],
            e2e=stats['e2e_p50_seconds'], tok_s=stats['output_tokens_per_second'],
            matched=stats['result_equivalent'], scored=stats['reference_scored'],
            application_matches=record['quality']['result_equivalent'],
            source=record['_recorded']['source']))
    return dict(label=LABEL, distinct_questions=4, caching=caching, concurrency=concurrency)


def render(root=ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = figure_data(root)
    folder = Path(root) / 'tutorial/figures'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#ffffff', 'axes.titleweight': 'bold', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'local-ai-inference-lab'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1))
    colors = ['#3568a8' if r['phase'] == 'cold' else '#168678' for r in data['caching']]
    for ax, metric, title in zip(axes, ('ttft', 'e2e'), ('First-content TTFT', 'Request end-to-end latency')):
        values = [r[metric] for r in data['caching']]
        ax.bar(range(4), values, color=colors, width=.62)
        ax.set_xticks(range(4), [r['precision'] + '\n' + r['phase'] for r in data['caching']])
        ax.set_ylabel('Median seconds')
        ax.set_title(title, pad=13)
        ax.set_ylim(0, max(values) * 1.28)
        ax.yaxis.grid(True, alpha=.18)
        ax.set_axisbelow(True)
        for i, (value, record) in enumerate(zip(values, data['caching'])):
            ax.text(i, value + max(values) * .025,
                    f'{value:.3f} s\n{record["matched"]}/{record["scored"]} SQL matches', ha='center', fontsize=9)
    fig.suptitle('Qwen: prefix reuse changes prefill more than total request time', fontsize=15, y=.98)
    fig.text(.5, .025, 'Recorded Dell measurement · cache-on paired phases · four distinct questions × five repeats\n'
             'Matches are first-pass reference-result matches; semantic review pending. Startup/reset/priming excluded.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .12, 1, .92))
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'01_prefix_cache.{suffix}', dpi=170, metadata={'Date': None} if suffix == 'svg' else {})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for point, color in zip(data['concurrency'], ('#3568a8', '#168678', '#9c6040')):
        ax.scatter(point['e2e'], point['tok_s'], s=100, color=color, zorder=3)
        ax.annotate(f'{point["clients"]} client(s)\n{point["matched"]}/{point["scored"]} direct SQL matches; '
                    f'{point["application_matches"]}/4 application matches',
                    (point['e2e'], point['tok_s']), xytext=(12, -6), textcoords='offset points', fontsize=10)
    ax.set_xlabel('Median request end-to-end latency (seconds)')
    ax.set_ylabel('Aggregate serving output (tokens / second)')
    ax.set_xlim(3.5, 12)
    ax.set_ylim(25, 92)
    ax.grid(True, alpha=.18)
    ax.set_title('Nemotron: more concurrency trades latency for aggregate throughput', pad=17)
    fig.text(.5, .025, 'Recorded Dell measurement · native vLLM · scheduler capacity fixed at 4\n'
             'Four distinct questions × five repeats per level. Aggregate throughput uses serving wall time; warmup excluded.\n'
             'Reference-result match is not reviewed semantic correctness. Achieved batch occupancy was not measured.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .14, 1, 1))
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'02_concurrency.{suffix}', dpi=170, metadata={'Date': None} if suffix == 'svg' else {})
    plt.close(fig)
    # Matplotlib SVG path lines contain cosmetic trailing spaces; keep generated Git diffs clean.
    for path in folder.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    (folder / 'figure_data.json').write_text(json.dumps(data, indent=2) + '\n')
    return folder


if __name__ == '__main__':
    print(render())
