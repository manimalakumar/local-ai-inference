"""Build the standalone local AI optimization tutorial, unexecuted by default."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    'nemotron': {
        'label': 'Nemotron 3 Nano 4B',
        'model_id': 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
        'fp16_snapshot': 'models/nemotron-3-nano-4b-frozen',
        'awq_snapshot': 'models/nemotron-3-nano-4b-awq-4bit-frozen',
        'base_revision': 'SET_EXACT_NEMOTRON_BASE_REVISION_AND_AWQ_DERIVATION',
        'awq_source': 'SET_VERIFIED_NEMOTRON_3_NANO_4B_AWQ_SOURCE',
        'filename': 'local AI optimization - Nemotron.ipynb',
        'description': 'The SQL generator in this notebook is **Nemotron 3 Nano 4B** '
            '([base checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)). '
            'It generates clinical SQL from the fixed schema and question through vLLM. '
            'The unquantized BF16 source weights are loaded as FP16 for this experiment; '
            'the comparison requires an AWQ W4A16 derivative of that exact base revision. '
            'Set its verified AWQ source and local checkpoint path before running; no '
            'Nemotron AWQ artifact or GPU compatibility is assumed.',
    },
    'qwen': {
        'label': 'Qwen3-4B',
        'model_id': 'Qwen/Qwen3-4B',
        'fp16_snapshot': 'models/qwen3-4b-frozen',
        'awq_snapshot': 'models/qwen3-4b-awq-4bit-frozen',
        'base_revision': 'SET_EXACT_QWEN3_4B_BASE_REVISION_AND_AWQ_DERIVATION',
        'awq_source': 'Qwen/Qwen3-4B-AWQ',
        'filename': 'local AI optimization - Qwen.ipynb',
        'description': 'The SQL generator in this notebook is **Qwen3-4B** '
            '([base checkpoint](https://huggingface.co/Qwen/Qwen3-4B)). '
            'It generates clinical SQL from the fixed schema and question through vLLM. '
            'The baseline loads the unquantized checkpoint as FP16; the W4A16 candidate is '
            '[Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ). '
            'Pin compatible base/AWQ revisions and set the local checkpoint paths before running.',
    },
}


def notebook_document(model='qwen'):
    if model == 'nemotron':
        sys.path.insert(0, str(ROOT / 'src'))
        from clinical_trial_mvp.tutorial.nemotron_notebook import notebook_document as nemotron_document
        return nemotron_document()
    selected = MODELS[model]
    cells = []
    def add(kind, text):
        source = text.strip() + '\n'
        item = {'cell_type': kind, 'id': hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    def md(text):
        add('markdown', text)
    def code(text):
        add('code', text)

    md('''# Optimizing Local LLM Inference
## Prefix Caching and AWQ on a Clinical Text-to-SQL Workload

### 1. Problem and learning outcomes
In this tutorial, we test two inference optimizations on the same local Text-to-SQL workload:
prefix caching avoids repeated shared prefill computation; AWQ W4A16 compresses weights
to 4-bit while retaining FP16 activations. We measure SQL correctness, TTFT, request
end-to-end latency and GPU memory, plus throughput and approximate TPOT.

You will learn to distinguish prefill from decode, explain the limits of caching,
explain what remains 16-bit in W4A16, and choose a configuration using performance
and quality together. This separate MVP4 notebook stays in the existing repository.
**Live execution is disabled initially; GPU measurements are produced on your GPU machine.**
''')
    md('''**Model formats and runtime history, including ExecuTorch**

| Stage | Runtime / hardware | Model format and precision |
|---|---|---|
| MVP1 | Mac local inference / Ollama | GGUF, including Nemotron 4B Q4_K_M |
| MVP2 | llama.cpp and Ollama / M4 Pro | Same Nemotron Q4_K_M GGUF |
| MVP3 | Mac paths; GPU vLLM/SGLang | GGUF on Mac; Hugging Face checkpoints on GPU |
| ExecuTorch portability case | Native ExecuTorch / M4 Pro CPU, XNNPACK | SmolLM2-135M HF checkpoint converted/exported to FP32 `.pte`; no GGUF |
| Current MVP4 optimization | vLLM / target GPU | HF FP16 and AWQ W4A16 checkpoints; GPU measurements pending |

The saved **October 29, 2026 ExecuTorch run** used `HuggingFaceTB/SmolLM2-135M`,
not the originally planned gated Llama 3.2 checkpoint. Native model load was **0.076 s**,
single-request inference including prefill **0.199 s**, and whole-process time **3.480 s**.
PTE + tokenizer footprint was **623.60 MiB**; sampled peak process RSS **1018.13 MiB**.
RSS is CPU process memory, not GPU VRAM. TTFT and clinical SQL correctness were not
measured in this text-completion portability smoke test. Its smaller model, FP32 precision,
CPU backend and different prompt prevent a controlled performance comparison with MVP4.

See [ExecuTorch reproduction and limitations](../tutorial/EXECUTORCH_MAC.md) and the
[saved hardware record](../experiment-tracker/tutorial/edge/20261029T161510_034051Z.json).
This historical capture adds no new run or fifth optimization variant.
''')
    md('''### 2. Inference anatomy

| Component | Used when | Intervention |
|---|---|---|
| Model weights | Prefill and decode | AWQ compresses weights |
| Activations | Current layer computation | Remain FP16 in W4A16 |
| KV cache | Retains previous-token attention state | Remains unquantized here |
| Prefix cache | Reuses matching KV blocks across requests | Avoids shared prefill computation |

**Prefill** processes the input and creates KV state. **Decode** generates tokens sequentially.
Caching should mainly improve TTFT; it does not directly accelerate decoding.
AWQ reduces weight storage, but kernel support determines whether latency improves.
W4A16 does not make activations or the KV cache 4-bit.
''')
    md('''### 3. Workload and hypotheses

| | Cache off | Cache on |
|---|---|---|
| FP16 weights / FP16 activations | Baseline | Cache effect |
| AWQ 4-bit weights / FP16 activations | Quantization effect | Combined effect |

We keep the existing four clinical questions, system rules, **identical full schema prefix**,
tokenizer, generation seed (42), temperature (0), token cap (1024), context (8192) and GPU fixed.
Full schema replaces per-question retrieved schemas for this controlled experiment: every
prompt uses `system instructions → rules/schema/guidance → unique question` via the existing
prompt builder. Prefix hashes are saved. Concurrency is 1 to keep reset/prime ordering controlled.

Cache-on: for each of four targets, repeat five times:

1. Reset and confirm success, then measure the **cold** target request.
2. Reset again; prime with a **different question**, sharing the identical schema prefix.
3. Measure the **warm** target using the exact same prompt as the cold target.

Thus each cache-on configuration has 20 cold and 20 warm measured requests; cache-off has
20 comparable requests. The warm target cannot reuse its own whole question from its paired
cold request. Shared blocks stop at the first differing token; short question fragments may
also match. Five repeats are illustrative, not robust tail-latency evidence.

Hypotheses: cache-on lowers warm TTFT; approximate TPOT may stay similar; AWQ needs less
weight storage but can be slower; generated SQL may change. GPU memory includes vLLM KV
reservations, so total allocation need not fall in proportion to weight compression.
''')
    code('''from pathlib import Path
import json
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import Markdown, display
from clinical_trial_mvp.tutorial.optimization import (
    VARIANTS, configurations, checkpoint_pair, launch_preview, run_variant, compare_pair,
    awq_smoke, preflight,
)
from clinical_trial_mvp.tutorial.cache_study import phase_table
from clinical_trial_mvp.tutorial.harness import measurements_markdown
RUN_TAG = 'run02'  # New tag for every set of fresh launches.
configs = configurations(run_tag=RUN_TAG)
FP16_SNAPSHOT = ROOT / 'models/SET_FP16_HF_SNAPSHOT'
AWQ_SNAPSHOT = ROOT / 'models/SET_AWQ_4BIT_HF_SNAPSHOT'
MODEL_ID = 'SET_BASE_MODEL_ID'
BASE_REVISION = 'SET_EXACT_BASE_REVISION_AND_AWQ_DERIVATION'
AWQ_SOURCE = 'SET_AWQ_CHECKPOINT_SOURCE'
VALIDATE_CHECKPOINTS = False
RUN_LIVE = False
ACTIVE = None  # 'awq_smoke' first, then one of VARIANTS, matching the resident server.
LOAD_SAVED = False
pair = None
smoke = None
results = {}
''')
    md('''### 4. Environment preflight and AWQ compatibility smoke test

Use the existing working [GPU environment](../SETUP.md). Launch Jupyter on the GPU host;
install `requirements-notebook.txt` in its environment if needed. Use the same pinned vLLM
build for all launches. Set the paths/model revisions above, then enable checkpoint validation.
This loads no GPU libraries and downloads no weights. AWQ must already be quantized; a
server flag does not convert FP16 weights.

The preflight reports the notebook environment. Launch provenance separately records the
actual server's PyTorch, CUDA and vLLM versions when the server uses another environment.
The CUDA capability in `nvidia-smi` is driver capability, not the server's Torch CUDA build.

<details><summary>Reproducibility note</summary>

Validation checks 4-bit AWQ metadata, an unquantized baseline, matching architecture and
byte-identical tokenizer/template files. Record the exact base revision and AWQ derivation:
matching shapes cannot prove common lineage. Each server is checked against the validated
weight fingerprint. Launch files must belong to the actual resident server. No other clients
or external KV connectors may use the server during reset/prime measurements.

</details>

**First launch the AWQ smoke command below in another terminal, wait for readiness, set
`ACTIVE='awq_smoke'` and `RUN_LIVE=True`, then run the smoke cell.** It generates one SQL
answer and executes it through the existing read-only validator. Failure blocks all four
variants; preserve its saved error as compatibility evidence. Success is compatibility,
not an acceptable final quality score. Stop the smoke server before the FP16 baseline.

For cache-on measurements, `/reset_prefix_cache` must explicitly return `success=true`.
Old empty HTTP 200 acknowledgements are insufficient: cold measurement fails visibly
rather than silently labeling a warm cache cold. See the run guide for the API limitation.
''')
    code('''if VALIDATE_CHECKPOINTS:
    pair = checkpoint_pair(FP16_SNAPSHOT, AWQ_SNAPSHOT, base_revision=BASE_REVISION + '; ' + AWQ_SOURCE)
info = preflight(configs[VARIANTS[0]], model_id=MODEL_ID, base_revision=BASE_REVISION,
                 awq_source=AWQ_SOURCE, execute=RUN_LIVE)
display(Markdown('| Preflight | Value |\\n|---|---|\\n' + '\\n'.join(f'| {k} | {v} |' for k, v in info.items())))
smoke_spec = dict(configs['awq_w4a16_cache_off'])
smoke_spec['provenance'] = f'experiment-tracker/mvp4/provenance/{RUN_TAG}-awq-smoke.json'
print('Launch AWQ smoke first:\\n' + launch_preview(smoke_spec, AWQ_SNAPSHOT))
''')
    code('''if RUN_LIVE and ACTIVE == 'awq_smoke':
    smoke = awq_smoke(smoke_spec, pair, execute=True, root=ROOT)
    print(json.dumps(smoke, indent=2, default=str))
else:
    print('AWQ smoke pending; launch AWQ and enable its cell on the GPU host.')
# To resume a GPU session, load the exact successful smoke JSON before the variant cells.
# smoke = json.loads((ROOT / 'experiment-tracker/mvp4/smoke/EXACT_RUN_ID.json').read_text())
''')
    md('''Every variant uses a new foreground server. The commands below use the existing launcher;
run only the matching command in a terminal, wait for readiness and confirm the GPU UUID
in its provenance file. Add the real `nvidia-smi` UUID if the launcher did not capture it.
Stop the previous engine with Ctrl-C first. Provenance files are never overwritten.
**Startup, readiness, excluded kernel warmup, resets and priming are outside request timing.**
GPU sampling surrounds each measured request; SQL validation/execution follows the request
clock. Idle reserved allocation remains included in GPU memory.
''')
    code('''for name in VARIANTS:
    snapshot = AWQ_SNAPSHOT if configs[name]['quantization'] else FP16_SNAPSHOT
    print(name + ':\\n' + launch_preview(configs[name], snapshot) + '\\n')
def run_one(name):
    global current
    current = run_variant(configs[name], pair, smoke=smoke,
                          execute=RUN_LIVE and ACTIVE == name, root=ROOT)
    if current['status'] != 'plan_only' or name not in results:
        results[name] = current
    display(Markdown(phase_table({name: results[name]})))
    print(current['status'], current.get('reason', ''))
''')
    for number, title, name, explanation in [
        (5, 'FP16 baseline', 'fp16_cache_off', 'Launch FP16/cache-off after a successful AWQ smoke test. This establishes uncached latency and correctness.'),
        (6, 'Prefix-cache experiment', 'fp16_cache_on', 'Restart with the same FP16 checkpoint and cache enabled. Cold and warm requests appear separately; warming uses another question with the same schema.'),
        (7, 'AWQ experiment', 'awq_w4a16_cache_off', 'Restart with the validated AWQ checkpoint and cache disabled. Weights are 4-bit, activations FP16. Compare this to the FP16 baseline.'),
        (8, 'Combined experiment', 'awq_w4a16_cache_on', 'Restart AWQ with prefix caching enabled. Test whether its cold/warm difference changes under W4A16.'),
    ]:
        md(f'### {number}. {title}\n\n{explanation}\n\nSet `ACTIVE={name!r}` and `RUN_LIVE=True` in the setup variables **without rerunning setup**, then run this cell after starting the matching server. Save all four configurations; do not use Run All with a single resident server.')
        code(f"run_one('{name}')")
    md('''### 9. Quality guardrail and saved results

**Mechanism layer:** direct SQL generation requests are timed and then scored separately
for cold, warm or cache-off conditions. No repairs hide their first-pass correctness.
Approximate TPOT is `(E2E - TTFT)/(output_tokens - 1)`, not measured per-token timestamps.
Aggregate output tok/s uses summed active request durations, excluding reset/prime gaps;
it is not total wall-clock useful task throughput.

**Application layer:** afterward, the inherited clinical plan → SQL → validation → DuckDB
→ repair path runs against the same questions/schema. The second table reports this
application's SQL result match and semantic review coverage. Its TTFT/E2E serving columns
are a supplemental warmed repeated run, not the controlled cold/warm mechanism results.
Agent quality latency (including plan/execution/repair) is printed separately below.

Four cases are a functional smoke test. Final quality claims require the full 23-reference-question
SQL evaluation set (30 total cases including policy cases in the current dataset).
Semantic review uses exact SQL fingerprints and remains unreviewed until a human reviews it.
Raw per-phase SQL, failures, results, reset confirmations and memory samples are saved.

Set `LOAD_SAVED=True` to recover this tag's latest successful records after copying
`experiment-tracker/mvp4/{runs,provenance,smoke}` back to the Mac.
''')
    code('''if LOAD_SAVED:
    for path in sorted((ROOT / 'experiment-tracker/mvp4/runs').glob('*.json')):
        saved = json.loads(path.read_text())
        for name, spec in configs.items():
            if (saved.get('configuration') == spec and saved.get('status') == 'completed'
                    and saved.get('evidence') == 'hardware' and saved.get('cache_study')):
                results[name] = saved
display(Markdown(phase_table(results)))
application = [r for r in results.values() if r.get('quality')]
display(Markdown(measurements_markdown(application)))
for r in application:
    q = r['quality']
    print(r['experiment'], 'clinical agent E2E p50/p95 seconds:',
          q.get('latency_p50_seconds'), q.get('latency_p95_seconds'))
''')
    md('''### 10. Findings and deployment decision

Compare cold and warm separately. Pairs must match environment, database, date, prompts,
shared prefix and checkpoint lineage. Missing or failed phases cannot produce speed ratios.
A ratio above 1 means the right-hand variant has lower latency. Check SQL correctness in
both phases before interpreting speed. No winner is established until GPU execution.
''')
    code('''comparisons = [
    ('fp16_cache_off', 'fp16_cache_on', 'prefix_cache'),
    ('fp16_cache_off', 'awq_w4a16_cache_off', 'quantization'),
    ('awq_w4a16_cache_off', 'awq_w4a16_cache_on', 'prefix_cache'),
    ('fp16_cache_on', 'awq_w4a16_cache_on', 'quantization'),
]
for left, right, factor in comparisons:
    for phase in ('cold', 'warm'):
        try:
            print(json.dumps(compare_pair(results.get(left, {}), results.get(right, {}),
                                          factor=factor, phase=phase), indent=2))
        except ValueError as exc:
            print(f'{left} -> {right} / {phase}: pending or unpaired: {exc}')
''')
    md('''| Workload condition | Deployment guidance to test |
|---|---|
| Repeated shared schema/system prefix | Consider prefix caching; inspect warm TTFT |
| Mostly unrelated prefixes | Cache benefit may be limited |
| Weights do not fit comfortably | Consider compatible AWQ W4A16 |
| FP16 fits and latency matters | Benchmark AWQ; do not assume it is faster |
| SQL quality falls after quantization | Retain FP16 or investigate recovery |
| Constrained GPU | Prioritize correctly completed tasks per unit total time |

Record observations here after GPU runs: cold/warm TTFT change, TPOT change, GPU allocation,
per-phase SQL match, application quality, and failures. Repeat fresh launches with a new tag
and reverse order before making a deployment choice. Do not estimate semantically correct
throughput while outputs remain unreviewed, or divide mixed phases by a serving-only clock.

### 11. Learning journal

| Situation | What happened | Root cause | Fix | Transferable lesson |
|---|---|---|---|---|
| AWQ checkpoint/kernel setup | Pending GPU smoke | Pending | Pending | Support is stack-specific |
| Cold reset unsupported | Pending | Legacy empty acknowledgement? | Verify API support | A requested reset is not proof |
| Cache gain small | Pending | Prefix length/block reuse? | Inspect prompts and counts | Prompt design controls reuse |
| Memory saving small | Pending | KV/allocator reservations? | Inspect startup allocation logs | Total VRAM is not weight size |
| SQL changed at temperature 0 | Pending | Numerical/backend differences? | Execute and review | Determinism is not semantic equivalence |

References: [prefix caching](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/),
[AWQ](https://docs.vllm.ai/en/v0.13.0/features/quantization/auto_awq/),
[confirmed reset API](https://docs.vllm.ai/en/v0.27.1/api/vllm/entrypoints/serve/dev/cache/api_router/).
See [GPU run instructions](../tutorial/MVP4_OPTIMIZATION.md).
''')
    replacements = {
        '# Optimizing Local LLM Inference\n': '# Optimizing Local LLM Inference — ' + selected['label'] + '\n',
        '### 1. Problem and learning outcomes\n': '### 1. Problem and learning outcomes\n' + selected['description'] + '\n\n',
        "RUN_TAG = 'run02'": "RUN_TAG = '" + model + "-run02'",
        'models/SET_FP16_HF_SNAPSHOT': selected['fp16_snapshot'],
        'models/SET_AWQ_4BIT_HF_SNAPSHOT': selected['awq_snapshot'],
        'SET_BASE_MODEL_ID': selected['model_id'],
        'SET_EXACT_BASE_REVISION_AND_AWQ_DERIVATION': selected['base_revision'],
        'SET_AWQ_CHECKPOINT_SOURCE': selected['awq_source'],
        '| Current MVP4 optimization | vLLM / target GPU | HF FP16 and AWQ W4A16 checkpoints; GPU measurements pending |':
            '| Current MVP4 optimization | vLLM / target GPU | ' + selected['label'] + ' HF FP16 and AWQ W4A16 checkpoints; GPU measurements pending |',
        'We keep the existing four clinical questions, system rules, **identical full schema prefix**,':
            'For ' + selected['label'] + ', we keep the existing four clinical questions, system rules, **identical full schema prefix**,',
        'The commands below use the existing launcher;':
            'The ' + selected['label'] + ' commands below use the existing launcher;',
    }
    for index, cell in enumerate(cells):
        source = ''.join(cell['source'])
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell['source'] = source.splitlines(keepends=True)
        cell['id'] = hashlib.sha256((str(index) + source).encode()).hexdigest()[:12]
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python', 'version': '3.11'}},
            'nbformat': 4, 'nbformat_minor': 5}


if __name__ == '__main__':
    for model, selected in MODELS.items():
        path = ROOT / 'notebooks' / selected['filename']
        path.write_text(json.dumps(notebook_document(model), indent=1) + '\n')
        print(path.name)
