"""Reproducible Nemotron notebook: core before conditional AWQ stretch."""
import hashlib


def notebook_document():
    cells = []
    def add(kind, text):
        source = text.strip() + '\n'
        cell = dict(cell_type=kind, id=hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                    metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    def md(text):
        add('markdown', text)
    def code(text):
        add('code', text)
    md('''# Optimizing Local LLM Inference — Nemotron 3 Nano 4B

## 1. Use case → workload → serving → optimization

A clinical-trial analyst asks operational questions over the included synthetic ten-table
DuckDB database. **Nemotron 3 Nano 4B** (`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`)
generates a clinical plan and SQL; the inherited read-only guards, execution, result scorer
and one repair check whether the answer matches reference results. vLLM serves local HF
weights loaded as **FP16**. This is a continuation of the clinical workload, not a new app.

The goal is lower latency and resource use without losing SQL correctness. Core work needs
only the unquantized checkpoint; **AWQ is optional stretch work and never gates core runs**.
Four questions (`eval_001`, `002`, `005`, `007`) form a functional smoke workload. Five
serving repeats give 20 direct SQL requests per configuration. Final quality conclusions
need the full 23-reference-question evaluation plus semantic review (30 total cases).
GPU measurements are pending. Live execution starts disabled; default cells run on CPU.
''')
    md('''## 2. Mechanism and experiment strategy

Prefill processes the prompt; decode generates output tokens. A shorter schema reduces
input work. Prefix caching reuses matching prefix state, mainly affecting TTFT. Client
concurrency submits overlapping requests; `max-num-seqs` limits server scheduler capacity.
vLLM dynamically batches active requests: neither a concurrency setting nor a scheduler
limit proves an achieved batch size. Reasoning changes generated work; output limits bound
that work and can truncate an otherwise valid answer.

Nemotron is a **hybrid Mamba/attention model**. Its state is not interchangeable with Qwen's
attention-only KV cache. This notebook holds Mamba SSM state at FP32 as in the model-card
serving example; FP16 here describes weights/activations, not every cache/state tensor.

| Priority | Experiment | Control |
|---|---|---|
| Core | FP16 baseline | Full schema, cache off, concurrency 1, scheduler capacity 4, reasoning off, cap 1024 |
| Core | Prefix caching | Same checkpoint/schema; confirmed-reset cold vs distinct-question-primed warm requests |
| Core | Context/schema reduction | Full vs existing frozen retrieved schema; same server/token cap |
| Core | Concurrency | 1 vs 2 vs 4 clients with scheduler capacity fixed at 4 |
| Core | Batching capacity | Compare `concurrency_4` vs `batch_capacity_1`; concurrency fixed at 4, capacity 4 vs 1 |
| Core | Reasoning | Off vs on; same cap 1024 and sampling settings |
| Core | Output limits | 256 vs 1024 vs 2048; reasoning off |
| Stretch | AWQ W4A16 | Only with a validated compatible derivative and successful AWQ smoke |

Treat every row as a controlled intervention. Smaller output caps may cause failures;
never drop truncated requests to make throughput look better. Temperature 0 and seed 42
are held fixed for this engineering comparison, not offered as general reasoning tuning.
''')
    code('''from pathlib import Path
import json
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import Markdown, display
from clinical_trial_mvp.tutorial import nemotron
from clinical_trial_mvp.tutorial.optimization import preflight, checkpoint_pair, awq_smoke
from clinical_trial_mvp.tutorial.cache_study import phase_table
RUN_TAG = 'nemotron-run02'
MODEL_ID = 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
FP16_SNAPSHOT = ROOT / 'models/nemotron-3-nano-4b-frozen'
AWQ_SNAPSHOT = ROOT / 'models/nemotron-3-nano-4b-awq-4bit-frozen'
BASE_REVISION = 'SET_EXACT_NEMOTRON_BASE_REVISION_AND_AWQ_DERIVATION'
AWQ_SOURCE = 'SET_VERIFIED_NEMOTRON_3_NANO_4B_AWQ_SOURCE'
configs = nemotron.configurations(RUN_TAG)
VARIANTS = tuple(configs)
VALIDATE_CHECKPOINTS = False
RUN_LIVE = False
ACTIVE = None
LOAD_SAVED = False
PREFIX_SUPPORTED = False  # Set only after checking actual build/backend support and startup logs.
RUN_AWQ = False
checkpoint = None
pair = None
smoke = None
results = {}
''')
    md('''## 3. Environment and model preflight

Use `SETUP.md` for the CPU notebook and separate Linux GPU engine environment. The model
card requires **vLLM >= 0.15.1**. Pin the actual installed vLLM/Torch/CUDA combination and
record the GPU SKU, VRAM and UUID; model support alone does not prove compatibility with
your GPU. Use a complete local snapshot, including reviewed custom model/tokenizer code.
The launcher enables trusted local model code and offline HF loading; it does not download
or execute a remote model download. Record the exact revision, then enable validation below.

The FP16 baseline can fail on an unsupported kernel, device or precision. Keep the failure
and fix that baseline first; do not silently substitute BF16 or a different model. vLLM
release/backend support for hybrid prefix caching varies; verify separately and leave
`PREFIX_SUPPORTED=False` if unsupported. A skipped cache row must not block other core work.
''')
    code('''if VALIDATE_CHECKPOINTS:
    checkpoint = nemotron.baseline_checkpoint(FP16_SNAPSHOT, revision=BASE_REVISION)
info = preflight(configs['fp16_baseline'], model_id=MODEL_ID, base_revision=BASE_REVISION,
                 awq_source='stretch only: ' + AWQ_SOURCE, execute=RUN_LIVE)
display(Markdown('| Preflight | Value |\\n|---|---|\\n' + '\\n'.join(f'| {k} | {v} |' for k, v in info.items())))
display(Markdown('| Experiment | Priority | Schema | Clients | Scheduler capacity | Reasoning | Output cap |\\n|---|---|---|---|---|---|---|\\n' +
    '\\n'.join(f'| {name} | {s["priority"]} | {s["profile"]} | {s["concurrency"]} | {s["max_num_seqs"]} | {s["reasoning"]} | {s["max_tokens"]} |'
              for name, s in configs.items())))
''')
    md('''## 4. Deploy one server and select its experiment

Launch commands run in a separate terminal from the repo root. Start with the FP16
baseline, wait for `/health`, inspect startup logs, and confirm the GPU UUID in metadata.
Use one resident engine. Stop it before a fresh launch with the next experiment's metadata.
The commands use the legacy served alias `clinical-trial-mvp3` for harness compatibility;
`MODEL_ID`, revision and the weight fingerprint establish the actual model identity.

The cold/warm cache protocol uses one excluded kernel-warmup request, then, per target:
reset → cold target → reset → prime using another question with the **identical full schema
prefix** → warm target. Resets must return `success=true`; legacy empty acknowledgements
cannot establish cold state. Twenty cold and twenty warm targets are reported separately.
Startup, health checks, resets, priming, DB execution and telemetry setup are outside the
individual TTFT/E2E clocks. Other clients and external cache connectors must be absent.

For each core experiment, run its printed command, set `ACTIVE` and `RUN_LIVE=True` in the
notebook session, then execute the corresponding cell. Do not rerun setup and discard state.
Request-only interventions can use a verified equivalent running server only if the selected
provenance file actually describes that server; separate launch records are the default.
''')
    code('''for name, spec in configs.items():
    if spec['priority'] == 'core':
        print(name + ':\\n' + nemotron.launch_preview(spec, FP16_SNAPSHOT) + '\\n')
def run_one(name):
    global current
    spec = configs[name]
    current = nemotron.run(spec, checkpoint, execute=RUN_LIVE and ACTIVE == name,
                           root=ROOT, prefix_supported=PREFIX_SUPPORTED, pair=pair, smoke=smoke)
    if current['status'] != 'plan_only' or name not in results:
        results[name] = current
    print(name, current['status'], current.get('reason', ''))
    if current.get('cache_study'):
        display(Markdown(phase_table({name: current})))
''')
    sections = [
        ('5. FP16 baseline', ['fp16_baseline'], 'Establish executable SQL and save baseline provenance before other interventions.'),
        ('6. Prefix-cache experiment', ['prefix_cache'], 'Verify installed hybrid-cache support before setting PREFIX_SUPPORTED=True. Cold/warm requests keep the identical schema prefix. Unsupported is a recorded skip, not zero performance.'),
        ('7. Context/schema reduction', ['schema_retrieved'], 'Compare with fp16_baseline. Reuse the existing frozen retrieval cache; encoding was precomputed, so timings exclude live embedding load/encoding. Audit both token reduction and clinical definition/SQL misses.'),
        ('8. Concurrency and dynamic batching', ['concurrency_2', 'concurrency_4', 'batch_capacity_1'], 'Compare concurrency 1/2/4 at capacity 4. Compare concurrency_4 to batch_capacity_1 to isolate server capacity. These experiments do not report scheduler traces or achieved batch occupancy.'),
        ('9. Reasoning and output budgets', ['reasoning_on', 'output_256', 'output_2048'], 'Compare each with fp16_baseline. The reasoning client toggles enable_thinking and removes the conflicting /no_think directive. Raw thought + answer output is saved; complete think tags are removed only for SQL parsing. TTFT is first generated content, which includes thoughts when enabled; final-answer TTFT is not measured. Truncation remains a failure, and output cap includes thought tokens.'),
    ]
    for title, names, explanation in sections:
        md('## ' + title + '\n\n' + explanation)
        for name in names:
            code(f"run_one('{name}')")
    md('''## 10. Optional stretch: validated AWQ W4A16

No compatible Nemotron AWQ checkpoint is assumed to exist. Obtain or produce a derivative
of the exact validated base revision; validate its config, tokenizer and packed 4-bit weights.
Model-specific kernel and hybrid-state support must pass an actual GPU smoke test. An
FP8 or GGUF artifact is not a substitute. Leave this section disabled until evidence exists.
Failure or absence must not alter any core result.

If a compatible artifact is available, set AWQ_SOURCE, paths and revisions; enable RUN_AWQ;
launch the printed AWQ command and set ACTIVE='awq_smoke' for the smoke cell. Then switch
ACTIVE='awq_stretch' to run the full stretch row on the same verified server.
''')
    code('''if RUN_AWQ:
    pair = checkpoint_pair(FP16_SNAPSHOT, AWQ_SNAPSHOT, base_revision=BASE_REVISION + '; ' + AWQ_SOURCE)
    print(nemotron.launch_preview(configs['awq_stretch'], AWQ_SNAPSHOT))
    if RUN_LIVE and ACTIVE == 'awq_smoke':
        smoke = awq_smoke(configs['awq_stretch'], pair, execute=True, root=ROOT)
        print(smoke['status'], smoke.get('reason', ''))
if RUN_AWQ:
    run_one('awq_stretch')
else:
    results['awq_stretch'] = nemotron.run(configs['awq_stretch'], execute=False, root=ROOT)
    print('AWQ stretch disabled; core experiments do not need AWQ.')
''')
    md('''## 11. Measurements, quality and deployment decision

Direct serving records include SQL correctness after each timed completion. The inherited
application evaluation separately runs serial planning, SQL validation/execution and repair
on four questions. Table request TTFT/E2E and application E2E are separate. Peak memory is
sampled device VRAM including resident weights and reserved state/cache, not isolated weight
size; unknown memory remains unknown. Only first-content TTFT is recorded for reasoning.

Review request failure counts, SQL execution/result matches and exact-SQL semantic coverage
alongside latency, prompt/output tokens and aggregate throughput. A p95 from 20 requests is
illustrative. Match model hash, database, date, prompts, seed and unchanged server settings
before attributing a difference to one factor. Schema reduction intentionally changes prompts.
Reasoning and output caps intentionally change generation work; do not compare output tok/s
alone. Unsupported cache/AWQ runs are unavailable, not performance scores.

Decision remains pending GPU evidence. Retain the smallest context/output budget and
concurrency that improve useful completed SQL tasks without unacceptable correctness loss.
''')
    code('''if LOAD_SAVED:
    for path in sorted((ROOT / 'experiment-tracker/mvp4/nemotron').glob('*.json')):
        saved = json.loads(path.read_text())
        for name, spec in configs.items():
            if saved.get('configuration') == spec:
                results[name] = saved
def fmt(value):
    return '—' if value is None else f'{value:.3f}'
rows = ['| Experiment/status | Request SQL match | Semantic | TTFT p50/p95 s | Request E2E p50/p95 s | Application E2E p50 s | tok/s | Peak VRAM GiB |',
        '|---|---|---|---|---|---|---|---|']
for name, result in results.items():
    q, s, m = result.get('quality') or {}, result.get('serving') or {}, result.get('memory') or {}
    peak = m.get('peak_bytes')
    semantic = s.get('semantic', {})
    review = f'{semantic.get("correct",0)}/{semantic["reviewed"]} reviewed' if semantic.get('reviewed') else 'unreviewed'
    rows.append(f'| {name} / {result["status"]} | {s.get("result_equivalent", "—")}/{s.get("reference_scored", "—")} | {review} | '
                f'{fmt(s.get("ttft_p50_seconds"))}/{fmt(s.get("ttft_p95_seconds"))} | '
                f'{fmt(s.get("e2e_p50_seconds"))}/{fmt(s.get("e2e_p95_seconds"))} | {fmt(q.get("latency_p50_seconds"))} | '
                f'{fmt(s.get("output_tokens_per_second"))} | {fmt(peak / 1024**3 if peak is not None else None)} |')
display(Markdown('\\n'.join(rows)))
for name, result in results.items():
    if result.get('cache_study'):
        display(Markdown(phase_table({name: result})))
''')
    md('''## 12. Learning journal and lineage

| Experiment | Observation | Fix or lesson |
|---|---|---|
| FP16 compatibility | Pending GPU run | Record exact GPU/build and baseline failure before changing settings |
| Hybrid prefix cache | Pending support verification | Installed backend/version determines availability |
| Schema reduction | Pending | Fewer tokens can omit important clinical definitions |
| Concurrency/batching | Pending | Client overlap and scheduler capacity are different controls |
| Reasoning/output limits | Pending | Thought tokens consume budget; truncated output is a failure |
| AWQ stretch | No compatible checkpoint validated | Does not block core work |

MVP1/MVP2 used Nemotron/Qwen GGUF on the Mac; MVP3 GPU serving moved to HF checkpoints.
The separate ExecuTorch case used SmolLM2-135M exported to FP32 PTE on M4 Pro CPU, not
this clinical SQL model or workload. Its saved portability measurement is documented in
`tutorial/EXECUTORCH_MAC.md`; it does not provide a Nemotron/vLLM performance comparison.

References: [NVIDIA model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16),
[vLLM hybrid model caveats](https://docs.vllm.ai/en/latest/usage/v1_guide/),
[Mamba cache controls](https://docs.vllm.ai/en/v0.31.0/configuration/engine_args/).
GPU run outcomes, including unsupported cases, must be recorded before claiming support.
''')
    return dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}},
                nbformat=4, nbformat_minor=5)
