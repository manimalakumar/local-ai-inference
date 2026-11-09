"""Four learner chapters; replay by default, with visible serving/evaluation mechanics."""
import hashlib
from clinical_trial_mvp.tutorial import nemotron, nim_study, optimization

FILENAMES = {
    ('vllm', 'qwen'): '01_qwen_cache_quantization.ipynb',
    ('vllm', 'nemotron'): '02_nemotron_optimization.ipynb',
    ('nim', 'qwen'): '03_qwen_nim.ipynb',
    ('nim', 'nemotron'): '04_nemotron_nim.ipynb',
}
ARCHIVE_FILENAMES = {
    ('vllm', 'qwen'): 'local AI optimization - Qwen.ipynb',
    ('vllm', 'nemotron'): 'local AI optimization - Nemotron.ipynb',
    ('nim', 'qwen'): 'local AI optimization - NIM Qwen.ipynb',
    ('nim', 'nemotron'): 'local AI optimization - NIM Nemotron.ipynb',
}


def notebook_document(runtime, model):
    cells = []
    def add(kind, source):
        source = source.strip() + '\n'
        cell = dict(cell_type=kind, id=hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                    metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    md = lambda text: add('markdown', text)
    code = lambda text: add('code', text)
    label = nim_study.MODELS[model]['label']
    model_id = nim_study.MODELS[model]['model_id']
    first = 'baseline' if runtime == 'nim' else 'fp16_baseline' if model == 'nemotron' else 'fp16_cache_off'
    chapter = ('Prefix caching and quantization' if model == 'qwen' else 'Context, concurrency and reasoning') if runtime == 'vllm' else f'{label} through NVIDIA NIM'
    md(f'''# {chapter}

*Local AI Inference Lab: Serving, Optimization and Application Quality*

**Question:** How do we make local AI faster and more efficient without losing useful task performance?

**Learn:** {'separate prefill/decode and test FP16/AWQ × caching with paired cold/warm requests' if runtime == 'vllm' and model == 'qwen' else 'investigate hybrid state, context, concurrency/capacity and reasoning budgets' if runtime == 'vllm' else 'inspect NIM profiles, packaged serving and application request contracts'}.
**Requirements:** [CPU replay prerequisites](../SETUP.md); [Live configuration](../tutorial/RUN_ALL.md) only for fresh GPU runs.
**Expected outputs:** recorded tables, a server command/request/evaluated SQL example,
and an observation → explanation → limitation → decision for every experiment.

{'The recorded Qwen FP16 cache-on run has cold/warm median TTFT of **2.019 → 0.083 s**, while request E2E changes **6.038 → 4.039 s**. Prefill reuse does not eliminate generation time. New to the workload? Read [00 — local serving](00_local_serving.ipynb), or start here for optimization.' if runtime == 'vllm' and model == 'qwen' else 'This second chapter applies the same quality-first method to a hybrid model. Read the Qwen chapter for the common mechanics.' if runtime == 'vllm' else 'This optional extension applies the same quality-first method to packaged deployment. Read the Qwen/vLLM chapter first and use the [shared NIM technical guide](../tutorial/NIM_MVP4.md).'}

These are observations from a small recorded Dell workload, not universal claims. A faster
response is not necessarily a better answer; different optimizations affect different stages.
Clinical-trial analytics supplies an executable application-quality test. We have not evaluated
multiple application domains. The synthetic database contains no patient data.
''')
    md(f'''## 1. Choose how to explore

| Mode | Behavior |
|---|---|
| **Replay — default** | Load checksum-verified **recorded Dell measurements** and reproduce tables without a GPU. |
| Plan | Show configuration and server commands without launching or downloading anything. |
| Live | Launch owned servers, benchmark, preserve evidence and clean up. Requires Linux GPU setup. |

Choose **Run All** now for replay. To change mode, create `tutorial/config/{runtime}-{model}.local.json`
using [the shared setup guide](../tutorial/RUN_ALL.md) and set `mode` to `replay`, `plan` or `live`.
All machine settings and flags belong in this configuration file. A live failure never silently
loads archived results. Replay ignores new experiment settings when presenting historical values.

Serving target: **{label}**, `{model_id}`. The checkpoint is loaded as **FP16** through
{'native vLLM in a separate GPU Python environment' if runtime == 'vllm' else 'a vLLM-backed NIM model-free container with a read-only local checkpoint mount'}.
{'The hybrid Mamba/attention architecture retains FP32 SSM state. Its BF16 source label does not change the requested FP16 experiment.' if model == 'nemotron' else 'The AWQ W4A16 candidate uses 4-bit weights and FP16 activations. The KV cache is not quantized by this weight format.' if runtime == 'vllm' else 'This chapter uses unquantized FP16 weights; its initial NIM recipe does not run AWQ variants.'}

Four distinct clinical questions are repeated five times per serving condition. They are not
20 independent clinical tasks. Application quality separately evaluates four cases, including
planning, validation, SQL execution and one repair. Exact-SQL semantic review is still pending.
'''.replace('\n+', '\n'))
    code(f'''from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import JSON, Markdown, display
from clinical_trial_mvp.tutorial.workflow import Study
from clinical_trial_mvp.tutorial.teaching import server_example, request_example, response_example, interpret
if 'study' in globals():
    study.close()
study = Study('{runtime}', '{model}', root=ROOT)
results = study.results
print('Mode:', study.mode)
print('Model:', '{model_id}')
print(study.prepare())
''')
    if runtime == 'vllm' and model == 'qwen':
        md('''## 2. Predict before measuring

Prefill processes the prompt; decode generates output tokens. A cache can reuse shared
prefix blocks. Quantization changes weight representation and kernels. Shorter context
changes both input work and the information available. More concurrency can improve
aggregate throughput while increasing per-request latency.

The application checks plan structure, SQL safety, schema use, execution and reference-result
match. Endpoint HTTP success alone says nothing about whether a useful answer was delivered.
Before each experiment, predict both its systems effect and its possible quality cost.
''')
    else:
        md('''## 2. Predict before measuring

Recap: serving speed and application quality are different outcomes. Use
[chapter 01](01_qwen_cache_quantization.ipynb) for prefill/decode, W4A16 and common metric
definitions. Here, focus on the model/deployment differences below and predict their effects.
''')
    if runtime == 'vllm' and model == 'qwen':
        md('''| | Cache off | Cache on |
|---|---|---|
| FP16 | Baseline | Prefix reuse |
| AWQ W4A16 | Weight representation | Combined effects |

Hold schema, four questions, seed 42, temperature 0, cap 1024 and context limit 8192 fixed.
An AWQ SQL compatibility smoke precedes all four live variants. For cache-on, each target
uses **reset → cold target → reset → different-question prime → identical target warm**,
with the identical shared schema prefix. Resets must acknowledge `success=true`.
Startup, health checks, warmup, reset, prime and SQL scoring are outside request timing.

The new native controller sets `VLLM_SERVER_DEV_MODE=1` consistently to expose reset controls.
The original reset-404 failure remains part of the evidence. See [its case study](../tutorial/CASE_STUDIES.md).
''')
    elif runtime == 'vllm':
        md('''Establish the FP16 baseline, then investigate verified hybrid prefix caching,
retrieved context, concurrency 1/2/4 at scheduler capacity 4, capacity 1 with four clients,
reasoning on, and output caps 256/2048. Client overlap and scheduler capacity are distinct.
Reasoning tokens consume output budget, and TTFT includes the first thought token.

Nemotron core has no AWQ dependency. AWQ is optional stretch work and never gates core runs.
Set `prefix_supported=true` only after verifying the installed hybrid backend. Unsupported
cache/AWQ rows are unavailable or skipped, not zero performance. No compatible Nemotron
AWQ artifact was obtained; its absence does not invalidate completed core experiments.
''')
    else:
        md(f'''NIM packages an inference backend; it does not imply a TensorRT conversion.
Select your own runnable non-LoRA profile after model/image/GPU discovery. TP1/PP1 means one
GPU and one pipeline stage. Startup still has to succeed after a runnable profile listing.
The core study changes schema and client concurrency while reusing a fixed deployment.

{'The original Qwen NIM smoke had 4/4 transport successes but 0/4 application SQL executions: every plan used an object for `filters`, while the application requires a string. The constrained rerun executed 3/4. The reusable config enables schema enforcement and records that setting.' if model == 'qwen' else 'The recorded Nemotron NIM runs used schema enforcement off; this remains its configurable default. Hybrid state and trusted local-code requirements must also be honored by the container backend.'}

Thinking parity with native vLLM remains unverified. The common `/no_think` prompt is not proof
of identical backend behavior. NIM versus native results are deployment observations until
model, request contracts, engine settings, date/database and thinking controls are matched.
Cache/AWQ extensions remain conditional; no new runtime is added to this lab.
''')
    md('''## 3. Compatibility before benchmarking

Smoke asks whether the path can generate and execute SQL. It is not a general quality score.
Replay loads the recorded smoke when available. Plan makes no model calls. Live stops on
failed required smoke, retains the failure and never substitutes replay measurements.
''')
    code('''smoke = study.smoke()
if isinstance(smoke, dict):
    print(smoke.get('_recorded', {}).get('label', study.mode), smoke.get('status'), smoke.get('reason', ''))
else:
    print(smoke)
''')
    names = (['baseline', 'schema_retrieved', 'concurrency_2', 'concurrency_4'] if runtime == 'nim' else
             list(nemotron.configurations()) if model == 'nemotron' else list(optimization.VARIANTS))
    md('## 4. Experiments and decisions\n\nRead each prediction before running its cell. Quality counts accompany speed; unavailable metrics stay unavailable.')
    questions = {
        'fp16_cache_off': 'What does an FP16 request cost without prefix caching?',
        'fp16_cache_on': 'Can shared-prefix reuse reduce first-token latency?',
        'awq_w4a16_cache_off': 'How does W4A16 change latency and task results?',
        'awq_w4a16_cache_on': 'Do caching and W4A16 help together?',
    }
    groups = {
        'fp16_baseline': 'Input work: full/retrieved schema and prefix caching',
        'concurrency_2': 'Serving capacity: clients and scheduler limits',
        'reasoning_on': 'Output work: reasoning and token caps',
        'awq_stretch': 'Optional precision stretch: compatible checkpoint required',
    }
    for i, name in enumerate(names):
        if runtime == 'vllm' and model == 'nemotron' and name in groups:
            md('### ' + groups[name] + '\n\nEach row is its own controlled comparison. No cumulative best configuration was measured.')
        md(('#### ' if runtime == 'vllm' and model == 'nemotron' else '### ') +
           (questions.get(name, name.replace('_', ' ')) if runtime == 'vllm' and model == 'qwen' else name.replace('_', ' ')))
        code(f'''result = study.run('{name}')
print(result.get('_recorded', {{}}).get('label', study.mode), '{name}:', result['status'])
display(Markdown(interpret(study, '{name}')))
if study.mode == 'plan':
    display(JSON(server_example(study, '{name}')))
''')
        if i == 0:
            md(f'''### Under the hood: three inspectable paths

**A. Server command.** The next cell shows the recorded portable argv or the planned command,
model revision, dtype, cache control and scheduler limits. A setting absent from the archive
is marked unknown rather than inferred.
{'Raw Docker argv was not captured; the shown shape is reconstructed from metadata. [NIM deployment](../src/clinical_trial_mvp/tutorial/nim_study.py).' if runtime == 'nim' else '[Native launcher](../src/clinical_trial_mvp/serving/launch.py).'}
''')
            code(f"display(JSON(server_example(study, '{first}')))")
            md('''**B. One HTTP request.** Inspect the full prompt, generation options and schema
contract. This uses the exact payload builder called by the transport. Replay reconstructs
it from the saved prompt and recorded flags; it is not a packet capture. No request is sent
by this inspection cell. [Request builder and timing](../src/clinical_trial_mvp/serving/client.py).
''')
            code(f"display(JSON(request_example(study, '{first}')))")
            md('''**C. One evaluated response.** Inspect generated SQL, safety/schema validation,
execution outcome, query rows, the reference query and the recorded result comparison.
A reference match is not a human semantic review. [Plan/SQL validation](../src/clinical_trial_mvp/agent/analyst.py)
· [reference scorer](../src/clinical_trial_mvp/evaluation/scorers.py).
''')
            code(f"display(JSON(response_example(study, '{first}')))")
    md('''## 5. Put performance beside quality

TTFT is first generated content. Request E2E measures generation/transport; application E2E
includes planning, validation, execution and repair. Cold/warm cache phases are separate from
supplemental warmed summaries. Semantic review remains unreviewed. Native VRAM telemetry
failed and stays missing; no memory-reduction conclusion is supported by those runs.
''')
    code('display(Markdown(study.report()))\ndisplay(Markdown(study.cache_report()))')
    if runtime == 'vllm' and model == 'qwen':
        md('''### Check output work before explaining AWQ latency

Compare output-token counts by phase and case before attributing E2E differences entirely
to faster computation. The next table uses the same target records as the paired cache
results. Its decode rate is a client estimate, not GPU kernel timing. Generated SQL/text
can differ even at similar token lengths.
''')
        code('''from clinical_trial_mvp.tutorial.token_analysis import analyze, table
token_analysis = analyze(study.results)
display(Markdown(table(token_analysis)))
''')
        md('''### Filled interpretation of the historical Qwen comparison

**Observation:** with caching enabled, FP16 cold/warm TTFT was **2.019 → 0.083 s** and
E2E was **6.038 → 4.039 s**; AWQ was **2.083 → 0.057 s** and **4.271 → 2.258 s**.
Direct reference matches stayed **10/20** in each FP16 phase and **15/20** in each AWQ phase.
**Explanation:** prefix reuse is consistent with less prefill work; AWQ also changed the
generation path and actual SQL. **Limitation:** this small historical workload does not
establish general speed/quality gains, reviewed semantics or native memory savings.
**Decision:** test caching for repeated schemas; adopt a validated AWQ checkpoint only after
checking useful task results and latency on the target machine. These conclusions interpret
the archive; any fresh Live run must be assessed using its own tables.

In the recorded runs, mean output lengths are FP16/AWQ **152/151.5 tokens cold**
and **150.75/151.5 warm**. Shorter output alone is not a sufficient aggregate explanation.
Similar lengths do not establish identical token content or isolate GPU compute.
[Per-case token audit and interpretation](../tutorial/RESULTS.md#output-length-before-awq-attribution).
Fresh live counts appear in the table above; these quoted values remain historical.
''')
    if runtime == 'vllm':
        figure = '01_prefix_cache.png' if model == 'qwen' else '02_concurrency.png'
        md(f'''### Recorded evidence figure

![Recorded Dell measurement](../tutorial/figures/{figure})

This figure always describes the archived Dell runs, including when the notebook is in live
mode. Fresh measurements appear in the tables above. [Figure sources](../tutorial/figures/figure_data.json)
· [reproduction script](../scripts/tutorial/render_findings.py).
''')
    md(f'''## 6. Finish, preserve and continue

Save with Ctrl+S. Replay changes only the notebook display; it does not rewrite measurements.
Live adds unique records and owns only the processes it starts.
{'Run the cleanup cell if pausing between successful live NIM cells.' if runtime == 'nim' else 'Native servers stop after each row; the final cleanup cell is safe to repeat.'}
Interrupted/failed operations clean up owned servers; a kernel/OS crash requires explicit
cleanup using the saved {'container ID' if runtime == 'nim' else 'process group'}.
''')
    code('print(study.close())')
    next_chapter = ('02_nemotron_optimization.ipynb' if (runtime, model) == ('vllm', 'qwen') else
                    '03_qwen_nim.ipynb' if runtime == 'vllm' else
                    '04_nemotron_nim.ipynb' if model == 'qwen' else '05_edge_portability_executorch.ipynb')
    md(f'''Continue to [the next chapter]({next_chapter}), or read:

- [What should I choose?](../tutorial/RESULTS.md#what-should-i-choose) — concise results and deployment decisions.
- [Troubleshooting case studies](../tutorial/CASE_STUDIES.md).
- [Four-case SQL review packet](../tutorial/SQL_REVIEW.md) — human review pending.
- [Fresh GPU validation checklist](../tutorial/GPU_VALIDATION.md).
- [Original executed notebook](history/dell-2026-11-07/{ARCHIVE_FILENAMES[(runtime, model)].replace(' ', '%20')}) and [checksum manifest](history/dell-2026-11-07/MANIFEST.json).

Historical iteration labels and source lineage are preserved in the evidence, not used as
the learner-facing chapter structure.
''')
    return dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
