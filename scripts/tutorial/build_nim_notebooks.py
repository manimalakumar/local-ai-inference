"""Build two additional MVP4 NIM notebooks; preserve the existing vLLM pair."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.nim_study import MODELS, template

FILENAMES = {'qwen': 'local AI optimization - NIM Qwen.ipynb',
             'nemotron': 'local AI optimization - NIM Nemotron.ipynb'}


def notebook_document(key):
    model = MODELS[key]
    cells = []
    def add(kind, text):
        source = text.strip() + '\n'
        item = dict(cell_type=kind, id=hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                    metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    def md(text):
        add('markdown', text)
    def code(text):
        add('code', text)
    md(f'''# MVP4 NIM optimization — {model['label']}

## 1. Continuing MVP3 → MVP4

The clinical-trial analyst asks operational questions over the same synthetic ten-table
DuckDB database. **{model['label']}** (`{model['model_id']}`) generates a plan and SQL;
the inherited read-only guard, execution, one repair and reference scorer evaluate the
answer. This is an additional NIM notebook alongside the two vLLM optimization notebooks.

We first investigate whether a runnable model-free NIM profile can serve this exact model
on the Dell, then optimize schema context and client concurrency. NIM packages an inference
backend; this recipe targets a **vLLM-backed NIM 2.x model-free image**. It is not a different
model or an automatic TensorRT comparison. No image/profile/GPU compatibility is assumed.
**Live execution defaults to off. Run All performs no Docker, install, download or GPU call.**
''')
    md('''## 2. Workload and measurements

Use the existing four questions `eval_001`, `002`, `005`, `007`. Each configuration runs
serial clinical quality, then timed SQL generation at its chosen client concurrency.
Keep temperature 0, seed 42, output cap 1024, context 8192 and five serving repeats (20
requests). An excluded serving warmup precedes timing. Container startup/readiness is
recorded separately and excluded from request TTFT/E2E.

| Experiment | Fixed deployment | Changed factor |
|---|---|---|
| SQL smoke | Exact staged model/profile | One serving repeat to validate execution |
| Baseline | FP16, cache off, scheduler capacity 4 | Full schema, one client |
| Schema reduction | Same container/profile | Frozen retrieved schema |
| Concurrency 2 | Same full schema/container | Two overlapping client requests |
| Concurrency 4 | Same full schema/container | Four overlapping client requests |

Measure SQL execution/result match, exact-SQL semantic review coverage, prompt tokens,
first-content TTFT p50/p95, request E2E p50/p95, application quality latency, aggregate tok/s,
request failures and sampled peak device GPU memory. Four questions are a functional smoke
test; final quality conclusions need all 23 reference-SQL cases plus review. Small-sample
p95 is descriptive. Total GPU memory includes reserved state/cache and other processes.
''')
    code(f'''from pathlib import Path
import json
import shlex
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import Markdown, display
from clinical_trial_mvp.tutorial import nim_study
from clinical_trial_mvp.tutorial.harness import measurements_markdown
MODEL_KEY = '{key}'
MODEL_ID = '{model['model_id']}'
RUN_TAG = 'nim-{key}-run01'
configs = nim_study.configurations(MODEL_KEY, run_tag=RUN_TAG)
PROFILE_PATH = ROOT / 'tutorial/nim_{key}.local.json'
PROFILE = nim_study.template(MODEL_KEY)
RUN_LIVE = False
RUN_SMOKE = False
LOAD_SAVED = False
results = {{}}
smoke_result = None
''')
    md(f'''## 3. Exact model and environment preflight

Use the same complete local HF snapshot/revision as the **{model['label']}** vLLM notebook.
This recipe mounts it read-only as `/model` with `NIM_MODEL_PATH=/model` and HF downloads
disabled. Select an accessible digest-pinned NIM 2.x vLLM model-free image that supports the
exact architecture and GPU. Neither a similarly named model nor successful native vLLM
serving establishes NIM support. Discovery/startup on the target host provides evidence.

Prepare Docker with NVIDIA GPU access and NGC registry access on the Dell. Credentials
stay in `NGC_API_KEY` and `HF_TOKEN` environment variables; never paste values into notebooks
or profiles. Stop native vLLM before deploying NIM. Create the ignored local profile:

```bash
.venv/bin/python scripts/tutorial/nim_deploy.py init --model {key} --profile tutorial/nim_{key}.local.json
```

Edit that local file: set the image digest, exact model revision, staged snapshot and single
GPU device. Keep the exact model ID and served alias. The template requests FP16, context
8192, cache off and scheduler capacity 4. Fingerprints bind discovery to deployment, but
architecture matching alone cannot prove lineage: record your actual snapshot revision.
''')
    if key == 'nemotron':
        md('''**Nemotron-specific check:** the target is `NVIDIA-Nemotron-3-Nano-4B-BF16`, not
Llama 3.1 Nemotron Nano 4B or the 30B Nemotron 3 Nano. This hybrid Mamba/attention model
uses FP32 Mamba SSM state and trusted reviewed local code in this recipe, while weights/
activations are requested as FP16. The NIM image's actual backend and your GPU must support
these controls. Missing support is a recorded outcome, not permission to substitute a model.''')
    code('''if PROFILE_PATH.exists():
    PROFILE = json.loads(PROFILE_PATH.read_text())
print('Requested model:', MODEL_ID)
print('Image:', PROFILE['image'])
print('Snapshot:', PROFILE['snapshot'])
print('GPU device:', PROFILE['gpu_device'])
print('Selected profile:', PROFILE['profile_id'] or 'pending discovery and explicit selection')
''')
    md(f'''## 4. Discover and select a runnable profile

After configuring the local file, preview first, then run on the Dell:

```bash
.venv/bin/python scripts/tutorial/nim_deploy.py discover --profile tutorial/nim_{key}.local.json
.venv/bin/python scripts/tutorial/nim_deploy.py discover --profile tutorial/nim_{key}.local.json --execute
```

Discovery saves sanitized output/errors, actual GPU name/UUID and the model/image/config
contract. Only IDs under **Compatible with system and runnable** qualify. Inspect the full
profile description and copy the chosen complete ID into `profile_id`. Low-memory and
incompatible profiles cannot run this experiment. Selection is explicit; verify that the
image/profile can honor the requested FP16 configuration and model architecture.
If no profile can run, retain the discovery evidence and continue the vLLM work independently.
''')
    code('''if PROFILE_PATH.exists():
    PROFILE = json.loads(PROFILE_PATH.read_text())
    print('Discovery succeeded:', PROFILE.get('discovery_succeeded', False))
    print('Runnable IDs:', PROFILE.get('runnable_profile_ids', []))
    print('Selected:', PROFILE.get('profile_id') or 'pending explicit selection')
else:
    print('Profile pending. Offline experiment plans remain available.')
''')
    md('''## 5. Deploy and smoke-test SQL

Run the printed deployment command in a terminal. It starts one named background container,
binds only loopback port 8000, waits for readiness and checks the served model alias through
`/v1/models`. Metadata is reserved before launch and never overwrites a prior deployment.
A changed model, image, revision, configuration or GPU requires fresh discovery.
Retain startup logs and review resolved backend settings: profile defaults can differ from
requested CLI values. Container/profile identity and staged weight identity are separate.

NIM is not assumed equivalent to native vLLM. Backend version, effective configuration and
thinking behavior need separate verification before a causal speed comparison. The adapter
uses standard OpenAI fields and the common `/no_think` prompt only; it does not claim a
backend-specific thinking switch was applied. Inspect the template/engine behavior.
''')
    code('''deploy_command = ['.venv/bin/python', 'scripts/tutorial/nim_deploy.py', 'deploy',
                  '--profile', str(PROFILE_PATH.relative_to(ROOT)),
                  '--metadata', configs['baseline']['provenance'], '--execute']
print(shlex.join(deploy_command))
print('After ready: set RUN_LIVE=True and RUN_SMOKE=True, then execute the next cell.')
''')
    code('''smoke_result = nim_study.run(configs['smoke'], execute=RUN_LIVE and RUN_SMOKE, root=ROOT)
results['smoke'] = smoke_result
print(smoke_result['status'], smoke_result.get('reason', ''))
display(Markdown(measurements_markdown([smoke_result])))
''')
    md('''## 6. Measure baseline, schema reduction and concurrency

After a completed smoke with `smoke_passed=True`, keep the same resident deployment and run
the following cells. Full/retrieved schema intentionally changes context; concurrency 1/2/4
holds the full schema and scheduler capacity 4 fixed. This is client overlap under dynamic
batching, not a measurement of achieved server batch occupancy. A smoke success means SQL
executed, not that every clinical answer is correct; reference and semantic review still apply.
''')
    code('''def run_one(name):
    execute = RUN_LIVE and bool(smoke_result and smoke_result.get('status') == 'completed'
                               and smoke_result.get('smoke_passed'))
    result = nim_study.run(configs[name], execute=execute, root=ROOT)
    if RUN_LIVE and not execute:
        result['reason'] = 'Complete the NIM SQL smoke test before benchmark configurations.'
    results[name] = result
    print(name, result['status'], result.get('reason', ''))
    display(Markdown(measurements_markdown([result])))
''')
    for name in ('baseline', 'schema_retrieved', 'concurrency_2', 'concurrency_4'):
        code(f"run_one('{name}')")
    md('''## 7. Conditional cache and precision extensions

These controls are not silently enabled by the core NIM study:

- Prefix cache off/on requires actual profile/backend support and a fresh deployment record.
  Verify proxy access to a reset API returning `success=true` before using the shared-schema
  cold/warm protocol. If NIM hides that endpoint, cold/warm results remain unavailable.
- AWQ W4A16 needs a validated compatible derivative of the exact base revision, matching
  tokenizer/template lineage and a supporting NIM profile/kernel plus successful smoke.
  FP8, NVFP4 and GGUF are different experiments. Nemotron AWQ remains stretch work.
- Reasoning and backend batch-capacity sweeps require exposed supported controls and new
  verified deployment settings. Prompt-only control does not establish thinking parity.

The initial deployment helper accepts only staged unquantized FP16 weights. These extensions
need a separate validated deployment recipe; they are not reported as completed here.
Record unavailable features rather than treating them as zero performance.
''')
    md('''## 8. Results and deployment decision

Reload matching dedicated NIM records with `LOAD_SAVED=True`. Keep failed/skipped outcomes.
Application quality E2E includes planning/execution/repair; serving TTFT/E2E is request
transport. Startup/readiness is separate. Unknown memory stays unknown, and GPU allocation
includes reserved cache/state. A four-case score cannot establish general clinical quality.

For native vLLM versus NIM, first verify the same model/tokenizer fingerprints, precision,
GPU, database/date, prompts, seed, output cap, cache/SSM state and thinking behavior. Inspect
bundled backend versions and effective settings too. Until then, label it a deployment
comparison and do not attribute a speed change to packaging. No winner is established until
real GPU measurements and quality review exist.
''')
    code('''if LOAD_SAVED:
    for path in sorted((ROOT / 'experiment-tracker/mvp4/nim/runs').glob('*.json')):
        saved = json.loads(path.read_text())
        for name, spec in configs.items():
            actual = saved.get('configuration', {})
            if actual.get('experiment') == spec['experiment'] and actual.get('provenance') == spec['provenance']:
                results[name] = saved
display(Markdown(measurements_markdown(list(results.values()))))
for name, result in results.items():
    if result.get('quality'):
        q = result['quality']
        print(name, 'application E2E p50/p95 s:', q.get('latency_p50_seconds'), q.get('latency_p95_seconds'))
    if result.get('server'):
        print(name, 'startup/readiness s:', result['server'].get('readiness_seconds'))
''')
    md(f'''## 9. Learning journal and cleanup

| Situation | Outcome | Next action |
|---|---|---|
| Exact {model['label']} architecture/GPU compatibility | Pending | Discover actual image profiles; never substitute a similar model |
| Precision/CLI overrides | Pending | Inspect resolved startup configuration |
| Clinical smoke and semantic review | Pending | Execute and review exact SQL outputs |
| Schema reduction | Pending | Audit lost clinical definitions |
| Concurrency | Pending | Distinguish client overlap and server capacity |
| Cache/precision extensions | Conditional | Verify exposed controls before adding experiments |

Stop this NIM before returning to native vLLM:

```bash
docker logs clinical-trial-nim-{key}
docker stop clinical-trial-nim-{key}
```

Retain the local profile, immutable metadata and raw records. Copy evidence for review;
exclude credentials and weights from Git.

References: [profile discovery](https://docs.nvidia.com/nim/large-language-models/latest/profiles.html),
[model-free configuration](https://docs.nvidia.com/nim/large-language-models/latest/reference/advanced-configuration.html),
[NIM 2.x model-path migration](https://docs.nvidia.com/nim/large-language-models/latest/reference/1.x-migration-guide.html).
See [MVP4 NIM run instructions](../tutorial/NIM_MVP4.md).
''')
    return dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}},
                nbformat=4, nbformat_minor=5)


if __name__ == '__main__':
    for key, filename in FILENAMES.items():
        (ROOT / 'notebooks' / filename).write_text(json.dumps(notebook_document(key), indent=1) + '\n')
        (ROOT / 'tutorial' / f'nim_{key}.example.json').write_text(json.dumps(template(key), indent=2) + '\n')
        print(filename)
