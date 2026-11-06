# MVP4: two additional model-specific NIM notebooks

MVP4 continues the cumulative clinical-trial workload through MVP3. It now has four current
notebooks: the existing Qwen/Nemotron vLLM studies and these two additional NIM studies:

- [NIM Qwen3-4B](../notebooks/local%20AI%20optimization%20-%20NIM%20Qwen.ipynb), targeting exactly `Qwen/Qwen3-4B`.
- [NIM Nemotron 3 Nano 4B](../notebooks/local%20AI%20optimization%20-%20NIM%20Nemotron.ipynb), targeting exactly `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.

Do not substitute Llama 3.1 Nemotron Nano 4B, Nemotron 30B, or another Qwen variant if the
requested model lacks support. These notebooks investigate compatibility before benchmarking;
neither exact model/profile/GPU combination has been validated on the Dell yet.

## Scope and continuity

Reuse the same complete local HF snapshot and exact revision as the corresponding vLLM
notebook, the synthetic ten-table DuckDB database, four fixed clinical questions, reference
SQL scorer and inherited plan/SQL/repair path. The NIM baseline requests FP16, full schema,
cache off, scheduler capacity 4 and one client. Hold the container/profile fixed while testing
full vs frozen retrieved schema and concurrency 1/2/4. Model startup and readiness are
recorded separately from request timing. Native vLLM and NIM are run one at a time.

This recipe deliberately targets **NIM 2.x model-free images with a vLLM backend**. The
documented image family is `nvcr.io/nim/nvidia/vllm-model-free-nim`; select an accessible
version, inspect its supported architectures/GPU requirements, and pin its actual digest.
Do not use an unpinned `latest` tag in a recorded profile. NIM is a packaged inference stack;
a vLLM-backed NIM versus native vLLM comparison is not automatically a different-engine test.
The [model-free image guidance](https://docs.nvidia.com/nemo-helix/latest/documentation/customizer-reference/customization-concepts)
and [NIM model-free configuration](https://docs.nvidia.com/nim/large-language-models/latest/reference/advanced-configuration.html)
describe that interface. The helper uses `NIM_MODEL_PATH` rather than the retired NIM 1.x
`NIM_MODEL_NAME` variable; see [the migration guide](https://docs.nvidia.com/nim/large-language-models/latest/reference/1.x-migration-guide.html).

## Dell prerequisites

Use the existing `.venv` notebook/CPU test environment; no new Python packages are needed.
Install/configure Docker and NVIDIA Container Toolkit on the target Linux GPU host. Ensure
Docker can access the selected GPU and that your account can pull the chosen NGC image.
Configure registry authentication separately. Pass `NGC_API_KEY` and optional `HF_TOKEN`
through terminal environment variables; never store values in notebook cells or JSON.
The helper passes environment names to Docker and redacts their values from discovery errors.

Local model snapshots are mounted read-only under `/model`, with HF downloads disabled.
Stage all model, tokenizer and required custom-code files before running discovery. NIM
may pull the selected image when explicitly executed; the notebook's default Run All never
launches Docker or performs any install/download/GPU/network action. Image/profile startup
can still fail because of actual architecture, driver, GPU memory or kernel limitations.

## Qwen: configure → discover → select → deploy

From the cloned repository root:

```bash
.venv/bin/python scripts/tutorial/nim_deploy.py init --model qwen --profile tutorial/nim_qwen.local.json
```

Edit `tutorial/nim_qwen.local.json`. Set the image to its digest-pinned reference, the exact
model revision, the existing `models/qwen3-4b-frozen` snapshot location and the single GPU
index/UUID. Keep the target model ID and served alias. The local file is ignored by Git.

Preview, then explicitly execute discovery:

```bash
.venv/bin/python scripts/tutorial/nim_deploy.py discover --profile tutorial/nim_qwen.local.json
.venv/bin/python scripts/tutorial/nim_deploy.py discover --profile tutorial/nim_qwen.local.json --execute
```

Inspect the saved discovery output. Copy a suitable complete ID from **Compatible with
system and runnable** into `profile_id`; selection is never automatic. Low-memory and
incompatible profiles cannot qualify. Choose a profile that can honor the requested precision
and backend settings. If none exists, retain the discovery evidence and leave this study
unmeasured. Changing model/image/revision/config or GPU requires fresh discovery.

Open the Qwen NIM notebook and choose a unique `RUN_TAG` in its setup cell. Section 5 prints
the exact deployment command and metadata path for that tag. An example for its default tag:

```bash
.venv/bin/python scripts/tutorial/nim_deploy.py deploy \
  --profile tutorial/nim_qwen.local.json \
  --metadata experiment-tracker/mvp4/nim/provenance/qwen-nim-qwen-run01.json \
  --execute
```

The helper reserves a new metadata file, starts `clinical-trial-nim-qwen`, waits for
`/v1/health/ready`, verifies the alias in `/v1/models`, and records readiness time separately.
It binds only `127.0.0.1:8000`. Keep the actual GPU UUID and deployment evidence. Inspect
resolved backend settings in the startup log; explicit CLI requests may interact with profile
defaults. Do not overwrite an old metadata file. A startup failure is recorded with no
hardware-success claim; inspect/stop the container before trying a new record.

In Jupyter, run setup and profile inspection cells, then set `RUN_LIVE=True` and
`RUN_SMOKE=True` in a new cell without resetting setup state. Run the smoke cell first.
Only a completed SQL smoke with `smoke_passed=True` enables baseline, context-reduction and
concurrency cells. The smoke runs the four clinical quality cases and one serving repeat;
it is not a final quality conclusion. Save the notebook after runs.

## Nemotron: same sequence, exact architecture

```bash
.venv/bin/python scripts/tutorial/nim_deploy.py init --model nemotron --profile tutorial/nim_nemotron.local.json
.venv/bin/python scripts/tutorial/nim_deploy.py discover --profile tutorial/nim_nemotron.local.json
.venv/bin/python scripts/tutorial/nim_deploy.py discover --profile tutorial/nim_nemotron.local.json --execute
```

Configure the file between init and discovery as for Qwen. Stage the same Nemotron HF base
snapshot used by the native vLLM study. The template adds trusted reviewed local model code
and FP32 Mamba SSM state while requesting FP16 weights/activations. The selected image's
backend and actual Dell GPU must support this hybrid architecture and these controls.
Do not infer NIM compatibility from successful native vLLM inference. Select a runnable
profile explicitly, open the Nemotron NIM notebook, and use its printed deployment command.
Its container name is `clinical-trial-nim-nemotron`. Stop the Qwen/native server first.

## What is measured and what stays conditional

The shared harness records SQL execution/result match, exact-SQL semantic review coverage,
prompt tokens, TTFT p50/p95, request E2E p50/p95, application latency, aggregate output tok/s,
failures, raw model/SQL attempts and sampled device GPU memory. Warmup is excluded from
serving timing. Memory includes resident/reserved cache/state and other GPU processes;
unknown is not zero. Four questions are a smoke workload; the full dataset has 23 executable
reference questions and 30 total cases. Semantic correctness remains unreviewed until
explicit exact-SQL human judgements exist.

The NIM adapter sends standard OpenAI fields and the common `/no_think` instruction only.
Thinking-mode parity with native backend-specific controls is **unverified**, especially for
hybrid Nemotron. Do not attribute latency differences to NIM packaging without establishing
thinking, backend version, effective settings, model/tokenizer, precision, GPU, data/date and
prompt parity. Deployment identity alone is insufficient. Records explicitly label this as
a deployment comparison pending that verification.

Prefix caching/cold-warm, AWQ/other precision profiles, reasoning toggles and backend batch
capacity are conditional extensions, not silently implemented claims. The initial helper
intentionally accepts only unquantized FP16. Cache-on requires verified profile controls,
new discovery/deployment evidence, and a proxy-exposed reset API that confirms success before
the shared-prefix cold/warm protocol can run. AWQ requires an exact validated derivative and
supporting profile/kernel, with a separate smoke; it remains stretch for Nemotron. Add those
recipes only after the baseline establishes which controls the actual NIM exposes.

## Evidence, tests and cleanup

Dedicated deployment metadata: `experiment-tracker/mvp4/nim/provenance/*.json`.
Dedicated benchmarks/failures/skips: `experiment-tracker/mvp4/nim/runs/*.json`.
The inherited harness also retains its ordinary successful hardware run copy. Keep local
profile/discovery files with the run evidence when transferring results, excluding credentials.
Each notebook can reload matching run records using `LOAD_SAVED=True`.

```bash
docker logs clinical-trial-nim-qwen
docker stop clinical-trial-nim-qwen
```

Use `clinical-trial-nim-nemotron` for the other model. Save logs before stopping a `--rm`
container. Stop NIM before returning to native vLLM or running the other NIM notebook.

CPU checks and reproducible builders:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/tutorial/build_nim_notebooks.py
```

The builder rewrites only the two NIM notebooks and their example templates; do not invoke
it to preserve GPU cell outputs. Tests fake Docker/inference, validate deployment/profile
binding and failures, run real read-only DuckDB scoring, and execute all default NIM cells
without GPU imports or external calls. They do not establish real NIM/GPU compatibility.
