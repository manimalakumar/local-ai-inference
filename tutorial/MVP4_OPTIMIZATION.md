# Qwen vLLM optimization: inherited four-variant workflow

For Nemotron core/stretch experiments, use the separate Nemotron notebook and [setup guide](../SETUP.md).

Open [local AI optimization](../notebooks/local%20AI%20optimization%20-%20Qwen.ipynb).
This separate notebook stays in the existing repo and reuses its clinical questions,
prompt builder, SQL validator, read-only DuckDB execution and reference scorer.

## Model/runtime history, including ExecuTorch

| Stage | Runtime / hardware | Artifact |
|---|---|---|
| MVP1 | Mac local inference / Ollama | GGUF, including Nemotron 4B Q4_K_M |
| MVP2 | llama.cpp and Ollama / M4 Pro | Same Nemotron Q4_K_M GGUF |
| MVP3 | Mac paths; GPU vLLM/SGLang | GGUF on Mac; HF checkpoints on GPU |
| ExecuTorch portability case | Native ExecuTorch / M4 Pro CPU, XNNPACK | HF SmolLM2-135M converted/exported to FP32 `.pte`; no GGUF |
| MVP4 optimization | vLLM / target GPU | HF FP16 and AWQ W4A16; measurements pending |

The [saved ExecuTorch measurement](../experiment-tracker/tutorial/edge/20261029T161510_034051Z.json)
from October 29, 2026 used `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, exported to an XNNPACK PTE and
executed through the native C++ text runner. The planned gated Llama 3.2 model
was not used. Measured native load: **0.076 s**; request inference including prompt
evaluation: **0.199 s**; whole process: **3.480 s**; model + tokenizer: **623.60 MiB**;
sampled peak runner RSS: **1018.13 MiB**.

This was one warm-filesystem CPU text-completion portability smoke test. It did
not measure clinical SQL correctness or TTFT. RSS is process memory, not GPU VRAM.
The model, precision, backend and workload differ from MVP4, so these values do
not establish a paired speed comparison. ExecuTorch remains a separate historical
portability case; the optimization experiment still has four vLLM variants.
See [reproduction, export validation and limitations](EXECUTORCH_MAC.md).

| Weights / activations | Cache off | Cache on |
|---|---|---|
| FP16 / FP16 | Baseline | Cold + warm |
| AWQ INT4 / FP16 (W4A16) | Quantization effect | Cold + warm |

The notebook teaches prefill/decode and weights/KV cache before introducing the
controlled experiment. Its 11 sections cover preflight, compatibility, the four
variants, quality, deployment decisions and a learning journal.

## Run on the GPU machine

1. Use the existing [setup guide](../SETUP.md) and working pinned vLLM environment.
   Run Jupyter on the GPU host for GPU-memory sampling. The notebook environment
   uses `requirements-notebook.txt`; GPU engines stay in their separate environment.
2. Set local `FP16_SNAPSHOT`, `AWQ_SNAPSHOT`, `MODEL_ID`, `BASE_REVISION` and
   `AWQ_SOURCE`. Record exact common model lineage and quantization derivation.
   Set a fresh `RUN_TAG`, then enable `VALIDATE_CHECKPOINTS`. This validates 4-bit
   AWQ, unquantized baseline, matching model shapes and identical tokenizer/template
   files, and hashes both weight snapshots. No weights are downloaded or converted.
3. Section 4 prints a separate AWQ smoke launch command. Run it in a terminal from
   the repo root. Wait for readiness, inspect startup logs, and ensure launch metadata
   contains the actual GPU UUID. Set `ACTIVE='awq_smoke'`, `RUN_LIVE=True` in the
   notebook session, then execute the smoke cell. This makes one SQL generation and
   executes it with the existing read-only guard. Completion/execution failure blocks
   all four variants and saves the error. SQL result equivalence is also recorded;
   compatibility success alone is not a quality conclusion.
4. Stop the smoke server. Section 4 also prints the four experiment launch commands.
   Launch **FP16 off → FP16 on → AWQ off → AWQ on**, one resident server at a time.
   Wait for readiness, confirm provenance matches that running server, then set
   `ACTIVE` to the exact variant and run its section 5–8 cell. Keep `RUN_LIVE=True`.
   Change these variables in the session without rerunning the setup cell, which
   resets in-memory results. Do not run all cells against one server. Other variant
   cells stay offline when `ACTIVE` does not match.
5. Sections 9–10 show per-phase measurements and paired comparisons. Keep all raw
   records. Use a new tag for another fresh set of launches and reverse order to
   examine variability. Section 11 records failures, fixes and transferable lessons.

Every run checks successful AWQ smoke evidence against the same checkpoint pair,
GPU, engine version, Torch/CUDA build and visible-device configuration. A resumed
notebook can reload the exact successful smoke file using the commented line in
section 4. The smoke server uses a different provenance filename from the four
experiment servers. Never overwrite an existing launch record.

## Cold and warm protocol

All variants now use the **same full schema**, replacing question-specific retrieved
contexts in this experiment. The existing SQL prompt builder places identical rules,
schema, clinical guidance and fixed plan placeholder before the unique question.
Prompts and shared-prefix hashes are saved. Concurrency is fixed at 1, temperature 0,
seed 42, context 8192, output cap 1024, and repeats 5 over four existing questions.

After readiness, one excluded request warms kernels/connection. For every cache-on
trial, with no concurrent clients or external KV connectors:

1. Confirm a prefix-cache reset; measure a cold target request.
2. Confirm another reset; send an excluded priming request with a **different question**
   and the identical shared schema prefix.
3. Measure the warm target using the **exact same prompt as the cold target**.

This produces 20 cold and 20 warm requests per cache-on variant, each scored for
SQL execution and result equivalence. The warm target cannot reuse its own full
question from the paired cold trial. Matching blocks may include short common question
fragments; only matching prefix blocks can be reused. Cache-off has 20 target requests
without resets or priming. A reset/priming failure prevents a completed comparison.

**Cold means empty prefix cache on an already loaded, kernel-warmed engine.** It does
not mean cold model startup. Model loading/startup, health checks, warmup, resets,
priming and SQL execution are outside measured request latency. Each request records
first-content TTFT and transport E2E, sampled peak device memory, token counts, raw SQL,
SQL execution, result equivalence and exact-SQL semantic review coverage.

### Reset API compatibility

The protocol requires `POST /reset_prefix_cache` to return `{"success": true}`.
An empty HTTP 200 response from older vLLM releases does not confirm success. The
notebook saves a failure and does not claim a cold measurement on such a server.
Check the installed engine's API before the full cache study. Do not upgrade only
one experimental variant; any engine change needs a fresh smoke test and full set.
The [confirmed-reset API](https://docs.vllm.ai/en/v0.27.1/api/vllm/entrypoints/serve/dev/cache/api_router/)
returns a success boolean; [older APIs](https://docs.vllm.ai/en/v0.20.0/api/vllm/entrypoints/serve/cache/api_router/)
explicitly do not check reset success. Cold-cache results remain unavailable if the
working GPU stack cannot provide that confirmation. No GPU results are fabricated.

## Metrics and quality interpretation

The primary table separates `cache_off`, `cold`, and `warm`. It reports SQL match,
semantic review, TTFT p50/p95, request E2E p50/p95, approximate TPOT p50, aggregate
output tok/s, and sampled peak GPU memory. TPOT is estimated from request-level
E2E/TTFT and token counts; it is not actual inter-token timestamp telemetry. Tokens/s
uses summed target-request durations and excludes reset/prime gaps. It does not
represent total wall-clock throughput or semantically correct tasks per minute.

The inherited application harness runs **after** these measurements, evaluating
clinical plan → SQL → validation/execution → repair. Application quality latency is
shown separately, with SQL match and human semantic review coverage. Its supplemental
serving table is a warmed repeated workload, not the controlled cold/warm study.
Four quality cases are a functional smoke test. The current dataset has **23 reference
SQL cases** out of 30 total; use that full evaluation plus semantic review before final
quality claims. Twenty requests per phase give illustrative p95 values only.

AWQ W4A16 compresses weights, not activations or KV cache. Both modes use FP16
activations; the existing server leaves KV dtype at auto (FP16 here). Total GPU
memory includes resident weights, reserved KV/allocator memory and other GPU processes.
Sampling can miss short peaks. Missing memory is unknown, never zero. Retain startup
allocation logs to interpret weight memory separately; do not infer compression ratio
from total `nvidia-smi` memory alone.

## Save, resume and validate

- Smoke attempts: `experiment-tracker/mvp4/smoke/*.json`.
- Variant records: `experiment-tracker/mvp4/runs/*.json`, including phase summaries,
  raw target/prime requests, reset confirmations, SQL results, memory and lineage.
- Launch provenance: `experiment-tracker/mvp4/provenance/*.json`.

Copy these folders back with the notebook for Mac review. Set `LOAD_SAVED=True` in
section 9 to reload this tag's latest successful records. Phase comparisons refuse
legacy results without cold/warm evidence, mismatched prompts/date/database/environment,
or failed requests. Successful full clinical harness runs also retain their inherited
tutorial run records.

CPU validation from the repo root:

```bash
python -m pytest tests/tutorial/test_optimization.py tests/tutorial/test_cache_study.py -q
```

Tests use fake inference and real read-only DuckDB scoring. They verify reset/prime
ordering, different-question priming, identical shared prefixes, SQL mismatch handling,
AWQ success/failure gates, checkpoint controls, phase summaries and every default
notebook cell without importing GPU libraries.

`python scripts/tutorial/build_optimization_notebook.py` rebuilds the clean unexecuted
notebook. Use it only when intentionally replacing saved outputs.

Mechanism references: [vLLM prefix caching](https://docs.vllm.ai/en/v0.13.0/features/automatic_prefix_caching/)
and [AWQ serving](https://docs.vllm.ai/en/v0.13.0/features/quantization/auto_awq/).
