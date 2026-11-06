# Setup and GPU execution

For the two additional NIM notebooks, follow [the NIM-specific deployment guide](tutorial/NIM_MVP4.md).
The vLLM setup and optimization workflows below remain unchanged.

Start from this repository root. Python 3.11 is the tested CPU interpreter. Install
`requirements-notebook.txt` into `.venv` and run `python -m pytest -q` there before moving
to GPU execution. The included synthetic database is ready; no database rebuild is needed.
`.env.example` is optional. Notebooks set database/schema settings explicitly and use HTTP
serving adapters; they do not load GGUF or start the inherited llama.cpp application.

## GPU engine environment

Use a separate Linux `.venv-vllm` environment with a compatible NVIDIA driver, CUDA/PyTorch
stack and pinned vLLM build. Nemotron requires **vLLM >= 0.15.1** according to its model card;
the exact build must also support the actual GPU, model and hybrid state kernels. Do not
upgrade the engine halfway through a comparison. Record `pip freeze`, startup logs and
`nvidia-smi`; real launch provenance records engine/Torch/CUDA versions and GPU identity.
No GPU package installation or model download is performed by the notebook.

Prepare local snapshots before launching:

| Model | Unquantized checkpoint | AWQ |
|---|---|---|
| Qwen | `Qwen/Qwen3-4B` → `models/qwen3-4b-frozen` | `Qwen/Qwen3-4B-AWQ` → `models/qwen3-4b-awq-4bit-frozen`; pin matching revisions and validate |
| Nemotron | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` → `models/nemotron-3-nano-4b-frozen`; loaded as FP16 | No validated compatible artifact supplied; optional stretch only |

Set exact base revisions in the notebook. Local folder names are conventions, not evidence
of model identity. Validation fingerprints all relevant model/tokenizer artifacts. Nemotron
snapshots must include the reviewed custom model/tokenizer code referenced by their config.
Its launcher enables trusted local code with HF downloads disabled. Keep local paths in
notebook variables and retain license/model notices with separately downloaded artifacts.

## Qwen workflow

Open `notebooks/local AI optimization - Qwen.ipynb`. Validate the exact base/AWQ pair;
launch and complete the separate AWQ smoke test first. Stop it, then execute FP16 off,
FP16 on, AWQ off and AWQ on one server at a time using the printed commands. Set `ACTIVE`
to the server actually running and enable `RUN_LIVE`. The existing Qwen protocol is detailed
in [the inherited optimization guide](tutorial/MVP4_OPTIMIZATION.md); this guide governs the
new split repository. Qwen's own full/retrieved/context/concurrency follow-up work is not
part of the carried-over four-variant experiment.

## Nemotron workflow

Open `notebooks/local AI optimization - Nemotron.ipynb`. Validate **only the unquantized
baseline** first. Keep `RUN_AWQ=False`; its absent checkpoint does not block core work.

The notebook prints commands using `scripts/serving/launch_nemotron.py`. Its default is a
preview; `--execute` launches the engine explicitly and creates a unique provenance JSON.
It requests FP16 weights/activations, FP32 Mamba SSM state and explicit scheduler capacity.
The notebook controls request reasoning with `enable_thinking` and removes `/no_think`
when reasoning is enabled. Raw thought content is retained in serving records; only the
final JSON after a complete closing think marker is scored. Incomplete reasoning is a
failure. First-content TTFT includes thought tokens; final-answer TTFT is not measured.

Run the FP16 baseline first. Then run prefix caching, schema reduction, concurrency 2/4,
capacity 1 at concurrency 4, reasoning on, and output caps 256/2048. All compare with the
baseline except the capacity experiment, which compares with `concurrency_4`. Use one
printed command/metadata record per fresh launch, wait for readiness, set `ACTIVE` to that
experiment, and execute its cell. The namespace setup cell clears results; change variables
in a separate cell while retaining the setup state. Avoid Run All against one resident server.

Prefix caching is a core investigation with a support gate: inspect the installed engine's
Nemotron/Mamba backend support and startup logs before setting `PREFIX_SUPPORTED=True`.
If unsupported, retain the skip and continue other experiments. The mechanism protocol
also requires a reset response containing `success=true`. An empty legacy HTTP 200 does
not confirm a cold cache; preserve the failure rather than mislabeling warm measurements.
No concurrent clients or external KV connectors may interfere with the reset/prime sequence.

If a Nemotron AWQ derivative is obtained or produced, set its source/revision/path, validate
its lineage and config, enable `RUN_AWQ`, launch its printed command, then run the smoke
cell with `ACTIVE='awq_smoke'`. Only after success run `ACTIVE='awq_stretch'`. Keep the
same engine, GPU and state configuration; unsupported stretch work does not invalidate core.

## Save and resume

Qwen results use `experiment-tracker/mvp4/{smoke,runs,provenance}/`; Nemotron results use
`experiment-tracker/mvp4/nemotron/` and the same provenance/smoke directories. Model-specific
run tags prevent collisions. Keep all completed, failed and skipped records, and copy them
back with the notebooks after GPU execution. `LOAD_SAVED=True` reloads matching configuration
records. Rebuild scripts intentionally replace notebook cell outputs, so do not use them to
save GPU results. Keep model and database fingerprints, date and unchanged settings matched
when interpreting a comparison.
