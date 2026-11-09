# Shared NIM technical guide

Used by [03 — Qwen NIM](../notebooks/03_qwen_nim.ipynb) and
[04 — Nemotron NIM](../notebooks/04_nemotron_nim.ipynb). Install prerequisites in
[SETUP.md](../SETUP.md); choose modes/run/cleanup in [RUN_ALL.md](RUN_ALL.md).
This guide covers the NIM-specific deployment contract rather than repeating installation.

## What is being deployed?

The tested `vllm-model-free-nim:2.1.5` image packages **vLLM**. It is not necessarily a different
inference engine from native vLLM and does not imply TensorRT conversion. One downloaded image
can serve either exact base model through separate container instances and local read-only
checkpoint mounts. On the 16 GiB Dell, run one GPU model server at a time.

Targets are exactly `Qwen/Qwen3-4B` and `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`. The initial
recipe loads unquantized weights as FP16, context 8192, prefix cache off, scheduler capacity 4.
Nemotron additionally requests FP32 Mamba SSM state and reviewed trusted local code. A similar
model name, BF16 source label or successful native launch does not prove these settings work
inside a particular container. Retain and inspect resolved engine settings in startup logs.

Both targets ran manually on the Dell in the archived study. Fresh validation of the later
Run All lifecycle remains pending. NIM/native gains are deployment observations until engine
versions, weights/tokenizer, prompts, date/database, precision/state, output settings and
thinking behavior are matched. The image digest identifies packaging, not the backend version
or proof of effective settings. No controlled runtime-speed winner is established.

## Discover on your actual model/image/GPU

After creating the appropriate config and supplying exact model revision/download settings:

```bash
.venv/bin/python scripts/tutorial/configure_mvp4.py --runtime nim --model qwen --discover
```

Use `--model nemotron` for chapter 04. Discovery starts a temporary inspection container,
not a benchmark server. It pins the locally downloaded image digest and saves the profile's
`discovery_output`, `runnable_profile_ids`, GPU UUID and model/config fingerprint contract.
Inspect the complete output; select only a runnable profile, not low-memory/incompatible rows.
Set your chosen full ID as `nim_profile_id` in `tutorial/config/nim-<model>.local.json`.
The helper never silently chooses another precision/backend or an arbitrary first profile.

| Profile component | Meaning |
|---|---|
| `vllm` | Serving backend supplied by this model-free image |
| `tp1` | No tensor splitting across multiple GPUs |
| `pp1` | One pipeline stage containing all layers |
| `feat_lora` | Adapter-loading capability; not needed by this base-model experiment |

The recorded single-GPU listing offered standard and LoRA variants. Your own discovery may
change with image/model/GPU. Generic profile names do not necessarily encode dtype; FP16 is
requested in launch settings and still needs actual startup verification. Rediscover after
changing the model, image, GPU or launch settings. A runnable listing alone is not a successful
model startup or executable SQL smoke.

## Binding, readiness and ownership

The controller stages the local snapshot read-only at `/model`, requests offline HF loading,
sets `NIM_MODEL_PATH`, and binds the API to loopback. Credentials stay in environment variables;
profiles and commands carry variable names, not secret values.

Discovery is bound to model revision, fingerprint, image, GPU and requested settings. Deployment
reserves fresh metadata, starts an owned container, waits for `/v1/health/ready` and verifies
`/v1/models` serves the intended alias. A changed contract fails before benchmarking.
The unique container name/ID and startup logs provide cleanup evidence. The fixed archived
manual container names should not be copied into new owned-session cleanup commands.

Core experiments reuse this deployment: smoke → full-schema baseline → retrieved schema →
concurrency 2 → concurrency 4. Request concurrency changes; scheduler capacity stays 4.
Startup/readiness is recorded separately. Shared cleanup behavior is documented in RUN_ALL.
[Implementation](../src/clinical_trial_mvp/tutorial/nim_study.py)
· [owned-session controller](../src/clinical_trial_mvp/tutorial/workflow.py).

## Preserve the contract failure and correction

The [original Qwen smoke](../experiment-tracker/mvp4/nim/runs/20261107T213636_946106Z.json)
had 4/4 successful transport requests but 0/4 application SQL executions. All four plans used
an object for `filters`; the strict plan contract requires a string. The initial recipe disabled
JSON-schema enforcement. Valid JSON was not enough.

The [schema-constrained rerun](../experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json)
executed 3/4 application queries and matched 2/4 references. Both attempts remain checksum-
protected. Qwen's reusable config defaults `structured_outputs=true`; Nemotron NIM retains
its recorded false setting. The flag is serialized per result and must be accounted for in
comparisons. Never coerce a malformed plan merely to pass smoke.

The adapter sends standard OpenAI fields and the common `/no_think` prompt. That prompt alone
does not verify backend thinking parity with native template controls. SQL smoke requires some
executable application SQL plus no serving failures; it is not an all-cases quality pass.

## Evidence boundaries

Transport TTFT/E2E, application E2E, reference-result match and human semantic review are
separate measurements. NIM direct-serving requests were not post-scored as SQL in this initial
recipe; do not call transport success direct SQL correctness. NIM VRAM samples exist, while
historical native memory sampling failed. They cannot fill each other's missing values.

Cold/warm caching, AWQ, reasoning and backend-capacity sweeps are conditional extensions,
not claims that the initial NIM recipe implements them. No more experiments are being added
for this publication pass. See [concise findings](RESULTS.md), [detailed evidence](DELL_RESULTS.md)
and [the contract case study](CASE_STUDIES.md#qwen-nim-contract-failure).

Official references: [profile selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html),
[model-free deployment](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-free-nim.html).
