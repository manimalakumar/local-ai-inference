# Configure, replay, plan, run live and clean up

Install prerequisites once using [SETUP.md](../SETUP.md). This guide contains execution
instructions, not a second installation transcript. Chapters remain separate:
00 is a read-only serving foundation; 01–04 are the optimization/deployment entry points.

## Replay — default

```bash
.venv/bin/python -m jupyter lab notebooks
```

Follow 00→04, or jump to `01_qwen_cache_quantization.ipynb`, then choose **Run All**.
No local config, GPU, model download, Docker or credentials are needed for replay. Each
measurement is explicitly labeled. Chapter 00 distinguishes historical Mac report values
and rounded engine summaries from complete recorded Dell measurements in chapters 01–04.

Replay selects specific records and validates checksums; it never silently fills missing data.
A failed live experiment never falls back to archived values. Saved outputs can be read on
GitHub without running anything. Output-token analysis and reference SQL use the existing data.

## Configure chapters 01–04

Create one ignored per-machine config for each chapter you will run:

```bash
.venv/bin/python scripts/tutorial/configure_mvp4.py --runtime vllm --model qwen
```

Use `--model nemotron` for chapter 02. Use `--runtime nim` with the appropriate model for
chapters 03/04. Edit the printed `tutorial/config/<runtime>-<model>.local.json` with an editor.
Example files are committed; machine-local files are not. Chapter 00 needs no live config.

| Key | Meaning |
|---|---|
| `mode` | `replay` loads recorded results, `plan` prints commands, `live` benchmarks on Linux GPU |
| `vllm_python` | Existing GPU Python, e.g. `/home/setup/.venv-vllm/bin/python`; no symlink needed |
| `base_snapshot`, `awq_snapshot` | Local checkpoint paths, defaulting to the existing `models/*-frozen` folders |
| `download_models` | Explicit opt-in to download/reuse snapshots before measurement |
| `base_revision`, `awq_revision` | Exact revisions; otherwise use existing model revision JSON files |
| `awq_source`, `awq_derivation` | Compatible repository and explicit attestation linking the derivative to the exact base revision |
| `run_awq` | Qwen's four-variant strategy requires AWQ smoke; Nemotron core defaults false |
| `prefix_supported` | Nemotron: enable only after actual hybrid-backend verification; otherwise the row skips |
| `structured_outputs` | Default on for Qwen NIM, off for Nemotron NIM, matching the recorded correction/settings |
| `nim_image`, `nim_profile_id` | Digest-pinned image and your explicitly selected runnable profile; see the NIM guide |
| `run_tag` | Optional label; a timestamp/UUID is always added to avoid collisions |
| `repeats` | Five by default: four distinct questions, not 20 independent clinical tasks |

The older `execute` boolean is accepted for existing configs; new configs use `mode`.

## Plan without executing

Set `mode` to `plan` and Run All. The notebook displays configuration and resolved/planned
server arguments. Unknown revisions/runtime defaults are labeled, not fabricated. No server,
model download or Docker process starts. Chapter 00 remains read-only in Plan mode.

## Prepare exact models for Live

The bases are `Qwen/Qwen3-4B` and `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, loaded as FP16.
Nemotron's SSM state remains FP32. The Qwen AWQ candidate is `Qwen/Qwen3-4B-AWQ`; validate
matching architecture/tokenizer and record derivation. No compatible Nemotron AWQ checkpoint
is supplied; it remains optional stretch. FP8, NVFP4 and GGUF are different experiments.

Existing Dell `models/qwen-revisions.json` and `models/nemotron-revision.json` are reused.
On a new host, provide pinned revisions or explicitly enable downloads. A missing revision
for an enabled download is resolved to a commit SHA and saved; set exact revisions to
reproduce an earlier run. Downloading precedes request clocks. Review custom model code
before enabling Nemotron's trusted local-code path. Models and caches stay out of Git.

NIM also requires per-model discovery and explicit profile selection; follow the
[shared technical guide](NIM_MVP4.md). Neither matching architecture nor a runnable listing
replaces actual startup and SQL smoke.

## Live execution and interpretation

Set `mode` to `live` and Run All on the Linux GPU host. Start with Qwen/vLLM. The controller
validates checkpoints, starts owned servers, waits for readiness, runs smoke, measures and
cleans up. Native variants use fresh servers; NIM request-level experiments share one fixed
deployment. No manual ACTIVE cells or copied launch commands are needed.

Startup/readiness, warmup, reset/prime, telemetry setup and SQL scoring stay outside individual
request timing. The native controller sets `VLLM_SERVER_DEV_MODE=1` consistently to expose
reset controls; confirmed resets still require JSON `success=true`. See the
[methodology](MVP4_OPTIMIZATION.md) for the precise cold/warm protocol and token-work checks.

Inspect observation → explanation → limitation → decision after each experiment. Request
TTFT/E2E is not application E2E; transport success is not SQL correctness. Compare output-token
counts before attributing latency to computation. NIM/native settings and thinking parity
remain unverified for a controlled speed comparison. Keep unknown memory unavailable.

## Cleanup, pause and resume

An occupied port 8000 fails visibly. The workflow never stops external servers. Exceptions
and interrupts stop owned servers and retain failed records. Native servers stop after each
row. If pausing between successful NIM cells, run the final cleanup cell before switching models.
After a kernel/OS crash, inspect saved process-group/container identity and stop that owned
server explicitly; Python cleanup cannot run after a crash.

Save with Ctrl+S; no Jupyter download is required. Replay writes no measurements. Live adds
uniquely named records under the existing model-specific directories. Native lifecycle metadata
uses `experiment-tracker/mvp4/workflow`; NIM deployment metadata uses `experiment-tracker/mvp4/nim/provenance`.
Re-run Replay to inspect the fixed original archive; live failures never trigger it automatically.

## Preserve, review and share

Builder scripts are the source of notebook prose and cells. They preserve outputs for
byte-identical code cells, including repeated cells in order; changed code is left unexecuted.
Historical outputs are not validation of edited prose or a new live run. Archive newly
executed outputs before rebuilding when they are required as complete session evidence. The unchanged original Dell archive and
its checksums remain under `notebooks/history/dell-2026-11-07`. Keep failures and required
provenance. Never regenerate the frozen database just to refresh outputs.

Before staging, review files for credentials, personal paths and sensitive output. Environments,
weights/caches, `*.local.json`, secrets, temporary ZIPs and unreviewed runtime logs are ignored.
Retain sanitized provenance required for reproducibility; do not rewrite checksum-protected
history casually. Raw new logs require explicit review/sanitization before inclusion.

```bash
git status --short
git add notebooks/ src/ scripts/ tests/ .github/ tutorial/ constraints/ evals/ data/
git add README.md SETUP.md VALIDATION.md THIRD_PARTY.md pyproject.toml requirements*.txt CPU_NOTEBOOK_ENVIRONMENT.json .gitignore
git diff --cached --stat
```

Commit/push privately after reviewing the staged content. Confirm hosted CPU CI. Use
`scripts/tutorial/export_tracked.py` only for a reviewed committed revision; it never zips
an environment or `.git` directory. Do not publish a working-directory ZIP. See the
[remaining validation checklist](PUBLICATION.md).

Chapter [05 — ExecuTorch portability](../notebooks/05_edge_portability_executorch.ipynb) is
also read-only recorded replay, with no Live edge run. It uses the same CPU environment
and stays separate from the clinical GPU comparisons.
