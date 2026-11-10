# Run the notebooks: configure, check, measure and clean up

Notebooks 00–06 default to **Live**. Read the notebook without executing to inspect
saved outputs where available. To collect new measurements, configure the relevant
runtime, then execute cells in order. Installation instructions live in [SETUP.md](../SETUP.md).

For Dell launch commands and standalone vLLM/SGLang checks, use [DELL_START.md](DELL_START.md).

## One visible execution control

Each notebook 01–06 starts with:

```python
MODE = "live"  # "plan" previews commands; "replay" displays archived results
```

The notebook passes this value to `Study(..., mode=MODE)` or `EdgeStudy(..., mode=MODE)`.
It overrides a stale `mode: "replay"` in a local JSON file. Remove `execute` from
older configurations: mode controls execution; contradictory legacy flags are rejected
when no explicit notebook mode is provided. The explicit notebook mode supersedes that
legacy flag. The first cell prints effective mode, model, hardware target and session ID.
Live begins with an empty results collection and never imports old measurements.

For CPU CI and deliberate offline execution, `LOCAL_AI_LAB_MODE=plan` or `replay`
overrides the notebook's default before construction. These modes make no inference,
Docker, download or GPU-probe calls. Select them explicitly before **Run All**.

## Configure notebooks 01–04

Create one ignored per-machine configuration for each notebook:

```bash
.venv/bin/python scripts/tutorial/configure_mvp4.py --runtime vllm --model qwen
```

Change `--model` to `nemotron` for 02; use `--runtime nim` with each model for 03/04.
Edit the printed `tutorial/config/<runtime>-<model>.local.json`. Example files are
committed; local paths and settings stay ignored. Reuse your existing Dell environments,
model snapshots and model revision JSON files. No project environment symlink is needed.

| Setting | Meaning |
|---|---|
| `vllm_python` | Actual CUDA Python, e.g. `/home/setup/.venv-vllm/bin/python` |
| `base_snapshot`, `awq_snapshot` | Existing frozen model directories |
| `base_revision`, `awq_revision` | Exact commits, or restore the existing model revision JSON |
| `download_models` | False by default; true explicitly permits checkpoint downloads |
| `awq_source`, `awq_derivation` | Exact compatible source and provenance linking AWQ to the base revision |
| `run_awq` | Required for Qwen's four variants; optional for Nemotron |
| `prefix_supported` | Nemotron: enable only after verifying installed hybrid-backend support |
| `structured_outputs` | Request schema enforcement; on by default for Qwen NIM |
| `nim_image`, `nim_profile_id` | Locally available digest-pinned image and explicitly selected runnable profile |
| `nim_profile_path` | Your per-model profile/discovery JSON |
| `repeats` | Repeats per four-question serving workload; default five |

The bases are `Qwen/Qwen3-4B` and `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`,
loaded as FP16. Nemotron retains FP32 SSM state. Qwen AWQ uses four-bit weights
and FP16 activations. No validated Nemotron AWQ checkpoint is supplied.

## Cell order on the Dell

1. **Imports/configuration:** choose mode and review paths/revisions.
2. **Preflight:** validate GPU environment, runtime, checkpoint paths and free port 8000.
   NIM also checks Docker, its local image, Docker GPU access and selected profile.
3. **Preparation/smoke:** validate artifacts and test required SQL compatibility.
4. **Experiments:** run each experiment cell separately. Live starts serving and sends
   new inference requests; it saves fresh request timestamps and a unique evidence file.
5. **Comparison:** inspect current-session tables, cold/warm phases, output tokens,
   request reconstruction, generated SQL and figures.
6. **Cleanup:** stop only servers or containers created by this study; save the notebook.

NIM needs the image already available locally; no notebook pulls an image or installs
packages. Follow [the NIM guide](NIM_MVP4.md) for image access and initial discovery.
If the profile ID is not yet known, configure the image/checkpoint, then use:

```bash
.venv/bin/python scripts/tutorial/configure_mvp4.py --runtime nim --model qwen --discover
```

Discovery prints runnable IDs; select yours in local configuration before preflight.
Native variants use separate owned servers. NIM reuses one owned deployment across its
request experiments. An occupied endpoint fails; the notebook does not stop an external server.

Cache-on uses separate cold/warm target requests with the identical shared schema prefix.
The controller sets `VLLM_SERVER_DEV_MODE=1`; a reset must acknowledge success. Model
startup/readiness, warmup, reset, prime and SQL scoring are outside request timing.
See [the measurement methodology](MVP4_OPTIMIZATION.md).

## Session evidence and failure handling

Every Live session has a fresh timestamp/UUID. Summaries show its ID and evidence paths
under `experiment-tracker/mvp4/workflow/`. Original runtime records remain in the existing
model-specific folders. Rerunning a cell writes another unique record; it does not overwrite
prior evidence. The summary selects the latest cell result in this session.

A missing result stays missing. A failed result remains failed, is saved and stops the cell;
no older success replaces it. Tables, figures, cache phases and token analysis use this
session only. Request inspection uses the current request builder and profile; reconstructed
payloads are explicitly identified as reconstructions, not packet captures.

Native servers stop after each row. Run the cleanup cell before leaving a live NIM session.
After a kernel or OS crash, use saved process/container identity for explicit cleanup.

## Notebooks 05–06: local ExecuTorch stages

Follow [the edge guide](EXECUTORCH_MAC.md), then copy the relevant example:

```bash
cp tutorial/config/edge-qwen.example.json tutorial/config/edge-qwen.local.json
cp tutorial/config/edge-nemotron.example.json tutorial/config/edge-nemotron.local.json
```

Set `edge_python` to your separate installed edge environment. Qwen's `artifact_dir`
contains source weights, conversion validation and the exported PTE. `reuse_export=true`
verifies existing export hashes and explicitly labels reused stages; **native generation
and SQL checking run afresh**. For a new conversion/export, set false and choose a new
artifact directory. Downloads remain opt-in. No HTTP serving benchmark is added.

Nemotron's `nemotron_config` points to the pinned local config. Notebook 06 executes a
fresh recipe check, without weights or native inference. Unsupported configuration is
reported honestly, with no SQL or latency measurements. Sessions save new evidence under
`experiment-tracker/tutorial/edge/sessions/<session-id>/`.

## Reference: previously measured results

Your existing Dell GPU results remain unchanged. See [RESULTS.md](RESULTS.md),
[the original executed Dell notebooks](../notebooks/history/dell-2026-11-07/),
and the preserved [notebooks before these changes](../notebooks/history/before-live-first-2026-11-10/).
Explicit `MODE="replay"` rebuilds the selected historical tables without GPU work.
No GPU rerun is needed to read or analyze that existing evidence. A new Dell run is needed
only to populate genuine Live outputs for the revised execution path.
