# Published execution evidence

The table describes what the committed artifacts establish. The GPU records do
**not** contain the tested repository SHA: an upload commit identifies where the
records were published, not necessarily the code executed. CPU CI validates the
current source in Plan and Replay; it does not run GPU inference or certify cleanup.
No new GPU run is required for the presentation and offline-portability fixes.

| Notebook | Tested source commit | Mode | Session / run ID | Completed, failed or skipped experiments | Cleanup outcome | Evidence |
|---|---|---|---|---|---|---|
| 00 Local serving | Not recorded | Live, Mac CPU | `live-nb00-20261110T051644_860809Z` | Full/retrieved context, local serving and a checked answer completed | Server externally managed; independent cleanup check not recorded | [Executed checkpoint](../experiment-tracker/tutorial/foundation/live-nb00-20261110T051644_860809Z/execution-checkpoint.ipynb), [server provenance](../experiment-tracker/tutorial/foundation/live-nb00-20261110T051644_860809Z/server.json) |
| 01 Qwen optimization | Not recorded; uploaded in `99e1a1e` | Live, Dell GPU | `20261110T233252-f6e2bc4728` | AWQ smoke and 12 benchmark variants completed; completion is separate from individual request failures and reference-result matches | Cleanup cell completed; no independent post-run process/VRAM snapshot | [Baseline](../experiment-tracker/mvp4/workflow/vllm-qwen-20261110T233252-f6e2bc4728-fp16_cache_off-5d535f84.json), [session records](../experiment-tracker/mvp4/workflow/), [executed notebook](../notebooks/01_qwen_optimization.ipynb) |
| 02 Nemotron optimization | Not recorded; uploaded in `99e1a1e` | Live, Dell GPU | `20261111T005817-d22e14d969` | 9 variants completed; prefix cache and AWQ skipped. Reasoning/output limits can still produce failed or truncated individual requests | Cleanup cell completed; no independent post-run process/VRAM snapshot | [Baseline](../experiment-tracker/mvp4/workflow/vllm-nemotron-20261111T005817-d22e14d969-fp16_baseline-9006770a.json), [session records](../experiment-tracker/mvp4/workflow/), [executed notebook](../notebooks/02_nemotron_optimization.ipynb) |
| 03 Qwen NIM | Not recorded; uploaded in `99e1a1e` | Live, Dell GPU | `20261111T032350-ebdb978c0d` | Smoke and 4 benchmark variants completed; individual request quality remains in each result | Owned-container cleanup cell returned successfully; independent GPU-release check not recorded | [Baseline](../experiment-tracker/mvp4/workflow/nim-qwen-20261111T032350-ebdb978c0d-baseline-4ed699e7.json), [profile provenance](../experiment-tracker/mvp4/nim/provenance/qwen-nim-qwen-20261111T032350-ebdb978c0d.json), [executed notebook](../notebooks/03_qwen_nim.ipynb) |
| 04 Nemotron NIM | Original source SHA not recorded | Replay of manual Live GPU measurements | Baseline `20261107T204614_919475Z`; no original automated session ID | Archived smoke and 4 benchmark variants completed; no current automated NIM deployment inferred from Replay | Replay starts no container; original automated cleanup not established | [Original baseline](../experiment-tracker/mvp4/nim/runs/20261107T204614_919475Z.json), [original executed notebook](../notebooks/history/dell-2026-11-07/local%20AI%20optimization%20-%20NIM%20Nemotron.ipynb) |
| 05 Qwen ExecuTorch | Not recorded | Live, Mac CPU; verified export reused | `20261110T121532-73d94917db` | Fresh native generation and SQL execution/reference check completed; conversion/export artifacts reused, not newly measured | Foreground child-process execution completed; no persistent server | [Native run](../experiment-tracker/tutorial/edge/sessions/20261110T121532-73d94917db/run-c0fa578c.json), [SQL check](../experiment-tracker/tutorial/edge/sessions/20261110T121532-73d94917db/sql_check-da715168.json) |
| 06 Nemotron ExecuTorch | Not recorded | Live compatibility probe, Mac CPU | `20261110T122034-8766f89a12` | Probe completed with `blocked_compatibility`; no native inference | Foreground probe completed; no persistent server | [Probe](../experiment-tracker/tutorial/edge/sessions/20261110T122034-8766f89a12/probe-e8d2934e.json) |

The final repair commit's exact CPU status is available in
[GitHub Actions](https://github.com/manimalakumar/local-ai-inference/actions).
Notebook schema/link/builder validation remains the first gate, followed by tests,
all-chapter Replay and Plan execution, and deterministic evidence-report checks.

The distinction between an experiment that completed and a correct answer
matters: the metrics here use **reference-result match**, not proven
application correctness. Human semantic review remains deferred. Native VRAM
records from 01–02 now have readings and sample counts, but they include the
server's memory reservation; no isolated weight-memory saving is inferred.

Remaining GPU validation work is limited to facts absent from these records:
exact source-revision attestation, independent cleanup/GPU-release verification,
interruption recovery on hardware, and a fresh automated Nemotron NIM run if that
path is to be claimed as validated. See [the checklist](GPU_VALIDATION.md).
