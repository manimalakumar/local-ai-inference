# Local AI Inference Lab: Serving, Optimization and Application Quality

How do we make local AI faster and more efficient without losing useful task performance?

A hands-on study of local serving on constrained hardware. Using Qwen and Nemotron, we
explore caching, quantization, context and concurrency through vLLM and NVIDIA NIM.
Clinical-trial analytics is the executable example workload: it checks whether a faster
response still produces a useful answer. We have not evaluated multiple application domains.

## What you will learn

- Distinguish prefill, decode and application work before choosing an optimization.
- Read latency and aggregate throughput beside SQL execution and result-match counts.
- Diagnose serving compatibility, request contracts and missing telemetry.
- Reproduce recorded findings without a GPU, then plan or run your own experiments.

**A faster response is not necessarily a better answer.** Different optimizations change
different parts of inference.

## Three measured findings

These are **recorded Dell measurements** from a Quadro RTX 5000 with 16 GiB VRAM.
They describe four distinct questions, not a general benchmark or clinical validation.

| Observation | What to investigate |
|---|---|
| Qwen FP16 paired cold/warm TTFT: **2.019 → 0.083 s**; request E2E: **6.038 → 4.039 s**. Direct result matches stayed **10/20** in both phases. | Prefix reuse avoids repeated prefill, while generation time remains. |
| Nemotron retrieved-schema application result match: **1/4**, versus **2/4** with full schema. | Fewer tokens can remove information needed by the application. |
| Qwen NIM transport succeeded **4/4**, but the original smoke executed **0/4** application SQL queries; schema enforcement rerun executed **3/4**. | Endpoint success is not application success; request contracts matter. |

[Sources, both main figures and boundaries](tutorial/RESULTS.md) · [full evidence tables](tutorial/DELL_RESULTS.md)

## Start here

**New to local serving? Start at [00 — Serving foundations and recorded comparisons](notebooks/00_local_serving.ipynb).**
**Already familiar? Start at [01 — Qwen: cache and quantization](notebooks/01_qwen_cache_quantization.ipynb).**

Install once, then select your chapter in Jupyter's file browser. Chapter 00 is historical
and read-only; it does not reproduce fresh llama.cpp/Ollama/vLLM/SGLang runs. Chapters 01–04
support Replay, Plan and Live. Saved outputs are clearly labeled recorded evidence:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt -c constraints/cpu-notebook-py311.txt
.venv/bin/python -m jupyter lab notebooks/
```

Choose **Run All**. Default replay needs no GPU, model download, Docker or credentials.
The notebook exposes the server command, request construction and an evaluated SQL response.

| Mode | What happens |
|---|---|
| **Replay — default (01–04)** | Load specific checksum-verified archived runs; every result is labeled **recorded Dell measurement**. |
| Plan (01–04) | Show configuration and launch commands without contacting a server. |
| Live (01–04) | Launch, benchmark, preserve new evidence and clean up owned processes. A failure never falls back to replay. |

Mode and machine settings live in a local configuration file. Follow [setup and configuration](tutorial/RUN_ALL.md)
for Plan/Live, optional pinned model downloads and profile discovery.

## Hardware and software

Replay/Plan require Python 3.11 and the pinned CPU notebook environment. Pins were installed
and tested on macOS arm64 with Python 3.11.15. A fresh remote clone passed Replay/Plan
and 204 tests; [hosted Linux CI passed](https://github.com/manimalakumar/local-ai-inference/actions/runs/42666644695)
for source commit `56feab1`. GPU engines are deliberately separate from those CPU constraints.
See [exact installed package metadata](CPU_NOTEBOOK_ENVIRONMENT.json) and the
[validation record](VALIDATION.md).

The archived live study used a **Quadro RTX 5000, 16 GiB**, driver **595.91.07**, native
**vLLM 0.32.0**, and a digest-pinned **NIM 2.1.5** model-free image. Models were Qwen3-4B and
NVIDIA Nemotron 3 Nano 4B, loaded as FP16. Nemotron retained FP32 SSM state. Compatibility
must be checked on your actual hardware/build; these observations do not establish support
for every 16 GB GPU. NIM requires system Docker and NVIDIA Container Toolkit.

## Experiment map

| Order | Chapter | Role |
|---|---|---|
| Foundation | [00 — Local serving](notebooks/00_local_serving.ipynb) | Workload/evaluation, full/retrieved context, historical llama.cpp/Ollama and vLLM/SGLang serving |
| Optimization | [01 — Qwen/vLLM](notebooks/01_qwen_cache_quantization.ipynb) | FP16/AWQ W4A16 × prefix cache off/on, with paired cold/warm interpretation |
| Next | [02 — Nemotron/vLLM](notebooks/02_nemotron_optimization.ipynb) | Hybrid caching, context, concurrency/capacity, reasoning and output budgets |
| Optional extension | [03 — Qwen/NIM](notebooks/03_qwen_nim.ipynb) | Packaging, profiles and structured-output compatibility |
| Optional extension | [04 — Nemotron/NIM](notebooks/04_nemotron_nim.ipynb) | Deploying the hybrid model through NIM |
| Optional CPU portability | [05 — ExecuTorch](notebooks/05_edge_portability_executorch.ipynb) | Recorded SmolLM2 export/run; separate from clinical-model rankings |

## Results and limitations

Read [the concise results page](tutorial/RESULTS.md). Paired cache phases are interpreted
separately from supplemental warmed summaries. Twenty serving requests repeat four clinical
questions. Reference-result match is distinct from human-reviewed semantic correctness.

**Deferred follow-up validation:** human semantic review and fresh checks of the updated
GPU automation/native VRAM sampling are parked for this iteration. Your completed Dell GPU
experiments remain available as recorded evidence. These follow-ups are not prerequisites
for reading or replaying the tutorial. Native memory-reduction conclusions remain unsupported;
wider quality claims require broader evaluation and review. NIM/native comparisons still
need matched settings, and no compatible Nemotron AWQ checkpoint was obtained.

[Current rubric and publication checklist](tutorial/PUBLICATION.md) · [GPU rerun checklist](tutorial/GPU_VALIDATION.md)
· [SQL review packet](tutorial/SQL_REVIEW.md) · [validation history](VALIDATION.md)

## Troubleshooting as case studies

- [Cache-reset 404](tutorial/CASE_STUDIES.md#cache-reset-404): a running server did not expose the endpoint needed by the experiment.
- [Qwen NIM contract failure](tutorial/CASE_STUDIES.md#qwen-nim-contract-failure): valid transport and plausible JSON violated the application schema.
- [Missing VRAM telemetry](tutorial/CASE_STUDIES.md#missing-vram-telemetry): the experiment ran but its measurement path was wrong.

The [detailed journal](tutorial/DELL_RESULTS.md#troubleshooting-journal) retains environment,
symlink, Docker, profile-selection and Git lessons without putting them in the quick start.

## Licensing and contributions

License work is deferred for this iteration. Existing [third-party notes](THIRD_PARTY.md)
remain intact; no license terms have been changed.

For private review, run the CPU suite and notebook/replay checks:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/tutorial/validate_notebooks.py --mode replay
.venv/bin/python scripts/tutorial/validate_notebooks.py --mode plan
```

A [CPU CI workflow](.github/workflows/cpu.yml) performs these checks. It makes no GPU claims.
Create release ZIPs only from a reviewed committed revision with
`python scripts/tutorial/export_tracked.py --output /tmp/local-ai-inference.zip`.
Uncommitted changes are not included; `.git`, caches, bytecode and model weights are excluded.

## Evidence, lineage and history

The project continues the cumulative work tagged `mvp4`; historical iteration labels remain
in records and source paths. [Baseline history](BASELINE.md), [source hashes](SOURCE_LINEAGE.json)
and the [unchanged Dell notebook archive](notebooks/history/dell-2026-11-07) provide audit context.
The archive manifest checksums notebooks, successes, failures and provenance. New runs never
overwrite them. The old `clinical-trial-mvp3` served alias remains for harness compatibility;
model revision/fingerprints establish identity. [ExecuTorch](tutorial/EXECUTORCH_MAC.md) is a
separate SmolLM2 FP32 CPU portability case, not a comparable clinical GPU experiment.
