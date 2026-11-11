# Local AI Inference Lab

## Serving, optimization, and application quality

Getting a model to run locally is a useful first step. The next questions are harder:
what makes it faster, what changes when several users arrive at once, and does the
application still get the right answer?

This lab follows those questions from local serving to inference optimization and
packaged deployment. We use Qwen and Nemotron with vLLM, SGLang and NVIDIA NIM, building on
baseline llama.cpp and Ollama experiments. A small clinical-trial analytics workflow
gives us something executable to evaluate: the model writes SQL, the application
checks it, and DuckDB runs it against synthetic data.

The point isn't to crown a runtime or model. It's to learn how to explain a measured
trade-off—and decide whether it helps the application.

## What you'll learn

- Separate input processing, token generation, and application work before tuning them.
- Test prefix caching, weight quantization, shorter context, and concurrent requests.
- Read first-token latency and throughput alongside executable SQL and result matches.
- Diagnose failures involving profiles, request schemas, cache resets, and telemetry.
- Inspect recorded results without a GPU, then plan or run your own experiments.

The workload and schema are introduced in chapter 00.

## Start here

New to local serving? Start at **00**. 
Already familiar? Start at **01** for optimization.
Each notebook begins with this workflow and highlights **You are here**.

| **Step** | **Notebook** | **Purpose** |
|---|---|---|
| 0. Understand the request | [00 — How a local AI request works](notebooks/00_local_serving.ipynb) | Follow a question through schema context, model serving and generated SQL, then evaluate the results to establish a quality baseline. |
| 1. Qwen optimization | [01 — Qwen: caching and quantization](notebooks/01_qwen_optimization.ipynb) | Compare FP16/AWQ W4A16 and prefix caching, with cold and warm requests reported separately. |
| 2. Nemotron optimization | [02 — Nemotron optimization](notebooks/02_nemotron_optimization.ipynb) | Test schema context, supported prefix caching, concurrent requests, scheduler capacity and output budgets. |
| 3. Qwen NIM (optional) | [03 — Qwen through NIM](notebooks/03_qwen_nim.ipynb) | Deploy Qwen in a container, check the plan format, and compare schema context and concurrent requests. |
| 4. Nemotron NIM (optional) | [04 — Nemotron through NIM](notebooks/04_nemotron_nim.ipynb) | Deploy the hybrid model through NIM and compare schema context and request concurrency. |
| 5. CPU portability (optional) | [05 — Qwen with ExecuTorch](notebooks/05_edge_portability_executorch.ipynb) | Follow Qwen3-4B conversion and CPU export, then inspect a generated SQL answer and its timing. |
| 6. Compatibility study (optional) | [06 — Nemotron with ExecuTorch](notebooks/06_nemotron_executorch.ipynb) | Check whether the installed exporter accepts Nemotron and identify what a working port still needs. |

Follow **00 → 01 → 02** to understand the request and then optimize the two models. Chapter 00 defaults to **local GGUF baselines from scratch**: prepare llama.cpp,
start a server and Run All to collect new SQL-quality and serving measurements. Chapters **03/04**
are optional NIM extensions for the two models. Chapters **05/06** form a separate CPU portability branch: Qwen has a native CPU
smoke result; Nemotron records the unresolved export-recipe compatibility check.

## Try it without a GPU

From a checkout of this repository, create a Python 3.11 environment and install
the notebook dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt -c constraints/cpu-notebook-py311.txt
.venv/bin/python -m jupyter lab notebooks/
```

| Mode | What happens |
|---|---|
| **Live – default** | Configure the target environment and run cells in order to collect new measurements. |
| **Plan** | Inspect the proposed configuration and launch commands without contacting an inference server. |
| **Replay – explicit** | Read the preserved measurements and rebuild their tables without inference. |

## A few results worth investigating

These observations come from an NVIDIA Quadro RTX 5000 with 16 GiB VRAM. The compact
optimization workload contains **four distinct questions**. Repeated requests help
us inspect timing; they do not turn four questions into a broad quality benchmark.

| What happened | Why it matters |
|---|---|
| With Qwen FP16 caching enabled, paired cold/warm median TTFT fell from **2.019 to 0.083 s**. Request completion time fell from **6.038 to 4.039 s**; direct reference matches stayed **10/20** in both phases. | Reusing input-prefix work can improve the first response much more than the complete request. |
| Nemotron's retrieved-schema application result matches were **1/4**, versus **2/4** with full schema. | Sending fewer tokens can also remove information the model needs. |
| The first Qwen NIM smoke returned all four responses but executed **0/4** application SQL queries. A schema-constrained rerun executed **3/4**. | A working endpoint and a working application are different milestones. |

## Running on your own GPU

The recorded study used native **vLLM 0.32.0** and a digest-pinned **NIM 2.1.5**
model-free image on a **Quadro RTX 5000, 16 GiB**, with driver **595.91.07**.
Qwen3-4B and NVIDIA Nemotron 3 Nano 4B were loaded as FP16; Nemotron retained FP32
Mamba SSM state. NIM used a vLLM backend—it was not automatically a TensorRT conversion.

GPU runtimes are separate from the CPU notebook environment. Start with
[installation prerequisites](SETUP.md), then [configuration and execution](tutorial/RUN_ALL.md).
NIM also requires Docker, NVIDIA Container Toolkit, and a profile selected for your
model/image/GPU combination. Compatibility on the recorded laptop doesn't establish
compatibility on every 16 GB GPU.

The CPU environment was pinned and tested on macOS arm64 with Python 3.11.15.
The supplied validation record reports a fresh-clone Replay/Plan check and **204 tests**;
[hosted Linux CI passed](https://github.com/manimalakumar/local-ai-inference/actions/runs/42666644695)
for commit `56feab1`. See [package metadata](CPU_NOTEBOOK_ENVIRONMENT.json) and
[validation history](VALIDATION.md) for the scope of those checks.

## What these experiments tell us

The dataset is synthetic and contains no patient data. We use clinical analytics as
one concrete workload, not as evidence of performance across application domains.

- Structured data execution, matching reference rows, and human-reviewed semantic correctness
  are different outcomes. Human semantic review is still a follow-up.
- Native VRAM sampling failed in the archived runs. Those records do not support
  a claim about memory savings. Keep missing readings as unavailable, not zero.
- NIM/native comparisons need matching model, prompts, request contracts, backend
  settings, and thinking behavior before attributing a difference to deployment.
- We did not obtain a validated Nemotron AWQ checkpoint. Qwen's AWQ experiment does
  not establish that the same path works for Nemotron.
- The later automated GPU lifecycle and corrected native-memory sampler still need
  fresh GPU validation. The completed GPU runs and offline notebook replay verify different parts of the workflow.


