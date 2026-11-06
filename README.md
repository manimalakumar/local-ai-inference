# MVP4: Optimization

Clinical Text-to-SQL inference optimization: **context, caching, concurrency and precision**.
**MVP4 is baselined on the cumulative work through MVP3**, including MVP1 context/retrieval
and quality lessons, MVP2 local serving comparisons, MVP3 GPU serving, and the separate
ExecuTorch portability case. It keeps the synthetic ten-table DuckDB database, SQL guards,
reference scorer and four-question smoke workload. The initial repository state is tagged
`mvp4`; inherited measurements are historical baseline evidence, not new optimization results.

| Notebook | Scope |
|---|---|
| [Qwen3-4B](notebooks/local%20AI%20optimization%20-%20Qwen.ipynb) | Existing FP16/AWQ W4A16 × cache off/on strategy; AWQ compatibility smoke first; separate cold/warm requests |
| [Nemotron 3 Nano 4B](notebooks/local%20AI%20optimization%20-%20Nemotron.ipynb) | Core FP16 baseline, prefix caching, full/retrieved schema, client concurrency, scheduler capacity, reasoning and output caps; optional validated AWQ stretch |
| [NIM Qwen3-4B](notebooks/local%20AI%20optimization%20-%20NIM%20Qwen.ipynb) | Exact Qwen model/profile discovery, deployment, SQL smoke, full/retrieved schema and concurrency 1/2/4 |
| [NIM Nemotron 3 Nano 4B](notebooks/local%20AI%20optimization%20-%20NIM%20Nemotron.ipynb) | Exact Nemotron model/profile discovery with hybrid-state checks, deployment, SQL smoke, context and concurrency |

**GPU measurements are pending.** Unit tests run on CPU with fake model transports and
real read-only SQL scoring. They do not establish kernel, checkpoint or GPU compatibility.
All four notebooks execute offline by default and record actual GPU evidence only after live runs.
The additional NIM notebooks have a [model-specific run guide](tutorial/NIM_MVP4.md). They do
not imply that a compatible NIM profile for either model has already been validated on the Dell.

## Start

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m jupyter lab notebooks
```

See [setup and GPU workflow](SETUP.md). Use one resident GPU server at a time and a fresh
run tag for each launch set. The notebook defaults point to **local HF snapshots**, not GGUF.
No models or environments are stored in Git. The included database and CSVs are synthetic.

## Experiment controls

- Qwen retains its four precision/cache configurations. The Qwen notebook's experiment
  text and cells are carried over from MVP3 without changing that strategy.
- Nemotron core has **no AWQ dependency**. Validate its unquantized snapshot and FP16
  compatibility first. A cache experiment can be skipped if the installed hybrid backend
  is unsupported; all other core experiments remain available.
- Concurrency means simultaneous client requests. Scheduler `max-num-seqs` is a separate
  capacity control; achieved batch occupancy is not measured. Compare concurrency 4 at
  capacity 4 against concurrency 4 at capacity 1 to isolate capacity.
- Cold cache means a confirmed empty prefix cache on an already loaded/warmed engine.
  Cache-on trials reset, measure cold, reset again, prime with a different question using
  the same schema prefix, then measure the identical target warm. Startup, readiness,
  resets and priming are outside target request timing.
- Review SQL execution/result match, exact-SQL semantic review coverage, TTFT, request E2E,
  application E2E and sampled GPU memory together. Four cases are a functional smoke test;
  the dataset has 23 reference-SQL questions out of 30 total cases. No unreviewed output
  is counted as reviewed semantic correctness.

Nemotron's [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)
requires vLLM >= 0.15.1 and documents `enable_thinking`; its serving example uses FP32 Mamba
SSM state. This repo holds that state precision fixed while loading weights as FP16.
[Hybrid-cache support](https://docs.vllm.ai/en/latest/usage/v1_guide/) and
[Mamba cache controls](https://docs.vllm.ai/en/v0.31.0/configuration/engine_args/) vary by
version/backend. The notebook requires actual support verification before enabling that row.

## Contents and lineage

See [the cumulative baseline](BASELINE.md) for the experiments and lessons inherited from
each iteration. Selected original reports and GPU evidence are retained under `baseline/`.

`src/clinical_trial_mvp/` contains the inherited clinical pipeline and the MVP4 study helpers.
`scripts/serving/` provides offline previews and explicit launches; `scripts/tutorial/` rebuilds
the vLLM and NIM notebook pairs. `tests/` checks the packaged workload and both serving paths.
`SOURCE_LINEAGE.json` records hashes at the time source files were inherited; it is not a
checksum manifest of the modified final files. `VALIDATION.md` records the local checks.

The legacy served alias `clinical-trial-mvp3` and the `experiment-tracker/mvp3/retrieval-cache/`
path are retained for compatibility. Model ID/revision and exact artifact fingerprints establish
model identity. Results belong under `experiment-tracker/mvp4/` and remain unmeasured until
your GPU runs. The small ExecuTorch record is explicitly historical CPU portability evidence.

The repository has no configured remote. Choose your GitHub repository and push when ready.
See [third-party notes](THIRD_PARTY.md) for inherited licensing context.
