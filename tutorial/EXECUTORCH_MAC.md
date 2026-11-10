# Qwen3-4B through ExecuTorch CPU

Start with [05 — Qwen with ExecuTorch](../notebooks/05_edge_portability_executorch.ipynb).
The notebook defaults to Live: configure `edge_python` and execute the stages in order.
Set `MODE="replay"` explicitly to inspect the verified November 9 record without edge execution.
See [per-machine configuration and stage controls](RUN_ALL.md#notebooks-0506-local-executorch-stages).
Live can reuse a hash-verified export explicitly; generation and SQL checks always run afresh.

## Reference: previously measured Qwen run

The exact `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` was downloaded,
converted with the official Qwen3 mapping, checked against HF FP32 logits, exported to an
XNNPACK PTE and executed through the native C++ TextLLMRunner Python bindings on Apple arm64 CPU.
Its base snapshot fingerprint matches the saved Dell model, not just the model name.

The CPU recipe uses **8da4w: dynamic INT8 activations / INT4 linear weights, INT8 embeddings**.
It is different from the vLLM **AWQ W4A16** experiment. Two FP32 conversion checks passed
before quantization (maximum absolute logit error 0.000044823; atol=rtol=0.001).
This checks the unquantized mapping, not quantized application quality generally.

[Native hardware record](../experiment-tracker/tutorial/edge/qwen3-4b-run-20261109T195845_533008Z.json)
· [separate SQL smoke](../experiment-tracker/tutorial/edge/qwen3-4b-sql-smoke-20261109.json).
Native model load was 1.688 s, first token 0.195 s and inference 0.876 s. These are one
warm-filesystem short-schema sample, not stable percentiles or an HTTP serving benchmark.
The generated eval_001 SQL executed and matched its reference on the frozen synthetic DB;
there was no answer-plan/repair loop or human semantic review. No controlled CPU/GPU gain is claimed.

## Prepare the Live environment and artifacts

Use a separate Apple Silicon Python 3.11 environment. The original working MVP3 published
edge environment was reused for this attempt; it was not modified. A fresh setup uses:

```bash
python3.11 -m venv .venv-edge
.venv-edge/bin/python -m pip install -r requirements-edge.txt
.venv-edge/bin/python scripts/tutorial/run_qwen_edge.py --stage download
.venv-edge/bin/python scripts/tutorial/run_qwen_edge.py --stage prepare
.venv-edge/bin/python scripts/tutorial/run_qwen_edge.py --stage reference
.venv-edge/bin/python scripts/tutorial/run_qwen_edge.py --stage validate
.venv-edge/bin/python scripts/tutorial/run_qwen_edge.py --stage export
.venv-edge/bin/python scripts/tutorial/run_qwen_edge.py --stage run
```

Stages are separate processes to avoid keeping both large reference/mapped models resident.
The tested host has 48 GiB RAM. Weights/conversion/PTE artifacts use substantial disk space
and remain in ignored `models/edge-qwen3-4b/`. Export took about five minutes on this host.
This is a measured compatibility example, not a guarantee for arbitrary CPUs or toolchains.
Each stage saves a new small result record; record paths are printed. Retain failures too.

After the native run, use the ordinary CPU notebook environment to check its SQL outside timing:

```bash
.venv/bin/python scripts/tutorial/check_edge_sql.py --record experiment-tracker/tutorial/edge/REPLACE_WITH_YOUR_NATIVE_RUN.json
```

The scoring script strips only a terminal Qwen `<|im_end|>` marker, preserves raw text and
runs bounded read-only SQL plus the existing reference comparator. It checks the local base
snapshot against Dell's recorded fingerprint. A result match is not a human semantic verdict.
New records do not automatically replace the notebook's explicitly selected/checksummed replay.

## Native kernel registration failure and fix

The [first run](../experiment-tracker/tutorial/edge/qwen3-4b-run-20261109T195758_399022Z.json)
failed loading `quantized_decomposed::embedding_byte.dtype_out`. The runner had registered
LLM custom ops but not the quantized embedding kernels. Importing `executorch.kernels.quantized`
before constructing TextLLMRunner fixed it; the same PTE ran without re-export.
The [diagnosis](../experiment-tracker/tutorial/edge/qwen3-4b-kernel-registration-diagnosis.json)
preserves the specific error separately from the original failed record.

The helper follows the [official Qwen3 example](https://github.com/pytorch/executorch/blob/v1.5.1/examples/models/qwen3/README.md)
and [native runner registration](https://github.com/pytorch/executorch/blob/v1.5.1/examples/models/llama/runner/native.py).
Package versions are pinned in [requirements-edge.txt](../requirements-edge.txt).

## Earlier SmolLM2 work remains historical

The [unchanged SmolLM2-135M record](../experiment-tracker/tutorial/edge/20261029T161510_034051Z.json),
[previous chapter copy](../notebooks/history/edge-smollm2-before-qwen.ipynb) and
[historical recipe](../baseline/mvp3/tutorial/EXECUTORCH_MAC.md) are preserved. The originally
planned Llama 3.2 1B checkpoint lacked access; SmolLM2 supplied the earlier small public model.
That choice did not establish that Qwen/Nemotron could not run through ExecuTorch. This new
Qwen result addresses Qwen only; Nemotron was not exported in this attempt.

PTE/tokenizer footprint excludes runtime libraries. Process RSS includes them and is not
VRAM. Sampling can miss brief peaks; operator fallback remains unreviewed. See
[the deployment decision table](RESULTS.md#what-should-i-choose).

## Nemotron: separate compatibility chapter

[Chapter 06](../notebooks/06_nemotron_executorch.ipynb) checks the same
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` revision used on Dell:
`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`.
The November 9 probe downloaded only the pinned configuration and attempted both
`nemotron_h` and `nemotron_3_nano_4b` against the installed ExecuTorch 1.5.1 LLM
configuration. Both were rejected. No weights, graph capture, PTE or inference
were attempted. This is a limitation of the installed standard recipe; a custom
adapter remains unvalidated. Qwen's adapter cannot preserve Nemotron's Mamba state.

Chapter 06 defaults to a fresh local configuration probe. Select `MODE="replay"` explicitly
to read the saved evidence without edge subprocesses. Set `nemotron_config` to the pinned
local config and `edge_python` to the installed edge environment. Downloads remain opt-in.
To repeat the config-only probe using your separately installed edge environment:

```bash
.venv-edge/bin/python scripts/tutorial/probe_nemotron_edge.py --output experiment-tracker/tutorial/edge/nemotron-compatibility-new.json
```

Choose a new output filename each time. Offline `--config /path/to/config.json`
is also supported, but that caller-provided file does not independently establish
revision identity. The probe does not execute remote model code.
A working port still requires validated hybrid weight conversion and state/cache
handling, graph capture, backend/operator coverage, PTE loading and native SQL
execution. Only then can TTFT, request latency and memory be measured.
