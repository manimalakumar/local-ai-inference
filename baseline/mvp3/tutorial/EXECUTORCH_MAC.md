# Real ExecuTorch run on Apple Silicon

On October 29, 2026, the public **HuggingFaceTB/SmolLM2-135M** model was converted, exported to an XNNPACK PTE, and run on the Apple M4 Pro CPU. The notebook now replays the saved hardware record. The originally planned Llama 3.2 1B checkpoint was gated and unavailable; it was not used for this measurement.

This is a separate edge portability case. It does not measure clinical Text-to-SQL quality or compete directly with the larger models in MVP1–MVP3.

## Reproduce from a fresh clone

From the repository root, with Python 3.11 installed:

```bash
python3.11 -m venv .venv-edge
.venv-edge/bin/python -m pip install --upgrade pip
.venv-edge/bin/python -m pip install -r requirements-edge.txt
.venv-edge/bin/python scripts/tutorial/run_edge_mac.py --download --export --run
```

The flags are explicit:

- `--download` obtains the public checkpoint at immutable revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` and its tokenizer.
- `--export` converts all checkpoint weights, compares two sets of output logits against the original Hugging Face model, and exports the XNNPACK PTE. Export runs offline after download.
- `--run` uses the native C++ `TextLLMRunner` through the Python bindings, records real observer timings, samples child-process RSS and saves a portable result. It does not perform PyTorch eager inference in place of ExecuTorch.

The pinned dependencies were installed and exercised on macOS arm64. Downloads require network access; the export and run use local artifacts. Allow roughly 2 GiB of local storage for source weights, converted weights and exported model, plus the isolated Python environment. The model is FP32, not a quantized optimization result.

The upstream SmolLM2 conversion helper depends on a TorchTune interface that did not match this toolchain. This recipe maps the pinned model's complete bias-free Llama architecture directly to the upstream `Llama2Model` parameterization, preserves the Hugging Face rotary layout, rejects missing/extra keys, and checks logits before export. Both input sequences passed at `atol=rtol=0.001`; the observed maximum absolute logit error was **0.000043869**. `llama2` names the exporter's architecture class, not the downloaded model's identity.

The native runner must register the packaged LLM custom kernels before loading the PTE. `edge_python_runner.py` imports those registrations; without them, `llama::update_cache.out` and `llama::custom_sdpa.out` are missing. Run it with the isolated interpreter, not the resolved underlying system interpreter.

## Measured run

| Metric | M4 Pro CPU / XNNPACK |
| --- | ---: |
| Native model load | 0.076 s |
| Native single-request inference, including prompt evaluation | 0.199 s |
| Whole child process, including Python/runtime initialization | 3.480 s |
| PTE + tokenizer footprint | 623.60 MiB |
| Sampled peak runner-process RSS | 1018.13 MiB |
| Native reported prompt/generated tokens | 4 / 31 |
| Reviewed operator fallback | Unknown |

These values come from the completed record in `experiment-tracker/tutorial/edge/`; unsuccessful attempts are retained there with `evidence: none`. The default notebook loads only completed hardware records. The footprint excludes Python/PyTorch libraries. Process RSS includes those loaded libraries and is not GPU memory. Sampling every 0.1 s can miss short peaks. The filesystem was warm after export and earlier smoke checks; this is one measured portability example, not a cold-start or percentile benchmark.

Native token counts are preserved as reported by the runner; callback text pieces are not used as a token estimate. Model-load time covers native loading and does not include Python import/startup. Whole-process time includes that overhead. No reviewed ETDump/ETRecord trace was produced, so fallback is not inferred from export-time delegation.

Prompt: `Once upon a time`

Generated text:

> , there was a little girl named Lily. She lived in a big house with her family, but she didn't have many toys to play with. One day

The base model is a small completion model, not a clinical instruction model. This sample establishes that the exported program generates text on this Mac.

## Rerun and update the notebook

Once artifacts exist:

```bash
.venv-edge/bin/python scripts/tutorial/run_edge_mac.py --run
.venv-edge/bin/python scripts/tutorial/validate_notebook.py --execute --output notebooks/clinical_trial_inference_tutorial.ipynb
```

Each run writes a new timestamped JSON record; prior attempts remain intact. The setup writes ignored `tutorial/edge.local.json` to configure live notebook execution. Leave `RUN_EDGE=False` to replay saved evidence without importing ExecuTorch or needing local weights. Set `RUN_EDGE=True` to measure the configured local PTE from the notebook. Use the command-line runner above to save a fully versioned new record.

## Files and Git

Track the scripts, `requirements-edge.txt`, notebook, this guide and small JSON evidence records. `.gitignore` excludes `.venv-edge/`, `models/`, `.local-serving/` logs and `tutorial/*.local.json`. No weights or environment files need to be committed. This work is in the standalone published repository; it has not been pushed automatically.

Original Llama 3.2 export support remains in `scripts/tutorial/export_edge.py`; that model path still needs its own access, toolchain and live validation. NIM, GPU optimizations and DLSIM have not been validated by this CPU run.

Primary references: [ExecuTorch model export and Python runtime](https://docs.pytorch.org/executorch/stable/getting-started.html), [native LLM runner](https://docs.pytorch.org/executorch/stable/llm/run-with-c-plus-plus.html), [SmolLM2 source model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M).
