# ExecuTorch: recorded CPU portability extension

Start with [05 — Edge portability with ExecuTorch](../notebooks/05_edge_portability_executorch.ipynb).
This read-only chapter adapts the existing SmolLM2 work. Run All in the CPU notebook environment;
it verifies and displays the October 29, 2026 Mac hardware record without installing an edge
runtime, downloading a model or conducting a new experiment.

The measured model was **HuggingFaceTB/SmolLM2-135M**, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, exported as **FP32 XNNPACK PTE** and executed
on Apple M4 Pro CPU. Llama 3.2 1B was originally planned but its checkpoint access was
unavailable. SmolLM2 supplied a public small-model portability example; it was not substituted
into the clinical-model comparisons.

The [completed hardware record](../experiment-tracker/tutorial/edge/20261029T161510_034051Z.json)
contains model/tokenizer/runner checksums, conversion-logit checks, measured timings,
footprint and sampled RSS. The chapter checks that record's checksum before displaying it.
Its builder is [build_edge_notebook.py](../scripts/tutorial/build_edge_notebook.py); edits
belong in [the chapter source](../src/clinical_trial_mvp/tutorial/edge_notebook.py).

The [unchanged historical guide](../baseline/mvp3/tutorial/EXECUTORCH_MAC.md) describes the
original conversion/export/runner recipe. Its commands reference the earlier project;
those scripts and local exported weights are not included as a newly validated live path
here. Chapter 05 provides recorded replay only. No new export/run was performed for this update.

One warm-filesystem sample does not establish cold-start percentiles, SQL quality,
concurrent-serving performance or clinical-model parity. Artifact size excludes runtime
libraries; process RSS includes them and is not VRAM. Runtime operator fallback remains
unreviewed. See [the deployment decision table](RESULTS.md#what-should-i-choose).
