# MVP4 local validation — November 5, 2026

The packaged repository passed **138 tests** in **18.04 seconds** from its own root.
The tests used the existing Python 3.11 CPU environment; package versions are in
`TESTED_ENVIRONMENT.json`. A fresh GPU installation or clean dependency reinstall
was not part of this check.

Command: `python -m pytest -q`.

Validated behavior includes:

- Every default code cell in both notebooks executes offline without GPU imports.
- Notebook contents match their reproducible builders; Qwen is preserved byte-for-byte
  against the inherited source hash recorded in `SOURCE_LINEAGE.json`.
- Real read-only DuckDB scoring checks generated SQL against the included synthetic data.
- Cold/warm phases verify reset ordering, shared schema, distinct-question priming,
  separate timing/memory summaries, and failure handling for unconfirmed resets.
- Qwen's checkpoint-pair and AWQ-smoke gates remain intact.
- Nemotron core planning and fake-transport runs do not require an AWQ artifact.
- Nemotron client concurrency and scheduler capacity are independently configured;
  actual metadata must match requested weights, version and hybrid-state settings.
- Reasoning reaches the HTTP payload as `enable_thinking=True`; conflicting prompt
  directives are changed, raw thought output is retained and final JSON is scored.
- Unsupported hybrid caching and missing AWQ stretch evidence do not block core work.

At this initial baseline check, the repository contained two current notebooks and no copied environment/model
weights or secrets. A text-path audit found no source-machine home paths. Historical
baseline reports and source notebooks are explicitly separate from new measurements.

**Pending on the GPU machine:** FP16/kernel compatibility, hybrid-cache support and
confirmed resets, a compatible Nemotron AWQ derivative if pursued, and all new performance
measurements. No GPU support, latency improvement or clinical-quality improvement is
inferred from the CPU unit tests.

## November 6, 2026 — two additional NIM notebooks

The full suite now passes **160 tests** in **20.38 seconds**. This adds separate NIM
notebooks for the exact Qwen3-4B and Nemotron 3 Nano 4B targets, bringing the current
notebook count to four. The existing two vLLM notebooks are unchanged; the `mvp4` tag
continues to identify the original cumulative baseline commit.

The additional checks validate model-specific profile templates and staged architecture,
digest-pinned image commands, environment-name-only credential handling and redaction,
explicit runnable-profile selection, immutable deployment metadata, model/image/config/GPU
binding, readiness and served-alias checks, startup failures, real read-only clinical SQL
scoring through fake transports, and every default NIM notebook cell without GPU imports,
Docker or external calls. Injected transports remain test evidence, not hardware results.

NIM container/image access, runnable profiles for these exact models on the Dell, bundled
backend support and effective precision/thinking settings still require actual target-host
verification. Cold/warm caching and AWQ are conditional extensions requiring separately
validated deployment recipes; the initial NIM helper accepts staged unquantized FP16 only.
