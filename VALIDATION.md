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

The repository contains only two current notebooks and no copied environment/model
weights or secrets. A text-path audit found no source-machine home paths. Historical
baseline reports and source notebooks are explicitly separate from new measurements.

**Pending on the GPU machine:** FP16/kernel compatibility, hybrid-cache support and
confirmed resets, a compatible Nemotron AWQ derivative if pursued, and all new performance
measurements. No GPU support, latency improvement or clinical-quality improvement is
inferred from the CPU unit tests.
