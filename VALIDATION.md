# Validation record

## November 9, 2026 — review follow-up and optional CPU portability

**204 tests pass in 51.41 seconds**. All six chapters (00–05) pass schema/link checks
and complete execution in real CPU Jupyter kernels in Replay and Plan modes. Replay outputs
are saved in the learner notebooks. Chapter 00 and optional chapter 05 are explicitly
read-only; only chapters 01–04 support fresh Live experiments.

Chapter 05 adapts the existing SmolLM2-135M FP32/XNNPACK record. Its evidence is checksum
verified, fallback and clinical quality remain unavailable, and no model/export experiment
was added. The original 65 protected Dell evidence hashes still match.

A reproducible human-review packet now inventories **61 distinct final SQL outputs** across
four questions from selected native/NIM application, direct-serving and cache-target paths,
plus five no-SQL outputs. Exact hashes and source JSON pointers are retained. These are
review tasks, not judgments; `semantic_reviews.json` stays empty until a human reviews them.

The Dell automated rerun and native memory validation remain pending on the GPU machine.
License work remains deferred. Private Git, clean-clone and hosted CI outcomes are recorded
separately below when performed; these local tests do not imply those checks passed.

## November 9, 2026 — serving foundation and publication polish

The full suite passes **201 tests in 42.58 seconds** in the pinned Python 3.11.15
CPU environment. All five chapters (00–04) pass notebook schema and local-link checks
and complete execution in real Jupyter kernels in **Replay and Plan** modes. Saved
notebook outputs are visibly labeled recorded evidence. No new GPU experiment ran.

Chapter 00 adapts the earlier serving tutorial through shared application/evaluation code
and historical evidence; it adds no duplicate application. Its foundation manifest checks
26 frozen inputs and source records, including the synthetic database fingerprint matched
to the Dell baseline. All 65 protected historical notebook/result/provenance hashes still
match. The schema, evaluation SQL, database and original failures are unchanged.

The Qwen output-token audit is reproducible from the paired cold/warm records, with aggregate
and per-case counts. Similar output lengths support the qualified latency interpretation;
client-estimated decode rates do not establish GPU kernel timings. Setup, execution, NIM
technical details and methodology now have separate documentation responsibilities.

Fresh automated Dell runs, native memory validation, human four-case semantic review,
a clean-clone Replay check of the reviewed commit and hosted CI remain pending. Changes
are local and unpushed. License work is deferred at the user's request.

## November 8, 2026 — learner chapters, recorded replay and pinned CPU setup

The full suite passes **195 tests in 41.30 seconds** in a freshly installed, isolated
Python **3.11.15** CPU notebook environment on macOS arm64. All **141 installed packages**
pass the dependency compatibility check. Exact pins are in
`constraints/cpu-notebook-py311.txt`; installed metadata is in `CPU_NOTEBOOK_ENVIRONMENT.json`.
GPU engine/Torch/CUDA packages are separate and were not installed or launched for this work.

All four numbered learner notebooks passed **nbformat schema validation, local source/link
checks, and complete execution in real Jupyter kernels** in both Replay and Plan modes.
The committed learner copies contain the CPU-replayed, visibly labeled recorded Dell outputs.
Unlike the earlier test environment described below, this new environment includes nbformat,
nbclient, Jupyter and Matplotlib. Real-kernel checks used local socket permission; no GPU or
external inference request was made.

Additional checks cover explicit recorded-run selection/checksums, unavailable-data handling,
absence of silent replay fallback after live failure, exact request payload construction,
recorded/planned server details, scored response inspection, paired-phase interpretation,
reproducible figure numeric inputs, pending human-review judgments, and clean Git-based source
export. Both figures were rendered and visually inspected. The clean-export test uses a
temporary committed repository and verifies that untracked caches/repository internals are absent.

All 65 protected historical notebook/result/provenance hashes still match. No original
measurements, failure records or human semantic-review judgments were rewritten.

The GitHub CPU workflow is configured but has not run on hosted Linux until these changes
are pushed. Fresh automated Dell GPU execution, native VRAM telemetry, human semantic SQL
review and project-license/publication-rights decisions remain **pending**. The archived
manual GPU runs do not validate the later automation. No public source release was created.

Earlier entries below describe their own source versions and environments.

## November 7, 2026 — preserved Dell evidence and reusable notebooks

The current suite passes **181 CPU tests**. All four clean notebooks match their builders
and every default code cell executes offline without GPU imports, Docker or network calls.
The original Dell notebooks are archived byte-for-byte. A checksum test verifies all 65
preserved notebook/result/provenance files, including original failures and inherited Qwen
source lineage. The raw GPU records were not rewritten.

New tests cover configuration typing and safe defaults, independent Nemotron core preparation,
unique provenance, explicit GPU Python, native readiness and process-group cleanup on failure,
refusal to touch external servers, NIM smoke gating and owned-container reuse/cleanup, retryable
cleanup failures, Qwen's schema-enforcement setting, and normalization of bare CUDA UUIDs for
NVIDIA-SMI memory sampling. Existing real DuckDB scoring and cache-protocol tests still pass.

Tested with the existing Python 3.11 CPU environment; no new dependency installation or GPU
launch was performed on the Mac. That environment lacks `nbformat`, so a separate nbformat
schema check was unavailable; JSON, builder equality and all code cells were checked by the
suite. The notebook requirements include nbformat for users installing the documented environment.

The new automation and UUID fix still require a **fresh Dell GPU run**. Historical native
GPU memory stays unavailable. Actual saved Qwen NIM unconstrained/constrained runs differ
in schema enforcement; backend thinking parity remains unverified. See [the findings](tutorial/DELL_RESULTS.md).

The earlier validation entries below describe their respective historical source versions.

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
