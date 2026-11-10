# Validation record

## November 9, 2026 — notebook 00 local-serving scope and readable results

Removed the vLLM/SGLang concurrency section, its execution cell and saved GPU
table from chapter 00. The remaining walkthrough covers context selection,
llama.cpp/Ollama serving and a structured answer, followed by baseline metrics.
The applicable review TODOs were retained; later notebook-only prose/TODO edits
were synchronized without restoring notes the owner had removed.

Schema-context and local-serving results now render as Markdown tables rather
than nested JSON. Measurements are unchanged. The worked answer now comes from
the actual `eval_001` Nemotron Q4_K_M/llama.cpp record found in MVP3, replacing
the vLLM answer previously displayed in this chapter. Its source JSONL and manifest
were copied byte-for-byte into `baseline/mvp1/experiment-tracker/local-answer-example/`.
The JSONL SHA-256, GGUF artifact identity and matching database fingerprint are
checked before loading. The displayed aggregate answer contains 44/48/40/38
subjects and retains the original execution/reference-result scores. No new
model inference or human semantic judgment is claimed.

Chapter 00's default/example Live configuration and SETUP instructions now use
a local CPU GGUF endpoint. New local configuration and foreground launch helpers
record artifact identity, runtime version, command and process PID. The starter
pins the original Nemotron Q4_K_M SHA-256 and rejects other files, occupied ports
and reused metadata. Its native end-to-end Live run has not been performed here;
the helper/config rejection paths are tested without launching an engine.

## November 9, 2026 — chapter 00 Replay and fresh baseline options

Chapter 00 now branches on `replay`, `live` or optional `plan`. Replay remains
offline and uses the identified supplied results. Live attaches to explicitly
configured, already running local endpoints and calls the existing clinical
benchmark; it never substitutes archived tables or SQL for a failed fresh row.
The previous executed notebook, including review TODOs and outputs, is preserved
as `notebooks/history/00_local_serving-before-live.ipynb`.

The starter setup downloads the pinned Qwen3-4B base revision and prints an explicit
foreground vLLM launch command. It uses a separate GPU interpreter, rejects an
occupied port/existing metadata filename, and retains new launch provenance.
Fresh notebook baselines cover full/retrieved schema, configured concurrency,
application quality and a worked answer from the new full-schema record. Optional
llama.cpp/Ollama sections stay visibly unselected unless endpoints are supplied.
The caller starts/stops its server; startup is outside request timing. Frozen data
and retrieval remain unchanged, so “from scratch” means new inference on fixed inputs.

Settings live in `tutorial/config/local-serving.local.json`, with a committed
example and Replay/Live instructions in SETUP.md. Live rows and failures are
saved under session-specific `experiment-tracker/tutorial/foundation/` paths.
The shared benchmark's optional `include_policy_cases` flag lets this baseline
evaluate all 30 selected quality cases while timing the 26 answerable serving
cases; the earlier callers' default behavior is unchanged. Default chapter-00
selection remains the four compact questions.

**243 tests passed in 44.37 seconds.** These include new tests for mode isolation,
failed-row persistence, current-session answer reuse, complete quality-case
selection, config/launch guards and non-overwriting setup. Notebook 00 passed real
offline Jupyter Replay and Plan execution with final sources; Replay outputs were
saved. All seven schema/link/builder checks passed. All 13 owner TODOs remain.
No fresh GPU or optional local-serving hardware execution was performed; Live
end-to-end validation remains pending. No new GPU measurements are claimed.

## November 9, 2026 — notebook prose review using the owner's TODOs

Read the accessible full history of the ChatGPT thread “Local AI Career Path”
and the owner's 13 TODO notes in chapter 00. Revised 58 Markdown cells across
chapters 00–06 to explain the application, each experiment's purpose, its hardware
and timing scope, and what to inspect in the next cell. The shared workflow now
calls chapter 00 “How a local AI request works.” Mode descriptions distinguish
reading saved measurements from running new inference. NIM chapters explain
containers, profiles, and the difference between database schema and plan JSON schema.

All 13 original TODO notes remain visible beside the corresponding revised prose;
only trailing whitespace was normalized. Chapter 00–05 prose and these review notes
are preserved in `chapter_prose.py`; chapter 06 text remains in its builder.
README workflow labels were synchronized. Reconstruction details were moved out
of the teaching narrative into the already ignored `.local-serving/notes.md`;
existing output origin fields, source records and audit documents remain intact.

Compared every code-cell object against a snapshot taken before the edits:
code, saved outputs, execution counts and code-cell metadata are unchanged in all
seven notebooks. All seven schema/local-link/builder-source checks passed, along
with **78 relevant existing tests in 2.21 seconds**. No new inference or notebook
kernel execution was needed for this prose-only revision. Changes remain local.

## November 9, 2026 — separate Nemotron ExecuTorch compatibility chapter

Added chapter 06 using the exact Dell model ID and revision:
`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16@dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`.
Reused the MVP3 edge environment without modification. Downloaded only pinned
`config.json`; the installed ExecuTorch 1.5.1/OmegaConf LLM export configuration
rejected both `nemotron_h` and `nemotron_3_nano_4b`. The original exceptions,
configuration, source checksum and installed model-class inventory are saved in
`experiment-tracker/tutorial/edge/nemotron-4b-compatibility-20261109.json`.
Chapter replay checks that record's SHA256 and exact model/revision.

This establishes a limitation of the installed standard recipe, not impossibility
of a custom port. We did not download weights, execute model repository code,
convert weights, capture a graph, export a PTE or run native Nemotron inference.
SQL correctness, TTFT, request latency and memory remain unavailable. No Qwen
adapter, smaller model or other runtime was substituted as Nemotron evidence.

README and all seven chapter introductions share the updated workflow. The
validator includes chapter 06 in schema/source checks and Replay/Plan execution.
All **227 unit tests passed in 47.28 seconds**; all seven chapter schema/link/builder
checks passed. Chapter 06 also passed real offline Jupyter Replay and Plan runs;
Replay outputs were saved. Other existing inference outputs were preserved.
No new GPU automation, human semantic reviews, hosted CI or clean installation
is claimed by these checks. Changes remain local and unpushed.

## November 9, 2026 — requested Qwen3-4B ExecuTorch attempt

At the owner's explicit request, the Qwen model from the GPU chapters was attempted on
ExecuTorch CPU. The existing edge environment was found in MVP3's published copy and reused
without modification. A separate MVP4 environment setup had been interrupted; it was not the
successful runtime. Installed versions used were ExecuTorch 1.5.1, Torch 2.15.0, TorchAO 0.18.0,
Transformers 5.22.0; the host is Apple arm64 with 48 GiB RAM. This is a new edge experiment,
not the deferred native-vLLM/NIM GPU lifecycle rerun or human semantic review.

Pinned Qwen3-4B revision: `1cfa9a7208912126459214e8b04321603b3df60c`. The complete base snapshot
fingerprint matches the recorded Dell FP16 model. Official Qwen3 conversion passed two HF
FP32 reference-logit checks at atol=rtol=0.001 (maximum absolute error 0.000044823).
The 4096-context XNNPACK 8da4w export completed in about five minutes. This uses dynamic INT8
activations/INT4 linear weights and INT8 embeddings, not AWQ W4A16.

The first native load failed due to missing `quantized_decomposed::embedding_byte.dtype_out`.
Adding `executorch.kernels.quantized` registration fixed the same PTE without re-export.
The successful native run recorded model load 1.688 s, first token 0.195 s, inference 0.876 s,
52 prompt/29 generated tokens, 2,480,823,486 bytes PTE+tokenizer and sampled peak process RSS
5,272,551,424 bytes. Native observer timings exclude Python startup; whole-process time
includes imports and artifact checking. One warm-filesystem sample is not a latency benchmark.

Generated eval_001 SQL executed and matched the existing frozen-database reference, after
stripping only the terminal Qwen `<|im_end|>` marker outside timing. This short-schema direct
smoke has no answer-plan/repair loop or human semantic verdict. No controlled CPU/GPU gain,
full clinical-task quality or reviewed operator-fallback conclusion is claimed.

Results/failure/SQL evidence and the previous SmolLM2 chapter are indexed by
`experiment-tracker/tutorial/edge/qwen3-4b-manifest.json`. The original 65 Dell hashes remain
unchanged. Chapter 05 now defaults to checksum-verified Qwen replay; SmolLM2 history is retained.
**97 relevant tests passed in 11.23 seconds**, all six static schema/link/builder checks pass,
and chapter 05 executed in a real CPU Jupyter kernel with replay outputs saved. No Torch or
model download is needed for that replay. Fresh export/run uses the explicit documented stages.

## November 9, 2026 — shared workflow and notebook location

README now uses a shared Step / Notebook / Purpose table under Start here. The first
Markdown block of every learner notebook (00–05) contains the same linked workflow and
one highlighted current location. The map separates the main serving/optimization path
from optional NIM and CPU-portability extensions.

The working notebooks' newer prose was preserved in the builder source. All six notebooks'
code cells, saved outputs, execution counts and code metadata matched their pre-change
hashes after regeneration. Static schema/link/builder checks passed for all six chapters;
**95 relevant tests passed in 10.11 seconds**. No model/server or new GPU run was used.

## November 9, 2026 — final local CPU readiness check

Source under test: **`7e0bd8ff0568fc2b93ef7d68a32820beadce84f3`**, with the owner's
documentation-only deferral changes applied. **222 tests passed in 43.29 seconds.** All six
notebooks passed schema, local-link and builder-source checks and complete real Jupyter
execution in both Replay and Plan modes. No executed notebook outputs or original records
were rewritten. The resumed full test run replaced an earlier user-interrupted attempt;
the interrupted attempt is not counted as a pass.

Commands used the existing pinned Python 3.11.15 environment:

```bash
/tmp/local-ai-inference-clean-20261109/.venv/bin/python -m pytest -q
/tmp/local-ai-inference-clean-20261109/.venv/bin/python scripts/tutorial/validate_notebooks.py --mode replay
/tmp/local-ai-inference-clean-20261109/.venv/bin/python scripts/tutorial/validate_notebooks.py --mode plan
```

Jupyter checks isolated runtime state with `IPYTHONDIR=/tmp/local-ai-final-ipython` and
`JUPYTER_RUNTIME_DIR=/tmp/local-ai-final-jupyter`. This was final local CPU validation,
not another clean dependency installation or a new hosted CI run. The earlier fresh-install,
remote-clone and hosted Linux CI results remain explicitly scoped to `56feab1` below.
Before changing repository visibility, push the final candidate and confirm CI on that commit.
Human SQL review and fresh GPU-automation/native-memory validation remain deferred; no new
GPU or model experiment ran. License work remains deferred. A documentation-only commit
records this outcome and exports the source candidate without environments or Git internals.

## November 9, 2026 — owner defers two follow-up validations

At the owner's request, human semantic review and fresh validation of the updated automated
GPU lifecycle/native memory sampling are parked to save time. They are excluded from the
current educational/replay polish rating and are not treated as completed checks. Original
manual GPU results, the 0/61 review status, missing native VRAM and both future checklists
remain intact. No model, experiment, scoring verdict or notebook result changed. The final
CPU clean run remains a separate end-of-polish activity. This update changes documentation only.

## November 9, 2026 — iterative source review; final clean run deferred

The supplied working-directory ZIP was inspected: **22,689 entries**, including **21,670
environment entries** and **602 Git-internal entries** (directory entries included). It is
not a release source package. The missing working directory was restored from its source
and repository files without importing that environment or the caches. The original ZIP
was left unchanged. A separate committed-file source export is used for the review artifact.

Fixes: valid human judgments no longer fail an empty-review assertion; review entry validates
exact case/hash, date, attribution and rationale, rejects duplicate/unknown entries atomically,
and refreshes packet coverage without rewriting measurements. Builders retain outputs only
for byte-identical code cells. All six chapters now receive static builder/source checks;
CI additionally rejects drift in the generated JSON/review packet. The exporter rejects
tracked local configs, raw logs and nested ZIPs as well as environments/weights/Git internals.

**93 affected tests passed in 43.14 seconds** using the existing isolated Python 3.11.15 CPU
environment. After adding an attributed-response overlay check, all **18 focused review/export/
builder regressions passed in 1.76 seconds**; this subset overlaps the affected-test run. The six notebook schema/link/builder checks passed in `--check-only` mode; no
kernels or servers were started by that check. All 65 historical and 26 foundation file
hashes match. Commands:

```bash
python -m pytest -q tests/tutorial/test_review_gaps.py tests/tutorial/test_harness.py tests/tutorial/test_replay.py tests/tutorial/test_edge_review.py tests/tutorial/test_nemotron.py tests/tutorial/test_nim_study.py
python scripts/tutorial/validate_notebooks.py --check-only
python scripts/tutorial/review_sql.py
```

The Python used was `/tmp/local-ai-inference-clean-20261109/.venv/bin/python`; this reused
an existing environment, not a new clean install. Review coverage remains **0/61**; no
human judgment was invented. **No final clean clone, full-suite run, Replay/Plan kernel run,
new hosted CI or GPU experiment was performed for this revision.** These remain scoped final
checks. See the [current rubric and explicit gaps](tutorial/PUBLICATION.md#current-iterative-review--november-9-2026).

## November 9, 2026 — private commit, hosted CI and fresh remote clone

Tested source commit: **`56feab13f6d5e7feb3cb3988658040b1fc81497b`**
(`Build replay-first inference lab chapters and preserve GPU evidence`). It was pushed to
`manimalakumar/local-ai-inference` on `main`; GitHub API metadata confirmed the repository
remains **private**. No public release or source ZIP was published.

The [hosted CPU workflow passed](https://github.com/manimalakumar/local-ai-inference/actions/runs/42666644695)
on that exact commit. Its Linux job installed the CPU constraints, ran the suite, executed
Replay and Plan, rebuilt figures/token analysis/review packet and checked whitespace.

A new HTTPS clone at `/tmp/local-ai-inference-clean-20261109` resolved to the same full SHA.
From that clean directory, the README environment/install path was exercised:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt -c constraints/cpu-notebook-py311.txt
.venv/bin/python scripts/tutorial/validate_notebooks.py --mode replay
.venv/bin/python scripts/tutorial/validate_notebooks.py --mode plan
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
git status --short
```

The interpreter used to create the new venv was the installed Python 3.11.15. Local kernel
checks additionally set `IPYTHONDIR=/tmp/local-ai-clean-ipython` and
`JUPYTER_RUNTIME_DIR=/tmp/local-ai-clean-jupyter` to isolate Jupyter state. All six notebooks
passed real-kernel execution in both modes; dependencies were compatible; **204 tests passed
in 47.40 seconds**; tracked files remained unchanged. No model/GPU/edge inference ran.

Human review of the distinct SQL outputs, fresh Dell execution of automated chapters 01–04
and native VRAM validation remain pending. Native memory-savings claims remain excluded.
License work is deferred. This entry records validation of the source commit above; a later
documentation-only commit records the outcome without claiming new GPU evidence.

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


## November 10, 2026 — live-first notebooks 01–06

- All seven learner notebooks passed schema, link and builder-source validation.
- All seven executed successfully in real CPU kernels in explicitly selected **Plan**
  and **Replay** modes. Those checks did not overwrite saved notebook outputs.
- **264 unit tests passed**, including mode precedence over stale local Replay,
  fresh inference dispatch, session-only reports/inspection/figures, failed reruns,
  unique evidence files, non-inference Plan/Replay and owned-server cleanup.
- Notebook 05 completed a genuine new Qwen CPU generation and SQL check. It
  explicitly reused a hash-verified export; it did not claim a fresh conversion/export.
  Session `20261110T121532-73d94917db`: generated SQL executed and matched its reference.
- Notebook 06 completed a fresh installed-exporter configuration probe.
  Session `20261110T122034-8766f89a12`: `blocked_compatibility`; no weights loaded,
  exported program, native inference or generation timing was claimed.
- Revised notebooks 01–04 have no saved outputs. Their existing Dell evidence and
  executed archives are preserved; no old Replay output was relabeled as Live.
  The new Linux GPU orchestration still needs a Dell run if fresh Live outputs for
  these revised notebooks are desired. Existing measurements remain usable through
  explicit Replay and the reference links without rerunning the GPU workload.

Current edge session records are under `experiment-tracker/tutorial/edge/sessions/`.
Notebook backups before these changes are under
`notebooks/history/before-live-first-2026-11-10/` with a SHA-256 manifest.

### Final pre-push check

All **272 tests passed** after adding the Dell launch scripts. All seven notebooks
passed schema, link and builder-source checks. Shell syntax and launcher dry-runs
passed. Model weights, virtual environments and local configuration files remain
excluded from Git. No new Dell GPU execution was performed for this push.
