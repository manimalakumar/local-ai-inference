"""Notebook teaching text; preserve reviewer TODOs when regenerating chapters.

Builders provide code/layout, and navigation.py supplies the shared workflow.
Edit these prose slots for chapters 00–05. Chapter 06 uses its own builder.
"""

CHAPTER_PROSE = {
    '00_local_serving.ipynb': {
        '7b5e354b23e7': r'''# How a local LLM answers questions using structured data

**Use case:** Help a clinical trial analyst answer questions about a study.
**Persona:** A clinical trial analyst.
**Job:** Analyze a clinical study using structured trial data.

The analyst asks questions such as, “How many enrolled subjects are in each
treatment arm?” The LLM needs more than the question: it needs context—the
database structure, the meaning of “enrolled,” and instructions for writing SQL.
The application then checks the generated SQL and runs it to produce the answer.

Along the way, we'll ask how much database context to send, how to serve the model, and how to
tell whether a quick response is also a useful answer. Our examples use **Qwen3-4B
and Nemotron 3 Nano 4B**, the two models used throughout to **compare how schema context and serving choices affect response time and structured answer quality**.

**Choose how to run notebook 00:**

| Option | What happens | What you need |
|---|---|---|
| **Replay** | Inspect the database and display the supplied experiment results | The notebook environment; no model download or inference server |
| **Live: run from scratch (default)** | Send the selected questions to your running model, execute its SQL, and save a new quality/timing baseline | Model weights, a compatible runtime and hardware, and the local configuration |

For the default Live mode, follow the [from-scratch setup in SETUP.md](../SETUP.md#notebook-00-replay-or-run-from-scratch), start the local server and choose **Run All**.
To explore the supplied results without a server, select **Replay** using the same setup guide.

In this notebook, **we compare full and retrieved schema context**, examine **local serving with llama.cpp and Ollama**, and **evaluate the generated SQL**. Together, these experiments **establish a baseline for response time and SQL answer quality**.
''',
        'af1ac4f75873': r'''### The questions we'll work through

We compare **Nemotron 3 Nano 4B (Q4_K_M)** and **Qwen3-4B (Q4_K_M)**
using full and retrieved schema context. The local-serving comparison uses the
same **Nemotron** GGUF file through llama.cpp and Ollama. The structured-answer
example also uses Nemotron through llama.cpp.

| Section | What we want to understand |
|---|---|
| 1. Workload | What structured data and questions must the application handle? |
| 2. Schema context | How do full and retrieved schema affect Nemotron and Qwen's response time and result quality? |
| 3. Local serving | How does the same Nemotron model behave through llama.cpp and Ollama? |
| 4. Structured answer | How does Nemotron's generated SQL become a structured answer with a quality check? |
| 5. Establish the baseline | What quality and timing baseline do these comparisons establish? |''',
        'b937d0a634e6': r'''## 1. The application and its data

Our application answers questions about subjects, sites, visits and adverse
events in a clinical trial. The database is synthetic; it contains no personally identifiable information (PII).
**DuckDB** is the SQL database engine that executes the generated queries.

Here's the path we'll follow:

**Question → schema context → model plan and SQL → validation → DuckDB → result comparison.**

A *schema* describes tables, columns and relationships. A *plan* states what the
answer should count or group before the model writes SQL. Validation checks
whether the generated query is allowed and executable; result comparison checks
its rows against a reference query.

Run the next cell to inspect the ten tables. This notebook reads the supplied
database without changing it. Using the same rows lets us compare the model's
query with the reference consistently.

The evaluation set contains **30 questions**, including **23 with reference SQL**.
The optimization chapters use four questions from that set for shorter experiments.
Repeating those four questions helps measure timing.
''',
        'ac6979b6039b': r'''## 2. Give the model the context it needs

To write SQL, the model needs table names, columns, joins and the definitions
behind the question. We can prepare that context in two ways:

- **Full schema:** include all the available schema descriptions.
- **Retrieved schema:** select the descriptions that appear relevant to this
  question, using embeddings to find related text.

Both approaches send text to the generation model. Retrieval reduces the amount
of text, but it may also leave out a definition or join the answer needs.

The next cell displays **both prompts for the same question**. Compare the full
prompt with the retrieved prompt: what was kept, and what was removed? The
displayed lengths count characters, not model tokens.

First inspect the fixed schema selection used by the baseline comparison.
Then run the embedding cell below to see how a fresh selection is produced.
The timing comparison measures answer generation and application work;
it excludes schema retrieval.
''',
        'embedding-invocation-explanation': r'''### Invoke the embedding model

**Nemotron-3-Embed-1B-BF16** selects schema context for both SQL-generating
models. It runs locally through Sentence Transformers, independently of the
OpenAI-compatible server used for plan and SQL generation.

The next cell loads the embedding model and calls `embedding_model.encode()`.
The retriever encodes schema descriptions with a **`passage: `** prefix and the
question with a **`query: `** prefix. It ranks their normalized vectors by
similarity, selects up to 12 facts, then adds required columns and join paths.
Read the selected facts and schema to see what the generation model could receive.

In **Live**, this is fresh embedding inference. Follow the
[embedding setup](../SETUP.md#notebook-00-embedding-model) first.
In **Replay**, this cell is skipped. The measured comparisons continue to use
the fixed retrieval selection displayed above so their inputs remain consistent;
this demonstration does not replace that selection or enter the latency table.
''',
        'first-api-explanation': r'''### Make the first model API call

Now send the first question and its full schema context to the running model.
The cell below shows the **OpenAI-compatible request**, executes
`POST /v1/chat/completions`, and prints the model's new response: an answer plan.
It uses **Nemotron 3 Nano 4B Q4_K_M through llama.cpp** with the default local setup.
The endpoint and served model name come from your local configuration.

Run the [local-server setup](../SETUP.md#notebook-00-replay-or-run-from-scratch)
first. **Live** executes this request; **Replay** skips it. This introductory
request is separate from the measured comparisons below. The application then
uses the plan in a second model call to generate SQL.
''',
        '0a756f9a063f': r'''### Compare full-schema and retrieved-schema answers

Now let's see what happened when each model answered using those two kinds of
context. This experiment used **Nemotron 3 Nano 4B and Qwen3-4B**, each in
**Q4_K_M GGUF** format through **llama-cpp-python 0.3.35**, on an **Apple M4 Pro
CPU** with GPU offload disabled. GGUF is the model-file format used by llama.cpp;
Q4_K_M identifies the quantized weight representation.

Both models receive the same context selected by Nemotron Embed. This keeps retrieval fixed for the comparison

Each model/context configuration in Replay answered the 30-question set once.
The next table compares result matches on the 23 questions with reference SQL and
**application latency**: planning, generation, checks and any repair. This is
the time to work through an answer, rather than the time to its first token.

**In Live:** the next cell runs both schema profiles against `context_server`
from your local config. The default selects four questions; choose all 30 IDs
to run the complete set. The fresh table includes application quality and direct
request timing, labeled separately.
''',
        'dd5f0ec80102': r'''### What changes with retrieved schema?

In this question set, both models had lower application latency with retrieved
schema. Reference-result matches decreased for Nemotron and increased slightly
for Qwen.

**Takeaway:** Reducing context can save time, but the model still needs the
right definitions and joins to produce a useful structured answer. Read latency
and result matches together when choosing how much context to provide.

See [the schema-comparison settings and detailed results](../baseline/mvp1/experiment-tracker/MVP1_Q4_30_CASE_REPORT.md).
''',
        '90c6cb614df5': r'''## 3. Serve the same model through llama.cpp and Ollama

Once we've chosen a model and prompt, the application needs a way to call the
model. **llama.cpp** gives us direct control over model execution. **Ollama**
provides model management and an API that the application can call.

Here we hold the **Nemotron 3 Nano 4B Q4_K_M GGUF file** fixed and compare those
two serving paths on an **Apple M4 Pro with Metal GPU offload enabled**.
Prompts, seed and output cap match; the measurements report no cached prompt
tokens. There are three measured requests per prompt for each runtime.

In Replay, run the next cell to compare **time to first token** and **complete request
time**. This measures a model-generation request. Section 2 measured the broader
application path on CPU, so the two sections answer different timing questions.

You can follow how the application calls each runtime in the
[runtime adapters](../src/clinical_trial_mvp/tutorial/adapters.py) and
[serving harness](../src/clinical_trial_mvp/serving/benchmark.py).

**In Live:** this section measures only the running llama.cpp/Ollama endpoints
listed in `local_servers`. Select the local serving endpoints you want to compare in your configuration.
To compare the two paths, supply metadata for the same GGUF artifact and matching
launch settings; these optional endpoint integrations still need fresh hardware validation.
''',
        '552eb017d529': r'''### Compare the serving paths

With selected schema, llama.cpp and Ollama produced matching outputs with similar
first-token and request-completion times. The table also includes full-schema
requests so you can compare the effect of context length within each runtime.

**Takeaway:** Runtime controls, API convenience and model management can guide
this serving choice alongside timing. These measurements cover individual model
requests; evaluate the resulting structured answers separately.

Each row summarizes three measured requests per prompt and runtime. See
[the serving setup, prompts and detailed results](../baseline/mvp2/experiment-tracker/mvp2/serving/MILESTONE_REPORT.md).
''',
        '18c915d61247': r'''## 4. Follow one question through to a structured answer

Let's connect the pieces with **Nemotron 3 Nano 4B Q4_K_M through llama.cpp**
on an **Apple M4 Pro CPU**. The next cell shows an actual question, the generated
SQL, its aggregate result rows and the reference-result check.

First identify what the question asks. Check whether the SQL keeps its filters
and grouping and counts the intended subjects. Then read the structured answer
and its quality checks. A query executing successfully is one check; returning
the expected result is another.

In Replay, the example comes from the supplied local-model evaluation. In Live,
the cell displays the full-schema application result collected in your session.

**Question → schema context → generated SQL → execution → structured answer → quality check.**
''',
        'db1301631036': r'''## 5. Establish the baseline

The table below brings together this notebook's schema-context, local-serving
and structured-answer results. Read each comparison's quality and timing together.
''',
    },
    '01_qwen_cache_quantization.ipynb': {
        'intro': r'''# Qwen: answer faster with caching and four-bit weights

We compare FP16 and AWQ W4A16 weights, each with prefix caching off and on. Caching reuses computation for shared instructions and schema; AWQ compresses weights while keeping activations at FP16.

The workload is structured clinical-trial questions on synthetic data with no personally identifiable information (PII). We compare response time with executed SQL and reference-result matches.

Configure your GPU environment below, then run the cells in order to collect new measurements. Each experiment saves its results separately. Completed outputs are visible when a live execution has been saved. For the previous Dell measurements, use the reference links at the end.

Follow [SETUP.md](../SETUP.md) and [the execution guide](../tutorial/RUN_ALL.md) first. The target is **Qwen3-4B**, `Qwen/Qwen3-4B`, on a **Linux NVIDIA GPU**. Model loading and server startup happen before request timing.
''',
        'configuration': r'''## 1. Imports and editable configuration

`MODE` controls execution: **live** collects new inference; **plan** previews commands; **replay** explicitly reads archived results. It overrides an older local JSON mode. CPU validation supplies `LOCAL_AI_LAB_MODE=plan` or `replay` before executing these cells.

Keep machine paths, exact checkpoint revisions and deployment settings in the per-notebook local JSON. Downloads are opt-in. Remove the obsolete `execute` flag; mode is the execution authority.
''',
        'preflight': r'''## 2. Preflight checks

Check the GPU, selected runtime, checkpoint paths/revisions and port 8000 before loading a model. NIM also checks Docker access, an already available image, GPU access from that image and your selected profile. Missing prerequisites stop with instructions; no package installation or image pull is automatic.
''',
        'prepare': r'''## 3. Model preparation and compatibility smoke

Preparation validates local checkpoint identity and precision. It downloads weights only when `download_models=true`. NIM verifies the selected profile against discovery for this deployment.

Qwen's vLLM study requires a successful AWQ SQL smoke before all four variants. Nemotron core needs no AWQ checkpoint: its optional stretch remains separate. NIM smoke tests both serving and SQL execution; a failed required smoke blocks dependent experiments.
''',
        'method': r'''## 4. Individual experiments

Hold the schema, four questions, seed 42, temperature 0, cap 1024 and context limit 8192 fixed. AWQ W4A16 changes weight storage, not KV-cache precision.

Cache-on targets use **reset → cold target → reset → different-question prime → identical target warm**, with an identical shared schema prefix. Cold and warm requests remain separate. Startup, readiness, warmup, reset, priming and SQL scoring are outside individual request timing.

Predict whether shared-prefix reuse helps TTFT more than complete-request latency. Each cell below starts its own server, makes new inference requests in Live mode, saves a new result and stops that server.
''',
        'experiment-fp16_cache_off': r'''### fp16 cache off

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'mechanism': r'''### Inspect the serving call and answer

The experiment above calls the model through the OpenAI-compatible `/v1/chat/completions` API. Inspect the launch command, request, generated plan/SQL and execution checks below. The request is constructed from the experiment prompt, schema profile and generation settings. In Plan, these details preview the configured call. The response shows the generated answer and its reference-result check.

[Request transport](../src/clinical_trial_mvp/serving/client.py) · [plan and SQL execution](../src/clinical_trial_mvp/serving/agent.py)
''',
        'experiment-fp16_cache_on': r'''### fp16 cache on

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-awq_w4a16_cache_off': r'''### awq w4a16 cache off

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-awq_w4a16_cache_on': r'''### awq w4a16 cache on

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'comparison': r'''## 5. Current-session comparison and interpretation

Read result matches alongside latency. TTFT measures first generated content; request completion covers generation and transport; application latency includes planning, validation, execution and repair. Missing memory telemetry stays unavailable. Failed rows remain failures.

Compare cold and warm TTFT. Did any reduction carry through to complete-request latency? For concurrency, did throughput improve within your latency budget? For precision or output caps, did the amount or correctness of generated SQL change?
''',
        'tokens': r'''### Check output work

Compare output tokens by phase and case before attributing latency differences to kernels or precision. Similar token counts do not establish identical SQL or computation.
''',
        'figure': r'''### Plot this session

The figure uses only successful measured rows selected in this session. Missing rows are not filled from another run. Plan produces no measurement figure.
''',
        'cleanup': r'''## 6. Cleanup

Save the notebook to retain genuine Live outputs. Native servers stop after each row. Stop the owned NIM container before switching models or pausing. Cleanup is safe to repeat and does not stop external servers.
''',
        'references': r'''## Reference: previously measured results

The previous Dell measurements remain unchanged in [the results guide](../tutorial/RESULTS.md) and [the original executed notebook](history/dell-2026-11-07/local%20AI%20optimization%20-%20Qwen.ipynb). Select `MODE="replay"` explicitly to rebuild their tables. They are not measurements from this Live session.

[Execution and troubleshooting](../tutorial/RUN_ALL.md) · [SQL review packet](../tutorial/SQL_REVIEW.md)
''',
    },
    '02_nemotron_optimization.ipynb': {
        'intro': r'''# Nemotron: context, caching, concurrency and output budgets

We keep FP16 weights and FP32 Mamba state, then test schema context, supported prefix caching, concurrent requests, scheduler capacity, reasoning and output limits.

The workload is structured clinical-trial questions on synthetic data with no personally identifiable information (PII). We compare response time with executed SQL and reference-result matches.

Configure your GPU environment below, then run the cells in order to collect new measurements. Each experiment saves its results separately. Completed outputs are visible when a live execution has been saved. For the previous Dell measurements, use the reference links at the end.

Follow [SETUP.md](../SETUP.md) and [the execution guide](../tutorial/RUN_ALL.md) first. The target is **Nemotron 3 Nano 4B**, `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, on a **Linux NVIDIA GPU**. Model loading and server startup happen before request timing.
''',
        'configuration': r'''## 1. Imports and editable configuration

`MODE` controls execution: **live** collects new inference; **plan** previews commands; **replay** explicitly reads archived results. It overrides an older local JSON mode. CPU validation supplies `LOCAL_AI_LAB_MODE=plan` or `replay` before executing these cells.

Keep machine paths, exact checkpoint revisions and deployment settings in the per-notebook local JSON. Downloads are opt-in. Remove the obsolete `execute` flag; mode is the execution authority.
''',
        'preflight': r'''## 2. Preflight checks

Check the GPU, selected runtime, checkpoint paths/revisions and port 8000 before loading a model. NIM also checks Docker access, an already available image, GPU access from that image and your selected profile. Missing prerequisites stop with instructions; no package installation or image pull is automatic.
''',
        'prepare': r'''## 3. Model preparation and compatibility smoke

Preparation validates local checkpoint identity and precision. It downloads weights only when `download_models=true`. NIM verifies the selected profile against discovery for this deployment.

Qwen's vLLM study requires a successful AWQ SQL smoke before all four variants. Nemotron core needs no AWQ checkpoint: its optional stretch remains separate. NIM smoke tests both serving and SQL execution; a failed required smoke blocks dependent experiments.
''',
        'method': r'''## 4. Individual experiments

Start with the FP16 baseline. Each intervention changes one control: supported prefix caching; retrieved schema; client concurrency at fixed scheduler capacity; scheduler capacity at four clients; reasoning; or output cap. Concurrency and achieved batching are different quantities.

Enable `prefix_supported` only after verifying support for the installed hybrid backend. Unsupported cache rows stay skipped. AWQ is optional stretch work requiring a validated checkpoint and never gates the core experiments.
''',
        'experiment-fp16_baseline': r'''### fp16 baseline

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'mechanism': r'''### Inspect the serving call and answer

The experiment above calls the model through the OpenAI-compatible `/v1/chat/completions` API. Inspect the launch command, request, generated plan/SQL and execution checks below. The request is constructed from the experiment prompt, schema profile and generation settings. In Plan, these details preview the configured call. The response shows the generated answer and its reference-result check.

[Request transport](../src/clinical_trial_mvp/serving/client.py) · [plan and SQL execution](../src/clinical_trial_mvp/serving/agent.py)
''',
        'experiment-prefix_cache': r'''### prefix cache

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-schema_retrieved': r'''### schema retrieved

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-concurrency_2': r'''### concurrency 2

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-concurrency_4': r'''### concurrency 4

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-batch_capacity_1': r'''### batch capacity 1

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-reasoning_on': r'''### reasoning on

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-output_256': r'''### output 256

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-output_2048': r'''### output 2048

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-awq_stretch': r'''### awq stretch

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'comparison': r'''## 5. Current-session comparison and interpretation

Read result matches alongside latency. TTFT measures first generated content; request completion covers generation and transport; application latency includes planning, validation, execution and repair. Missing memory telemetry stays unavailable. Failed rows remain failures.

Compare cold and warm TTFT. Did any reduction carry through to complete-request latency? For concurrency, did throughput improve within your latency budget? For precision or output caps, did the amount or correctness of generated SQL change?
''',
        'figure': r'''### Plot this session

The figure uses only successful measured rows selected in this session. Missing rows are not filled from another run. Plan produces no measurement figure.
''',
        'cleanup': r'''## 6. Cleanup

Save the notebook to retain genuine Live outputs. Native servers stop after each row. Stop the owned NIM container before switching models or pausing. Cleanup is safe to repeat and does not stop external servers.
''',
        'references': r'''## Reference: previously measured results

The previous Dell measurements remain unchanged in [the results guide](../tutorial/RESULTS.md) and [the original executed notebook](history/dell-2026-11-07/local%20AI%20optimization%20-%20Nemotron.ipynb). Select `MODE="replay"` explicitly to rebuild their tables. They are not measurements from this Live session.

[Execution and troubleshooting](../tutorial/RUN_ALL.md) · [SQL review packet](../tutorial/SQL_REVIEW.md)
''',
    },
    '03_qwen_nim.ipynb': {
        'intro': r'''# Qwen3-4B through NVIDIA NIM

We deploy a vLLM-backed NIM container, check the application response contract, then compare schema context and one, two and four concurrent requests. A NIM profile selects a deployment configuration for the checkpoint and GPU.

The workload is structured clinical-trial questions on synthetic data with no personally identifiable information (PII). We compare response time with executed SQL and reference-result matches.

Configure your GPU environment below, then run the cells in order to collect new measurements. Each experiment saves its results separately. Completed outputs are visible when a live execution has been saved. For the previous Dell measurements, use the reference links at the end.

Follow [SETUP.md](../SETUP.md) and [the execution guide](../tutorial/RUN_ALL.md) first. The target is **Qwen3-4B**, `Qwen/Qwen3-4B`, on a **Linux NVIDIA GPU**. Model loading and server startup happen before request timing.
''',
        'configuration': r'''## 1. Imports and editable configuration

`MODE` controls execution: **live** collects new inference; **plan** previews commands; **replay** explicitly reads archived results. It overrides an older local JSON mode. CPU validation supplies `LOCAL_AI_LAB_MODE=plan` or `replay` before executing these cells.

Keep machine paths, exact checkpoint revisions and deployment settings in the per-notebook local JSON. Downloads are opt-in. Remove the obsolete `execute` flag; mode is the execution authority.
''',
        'preflight': r'''## 2. Preflight checks

Check the GPU, selected runtime, checkpoint paths/revisions and port 8000 before loading a model. NIM also checks Docker access, an already available image, GPU access from that image and your selected profile. Missing prerequisites stop with instructions; no package installation or image pull is automatic.
''',
        'prepare': r'''## 3. Model preparation and compatibility smoke

Preparation validates local checkpoint identity and precision. It downloads weights only when `download_models=true`. NIM verifies the selected profile against discovery for this deployment.

Qwen's vLLM study requires a successful AWQ SQL smoke before all four variants. Nemotron core needs no AWQ checkpoint: its optional stretch remains separate. NIM smoke tests both serving and SQL execution; a failed required smoke blocks dependent experiments.
''',
        'method': r'''## 4. Individual experiments

Keep the NIM deployment fixed while changing full/retrieved schema or client concurrency. TP1/PP1 uses one GPU and one pipeline stage. A runnable profile still needs successful startup and a SQL smoke.

Check response structure as well as HTTP success. Qwen enables schema enforcement by default; Nemotron's setting is explicit in configuration. Compare the request contract before interpreting differences from native vLLM. The owned container stays running between these cells until cleanup.
''',
        'experiment-baseline': r'''### baseline

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'mechanism': r'''### Inspect the serving call and answer

The experiment above calls the model through the OpenAI-compatible `/v1/chat/completions` API. Inspect the launch command, request, generated plan/SQL and execution checks below. The request is constructed from the experiment prompt, schema profile and generation settings. In Plan, these details preview the configured call. The response shows the generated answer and its reference-result check.

[Request transport](../src/clinical_trial_mvp/serving/client.py) · [plan and SQL execution](../src/clinical_trial_mvp/serving/agent.py)
''',
        'experiment-schema_retrieved': r'''### schema retrieved

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-concurrency_2': r'''### concurrency 2

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-concurrency_4': r'''### concurrency 4

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'comparison': r'''## 5. Current-session comparison and interpretation

Read result matches alongside latency. TTFT measures first generated content; request completion covers generation and transport; application latency includes planning, validation, execution and repair. Missing memory telemetry stays unavailable. Failed rows remain failures.

Compare cold and warm TTFT. Did any reduction carry through to complete-request latency? For concurrency, did throughput improve within your latency budget? For precision or output caps, did the amount or correctness of generated SQL change?
''',
        'figure': r'''### Plot this session

The figure uses only successful measured rows selected in this session. Missing rows are not filled from another run. Plan produces no measurement figure.
''',
        'cleanup': r'''## 6. Cleanup

Save the notebook to retain genuine Live outputs. Native servers stop after each row. Stop the owned NIM container before switching models or pausing. Cleanup is safe to repeat and does not stop external servers.
''',
        'references': r'''## Reference: previously measured results

The previous Dell measurements remain unchanged in [the results guide](../tutorial/RESULTS.md) and [the original executed notebook](history/dell-2026-11-07/local%20AI%20optimization%20-%20NIM%20Qwen.ipynb). Select `MODE="replay"` explicitly to rebuild their tables. They are not measurements from this Live session.

[Execution and troubleshooting](../tutorial/RUN_ALL.md) · [SQL review packet](../tutorial/SQL_REVIEW.md)
''',
    },
    '04_nemotron_nim.ipynb': {
        'intro': r'''# Nemotron 3 Nano 4B through NVIDIA NIM

We deploy a vLLM-backed NIM container, check the application response contract, then compare schema context and one, two and four concurrent requests. A NIM profile selects a deployment configuration for the checkpoint and GPU.

The workload is structured clinical-trial questions on synthetic data with no personally identifiable information (PII). We compare response time with executed SQL and reference-result matches.

Configure your GPU environment below, then run the cells in order to collect new measurements. Each experiment saves its results separately. Completed outputs are visible when a live execution has been saved. For the previous Dell measurements, use the reference links at the end.

Follow [SETUP.md](../SETUP.md) and [the execution guide](../tutorial/RUN_ALL.md) first. The target is **Nemotron 3 Nano 4B**, `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, on a **Linux NVIDIA GPU**. Model loading and server startup happen before request timing.
''',
        'configuration': r'''## 1. Imports and editable configuration

`MODE` controls execution: **live** collects new inference; **plan** previews commands; **replay** explicitly reads archived results. It overrides an older local JSON mode. CPU validation supplies `LOCAL_AI_LAB_MODE=plan` or `replay` before executing these cells.

Keep machine paths, exact checkpoint revisions and deployment settings in the per-notebook local JSON. Downloads are opt-in. Remove the obsolete `execute` flag; mode is the execution authority.
''',
        'preflight': r'''## 2. Preflight checks

Check the GPU, selected runtime, checkpoint paths/revisions and port 8000 before loading a model. NIM also checks Docker access, an already available image, GPU access from that image and your selected profile. Missing prerequisites stop with instructions; no package installation or image pull is automatic.
''',
        'prepare': r'''## 3. Model preparation and compatibility smoke

Preparation validates local checkpoint identity and precision. It downloads weights only when `download_models=true`. NIM verifies the selected profile against discovery for this deployment.

Qwen's vLLM study requires a successful AWQ SQL smoke before all four variants. Nemotron core needs no AWQ checkpoint: its optional stretch remains separate. NIM smoke tests both serving and SQL execution; a failed required smoke blocks dependent experiments.
''',
        'method': r'''## 4. Individual experiments

Keep the NIM deployment fixed while changing full/retrieved schema or client concurrency. TP1/PP1 uses one GPU and one pipeline stage. A runnable profile still needs successful startup and a SQL smoke.

Check response structure as well as HTTP success. Qwen enables schema enforcement by default; Nemotron's setting is explicit in configuration. Compare the request contract before interpreting differences from native vLLM. The owned container stays running between these cells until cleanup.
''',
        'experiment-baseline': r'''### baseline

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'mechanism': r'''### Inspect the serving call and answer

The experiment above calls the model through the OpenAI-compatible `/v1/chat/completions` API. Inspect the launch command, request, generated plan/SQL and execution checks below. The request is constructed from the experiment prompt, schema profile and generation settings. In Plan, these details preview the configured call. The response shows the generated answer and its reference-result check.

[Request transport](../src/clinical_trial_mvp/serving/client.py) · [plan and SQL execution](../src/clinical_trial_mvp/serving/agent.py)
''',
        'experiment-schema_retrieved': r'''### schema retrieved

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-concurrency_2': r'''### concurrency 2

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'experiment-concurrency_4': r'''### concurrency 4

Predict the effect on latency, output work and structured-answer correctness before executing this cell.
''',
        'comparison': r'''## 5. Current-session comparison and interpretation

Read result matches alongside latency. TTFT measures first generated content; request completion covers generation and transport; application latency includes planning, validation, execution and repair. Missing memory telemetry stays unavailable. Failed rows remain failures.

Compare cold and warm TTFT. Did any reduction carry through to complete-request latency? For concurrency, did throughput improve within your latency budget? For precision or output caps, did the amount or correctness of generated SQL change?
''',
        'figure': r'''### Plot this session

The figure uses only successful measured rows selected in this session. Missing rows are not filled from another run. Plan produces no measurement figure.
''',
        'cleanup': r'''## 6. Cleanup

Save the notebook to retain genuine Live outputs. Native servers stop after each row. Stop the owned NIM container before switching models or pausing. Cleanup is safe to repeat and does not stop external servers.
''',
        'references': r'''## Reference: previously measured results

The previous Dell measurements remain unchanged in [the results guide](../tutorial/RESULTS.md) and [the original executed notebook](history/dell-2026-11-07/local%20AI%20optimization%20-%20NIM%20Nemotron.ipynb). Select `MODE="replay"` explicitly to rebuild their tables. They are not measurements from this Live session.

[Execution and troubleshooting](../tutorial/RUN_ALL.md) · [SQL review packet](../tutorial/SQL_REVIEW.md)
''',
    },
    '05_edge_portability_executorch.ipynb': {
        'edge-intro': r'''# Qwen: export a model and run a structured question on CPU

The serving chapters used an HTTP API. Here we package the same **Qwen3-4B** base revision as an **ExecuTorch XNNPACK** program and invoke its native CPU runner. The recipe uses **INT4 linear weights with dynamic INT8 activations and INT8 embeddings**, which differs from AWQ W4A16.

We test one direct clinical SQL question with a short schema. This is a portability smoke, separate from the full planning/repair workload and GPU serving benchmarks.

Configure the edge environment below, then run the cells in order. Reuse a validated export explicitly or perform conversion and export. **Generation always runs afresh in Live mode.** Install the [notebook environment](../SETUP.md). See [the edge setup guide](../tutorial/EXECUTORCH_MAC.md).
''',
        'edge-config': r'''## 1. Imports and editable configuration

`MODE="live"` performs the stages below. Choose **plan** to inspect without edge execution or **replay** to read the explicitly selected prior record. The notebook mode overrides an old local JSON mode. No stage installs packages or downloads weights implicitly.
''',
        'edge-preflight': r'''## 2. Preflight

Check the configured edge Python and installed runtime before any model work. Qwen requires a pinned source checkpoint and either a validated reusable export or explicit conversion/export. Nemotron needs only its pinned configuration. Missing prerequisites stop with instructions.
''',
        'edge-export': r'''## 3. Prepare and validate the CPU program

Stages run as separate processes so large reference and converted models do not stay resident together. `download_models=false` uses staged source weights. Set it to true only to request a download.

With `reuse_export=true`, conversion, reference, validation and export cells verify the recorded artifact hashes and report **reused_artifact**. They do not claim a new export. To build afresh, set `reuse_export=false` and select a new `artifact_dir` in the local config. Conversion uses the official Qwen3 weight mapping; FP32 logit checks must pass before quantized export.

The native recipe and generation call are in [run_qwen_edge.py](../scripts/tutorial/run_qwen_edge.py).
''',
        'edge-inference': r'''## 4. Invoke the model and check its answer

The next cell starts a **new native generation process** with the verified `.pte` program and tokenizer. The runner calls `TextLLMRunner.generate()` with temperature zero, a short clinical schema and an output cap of 192 tokens. It uses no HTTP serving API and loads no saved answer. Model-load time is separate from inference timing.

The following cell executes the newly generated SQL in DuckDB and compares its rows with the reference. SQL scoring occurs after native timing.
''',
        'edge-metrics': r'''## 5. Read this session's measurements

Compare model loading, first-token time, inference latency, artifact footprint and process RSS. These are one short-question sample, not serving percentiles. RSS is CPU process memory, not GPU memory. Inspect generated SQL and its reference-result check alongside timing.
''',
        'edge-cleanup': r'''## Cleanup

The edge stages execute in foreground child processes and retain no serving process. Save the notebook to retain genuine current outputs. Failures stay in the session evidence.
''',
        'edge-reference': r'''## Reference: previously measured results

The [edge guide](../tutorial/EXECUTORCH_MAC.md) preserves the previous Qwen native run and Nemotron compatibility check. Choose `MODE="replay"` explicitly to inspect those records. The saved artifacts and historical records remain unchanged; they are not results from a new Live execution.
''',
    },
    '06_nemotron_executorch.ipynb': {
        'probe-intro': r'''# Nemotron: check the CPU export path before loading weights

Can the same **Nemotron 3 Nano 4B** hybrid checkpoint used in the serving chapters use the Qwen CPU deployment path? First test whether the installed ExecuTorch export recipe accepts its architecture.

Nemotron-H includes Mamba recurrent state alongside attention. Qwen's adapter does not supply that state mapping. The following cells execute a **fresh configuration-only probe** of the installed exporter. Unsupported configuration is a valid finding; this chapter does not run native inference or report generation timings.

Install the [notebook environment](../SETUP.md). Follow [the edge setup guide](../tutorial/EXECUTORCH_MAC.md), configure your edge Python, then run the cells in order. Downloads are opt-in; this check needs only the pinned `config.json`, not model weights.
''',
        'edge-config': r'''## 1. Imports and editable configuration

`MODE="live"` performs the stages below. Choose **plan** to inspect without edge execution or **replay** to read the explicitly selected prior record. The notebook mode overrides an old local JSON mode. No stage installs packages or downloads weights implicitly.
''',
        'edge-preflight': r'''## 2. Preflight

Check the configured edge Python and installed runtime before any model work. Qwen requires a pinned source checkpoint and either a validated reusable export or explicit conversion/export. Nemotron needs only its pinned configuration. Missing prerequisites stop with instructions.
''',
        'probe-mechanism': r'''## 3. Execute the compatibility probe

The next cell invokes [probe_nemotron_edge.py](../scripts/tutorial/probe_nemotron_edge.py) using your installed edge environment. It tests `nemotron_h` and `nemotron_3_nano_4b` against the installed model configuration enum and saves a unique result.

It reports exporter version, accepted model types and each attempt. If a recipe is accepted, conversion and native inference still require validation; acceptance alone supplies no performance result.
''',
        'probe-interpretation': r'''## 4. Interpret the finding

If the result is **blocked_compatibility**, the installed standard recipe lacks this model configuration. It does not establish that a custom port is impossible. A working port would need validated Mamba state handling, weight conversion, graph capture, supported operators, PTE loading and native generation before SQL or timing could be evaluated.
''',
        'edge-cleanup': r'''## Cleanup

The edge stages execute in foreground child processes and retain no serving process. Save the notebook to retain genuine current outputs. Failures stay in the session evidence.
''',
        'edge-reference': r'''## Reference: previously measured results

The [edge guide](../tutorial/EXECUTORCH_MAC.md) preserves the previous Qwen native run and Nemotron compatibility check. Choose `MODE="replay"` explicitly to inspect those records. The saved artifacts and historical records remain unchanged; they are not results from a new Live execution.
''',
    },
}

# Shared controls for the two native vLLM studies. NIM/edge chapters keep their scope.
OPTIMIZATION_CONTROLS = '''| Control | What it changes |
|---|---|
| Schema context | Compare the full schema with a smaller retrieved selection. |
| Prefix caching | Reuse processing of an identical shared schema prefix; report cold and warm requests separately. |
| Concurrency | Change how many client requests overlap: 1, 2 or 4. |
| Scheduling capacity | Allow 1, 2 or 4 sequences in the server scheduler, keeping four client requests concurrent; actual batch occupancy is not measured. |
| Reasoning | Compare reasoning disabled and enabled; reasoning tokens count within the output budget. |
| Output limit | Cap generated tokens at 256, 1,024 or 2,048; the context-window limit remains 8,192. |
| Precision | Compare FP16 with validated AWQ 4-bit weights and FP16 activations (W4A16). |'''

OPTIMIZATION_EXPLANATIONS = {
    'fp16_cache_off': 'Establish the Qwen FP16 baseline with prefix caching disabled, one client, scheduling capacity 4 and an output limit of 1,024 tokens.',
    'fp16_baseline': 'Establish the Nemotron FP16 baseline with prefix caching disabled, one client, scheduling capacity 4 and an output limit of 1,024 tokens; Mamba state stays FP32.',
    'fp16_cache_on': 'Change only prefix caching from fp16_cache_off; compare cold and warm requests using an identical shared schema prefix.',
    'awq_w4a16_cache_off': 'Change only precision from fp16_cache_off to AWQ W4A16, after the checkpoint passes its compatibility smoke.',
    'awq_w4a16_cache_on': 'Change only prefix caching from awq_w4a16_cache_off; report cold and warm requests separately.',
    'prefix_cache': 'Change only prefix caching from fp16_baseline, after verifying hybrid-backend support; report cold and warm requests separately.',
    'schema_retrieved': 'Replace the full schema with retrieved schema context while keeping the FP16 baseline settings fixed.',
    'concurrency_2': 'Send two overlapping client requests while keeping server scheduling capacity at 4; compare with the one-client FP16 baseline.',
    'concurrency_4': 'Send four overlapping client requests with scheduling capacity 4; this also supplies the capacity-4 comparison for the next two experiments.',
    'batch_capacity_1': 'Keep four clients concurrent but limit scheduling capacity to 1; compare with batch_capacity_2 and concurrency_4 (capacity 4).',
    'batch_capacity_2': 'Keep four clients concurrent but limit scheduling capacity to 2; compare capacity 1 → 2 → 4, without assuming achieved batch size.',
    'reasoning_on': 'Enable reasoning while holding the FP16 baseline settings and 1,024-token output limit fixed; TTFT includes reasoning and truncation remains a failure.',
    'output_256': 'Reduce only the FP16 baseline output limit from 1,024 to 256 generated tokens; keep the context-window limit at 8,192.',
    'output_2048': 'Increase only the FP16 baseline output limit from 1,024 to 2,048 generated tokens; keep the context-window limit at 8,192.',
    'awq_stretch': 'Try AWQ W4A16 against the FP16 baseline only after validating the exact checkpoint pair and passing an AWQ SQL smoke; Mamba state stays FP32.',
}

for _filename in ('01_qwen_cache_quantization.ipynb', '02_nemotron_optimization.ipynb'):
    _chapter = CHAPTER_PROSE[_filename]
    _chapter['method'] += '\n' + OPTIMIZATION_CONTROLS + '\n'
    for _name, _explanation in OPTIMIZATION_EXPLANATIONS.items():
        if _name in ('fp16_baseline', 'prefix_cache', 'awq_stretch') and _filename.startswith('01'):
            continue
        if _name in ('fp16_cache_off', 'fp16_cache_on', 'awq_w4a16_cache_off', 'awq_w4a16_cache_on') and _filename.startswith('02'):
            continue
        _chapter['experiment-' + _name] = '### ' + _name.replace('_', ' ') + '\n\n' + _explanation + '\n'

CHAPTER_PROSE['01_qwen_cache_quantization.ipynb']['intro'] = CHAPTER_PROSE['01_qwen_cache_quantization.ipynb']['intro'].replace(
    '# Qwen: answer faster with caching and four-bit weights', '# Qwen: context, caching, concurrency and precision').replace(
    'AWQ compresses weights while keeping activations at FP16.',
    'AWQ compresses weights while keeping activations at FP16. We then test schema context, concurrency, scheduling capacity, reasoning and output limits against the FP16 cache-off baseline.')
CHAPTER_PROSE['01_qwen_cache_quantization.ipynb']['method'] = CHAPTER_PROSE['01_qwen_cache_quantization.ipynb']['method'].replace(
    'Hold the schema, four questions, seed 42, temperature 0, cap 1024 and context limit 8192 fixed.',
    'For the first four precision/cache variants, hold the schema, four questions, seed 42, temperature 0, output cap 1,024, scheduling capacity 4 and context limit 8,192 fixed. Subsequent experiments each change one control from fp16_cache_off; capacity experiments keep concurrency at four clients. Compare capacity 1/2/4 using batch_capacity_1, batch_capacity_2 and concurrency_4.')
CHAPTER_PROSE['02_nemotron_optimization.ipynb']['prepare'] += '''
For the optional AWQ attempt, edit `tutorial/config/vllm-nemotron.local.json` before starting the notebook: set `run_awq=true`, `awq_source` to a compatible checkpoint repository, `awq_revision` to its exact 40-character commit SHA, `awq_derivation` to the documented link to your exact base revision, and `awq_snapshot` to its local directory. Set `download_models=true` only if you want the pinned checkpoint downloaded. The final AWQ cell validates architecture, tokenizer and 4-bit AWQ metadata, starts the server and runs a SQL smoke before benchmarking. Without a compatible source, leave `run_awq=false`; the core experiments still run. Other four-bit formats are not interchangeable with AWQ.
'''
