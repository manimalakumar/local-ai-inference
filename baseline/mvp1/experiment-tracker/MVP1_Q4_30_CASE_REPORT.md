# MVP1 Q4: complete 30-question comparison

Date: 2026-09-28. This replaces the one-question pilot as the measured Q4 baseline. Each of four configurations ran the same 30 cases once, serially, on the same synthetic DuckDB database and evaluation dataset. Model settings were temperature 0, seed 42, 8,192-token context, 1,024 output-token limit, CPU inference (`gpu_layers=0`), reasoning disabled, and at most one SQL repair. The generation models were Nemotron 3 Nano 4B Q4_K_M and Qwen3 4B Q4_K_M. Retrieved-schema runs used the frozen Nemotron 3 Embed 1B BF16 cache; full-schema runs used the complete schema guidance. All four manifests have the same case-ID, dataset, database, scorer, and code-commit hashes. Each run has 30 records.

The 30 cases comprise 23 SQL questions with executable reference queries, three SQL-eligible questions without reference queries, and four policy or escalation cases. **Final result equivalence on the 23 reference cases is the primary automatic quality metric.** It compares returned values, rather than SQL text. It can still reward a semantically wrong query that happens to return the same rows on this one database; semantic review and counterfactual fixtures remain necessary. A single run per configuration does not measure run-to-run variance.

## Main results

| Model | Schema context | First-pass correct / 23 | Final equivalent / 23 | SQL executed / 26 | Median latency / question | Median prompt tokens / invocation | Median first token / invocation | Peak process RAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nemotron 3 Nano 4B Q4 | Full | 9 | **10** | 16 | 79.795 s | 2,477 | 32.371 s | 4.34 GB |
| Nemotron 3 Nano 4B Q4 | Retrieved | 6 | 8 | 12 | 29.500 s | 555.5 | 7.761 s | 4.31 GB |
| Qwen3 4B Q4 | Full | 5 | 5 | 9 | 62.860 s | 2,402 | 24.740 s | 6.59 GB |
| Qwen3 4B Q4 | Retrieved | 5 | 6 | 8 | 21.146 s | 551 | 4.365 s | 6.56 GB |

The [30-question case review](MVP1_Q4_CASE_REVIEW.md) records semantic correctness, execution, and latency for every model and question; its [structured audit](MVP1_Q4_CASE_AUDIT.json) records the model-specific reason and retrieved facts. After SQL and policy-response review, the semantically correct counts among the **23 reference SQL cases** are Nemotron full **9/23**, Nemotron retrieved **7/23**, Qwen full **5/23**, and Qwen retrieved **5/23**. In addition, all four runs correctly handled the four deterministic policy cases; all four partially handled `eval_014` by asking for a performance metric and avoiding a causal explanation without checking Site 17's existence. These are inspected judgments on this case set, not proof over all possible databases.

Three baseline row matches are downgraded by semantic review: Nemotron retrieved `eval_020` filters out a nullable screening status, and Nemotron full plus Qwen retrieved `eval_023` omit the required non-null enrollment-date filter. The current synthetic data has no rows that expose those defects. The targeted counterfactual fixture covers five other cases; it confirms Nemotron full on `eval_002` and `eval_021`, and Nemotron retrieved on `eval_021`. It supplies no broad semantic guarantee.

Retrieval reduced median prompt tokens per model invocation by 77.6% for Nemotron and 77.1% for Qwen. Median end-to-end question latency fell by 63.0% and 66.4%, respectively. These are observed medians, not isolated retrieval overhead estimates. Nemotron's final row-equivalence count decreased by two with retrieval; Qwen's increased by one. The count changes hide different case outcomes: Nemotron full alone matched `eval_002`, `eval_023`, and `eval_030`, while retrieved alone matched `eval_026`. Qwen full alone matched `eval_022`, while retrieved alone matched `eval_023` and `eval_026`. The small paired set does not justify a general claim that retrieval improves or harms accuracy.

## Other tracked measures

| Measure | Nemotron full | Nemotron retrieved | Qwen full | Qwen retrieved |
| --- | ---: | ---: | ---: | ---: |
| SQL generated / 26 eligible | 25 | 25 | 22 | 25 |
| SQL safety pass / generated | 24/25 | 23/25 | 11/22 | 11/25 |
| Schema compliance / applicable safe SQL | 16/20 | 12/16 | 9/11 | 8/11 |
| Repair success / attempted | 1/12 | 2/15 | 0/15 | 1/18 |
| Average tool calls / question | 1.433 | 1.300 | 1.200 | 1.167 |
| Median generated tokens / second | 29.914 | 30.721 | 26.535 | 30.964 |
| Local resolution / 30 | 14 | 12 | 9 | 10 |
| Question timeouts | 0 | 0 | 2 | 0 |

SQL execution rates over the 26 SQL-eligible questions are **61.5%**, **46.2%**, **34.6%**, and **30.8%** in table order. Per-question execution status and latency are in the case review; the raw JSONL records include first and final SQL, errors, plans, and tool calls.

The two Qwen full-schema timeouts were `eval_029` and `eval_030`, at the configured 300-second per-question limit. All four runs recorded 30/30 correct initial tool-selection paths, 4/4 policy escalations, 30/30 repair-loop termination, and zero unnecessary repairs by the current deterministic checks. The clarification check passed 26/26 SQL-eligible labels, but only `eval_014` is positively labeled as requiring clarification; it is also the sole unsupported causal-claim rejection case (1/1). The unsafe-request case passed 1/1, and all ten separate SQL-write probes were blocked in each run. These small policy denominators do not establish broad safety or clarification accuracy.

The automated schema-linking score is 23/23 in all four runs. It measures whether the context contains tables and columns used in the reference query; it does **not** measure whether the model chose the right tables, columns, joins, or definitions in its generated SQL. Full-schema coverage is expected by construction. In the retrieved mode, the frozen cache had already been checked for 23/23 reference-schema coverage. High reference-schema coverage therefore does not explain the low row-equivalence scores by itself; the case audit records both generation failures and missing definition facts.

**Retrieved-schema misses:** The reference checker found zero missing reference tables or qualified columns across the 23 reference queries. It does not check unqualified columns or the seven cases without reference SQL. A separate audit of clinical definitions found ten missing definition facts across eight retrieved prompts: `eval_003` and `eval_004` omitted enrollment; `eval_006` and `eval_010` omitted enrollment and rate; `eval_007`, `eval_008`, `eval_011`, and `eval_029` omitted rate. These are existing corpus definitions whose question triggers fired but whose facts fell outside the retrieved top 12. The corpus also lacks a specific definition of the previous calendar month and of an unusual site signal; those are corpus gaps, not top-12 retrieval misses. The same frozen cache served both retrieved-schema SQL models, so their retrieval misses are identical. The audit does not establish that any one omission caused a particular generation failure.

**Model roles:** Schema retrieval here used the local `nvidia/Nemotron-3-Embed-1B-BF16` embedding model (downloaded revision `c0c9fea93ea424587517f2c59e20db9f1d6bf615`) via `sentence-transformers 6.1.0`. Schema facts were encoded with a `passage:` prefix and questions with a `query:` prefix, then normalized for similarity ranking. Retrieval used a reviewed schema-fact corpus, top 12 similarity hits, at most four seed tables, and relationship-path enrichment. The [model registry](MVP1_MODEL_MATRIX.json) records its source. Nemotron 3 Nano 4B and Qwen3 4B generated SQL. This table compares the **SQL-generating models and full versus retrieved schema paths**. It does not compare Qwen embeddings with BGE embeddings; neither was used to produce these retrieved schemas.

**Generation settings and hardware:** Both Q4_K_M models ran through `llama-cpp-python 0.3.35` on a MacBook Pro with Apple M4 Pro (14 CPU cores, 20 GPU cores), 48 GB unified memory, and macOS 26.7. Inference was CPU-only (`gpu_layers=0`); the GPU was present but unused for generation. Runs were serial. Context size was 8,192 tokens, maximum output 1,024 tokens, temperature 0, seed 42, reasoning disabled, maximum one repair, and 300-second per-question timeout. Generation model file SHA-256 values are in each run manifest. The frozen retrieval cache SHA-256 was `ab6995fe42b357898873b4f16ad790960a85dac815ea94a37651023401cab75b`; corpus SHA-256 was `5962f5dc922705d74b3ab18be77f326693f45d068cf9f0e6461cd1144a02eab1`. The evaluated code commit was `b356e1b` in all four runs.

The retrieved prompts were built **before** SQL evaluation in a separate embedding-model process and read from a frozen cache during generation. Thus the displayed end-to-end question latency excludes embedding-model startup and query encoding. Cache selection/assembly took a median 5.0 ms per question during precomputation, but this is not a complete interactive retrieval latency. The prompt-token medians are per model invocation, while the latency medians are per question, including any repair.

Median local privacy-check time per question was 0.162, 0.208, 0.139, and 0.294 milliseconds in table order. This covers preflight policy and local SQL-safety validation only; it excludes MCP display pseudonymization and therefore is not a complete privacy-overhead figure. Peak RAM is the evaluation process high-water mark. macOS unified memory prevents a separate comparable VRAM figure here. Cold-start measurements were 0.899, 2.823, 2.349, and 2.594 seconds in table order; cache state and run order make these unsuitable for a model ranking.

## Reproducibility and next iteration

Each run retains a detailed JSONL trace, summary, checksummed manifest, and source-code ZIP in `results/`:

| Configuration | Run ID |
| --- | --- |
| Nemotron full | [`eval_20260928T230459_358876Z`](results/eval_20260928T230459_358876Z_summary.json) |
| Nemotron retrieved | [`eval_20260928T224053_842728Z`](results/eval_20260928T224053_842728Z_summary.json) |
| Qwen full | [`eval_20260928T234555_701456Z`](results/eval_20260928T234555_701456Z_summary.json) |
| Qwen retrieved | [`eval_20260928T225433_213501Z`](results/eval_20260928T225433_213501Z_summary.json) |

The targeted counterfactual reports are saved for [Nemotron full](results/eval_20260928T230459_358876Z_counterfactual_v1.json), [Nemotron retrieved](results/eval_20260928T224053_842728Z_counterfactual_v1.json), [Qwen full](results/eval_20260928T234555_701456Z_counterfactual_v1.json), and [Qwen retrieved](results/eval_20260928T225433_213501Z_counterfactual_v1.json), each with a checksum sidecar.

This is the first complete Q4 slice: two models × two schema modes, with reasoning off and one repair available. Higher precision, reasoning on, and zero-repair configurations remain separate iterations, to be run serially given the machine's memory pressure. The semantic review and five-case counterfactual checks are preserved separately from the immutable raw runs. A broader perturbation suite would test the remaining semantic assumptions and quantify how often row equivalence is accidental.
