# MVP4 baseline: cumulative work through MVP3

MVP4 is a new tagged iteration of the cumulative project. Its application code and workload
come from the MVP3 working tree, which carries the earlier MVP1/MVP2 foundation. The tag
`mvp4` identifies the initial assembled MVP4 repository. It does not designate a fresh GPU
benchmark or imply that earlier results were rerun.

| Prior work | What MVP4 inherits | Lesson carried forward |
|---|---|---|
| Original MVP foundation | Ten-table synthetic DuckDB workload, SQL/privacy guards, reference queries and policy cases | Generated/executable SQL is not necessarily a correct clinical answer |
| MVP1 | Full/retrieved schemas, frozen retrieval caches, Nemotron/Qwen Q4 GGUF comparisons, semantic review | Context reduction can improve latency while losing clinical definitions or correctness |
| MVP2 | Same-GGUF llama.cpp/Ollama serving comparison and reproducibility controls | Hold model artifacts, prompts and serving settings fixed before attributing speed changes |
| MVP3 | HF checkpoint vLLM/SGLang transport, concurrency harness, provenance, SQL quality evaluation | Runtime measurements require matched model/data/environment evidence |
| ExecuTorch portability case retained in MVP3 | SmolLM2-135M HF → FP32 PTE, native XNNPACK CPU run | Portability success on a different model/workload is not a clinical SQL speed/quality comparison |

## Preserved baseline outcomes

MVP1's complete 30-case Q4 comparison found Nemotron full schema matched **10/23** reference
results, with **9/23** reviewed semantic answers, at **79.795 s** median question latency.
Retrieved schema matched **8/23** (**7/23** reviewed) at **29.500 s**. Qwen full/retrieved
matched **5/23** and **6/23** (both **5/23** reviewed) at **62.860 s** and **21.146 s**.
Retrieval reduced median prompt tokens by about 77%, but did not consistently improve
correctness. These CPU GGUF measurements used precomputed retrieval caches, and were
single engineering runs per configuration, not a held-out quality estimate.
See [the original Q4 report](baseline/mvp1/experiment-tracker/MVP1_Q4_30_CASE_REPORT.md)
and [embedding comparison](baseline/mvp1/experiment-tracker/EMBEDDING_MODEL_COMPARISON.md).

MVP2 demonstrated the same Nemotron Q4_K_M GGUF through native llama.cpp and Ollama on
M4 Pro. Its small paired serving screen produced matching outputs for the paired prompts;
it did not establish a general speed ranking or a new clinical quality score.
See [the original milestone](baseline/mvp2/experiment-tracker/mvp2/serving/MILESTONE_REPORT.md).

MVP3's imported Dell notebook summaries retain vLLM/SGLang concurrency 1/2/4 results.
They report 3/4 SQL executions and 2/4 reference matches in their displayed final tables.
They lack complete raw Dell request records and launch/model/database fingerprints;
they remain explicitly labeled notebook summaries with unverified pairing. Source notebook
archives and the imported summary JSONs are preserved under `baseline/mvp3/`.
See [the original results/portability report](baseline/mvp3/tutorial/RESULTS_PORTABILITY.md).

The separate ExecuTorch case used SmolLM2-135M on M4 Pro CPU, with **0.199 s** native
single-request inference, **0.076 s** model load and about **1018 MiB** sampled process
RSS. It did not measure clinical SQL correctness or GPU VRAM. See
[the retained reproduction record](tutorial/EXECUTORCH_MAC.md).

## New MVP4 work

The new optimization experiments build on those controls and lessons. Qwen retains the
FP16/AWQ cache experiment; Nemotron prioritizes its FP16 baseline, caching investigation,
context reduction, concurrency/batching and reasoning/output budgets, with compatible
AWQ as stretch work. The shared workload and safety/scoring foundation are preserved.
Core Nemotron work does not depend on an AWQ artifact.

GPU measurements for these MVP4 experiments remain pending. Do not combine earlier Mac
GGUF, imported Dell HF summaries, and CPU PTE measurements into a paired speed comparison.
Source hashes, ancestor Git heads and working-tree status are recorded in
`SOURCE_LINEAGE.json`. Original historical reports retain their original links and dates;
some links require the original iteration because only selected baseline evidence is packaged.
