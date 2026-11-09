"""Chapter 00 adapts the existing serving tutorial; it adds no benchmark/runtime."""
import hashlib


FILENAME = '00_local_serving.ipynb'


def notebook_document():
    cells = []
    def add(kind, text):
        source = text.strip() + '\n'
        cell = dict(cell_type=kind, id=hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                    metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    md = lambda text: add('markdown', text)
    code = lambda text: add('code', text)
    md('''# Serving foundations and recorded comparisons

*Local AI Inference Lab: Serving, Optimization and Application Quality*

**Learn:** follow a question through schema selection, model serving, plan/SQL validation,
read-only execution and result scoring; distinguish application quality from transport speed.
Understand the earlier llama.cpp/Ollama and vLLM/SGLang serving paths before optimizing them.

**Requirements:** the [CPU notebook environment](../SETUP.md), included frozen synthetic
database and archived evidence. No GPU, model download, Docker or credentials are needed.

**Expected outputs:** a verified read-only table inventory, full/retrieved prompt examples,
clearly labeled historical serving tables, and one recorded SQL example with evaluation.

This adapts the existing serving tutorial's workload/context/runtime sections. It reuses
the shared schema, retrieval cache, adapters and scorer; it does not import a duplicate app.
[Adaptation and data fingerprints](../tutorial/serving_foundation.json) identify the source.
**Historical/read-only:** this chapter does not provide fresh llama.cpp/Ollama/vLLM/SGLang
reproduction. It has no Live serving support. Fresh optimization runs are supported in chapters 01–04. Historical report values and rounded notebook summaries are distinct evidence.
''')
    code('''from pathlib import Path
import os
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import JSON, Markdown, display
from clinical_trial_mvp.tutorial.foundation import inventory, contexts, mac_history, gpu_history, history_table
print('Read-only foundation; mode:', os.environ.get('LOCAL_AI_LAB_MODE', 'replay'))
print('Historical evidence is labeled; no fresh model calls are made.')
''')
    md('''### Model and hardware map before reading the results

| Experiment | Model/artifact from archived evidence | Hardware | Evidence limitation |
|---|---|---|---|
| Context selection | Nemotron 3 Nano 4B and Qwen3 4B, Q4_K_M GGUF; llama-cpp-python 0.3.35, context 8192, seed 42, cap 1024; exact file hashes not included in this report | M4 Pro, CPU-only (`gpu_layers=0`) | Historical application measurements; retrieval was precomputed |
| llama.cpp/Ollama | Same Nemotron 3 Nano 4B Q4_K_M GGUF; SHA-256 `be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2` | M4 Pro, Metal offload | Three requests per prompt/runtime; direct serving screen |
| vLLM/SGLang | Exact model ID/revision/fingerprint unavailable; alias or directory name is insufficient; FP16 requested | Dell Precision 7740 / Quadro RTX 5000 16 GB | Rounded notebook summaries; incomplete provenance and unverified pairing |

Sources: [context report](../baseline/mvp1/experiment-tracker/MVP1_Q4_30_CASE_REPORT.md),
[Mac serving report](../baseline/mvp2/experiment-tracker/mvp2/serving/MILESTONE_REPORT.md),
[GPU evidence limitations](../baseline/mvp3/tutorial/RESULTS_PORTABILITY.md).
''')
    md('''## 1. Keep the workload and database fixed

**Question → full/retrieved schema text → model plan → SQL → validation → DuckDB → scorer.**

The ten-table database and CSVs are synthetic. The database fingerprint matches the saved
Dell optimization baseline. Do not regenerate it to replay results: new rows can change a
query's reference match. Human-reviewed semantic correctness is separate from matching values
on one database. The full evaluation set has 30 cases, including 23 reference-SQL questions.
The later compact experiments reuse four distinct questions with repeated serving requests.
''')
    code('''database = inventory(ROOT)
display(JSON(database))
''')
    md('''## 2. Full versus retrieved schema

Retrieval selects schema facts; the generation model still receives text. We reuse the frozen
retrieval cache, so this inspection loads no embedding model. Inspect the selected tables
and actual prompts, especially required joins, grouping, denominators and clinical definitions.
Character lengths below are **not token counts**. Retrieval precomputation is outside the
historical generation measurements; it is not a full interactive retrieval latency estimate.
''')
    code('''context = contexts(ROOT)
display(JSON({'note': context['note'], 'cases': [
    {key: value for key, value in row.items() if key not in ('full_prompt', 'retrieved_prompt')}
    for row in context['rows']]}))
display(JSON({'full_prompt': context['rows'][0]['full_prompt'],
              'retrieved_prompt': context['rows'][0]['retrieved_prompt']}))
''')
    md('''### Historical context findings

The earlier CPU/GGUF report evaluated 30 cases once per configuration, with 23 reference
queries. These are **historical Mac report values**, not the later four-case GPU workload.
Question/application latency includes planning and possible repair; it is not request TTFT.
''')
    code('''history = mac_history(ROOT)
display(JSON({'label': history['origin'], 'context_report': history['context']}))
''')
    md('''**Observation:** Nemotron's historical retrieved context reduced median question latency
from 79.795 to 29.500 s while reference matches changed 10/23 → 8/23. Qwen's matches changed
5/23 → 6/23. **Explanation:** less input work and less information can occur together.
**Limitation:** a single small synthetic dataset and precomputed retrieval do not establish
general accuracy or online retrieval overhead. **Decision:** keep required definitions and
review SQL semantics before accepting context savings.

[Original context report](../baseline/mvp1/experiment-tracker/MVP1_Q4_30_CASE_REPORT.md).
''')
    md('''## 3. Mac serving: llama.cpp and Ollama

Both paths used the same Nemotron Q4_K_M GGUF on M4 Pro. The aligned direct-serving screen
used the same prompt bytes, seed, cap and zero observed cached prompt tokens. Each prompt
had three measured requests per runtime. This compares serving packages/APIs around the
same GGUF, not unrelated models or inference algorithms.

The earlier CPU context report and Metal-offloaded serving screen are separate experiments.
Do not pool their times. Reuse [the existing adapters](../src/clinical_trial_mvp/tutorial/adapters.py)
and [common serving harness](../src/clinical_trial_mvp/serving/benchmark.py), rather than copying the app.
''')
    code("display(JSON({'label': history['origin'], 'serving_screen': history['serving'], 'limitations': history['limitations']}))")
    md('''**Observation:** the selected-prompt screen recorded TTFT p50 0.400/0.414 s and E2E
1.393/1.404 s for llama.cpp/Ollama. Paired output hashes matched in that screen.
**Explanation:** serving APIs/lifecycle can differ while loading the same model artifact.
**Limitation:** matching generated text is not proof of clinical SQL correctness; three
requests do not support a general speed ranking. **Decision:** choose an API/lifecycle for
the deployment need, then use the shared application-quality path to evaluate useful answers.

[Original serving report](../baseline/mvp2/experiment-tracker/mvp2/serving/MILESTONE_REPORT.md).
''')
    md('''## 4. Dell serving: vLLM and SGLang

GPU serving used frozen Hugging Face checkpoints, not the Mac GGUF. The earlier notebook
outputs cover concurrency 1/2/4. They are **rounded historical notebook summaries**, with
missing raw requests and incomplete model/database/date provenance. Pairing is unverified.
Values are replayed only after checking the imported JSON and source notebook checksums.
''')
    code('''gpu = gpu_history(ROOT)
display(Markdown(history_table(gpu)))
''')
    md('''**Observation:** both imported runtime tables report 3/4 SQL executions and 2/4 result
matches. **Explanation:** serial application scoring and concurrent serving characterize
different work. **Limitation:** the rounded summaries cannot independently prove a controlled
engine-speed comparison; peak VRAM was not captured. **Decision:** preserve them as context,
then use complete raw records and matching provenance for fresh optimization decisions.

[Imported evidence limitations](../baseline/mvp3/tutorial/RESULTS_PORTABILITY.md).
''')
    md('''## 5. Inspect a request and an evaluated answer

The older engine summaries lack raw SQL. The example below therefore uses the **recorded
Dell Qwen optimization baseline from November 6**, explicitly labeled, through the same
application path. It does not pretend to recover missing SQL from the older summaries.

The request is reconstructed from a saved prompt and flags using the same transport payload
builder; the archive did not capture raw HTTP packets. No request is sent by inspection.
[Request construction/timing](../src/clinical_trial_mvp/serving/client.py)
· [SQL guards and plan validation](../src/clinical_trial_mvp/agent/analyst.py)
· [result scorer](../src/clinical_trial_mvp/evaluation/scorers.py).
''')
    code('''from clinical_trial_mvp.tutorial.workflow import Study, template
from clinical_trial_mvp.tutorial.teaching import request_example, response_example
config = template('vllm', 'qwen')
config['mode'] = 'replay'
example = Study('vllm', 'qwen', root=ROOT, config=config)
example.run('fp16_cache_off')
display(JSON(request_example(example, 'fp16_cache_off')))
display(JSON(response_example(example, 'fp16_cache_off')))
''')
    md('''## 6. Carry these controls into optimization

| Layer | Measure | Do not infer |
|---|---|---|
| Transport/serving | first-content TTFT, request E2E, aggregate output tokens/s | SQL correctness from HTTP success |
| Generation | prompt/output work, stop/truncation, decode-rate estimate | GPU kernel time from client timing |
| Application | plan contract, SQL safety/execution, reference match, repair | semantic correctness from matching rows alone |

Keep model/tokenizer, database/date, prompts, precision, generation settings and server
configuration visible. Cache conditions and output lengths can change interpretation.
Startup stays outside individual request timing. Missing measurements stay unavailable.

Continue to [01 — Qwen cache and quantization](01_qwen_cache_quantization.ipynb), or jump to
[02 — Nemotron optimization](02_nemotron_optimization.ipynb). NIM extensions follow as
[03](03_qwen_nim.ipynb) and [04](04_nemotron_nim.ipynb). Shared instructions:
[install prerequisites](../SETUP.md) → [configure and run](../tutorial/RUN_ALL.md).
Finish with [What should I choose?](../tutorial/RESULTS.md#what-should-i-choose).
''')
    return dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
