"""Optional CPU portability chapter adapted from the measured SmolLM2 case."""
import hashlib
import json

FILENAME = '05_edge_portability_executorch.ipynb'
RECORD = 'experiment-tracker/tutorial/edge/20261029T161510_034051Z.json'
# Frozen historical evidence, never a latest-file glob or a substitute inference run.
RECORD_SHA256 = '04076d8f693e18a503335da98f296d5af854a539389027f705b8b602f2d36982'


def recorded_edge(root):
    path = root / RECORD
    if hashlib.sha256(path.read_bytes()).hexdigest() != RECORD_SHA256:
        raise ValueError('ExecuTorch historical evidence checksum mismatch')
    result = json.loads(path.read_text())
    if result['status'] != 'completed' or result['evidence'] != 'hardware':
        raise ValueError('Expected completed hardware portability evidence')
    return {'label': 'recorded Mac CPU portability measurement — October 29, 2026',
            'source': RECORD, 'source_sha256': RECORD_SHA256, **result}


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
    md('''# Edge portability with ExecuTorch — optional CPU branch

*Local AI Inference Lab: Serving, Optimization and Application Quality*

**Learn:** distinguish checkpoint conversion, export, native execution, artifact footprint,
and process memory; understand why portability needs its own evidence.
**Requirements:** the [CPU notebook environment](../SETUP.md) and included historical JSON.
No ExecuTorch/Torch install, model download, GPU or credentials are needed for this chapter.
**Expected outputs:** verified recorded artifact identity, timing/memory table, generated sample
and an observation → explanation → limitation → decision.

**Read-only replay:** this adapts sections 17–18 of the earlier serving notebook. It does
not launch an exporter or perform a new model experiment, in Replay, Plan or Live settings.
It always shows the completed **recorded Mac CPU portability measurement**, separately from
recorded Dell measurements. The four optimization/NIM chapters keep their own Live workflow.

SmolLM2 is a small base completion model. This is not a clinical-model ranking or a
Text-to-SQL quality measurement. The planned Llama 3.2 1B path lacked checkpoint access;
the public SmolLM2 checkpoint supplied the existing portability demonstration.
''')
    code('''from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import JSON, Markdown, display
from clinical_trial_mvp.tutorial.edge_notebook import recorded_edge
edge = recorded_edge(ROOT)
print(edge['label'])
display(JSON({k: edge[k] for k in ('source', 'source_sha256', 'model_name', 'model_revision',
    'model_sha256', 'tokenizer_sha256', 'runner_sha256', 'backend', 'hardware', 'dtype',
    'max_context_length', 'threads', 'versions')}))
''')
    md('''## 1. Follow the existing export and run boundary

**Pinned Hugging Face SmolLM2-135M → complete FP32 weight mapping → reference-logit checks
→ XNNPACK `.pte` export → native C++ TextLLMRunner via Python bindings.**

The exporter used the upstream Llama2Model architecture class; that name does not change
the downloaded model identity. The recorded conversion checked two input sequences against
the original HF model at atol=rtol=0.001. This validates that conversion check, not SQL quality.
Native custom-kernel registration was required before loading the exported program.

The archived [guide](../baseline/mvp3/tutorial/EXECUTORCH_MAC.md) describes the historical
recipe and its then-current scripts. Those old source paths are archival context, not a
fresh-clone setup promise. [Current scope and evidence](../tutorial/EXECUTORCH_MAC.md).
''')
    code("display(JSON(edge['checkpoint_validation']))")
    md('''## 2. Keep model load, inference and whole-process time separate

Model load excludes Python initialization. Inference includes prompt evaluation.
Whole-process wall time includes child startup and runtime initialization. Artifact size
excludes the Python/PyTorch libraries. Sampled process RSS includes loaded libraries and
is not GPU memory. Sampling can miss brief peaks; operator fallback remains unreviewed.
''')
    code('''rows = [
    ('Native model load', f"{edge['startup_seconds']:.3f} s"),
    ('Native single request, including prompt evaluation', f"{edge['single_request_seconds']:.3f} s"),
    ('Whole child process', f"{edge['wall_seconds']:.3f} s"),
    ('PTE + tokenizer footprint', f"{edge['footprint_bytes'] / 2**20:.2f} MiB"),
    ('Sampled peak process RSS', f"{edge['memory']['peak_bytes'] / 2**20:.2f} MiB"),
    ('Native prompt/generated tokens', f"{edge['generation']['num_prompt_tokens']} / {edge['generation']['num_generated_tokens']}"),
    ('Reviewed operator fallback', 'unavailable' if edge['operator_fallback_count'] is None else str(edge['operator_fallback_count'])),
]
display(Markdown('| Metric | Recorded Mac CPU value |\\n|---|---|\\n' +
                 '\\n'.join(f'| {name} | {value} |' for name, value in rows)))
display(JSON({'prompt': edge['generation']['prompt'], 'generated_text': edge['generation']['text'],
              'limitations': edge['limitations']}))
''')
    md('''## 3. Interpret portability separately

**Observation:** the recorded M4 Pro CPU/XNNPACK run loaded in **0.076 s**, generated its
single sample in **0.199 s** including prompt evaluation, and took **3.480 s** for the whole
child process. The model/tokenizer footprint was **623.60 MiB**, sampled peak RSS **1018.13 MiB**.
**Explanation:** an exported program can run through the native edge runtime while startup,
artifact size and runtime libraries remain distinct deployment costs.
**Limitation:** one warm-filesystem smoke sample; FP32 weights, small model, no clinical task,
no concurrent-server benchmark and no reviewed fallback trace. It establishes neither cold
start percentiles nor clinical parity with Qwen/Nemotron. Export delegation is not runtime
fallback evidence.
**Decision:** use the portability branch to assess a suitable embedded workload and its
runtime/artifact budget. Validate that workload separately before selecting an edge model.

Continue with [What should I choose?](../tutorial/RESULTS.md#what-should-i-choose), return to
[01 — Qwen optimization](01_qwen_cache_quantization.ipynb), or inspect the
[original hardware record](../experiment-tracker/tutorial/edge/20261029T161510_034051Z.json).
''')
    return dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}}, nbformat=4, nbformat_minor=5)
