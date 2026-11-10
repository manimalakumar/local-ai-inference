"""Live-first edge tutorials: a Qwen generation path and a Nemotron recipe probe."""

def raw_document(model):
    cells=[]
    def md(slot,text):cells.append(dict(cell_type='markdown',id=slot,metadata={},source=(text.strip()+'\n').splitlines(keepends=True)))
    def code(slot,text):cells.append(dict(cell_type='code',id='code-'+slot,metadata={},source=(text.strip()+'\n').splitlines(keepends=True),execution_count=None,outputs=[]))
    qwen=model=='qwen'
    label='Qwen/Qwen3-4B' if qwen else 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
    md('edge-intro' if qwen else 'probe-intro', '''# Qwen: export a model and run a structured question on CPU

The serving chapters used an HTTP API. Here we package the same **Qwen3-4B** base revision as an **ExecuTorch XNNPACK** program and invoke its native CPU runner. The recipe uses **INT4 linear weights with dynamic INT8 activations and INT8 embeddings**, which differs from AWQ W4A16.

We test one direct clinical SQL question with a short schema. This is a portability smoke, separate from the full planning/repair workload and GPU serving benchmarks.

Configure the edge environment below, then run the cells in order. Reuse a validated export explicitly or perform conversion and export. **Generation always runs afresh in Live mode.** Install the [notebook environment](../SETUP.md). See [the edge setup guide](../tutorial/EXECUTORCH_MAC.md).''' if qwen else '''# Nemotron: check the CPU export path before loading weights

Can the same **Nemotron 3 Nano 4B** hybrid checkpoint used in the serving chapters use the Qwen CPU deployment path? First test whether the installed ExecuTorch export recipe accepts its architecture.

Nemotron-H includes Mamba recurrent state alongside attention. Qwen's adapter does not supply that state mapping. The following cells execute a **fresh configuration-only probe** of the installed exporter. Unsupported configuration is a valid finding; this chapter does not run native inference or report generation timings.

Install the [notebook environment](../SETUP.md). Follow [the edge setup guide](../tutorial/EXECUTORCH_MAC.md), configure your edge Python, then run the cells in order. Downloads are opt-in; this check needs only the pinned `config.json`, not model weights.''')
    md('edge-config','''## 1. Imports and editable configuration

`MODE="live"` performs the stages below. Choose **plan** to inspect without edge execution or **replay** to read the explicitly selected prior record. The notebook mode overrides an old local JSON mode. No stage installs packages or downloads weights implicitly.''')
    code('edge-configure',f'''from pathlib import Path
import os
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import JSON, Markdown, display
from clinical_trial_mvp.tutorial.edge_workflow import EdgeStudy

MODE = "live"  # "plan" previews stages; "replay" displays archived results
MODE = os.environ.get('LOCAL_AI_LAB_MODE', MODE)  # Explicit CPU validation override.
# Machine settings: tutorial/config/edge-{model}.local.json
study = EdgeStudy('{model}', root=ROOT, mode=MODE)
print('Effective mode:', study.mode)
print('Model:', '{label}')
print('Hardware target: local CPU / separate ExecuTorch environment')
print('Session:', study.session_id)
print('Current-session measurements:', len(study.results))
display(JSON(study.config))''')
    md('edge-preflight','''## 2. Preflight

Check the configured edge Python and installed runtime before any model work. Qwen requires a pinned source checkpoint and either a validated reusable export or explicit conversion/export. Nemotron needs only its pinned configuration. Missing prerequisites stop with instructions.''')
    code('edge-preflight-check','display(JSON(study.preflight()))')
    if qwen:
        md('edge-export','''## 3. Prepare and validate the CPU program

Stages run as separate processes so large reference and converted models do not stay resident together. `download_models=false` uses staged source weights. Set it to true only to request a download.

With `reuse_export=true`, conversion, reference, validation and export cells verify the recorded artifact hashes and report **reused_artifact**. They do not claim a new export. To build afresh, set `reuse_export=false` and select a new `artifact_dir` in the local config. Conversion uses the official Qwen3 weight mapping; FP32 logit checks must pass before quantized export.

The native recipe and generation call are in [run_qwen_edge.py](../scripts/tutorial/run_qwen_edge.py).''')
        for stage in ('download','prepare','reference','validate','export'):
            code('edge-'+stage,f"stage_result = study.stage('{stage}')\nfrom clinical_trial_mvp.tutorial.result_display import result_table\ndisplay(Markdown(result_table(stage_result)))")
        md('edge-inference','''## 4. Invoke the model and check its answer

The next cell starts a **new native generation process** with the verified `.pte` program and tokenizer. The runner calls `TextLLMRunner.generate()` with temperature zero, a short clinical schema and an output cap of 192 tokens. It uses no HTTP serving API and loads no saved answer. Model-load time is separate from inference timing.

The following cell executes the newly generated SQL in DuckDB and compares its rows with the reference. SQL scoring occurs after native timing.''')
        code('edge-generate',"edge = study.stage('run')\nfrom clinical_trial_mvp.tutorial.result_display import result_table\ndisplay(Markdown(result_table(edge)))")
        code('edge-sql-check',"sql_check = study.stage('sql_check')\nfrom clinical_trial_mvp.tutorial.result_display import result_table\ndisplay(Markdown(result_table(sql_check)))")
        md('edge-metrics','''## 5. Read this session's measurements

Compare model loading, first-token time, inference latency, artifact footprint and process RSS. These are one short-question sample, not serving percentiles. RSS is CPU process memory, not GPU memory. Inspect generated SQL and its reference-result check alongside timing.''')
        code('edge-metrics', '''current = study.results.get('run', {})
if current.get('session_id') == study.session_id and current.get('status') == 'completed':
    observer = current.get('native_observer') or {}
    ttft = ((observer['first_token_ms'] - observer['inference_start_ms']) / 1000
            if 'first_token_ms' in observer and 'inference_start_ms' in observer else None)
    from clinical_trial_mvp.tutorial.result_display import result_table
    display(Markdown(result_table({'session': study.session_id,
        'model_load_seconds': current.get('startup_seconds'), 'ttft_seconds': ttft,
        'inference_seconds': current.get('single_request_seconds'),
        'footprint_bytes': current.get('footprint_bytes'), 'memory': current.get('memory'),
        'generation': current.get('generation'),
        'sql_check': study.results.get('sql_check')})))
else:
    print('Native measurements are unavailable until generation completes in this session.')
display(Markdown(study.report()))''')
    else:
        md('probe-mechanism','''## 3. Execute the compatibility probe

The next cell invokes [probe_nemotron_edge.py](../scripts/tutorial/probe_nemotron_edge.py) using your installed edge environment. It tests `nemotron_h` and `nemotron_3_nano_4b` against the installed model configuration enum and saves a unique result.

It reports exporter version, accepted model types and each attempt. If a recipe is accepted, conversion and native inference still require validation; acceptance alone supplies no performance result.''')
        code('edge-probe',"probe = study.probe()\nfrom clinical_trial_mvp.tutorial.result_display import result_table\ndisplay(Markdown(result_table(probe)))")
        md('probe-interpretation','''## 4. Interpret the finding

If the result is **blocked_compatibility**, the installed standard recipe lacks this model configuration. It does not establish that a custom port is impossible. A working port would need validated Mamba state handling, weight conversion, graph capture, supported operators, PTE loading and native generation before SQL or timing could be evaluated.''')
        code('edge-probe-summary', '''display(Markdown(study.report()))
from clinical_trial_mvp.tutorial.result_display import result_table
display(Markdown(result_table({'native_inference_run': probe.get('native_inference_run', False),
              'SQL_correctness': probe.get('sql_correctness'),
              'TTFT': probe.get('ttft_seconds'),
              'request_latency': probe.get('end_to_end_seconds'),
              'peak_memory': probe.get('peak_memory_bytes')})))''')
    md('edge-cleanup','''## Cleanup

The edge stages execute in foreground child processes and retain no serving process. Save the notebook to retain genuine current outputs. Failures stay in the session evidence.''')
    code('edge-cleanup',"print(study.close())")
    md('edge-reference','''## Reference: previously measured results

The [edge guide](../tutorial/EXECUTORCH_MAC.md) preserves the previous Qwen native run and Nemotron compatibility check. Choose `MODE="replay"` explicitly to inspect those records. The saved artifacts and historical records remain unchanged; they are not results from a new Live execution.''')
    return dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5)
