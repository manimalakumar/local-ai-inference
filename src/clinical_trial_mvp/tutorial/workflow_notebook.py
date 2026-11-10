"""Four live-first chapters with explicit per-experiment execution."""
import hashlib
from clinical_trial_mvp.tutorial import nemotron, nim_study, optimization

FILENAMES = {
    ('vllm','qwen'): '01_qwen_optimization.ipynb',
    ('vllm','nemotron'): '02_nemotron_optimization.ipynb',
    ('nim','qwen'): '03_qwen_nim.ipynb',
    ('nim','nemotron'): '04_nemotron_nim.ipynb',
}
ARCHIVE_FILENAMES = {
    ('vllm','qwen'): 'local AI optimization - Qwen.ipynb',
    ('vllm','nemotron'): 'local AI optimization - Nemotron.ipynb',
    ('nim','qwen'): 'local AI optimization - NIM Qwen.ipynb',
    ('nim','nemotron'): 'local AI optimization - NIM Nemotron.ipynb',
}


def _raw_document(runtime, model):
    cells=[]
    def md(slot,text):cells.append(dict(cell_type='markdown',id=slot,metadata={},source=(text.strip()+'\n').splitlines(keepends=True)))
    def code(slot,text):cells.append(dict(cell_type='code',id=slot,metadata={},source=(text.strip()+'\n').splitlines(keepends=True),execution_count=None,outputs=[]))
    label=nim_study.MODELS[model]['label'];model_id=nim_study.MODELS[model]['model_id']
    title=('Qwen: answer faster with caching and four-bit weights' if model=='qwen' else 'Nemotron: context, caching, concurrency and output budgets') if runtime=='vllm' else label+' through NVIDIA NIM'
    purpose=('We compare FP16 and AWQ W4A16 weights, each with prefix caching off and on. Caching reuses computation for shared instructions and schema; AWQ compresses weights while keeping activations at FP16.' if model=='qwen' else 'We keep FP16 weights and FP32 Mamba state, then test schema context, supported prefix caching, concurrent requests, scheduler capacity, reasoning and output limits.') if runtime=='vllm' else 'We deploy a vLLM-backed NIM container, check the application response contract, then compare schema context and one, two and four concurrent requests. A NIM profile selects a deployment configuration for the checkpoint and GPU.'
    md('intro',f'''# {title}

{purpose}

The workload is structured clinical-trial questions on synthetic data with no personally identifiable information (PII). We compare response time with executed SQL and reference-result matches.

Configure your GPU environment below, then run the cells in order to collect new measurements. Each experiment saves its results separately. Completed outputs are visible when a live execution has been saved. For the previous Dell measurements, use the reference links at the end.

Follow [SETUP.md](../SETUP.md) and [the execution guide](../tutorial/RUN_ALL.md) first. The target is **{label}**, `{model_id}`, on a **Linux NVIDIA GPU**. Model loading and server startup happen before request timing.''')
    md('configuration','''## 1. Imports and editable configuration

`MODE` controls execution: **live** collects new inference; **plan** previews commands; **replay** explicitly reads archived results. It overrides an older local JSON mode. CPU validation supplies `LOCAL_AI_LAB_MODE=plan` or `replay` before executing these cells.

Keep machine paths, exact checkpoint revisions and deployment settings in the per-notebook local JSON. Downloads are opt-in. Remove the obsolete `execute` flag; mode is the execution authority.''')
    code('configure',f'''from pathlib import Path
import os
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'src/clinical_trial_mvp').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from IPython.display import JSON, Markdown, display
from clinical_trial_mvp.tutorial.workflow import Study, read_config
from clinical_trial_mvp.tutorial.teaching import server_example, request_example, response_example, interpret

MODE = "live"  # "plan" previews commands; "replay" displays archived results
MODE = os.environ.get('LOCAL_AI_LAB_MODE', MODE)  # Explicit CPU validation override.
CONFIG = read_config('{runtime}', '{model}', root=ROOT, mode=MODE)
# Edit CONFIG here or the matching tutorial/config/*.local.json before continuing.
if 'study' in globals():
    study.close()
study = Study('{runtime}', '{model}', root=ROOT, config=CONFIG, mode=MODE)
results = study.results
print('Effective mode:', study.mode)
print('Model:', '{model_id}')
print('Hardware target: Linux / NVIDIA GPU; actual device checked next')
print('Session:', study.session_id)
print('Current-session measurements:', len(results))''')
    md('preflight','''## 2. Preflight checks

Check the GPU, selected runtime, checkpoint paths/revisions and port 8000 before loading a model. NIM also checks Docker access, an already available image, GPU access from that image and your selected profile. Missing prerequisites stop with instructions; no package installation or image pull is automatic.''')
    code('preflight-check',"display(JSON(study.preflight()))")
    md('prepare','''## 3. Model preparation and compatibility smoke

Preparation validates local checkpoint identity and precision. It downloads weights only when `download_models=true`. NIM verifies the selected profile against discovery for this deployment.

Qwen's vLLM study requires a successful AWQ SQL smoke before all four variants. Nemotron core needs no AWQ checkpoint: its optional stretch remains separate. NIM smoke tests both serving and SQL execution; a failed required smoke blocks dependent experiments.''')
    code('prepare-model',"print(study.prepare())")
    code('compatibility-smoke',"smoke = study.smoke()\nfrom clinical_trial_mvp.tutorial.result_display import result_table\ndisplay(Markdown(result_table(smoke)))")
    if runtime=='vllm' and model=='qwen':
        md('method','''## 4. Individual experiments

Hold the schema, four questions, seed 42, temperature 0, cap 1024 and context limit 8192 fixed. AWQ W4A16 changes weight storage, not KV-cache precision.

Cache-on targets use **reset → cold target → reset → different-question prime → identical target warm**, with an identical shared schema prefix. Cold and warm requests remain separate. Startup, readiness, warmup, reset, priming and SQL scoring are outside individual request timing.

Predict whether shared-prefix reuse helps TTFT more than complete-request latency. Each cell below starts its own server, makes new inference requests in Live mode, saves a new result and stops that server.''')
    elif runtime=='vllm':
        md('method','''## 4. Individual experiments

Start with the FP16 baseline. Each intervention changes one control: supported prefix caching; retrieved schema; client concurrency at fixed scheduler capacity; scheduler capacity at four clients; reasoning; or output cap. Concurrency and achieved batching are different quantities.

Enable `prefix_supported` only after verifying support for the installed hybrid backend. Unsupported cache rows stay skipped. AWQ is optional stretch work requiring a validated checkpoint and never gates the core experiments.''')
    else:
        md('method','''## 4. Individual experiments

Keep the NIM deployment fixed while changing full/retrieved schema or client concurrency. TP1/PP1 uses one GPU and one pipeline stage. A runnable profile still needs successful startup and a SQL smoke.

Check response structure as well as HTTP success. Qwen enables schema enforcement by default; Nemotron's setting is explicit in configuration. Compare the request contract before interpreting differences from native vLLM. The owned container stays running between these cells until cleanup.''')
    names=['baseline','schema_retrieved','concurrency_2','concurrency_4'] if runtime=='nim' else list(nemotron.configurations()) if model=='nemotron' else list(optimization.study_configurations())
    first=names[0]
    for name in names:
        md('experiment-'+name,'### '+name.replace('_',' ')+'\n\nPredict the effect on latency, output work and structured-answer correctness before executing this cell.')
        code('run-'+name,f'''result = study.run('{name}')  # Live: launches serving and makes new inference requests.
from clinical_trial_mvp.tutorial.result_display import result_table
display(Markdown(result_table(result)))
display(Markdown(interpret(study, '{name}')))''')
        if name==first:
            md('mechanism','''### Inspect the serving call and answer

The experiment above calls the model through the OpenAI-compatible `/v1/chat/completions` API. Inspect the launch command, request, generated plan/SQL and execution checks below. The request is constructed from the experiment prompt, schema profile and generation settings. In Plan, these details preview the configured call. The response shows the generated answer and its reference-result check.

[Request transport](../src/clinical_trial_mvp/serving/client.py) · [plan and SQL execution](../src/clinical_trial_mvp/serving/agent.py)''')
            code('inspect-current-request',f"display(JSON(server_example(study, '{first}')))\ndisplay(JSON(request_example(study, '{first}')))\ndisplay(JSON(response_example(study, '{first}')))")
    md('comparison','''## 5. Current-session comparison and interpretation

Read result matches alongside latency. TTFT measures first generated content; request completion covers generation and transport; application latency includes planning, validation, execution and repair. Missing memory telemetry stays unavailable. Failed rows remain failures.

Compare cold and warm TTFT. Did any reduction carry through to complete-request latency? For concurrency, did throughput improve within your latency budget? For precision or output caps, did the amount or correctness of generated SQL change?''')
    code('session-summary','display(Markdown(study.report()))\ndisplay(Markdown(study.cache_report()))')
    if runtime=='vllm' and model=='qwen':
        md('tokens','''### Check output work

Compare output tokens by phase and case before attributing latency differences to kernels or precision. Similar token counts do not establish identical SQL or computation.''')
        code('session-tokens',"from clinical_trial_mvp.tutorial.token_analysis import analyze, table\ntoken_analysis = analyze(study.current_results())\ndisplay(Markdown(table(token_analysis)))")
    md('figure','''### Plot this session

The figure uses only successful measured rows selected in this session. Missing rows are not filled from another run. Plan produces no measurement figure.''')
    code('session-figure',"from clinical_trial_mvp.tutorial.session_figures import figure\nfig = figure(study)\nif fig is not None:\n    display(fig)\n    import matplotlib.pyplot as plt\n    plt.close(fig)\nelse:\n    print('No measured rows available for a figure.')")
    md('cleanup','''## 6. Cleanup

Save the notebook to retain genuine Live outputs. Native servers stop after each row. Stop the owned NIM container before switching models or pausing. Cleanup is safe to repeat and does not stop external servers.''')
    code('cleanup-owned-servers','print(study.close())')
    md('references',f'''## Reference: previously measured results

The previous Dell measurements remain unchanged in [the results guide](../tutorial/RESULTS.md) and [the original executed notebook](history/dell-2026-11-07/{ARCHIVE_FILENAMES[(runtime,model)].replace(' ','%20')}). Select `MODE="replay"` explicitly to rebuild their tables. They are not measurements from this Live session.

[Execution and troubleshooting](../tutorial/RUN_ALL.md) · [SQL review packet](../tutorial/SQL_REVIEW.md)''')
    return dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}},nbformat=4,nbformat_minor=5)


def notebook_document(runtime,model):
    from clinical_trial_mvp.tutorial.navigation import apply_chapter_prose
    return apply_chapter_prose(_raw_document(runtime,model),FILENAMES[(runtime,model)])
