# Start on the Dell

The project directory is `/home/setup/projects/local-ai-inference`.
Save the notebooks and stop old notebook kernels before pulling changes.
These instructions assume the updated files have already been copied or pushed/pulled;
the scripts are not available on the Dell until that transfer is complete.

## Notebook 01: use Run All

The revised Qwen optimization notebook has no executed outputs because its prior
Replay outputs were cleared. It is ready for **Live Run All** after configuration
and preflight succeed. It runs the required AWQ smoke, then FP16/cache-off,
FP16/cache-on, AWQ/cache-off and AWQ/cache-on as separate experiment cells.

Start Jupyter from any terminal; shell activation does not choose the interpreters:

```bash
cd /home/setup/projects/local-ai-inference
bash scripts/dell/start_jupyter.sh
```

Open the printed local URL containing the token. Choose kernel
**Clinical Trial MVP4 (notebook)**, confirm `MODE="live"`, and choose **Run All**.
The script runs Jupyter on `127.0.0.1:8888` with no automatic browser launch.
Keep its terminal open. The notebook Python needs `requirements-notebook.txt`;
it does not need Torch/vLLM. The inference Python is separately configured as
`/home/setup/.venv-vllm/bin/python` by default. No environment symlink is created.

The script preserves existing configuration files. Review
`tutorial/config/vllm-qwen.local.json` before the run:

- `vllm_python`: the actual working absolute CUDA Python path.
- `base_snapshot` and `awq_snapshot`: the two downloaded frozen Qwen folders.
- Exact revisions: restore existing `models/qwen-revisions.json` or set
  `base_revision` and `awq_revision` explicitly.
- `awq_derivation`: retain or record the provenance linking the AWQ source to the
  exact base revision. The script does not invent this attestation.
- `download_models=false` reuses staged weights; no implicit download or installation.

Do **not** manually start vLLM, SGLang or NIM before Run All. The optimization
notebooks own port 8000, launch servers, wait for readiness and stop them. A server
already on that port produces an actionable error. Notebook 01 uses **vLLM**;
it does not run a SGLang optimization comparison.

Start other notebooks or use an existing notebook environment:

```bash
bash scripts/dell/start_jupyter.sh notebooks/02_nemotron_optimization.ipynb
NOTEBOOK_PYTHON=/path/to/notebook-env/bin/python bash scripts/dell/start_jupyter.sh
JUPYTER_PORT=8889 bash scripts/dell/start_jupyter.sh
```

Run only one notebook GPU workload at a time. Stop its owned servers with the
notebook cleanup cell, save, then interrupt Jupyter with Ctrl+C when finished.

## Optional manual vLLM/SGLang endpoint checks

These commands are separate from notebook 01 Run All. Use **one engine at a time**
on the 16 GiB GPU. Different ports do not provide additional GPU memory.
The default manual port is 8001. The controller itself needs only system Python 3;
each engine runs through its exact existing environment Python.

vLLM Qwen FP16, cache off:

```bash
cd /home/setup/projects/local-ai-inference
bash scripts/dell/serve.sh vllm --python /home/setup/.venv-vllm/bin/python
```

Wait for **READY**, then in another terminal:

```bash
curl --fail http://127.0.0.1:8001/v1/models
```

Stop that server with **Ctrl+C in its launch terminal** before the next command.
The script stops only its own process group and records shutdown. AWQ cache on:

```bash
bash scripts/dell/serve.sh vllm --python /home/setup/.venv-vllm/bin/python --awq --prefix-cache
```

SGLang Qwen FP16, radix cache off:

```bash
bash scripts/dell/serve.sh sglang --python /home/setup/.venv-sglang/bin/python
```

For SGLang radix caching, add `--prefix-cache`. Native cache control flags differ;
this is an endpoint check, not a paired benchmark. Nemotron is supported by this
manual script through vLLM only:

```bash
bash scripts/dell/serve.sh vllm --python /home/setup/.venv-vllm/bin/python --model nemotron
```

Inspect commands without GPU work:

```bash
bash scripts/dell/serve.sh vllm --awq --prefix-cache --dry-run
bash scripts/dell/serve.sh sglang --dry-run
```

The manual vLLM launcher consistently exports `VLLM_SERVER_DEV_MODE=1`; FP16
activations are explicit. An occupied port is rejected, not reused or killed.
No package installation, model download, container pull or environment symlink
is automatic. Each launch has a unique log and metadata directory under
`experiment-tracker/mvp4/manual-servers/`. If startup fails, inspect the printed
`server.log`; a model listing alone does not prove AWQ or SQL correctness. Notebook
01 performs its own required compatibility smoke.

See [the complete execution guide](RUN_ALL.md) for experiment and evidence details.
