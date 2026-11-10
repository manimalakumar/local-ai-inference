# Install prerequisites

One shared installation guide for the existing `local-ai-inference` repository. Read
[00 — local serving](notebooks/00_local_serving.ipynb) for the foundation or jump directly to
[01 — Qwen optimization](notebooks/01_qwen_cache_quantization.ipynb). Configuration, modes,
profile selection, execution and cleanup belong in [RUN_ALL.md](tutorial/RUN_ALL.md).

## CPU notebook environment — all replay and plan chapters

From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt -c constraints/cpu-notebook-py311.txt
.venv/bin/python -m pytest -q
```

The exact CPU pins were tested on macOS arm64, Python 3.11.15. Hosted Linux CI passed
for source commit `56feab1`; see [the scoped validation record](VALIDATION.md). This environment includes Jupyter, replay, plotting and test tools.
It installs no vLLM, GPU Torch or model weights. See `CPU_NOTEBOOK_ENVIRONMENT.json`.

## Notebook 00: Replay or run from scratch

Notebook 00 establishes a local baseline: full/retrieved schema context,
llama.cpp/Ollama serving and structured-answer quality.

| Option | Setup | Result |
|---|---|---|
| **Replay — optional** | Install the notebook environment above and Run All | Read the supplied tables and a checked Nemotron GGUF answer |
| **Live — default, run from scratch** | Prepare a local GGUF server using the steps below, then Run All | Generate plans/SQL, execute queries and measure your baseline |

Both use the supplied synthetic database, question set and retrieval selection.
From scratch means new inference on those fixed inputs. Your outputs and timings
can differ; the new run retains its actual model fingerprint and settings.

### Fresh local baseline with Nemotron and llama.cpp

Use Python 3.11 for the notebook environment. Create a separate environment for
the local inference server, or reuse your existing compatible environment:

```bash
python3.11 -m venv .venv-llama
.venv-llama/bin/python -m pip install 'llama-cpp-python[server]==0.3.35'
```

For platform/build prerequisites, use the official
[llama-cpp-python installation and server instructions](https://llama-cpp-python.readthedocs.io/en/latest/server/).
The starter runs on CPU (`n_gpu_layers=0`) with context 8192 and seed 42.
It requires no NVIDIA GPU or Docker.

Download the Nemotron Q4_K_M GGUF used by the local baseline:

```bash
.venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF', filename='NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf', local_dir='models')"
```

The source is the [NVIDIA GGUF repository](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF).
The launch helper checks the file's SHA-256 against the tutorial artifact:
`be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2`.

Prepare the local configuration:

```bash
.venv/bin/python scripts/tutorial/configure_local_foundation.py --gguf models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf --server-python .venv-llama/bin/python
```

This creates `tutorial/config/local-serving.local.json` and prints the server
command. Existing local configs are preserved; edit yours or choose a new
`--config` path if one already exists. The committed local example is
`tutorial/config/local-serving.example.json`.

1. Run the printed command in a separate terminal, from the repository root.
2. Wait for model loading to finish, then check readiness:

   ```bash
   curl --fail http://127.0.0.1:8080/v1/models
   ```

3. Start Jupyter or select the notebook's Python 3.11 environment in your editor:

   ```bash
   .venv/bin/python -m jupyter lab notebooks/00_local_serving.ipynb
   ```

4. Restart the kernel and **Run All**. The first cell should report
   **`Chapter 00 mode: live`**. `LOCAL_AI_LAB_MODE` overrides the JSON mode if set
   in the terminal that launched Jupyter.
5. Read the tables and generated structured answer, save the notebook, and stop
   the server with **Ctrl+C** in its terminal.

The starter evaluates four questions with full/retrieved schema, one serial
application-quality pass per profile and three direct-serving repetitions per
question. One serving warmup is excluded from timing. The worked answer comes
from your full-schema result. Model loading happens before request timing.

Settings are in `tutorial/config/local-serving.local.json`:

| Setting | Purpose |
|---|---|
| `mode` | `replay`, `live`, or optional `plan` to inspect rows without making requests |
| `case_ids` | Questions to evaluate; the default is the four compact questions |
| `repeats`, `warmups`, `max_tokens` | Direct-serving repeats, warmups and output cap |
| `context_server` | The named endpoint for full/retrieved context and the worked answer |
| `local_servers` | The named llama.cpp/Ollama endpoints to compare |
| `servers` | Model alias, endpoint, precision, context and actual launch metadata |

For the full evaluation, set `case_ids` to every ID in `evals/evaluations.json`.
Quality scoring includes all 30 questions; direct-serving timing includes the 26
answerable questions. You can print the ID list with:

```bash
.venv/bin/python -c 'import json; print(json.dumps([c["id"] for c in json.load(open("evals/evaluations.json"))]))'
```

Fresh rows and failures are saved under
`experiment-tracker/tutorial/foundation/<session-id>/`. Complete benchmark records
are also retained by the shared harness under `experiment-tracker/tutorial/runs/`.
The launch helper records the actual server PID for CPU process-RSS sampling;
unavailable samples remain unavailable. A failed run stops with its saved error.

### Notebook 00 embedding model

Live mode's embedding demonstration needs Sentence Transformers in the notebook
kernel and a local copy of the embedding model. From the repository root:

```bash
.venv/bin/python -m pip install -r requirements-embedding.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nvidia/Nemotron-3-Embed-1B-BF16', revision='c0c9fea93ea424587517f2c59e20db9f1d6bf615', local_dir='models/Nemotron-3-Embed-1B-BF16')"
```

Restart the notebook kernel after installing. The cell loads only local files,
uses CPU, and embeds schema descriptions and the question. It requires no separate
embedding server. Set `embedding_model_path` in your local JSON if the weights
are elsewhere. Replay does not need these dependencies or weights.
The measured baseline keeps its frozen retrieval selection; the embedding cell
shows fresh retrieval separately and does not overwrite the cache.

### Optional Ollama comparison

Use the same GGUF file in Ollama and add its running endpoint to `servers` and
`local_servers`. Match context, generation limits and hardware settings when
comparing runtimes. The [local-serving experiment report](baseline/mvp2/experiment-tracker/mvp2/serving/MILESTONE_REPORT.md)
contains the original model/import and serving settings.

Each endpoint needs its own launch metadata: `runtime`, `model`, `engine_version`,
`hardware`, `model_sha256`, `evidence: "hardware"`, and `launch_settings` matching
the configured context/precision. Use the actual artifact hash and runtime version.
The starter creates these automatically for llama.cpp; optional Ollama endpoint
configuration remains a manual setup and needs fresh hardware validation.

### Choose Replay

Set `mode` to `replay` and restart the kernel, or start Jupyter with:

```bash
LOCAL_AI_LAB_MODE=replay .venv/bin/python -m jupyter lab notebooks/00_local_serving.ipynb
```

Replay needs no model or server. The new local Live starter is covered by CPU
configuration/guard tests; its end-to-end native run still needs validation.
Continue to chapters 01–04 for GPU optimization and NIM deployment.

## Preserve the synthetic data

The supplied ten-table DuckDB database and CSVs are synthetic, inherited from the existing
clinical-trial workload. The generation source is `src/clinical_trial_mvp/data/synthetic_data.py`.
Keep the frozen database: regenerating it can change historical reference-result comparisons.
Its SHA-256 matches the recorded Dell baseline:

```text
f1a7100697b0980b56359632fde49d042fb147260da03931a53a3cbfa7f6830e
```

[Data/schema/evaluation fingerprints](tutorial/serving_foundation.json) cover the database,
CSVs, schema, reference cases and frozen retrieval cache. Chapter 00 checks the database and
opens it read-only. No database rebuild is part of installation or replay.

## Native vLLM — optional, for live GPU chapters

Reuse a separate compatible Linux GPU environment. On the Dell the working interpreter is
`/home/setup/.venv-vllm/bin/python`, with vLLM 0.32.0. Verify it directly:

```bash
/home/setup/.venv-vllm/bin/python -c 'import sys, vllm, torch; print(sys.executable); print(vllm.__version__); print(torch.cuda.is_available())'
```

No project symlink is required. For a new machine use the official
[vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/) for the
actual GPU/driver stack. Keep the engine/Torch/CUDA build fixed across a comparison.
Jupyter may run in the separate CPU environment. Replay and Plan need no GPU engine.

## Docker and NVIDIA Toolkit — system-wide, once per machine

These are reusable across projects. Follow the official [Docker Ubuntu guide](https://docs.docker.com/engine/install/ubuntu/)
and [NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Keep your working driver. For existing conflicting packages or other OS versions, consult
those prerequisites rather than blindly removing system packages.

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg2
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF_DOCKER
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF_DOCKER
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run --rm hello-world
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi
sudo usermod -aG docker "$USER"
```

The Docker group grants administrator-level access. Log out/in, or use `newgrp docker`.
Ubuntu 26.04 provides `newgrp` in `util-linux-extra`; install that package if missing.
Verify `docker ps` and the GPU test without sudo. **Restart Jupyter from the group-enabled
terminal**: a running Jupyter process does not acquire new membership. `--rm` removes only
the temporary test container, not your files or downloaded Ubuntu image.

## NIM container image — optional prerequisite

After Docker GPU access works, download the same model-free image used by both NIM chapters:

```bash
docker pull nvcr.io/nim/nvidia/vllm-model-free-nim:2.1.5
```

No second Docker installation/image is needed for the second model. The live workflow pins
the local image digest; profile discovery and per-model settings are explained in the
[shared NIM technical guide](tutorial/NIM_MVP4.md). For gated/private images follow
[NVIDIA's registry authentication instructions](https://docs.nvidia.com/nim/large-language-models/latest/get-started/installation.html).
Credentials belong in environment variables, never notebooks, configs or Git.

Once prerequisites are installed, continue to [configure and run](tutorial/RUN_ALL.md).
