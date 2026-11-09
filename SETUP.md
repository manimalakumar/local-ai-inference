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

The exact CPU pins were tested on macOS arm64, Python 3.11.15. Hosted Linux CI remains
pending its first push. This environment includes Jupyter, replay, plotting and test tools.
It installs no vLLM, GPU Torch or model weights. See `CPU_NOTEBOOK_ENVIRONMENT.json`.

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
