"""Prepare matching commands; execute only when explicitly requested on Linux CUDA."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

from clinical_trial_mvp.serving.benchmark import snapshot_fingerprint
from clinical_trial_mvp.serving.portable import display_command, portable_text


def launch_command(runtime, snapshot, *, port=8000, context=8192, tp=1, dtype="bfloat16", prefix_cache=False, quantization=None,
                   max_num_seqs=None, trust_remote_code=False, mamba_ssm_cache_dtype=None):
    if runtime not in {"vllm", "sglang"} or not 1 <= port <= 65535 or context < 1 or tp < 1:
        raise ValueError("Invalid runtime, port, context, or tensor parallel size")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError("Use the same explicit floating point dtype for both engines")
    common = ["--host", "127.0.0.1", "--port", str(port), "--served-model-name", "clinical-trial-mvp3",
              "--dtype", dtype]
    if quantization:
        if quantization not in {"awq", "gptq", "fp8"}:
            raise ValueError("Unsupported quantization selection")
        common.extend(["--quantization", quantization])
    extra = []
    if max_num_seqs is not None:
        if type(max_num_seqs) is not int or max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")
        extra += ["--max-num-seqs", str(max_num_seqs)]
    if trust_remote_code:
        extra += ["--trust-remote-code"]
    if mamba_ssm_cache_dtype is not None:
        if mamba_ssm_cache_dtype not in {"float16", "float32", "bfloat16"}:
            raise ValueError("Invalid Mamba state dtype")
        extra += ["--mamba-ssm-cache-dtype", mamba_ssm_cache_dtype]
    if runtime != "vllm" and extra:
        raise ValueError("Additional launch options are for vLLM only")
    if runtime == "vllm":
        return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(snapshot), *common,
                "--max-model-len", str(context), "--tensor-parallel-size", str(tp),
                "--generation-config", "vllm",
                "--enable-prefix-caching" if prefix_cache else "--no-enable-prefix-caching", *extra]
    return [sys.executable, "-m", "sglang.launch_server", "--model-path", str(snapshot), *common,
            "--context-length", str(context), "--tp-size", str(tp), "--sampling-defaults", "openai",
            *([] if prefix_cache else ["--disable-radix-cache"])]


def launch(runtime, snapshot: Path, metadata_path: Path, **options):
    if platform.system() != "Linux":
        raise ValueError("Live engine launch requires a Linux CUDA host; use command preview on this Mac")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU is not available")
    command = launch_command(runtime, snapshot.resolve(), **options)
    hardware = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"
    ], text=True).strip()
    metadata = {"runtime": runtime, "engine_version": importlib.metadata.version(runtime),
                "model_sha256": snapshot_fingerprint(snapshot), "hardware": hardware,
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "evidence": "hardware",
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "")) or None,
                "launch_settings": {"context": options.get("context", 8192), "tp": options.get("tp", 1),
                                    "dtype": options.get("dtype", "bfloat16"),
                                    "quantization": options.get("quantization"),
                                    "prefix_cache": options.get("prefix_cache", False)},
                "command": display_command(command), "model": "clinical-trial-mvp3"}
    for key in ("max_num_seqs", "trust_remote_code", "mamba_ssm_cache_dtype"):
        if key in options:
            metadata["launch_settings"][key] = options[key]
    import json
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("x") as output:
        output.write(json.dumps(metadata, indent=2) + "\n")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    # No shell invocation or implicit model downloads. This foreground process owns the GPU.
    with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace") as process:
        try:
            for line in process.stdout:
                print(portable_text(line), end="", flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
