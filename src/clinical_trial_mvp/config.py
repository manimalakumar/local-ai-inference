"""Load local paths, model settings, and query limits from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def _bool_from_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    db_path: Path
    csv_dir: Path
    model_path: Path
    hf_token: str
    hf_model_repo: str
    hf_model_file: str
    model_runtime: str
    llm_n_ctx: int
    llm_n_gpu_layers: int
    llm_temperature: float
    llm_max_tokens: int
    llm_seed: int | None
    ollama_deploy_model: bool
    ollama_base_url: str
    ollama_model_name: str
    ollama_modelfile: Path
    sql_result_row_limit: int
    sql_execution_timeout_seconds: float
    privacy_salt: str
    schema_context_mode: str = "retrieved"
    embed_model_path: Path | None = None
    embed_model_id: str = "nemotron_embed_1b"
    embed_top_k: int = 12
    embed_max_seed_tables: int = 4
    max_repairs: int = 1
    reasoning_enabled: bool = False
    retrieval_cache_path: Path | None = None
    llama_server_base_url: str = "http://127.0.0.1:8080"
    bench_ttft_slo_seconds: float = 5.0
    bench_e2e_slo_seconds: float = 30.0
    llama_server_parallel: int = 1
    llama_server_gpu_layers: int = 99
    bench_ollama_model_name: str = "clinical-trial-mvp2-bench"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            project_root=PROJECT_ROOT,
            db_path=_path_from_env("CLINICAL_TRIAL_DB_PATH", "data/db/clinical_trials.duckdb"),
            csv_dir=_path_from_env("CLINICAL_TRIAL_CSV_DIR", "data/raw"),
            model_path=_path_from_env(
                "MODEL_PATH",
                "models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
            ),
            hf_token=os.getenv("HF_TOKEN", ""),
            hf_model_repo=os.getenv(
                "HF_MODEL_REPO",
                "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF",
            ),
            hf_model_file=os.getenv(
                "HF_MODEL_FILE",
                "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
            ),
            model_runtime=os.getenv("MODEL_RUNTIME", "llama_cpp").strip().lower(),
            llm_n_ctx=int(os.getenv("LLM_N_CTX", "8192")),
            llm_n_gpu_layers=int(os.getenv("LLM_N_GPU_LAYERS", "0")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            llm_seed=int(os.environ["LLM_SEED"]) if os.getenv("LLM_SEED", "").strip() else None,
            ollama_deploy_model=_bool_from_env("OLLAMA_DEPLOY_MODEL", False),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
            ollama_model_name=os.getenv("OLLAMA_MODEL_NAME", "clinical-trial-nemotron").strip(),
            ollama_modelfile=_path_from_env("OLLAMA_MODELFILE", "ollama/Modelfile"),
            sql_result_row_limit=int(os.getenv("SQL_RESULT_ROW_LIMIT", "100")),
            sql_execution_timeout_seconds=float(os.getenv("SQL_EXECUTION_TIMEOUT_SECONDS", "10")),
            privacy_salt=os.getenv("PRIVACY_SALT", "replace-this-local-salt"),
            schema_context_mode=os.getenv("SCHEMA_CONTEXT_MODE", "retrieved").strip().lower(),
            embed_model_path=_path_from_env("EMBED_MODEL_PATH", "models/Nemotron-3-Embed-1B-BF16"),
            embed_model_id=os.getenv("EMBED_MODEL_ID", "nemotron_embed_1b").strip().lower(),
            embed_top_k=int(os.getenv("EMBED_TOP_K", "12")),
            embed_max_seed_tables=int(os.getenv("EMBED_MAX_SEED_TABLES", "4")),
            max_repairs=int(os.getenv("MAX_REPAIRS", "1")),
            reasoning_enabled=_bool_from_env("LLM_REASONING_ENABLED", False),
            retrieval_cache_path=_path_from_env("EMBED_RETRIEVAL_CACHE", "experiment-tracker/mvp3/retrieval-cache/current/evaluations.json"),
            llama_server_base_url=os.getenv("LLAMA_SERVER_BASE_URL", "http://127.0.0.1:8080").strip(),
            bench_ttft_slo_seconds=float(os.getenv("BENCH_TTFT_SLO_SECONDS", "5")),
            bench_e2e_slo_seconds=float(os.getenv("BENCH_E2E_SLO_SECONDS", "30")),
            llama_server_parallel=int(os.getenv("LLAMA_SERVER_PARALLEL", "1")),
            llama_server_gpu_layers=int(os.getenv("LLAMA_SERVER_GPU_LAYERS", "99")),
            bench_ollama_model_name=os.getenv("BENCH_OLLAMA_MODEL_NAME", "clinical-trial-mvp2-bench").strip(),
        )
