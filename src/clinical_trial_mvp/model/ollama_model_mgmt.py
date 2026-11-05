"""Create and manage the optional local Ollama model."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from clinical_trial_mvp.config import Settings


@dataclass(frozen=True)
class OllamaCommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _ollama_env(settings: Settings) -> dict[str, str]:
    import os

    env = os.environ.copy()
    env["OLLAMA_HOST"] = settings.ollama_base_url
    return env


def run_ollama_command(
    args: list[str],
    settings: Settings | None = None,
    timeout: int = 300,
) -> OllamaCommandResult:
    active_settings = settings or Settings.from_env()
    executable = shutil.which("ollama")
    if executable is None:
        raise RuntimeError("Ollama CLI was not found on PATH.")

    command = [executable, *args]
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        env=_ollama_env(active_settings),
        text=True,
        timeout=timeout,
    )
    return OllamaCommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def verify_ollama(settings: Settings | None = None) -> dict[str, str | bool]:
    active_settings = settings or Settings.from_env()
    executable = shutil.which("ollama")
    installed = executable is not None
    version = ""
    service_ready = False
    list_output = ""
    error = ""

    if installed:
        version_result = run_ollama_command(["--version"], active_settings, timeout=15)
        version = version_result.stdout or version_result.stderr
        list_result = run_ollama_command(["list"], active_settings, timeout=15)
        service_ready = list_result.returncode == 0
        list_output = list_result.stdout
        error = list_result.stderr

    return {
        "installed": installed,
        "executable": executable or "",
        "version": version,
        "service_ready": service_ready,
        "base_url": active_settings.ollama_base_url,
        "list_output": list_output,
        "error": error,
    }


def list_ollama_model_names(settings: Settings | None = None) -> list[str]:
    result = run_ollama_command(["list"], settings, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Unable to list Ollama models.")

    names: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def ollama_model_exists(settings: Settings | None = None) -> bool:
    active_settings = settings or Settings.from_env()
    wanted = active_settings.ollama_model_name
    names = list_ollama_model_names(active_settings)
    return wanted in names or f"{wanted}:latest" in names


def render_modelfile(settings: Settings | None = None) -> str:
    active_settings = settings or Settings.from_env()
    model_path = active_settings.model_path.resolve()
    return f"""FROM {model_path}

PARAMETER temperature {active_settings.llm_temperature}
PARAMETER num_ctx {active_settings.llm_n_ctx}
PARAMETER num_predict {active_settings.llm_max_tokens}

SYSTEM \"\"\"You are Clinical Trial Analyst, a local-only clinical data analyst. Use only the supplied schema and query results. Prefer aggregate answers and do not expose raw subject identifiers.\"\"\"
"""


def write_project_modelfile(settings: Settings | None = None) -> Path:
    active_settings = settings or Settings.from_env()
    active_settings.ollama_modelfile.parent.mkdir(parents=True, exist_ok=True)
    active_settings.ollama_modelfile.write_text(render_modelfile(active_settings), encoding="utf-8")
    return active_settings.ollama_modelfile


def deploy_ollama_model(settings: Settings | None = None, force: bool = False) -> bool:
    active_settings = settings or Settings.from_env()
    if not active_settings.model_path.exists():
        raise FileNotFoundError(
            f"GGUF model file not found: {active_settings.model_path}. "
            "Run scripts/model/download_model.py first or update MODEL_PATH."
        )

    if not force and ollama_model_exists(active_settings):
        return False

    rendered = render_modelfile(active_settings)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".Modelfile", delete=False) as handle:
        handle.write(rendered)
        temp_modelfile = Path(handle.name)

    try:
        result = run_ollama_command(
            ["create", active_settings.ollama_model_name, "-f", str(temp_modelfile)],
            active_settings,
            timeout=1800,
        )
    finally:
        temp_modelfile.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Ollama model deployment failed.")

    write_project_modelfile(active_settings)
    return True


def ensure_ollama_model(settings: Settings | None = None) -> bool:
    active_settings = settings or Settings.from_env()
    if not active_settings.ollama_deploy_model:
        return False
    return deploy_ollama_model(active_settings, force=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the project's local Ollama GGUF deployment.")
    parser.add_argument(
        "command",
        choices=["status", "list", "write-modelfile", "deploy", "ensure"],
        nargs="?",
        default="status",
    )
    parser.add_argument("--force", action="store_true", help="Recreate the Ollama model if it already exists.")
    args = parser.parse_args()

    settings = Settings.from_env()

    if args.command == "status":
        status = verify_ollama(settings)
        print(f"installed: {status['installed']}")
        print(f"executable: {status['executable']}")
        print(f"version: {status['version']}")
        print(f"service_ready: {status['service_ready']}")
        print(f"base_url: {status['base_url']}")
        print(f"model_name: {settings.ollama_model_name}")
        print(f"model_exists: {ollama_model_exists(settings) if status['service_ready'] else False}")
        print(f"gguf_path: {settings.model_path}")
        print(f"gguf_exists: {settings.model_path.exists()}")
        print(f"deploy_flag: {settings.ollama_deploy_model}")
        if status["error"]:
            print(f"error: {status['error']}")
        return

    if args.command == "list":
        for name in list_ollama_model_names(settings):
            print(name)
        return

    if args.command == "write-modelfile":
        print(write_project_modelfile(settings))
        return

    if args.command == "deploy":
        created = deploy_ollama_model(settings, force=args.force)
        print("deployed" if created else "already exists")
        return

    if args.command == "ensure":
        created = ensure_ollama_model(settings)
        print("deployed" if created else "skipped")


if __name__ == "__main__":
    main()
