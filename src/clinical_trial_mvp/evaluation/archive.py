"""Preserve a compact, immutable code snapshot for each model iteration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


CODE_DIRECTORIES = ("src", "scripts", "tests", "sql")
CODE_FILES = (
    "pyproject.toml", "requirements.txt", "requirements-embed.txt", ".python-version", ".env.example",
    "README.md",
    "evals/evaluations.json",
)
CODE_DIRECTORIES += ("experiment-tracker",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_iteration(project_root: Path, run_id: str) -> Path:
    """Create one non-overwriting ZIP with code, tests, SQL, evals, and logs."""
    if not run_id.startswith("eval_") or "/" in run_id or "\\" in run_id:
        raise ValueError(f"Invalid run ID: {run_id}")
    legacy_output = project_root / "evals/results" / f"{run_id}_code.zip"
    output = legacy_output if legacy_output.exists() else project_root / "experiment-tracker/results" / f"{run_id}_code.zip"
    if output.exists():
        digest_path = output.with_suffix(".zip.sha256")
        if not digest_path.exists() or json.loads(digest_path.read_text())["sha256"] != _sha256(output):
            raise ValueError(f"Existing code ZIP failed checksum verification: {output}")
        return output
    paths = [project_root / name for name in CODE_FILES if (project_root / name).is_file()]
    for directory in CODE_DIRECTORIES:
        paths.extend(
            path for path in (project_root / directory).rglob("*")
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
            and path.suffix != ".pyc" and path.name != "catalog.json"
            and "results" not in path.relative_to(project_root / directory).parts
        )
    paths = sorted(set(paths), key=lambda path: str(path.relative_to(project_root)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            for path in paths:
                archive.write(path, arcname=str(path.relative_to(project_root)))
    digest_path = output.with_suffix(".zip.sha256")
    with digest_path.open("x", encoding="utf-8") as checksum:
        checksum.write(json.dumps({"file": output.name, "sha256": _sha256(output)}) + "\n")
    return output
