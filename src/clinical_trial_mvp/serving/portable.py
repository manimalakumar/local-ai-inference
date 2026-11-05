"""Portable paths for shared manifests, previews, and console messages."""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def portable_text(value, root=PROJECT_ROOT):
    text = str(value).replace(str(root), ".")
    # External home/temporary paths do not belong in shared output.
    return re.sub(r"/(?:Users|home|private|tmp|var/folders)/[^\s\"'<>]*", "<local-path>", text)


def portable_value(value):
    if isinstance(value, dict):
        return {portable_text(k): portable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_value(v) for v in value]
    return portable_text(value) if isinstance(value, (str, Path)) else value


def display_command(command):
    # Use the active interpreter to execute; display a portable invocation.
    return ["python", *[portable_text(arg) for arg in command[1:]]]
