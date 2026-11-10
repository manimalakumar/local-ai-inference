#!/usr/bin/env bash
# Manual endpoint checks only; notebooks 01–04 launch their own servers.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$TASK_REPO_ROOT"
exec "${CONTROL_PYTHON:-python3}" "$TASK_REPO_ROOT/scripts/dell/serve.py" "$@"
