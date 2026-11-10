#!/usr/bin/env bash
# Use the notebook environment; serving engines stay in their own environments.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_NOTEBOOK_PYTHON="${NOTEBOOK_PYTHON:-$TASK_REPO_ROOT/.venv/bin/python}"
TASK_VLLM_PYTHON="${VLLM_PYTHON:-$HOME/.venv-vllm/bin/python}"
TASK_JUPYTER_PORT="${JUPYTER_PORT:-8888}"
TASK_NOTEBOOK="${1:-notebooks/01_qwen_cache_quantization.ipynb}"
cd -- "$TASK_REPO_ROOT"
if [[ ! -x "$TASK_NOTEBOOK_PYTHON" ]]; then
    echo "Notebook Python missing: $TASK_NOTEBOOK_PYTHON" >&2
    echo "Follow SETUP.md to install requirements-notebook.txt, or set NOTEBOOK_PYTHON to your existing Jupyter Python." >&2
    exit 1
fi
"$TASK_NOTEBOOK_PYTHON" - "$TASK_NOTEBOOK" "$TASK_JUPYTER_PORT" <<'PY'
import importlib.util
from pathlib import Path
import socket
import sys
if not Path(sys.argv[1]).is_file():
    raise SystemExit('Notebook missing: ' + sys.argv[1])
for name in ('jupyterlab', 'ipykernel', 'langchain_core', 'duckdb', 'mcp'):
    if importlib.util.find_spec(name) is None:
        raise SystemExit('Notebook environment missing ' + name + '; install requirements-notebook.txt using NOTEBOOK_PYTHON (see SETUP.md)')
port = int(sys.argv[2])
if not 1 <= port <= 65535:
    raise SystemExit('JUPYTER_PORT must be between 1 and 65535')
with socket.socket() as probe:
    if probe.connect_ex(('127.0.0.1', port)) == 0:
        raise SystemExit('Jupyter port is occupied; use the existing server or choose JUPYTER_PORT=8889')
PY
# Register the exact notebook interpreter, regardless of the active shell venv.
"$TASK_NOTEBOOK_PYTHON" -m ipykernel install --user --name clinical-trial-mvp4 --display-name 'Clinical Trial MVP4 (notebook)'
# Seed only a missing vLLM config. Preserve every existing machine setting.
"$TASK_NOTEBOOK_PYTHON" - "$TASK_REPO_ROOT" "$TASK_VLLM_PYTHON" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
for model in ('qwen', 'nemotron'):
    path = root / f'tutorial/config/vllm-{model}.local.json'
    if path.exists():
        print('Keeping existing configuration:', path)
        continue
    cfg = json.loads((root / f'tutorial/config/vllm-{model}.example.json').read_text())
    cfg['vllm_python'] = str(Path(sys.argv[2]).expanduser())
    with path.open('x') as stream:
        json.dump(cfg, stream, indent=2)
        stream.write('\n')
    print('Created configuration; review checkpoint revisions and AWQ derivation:', path)
PY
# Clear stale validation mode; the visible notebook MODE remains authoritative.
unset LOCAL_AI_LAB_MODE
printf '%s\n' "Notebook Python: $TASK_NOTEBOOK_PYTHON" "GPU runtime Python (separate): $TASK_VLLM_PYTHON" \
    'Select kernel: Clinical Trial MVP4 (notebook)' \
    'Notebooks 01–04 own their servers; do not start a manual server before Run All.' \
    'Keep this terminal open. Stop Jupyter with Ctrl+C after saving and notebook cleanup.'
exec "$TASK_NOTEBOOK_PYTHON" -m jupyter lab "$TASK_NOTEBOOK" \
    --ip=127.0.0.1 --port="$TASK_JUPYTER_PORT" --ServerApp.port_retries=0 --no-browser
