"""Clean export of a committed revision, never a working-directory ZIP."""
import argparse
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def forbidden(name):
    parts = PurePosixPath(name).parts
    return (any(p in {'.git', '.pytest_cache', '.cache', '.ruff_cache', '.mypy_cache', '.tox', '__pycache__', '.ipynb_checkpoints', 'models'}
                or p.startswith('.venv') for p in parts)
            or name.endswith(('.pyc', '.pyo', '.gguf', '.safetensors', '.pt', '.pth', '.pte', '.bin', '.DS_Store'))
            or any(p == '.env' or (p.startswith('.env.') and p != '.env.example') for p in parts))


def export(root, output, revision='HEAD'):
    root, output = Path(root), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Export already exists: ' + str(output))
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=root, text=True).strip()
    names = subprocess.check_output(['git', 'ls-tree', '-rz', '--name-only', commit], cwd=root).decode().split('\0')
    bad = [name for name in names if name and forbidden(name)]
    if bad:
        raise ValueError('Committed source contains excluded artifacts: ' + ', '.join(bad))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'archive', '--format=zip', '--prefix=local-ai-inference/',
                    '--output=' + str(output), commit], cwd=root, check=True)
    with zipfile.ZipFile(output) as archive:
        if any(forbidden(name) for name in archive.namelist()):
            raise ValueError('Export verification failed')
    return commit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', default='HEAD')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print('Exported committed revision:', export(root, args.output, args.revision))
    print('Uncommitted changes are not included. Publication rights must be resolved separately.')
