"""Validate chapter schemas/links and execute replay or plan in real CPU kernels."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.workflow_notebook import FILENAMES
from clinical_trial_mvp.tutorial.serving_notebook import FILENAME as FOUNDATION
from clinical_trial_mvp.tutorial.edge_notebook import FILENAME as EDGE


def local_links(document, path):
    for cell in document['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        for link in re.findall(r'\]\(([^)]+)\)', ''.join(cell['source'])):
            if link.startswith(('https:', 'http:', '#')):
                continue
            target = path.parent / unquote(link.split('#')[0])
            if not target.exists():
                raise ValueError(f'Broken chapter link in {path.name}: {link}')


def main(argv=None):
    import nbformat
    from nbclient import NotebookClient
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['replay', 'plan'], default='replay')
    parser.add_argument('--write-replay', action='store_true', help='Save freshly executed replay outputs into the learner chapters')
    args = parser.parse_args(argv)
    if args.write_replay and args.mode != 'replay':
        parser.error('--write-replay requires replay mode')
    for name in (FOUNDATION, *FILENAMES.values(), EDGE):
        path = ROOT / 'notebooks' / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        local_links(notebook, path)
        client = NotebookClient(notebook, timeout=180, kernel_name='python3',
                                resources={'metadata': {'path': str(path.parent)}})
        client.execute(env={**os.environ, 'LOCAL_AI_LAB_MODE': args.mode,
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        nbformat.validate(notebook)
        text = json.dumps(notebook)
        if args.mode == 'replay' and (('recorded Mac CPU portability measurement' if name == EDGE else 'recorded Dell measurement') not in text):
            raise ValueError('Replay did not expose recorded-measurement labels: ' + name)
        if args.write_replay:
            nbformat.write(notebook, path)
        print(f'{name}: schema, links and {args.mode} execution passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
