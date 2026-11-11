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
from clinical_trial_mvp.tutorial.nemotron_edge_notebook import FILENAME as NEMOTRON_EDGE, notebook_document as nemotron_edge_document
from clinical_trial_mvp.tutorial.notebook_io import sources, source_text
from clinical_trial_mvp.tutorial.serving_notebook import notebook_document as foundation_document
from clinical_trial_mvp.tutorial.edge_notebook import notebook_document as edge_document
from clinical_trial_mvp.tutorial.workflow_notebook import notebook_document as workflow_document


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


def validate_sources(notebook, expected, name):
    if sources(notebook) == sources(expected):
        return
    actual_cells, expected_cells = notebook['cells'], expected['cells']
    for index in range(max(len(actual_cells), len(expected_cells))):
        actual = actual_cells[index] if index < len(actual_cells) else None
        wanted = expected_cells[index] if index < len(expected_cells) else None
        if actual is None or wanted is None or (actual['cell_type'], source_text(actual)) != (wanted['cell_type'], source_text(wanted)):
            actual_id = actual.get('id', '?') if actual else '<missing>'
            wanted_id = wanted.get('id', '?') if wanted else '<none>'
            raise ValueError(f'Chapter source differs from builder: {name}; cell {index + 1}: '
                             f'found {actual_id}, expected {wanted_id}. Update the builder/prose source '
                             'and regenerate with notebook_io.write_notebook to preserve unchanged live outputs.')


def main(argv=None):
    import nbformat
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['replay', 'plan'], default='replay')
    parser.add_argument('--check-only', action='store_true', help='Check schema, local links and builder sources; start no kernels')
    parser.add_argument('--write-replay', action='store_true', help='Save freshly executed replay outputs into the learner chapters')
    args = parser.parse_args(argv)
    if args.write_replay and (args.mode != 'replay' or args.check_only):
        parser.error('--write-replay requires replay execution, without --check-only')
    for name in (FOUNDATION, *FILENAMES.values(), EDGE, NEMOTRON_EDGE):
        path = ROOT / 'notebooks' / name
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        local_links(notebook, path)
        expected = (foundation_document() if name == FOUNDATION else edge_document() if name == EDGE else
                    nemotron_edge_document() if name == NEMOTRON_EDGE else
                    workflow_document(*next(key for key, filename in FILENAMES.items() if filename == name)))
        validate_sources(notebook, expected, name)
        if args.check_only:
            print(f'{name}: schema, links and builder sources passed (no execution)', flush=True)
            continue
        from nbclient import NotebookClient
        client = NotebookClient(notebook, timeout=180, kernel_name='python3',
                                resources={'metadata': {'path': str(path.parent)}})
        client.execute(env={**os.environ, 'LOCAL_AI_LAB_MODE': args.mode,
                            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        nbformat.validate(notebook)
        text = json.dumps(notebook)
        label = ('Loading results.' if name == FOUNDATION else 'recorded local compatibility probe' if name == NEMOTRON_EDGE else
                 'recorded Mac CPU portability measurement' if name == EDGE else 'recorded Dell measurement')
        if args.mode == 'replay' and label not in text:
            raise ValueError('Replay did not expose recorded-measurement labels: ' + name)
        if args.write_replay:
            nbformat.write(notebook, path)
        print(f'{name}: schema, links and {args.mode} execution passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
