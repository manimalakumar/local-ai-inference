"""Regenerate source without discarding outputs of unchanged code cells."""
from collections import defaultdict, deque
from copy import deepcopy
import json
from pathlib import Path


def source_text(cell):
    source = cell['source']
    return source if isinstance(source, str) else ''.join(source)


def sources(document):
    return [(cell['cell_type'], source_text(cell)) for cell in document['cells']]


def write_notebook(path, document):
    """Keep outputs only for byte-identical code, in occurrence order.

    Changed code starts unexecuted. Historical archives are never passed here.
    Output provenance remains historical until a fresh execution replaces it.
    """
    path = Path(path)
    document = deepcopy(document)
    previous = defaultdict(deque)
    if path.exists():
        old = json.loads(path.read_text())
        for cell in old['cells']:
            if cell['cell_type'] == 'code':
                previous[source_text(cell)].append(cell)
    for cell in document['cells']:
        matches = previous[source_text(cell)] if cell['cell_type'] == 'code' else ()
        if matches:
            old = matches.popleft()
            cell['outputs'] = deepcopy(old.get('outputs', []))
            cell['execution_count'] = old.get('execution_count')
            if 'execution' in old.get('metadata', {}):
                cell['metadata']['execution'] = deepcopy(old['metadata']['execution'])
    path.write_text(json.dumps(document, indent=1) + '\n')
