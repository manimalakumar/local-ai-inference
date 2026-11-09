"""Explicit, checksum-verified Dell evidence. Never a fallback for a failed live run."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


LABEL = 'recorded Dell measurement'
MANIFEST = 'notebooks/history/dell-2026-11-07/MANIFEST.json'


class RecordedStudy:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / 'tutorial/recorded_runs.json').read_text())
        self.hashes = json.loads((self.root / MANIFEST).read_text())['files_sha256']

    def read(self, path):
        if path not in self.hashes:
            raise ValueError('Replay record is not in the preserved Dell manifest: ' + path)
        data = (self.root / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != self.hashes[path]:
            raise ValueError('Recorded evidence checksum mismatch: ' + path)
        result = json.loads(data)
        result['_recorded'] = dict(label=LABEL, source=path, source_sha256=self.hashes[path],
                                   origin='Dell Quadro RTX 5000; November 6–7, 2026')
        return result

    def result(self, runtime, model, name):
        path = self.index['studies'][f'{runtime}-{model}'].get(name)
        if path is None:
            return dict(experiment=f'{model}/{name}', status='unavailable', evidence='none',
                        reason='No such measurement was recorded in the Dell archive; no values estimated.')
        return self.read(path)

    def failure(self, name):
        return self.read(self.index['failures'][name])


def display_result(result):
    """Label presentation copies; leave original measurement fields unchanged."""
    result = deepcopy(result)
    if result.get('_recorded'):
        result['experiment'] = result.get('experiment', 'smoke') + ' · ' + LABEL
    return result


def evidence_links(result):
    recorded = result.get('_recorded')
    return ('**' + LABEL + '** — [' + recorded['source'] + '](../' + recorded['source'] + ')') if recorded else ''
