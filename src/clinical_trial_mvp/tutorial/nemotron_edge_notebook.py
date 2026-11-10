"""Nemotron-H compatibility study; no fabricated native inference measurements."""
import hashlib
import json
from pathlib import Path

from .navigation import workflow_intro

FILENAME = '06_nemotron_executorch.ipynb'
RECORD = 'experiment-tracker/tutorial/edge/nemotron-4b-compatibility-20261109.json'
RECORD_SHA256 = '8ac204288d5251edc9a57419580e411c3cc1b54c282452605e6cf955c70a508f'


def recorded_probe(root):
    raw = (Path(root) / RECORD).read_bytes()
    if hashlib.sha256(raw).hexdigest() != RECORD_SHA256:
        raise ValueError('Nemotron compatibility evidence checksum mismatch')
    result = json.loads(raw)
    if (result['status'] != 'blocked_compatibility' or result['native_inference_run']
        or result['model_id'] != 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
        or result['model_revision'] != 'dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f'):
        raise ValueError('Expected the pinned Nemotron compatibility probe, without inference claims')
    return dict(result, label='recorded local compatibility probe — not a native inference run',
                source=RECORD, source_sha256=RECORD_SHA256)


def notebook_document():
    from clinical_trial_mvp.tutorial.edge_chapters import raw_document
    from clinical_trial_mvp.tutorial.navigation import apply_chapter_prose
    return apply_chapter_prose(raw_document('nemotron'), FILENAME)
