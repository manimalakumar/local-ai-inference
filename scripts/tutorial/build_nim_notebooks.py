"""Build two additional MVP4 NIM notebooks; preserve the existing vLLM pair."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.nim_study import MODELS, template

FILENAMES = {'qwen': '03_qwen_nim.ipynb',
             'nemotron': '04_nemotron_nim.ipynb'}


def notebook_document(key):
    from clinical_trial_mvp.tutorial.workflow_notebook import notebook_document as build
    return build('nim', key)


from clinical_trial_mvp.tutorial.notebook_io import write_notebook

if __name__ == '__main__':
    for key, filename in FILENAMES.items():
        write_notebook(ROOT / 'notebooks' / filename, notebook_document(key))
        (ROOT / 'tutorial' / f'nim_{key}.example.json').write_text(json.dumps(template(key), indent=2) + '\n')
        print(filename)
