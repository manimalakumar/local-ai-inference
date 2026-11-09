"""Adapt the prior serving foundation into chapter 00; preserve archived evidence."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.serving_notebook import FILENAME, notebook_document

from clinical_trial_mvp.tutorial.notebook_io import write_notebook

if __name__ == '__main__':
    path = ROOT / 'notebooks' / FILENAME
    write_notebook(path, notebook_document())
    print(path.name)
