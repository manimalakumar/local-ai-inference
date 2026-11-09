"""Adapt the existing CPU portability case into optional chapter 05."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.edge_notebook import FILENAME, notebook_document

if __name__ == '__main__':
    path = ROOT / 'notebooks' / FILENAME
    path.write_text(json.dumps(notebook_document(), indent=1) + '\n')
    print(path.name)
