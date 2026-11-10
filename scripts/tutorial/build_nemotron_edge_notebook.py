"""Build chapter 06, preserving existing outputs of unchanged code cells."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.nemotron_edge_notebook import FILENAME, notebook_document
from clinical_trial_mvp.tutorial.notebook_io import write_notebook

if __name__ == '__main__':
    write_notebook(ROOT / 'notebooks' / FILENAME, notebook_document())
    print(FILENAME)
