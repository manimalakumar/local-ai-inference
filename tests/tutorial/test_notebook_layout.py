"""Publish only the learner chapters; retain measurement evidence outside that folder."""
from pathlib import Path
import subprocess

from clinical_trial_mvp.tutorial.harness import ROOT
from clinical_trial_mvp.tutorial.workflow_notebook import FILENAMES
from clinical_trial_mvp.tutorial.serving_notebook import FILENAME as FOUNDATION
from clinical_trial_mvp.tutorial.edge_notebook import FILENAME as EDGE
from clinical_trial_mvp.tutorial.nemotron_edge_notebook import FILENAME as NEMOTRON_EDGE


def test_repository_has_one_notebook_folder_with_only_current_chapters():
    listed = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--', '*.ipynb'],
        cwd=ROOT, text=True).splitlines()
    present = {name for name in listed if (ROOT/name).is_file()}
    expected = {'notebooks/' + name for name in (FOUNDATION, *FILENAMES.values(), EDGE, NEMOTRON_EDGE)}
    assert present == expected
    assert {str(Path(name).parent) for name in present} == {'notebooks'}
