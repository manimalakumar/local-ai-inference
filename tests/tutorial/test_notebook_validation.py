"""Make source drift actionable without weakening the notebook validation gate."""
import importlib.util
from copy import deepcopy
import pytest
from clinical_trial_mvp.tutorial.harness import ROOT


def test_source_drift_reports_notebook_and_first_mismatched_cell():
    path = ROOT/'scripts/tutorial/validate_notebooks.py'
    spec = importlib.util.spec_from_file_location('notebook_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {'cells':[{'id':'preflight-check','cell_type':'code','source':['study.preflight()\n']}]}
    actual = deepcopy(expected)
    actual['cells'][0]['source'] = ['subprocess.run(["docker", "info"])\n']
    with pytest.raises(ValueError, match='example.ipynb; cell 1: found preflight-check'):
        module.validate_sources(actual, expected, 'example.ipynb')
    actual['cells'][0]['source'] = expected['cells'][0]['source']
    actual['cells'][0]['outputs'] = [{'output_type':'stream','name':'stdout','text':['genuine result']}]
    module.validate_sources(actual, expected, 'example.ipynb')
