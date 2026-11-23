"""Keep actual JSON readable in static viewers without replacing native Jupyter data."""
from copy import deepcopy
import json
import subprocess

from IPython.core.formatters import DisplayFormatter

from clinical_trial_mvp.tutorial.notebook_display import JSON
from clinical_trial_mvp.tutorial.harness import ROOT


def test_json_display_keeps_native_data_and_metadata_with_readable_fallbacks():
    payload = {'status':'completed', 'reference_result_matches':2, 'latency':0.083,
               'missing':None, 'schema':['subjects','treatment_arm'], 'text':'μg'}
    before = deepcopy(payload)
    value = JSON(payload, expanded=True, root='result')
    data, metadata = DisplayFormatter().format(value)
    assert data['application/json'] == before
    assert json.loads(data['text/plain']) == before
    assert 'IPython.core.display.JSON object' not in data['text/plain']
    assert data['text/markdown'].startswith('```json\n')
    assert json.loads(data['text/markdown'].split('\n',1)[1].rsplit('\n',1)[0]) == before
    assert metadata['application/json']['expanded'] is True
    assert metadata['application/json']['root'] == 'result'
    assert payload == before


def test_nested_code_fences_in_model_requests_do_not_break_json_rendering():
    payload = {'messages':[{'content':'```sql\nSELECT 1;\n```'}]}
    data, _ = DisplayFormatter().format(JSON(payload))
    rendered = data['text/markdown']
    assert rendered.startswith('````json\n') and rendered.endswith('\n````')
    assert json.loads(rendered.split('\n',1)[1].rsplit('\n',1)[0]) == payload


def test_published_notebooks_include_real_json_in_static_output_views():
    count = 0
    files = subprocess.check_output(['git','ls-files','*.ipynb'],cwd=ROOT,text=True).splitlines()
    for name in files:
        document = json.loads((ROOT/name).read_text())
        for cell in document['cells']:
            for output in cell.get('outputs',[]):
                data = output.get('data',{})
                if 'application/json' not in data:
                    continue
                plain = data['text/plain']
                plain = plain if isinstance(plain,str) else ''.join(plain)
                assert json.loads(plain) == data['application/json'], (name,cell.get('id'))
                assert 'text/markdown' in data
                count += 1
    assert count >= 26
