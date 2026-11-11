"""Tables retain actual counts, missing values and the original result payload."""
from copy import deepcopy
from html import unescape
import json
import re

from clinical_trial_mvp.tutorial.result_display import result_table


def test_result_tables_preserve_payload_and_distinguish_phase_denominators():
    result=dict(experiment='FP16/cache-on',status='completed',session_id='current',
        quality=dict(cases=4,sql_executed=3,result_equivalent=2,reference_scored=4),
        serving=dict(ttft_p50_seconds=.1,e2e_p50_seconds=1.25),memory=dict(peak_bytes=None),
        cache_study={'phases':{'cold':dict(requests=20,sql_executed=15,result_equivalent=10,reference_scored=20,ttft_p50_seconds=2.019),
                               'warm':dict(requests=20,sql_executed=15,result_equivalent=10,reference_scored=20,ttft_p50_seconds=.083)}})
    before=deepcopy(result);text=result_table(result)
    assert '| cold | completed | 15/20 | 10/20 | 2.019' in text
    assert '| warm | completed | 15/20 | 10/20 | 0.083' in text
    assert '| 3/4 | 2/4 |' in text
    payload=json.loads(unescape(re.search(r'<pre>(.*?)</pre>',text,re.S).group(1)))
    assert payload==result==before


def test_failed_and_missing_results_do_not_gain_success_or_zero_metrics():
    text=result_table(dict(status='failed',reason='No response',quality=None,serving=None))
    visible=text.split('<details>')[0]
    assert '| failed |' in visible and '0.000' not in visible
    assert 'No response' in visible


def test_native_tables_keep_startup_inference_and_rss_scopes_separate():
    text=result_table(dict(stage='run',status='completed',startup_seconds=1.688,single_request_seconds=.876,
        native_observer=dict(first_token_ms=1195,inference_start_ms=1000),
        memory=dict(peak_bytes=2**30,scope='whole process RSS; not GPU memory')))
    assert '| 0.195 | 0.876 | 1.688 | 1.000 |' in text
    assert 'whole process RSS; not GPU memory' in text


def test_probe_table_does_not_imply_inference_and_shows_compatibility_attempts():
    text=result_table(dict(status='blocked_compatibility',native_inference_run=False,
        attempts=[dict(model_class='nemotron_h',accepted=False,reason='unsupported')]))
    assert '| nemotron_h | No | unsupported |' in text
    assert 'Native inference run: **No**' in text
    assert '0.000' not in text


def test_evidence_backed_tables_link_raw_records_without_repeating_them():
    result = dict(status='completed', session_evidence_path='experiment-tracker/mvp4/workflow/example.json',
                  quality=dict(cases=4,sql_executed=3,result_equivalent=2,reference_scored=4),
                  records={'serving':[{'content':'verbose-token-stream'}]*20})
    before = deepcopy(result)
    text = result_table(result)
    assert '| 3/4 | 2/4 |' in text
    assert '[Full request and result records](../experiment-tracker/mvp4/workflow/example.json)' in text
    assert 'verbose-token-stream' not in text and '<details>' not in text
    assert result == before
