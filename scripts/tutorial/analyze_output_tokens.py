"""Reproduce the token-length audit from checksum-verified recorded target requests."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.replay import RecordedStudy
from clinical_trial_mvp.tutorial.token_analysis import analyze, table


def main():
    replay = RecordedStudy(ROOT)
    results = {name: replay.result('vllm', 'qwen', name)
               for name in ('fp16_cache_on', 'awq_w4a16_cache_on')}
    analysis = analyze(results)
    analysis['label'] = 'recorded Dell measurement'
    (ROOT / 'tutorial/output_token_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(table(analysis))


if __name__ == '__main__':
    main()
