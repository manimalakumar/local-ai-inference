"""Qwen3-4B CPU portability, with the original SmolLM2 record retained."""
import hashlib
import json

FILENAME = '05_edge_portability_executorch.ipynb'
RECORD = 'experiment-tracker/tutorial/edge/20261029T161510_034051Z.json'
# Frozen historical evidence, never a latest-file glob or a substitute inference run.
RECORD_SHA256 = '04076d8f693e18a503335da98f296d5af854a539389027f705b8b602f2d36982'


def recorded_edge(root):
    path = root / RECORD
    if hashlib.sha256(path.read_bytes()).hexdigest() != RECORD_SHA256:
        raise ValueError('ExecuTorch historical evidence checksum mismatch')
    result = json.loads(path.read_text())
    if result['status'] != 'completed' or result['evidence'] != 'hardware':
        raise ValueError('Expected completed hardware portability evidence')
    return {'label': 'recorded Mac CPU portability measurement — October 29, 2026',
            'source': RECORD, 'source_sha256': RECORD_SHA256, **result}


QWEN_RECORD = 'experiment-tracker/tutorial/edge/qwen3-4b-run-20261109T195845_533008Z.json'
QWEN_RECORD_SHA256 = '9716935d2ea971f14927b3f6b157a816dc45a1d1e15d4de2a4aec83362a43014'
QWEN_SQL_RECORD = 'experiment-tracker/tutorial/edge/qwen3-4b-sql-smoke-20261109.json'
QWEN_SQL_SHA256 = 'da3a170b16581d07e6a6c4526f4527d156d19cb3eaccbf6dd062c280118ccc52'


def recorded_qwen_edge(root):
    from pathlib import Path
    root=Path(root)
    def checked(path,digest):
        raw=(root/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Qwen ExecuTorch evidence checksum mismatch')
        return json.loads(raw)
    record=checked(QWEN_RECORD,QWEN_RECORD_SHA256)
    smoke=checked(QWEN_SQL_RECORD,QWEN_SQL_SHA256)
    if (record['model_id']!='Qwen/Qwen3-4B' or record['status']!='completed' or record['evidence']!='hardware'
        or smoke['source_sha256']!=QWEN_RECORD_SHA256 or not smoke['matches_dell_base_snapshot']):
        raise ValueError('Expected verified native Qwen3-4B hardware evidence, not a fallback model')
    observer=record['native_observer']
    ttft=(observer['first_token_ms']-observer['inference_start_ms'])/1000
    return {**record,'label':'recorded Mac CPU portability measurement — Qwen3-4B, November 9, 2026',
            'source':QWEN_RECORD,'source_sha256':QWEN_RECORD_SHA256,
            'max_context_length':record['context'],'limitations':record['limitation'],
            'native_ttft_seconds':ttft,'sql_smoke':smoke}


def notebook_document():
    from clinical_trial_mvp.tutorial.edge_chapters import raw_document
    from clinical_trial_mvp.tutorial.navigation import apply_chapter_prose
    return apply_chapter_prose(raw_document('qwen'), FILENAME)
