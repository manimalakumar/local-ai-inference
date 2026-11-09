"""Build the standalone local AI optimization tutorial, unexecuted by default."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    'nemotron': {
        'label': 'Nemotron 3 Nano 4B',
        'model_id': 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
        'fp16_snapshot': 'models/nemotron-3-nano-4b-frozen',
        'awq_snapshot': 'models/nemotron-3-nano-4b-awq-4bit-frozen',
        'base_revision': 'SET_EXACT_NEMOTRON_BASE_REVISION_AND_AWQ_DERIVATION',
        'awq_source': 'SET_VERIFIED_NEMOTRON_3_NANO_4B_AWQ_SOURCE',
        'filename': '02_nemotron_optimization.ipynb',
        'description': 'The SQL generator in this notebook is **Nemotron 3 Nano 4B** '
            '([base checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16)). '
            'It generates clinical SQL from the fixed schema and question through vLLM. '
            'The unquantized BF16 source weights are loaded as FP16 for this experiment; '
            'the comparison requires an AWQ W4A16 derivative of that exact base revision. '
            'Set its verified AWQ source and local checkpoint path before running; no '
            'Nemotron AWQ artifact or GPU compatibility is assumed.',
    },
    'qwen': {
        'label': 'Qwen3-4B',
        'model_id': 'Qwen/Qwen3-4B',
        'fp16_snapshot': 'models/qwen3-4b-frozen',
        'awq_snapshot': 'models/qwen3-4b-awq-4bit-frozen',
        'base_revision': 'SET_EXACT_QWEN3_4B_BASE_REVISION_AND_AWQ_DERIVATION',
        'awq_source': 'Qwen/Qwen3-4B-AWQ',
        'filename': '01_qwen_cache_quantization.ipynb',
        'description': 'The SQL generator in this notebook is **Qwen3-4B** '
            '([base checkpoint](https://huggingface.co/Qwen/Qwen3-4B)). '
            'It generates clinical SQL from the fixed schema and question through vLLM. '
            'The baseline loads the unquantized checkpoint as FP16; the W4A16 candidate is '
            '[Qwen/Qwen3-4B-AWQ](https://huggingface.co/Qwen/Qwen3-4B-AWQ). '
            'Pin compatible base/AWQ revisions and set the local checkpoint paths before running.',
    },
}


def notebook_document(model='qwen'):
    sys.path.insert(0, str(ROOT / 'src'))
    from clinical_trial_mvp.tutorial.workflow_notebook import notebook_document as build
    return build('vllm', model)


if __name__ == '__main__':
    for model, selected in MODELS.items():
        path = ROOT / 'notebooks' / selected['filename']
        path.write_text(json.dumps(notebook_document(model), indent=1) + '\n')
        print(path.name)
