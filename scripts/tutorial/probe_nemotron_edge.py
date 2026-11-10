"""Check the exact Dell Nemotron against the installed ExecuTorch LLM recipe.

Downloads config.json only. This probes export configuration, not graph capture,
weight conversion, inference, or general feasibility with a custom adapter.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16'
REVISION = 'dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f'


def probe(config):
    from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType
    from omegaconf import OmegaConf
    if config.get('model_type') != 'nemotron_h' or config.get('hidden_size') != 3136:
        raise ValueError('Expected the pinned 4B Nemotron hybrid configuration')
    attempts = []
    # Check both the HF architecture identifier and an explicit 4B recipe name.
    for candidate in ('nemotron_h', 'nemotron_3_nano_4b'):
        try:
            OmegaConf.merge(OmegaConf.structured(LlmConfig), {'base': {'model_class': candidate}})
        except Exception as exc:
            attempts.append({'model_class': candidate, 'accepted': False,
                             'error_type': type(exc).__name__, 'reason': str(exc)})
        else:
            attempts.append({'model_class': candidate, 'accepted': True})
    accepted = any(a['accepted'] for a in attempts)
    source = Path(inspect.getfile(ModelType)).read_bytes()
    return dict(status='recipe_available_unvalidated' if accepted else 'blocked_compatibility',
        evidence='local_export_configuration_probe',
        model_id=MODEL, model_revision=REVISION,
        architecture=config['architectures'], model_type=config['model_type'],
        hybrid_override_pattern=config['hybrid_override_pattern'],
        config=config, config_sha256=None,
        installed_model_types=[m.value for m in ModelType],
        exporter_config_source_sha256=hashlib.sha256(source).hexdigest(),
        attempts=attempts, weights_downloaded=False, graph_capture_attempted=False,
        pte_produced=False, native_inference_run=False, sql_correctness=None,
        ttft_seconds=None, end_to_end_seconds=None, peak_memory_bytes=None,
        scope='The installed standard LLM export recipe only; no claim that a custom Nemotron adapter is impossible.',
        next_steps=['Implement or obtain a validated Nemotron-H export adapter preserving Mamba state and attention caches.',
                    'Compare converted FP32 logits with the pinned Hugging Face checkpoint.',
                    'Validate torch.export graph capture, backend/operator coverage, PTE load and native generation.',
                    'Then measure SQL correctness, TTFT, request latency and process memory separately from startup.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Use an existing exact pinned config.json offline')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path to preserve previous probe evidence')
    if args.config:
        path = args.config
        identity = 'Caller-provided configuration; revision must be checked independently'
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(MODEL, 'config.json', revision=REVISION,
                    local_dir=str(ROOT / 'models/edge-nemotron-4b/source')))
        identity = 'Config downloaded from the exact Dell base revision; weights not downloaded'
    raw = path.read_bytes()
    result = probe(json.loads(raw))
    result.update(config_sha256=hashlib.sha256(raw).hexdigest(), source_identity=identity,
        measured_at=datetime.now(timezone.utc).isoformat(),
        versions={p:importlib.metadata.version(p) for p in ('executorch','torch','transformers','omegaconf')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
