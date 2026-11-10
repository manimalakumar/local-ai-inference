"""Prepare chapter 00's Qwen GPU baseline; server launch is an explicit command."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.foundation_study import template

MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=['vllm', 'sglang'], default='vllm')
    parser.add_argument('--mode', choices=['live', 'plan'], default='live')
    parser.add_argument('--snapshot', type=Path, default=ROOT / 'models/qwen3-4b-frozen')
    parser.add_argument('--server-python', required=True, help='Python in your separate GPU environment')
    parser.add_argument('--download', action='store_true', help='Download the exact Qwen base revision')
    parser.add_argument('--config', type=Path, default=ROOT / 'tutorial/config/local-serving.local.json')
    args = parser.parse_args(argv)
    if args.config.exists():
        parser.error('Configuration already exists; edit it or choose a new --config path')
    snapshot = args.snapshot.expanduser().resolve()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=str(snapshot))
    elif args.mode == 'live' and not any(snapshot.glob('*.safetensors')):
        parser.error('Supply the Qwen HF snapshot with weights, or use --download')
    if args.mode == 'live':
        model_config = json.loads((snapshot / 'config.json').read_text())
        if (model_config.get('model_type') != 'qwen3' or model_config.get('hidden_size') != 2560
            or model_config.get('num_hidden_layers') != 36 or model_config.get('quantization_config')):
            parser.error('This starter requires the unquantized Qwen3-4B base snapshot')
    cfg = template()
    cfg['mode'] = args.mode
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    metadata = ROOT / f'experiment-tracker/tutorial/foundation/provenance/{stamp}-{args.runtime}.json'
    cfg['servers']['qwen'].update(runtime=args.runtime, provenance=str(metadata))
    args.config.parent.mkdir(parents=True, exist_ok=True)
    with args.config.open('x') as output:
        json.dump(cfg, output, indent=2)
        output.write('\n')
    command = [str(Path(args.server_python).expanduser()), str(ROOT / 'scripts/tutorial/launch_foundation.py'),
        '--runtime', args.runtime, '--snapshot', str(snapshot), '--dtype', 'float16',
        '--context', '8192', '--metadata', str(metadata)]
    print('Saved configuration:', args.config)
    print('Start this server in a separate terminal and wait until it is ready:')
    print(shlex.join(command))
    print('Then restart the notebook kernel and Run All. Stop your server with Ctrl+C when finished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
