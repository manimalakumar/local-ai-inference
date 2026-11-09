"""Create an ignored per-machine configuration, optionally discover NIM profiles."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.workflow import Study, template


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=['vllm', 'nim'], required=True)
    parser.add_argument('--model', choices=['qwen', 'nemotron'], required=True)
    parser.add_argument('--discover', action='store_true', help='Run NIM discovery on the Linux GPU host without starting a server')
    args = parser.parse_args(argv)
    path = ROOT / f'tutorial/config/{args.runtime}-{args.model}.local.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('x') as stream:
            config = template(args.runtime, args.model)
            config.pop('execute')  # Legacy flag is accepted, but new users choose mode explicitly.
            json.dump(config, stream, indent=2)
            stream.write('\n')
    print('Configuration:', path)
    if args.discover:
        if args.runtime != 'nim':
            parser.error('--discover requires --runtime nim')
        study = Study(args.runtime, args.model, root=ROOT)
        study.config['execute'] = True
        study.config['mode'] = 'live'
        try:
            print(study.prepare())
        except ValueError as exc:
            print(exc)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
