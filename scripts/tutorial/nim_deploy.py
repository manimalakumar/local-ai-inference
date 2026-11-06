"""Configure, discover or deploy a model-specific NIM; previews are the default."""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.nim_study import commands, deploy, discover, safe_text, template


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'discover', 'deploy'])
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--model', choices=['qwen', 'nemotron'])
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.action == 'init':
            if not args.model:
                raise ValueError('init requires --model qwen or --model nemotron')
            args.profile.parent.mkdir(parents=True, exist_ok=True)
            with args.profile.open('x') as output:
                json.dump(template(args.model), output, indent=2)
                output.write('\n')
            print('Profile created. Set the image digest, local snapshot and exact model revision.')
            return 0
        profile = json.loads(args.profile.read_text())
        print(shlex.join(commands(profile, root=ROOT)[args.action]))
        if not args.execute:
            print('Preview only; add --execute on the Linux GPU host.')
            return 0
        if args.action == 'discover':
            profile = discover(profile, root=ROOT)
            args.profile.write_text(json.dumps(profile, indent=2) + '\n')
            print('Runnable profile IDs:', profile['runnable_profile_ids'])
            print('Choose a compatible profile_id explicitly before deployment.')
            if not profile['discovery_succeeded']:
                print('Discovery failed; sanitized details were saved in the local profile.')
                return 1
            return 0
        if args.metadata is None:
            raise ValueError('Live deployment requires a new --metadata path')
        result = deploy(profile, args.metadata, root=ROOT, timeout=args.timeout)
        print(result['status'], result.get('reason', ''))
        return 0 if result['status'] == 'ready' else 1
    except (OSError, ValueError, KeyError) as exc:
        parser.error(safe_text(exc))


if __name__ == '__main__':
    raise SystemExit(main())
