"""Preview or launch local Nemotron with recorded scheduler and hybrid-state settings."""
import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.serving.launch import launch, launch_command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=4)
    parser.add_argument('--prefix-cache', action='store_true')
    parser.add_argument('--quantization', choices=['awq'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    options = dict(dtype='float16', context=args.context, max_num_seqs=args.max_num_seqs,
                   prefix_cache=args.prefix_cache, quantization=args.quantization,
                   trust_remote_code=True, mamba_ssm_cache_dtype='float32')
    if args.execute:
        return launch('vllm', args.snapshot, args.metadata, **options)
    print(shlex.join(launch_command('vllm', args.snapshot, **options)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
