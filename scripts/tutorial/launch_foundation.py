"""Start the baseline's foreground GPU server after checking its port/metadata."""
import argparse
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main(argv=None, *, launcher=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', required=True, choices=['vllm', 'sglang'])
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--metadata', required=True, type=Path)
    parser.add_argument('--dtype', choices=['float16', 'bfloat16'], default='float16')
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('Invalid port')
    if args.metadata.exists():
        parser.error('Metadata already exists; choose a new file for this server launch')
    try:
        with socket.create_connection(('127.0.0.1', args.port), timeout=1):
            parser.error('Port already occupied; stop your previous server or choose another port before launch')
    except ConnectionRefusedError:
        pass
    if launcher is None:
        from clinical_trial_mvp.serving.launch import launch
        launcher = launch
    return launcher(args.runtime, args.snapshot, args.metadata, dtype=args.dtype,
                    context=args.context, port=args.port, prefix_cache=False)


if __name__ == '__main__':
    raise SystemExit(main())
