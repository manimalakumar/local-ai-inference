"""Prepare notebook 00's local GGUF baseline; print a foreground server command."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from clinical_trial_mvp.tutorial.foundation_study import local_template


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf',required=True,type=Path)
    parser.add_argument('--server-python',required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'tutorial/config/local-serving.local.json')
    parser.add_argument('--mode',choices=['live','plan'],default='live')
    args=parser.parse_args(argv)
    if args.config.exists():parser.error('Local config exists; edit it or choose a new --config path')
    gguf=args.gguf.expanduser().resolve()
    if args.mode=='live' and (not gguf.is_file() or gguf.suffix.lower()!='.gguf'):
        parser.error('Supply the downloaded Q4_K_M GGUF file')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    metadata=ROOT/f'experiment-tracker/tutorial/foundation/provenance/{stamp}-llama.json'
    cfg=local_template();cfg['mode']=args.mode
    cfg['servers']['nemotron']['provenance']=str(metadata)
    args.config.parent.mkdir(parents=True,exist_ok=True)
    with args.config.open('x') as stream:json.dump(cfg,stream,indent=2);stream.write('\n')
    command=[str(Path(args.server_python).expanduser()),str(ROOT/'scripts/tutorial/launch_local_foundation.py'),
             '--gguf',str(gguf),'--metadata',str(metadata)]
    print('Local configuration:',args.config)
    print('Start this command in a separate terminal:');print(shlex.join(command))
    print('Wait for readiness, restart notebook 00 and Run All. Stop the server with Ctrl+C when finished.')
    return 0


if __name__=='__main__':raise SystemExit(main())
