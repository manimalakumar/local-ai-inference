"""Run llama-cpp-python on a local GGUF, preserving actual launch provenance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf',required=True,type=Path)
    parser.add_argument('--metadata',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.metadata.exists():parser.error('Choose a new metadata path')
    gguf=args.gguf.expanduser().resolve()
    if not gguf.is_file() or gguf.suffix.lower()!='.gguf':parser.error('Supply a local GGUF file')
    try:
        with socket.create_connection(('127.0.0.1',8080),timeout=1):parser.error('Port 8080 is occupied')
    except ConnectionRefusedError:pass
    # The server uses the embedded GGUF tokenizer and template.
    digest=hashlib.sha256()
    with gguf.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest() != 'be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2':
        parser.error('This starter requires the tutorial Nemotron Q4_K_M artifact; GGUF SHA-256 differs')
    command=[sys.executable,'-m','llama_cpp.server','--model',str(gguf),
        '--host','127.0.0.1','--port','8080','--model_alias','clinical-trial-local',
        '--n_ctx','8192','--n_gpu_layers','0','--seed','42']
    version=importlib.metadata.version('llama-cpp-python')
    with subprocess.Popen(command) as process:
        metadata=dict(runtime='llama_cpp',model='clinical-trial-local',engine_version=version,
            model_sha256=digest.hexdigest(),hardware=platform.platform()+' / '+platform.machine()+' / CPU',
            server_pid=process.pid,evidence='hardware',command=command,
            launch_settings=dict(context=8192,dtype='Q4_K_M',gpu_layers=0,seed=42))
        try:
            args.metadata.parent.mkdir(parents=True,exist_ok=True)
            with args.metadata.open('x') as stream:json.dump(metadata,stream,indent=2);stream.write('\n')
            return process.wait()
        except BaseException:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise


if __name__=='__main__':raise SystemExit(main())
