"""Foreground manual vLLM/SGLang server. Standard library only in the controller.

The engine runs in its own exact Python environment. No installs, symlinks,
downloads or external process termination. Not used by optimization notebook Run All.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[2]


def snapshot_fingerprint(folder):
    # Same file-selection and canonical JSON contract as serving.benchmark.
    # Kept stdlib-only so the engine environment needs no notebook/agent packages.
    files = sorted(p for p in folder.rglob('*') if p.is_file() and
        p.suffix in {'.safetensors', '.bin', '.json', '.model', '.jinja', '.txt'} and
        '.cache' not in p.relative_to(folder).parts)
    if not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError('Local checkpoint contains no Hugging Face weights')
    hashes = {}
    for path in files:
        with path.open('rb') as stream:
            hashes[str(path.relative_to(folder))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def command_for(args, snapshot, python):
    common = ['--host', '127.0.0.1', '--port', str(args.port),
              '--served-model-name', 'clinical-trial-mvp3', '--dtype', 'float16']
    if args.awq:
        common += ['--quantization', 'awq']
    if args.runtime == 'vllm':
        command = [str(python), '-m', 'vllm.entrypoints.cli.main', 'serve', str(snapshot), *common,
                   '--max-model-len', str(args.context), '--tensor-parallel-size', '1',
                   '--generation-config', 'vllm',
                   '--enable-prefix-caching' if args.prefix_cache else '--no-enable-prefix-caching']
        if args.model == 'nemotron':
            command += ['--trust-remote-code', '--mamba-ssm-cache-dtype', 'float32']
        return command
    if args.model == 'nemotron':
        raise ValueError('The Nemotron SGLang path is not validated here; use notebook 02 through vLLM')
    return [str(python), '-m', 'sglang.launch_server', '--model-path', str(snapshot), *common,
            '--context-length', str(args.context), '--tp-size', '1', '--sampling-defaults', 'openai',
            *([] if args.prefix_cache else ['--disable-radix-cache'])]


def stop_owned(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runtime', choices=['vllm', 'sglang'])
    parser.add_argument('--model', choices=['qwen', 'nemotron'], default='qwen')
    parser.add_argument('--python', type=Path, help='Exact engine Python; defaults to ~/.venv-<runtime>/bin/python')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--awq', action='store_true')
    parser.add_argument('--prefix-cache', action='store_true')
    parser.add_argument('--port', type=int, default=8001, help='Manual endpoint; notebooks own port 8000')
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--readiness-timeout', type=float, default=1800)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.context < 1 or not 0 < args.readiness_timeout <= 7200:
        parser.error('Use a valid port, positive context and readiness timeout <= 7200 seconds')
    if args.model == 'nemotron' and args.awq:
        parser.error('No validated Nemotron AWQ artifact is supplied; use notebook 02 stretch controls')
    python = (args.python or Path.home()/f'.venv-{args.runtime}/bin/python').expanduser().absolute()
    model_folder = 'qwen3-4b' if args.model == 'qwen' else 'nemotron-3-nano-4b'
    snapshot = args.snapshot or ROOT/'models'/(model_folder + ('-awq-4bit-frozen' if args.awq else '-frozen'))
    snapshot = snapshot.expanduser().absolute()
    try:
        command = command_for(args, snapshot, python)
    except ValueError as exc:
        parser.error(str(exc))
    print('Engine command:', shlex.join(command), flush=True)
    if args.runtime == 'vllm':
        print('Environment: VLLM_SERVER_DEV_MODE=1 (required for cache reset endpoints)', flush=True)
    if args.dry_run:
        return 0
    if platform.system() != 'Linux':
        parser.error('Run live serving on your Linux NVIDIA GPU machine; --dry-run works without a GPU')
    if not python.is_file():
        parser.error('Engine Python missing or a broken/cyclic symlink: ' + str(python) + '; use --python with the working absolute path')
    if not (snapshot/'config.json').is_file():
        parser.error('Local checkpoint missing: ' + str(snapshot) + '; stage it first; no download is automatic')
    checkpoint = json.loads((snapshot/'config.json').read_text())
    quantized = checkpoint.get('quantization_config') or {}
    if args.awq != bool(quantized) or (args.awq and quantized.get('quant_method') != 'awq'):
        parser.error('Checkpoint precision differs from --awq; select the correct base/AWQ snapshot')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            parser.error('Port occupied; stop that server yourself or choose --port. No external server will be killed.')
    probe_code = ("import importlib.metadata as m,json,torch; "
                  f"print(json.dumps(dict(runtime=m.version({args.runtime!r}),torch=torch.__version__,cuda=torch.version.cuda,available=torch.cuda.is_available())))")
    try:
        info = json.loads(subprocess.check_output([str(python), '-c', probe_code], text=True, timeout=120).strip().splitlines()[-1])
        if not info['available']:
            parser.error('CUDA is unavailable in the selected engine Python')
        hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.total,driver_version',
                                           '--format=csv,noheader'], text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error('Runtime/GPU check failed: ' + str(exc))
    print('Hashing local checkpoint for launch provenance (before request timing).', flush=True)
    fingerprint = snapshot_fingerprint(snapshot)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    folder = ROOT/'experiment-tracker/mvp4/manual-servers'/stamp
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(runtime=args.runtime, engine_version=info['runtime'], torch_version=info['torch'],
                    model_sha256=fingerprint, evidence='hardware',
                    cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES', 'all'),
                    cuda_version=info['cuda'], hardware=hardware, model='clinical-trial-mvp3',
                    command=command, snapshot=str(snapshot), server_dev_mode=args.runtime == 'vllm',
                    status='starting', started_at=datetime.now(timezone.utc).isoformat(),
                    launch_settings=dict(context=args.context, dtype='float16', quantization='awq' if args.awq else None,
                                         prefix_cache=args.prefix_cache, port=args.port))
    metadata_path = folder/'server.json'
    def save():
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    save()
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.runtime == 'vllm':
        environment['VLLM_SERVER_DEV_MODE'] = '1'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (folder/'server.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        metadata['server_pid'] = process.pid
        save()
        print('Startup log:', folder/'server.log', flush=True)
        print('Metadata:', metadata_path, flush=True)
        try:
            deadline = time.monotonic() + args.readiness_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError('Engine exited during startup; inspect server.log')
                try:
                    with opener.open(f'http://127.0.0.1:{args.port}/health', timeout=5) as response:
                        response.read()
                    with opener.open(f'http://127.0.0.1:{args.port}/v1/models', timeout=5) as response:
                        models = json.load(response)
                    if 'clinical-trial-mvp3' not in {m.get('id') for m in models.get('data', [])}:
                        raise ValueError('Served model alias differs')
                    metadata['status'] = 'ready';save()
                    print(f'READY: http://127.0.0.1:{args.port}/v1 | {args.runtime} | {args.model} | '
                          f'{"AWQ W4A16" if args.awq else "FP16"} | prefix cache {args.prefix_cache}', flush=True)
                    print('Ctrl+C stops this owned server. Stop it before running optimization notebooks.', flush=True)
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            else:
                raise RuntimeError('Readiness timed out; inspect server.log')
            code = process.wait()
            if code:
                raise RuntimeError('Engine exited with code ' + str(code))
        except KeyboardInterrupt:
            print('Stopping owned server.', flush=True)
        except BaseException as exc:
            metadata.update(status='failed', reason=str(exc));save()
            raise
        finally:
            stop_owned(process)
            metadata['stopped_at'] = datetime.now(timezone.utc).isoformat()
            if metadata['status'] != 'failed':metadata['status'] = 'stopped'
            save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
