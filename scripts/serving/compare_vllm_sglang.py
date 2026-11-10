"""MVP3: preview launch commands, run one GPU engine, then compare saved runs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.evaluation.dataset import sha256_file
from clinical_trial_mvp.serving.benchmark import Workload, prepare, run_quality, run_serving, compare
from clinical_trial_mvp.serving.client import Generation, OpenAIClient
from clinical_trial_mvp.serving.launch import launch, launch_command
from clinical_trial_mvp.serving.portable import display_command, portable_text, portable_value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("launch", help="Preview a launch command, or execute on a Linux CUDA host")
    start.add_argument("--runtime", choices=["vllm", "sglang"], required=True)
    start.add_argument("--snapshot", type=Path, required=True)
    start.add_argument("--port", type=int, default=8000)
    start.add_argument("--context", type=int, default=8192)
    start.add_argument("--max-num-seqs", type=int)
    start.add_argument("--tp", type=int, default=1)
    start.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    start.add_argument("--prefix-cache", action="store_true")
    start.add_argument("--quantization", choices=["awq", "gptq", "fp8"])
    start.add_argument("--execute", action="store_true")
    start.add_argument("--metadata", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--runtime", choices=["vllm", "sglang"], required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8000")
    run.add_argument("--server-metadata", type=Path)
    run.add_argument("--mode", choices=["serving", "quality"], default="serving")
    run.add_argument("--profile", choices=["full", "retrieved"], default="full")
    run.add_argument("--case", action="append", default=[])
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--max-repairs", type=int, choices=[0, 1], default=1)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--timeout", type=float, default=180)
    run.add_argument("--ttft-slo", type=float, default=5)
    run.add_argument("--e2e-slo", type=float, default=30)
    run.add_argument("--output", type=Path, default=ROOT / "experiment-tracker/mvp3/serving/runs")
    run.add_argument("--dry-run", action="store_true")
    paired = commands.add_parser("compare")
    paired.add_argument("left", type=Path)
    paired.add_argument("right", type=Path)
    paired.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "launch":
            options = {key: getattr(args, key) for key in ("port", "context", "tp", "dtype", "prefix_cache", "quantization", "max_num_seqs")}
            if args.execute:
                if args.metadata is None:
                    raise ValueError("--execute requires a new --metadata path")
                return launch(args.runtime, args.snapshot, args.metadata, **options)
            print(shlex.join(display_command(launch_command(args.runtime, args.snapshot, **options))))
            return 0
        if args.command == "compare":
            manifests = [json.loads(path.read_text()) for path in (args.left, args.right)]
            for path, manifest in zip((args.left, args.right), manifests):
                if manifest.get("samples_sha256") != sha256_file(path.parent / "samples.jsonl"):
                    raise ValueError(f"Sample checksum mismatch: {path}")
            result = compare(*manifests)
            result["source_sha256"] = {m["runtime"]: sha256_file(p)
                                       for p, m in zip((args.left, args.right), manifests)}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                output.write(json.dumps(portable_value(result), indent=2) + "\n")
            print(portable_text(args.output))
            return 0
        workload = Workload(args.mode, args.profile, args.repeats, args.warmups, args.concurrency,
                            args.max_repairs, args.ttft_slo, args.e2e_slo)
        generation = Generation(max_tokens=args.max_tokens, seed=args.seed, timeout_seconds=args.timeout)
        client = OpenAIClient(args.base_url, "clinical-trial-mvp3", generation)
        cases, contexts, prompts, inputs = prepare(ROOT, workload, args.case)
        if args.dry_run:
            print(json.dumps({"status": "plan_only", "runtime": args.runtime,
                              "workload": asdict(workload), "generation": asdict(generation), "inputs": inputs}, indent=2))
            return 0
        if args.server_metadata is None:
            raise ValueError("Live runs require --server-metadata from the launch command")
        server = json.loads(args.server_metadata.read_text())
        required = {"runtime", "engine_version", "model_sha256", "hardware", "launch_settings",
                    "command", "cuda_visible_devices", "model"}
        if not required.issubset(server) or any(server[k] in (None, "", {}) for k in required):
            raise ValueError("Server metadata is incomplete")
        if server["runtime"] != args.runtime or server["model"] != client.model:
            raise ValueError("Server metadata runtime/model does not match this run")
        settings = Settings.from_env()
        # Calendar-relative SQL uses DuckDB CURRENT_DATE; require the same evaluation date.
        import duckdb
        with duckdb.connect() as connection:
            sql_date = str(connection.execute("SELECT CURRENT_DATE").fetchone()[0])
        contract = {"version": "mvp3-v1", "inputs": inputs, "workload": asdict(workload),
                    "generation": asdict(generation), "model_sha256": server["model_sha256"],
                    "hardware": server["hardware"], "cuda_visible_devices": server["cuda_visible_devices"],
                    "launch_settings": server["launch_settings"], "evaluation_date": sql_date,
                    "database_sha256": sha256_file(settings.db_path),
                    "sql_limits": {"rows": settings.sql_result_row_limit,
                                   "timeout": settings.sql_execution_timeout_seconds}}
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.runtime
        folder = args.output / run_id
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {"run_id": run_id, "runtime": args.runtime, "status": "running",
                    "contract": contract, "server": server, "endpoint": client.base_url}
        manifest_path = folder / "manifest.json"
        manifest_path.write_text(json.dumps(portable_value(manifest), indent=2) + "\n")
        try:
            with (folder / "samples.jsonl").open("x") as details:
                def emit(record):
                    details.write(json.dumps(portable_value(record), default=str) + "\n")
                    details.flush()
                manifest["summary"] = (
                    run_serving(client, cases, prompts, workload, emit) if args.mode == "serving" else
                    run_quality(client, settings, cases, contexts, workload, emit))
            manifest["status"] = "completed"
            manifest["samples_sha256"] = sha256_file(folder / "samples.jsonl")
        except BaseException as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            manifest_path.write_text(json.dumps(portable_value(manifest), indent=2) + "\n")
        print(portable_text(manifest_path))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        parser.error(portable_text(exc))


if __name__ == "__main__":
    raise SystemExit(main())
