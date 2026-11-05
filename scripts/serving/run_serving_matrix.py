"""Preview or execute MVP3's eight workloads against one already running engine."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from clinical_trial_mvp.serving.portable import display_command


def matrix_commands(runtime, metadata=None, case_ids=(), base_url="http://127.0.0.1:8000"):
    matrix = json.loads((ROOT / "serving/matrix.json").read_text())
    if runtime not in matrix["runtimes"]:
        raise ValueError("Unknown runtime")
    for mode in ("serving", "quality"):
        for profile in matrix["profiles"]:
            for concurrency in matrix[mode]["concurrency"]:
                config = matrix[mode]
                command = [sys.executable, str(ROOT / "scripts/serving/compare_vllm_sglang.py"),
                           "run", "--runtime", runtime, "--mode", mode, "--profile", profile,
                           "--concurrency", str(concurrency), "--repeats", str(config["repeats"]),
                           "--warmups", str(config["warmups"]), "--max-tokens", str(config["max_tokens"]),
                           "--base-url", base_url]
                if metadata:
                    command.extend(["--server-metadata", str(metadata)])
                for case in case_ids:
                    command.extend(["--case", case])
                yield command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, choices=["vllm", "sglang"])
    parser.add_argument("--server-metadata", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute and not args.server_metadata:
        parser.error("--execute requires --server-metadata from launch")
    for command in matrix_commands(args.runtime, args.server_metadata, args.case, args.base_url):
        print(shlex.join(display_command(command)), flush=True)
        if args.execute:
            result = subprocess.run(command, check=False)
            if result.returncode:
                parser.exit(result.returncode, f"Workload failed with exit code {result.returncode}; matrix stopped.\n")


if __name__ == "__main__":
    main()
