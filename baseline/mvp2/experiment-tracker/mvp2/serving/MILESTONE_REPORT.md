# MVP2 local serving milestone — 2026-10-25

## Achieved

The same local Nemotron 3 Nano 4B Q4_K_M GGUF (SHA-256 `be5d9a656a51922f24f1f09a759cebb694e1f5d9728bf0ef9f8c972c5a0b5ef2`) served through two loopback endpoints on the M4 Pro:

| Runtime | Version and setup | Verified routes |
| --- | --- | --- |
| Native llama.cpp | Official macOS arm64 build `b11887`, commit `6050b467c`; archive SHA-256 `13d5a906ab066f57a937a24e345eb541cb423b249c2254ed83cb2a24bba3f502`; Metal offload requested with `--gpu-layers 99`, one slot, 8,192 context | `/health`, `/completion`, `/tokenize`, `/metrics` |
| Ollama | `0.40.5`; GGUF imported as `clinical-trial-mvp2-bench`; `ollama ps` reported 100% GPU, one slot, 8,192 context | `/api/version`, `/api/generate` |

The installed Ollama process launches its own bundled llama-server runner. This comparison is of the two serving packages and APIs with the same GGUF, not of unrelated inference algorithms. Both servers were kept resident, but measured requests were sent one at a time. Neither endpoint was exposed beyond loopback.

## Paired single-request screen

The [aligned raw run](paired_20261025T223525_745196Z.json) used the same prompt bytes for each runtime, temperature 0, seed 42, repetition penalty 1, 8,192 context, and a 64-token output cap. Each profile had one warmup and three measured requests per runtime. A changing prompt prefix limited cache reuse; both servers reported zero cached prompt tokens. Selected schema used `eval_001` (215 measured prompt tokens), full schema used `eval_002` (2,177 tokens). The prompt ends in `SQL:\nSELECT ` because an unseeded full-schema raw completion ended immediately without text in a diagnostic check. These are direct serving calls, not the complete agent path.

| Prompt | Runtime | Successful | TTFT p50 | E2E p50 | E2E p95* | Mean server TPOT | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected | llama.cpp | 3/3 | 0.400 s | 1.393 s | 1.404 s | 18.8 ms | 53 |
| Selected | Ollama | 3/3 | 0.414 s | 1.404 s | 1.418 s | 18.7 ms | 53 |
| Full | llama.cpp | 3/3 | 3.453 s | 4.568 s | 4.572 s | 18.8 ms | 59 |
| Full | Ollama | 3/3 | 3.669 s | 4.799 s | 4.849 s | 19.5 ms | 59 |

All 12 measured requests met the configured TTFT ≤ 5 s and end-to-end ≤ 30 s SLOs. For each profile, both runtimes produced the same response hash across the paired samples. This confirms that the compared raw requests had matching observed output for these cases; it does not establish SQL correctness. *The p95 values are descriptive interpolations from only three requests, not stable tail estimates.* The small differences do not support a general speed ranking.

An earlier [two-slot native pilot](paired_20261025T223415_592320Z.json) also completed, but Ollama was on one slot. It remains recorded and is excluded from the aligned table.

## Reproduction

1. Run `scripts/serving/setup_llama_cpp.sh` to obtain the pinned native build; its SHA-256 is checked. Model files are local and ignored by Git.
2. Start the installed Ollama service and run `ollama create clinical-trial-mvp2-bench -f ollama/Modelfile`. The model is imported locally from the same GGUF, with no quantization change.
3. Run `.venv/bin/python scripts/serving/start_llama_cpp.py` in a terminal. Central `.env` settings select the GGUF, loopback URL, context, slots, and GPU layer request.
4. Run `.venv/bin/python scripts/serving/compare_llama_ollama.py --repeats 3 --output-tokens 64`. The runner saves a new immutable JSON record in this folder.

The temporary native server was stopped and the imported Ollama model was unloaded after this run to release resources. The Ollama model remains registered locally; restart the native launcher before rerunning the screen.

The official [llama.cpp server guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) documents the native endpoints and metrics. Ollama documents [GGUF import](https://docs.ollama.com/import) and [raw generation plus token timing fields](https://docs.ollama.com/api/generate).

## Still pending

The full 12-configuration matrix, concurrency 2/4, synthetic 512/2K/4K sweeps, stable p95/goodput under load, resource sampling, failure recovery, and a fresh 30-question quality evaluation through each serving route have **not** been completed. Apple unified memory does not provide a separate comparable VRAM figure in the current harness. Do not interpret the inherited MVP1 30-question scores as new MVP2 quality results.
