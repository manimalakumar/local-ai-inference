# Dell MVP4 findings — November 6–7, 2026

These are saved hardware records from the Quadro RTX 5000 (16 GiB), not new Mac measurements. Native vLLM was 0.32.0; NIM used the digest-pinned 2.1.5 image. Original successes and failures are unchanged. The [archive manifest](../notebooks/history/dell-2026-11-07/MANIFEST.json) records their hashes and source commit.

## Application quality and request timing

Application SQL execution and result matches below use four cases, including one repair. Timing is the separate direct-serving workload; it is not application E2E. Cache-on timing here is the supplemental warmed serving run; use the following phase table for controlled cold/warm timing. NIM direct-serving records were not post-scored as SQL in this recipe; do not label transport success as direct SQL correctness.

| Run | Status | Application SQL executed | Application result match | Request TTFT p50 s | Request E2E p50/p95 s | Peak VRAM GiB | Schema enforcement |
|---|---|---|---|---|---|---|---|
| [MVP4/Optimization/fp16_cache_off](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) | completed | 3/4 | 2/4 | 2.049 | 6.081/7.653 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Optimization/fp16_cache_on](../experiment-tracker/mvp4/runs/20261106T222539_364377Z.json) | failed | —/4 | —/4 | unavailable | unavailable/unavailable | unavailable | native schema request; reasoning variant differs |
| [MVP4/Optimization/fp16_cache_on](../experiment-tracker/mvp4/runs/20261106T222610_177247Z.json) | failed | —/4 | —/4 | unavailable | unavailable/unavailable | unavailable | native schema request; reasoning variant differs |
| [MVP4/Optimization/fp16_cache_on](../experiment-tracker/mvp4/runs/20261106T224700_600112Z.json) | completed | 3/4 | 2/4 | 0.079 | 4.063/5.702 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Optimization/awq_w4a16_cache_off](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) | completed | 3/4 | 2/4 | 2.103 | 4.310/5.123 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Optimization/awq_w4a16_cache_on](../experiment-tracker/mvp4/runs/20261106T232408_684730Z.json) | completed | 3/4 | 2/4 | 0.055 | 2.282/3.093 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/fp16_baseline](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) | completed | 3/4 | 2/4 | 1.047 | 4.251/4.892 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/prefix_cache](../experiment-tracker/mvp4/nemotron/20261107T023148_583457Z.json) | completed | 3/4 | 2/4 | 0.319 | 3.575/4.207 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/schema_retrieved](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) | completed | 1/4 | 1/4 | 0.204 | 2.593/3.546 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/concurrency_2](../experiment-tracker/mvp4/nemotron/20261107T025050_224975Z.json) | completed | 3/4 | 2/4 | 1.099 | 5.362/7.011 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/concurrency_4](../experiment-tracker/mvp4/nemotron/20261107T030851_257660Z.json) | completed | 3/4 | 2/4 | 1.113 | 7.699/10.557 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/batch_capacity_1](../experiment-tracker/mvp4/nemotron/20261107T031845_031809Z.json) | completed | 3/4 | 2/4 | 13.389 | 16.704/16.748 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/reasoning_on](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) | completed | 2/4 | 2/4 | 1.060 | 9.104/16.138 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/output_256](../experiment-tracker/mvp4/nemotron/20261107T034418_092327Z.json) | completed | 2/4 | 2/4 | 1.059 | 4.265/4.918 | unavailable | native schema request; reasoning variant differs |
| [MVP4/Nemotron/output_2048](../experiment-tracker/mvp4/nemotron/20261107T035411_952257Z.json) | completed | 3/4 | 2/4 | 1.054 | 4.268/4.914 | unavailable | native schema request; reasoning variant differs |
| [MVP4/NIM/nemotron/smoke](../experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json) | completed | 3/4 | 2/4 | 1.061 | 8.872/12.333 | 13.853 | False |
| [MVP4/NIM/nemotron/baseline](../experiment-tracker/mvp4/nim/runs/20261107T204614_919475Z.json) | completed | 3/4 | 2/4 | 1.063 | 8.889/12.469 | 13.853 | False |
| [MVP4/NIM/nemotron/schema_retrieved](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) | completed | 3/4 | 2/4 | 0.208 | 5.768/10.374 | 13.853 | False |
| [MVP4/NIM/nemotron/concurrency_2](../experiment-tracker/mvp4/nim/runs/20261107T205602_380125Z.json) | completed | 3/4 | 2/4 | 1.113 | 10.392/13.710 | 13.853 | False |
| [MVP4/NIM/nemotron/concurrency_4](../experiment-tracker/mvp4/nim/runs/20261107T205940_074402Z.json) | completed | 3/4 | 2/4 | 1.136 | 12.044/22.403 | 13.892 | False |
| [MVP4/NIM/qwen/smoke](../experiment-tracker/mvp4/nim/runs/20261107T213636_946106Z.json) | failed | 0/4 | 0/4 | 2.021 | 6.151/7.543 | 13.950 | False |
| [MVP4/NIM/qwen/smoke](../experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json) | completed | 3/4 | 2/4 | 2.031 | 6.172/7.567 | 13.950 | True |
| [MVP4/NIM/qwen/baseline](../experiment-tracker/mvp4/nim/runs/20261107T215646_526562Z.json) | completed | 3/4 | 2/4 | 2.042 | 6.200/7.775 | 13.950 | True |
| [MVP4/NIM/qwen/schema_retrieved](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) | completed | 4/4 | 3/4 | 0.194 | 2.735/3.241 | 13.950 | True |
| [MVP4/NIM/qwen/concurrency_2](../experiment-tracker/mvp4/nim/runs/20261107T220932_908954Z.json) | completed | 3/4 | 2/4 | 2.087 | 8.595/12.904 | 13.950 | True |
| [MVP4/NIM/qwen/concurrency_4](../experiment-tracker/mvp4/nim/runs/20261107T221207_165140Z.json) | completed | 3/4 | 2/4 | 2.136 | 14.085/21.368 | 13.950 | True |

## Controlled cache phases

Each target uses the same shared schema prefix. Warm requests are primed with a different question after a confirmed reset. Startup and priming are excluded. These rows report direct, first-pass SQL reference-result matches, without application repairs.

| Run | Phase | SQL result match | TTFT p50/p95 s | E2E p50/p95 s |
|---|---|---|---|---|
| [MVP4/Optimization/fp16_cache_off](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) | cache_off | 10/20 | 2.005/2.014 | 5.994/7.601 |
| [MVP4/Optimization/fp16_cache_on](../experiment-tracker/mvp4/runs/20261106T224700_600112Z.json) | cold | 10/20 | 2.019/2.033 | 6.038/7.644 |
| [MVP4/Optimization/fp16_cache_on](../experiment-tracker/mvp4/runs/20261106T224700_600112Z.json) | warm | 10/20 | 0.083/0.102 | 4.039/5.691 |
| [MVP4/Optimization/awq_w4a16_cache_off](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) | cache_off | 15/20 | 2.070/2.078 | 4.241/5.076 |
| [MVP4/Optimization/awq_w4a16_cache_on](../experiment-tracker/mvp4/runs/20261106T232408_684730Z.json) | cold | 15/20 | 2.083/2.096 | 4.271/5.110 |
| [MVP4/Optimization/awq_w4a16_cache_on](../experiment-tracker/mvp4/runs/20261106T232408_684730Z.json) | warm | 15/20 | 0.057/0.077 | 2.258/3.079 |
| [MVP4/Nemotron/prefix_cache](../experiment-tracker/mvp4/nemotron/20261107T023148_583457Z.json) | cold | 15/20 | 1.101/1.109 | 4.327/4.968 |
| [MVP4/Nemotron/prefix_cache](../experiment-tracker/mvp4/nemotron/20261107T023148_583457Z.json) | warm | 15/20 | 0.300/0.305 | 3.529/4.179 |

## What can be concluded

- Both exact base models ran on native vLLM and the NIM container. Nemotron hybrid caching completed on this installed stack. That evidence is specific to this GPU/build.
- Completed means the experiment finished, not that all SQL was correct. Semantic review remains unreviewed; four questions cannot establish general clinical accuracy.
- Native GPU memory samples are unavailable: PyTorch reported a bare UUID, and the sampler passed it to `nvidia-smi` without `GPU-`. The new launcher/sampler normalizes that format. Historical values remain null; memory comparisons need fresh GPU runs.
- The original unconstrained Qwen NIM smoke had 4/4 transport successes but 0/4 application SQL executions: every plan returned an object for `filters`, while the contract requires a string. Enabling the existing JSON-schema option produced a completed rerun. Both records remain saved; future Qwen NIM configs default to enforcement on.
- Nemotron NIM used schema enforcement off; Qwen NIM reruns used it on. Dates, prompts, backend versions and thinking controls also require matching. Do not claim a causal NIM-versus-native speed gain from these records.
- No validated Nemotron AWQ checkpoint was obtained. AWQ remains optional stretch work; FP8/GGUF are not equivalent substitutes.

## Troubleshooting journal

| Observed problem | Cause / resolution captured in the reusable workflow |
|---|---|
| vLLM package appeared missing | The terminal used the SGLang Python. Configure the actual GPU `vllm_python`, independent of the Jupyter environment. |
| Project vLLM symlink loop | Use the absolute GPU Python directly; no symlink is required. Ignore environment directories and symlinks. |
| Copy/paste indentation and Python pasted into Bash | Notebook users edit JSON configuration and use Run All. Optional CLI helper scripts are supplied in the repo. |
| Cache reset HTTP 404 | Native vLLM development endpoints require `VLLM_SERVER_DEV_MODE=1`; the workflow enables it on all native launches and records it. Reset helpers still require `success=true`. |
| Provenance file already exists after restart | Each notebook session and launch gets a unique provenance path. Earlier records are retained. |
| Startup included in measurement concerns | Readiness is recorded separately; request clocks remain inside the existing transport helpers. |
| Native GPU memory absent | Bare CUDA UUID was missing the NVIDIA-SMI prefix. Normalize future UUIDs; retain missing historical values. |
| Docker was not installed | Install system Docker Engine plus NVIDIA Container Toolkit once, then verify GPU access. These are not pip requirements. |
| Docker socket permission denied | Activate group membership and restart Jupyter from that session. Ubuntu 26.04 provides `newgrp` in `util-linux-extra`. |
| NIM profile IDs unclear | Discover per model/image/GPU, inspect standard versus LoRA and TP/PP, select explicitly in local config. |
| Qwen NIM plan contract failed | Preserve the failed smoke; enable schema enforcement as an explicit recorded request setting and rerun. Never coerce invalid plans to hide the failure. |
| Many manual start/stop and ACTIVE cells | New notebooks use config-driven server ownership, readiness checks, smoke gates, automatic cleanup and fresh metadata. |
| Git staging failed | `git add notebooks/ ...` uses a plural directory; never concatenate it to `--`. Keep weights/environments/credentials out of Git. |
| Git author unknown | Set repository-local `git config user.name` and `git config user.email`; this differs from GitHub authentication. |

## Reproduce and preserve

Use [SETUP.md](../SETUP.md). The four clean notebooks never rebuild the archived executions. Rebuilder scripts touch only their four named reusable notebooks. The unchanged raw evidence remains under `experiment-tracker/mvp4`; future runs add unique files. Unit tests verify every archived hash.
