# MVP1 Q4 case-level semantic review

Each cell shows semantic review / SQL executed / end-to-end question latency. C = correct on inspected criteria, I = incorrect, P = partial; Y/N/— = SQL executed, failed, or not applicable. Detailed, model-specific review notes and retrieval facts are in [the structured audit](MVP1_Q4_CASE_AUDIT.json). This review includes all 30 questions, including the seven without reference SQL.

| Case | Nemotron full | Nemotron retrieved | Qwen full | Qwen retrieved |
| --- | ---: | ---: | ---: | ---: |
| `eval_001` | C / Y / 92.9s | C / Y / 29.0s | I / N / 59.8s | I / N / 20.1s |
| `eval_002` | C / Y / 95.5s | I / N / 44.3s | I / N / 59.2s | I / N / 18.1s |
| `eval_003` | I / N / 145.8s | I / N / 40.6s | I / N / 67.7s | I / Y / 14.8s |
| `eval_004` | I / N / 133.6s | I / N / 41.1s | I / Y / 55.0s | I / N / 22.3s |
| `eval_005` | I / N / 137.5s | I / N / 37.6s | I / N / 62.6s | I / N / 19.8s |
| `eval_006` | I / Y / 64.8s | I / N / 58.0s | I / N / 70.5s | I / N / 35.0s |
| `eval_007` | I / Y / 120.9s | I / Y / 36.2s | I / N / 59.2s | I / N / 36.5s |
| `eval_008` | I / Y / 127.3s | I / N / 49.7s | I / Y / 52.0s | I / N / 25.8s |
| `eval_009` | I / Y / 84.5s | I / Y / 18.0s | I / Y / 57.0s | I / N / 20.7s |
| `eval_010` | I / N / 172.5s | I / Y / 36.9s | I / N / 176.2s | I / N / 29.5s |
| `eval_011` | I / N / 117.7s | I / N / 36.3s | I / N / 81.8s | I / N / 27.4s |
| `eval_012` | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s |
| `eval_013` | I / Y / 68.5s | I / N / 44.2s | I / N / 75.4s | I / N / 29.8s |
| `eval_014` | P / N / 30.2s | P / N / 7.8s | P / N / 31.7s | P / N / 5.9s |
| `eval_015` | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s |
| `eval_016` | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s |
| `eval_017` | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s | C / — / 0.0s |
| `eval_018` | C / Y / 60.7s | C / Y / 17.6s | C / Y / 35.1s | C / Y / 12.9s |
| `eval_019` | C / Y / 63.9s | C / Y / 17.1s | C / Y / 64.3s | C / Y / 13.0s |
| `eval_020` | C / Y / 64.7s | I / Y / 14.9s | C / Y / 60.6s | C / Y / 13.7s |
| `eval_021` | C / Y / 72.4s | C / Y / 25.8s | I / N / 63.1s | I / N / 22.5s |
| `eval_022` | C / Y / 72.7s | C / Y / 17.3s | C / Y / 54.2s | I / Y / 21.6s |
| `eval_023` | I / Y / 75.1s | I / Y / 22.9s | I / N / 72.7s | I / Y / 19.2s |
| `eval_024` | I / N / 98.9s | I / N / 29.8s | I / N / 114.0s | I / N / 29.9s |
| `eval_025` | C / Y / 56.6s | C / Y / 16.2s | C / Y / 160.9s | C / Y / 16.9s |
| `eval_026` | I / N / 99.2s | C / Y / 30.2s | I / Y / 268.9s | C / Y / 25.4s |
| `eval_027` | I / N / 98.8s | I / N / 29.8s | I / N / 176.4s | I / N / 34.5s |
| `eval_028` | I / N / 97.4s | I / N / 29.2s | I / N / 203.8s | I / N / 27.9s |
| `eval_029` | I / Y / 74.3s | I / N / 33.4s | I / N / 300.2s | I / N / 34.9s |
| `eval_030` | C / Y / 111.9s | I / N / 39.2s | I / N / 303.9s | I / N / 32.0s |

A row-equivalent answer can still be marked incorrect after SQL review. `eval_020` in Nemotron retrieved excludes a nullable status; `eval_023` in Nemotron full and Qwen retrieved omits the non-null enrollment-date filter. All four `eval_014` results ask a useful metric clarification and avoid a causal explanation, but none checks whether Site 17 exists, so they are partial against the complete case criteria.

The five targeted counterfactual cases are `eval_002`, `eval_005`, `eval_021`, `eval_024`, and `eval_027`. Their per-run counterfactual records are saved alongside the raw results. A counterfactual match supports only those fixture conditions.
