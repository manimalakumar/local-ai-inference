# One master notebook across the Mac and NVIDIA machine

The master is `notebooks/clinical_trial_inference_tutorial.ipynb`. Its default execution loads saved results and does not contact model servers. It now shows four original Mac clinical configurations, six Dell MVP3 configurations, and the real Mac ExecuTorch case together. NIM, optimization and DLSIM retain their unmeasured states.

## What was recovered from the supplied GPU notebooks

Both original GPU notebooks remain unchanged. Sanitized archive copies are under `tutorial/history/gpu/`. The importer reads saved display output and literal configuration overrides; it never executes notebook source, launches a server or recreates metadata from a cell.

Six JSON summaries live under `experiment-tracker/tutorial/imported/`: vLLM and SGLang at concurrency 1, 2 and 4. Each retains its source notebook SHA-256, output cell index/execution count, archived-copy checksum and original displayed row. The final MVP3 measurement tables are used. The SGLang notebook also contains an older, pre-correction diagnostic output; those older values are not mixed with its later final table.

| Concurrency | vLLM TTFT p50 s | vLLM tok/s | SGLang TTFT p50 s | SGLang tok/s |
| --- | --- | --- | --- | --- |
| 1 | ≈0.19 | ≈40.67 | ≈0.13 | ≈43.49 |
| 2 | ≈0.24 | ≈67.09 | ≈0.17 | ≈75.35 |
| 4 | ≈0.37 | ≈105.37 | ≈0.31 | ≈127.68 |

Both final tables show 3/4 SQL executions and 2/4 execution-result matches at each concurrency. Semantic correctness is unreviewed. Peak VRAM is unknown. These are displayed, rounded figures from the provided notebooks; they do not establish an independently paired runtime comparison.

The notebooks do not contain the complete saved run JSON, per-request traces, SQL attempts, model/database fingerprints, launch metadata or measurement dates. Code that computes a fingerprint or writes a JSON file does not embed that resulting file in an `.ipynb`. These missing fields remain unknown. Older diagnostic output contains some full-precision figures, but it describes an earlier run and cannot supply precision for the final table.

## Evidence in the master

- Existing Mac clinical and edge records retain their original hardware evidence and provenance.
- Imported GPU summaries use `evidence: notebook_output`, displayed as **notebook summary** with **≈** on rounded metrics. They are kept outside the full raw run directory.
- Injected or unverified test results remain excluded from hardware tables.
- Full same-database raw records take precedence if later supplied for the same experiment.
- Pairing remains unverified for imported summaries. No model/date/database identity is invented from a rendered pairing message.

When live engines are disabled, the session replays an existing result's recorded hardware/settings even if the current machine's configuration differs. It does not replace historical measurements with a new unmeasured plan. To measure a changed configuration, enable that runtime and run it explicitly.

## Moving future complete runs between machines

After a real benchmark on the GPU host, retain:

```text
experiment-tracker/tutorial/runs/*.json
experiment-tracker/tutorial/provenance/*.json
```

For the Mac edge case, also retain `experiment-tracker/tutorial/edge/*.json`. The run records contain detailed samples and input fingerprints; provenance records identify the actual launch. Keep model weights, Python environments, compiled engines and credentials outside Git. Review manually created provenance for absolute machine paths before publishing; the normal harness saves portable paths.

Commit the small JSON evidence and notebook in the same repository, push from the machine that generated them, and pull on the other machine. Then run the master with:

```python
LIVE_RUNTIMES = set()
RUN_NIM = False
RUN_EDGE = False
```

The next default execution loads records and rebuilds the consolidated outputs. Calendar-relative SQL depends on the run date; preserve it. Raw clinical records for a different database are not silently loaded as current-database evidence. Unknown database identity in imported notebook summaries stays explicitly labelled as unknown.

To execute and save the master without inference servers:

```bash
python scripts/tutorial/validate_notebook.py --execute --output notebooks/clinical_trial_inference_tutorial.ipynb
```

Use an environment installed with `requirements-notebook.txt` (or the isolated edge environment, which includes those dependencies).

## Import another output-only notebook

```bash
python scripts/tutorial/import_notebook_results.py path/to/vllm-run.ipynb --runtime vllm
python scripts/tutorial/import_notebook_results.py path/to/sglang-run.ipynb --runtime sglang
```

Repeat imports of the same file are idempotent. Existing summary/archive files are not overwritten with different content. A new notebook can add another source snapshot. Source imports are deliberately limited to the known MVP3 table format; a missing or changed table is reported rather than guessed.

For complete precision and independently checked pairing of the current Dell runs, obtain the original JSON files from the two directories above. The consolidated notebook is usable now; those files can upgrade its evidence later.
