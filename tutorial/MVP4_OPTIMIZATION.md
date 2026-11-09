# Cache and quantization methodology

This guide holds the unique experimental methods behind chapter 01. Installation belongs in
[SETUP.md](../SETUP.md), configuration/execution in [RUN_ALL.md](RUN_ALL.md), and observations in
[RESULTS.md](RESULTS.md). It deliberately avoids another command-by-command launch transcript.

## Match the intervention to inference work

Prefill processes the prompt; decode generates tokens sequentially. Prefix caching reuses
matching shared prompt blocks, so warm TTFT is the first metric to investigate. AWQ W4A16
changes weights to four bits while retaining FP16 activations; it does not make the KV cache
four-bit. Kernel/backend compatibility matters. Weight compression is not a guarantee of
faster responses, correct SQL or lower total device allocation.

The Qwen matrix holds the workload fixed while varying unquantized FP16 versus validated AWQ,
and prefix cache off versus on. Model/tokenizer lineage is explicitly attested and fingerprinted.
Shapes and tokenizer equality cannot by themselves prove derivation. A successful AWQ SQL
compatibility smoke is required before all four live variants; failure is retained as evidence.

## Controlled cold/warm requests

The four target questions use the identical shared system/schema prefix. At concurrency one,
for every target/repeat: reset → cold target → reset → prime a different question with that
same prefix → identical target warm. A target is not primed by its own full question.
The first differing tokens bound prefix reuse; small shared question fragments can also match.

Every reset must explicitly acknowledge JSON `success=true`. HTTP 200 with an empty body is
insufficient. The installed native vLLM build requires development endpoints to expose reset;
the controller enables that consistently across variants. Other clients/external cache connectors
must not interfere. Kernel/connection warmup is excluded before the controlled sequence.

Each cache-on variant yields 20 cold and 20 warm requests: four distinct questions × five
repeats. Cache-off yields 20 requests. Keep phase summaries separate from the supplemental
warmed serving run and from application plan/SQL/repair. Startup/readiness, resets, priming,
telemetry setup, DB execution and SQL scoring are outside the individual request clocks.
[Reset/prime and SQL scoring implementation](../src/clinical_trial_mvp/tutorial/cache_study.py).

## Inspect output work before interpreting E2E

A token cap is not an observed output length. Compare actual counts overall and per case,
within cold/warm phases, before attributing E2E changes to computation. Generated SQL,
rationale and token identities can change under quantization. Similar lengths reduce one
simple explanation but do not make generated work identical or isolate kernel performance.

The [recorded token audit](output_token_analysis.json) and
[its interpretation](RESULTS.md#output-length-before-awq-attribution) use the same per-target
cache-on records as the phase findings. The analysis is reproduced by
`scripts/tutorial/analyze_output_tokens.py`; missing counts remain unavailable.
Client-estimated decode rate uses output usage and E2E-minus-TTFT. It is not server decode
profiling. Do not subtract unrelated summary medians as if they were a per-request decomposition.

## Quality and evidence controls

First-pass direct SQL is scored after timing; repairs do not hide those errors. Application
quality separately includes planning, validation, execution and one repair. Execution-result
match does not establish reviewed semantics. Preserve failed/stopped generations and malformed
contracts. Review quality counts alongside latency and throughput, not output tokens/s alone.

Native historical VRAM is unavailable. Total sampled device allocation would include weights,
reserved KV/state and other processes; it is not an isolated weight-size measurement.
The UUID normalization fix still needs fresh GPU telemetry. No memory-reduction conclusion is
supported by absent native samples. Nemotron hybrid-cache support and state controls are
model-specific differences discussed in chapter 02; its AWQ stretch remains optional.

The paired-comparison helper rejects mismatched data/date, prompts, model fingerprints,
engine/GPU settings or failed phase evidence. Even accepted pairs are small engineering
observations; broader quality claims require the full reference set and semantic review.
