# Three debugging decisions worth learning

## Cache-reset 404

**Symptom:** the server started and ordinary inference worked, but the cache study returned HTTP 404.

**Diagnostic evidence:** the [saved failed run](../experiment-tracker/mvp4/runs/20261106T222539_364377Z.json)
records the reset failure. Later launches with development endpoints exposed completed the
[paired cold/warm study](../experiment-tracker/mvp4/runs/20261106T224700_600112Z.json).

**Cause:** the running vLLM server did not expose `/reset_prefix_cache`. In the installed
0.32.0 build, these development endpoints require `VLLM_SERVER_DEV_MODE=1`.

**Correction:** enable the setting before launching, then require JSON `success=true` from
every reset. The reusable controller sets it consistently across native variants and records it.
A 200 with an empty body is not sufficient evidence of a cold cache.

**Reusable lesson:** verify experimental control endpoints, not only model readiness. Preserve
the failed attempt; do not relabel uncontrolled requests as cold.

[Reset verification source](../src/clinical_trial_mvp/tutorial/cache_study.py)
· [vLLM environment documentation](https://docs.vllm.ai/en/v0.32.0/configuration/env_vars/).

## Qwen NIM contract failure

**Symptom:** four requests completed successfully, but no application SQL executed.

**Diagnostic evidence:** all four plans in the [original smoke](../experiment-tracker/mvp4/nim/runs/20261107T213636_946106Z.json)
returned `filters` as `{}` or a populated object. The parser raised `Plan fields must be strings`.
The direct-serving path also produced plausible SQL, which did not fix the failed planning stage.

**Cause:** the initial NIM recipe disabled schema enforcement. Valid JSON did not imply valid
application structure. The plan contract requires string fields except boolean `needs_clarification`.

**Correction:** enable the existing JSON-schema request option, keep strict validation, and
save the changed request setting. The [constrained smoke](../experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json)
executed 3/4 application queries and matched 2/4 references. Both attempts remain available.

**Reusable lesson:** instrument transport, plan validation, SQL execution and result match
separately. Do not coerce malformed plans just to make a benchmark pass. Contract changes
must be visible when comparing deployments.

[Request builder](../src/clinical_trial_mvp/serving/client.py)
· [strict plan parser](../src/clinical_trial_mvp/agent/analyst.py).

## Missing VRAM telemetry

**Symptom:** native experiments completed, but peak GPU memory was null with no valid samples.

**Diagnostic evidence:** the [native baseline](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json)
reports `sample_count: 0` and `error: CalledProcessError`. Native provenance contains a bare
CUDA UUID; NIM provenance contains the equivalent `GPU-...` identifier and collected samples.

**Cause:** the sampler passed the bare UUID directly to `nvidia-smi --id`, which expects the
canonical identifier on this stack. The model ran; the measurement pipeline failed.

**Correction:** normalize bare CUDA UUIDs to the NVIDIA-SMI form in future launches/samples.
A regression test checks the command. Fresh GPU telemetry validation remains outstanding.

**Reusable lesson:** missing is not zero. Keep old values null, separate model success from
measurement success, and never claim weight-memory reduction from absent native telemetry.

[UUID normalization](../src/clinical_trial_mvp/serving/device.py)
· [memory sampler](../src/clinical_trial_mvp/tutorial/memory.py).
