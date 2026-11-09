# Fresh GPU validation of the automated workflow — pending

The archived manual Dell runs are not validation of the later automated lifecycle. CPU tests
and replay cannot establish kernel compatibility, GPU cleanup or working native memory sampling.
This checklist is intentionally incomplete until a new Linux GPU execution supplies evidence.

1. Pull the reviewed code to Dell and preserve any local executed notebooks/results first.
2. Use the working GPU vLLM environment and the documented CPU notebook constraints. Confirm
   Docker GPU access without sudo; restart Jupyter from the Docker-enabled session.
3. Create each per-notebook local config. Set `mode: "live"`, the actual GPU Python/model
   paths and pinned revisions. For Qwen, record AWQ derivation. For NIM, discover your own
   model/image/GPU profile and set `nim_profile_id`. Do not copy a replay profile ID blindly.
4. Start with `01_qwen_cache_quantization.ipynb` and Run All. Retain all four fresh native
   launches, AWQ smoke, reset confirmations, paired cold/warm records, logs and cleanup state.
5. Run the other chapters in order, one model/server at a time. Enable Nemotron hybrid caching
   only after installed-backend verification; AWQ remains optional and requires its own checkpoint.
6. For every new native result, verify `memory.peak_bytes` is non-null, `sample_count > 0` and
   `error` is null. Check the canonical GPU UUID. Missing values remain failures of telemetry,
   not evidence of zero VRAM use or a memory reduction.
7. Confirm the recorded readiness/startup duration is separate from request timing. Check the
   resolved dtype/cache/scheduler settings in logs, not just requested CLI arguments.
8. After successful completion and an intentional interrupted/failed operation, verify that
   owned servers stop. Verify an unrelated running server is refused rather than stopped.
9. Preserve a new executed copy plus raw records/provenance. Never overwrite the November 6–7
   archive. Report model hashes, GPU UUID, driver, engine/image and request/schema settings.

Only after reviewing these artifacts should the publication status change to GPU validated.
Run the full 23-reference-case workload before making broader application-quality claims;
human semantic review remains a separate step. No fresh GPU run has been performed by Codex
on this Mac, and no absent measurements have been inferred from the archive.
