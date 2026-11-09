# Publication readiness

| Item | Current evidence | Status |
|---|---|---|
| Project rights and license | Owner asked to defer license work; inherited/model notes stay intact | Deferred for this iteration |
| New automated GPU lifecycle | CPU tests only; manual Dell runs and CPU replay are separate evidence | **Fresh Dell run pending** |
| Native VRAM telemetry | Historical samples unavailable; UUID fix has CPU regression coverage | **GPU telemetry rerun pending** |
| SQL semantic review | Four-question, 61-distinct-output packet prepared across selected variants; no human judgments entered | **Human review pending** |
| Broader application-quality claims | Four distinct questions; full set contains 23 reference cases | **Full evaluation/review pending** |
| CPU dependency pins | Isolated Python 3.11.15 CPU notebook environment and exact constraints | Local validation recorded in VALIDATION.md |
| CPU CI | Hosted Linux workflow passed for source commit `56feab1`; suite, schema/links, Replay/Plan and generated artifacts | [Passed run](https://github.com/manimalakumar/local-ai-inference/actions/runs/42666644695) |
| Clean clone replay | Fresh remote clone of `56feab1`, new pinned CPU environment; six chapters Replay/Plan and 204 tests pass | Passed on macOS arm64; commands in [VALIDATION.md](../VALIDATION.md) |
| Clean source ZIP | Committed-revision export helper excludes repository internals/caches/weights | Helper testable locally; no public release created |

Keep the existing [third-party notes](../THIRD_PARTY.md). No model weights are included.
License selection is deferred; current work focuses on publication polish and validation.

No new runtime is needed. The finish line is a clear front-door tutorial, visible mechanisms,
reviewed examples, reproducible CPU replay and separately validated live GPU execution.
