# Publication readiness

| Item | Current evidence | Status |
|---|---|---|
| Project rights and license | Owner asked to defer license work; inherited/model notes stay intact | Deferred for this iteration |
| New automated GPU lifecycle | CPU tests only; manual Dell runs and CPU replay are separate evidence | **Fresh Dell run pending** |
| Native VRAM telemetry | Historical samples unavailable; UUID fix has CPU regression coverage | **GPU telemetry rerun pending** |
| SQL semantic review | Four-question, 61-distinct-output packet and validated review-entry/coverage tool; no human judgments entered | **Human review pending** |
| Broader application-quality claims | Four distinct questions; full set contains 23 reference cases | **Full evaluation/review pending** |
| CPU dependency pins | Isolated Python 3.11.15 CPU notebook environment and exact constraints | Local validation recorded in VALIDATION.md |
| CPU CI | Hosted Linux workflow passed for source commit `56feab1`; suite, schema/links, Replay/Plan and generated artifacts | [Passed run](https://github.com/manimalakumar/local-ai-inference/actions/runs/42666644695) |
| Clean clone replay | Fresh remote clone of `56feab1`, new pinned CPU environment; six chapters Replay/Plan and 204 tests pass | Passed on macOS arm64; commands in [VALIDATION.md](../VALIDATION.md) |
| Clean source ZIP | Committed-revision export helper excludes repository internals/caches/weights | Helper testable locally; no public release created |

Keep the existing [third-party notes](../THIRD_PARTY.md). No model weights are included.
License selection is deferred; current work focuses on publication polish and validation.

No new runtime is needed. The finish line is a clear front-door tutorial, visible mechanisms,
reviewed examples, reproducible CPU replay and separately validated live GPU execution.

## Current iterative review — November 9, 2026

Scores below are an internal source review against this educational scope, not an independent
certification. **9/10 is the target, not a reason to round up missing evidence.** The original
manual runs and earlier hosted/clean-clone check retain their own commit scope. A final clean
run of this revision is deliberately deferred until the remaining work is ready.

| Rubric | Current score / 10 | Basis and remaining condition |
|---|---:|---|
| Purpose, workload and model story | 9 | Concrete clinical Text-to-SQL example; exact model roles and known identity gaps are visible |
| Learning progression and mechanisms | 9 | 00→04 plus separate optional CPU chapter 05; inspectable request/SQL and filled interpretations |
| Experimental design and controls | 9 | Defined 2×2 matrix; cold/warm separated, distinct-question prime, startup excluded, no cumulative-best claim |
| Measurement reporting and restraint | 9 | Transport/application quality separated; token audit; missing VRAM and small sample limitations explicit |
| Evidence preservation and traceability | 9 | 65 original Dell hashes plus 26 foundation hashes verified; failures and frozen synthetic DB retained |
| Maintainability and regression checks | 9, provisional | Output-preserving builders, six-chapter static source checks, review-entry regressions and generated JSON drift gate; final full run pending |
| Learner execution design | 9, provisional | Replay default, mode/config separation, owned cleanup design; prior clean-clone result is scoped to `56feab1` |
| Source package hygiene | 9 for source export; supplied ZIP fails | Export rejects environments, Git internals, weights, tracked local configs/raw logs/nested ZIPs; use reviewed source-only export |
| Semantic validation evidence | 6.5 — below target | 0/61 distinct SQL judgments; a packet/tool is preparation, not completed human review |
| Automated GPU validation evidence | 6.5 — below target | New lifecycle/UUID telemetry fix still needs clean Dell execution; no native memory-saving claim |

License work is excluded from this iteration's rating at the owner's request; it remains deferred.

To reach at least 9 in the last two rows: review all distinct outputs supporting the retained
headline comparisons with a named reviewer, ISO date, exact hash and rationale; then run clean
chapters 01–04 on Dell, capture failures as well as success, verify cleanup and native memory
or explicitly retain it as unavailable. Keep conclusions within the evidence actually obtained.
The final CPU Replay/Plan/CI/clean-clone pass must identify the exact final source commit.
A successful model response does not by itself satisfy any of those gates.
