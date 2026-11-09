# Publication readiness

**Current scope decision:** the owner has deferred human SQL semantic review and fresh
validation of the updated GPU automation to save time. They are follow-up work, not blockers
for the current educational/replay polish. Completed manual Dell GPU experiments remain valid
historical evidence. Deferral does not mark either follow-up as completed or upgrade its score.

| Item | Current evidence | Status |
|---|---|---|
| Project rights and license | Owner asked to defer license work; inherited/model notes stay intact | Deferred for this iteration |
| New automated GPU lifecycle | CPU tests only; manual Dell runs and CPU replay are separate evidence | Deferred by owner |
| Native VRAM telemetry | Historical samples unavailable; UUID fix has CPU regression coverage | Deferred with GPU automation validation |
| SQL semantic review | Four-question, 61-distinct-output packet and validated review-entry/coverage tool; no human judgments entered | Deferred by owner |
| Broader application-quality claims | Four distinct questions; full set contains 23 reference cases | Outside current claims |
| CPU dependency pins | Isolated Python 3.11.15 CPU notebook environment and exact constraints | Local validation recorded in VALIDATION.md |
| CPU CI | Hosted Linux workflow passed for source commit `56feab1`; suite, schema/links, Replay/Plan and generated artifacts | [Passed run](https://github.com/manimalakumar/local-ai-inference/actions/runs/42666644695) |
| Clean clone replay | Fresh remote clone of `56feab1`, new pinned CPU environment; six chapters Replay/Plan and 204 tests pass | Passed on macOS arm64; commands in [VALIDATION.md](../VALIDATION.md) |
| Clean source ZIP | Source-only review export inspected; environments, Git internals, local configs and weights excluded | Source candidate prepared; original working-directory ZIP must not be published |

Keep the existing [third-party notes](../THIRD_PARTY.md). No model weights are included.
License selection is deferred; current work focuses on publication polish and validation.

No new runtime is needed. Current work focuses on a clear tutorial, visible mechanisms,
preserved experiment results and reproducible CPU replay. The two deferred follow-ups retain
their checklists for later use.

## Current iterative review — November 9, 2026

Scores below are an internal source review against this educational scope, not an independent
certification. **9/10 is the target, not a reason to round up missing evidence.** The original
manual runs and earlier hosted/clean-clone check retain their own commit scope. Final local
CPU validation passed: 222 tests and all six chapters in Replay/Plan. The remaining technical
release step is a final push and successful hosted CI before changing repository visibility.

| Rubric | Current score / 10 | Basis and remaining condition |
|---|---:|---|
| Purpose, workload and model story | 9 | Concrete clinical Text-to-SQL example; exact model roles and known identity gaps are visible |
| Learning progression and mechanisms | 9 | 00→04 plus separate optional CPU chapter 05; inspectable request/SQL and filled interpretations |
| Experimental design and controls | 9 | Defined 2×2 matrix; cold/warm separated, distinct-question prime, startup excluded, no cumulative-best claim |
| Measurement reporting and restraint | 9 | Transport/application quality separated; token audit; missing VRAM and small sample limitations explicit |
| Evidence preservation and traceability | 9 | 65 original Dell hashes plus 26 foundation hashes verified; failures and frozen synthetic DB retained |
| Maintainability and regression checks | 9 | Output-preserving builders, review-entry regressions and generated-data drift gate; final local suite passes 222 tests |
| Learner execution design | 9 for Replay/Plan | All six chapters pass final local kernel execution; prior fresh-install/remote-clone result is scoped to `56feab1`; Live automation checks deferred |
| Source package hygiene | 9 for source export; supplied ZIP fails | Export rejects environments, Git internals, weights, tracked local configs/raw logs/nested ZIPs; use reviewed source-only export |
| Semantic validation evidence | Deferred; excluded from current rating | 0/61 distinct SQL judgments; packet/tool retained for future review |
| Automated GPU validation evidence | Deferred; excluded from current rating | Completed manual GPU runs retained; new lifecycle/UUID telemetry checks remain unverified |

License work is excluded from this iteration's rating at the owner's request; it remains deferred.

When these follow-ups resume, use [SQL_REVIEW.md](SQL_REVIEW.md) and
[GPU_VALIDATION.md](GPU_VALIDATION.md). Reference-result matches remain separate from human
semantic judgments, and native memory remains unavailable. Final local CPU Replay/Plan checks passed; the final candidate still needs a push and hosted
CI result tied to that exact commit. The prior fresh-install/remote-clone check remains scoped.
