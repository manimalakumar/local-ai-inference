# Four-case SQL review packet

**Deferred by the owner for this iteration.** The packet and entry tool are retained for
later use; completing them is not part of the current educational/replay polish.

**Human semantic review: pending, 0/4 cases reviewed.** The following are recorded Dell application outputs from the Qwen full-schema baseline. Machine execution and reference-result scores are shown as evidence, not human judgments. Repetition of these cases is not broader coverage.

A reviewer should compare the question, plan, final SQL and reference definition, including grouping, denominator, date window, serious-versus-severe distinctions and required zero-count groups. Enter a named reviewer, date, exact SQL hash and boolean judgment in `tutorial/semantic_reviews.json` only after review. Do not copy the automated match bit as a human verdict.

**Required comparison coverage:** continue with the [61-distinct-output packet](SQL_REVIEW_DISTINCT.md)
and [structured queue](sql_review_distinct.json). It includes selected native/NIM application,
direct-serving and paired cold/warm target outputs. Review each exact SQL, not only these
four baseline examples. Human judgments remain empty until a person reviews them.

Source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json). These examples do not review every optimization variant; each distinct SQL output needs its own exact-hash review.

## eval_001

How many subjects are Enrolled or Randomized by treatment arm?

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT sub.subject_id) AS enrolled_or_randomized_count
FROM subjects AS sub
WHERE sub.screening_status IN ('Enrolled', 'Randomized')
GROUP BY sub.treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

**Recorded machine evidence:**

```json
{
  "execution_success": true,
  "error": null,
  "plan_error": null,
  "scores": {
    "sql_generated": true,
    "sql_safety": true,
    "schema_compliance": true,
    "sql_execution_success": true,
    "result_equivalence": true,
    "first_pass_accuracy": true,
    "repair_count": 0,
    "repair_success": null,
    "schema_linking_accuracy": null
  },
  "sql_sha256": "a5d7ad1b4a994ce855750973efe540d1c59f13cfc1c19a0ccc3f4ba26922b54a"
}
```

**Human judgment:** pending.

## eval_002

How many subjects have enrolled at each site?

**Generated final SQL:**

```sql
SELECT st.site_id, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects
FROM sites AS st
LEFT JOIN subjects AS sub ON st.site_id = sub.site_id
AND sub.screening_status IN ('Enrolled', 'Randomized')
GROUP BY st.site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

**Recorded machine evidence:**

```json
{
  "execution_success": true,
  "error": null,
  "plan_error": null,
  "scores": {
    "sql_generated": true,
    "sql_safety": true,
    "schema_compliance": true,
    "sql_execution_success": true,
    "result_equivalence": true,
    "first_pass_accuracy": true,
    "repair_count": 0,
    "repair_success": null,
    "schema_linking_accuracy": null
  },
  "sql_sha256": "bd5f0043db6ca33c3fdfe93dc726ffc93791675fdd630a85a4b9c8dee9625343"
}
```

**Human judgment:** pending.

## eval_005

How many scheduled visits were missed last month?

**Generated final SQL:**

```sql
SELECT COUNT(visit_id) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= DATE('now', '-1 month')
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

**Recorded machine evidence:**

```json
{
  "execution_success": false,
  "error": "SQL parser rejected query: Parser Error: Wrong number of arguments provided to DATE function",
  "plan_error": null,
  "scores": {
    "sql_generated": true,
    "sql_safety": false,
    "schema_compliance": null,
    "sql_execution_success": false,
    "result_equivalence": false,
    "first_pass_accuracy": false,
    "repair_count": 1,
    "repair_success": false,
    "schema_linking_accuracy": null
  },
  "sql_sha256": "b1cb6f13a9557df4606c456d9f5c8a7b330c16a6645c9f5496821d6c0e4a027b"
}
```

**Human judgment:** pending.

## eval_007

Compare serious adverse-event rates across treatment arms.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT sub.subject_id) AS total_subjects, COUNT(DISTINCT ae.subject_id) AS serious_adverse_events
FROM subjects AS sub
JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id
WHERE ae.serious = true
GROUP BY sub.treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

**Recorded machine evidence:**

```json
{
  "execution_success": true,
  "error": null,
  "plan_error": null,
  "scores": {
    "sql_generated": true,
    "sql_safety": true,
    "schema_compliance": true,
    "sql_execution_success": true,
    "result_equivalence": false,
    "first_pass_accuracy": false,
    "repair_count": 0,
    "repair_success": null,
    "schema_linking_accuracy": null
  },
  "sql_sha256": "0b067ad725242ef3e7ae5eb48946e68546cbd504012c9ec5468e252f801748f2"
}
```

**Human judgment:** pending.

## Enter an actual human judgment

Inspect the SQL and its source plan/repair, then create a JSON object in a local file, e.g.
`/tmp/sql-review-entry.json`, with `case_id`, the packet's full `sql_sha256`, `reviewer`,
ISO `reviewed_at`, boolean `correct` and a substantive `rationale`. Supply your real review;
the tool does not generate a judgment or establish the reviewer's identity.

```bash
.venv/bin/python scripts/tutorial/review_sql.py
.venv/bin/python scripts/tutorial/review_sql.py --record /tmp/sql-review-entry.json
.venv/bin/python scripts/tutorial/build_sql_review_packet.py
```

The first command only reports coverage. Recording rejects unknown case/hash pairs, missing
attribution/rationale, invalid dates and duplicate reviews before atomically updating
`semantic_reviews.json`. It never alters the frozen result records. Rebuild the packet after
recording reviews to update its coverage and attributed judgments. Tests accept valid human
reviews; they do not require the file to remain empty. A wrong SQL verdict can still be
`correct: false` even when the archived reference-result bit is true.
