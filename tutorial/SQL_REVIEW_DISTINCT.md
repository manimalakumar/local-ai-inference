# Distinct-output SQL review packet

**Human review: 0/61 distinct SQL outputs reviewed across four questions.**
The [baseline packet](SQL_REVIEW.md) is an introduction, not coverage of the headline comparisons.
This packet includes all selected native/NIM variants and smokes with available records,
application final attempts, direct serving responses and paired cold/warm target SQL.
Cache priming requests are excluded because they are not measured targets. Original failed
runs remain in the archive; no-SQL responses are inventoried separately, not counted as SQL.

Review the definitions, plan and any repair in each source record, the final SQL, required
zero-count groups, population/denominator, serious versus severe, and previous-calendar-month
boundaries. Machine reference match is evidence, not a human verdict. The same exact SQL can
have many occurrences; repetition is not broader question coverage.

Use [the complete structured queue](sql_review_distinct.json) for every source JSON pointer,
phase, variant, available machine scores and no-SQL failure. Each occurrence identifies where
to read its original plan/attempt history. The SQL hash strips only outer whitespace and
matches the scorer's review lookup; it does not normalize away semantic differences.

After actual human review, add an entry to `tutorial/semantic_reviews.json` with `case_id`,
`sql_sha256`, `reviewer`, `reviewed_at` (ISO date/time), boolean `correct`, and `rationale`.
Do not fill these from automated scores or identify an AI draft as a human review.
This generator only rebuilds the queue/packet and never changes submitted judgments.

## eval_001 — 026c7c85ff8b

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `026c7c85ff8b53171e860e089ef71f200e4b51849e2025ad4b9ec6f2cc530c57` · occurrences: 5.

Scopes: nim-nemotron/baseline/quality, nim-nemotron/concurrency_2/quality, nim-nemotron/concurrency_4/quality, nim-nemotron/smoke/quality, vllm-nemotron/reasoning_on/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS enrolled_subjects FROM subjects WHERE screening_status IN ('Enrolled','Randomized') GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 1f7b28260cf5

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `1f7b28260cf56632e3982534759dd99caa07c2d68fe06cf08e225c33ea013946` · occurrences: 52.

Scopes: vllm-nemotron/batch_capacity_1/quality, vllm-nemotron/batch_capacity_1/serving, vllm-nemotron/concurrency_2/quality, vllm-nemotron/concurrency_2/serving, vllm-nemotron/concurrency_4/quality, vllm-nemotron/concurrency_4/serving, vllm-nemotron/fp16_baseline/quality, vllm-nemotron/fp16_baseline/serving, vllm-nemotron/output_2048/quality, vllm-nemotron/output_2048/serving, vllm-nemotron/output_256/quality, vllm-nemotron/output_256/serving, vllm-nemotron/prefix_cache/cold, vllm-nemotron/prefix_cache/quality, vllm-nemotron/prefix_cache/serving, vllm-nemotron/prefix_cache/warm.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS enrolled_subjects FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 1fd5a35e1d7d

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `1fd5a35e1d7d42df76755d28bb72edf29137ebdc5279a44ebc4cb68f8279e446` · occurrences: 4.

Scopes: nim-qwen/baseline/quality, nim-qwen/concurrency_2/quality, nim-qwen/concurrency_4/quality, nim-qwen/smoke/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json](../experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS enrolled_or_randomized_count
FROM subjects
WHERE screening_status IN ('Enrolled', 'Randomized')
GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 2aafff04933e

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `2aafff04933e81deccfdcb811089b9e71ee521926fb67cde3da190f4a669cf3c` · occurrences: 6.

Scopes: vllm-nemotron/schema_retrieved/quality, vllm-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 3d3251d6cb39

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `3d3251d6cb3998969fef49ac9b61ff44f37596e4b4e6e1519ef1f709078a4507` · occurrences: 1.

Scopes: nim-qwen/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(subject_id) AS count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 3d5502d13637

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `3d5502d1363722b399d4bd79a40d250f6a070ffe60e270319ea94c72f77ed05e` · occurrences: 5.

Scopes: nim-qwen/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/serving/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 7f23a9b8058f

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `7f23a9b8058fb27adbb5715e7d1bdb59e64ba720b2600cec4f0bbdb53396fae2` · occurrences: 16.

Scopes: nim-nemotron/baseline/serving, nim-nemotron/concurrency_2/serving, nim-nemotron/concurrency_4/serving, nim-nemotron/smoke/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json](../experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json) at `/records/serving/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 960927320e2f

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `960927320e2fe0ed7090ad46c06492736e13e88a4ff3d1993329c0dece8d8674` · occurrences: 5.

Scopes: vllm-nemotron/reasoning_on/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/serving/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS enrolled_subjects FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — 9615db4f08f2

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `9615db4f08f25c3adedfc9d45b76c732832d5aab81fbe5d8ab993aa791e62585` · occurrences: 25.

Scopes: vllm-qwen/awq_w4a16_cache_off/cache_off, vllm-qwen/awq_w4a16_cache_off/serving, vllm-qwen/awq_w4a16_cache_on/cold, vllm-qwen/awq_w4a16_cache_on/serving, vllm-qwen/awq_w4a16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T230608_780309Z.json](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) at `/records/serving/0`.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT sub.subject_id) AS enrolled_or_randomized_subjects
FROM subjects AS sub
JOIN sites AS st ON sub.site_id = st.site_id
WHERE sub.screening_status IN ('Enrolled', 'Randomized')
GROUP BY sub.treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — a5795147f78b

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `a5795147f78b69338398bd3e963cf203ed324ad0b29885dbe21de4b9add68a98` · occurrences: 5.

Scopes: nim-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/serving/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(*) FROM subjects WHERE screening_status IN ('Enrolled','Randomized') GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — a5d7ad1b4a99

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `a5d7ad1b4a994ce855750973efe540d1c59f13cfc1c19a0ccc3f4ba26922b54a` · occurrences: 43.

Scopes: nim-qwen/baseline/serving, nim-qwen/concurrency_2/serving, nim-qwen/concurrency_4/serving, nim-qwen/smoke/serving, vllm-qwen/fp16_cache_off/cache_off, vllm-qwen/fp16_cache_off/quality, vllm-qwen/fp16_cache_off/serving, vllm-qwen/fp16_cache_on/cold, vllm-qwen/fp16_cache_on/quality, vllm-qwen/fp16_cache_on/serving, vllm-qwen/fp16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/quality/0`.

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

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — b06678550d24

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `b06678550d243403d5faf2ea5278ceefb971ff6235c0ab945fe9b685f4bf5ca3` · occurrences: 1.

Scopes: nim-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(subject_id) FROM subjects WHERE screening_status IN ('Enrolled','Randomized') GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_001 — c26776c76043

How many subjects are Enrolled or Randomized by treatment arm?

SHA-256: `c26776c760436cf2a8d725665d4c2aad320c93393887700cbbbe5d89376d2770` · occurrences: 2.

Scopes: vllm-qwen/awq_w4a16_cache_off/quality, vllm-qwen/awq_w4a16_cache_on/quality.

First source: [experiment-tracker/mvp4/runs/20261106T230608_780309Z.json](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) at `/records/quality/0`.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT sub.subject_id) AS count
FROM subjects AS sub
WHERE sub.screening_status IN ('Enrolled', 'Randomized')
GROUP BY sub.treatment_arm;
```

**Reference SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS subject_count FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm ORDER BY treatment_arm
```

Reference assumptions: Count subjects whose screening_status is Enrolled or Randomized.

**Human judgment and rationale:** pending.

## eval_002 — 0c552b5b92a9

How many subjects have enrolled at each site?

SHA-256: `0c552b5b92a9ea2609a2a229662242b7eb3a3e6b26024e3e236aba5bc24a3d26` · occurrences: 1.

Scopes: vllm-nemotron/reasoning_on/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/quality/1`.

**Generated final SQL:**

```sql
SELECT st.site_id, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects
FROM sites AS st
LEFT JOIN subjects AS sub ON st.site_id = sub.site_id
AND sub.screening_status IN ('Enrolled','Randomized')
GROUP BY st.site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 19992e4b13a8

How many subjects have enrolled at each site?

SHA-256: `19992e4b13a890b930f9f81a3c49caeef331736dcfc71e734752c4f2923dd431` · occurrences: 5.

Scopes: nim-qwen/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/serving/1`.

**Generated final SQL:**

```sql
SELECT sites.site_id, COUNT(subjects.subject_id) AS enrolled_subjects FROM sites INNER JOIN subjects ON sites.site_id = subjects.site_id WHERE subjects.screening_status IN ('Enrolled', 'Randomized') GROUP BY sites.site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 2e159344c89f

How many subjects have enrolled at each site?

SHA-256: `2e159344c89faaafa9e7059483c6886bda2dcda8924297c25437b4f9f1ac7ba2` · occurrences: 5.

Scopes: nim-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/serving/1`.

**Generated final SQL:**

```sql
SELECT site_id, COUNT(*) AS enrolled_subjects FROM subjects WHERE screening_status IN ('Enrolled','Randomized') GROUP BY site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 2e245067c84d

How many subjects have enrolled at each site?

SHA-256: `2e245067c84da7c25e15d9737e49c5662564645d0a7a56bcb2c6427032d2f7ef` · occurrences: 21.

Scopes: nim-nemotron/baseline/serving, nim-nemotron/concurrency_2/serving, nim-nemotron/concurrency_4/serving, nim-nemotron/smoke/serving, vllm-nemotron/reasoning_on/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/serving/1`.

**Generated final SQL:**

```sql
SELECT site_id, COUNT(DISTINCT subject_id) AS enrolled_subjects FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 5fdc259e287e

How many subjects have enrolled at each site?

SHA-256: `5fdc259e287ecbcc33309d4384a63849291b30d74716af32f0760f6190a2646f` · occurrences: 1.

Scopes: vllm-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/quality/1`.

**Generated final SQL:**

```sql
SELECT s.site_id, COUNT(DISTINCT s.subject_id) AS subject_count FROM sites s JOIN subjects sub ON s.site_id = sub.site_id WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 61650dded521

How many subjects have enrolled at each site?

SHA-256: `61650dded52165bbdb6272b7475e72d04af7eb192b92904cdc5b5e7435d7b508` · occurrences: 7.

Scopes: vllm-nemotron/batch_capacity_1/quality, vllm-nemotron/concurrency_2/quality, vllm-nemotron/concurrency_4/quality, vllm-nemotron/fp16_baseline/quality, vllm-nemotron/output_2048/quality, vllm-nemotron/output_256/quality, vllm-nemotron/prefix_cache/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/quality/1`.

**Generated final SQL:**

```sql
SELECT st.site_id, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects FROM sites AS st LEFT JOIN subjects AS sub ON st.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY st.site_id
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 8f51aec81c6b

How many subjects have enrolled at each site?

SHA-256: `8f51aec81c6b0e7b2f34ae3d2bc37b8084cb9a450e110a1f7692d525c949b00a` · occurrences: 1.

Scopes: nim-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/quality/1`.

**Generated final SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS subject_count FROM sites s LEFT JOIN subjects sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled','Randomized') GROUP BY s.site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — 99167a5e217e

How many subjects have enrolled at each site?

SHA-256: `99167a5e217eea41b210aa1dd7119fb738fb79940d3422f8c0c1d269d0e84975` · occurrences: 31.

Scopes: nim-qwen/baseline/quality, nim-qwen/concurrency_2/quality, nim-qwen/concurrency_4/quality, nim-qwen/smoke/quality, vllm-qwen/awq_w4a16_cache_off/cache_off, vllm-qwen/awq_w4a16_cache_off/quality, vllm-qwen/awq_w4a16_cache_off/serving, vllm-qwen/awq_w4a16_cache_on/cold, vllm-qwen/awq_w4a16_cache_on/quality, vllm-qwen/awq_w4a16_cache_on/serving, vllm-qwen/awq_w4a16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T230608_780309Z.json](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) at `/records/quality/1`.

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

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — a0e5a7d079c9

How many subjects have enrolled at each site?

SHA-256: `a0e5a7d079c9b7201d07cc4575621e2e6a97451ce162bc3a78ef6b901486f333` · occurrences: 45.

Scopes: vllm-nemotron/batch_capacity_1/serving, vllm-nemotron/concurrency_2/serving, vllm-nemotron/concurrency_4/serving, vllm-nemotron/fp16_baseline/serving, vllm-nemotron/output_2048/serving, vllm-nemotron/output_256/serving, vllm-nemotron/prefix_cache/cold, vllm-nemotron/prefix_cache/serving, vllm-nemotron/prefix_cache/warm.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/serving/1`.

**Generated final SQL:**

```sql
SELECT st.site_id, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects FROM sites AS st JOIN subjects AS sub ON st.site_id = sub.site_id WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY st.site_id
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — bd5f0043db6c

How many subjects have enrolled at each site?

SHA-256: `bd5f0043db6ca33c3fdfe93dc726ffc93791675fdd630a85a4b9c8dee9625343` · occurrences: 2.

Scopes: vllm-qwen/fp16_cache_off/quality, vllm-qwen/fp16_cache_on/quality.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/quality/1`.

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

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — c1407d33fb56

How many subjects have enrolled at each site?

SHA-256: `c1407d33fb567efdf5690b9c2eca8f7966d26e04239662c8a2c83ff2c6982d51` · occurrences: 41.

Scopes: nim-qwen/baseline/serving, nim-qwen/concurrency_2/serving, nim-qwen/concurrency_4/serving, nim-qwen/smoke/serving, vllm-qwen/fp16_cache_off/cache_off, vllm-qwen/fp16_cache_off/serving, vllm-qwen/fp16_cache_on/cold, vllm-qwen/fp16_cache_on/serving, vllm-qwen/fp16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/serving/1`.

**Generated final SQL:**

```sql
SELECT sites.site_id, COUNT(DISTINCT subjects.subject_id) AS enrolled_subjects
FROM sites
LEFT JOIN subjects ON sites.site_id = subjects.site_id
WHERE subjects.screening_status IN ('Enrolled', 'Randomized')
GROUP BY sites.site_id;
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — e676b7391a8b

How many subjects have enrolled at each site?

SHA-256: `e676b7391a8b13252d623f54cc332cc656599e087bf6f7bdef7e7ededafa4276` · occurrences: 5.

Scopes: vllm-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/serving/1`.

**Generated final SQL:**

```sql
SELECT site_id, COUNT(*) AS enrolled_subjects FROM subjects WHERE screening_status IN ('Enrolled', 'Randomized') GROUP BY site_id
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_002 — eaf611b7cdc4

How many subjects have enrolled at each site?

SHA-256: `eaf611b7cdc4d037a7329bf7755e73fdf731b1a26813d8dbfbbcc62b6d9639bc` · occurrences: 1.

Scopes: nim-qwen/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/quality/1`.

**Generated final SQL:**

```sql
SELECT sites.site_id, COUNT(subjects.subject_id) AS num_enrolled_subjects FROM sites LEFT JOIN subjects ON sites.site_id = subjects.site_id AND subjects.screening_status IN ('Enrolled', 'Randomized') GROUP BY sites.site_id
```

**Reference SQL:**

```sql
SELECT s.site_id, COUNT(sub.subject_id) AS enrolled_subjects FROM sites AS s LEFT JOIN subjects AS sub ON s.site_id = sub.site_id AND sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY s.site_id ORDER BY s.site_id
```

Reference assumptions: Treat Enrolled and Randomized as enrolled; include sites with zero qualifying subjects.

**Human judgment and rationale:** pending.

## eval_005 — 0116080d4ed6

How many scheduled visits were missed last month?

SHA-256: `0116080d4ed600a92fe7cc4e3a47b9328886984c7df2a61f87125d3756818d76` · occurrences: 4.

Scopes: nim-qwen/baseline/quality, nim-qwen/concurrency_2/quality, nim-qwen/concurrency_4/quality, nim-qwen/smoke/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json](../experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(DISTINCT visit_id) FROM visits WHERE status = 'Missed' AND planned_date >= DATE('2023-01-01') AND planned_date < DATE('2023-02-01')
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 343549ee9328

How many scheduled visits were missed last month?

SHA-256: `343549ee9328decb711cb2dd122020ee8cb6d58e8ad706974edd86e384cf0010` · occurrences: 1.

Scopes: nim-nemotron/concurrency_4/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205940_074402Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205940_074402Z.json) at `/records/serving/6`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= DATEFROMPARTS(EXTRACT(YEAR FROM CURRENT_DATE)-1, EXTRACT(MONTH FROM CURRENT_DATE)-1, 1) AND planned_date < DATEFROMPARTS(EXTRACT(YEAR FROM CURRENT_DATE), EXTRACT(MONTH FROM CURRENT_DATE), 1);
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 3fe5528b25cd

How many scheduled visits were missed last month?

SHA-256: `3fe5528b25cdb7502cd0dc2997a264d07540efae4095ddbde9625113dcf497bf` · occurrences: 1.

Scopes: vllm-nemotron/reasoning_on/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(visit_id) AS missed_visits FROM visits WHERE status='Missed' AND planned_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND planned_date < DATE_TRUNC('month', CURRENT_DATE);
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 485f045de50e

How many scheduled visits were missed last month?

SHA-256: `485f045de50e2b08dd1afd715e1e03c470c8b1ebf0c205b50b071c5658be6887` · occurrences: 7.

Scopes: vllm-nemotron/batch_capacity_1/quality, vllm-nemotron/concurrency_2/quality, vllm-nemotron/concurrency_4/quality, vllm-nemotron/fp16_baseline/quality, vllm-nemotron/output_2048/quality, vllm-nemotron/output_256/quality, vllm-nemotron/prefix_cache/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(DISTINCT visit_id) AS missed_visits FROM visits WHERE actual_date IS NULL AND planned_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < DATE_TRUNC('month', CURRENT_DATE)
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 4943eb953b16

How many scheduled visits were missed last month?

SHA-256: `4943eb953b16cf3d5d86010207001d009b77279a08eb6a45b9613b1a0a51d8b5` · occurrences: 4.

Scopes: nim-nemotron/baseline/quality, nim-nemotron/concurrency_2/quality, nim-nemotron/concurrency_4/quality, nim-nemotron/smoke/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json](../experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits
FROM visits
WHERE status = 'Missed'
  AND planned_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
  AND planned_date < DATE_TRUNC('month', CURRENT_DATE);
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 553685f3b29b

How many scheduled visits were missed last month?

SHA-256: `553685f3b29b0bdec7876513800473723fb97cd63fbca7dd82ba79a8d89d2975` · occurrences: 45.

Scopes: vllm-nemotron/batch_capacity_1/serving, vllm-nemotron/concurrency_2/serving, vllm-nemotron/concurrency_4/serving, vllm-nemotron/fp16_baseline/serving, vllm-nemotron/output_2048/serving, vllm-nemotron/output_256/serving, vllm-nemotron/prefix_cache/cold, vllm-nemotron/prefix_cache/serving, vllm-nemotron/prefix_cache/warm.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(DISTINCT v.visit_id) AS missed_visits FROM visits AS v WHERE v.status = 'Missed' AND v.planned_date >= CURRENT_DATE - INTERVAL '1 month' AND v.planned_date < CURRENT_DATE
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 58363c8ae808

How many scheduled visits were missed last month?

SHA-256: `58363c8ae8080e7efbec281bcdb4dbb4da355b17c91662a20af446a5d067905e` · occurrences: 5.

Scopes: vllm-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT visit_name, COUNT(*) AS missed_count FROM visits WHERE status = 'Missed' AND actual_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' AND actual_date < DATE_TRUNC('month', CURRENT_DATE) GROUP BY visit_name
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 6cec4a54d717

How many scheduled visits were missed last month?

SHA-256: `6cec4a54d71788acb02e143386baa3865cd49e48cfda24f293a0fde4fa4c2751` · occurrences: 1.

Scopes: vllm-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS count FROM visits WHERE status='Missed' AND planned_date >= DATE('now', '-1 month') AND planned_date < DATE('now', '0 month')
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 734e48a79135

How many scheduled visits were missed last month?

SHA-256: `734e48a79135d4b39c5b8a55ca96a5a6b391c22c51789a83a7007c44560be8c9` · occurrences: 5.

Scopes: nim-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) FROM visits WHERE status='Missed' AND planned_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month' AND planned_date < DATE_TRUNC('month', CURRENT_DATE);
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 82a629145b0d

How many scheduled visits were missed last month?

SHA-256: `82a629145b0d01b6b8951e36beeef0269d596e3354f45f8a49934257c4a54010` · occurrences: 41.

Scopes: nim-qwen/baseline/serving, nim-qwen/concurrency_2/serving, nim-qwen/concurrency_4/serving, nim-qwen/smoke/serving, vllm-qwen/fp16_cache_off/cache_off, vllm-qwen/fp16_cache_off/serving, vllm-qwen/fp16_cache_on/cold, vllm-qwen/fp16_cache_on/serving, vllm-qwen/fp16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= DATE('now', '-1 month') AND planned_date < DATE('now')
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 92c68fdb4f2f

How many scheduled visits were missed last month?

SHA-256: `92c68fdb4f2f3d9fd8078219a35997c6cecce15fea420542701d955d0dbff003` · occurrences: 1.

Scopes: nim-qwen/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) FROM visits WHERE status = 'Missed' AND planned_date >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 MONTH' AND planned_date < DATE_TRUNC('month', CURRENT_DATE)
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — 97d8ce22ddff

How many scheduled visits were missed last month?

SHA-256: `97d8ce22ddff91bc88d9871aa40963c1f6f0e2cd3dc6a4f0ca7780cc3fb5ed72` · occurrences: 14.

Scopes: nim-nemotron/baseline/serving, nim-nemotron/concurrency_2/serving, nim-nemotron/concurrency_4/serving, nim-nemotron/smoke/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json](../experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND DATE_TRUNC('month', actual_date) = DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month';
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — a903c84756de

How many scheduled visits were missed last month?

SHA-256: `a903c84756de915ce13ed2dbed020f80060035633aaecbc43e7efefa7cc46a81` · occurrences: 1.

Scopes: nim-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) FROM visits WHERE status='Missed' AND planned_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') AND planned_date < DATE_TRUNC('month', CURRENT_DATE);
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — b1cb6f13a955

How many scheduled visits were missed last month?

SHA-256: `b1cb6f13a9557df4606c456d9f5c8a7b330c16a6645c9f5496821d6c0e4a027b` · occurrences: 2.

Scopes: vllm-qwen/fp16_cache_off/quality, vllm-qwen/fp16_cache_on/quality.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(visit_id) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= DATE('now', '-1 month')
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — bd6ab6967908

How many scheduled visits were missed last month?

SHA-256: `bd6ab69679089244ad7d5e9f12a5871448e9c17df4138bfbdad867e2b9a16be3` · occurrences: 2.

Scopes: vllm-qwen/awq_w4a16_cache_off/quality, vllm-qwen/awq_w4a16_cache_on/quality.

First source: [experiment-tracker/mvp4/runs/20261106T230608_780309Z.json](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) at `/records/quality/2`.

**Generated final SQL:**

```sql
SELECT COUNT(visit_id) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date BETWEEN DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH) AND CURRENT_DATE - INTERVAL 1 DAY;
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — c49b96707014

How many scheduled visits were missed last month?

SHA-256: `c49b96707014e40dcdf3c95b3aaa5f9edaa18762946e1b78a39c33fda2e4edf9` · occurrences: 1.

Scopes: nim-nemotron/concurrency_4/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205940_074402Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205940_074402Z.json) at `/records/serving/10`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status='Missed' AND MONTH(planned_date)=EXTRACT(MONTH FROM CURRENT_DATE)-1 AND YEAR(planned_date)=YEAR(CURRENT_DATE);
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — c7f00596aa05

How many scheduled visits were missed last month?

SHA-256: `c7f00596aa05b7adb2e6235523391423ef40e5c9c60bea023c1ca53935c5ad80` · occurrences: 5.

Scopes: vllm-nemotron/reasoning_on/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE actual_date IS NULL AND planned_date BETWEEN DATE('now') - INTERVAL '1 month' AND DATE('now') - INTERVAL '1 day';
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — d4435181a061

How many scheduled visits were missed last month?

SHA-256: `d4435181a0613724ad695f58aebcfdb9477f1c0cecdd8f9fcb3554d5974c2094` · occurrences: 5.

Scopes: nim-qwen/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) FROM visits WHERE status = 'Missed' AND planned_date >= DATE('2023-01-01') AND planned_date < DATE('2023-02-01');
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_005 — ee5ddcb7d322

How many scheduled visits were missed last month?

SHA-256: `ee5ddcb7d3220bd7f7d99a3e42eff9078725a862cb206323b3c40d6c1c7ad737` · occurrences: 25.

Scopes: vllm-qwen/awq_w4a16_cache_off/cache_off, vllm-qwen/awq_w4a16_cache_off/serving, vllm-qwen/awq_w4a16_cache_on/cold, vllm-qwen/awq_w4a16_cache_on/serving, vllm-qwen/awq_w4a16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T230608_780309Z.json](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) at `/records/serving/2`.

**Generated final SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= DATE('2023-01-01') AND planned_date < DATE('2023-02-01')
```

**Reference SQL:**

```sql
SELECT COUNT(*) AS missed_visits FROM visits WHERE status = 'Missed' AND planned_date >= date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH AND planned_date < date_trunc('month', CURRENT_DATE)
```

Reference assumptions: Last month means the previous calendar month; use planned_date and status Missed.

**Human judgment and rationale:** pending.

## eval_007 — 0b067ad72524

Compare serious adverse-event rates across treatment arms.

SHA-256: `0b067ad725242ef3e7ae5eb48946e68546cbd504012c9ec5468e252f801748f2` · occurrences: 4.

Scopes: vllm-qwen/awq_w4a16_cache_off/quality, vllm-qwen/awq_w4a16_cache_on/quality, vllm-qwen/fp16_cache_off/quality, vllm-qwen/fp16_cache_on/quality.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/quality/3`.

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

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 0e4f0861e110

Compare serious adverse-event rates across treatment arms.

SHA-256: `0e4f0861e11065be65c202c7345c807f8915cef969e2d99e0e21b52de97de6d4` · occurrences: 5.

Scopes: vllm-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS serious_adverse_event_count, COUNT(*) FILTER (WHERE serious = true) AS serious_adverse_event_count_total, COUNT(*) FILTER (WHERE serious = true) / COUNT(*) AS serious_adverse_event_rate FROM subjects JOIN adverse_events ON subjects.subject_id = adverse_events.subject_id WHERE adverse_events.serious = true GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 1855b61034e6

Compare serious adverse-event rates across treatment arms.

SHA-256: `1855b61034e65aba80c4e0cd819b2eba3d7dcaf959ec3851709e1dd0af63820c` · occurrences: 25.

Scopes: vllm-qwen/awq_w4a16_cache_off/cache_off, vllm-qwen/awq_w4a16_cache_off/serving, vllm-qwen/awq_w4a16_cache_on/cold, vllm-qwen/awq_w4a16_cache_on/serving, vllm-qwen/awq_w4a16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T230608_780309Z.json](../experiment-tracker/mvp4/runs/20261106T230608_780309Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS serious_adverse_events, COUNT(DISTINCT sub.subject_id) AS total_subjects, (COUNT(DISTINCT ae.subject_id) * 1.0 / COUNT(DISTINCT sub.subject_id)) AS serious_adverse_event_rate
FROM adverse_events AS ae
JOIN subjects AS sub ON ae.subject_id = sub.subject_id
WHERE ae.serious = TRUE
GROUP BY sub.treatment_arm
ORDER BY serious_adverse_event_rate DESC;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 3080af8b1179

Compare serious adverse-event rates across treatment arms.

SHA-256: `3080af8b11798027df590545f7160647cca7431bb64235a26e2136949c85e152` · occurrences: 1.

Scopes: nim-qwen/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
WITH filtered_subjects AS (SELECT subjects.treatment_arm, COUNT(*) AS total_subjects FROM subjects WHERE subjects.treatment_arm IS NOT NULL AND screening_status IN ('Enrolled', 'Randomized') GROUP BY subjects.treatment_arm), serious_events AS (SELECT adverse_events.subject_id, adverse_events.onset_date FROM adverse_events WHERE adverse_events.serious = TRUE AND adverse_events.subject_id IN (SELECT subjects.subject_id FROM subjects WHERE subjects.treatment_arm IS NOT NULL AND screening_status IN ('Enrolled', 'Randomized'))) SELECT filtered_subjects.treatment_arm, COUNT(serious_events.subject_id) * 1.0 / total_subjects AS serious_rate FROM filtered_subjects INNER JOIN serious_events ON filtered_subjects.treatment_arm = serious_events.subject_id GROUP BY filtered_subjects.treatment_arm, total_subjects;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 48e60efd10a7

Compare serious adverse-event rates across treatment arms.

SHA-256: `48e60efd10a7c087d389b122241cecb5d7b54a46813e2054972c62de954de92f` · occurrences: 45.

Scopes: vllm-nemotron/batch_capacity_1/serving, vllm-nemotron/concurrency_2/serving, vllm-nemotron/concurrency_4/serving, vllm-nemotron/fp16_baseline/serving, vllm-nemotron/output_2048/serving, vllm-nemotron/output_256/serving, vllm-nemotron/prefix_cache/cold, vllm-nemotron/prefix_cache/serving, vllm-nemotron/prefix_cache/warm.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS serious_adverse_events, COUNT(DISTINCT subject_id) FILTER (WHERE serious) AS serious_adverse_events_total, COUNT(DISTINCT subject_id) FILTER (WHERE serious) / COUNT(DISTINCT subject_id) AS serious_adverse_event_rate FROM adverse_events JOIN subjects ON adverse_events.subject_id = subjects.subject_id GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 5ce0fcc43dbd

Compare serious adverse-event rates across treatment arms.

SHA-256: `5ce0fcc43dbd58f7aca1c7b710119068f6359acf585c55d5e22f3fe4fc844886` · occurrences: 5.

Scopes: nim-nemotron/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS serious_adv_event_count
FROM subjects
JOIN adverse_events ON subjects.subject_id = adverse_events.subject_id
WHERE serious = true
GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 68c450fbe635

Compare serious adverse-event rates across treatment arms.

SHA-256: `68c450fbe6359a3217f8b80fd6ab45a209cb3c70c85c2e0104b6ddb8e0c87dd7` · occurrences: 1.

Scopes: vllm-nemotron/reasoning_on/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm,
       COUNT(CASE WHEN serious THEN 1 END) AS serious_events,
       COUNT(*) AS total_subjects,
       ROUND(COUNT(CASE WHEN serious THEN 1 END)::DOUBLE / COUNT(*)::DOUBLE, 4) AS serious_rate
FROM subjects
WHERE treatment_arm IS NOT NULL
  AND screening_status IN ('Enrolled','Randomized')
GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 6ba4bc3ebd64

Compare serious adverse-event rates across treatment arms.

SHA-256: `6ba4bc3ebd64f7ab9a8878ec4eaa3c74f86200c4277f87051ff0e96b2ac5f7be` · occurrences: 4.

Scopes: nim-nemotron/baseline/quality, nim-nemotron/concurrency_2/quality, nim-nemotron/concurrency_4/quality, nim-nemotron/smoke/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json](../experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm,
       COUNT(DISTINCT ae.subject_id) AS serious_adv_events,
       COUNT(DISTINCT sub.subject_id) AS enrolled_subjects
FROM subjects sub
LEFT JOIN adverse_events ae ON ae.subject_id = sub.subject_id AND ae.serious = true
WHERE sub.treatment_arm IS NOT NULL
  AND sub.screening_status IN ('Enrolled','Randomized')
GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 88b9b1be2673

Compare serious adverse-event rates across treatment arms.

SHA-256: `88b9b1be2673565f8a67177d7c164e7c714b739b0716d7daab9df1897e22d783` · occurrences: 1.

Scopes: vllm-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json](../experiment-tracker/mvp4/nemotron/20261107T024417_765664Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(CASE WHEN serious = true THEN 1 END) / COUNT(*) AS serious_adverse_event_rate FROM subjects WHERE treatment_arm IS NOT NULL AND treatment_arm <> 'Not Assigned' AND screening_status IN ('Enrolled', 'Randomized') GROUP BY treatment_arm
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 8de729c9e4eb

Compare serious adverse-event rates across treatment arms.

SHA-256: `8de729c9e4ebdb97e7a6d599d2889ae28bd225f0851e6fbef0499a3544c71338` · occurrences: 5.

Scopes: vllm-nemotron/reasoning_on/serving.

First source: [experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json](../experiment-tracker/mvp4/nemotron/20261107T033328_571911Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subject_id) AS enrolled_subjects, COUNT(CASE WHEN severity='Severe' AND serious THEN 1 END) AS serious_events, ROUND(COUNT(CASE WHEN severity='Severe' AND serious THEN 1 END)::DOUBLE / COUNT(DISTINCT subject_id), 2) AS serious_rate FROM subjects s LEFT JOIN adverse_events ae ON s.subject_id = ae.subject_id WHERE s.screening_status IN ('Enrolled','Randomized') GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — 9423fb59b3c7

Compare serious adverse-event rates across treatment arms.

SHA-256: `9423fb59b3c7cf56eb6a06b444553e10ee33ff889aba239be5057713d370513b` · occurrences: 4.

Scopes: nim-qwen/baseline/quality, nim-qwen/concurrency_2/quality, nim-qwen/concurrency_4/quality, nim-qwen/smoke/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json](../experiment-tracker/mvp4/nim/runs/20261107T215439_693958Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT sub.subject_id) AS total_subjects, COUNT(DISTINCT ae.subject_id) AS serious_adverse_events
FROM subjects AS sub
JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id
WHERE ae.serious = true
  AND sub.treatment_arm IS NOT NULL
GROUP BY sub.treatment_arm
ORDER BY serious_adverse_events / total_subjects DESC;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — ae3cd07fbd4c

Compare serious adverse-event rates across treatment arms.

SHA-256: `ae3cd07fbd4cd902eea6b9919b8e06237862b1d8e593cc19acb6f5dddb4e4717` · occurrences: 16.

Scopes: nim-nemotron/baseline/serving, nim-nemotron/concurrency_2/serving, nim-nemotron/concurrency_4/serving, nim-nemotron/smoke/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json](../experiment-tracker/mvp4/nim/runs/20261107T204145_046415Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm,
       COUNT(DISTINCT sub.subject_id) AS enrolled_subjects,
       COUNT(CASE WHEN ae.severity='Severe' AND ae.severe THEN 1 END) AS serious_events,
       ROUND(COUNT(CASE WHEN ae.severity='Severe' AND ae.severe THEN 1 END)::DOUBLE / COUNT(DISTINCT sub.subject_id), 2) AS serious_rate
FROM subjects AS sub
LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id
WHERE sub.screening_status IN ('Enrolled', 'Randomized')
GROUP BY sub.treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — b09a3a21c434

Compare serious adverse-event rates across treatment arms.

SHA-256: `b09a3a21c43445da3652a3aa5c6d76c4e84c1709aea32bc7ad17bba885be8736` · occurrences: 5.

Scopes: nim-qwen/schema_retrieved/serving.

First source: [experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json](../experiment-tracker/mvp4/nim/runs/20261107T220801_623823Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(*) AS total_subjects, SUM(CASE WHEN serious THEN 1 ELSE 0 END) AS serious_adverse_events FROM adverse_events JOIN subjects ON adverse_events.subject_id = subjects.subject_id GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — e23fe05fe155

Compare serious adverse-event rates across treatment arms.

SHA-256: `e23fe05fe1551a2d493a6199a16cddee9e7d66e86138e3eed3bd86eed5d5bf49` · occurrences: 1.

Scopes: nim-nemotron/schema_retrieved/quality.

First source: [experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json](../experiment-tracker/mvp4/nim/runs/20261107T205138_075077Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
WITH ae AS (
  SELECT e.subject_id, e.serious, e.event_term, e.severity, e.onset_date, e.outcome
  FROM adverse_events e
  JOIN subjects s ON e.subject_id = s.subject_id
  WHERE e.serious = true AND s.treatment_arm <> 'Not Assigned'
)
SELECT treatment_arm, COUNT(*) / COUNT(s.subject_id) AS serious_event_rate
FROM ae a
JOIN subjects s ON a.subject_id = s.subject_id
GROUP BY treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — efeb1919cabb

Compare serious adverse-event rates across treatment arms.

SHA-256: `efeb1919cabb75c887eefd8b79c897703c669cee59210e47e081023ebe4fe2db` · occurrences: 6.

Scopes: vllm-nemotron/batch_capacity_1/quality, vllm-nemotron/concurrency_2/quality, vllm-nemotron/concurrency_4/quality, vllm-nemotron/fp16_baseline/quality, vllm-nemotron/output_2048/quality, vllm-nemotron/prefix_cache/quality.

First source: [experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json](../experiment-tracker/mvp4/nemotron/20261107T002501_223219Z.json) at `/records/quality/3`.

**Generated final SQL:**

```sql
SELECT treatment_arm, COUNT(DISTINCT subjects.subject_id) AS serious_adverse_events, COUNT(DISTINCT subjects.subject_id) FILTER (WHERE serious) AS serious_adverse_events_total, COUNT(DISTINCT subjects.subject_id) FILTER (WHERE screening_status IN ('Enrolled', 'Randomized')) AS enrolled_subjects, COUNT(DISTINCT subjects.subject_id) FILTER (WHERE screening_status IN ('Enrolled', 'Randomized')) / COUNT(DISTINCT subjects.subject_id) FILTER (WHERE screening_status IN ('Enrolled', 'Randomized')) AS rate FROM subjects JOIN adverse_events ON subjects.subject_id = adverse_events.subject_id WHERE serious = true AND treatment_arm IS NOT NULL GROUP BY treatment_arm HAVING COUNT(DISTINCT subjects.subject_id) FILTER (WHERE serious) IS NOT NULL
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.

## eval_007 — f9ded738a71e

Compare serious adverse-event rates across treatment arms.

SHA-256: `f9ded738a71edfdc6f12eb510f22053d06adaaa3f29a23fe789f437a174acd13` · occurrences: 41.

Scopes: nim-qwen/baseline/serving, nim-qwen/concurrency_2/serving, nim-qwen/concurrency_4/serving, nim-qwen/smoke/serving, vllm-qwen/fp16_cache_off/cache_off, vllm-qwen/fp16_cache_off/serving, vllm-qwen/fp16_cache_on/cold, vllm-qwen/fp16_cache_on/serving, vllm-qwen/fp16_cache_on/warm.

First source: [experiment-tracker/mvp4/runs/20261106T221400_918883Z.json](../experiment-tracker/mvp4/runs/20261106T221400_918883Z.json) at `/records/serving/3`.

**Generated final SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS serious_adverse_events, COUNT(DISTINCT sub.subject_id) AS total_enrolled_subjects, COUNT(DISTINCT ae.subject_id) * 1.0 / COUNT(DISTINCT sub.subject_id) AS serious_adverse_event_rate
FROM subjects AS sub
JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id
WHERE ae.serious = TRUE
  AND sub.screening_status IN ('Enrolled', 'Randomized')
GROUP BY sub.treatment_arm;
```

**Reference SQL:**

```sql
SELECT sub.treatment_arm, COUNT(DISTINCT ae.subject_id) AS affected_subjects, COUNT(DISTINCT sub.subject_id) AS enrolled_subjects, ROUND(100.0 * COUNT(DISTINCT ae.subject_id) / NULLIF(COUNT(DISTINCT sub.subject_id), 0), 2) AS serious_ae_percent FROM subjects AS sub LEFT JOIN adverse_events AS ae ON sub.subject_id = ae.subject_id AND ae.serious = true WHERE sub.screening_status IN ('Enrolled', 'Randomized') GROUP BY sub.treatment_arm ORDER BY sub.treatment_arm
```

Reference assumptions: Rate means percent of Enrolled or Randomized subjects with at least one serious adverse event, not events per subject.

**Human judgment and rationale:** pending.
