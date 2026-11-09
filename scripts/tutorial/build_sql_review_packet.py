"""Inventory distinct recorded SQL outputs without entering human judgments."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.agent.analyst import ClinicalTrialAnalystAgent
from clinical_trial_mvp.tutorial.replay import RecordedStudy


def build(root=ROOT):
    archive = RecordedStudy(root)
    cases = {c['id']: c for c in json.loads((root / 'evals/evaluations.json').read_text())}
    entries, no_sql = {}, []
    def add(record, occurrence, parse=False):
        attempts = record.get('attempts', [])
        sql = (attempts[-1].get('sql') or '').strip() if attempts else ''
        parse_error = None
        if parse and not sql:
            try:
                sql = ClinicalTrialAnalystAgent._parse_sql_json(record.get('content') or '')['sql'].strip()
            except (ValueError, KeyError, TypeError) as exc:
                parse_error = str(exc)
        occurrence = {**occurrence, 'case_id': record['case_id'], 'repeat': record.get('repeat'),
                      'machine_scores': record.get('scores'), 'plan_error': record.get('plan_error')}
        if not sql:
            no_sql.append({**occurrence, 'parse_error': parse_error,
                           'reason': 'No final executable SQL string recorded/parsed; not a semantic SQL verdict.'})
            return
        digest = hashlib.sha256(sql.encode()).hexdigest()
        key = record['case_id'], digest
        if key not in entries:
            case = cases[record['case_id']]
            entries[key] = dict(case_id=key[0], question=case['question'], sql=sql, sql_sha256=digest,
                reference_sql=case['reference_sql'], reference_assumptions=case.get('reference_assumptions'),
                success_criteria=case.get('success_criteria'), occurrences=[],
                reviewer=None, reviewed_at=None, correct=None, rationale=None)
        entries[key]['occurrences'].append(occurrence)
    for study, variants in archive.index['studies'].items():
        runtime, model = study.split('-', 1)
        for name, source in variants.items():
            result = archive.result(runtime, model, name)
            records = result.get('records', {})
            if not isinstance(records, dict):
                continue
            for section in ('quality', 'serving'):
                for i, row in enumerate(records.get(section, [])):
                    add(row, dict(study=study, variant=name, source=source,
                        pointer=f'/records/{section}/{i}', path=section), parse=section == 'serving')
            for i, row in enumerate(result.get('cache_study', {}).get('records', [])):
                add(row, dict(study=study, variant=name, source=source,
                    pointer=f'/cache_study/records/{i}', path='paired-target', phase=row['phase']))
    return dict(description='Human-review queue, not judgments. Exact final SQL after outer whitespace stripping; deduplicated only by case ID + SHA-256. All occurrences retain their source JSON pointers. No-SQL/parse failures remain explicit.',
                reviewed_outputs=0, distinct_outputs=len(entries),
                case_ids=sorted({key[0] for key in entries}),
                outputs=sorted(entries.values(), key=lambda e:(e['case_id'],e['sql_sha256'])),
                no_sql_outputs=no_sql)


def render(packet):
    text = f'''# Distinct-output SQL review packet

**Human review pending: 0/{packet['distinct_outputs']} distinct SQL outputs reviewed across four questions.**
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

'''
    for e in packet['outputs']:
        scopes=sorted({f"{o['study']}/{o['variant']}/{o.get('phase',o['path'])}" for o in e['occurrences']})
        first=e['occurrences'][0]
        text += (f"## {e['case_id']} — {e['sql_sha256'][:12]}\n\n{e['question']}\n\n"
                 f"SHA-256: `{e['sql_sha256']}` · occurrences: {len(e['occurrences'])}.\n\n"
                 f"Scopes: {', '.join(scopes)}.\n\n"
                 f"First source: [{first['source']}](../{first['source']}) at `{first['pointer']}`.\n\n"
                 f"**Generated final SQL:**\n\n```sql\n{e['sql']}\n```\n\n"
                 f"**Reference SQL:**\n\n```sql\n{e['reference_sql']}\n```\n\n"
                 f"Reference assumptions: {e['reference_assumptions']}\n\n"
                 "**Human judgment and rationale:** pending.\n\n")
    return text.rstrip() + '\n'


if __name__ == '__main__':
    packet=build()
    (ROOT/'tutorial/sql_review_distinct.json').write_text(json.dumps(packet,indent=2)+'\n')
    (ROOT/'tutorial/SQL_REVIEW_DISTINCT.md').write_text(render(packet))
    print(f"Prepared {packet['distinct_outputs']} distinct SQL outputs; {len(packet['no_sql_outputs'])} no-SQL outputs; no judgments entered")
