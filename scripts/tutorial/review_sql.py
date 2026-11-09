"""Inspect review coverage or record one explicitly supplied human review."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from clinical_trial_mvp.tutorial.reviews import validate_reviews


def status(root=ROOT):
    root = Path(root)
    packet = json.loads((root / 'tutorial/sql_review_distinct.json').read_text())
    allowed = {(e['case_id'], e['sql_sha256']) for e in packet['outputs']}
    reviews = json.loads((root / 'tutorial/semantic_reviews.json').read_text())['reviews']
    index = validate_reviews(reviews, allowed=allowed)
    return dict(distinct_outputs=len(allowed), reviewed_outputs=len(index),
                unreviewed_outputs=len(allowed)-len(index),
                correct_outputs=sum(r['correct'] for r in index.values()),
                cases=[dict(case_id=case, total=sum(key[0] == case for key in allowed),
                            reviewed=sum(key[0] == case for key in index))
                       for case in sorted({key[0] for key in allowed})],
                limitation='Counts are exact distinct SQL judgments, not independent tasks or clinical accuracy.')


def record(entry, root=ROOT):
    root = Path(root)
    status(root)  # Validate current records before appending; no partial update.
    packet = json.loads((root / 'tutorial/sql_review_distinct.json').read_text())
    allowed = {(e['case_id'], e['sql_sha256']) for e in packet['outputs']}
    path = root / 'tutorial/semantic_reviews.json'
    document = json.loads(path.read_text())
    document['reviews'].append(entry)
    validate_reviews(document['reviews'], allowed=allowed)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(document, indent=2) + '\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return status(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, help='JSON object containing an actual human judgment; never generated automatically')
    args = parser.parse_args()
    result = record(json.loads(args.record.read_text())) if args.record else status()
    print(json.dumps(result, indent=2))
