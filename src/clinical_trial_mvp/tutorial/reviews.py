"""Validate attributed human judgments; never infer a verdict from reference matches."""
from datetime import datetime
import re


def validate_reviews(reviews, *, allowed=None):
    if not isinstance(reviews, list):
        raise ValueError('reviews must be a list')
    index = {}
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError('Each semantic review must be an object')
        for field in ('case_id', 'reviewer', 'reviewed_at', 'rationale'):
            if not isinstance(review.get(field), str) or not review[field].strip():
                raise ValueError('Semantic reviews require nonempty ' + field)
        if type(review.get('correct')) is not bool:
            raise ValueError('Semantic reviews require a boolean verdict')
        if not isinstance(review.get('sql_sha256'), str) or not re.fullmatch(r'[0-9a-f]{64}', review['sql_sha256']):
            raise ValueError('Semantic reviews require the exact SQL SHA-256')
        try:
            datetime.fromisoformat(review['reviewed_at'].replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('reviewed_at must be an ISO date/time') from exc
        key = review['case_id'], review['sql_sha256']
        if key in index:
            raise ValueError('Duplicate semantic review')
        if allowed is not None and key not in allowed:
            raise ValueError('Review case/hash is absent from the distinct-output packet')
        index[key] = review
    return index
