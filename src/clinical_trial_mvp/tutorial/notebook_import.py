"""Recover displayed benchmark summaries without executing source notebook cells."""

import ast
import hashlib
import json
import math
from pathlib import Path
import re

from clinical_trial_mvp.serving.portable import portable_text, portable_value

FORMAT = 'tutorial-notebook-summary-v1'


def _text(value):
    return ''.join(value) if isinstance(value, list) else str(value)


def _number(value):
    if value.strip().startswith(('—', '-')):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError('Invalid displayed measurement')
    return number


def _ratio(value):
    matched = re.fullmatch(r'(\d+)/(\d+)', value.strip())
    if not matched:
        raise ValueError('Expected a displayed count/total')
    numerator, denominator = map(int, matched.groups())
    if not 0 <= numerator <= denominator or denominator < 1:
        raise ValueError('Invalid displayed count/total')
    return numerator, denominator


def _static_configuration(notebook, runtime):
    """Read literal runtime overrides; never evaluate a cell or expression."""
    result = {}
    for cell in notebook['cells']:
        if cell.get('cell_type') != 'code':
            continue
        try:
            tree = ast.parse(_text(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'update':
                continue
            target = node.func.value
            if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'configs' and isinstance(target.slice, ast.Constant)
                    and target.slice.value == 'rtx_' + runtime):
                continue
            for item in node.keywords:
                try:
                    value = ast.literal_eval(item.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if item.arg in {'hardware', 'dtype', 'provenance', 'context', 'model', 'profile', 'repeats'}:
                    result[item.arg] = value
    return result


def extract_notebook_summaries(path, runtime):
    """Use final MVP3 table outputs, not older diagnostic text or carried-over plans."""
    if runtime not in {'vllm', 'sglang'}:
        raise ValueError('Choose vllm or sglang')
    path = Path(path)
    raw = path.read_bytes()
    notebook = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    config = _static_configuration(notebook, runtime)
    tables = []
    for index, cell in enumerate(notebook['cells']):
        if cell.get('cell_type') != 'code' or not re.search(r'\bmvp3\s*=', _text(cell.get('source', []))):
            continue
        for output in cell.get('outputs', []):
            markdown = _text(output.get('data', {}).get('text/markdown', ''))
            if '| Configuration/status | SQL | Result match |' in markdown:
                tables.append((index, cell.get('execution_count'), markdown))
    if not tables:
        raise ValueError('No displayed MVP3 measurements table found')
    results = {}
    for index, count, markdown in tables:
        for line in markdown.splitlines():
            fields = [part.strip() for part in line.strip().split('|')[1:-1]]
            if len(fields) != 9 or not re.fullmatch(r'MVP3/' + runtime + r'/c[124]', fields[0]):
                continue
            executed, cases = _ratio(fields[1])
            matches, scored = _ratio(fields[2])
            semantic = re.fullmatch(r'(\d+)/(\d+) reviewed; (\d+) total', fields[3])
            if not semantic:
                raise ValueError('Unknown semantic review table format')
            correct, reviewed, total = map(int, semantic.groups())
            if not 0 <= correct <= reviewed <= total or total != cases or scored > cases:
                raise ValueError('Inconsistent displayed quality/review counts')
            latency = fields[6].split('/')
            if len(latency) != 2:
                raise ValueError('Expected latency p50/p95')
            peak = None
            if not fields[8].startswith(('—', '-')):
                if not fields[8].endswith(' GiB'):
                    raise ValueError('Unknown memory unit')
                peak = _number(fields[8][:-4]) * 1024**3
            concurrency = int(fields[0].rsplit('c', 1)[1])
            result = {
                'format': FORMAT, 'run_id': f'notebook-{digest[:16]}-{runtime}-c{concurrency}',
                'experiment': fields[0], 'runtime': runtime,
                'hardware': config.get('hardware', 'Hardware not recorded in notebook overrides'),
                'status': 'completed', 'evidence': 'notebook_output',
                'configuration': {**config, 'runtime': runtime, 'concurrency': concurrency},
                'quality': {'cases': cases, 'sql_executed': executed, 'reference_scored': scored,
                            'result_equivalent': matches,
                            'semantic': {'correct': correct, 'reviewed': reviewed, 'total': total}},
                'serving': {'ttft_p50_seconds': _number(fields[5]),
                            'e2e_p50_seconds': _number(latency[0]), 'e2e_p95_seconds': _number(latency[1]),
                            'output_tokens_per_second': _number(fields[7]),
                            'mean_prompt_tokens': _number(fields[4]),
                            'requests': None, 'successful': None, 'failed': None},
                'memory': {'peak_bytes': peak, 'scope': 'displayed source notebook; sampler details unavailable'},
                'server': None, 'database_sha256': None, 'evaluation_date': None, 'records': None,
                'source': {'kind': 'executed_notebook_output', 'filename': path.name,
                           'original_sha256': digest, 'cell_index': index, 'execution_count': count,
                           'displayed_row': line.strip()},
                'limitations': 'Displayed values only, rounded by the source notebook. Raw requests, SQL, '
                               'launch provenance, model/database fingerprints and run date are unavailable. '
                               'Not independently paired with other runtime records.',
            }
            results[fields[0]] = result
    if not results:
        raise ValueError('No completed displayed rows for the selected runtime')
    return list(results.values())


def import_notebook(path, runtime, root):
    """Save append-only summary JSON and a sanitized, non-executed source copy."""
    path, root = Path(path), Path(root)
    results = extract_notebook_summaries(path, runtime)
    digest = results[0]['source']['original_sha256']
    archive_relative = Path('tutorial/history/gpu') / f'{runtime}-{digest[:16]}.ipynb'
    archive = root / archive_relative
    notebook = json.loads(path.read_bytes())
    notebook.get('metadata', {}).pop('widgets', None)
    for cell in notebook['cells']:
        cell.setdefault('metadata', {})['trusted'] = False
    sanitized = json.dumps(portable_value(notebook), indent=1, ensure_ascii=False) + '\n'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and archive.read_text() != sanitized:
        raise FileExistsError('Archived source copy differs; originals will not be replaced')
    if not archive.exists():
        archive.write_text(sanitized)
    destination = root / 'experiment-tracker/tutorial/imported'
    destination.mkdir(parents=True, exist_ok=True)
    for result in results:
        result['source'].update(archive=str(archive_relative),
            archived_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        payload = json.dumps(portable_value(result), indent=2, ensure_ascii=False) + '\n'
        output = destination / (result['run_id'] + '.json')
        if output.exists() and output.read_text() != payload:
            raise FileExistsError('Imported result differs; existing evidence will not be replaced')
        if not output.exists():
            output.write_text(payload)
    return results
