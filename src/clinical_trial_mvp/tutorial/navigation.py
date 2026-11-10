"""One learning map shared by the README and all chapter introductions."""
CHAPTERS = (
    ('0. Understand the request', '00_local_serving.ipynb', '00 — How a local AI request works',
     'Follow a question through schema context, model serving and generated SQL, then evaluate the results to establish a quality baseline.'),
    ('1. Qwen optimization', '01_qwen_optimization.ipynb', '01 — Qwen: caching and quantization',
     'Compare FP16/AWQ W4A16 and prefix caching, with cold and warm requests reported separately.'),
    ('2. Nemotron optimization', '02_nemotron_optimization.ipynb', '02 — Nemotron optimization',
     'Test schema context, supported prefix caching, concurrent requests, scheduler capacity and output budgets.'),
    ('3. Qwen NIM (optional)', '03_qwen_nim.ipynb', '03 — Qwen through NIM',
     'Deploy Qwen in a container, check the plan format, and compare schema context and concurrent requests.'),
    ('4. Nemotron NIM (optional)', '04_nemotron_nim.ipynb', '04 — Nemotron through NIM',
     'Deploy the hybrid model through NIM and compare schema context and request concurrency.'),
    ('5. CPU portability (optional)', '05_edge_portability_executorch.ipynb', '05 — Qwen with ExecuTorch',
     'Follow Qwen3-4B conversion and CPU export, then inspect a generated SQL answer and its timing.'),
    ('6. Compatibility study (optional)', '06_nemotron_executorch.ipynb', '06 — Nemotron with ExecuTorch',
     'Check whether the installed exporter accepts Nemotron and identify what a working port still needs.'),
)


def workflow_table(current=None, *, prefix='notebooks/'):
    if current is not None and current not in {row[1] for row in CHAPTERS}:
        raise ValueError('Unknown workflow chapter: ' + current)
    rows = ['| **Step** | **Notebook** | **Purpose** |', '|---|---|---|']
    for step, filename, title, purpose in CHAPTERS:
        if filename == current:
            step = '**→ ' + step + ' (current)**'
        rows.append(f'| {step} | [{title}]({prefix}{filename}) | {purpose} |')
    return '\n'.join(rows)


def workflow_intro(filename):
    step, _, title, _ = next(row for row in CHAPTERS if row[1] == filename)
    return (f'**You are here → {title}.**\n\n'
            '**Main workflow:** understand a local AI request → optimize Qwen → optimize Nemotron.\n'
            '**Optional extensions:** model-specific NIM deployment; separate CPU portability branch.\n\n'
            + workflow_table(filename, prefix='') + '\n\n')


def apply_chapter_prose(document, filename):
    """Keep edited teaching prose in source and insert navigation in the first cell.

    Stable builder cell IDs identify prose slots; code and measured outputs are not edited.
    If the structural builder changes, update the prose slots explicitly.
    """
    from clinical_trial_mvp.tutorial.chapter_prose import CHAPTER_PROSE
    prose = CHAPTER_PROSE[filename]
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'markdown']
    if {cell['id'] for cell in cells} != set(prose):
        raise ValueError('Chapter structure changed; synchronize chapter_prose.py: ' + filename)
    for cell in cells:
        cell['source'] = prose[cell['id']].splitlines(keepends=True)
    first = document['cells'][0]
    title, _, rest = ''.join(first['source']).partition('\n')
    first['source'] = (title + '\n\n' + workflow_intro(filename) + rest.lstrip('\n')).splitlines(keepends=True)
    return document
