"""Select model-specific MVP3 views while retaining the original MVP1/MVP2 evidence."""

from copy import deepcopy
import json
from pathlib import Path

from clinical_trial_mvp.tutorial.harness import TutorialSession, load_configurations


def model_profile(root, family):
    profiles = json.loads((Path(root) / 'tutorial/model_experiments.json').read_text())
    if family not in profiles:
        raise ValueError('Choose the qwen or nemotron MVP3 notebook')
    return profiles[family]


def model_configurations(root, family):
    profile = model_profile(root, family)
    configs = load_configurations(root)
    for runtime in ('vllm', 'sglang'):
        configs['rtx_' + runtime].update(
            hardware='Dell Precision 7740 / Quadro RTX 5000 16 GB', dtype='float16',
            provenance=profile['provenance'][runtime], model_family=family,
            snapshot=profile['snapshot'])
    return configs


def select_model_results(results, family, profile):
    """Never reuse another model's results or modify source evidence in place."""
    selected = {}
    legacy_names = {f'MVP3/{runtime}/c{n}' for runtime in ('vllm', 'sglang') for n in (1, 2, 4)}
    for original in results:
        name = original['experiment']
        result = deepcopy(original)
        if name.startswith(('MVP1/', 'MVP2/')):
            pass
        elif name.startswith(f'MVP3/{family}/'):
            pass
        elif (family == 'qwen' and name in legacy_names and original.get('evidence') == 'notebook_output'
              and original.get('source', {}).get('original_sha256') in profile['legacy_notebook_sha256']):
            result['source_experiment'] = name
            result['experiment'] = name.replace('MVP3/', 'MVP3/qwen/', 1)
            result['model_family'] = 'qwen'
            result['configuration'] = {**result.get('configuration', {}),
                'experiment': result['experiment'], 'model_family': 'qwen'}
        else:
            continue
        existing = selected.get(result['experiment'])
        if existing and existing.get('evidence') == 'hardware' and result.get('evidence') != 'hardware':
            continue
        selected[result['experiment']] = result
    return list(selected.values())


def model_session(root, family):
    session = TutorialSession(root)
    session.load_saved()
    session.results = select_model_results(session.results, family, model_profile(root, family))
    return session
