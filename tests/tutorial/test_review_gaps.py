"""Regression checks for the iterative review, without a clean/live run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from clinical_trial_mvp.tutorial.harness import ROOT, semantic_scores
from clinical_trial_mvp.tutorial.notebook_io import write_notebook
from clinical_trial_mvp.tutorial.reviews import validate_reviews


def script(name):
    spec=importlib.util.spec_from_file_location(name, ROOT/f'scripts/tutorial/{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def code(source, count=None, text=None):
    return dict(cell_type='code', source=[source], metadata={}, execution_count=count,
                outputs=[] if text is None else [dict(output_type='stream', name='stdout', text=text)])


def test_rebuild_retains_outputs_only_for_identical_code_and_orders_duplicates(tmp_path):
    old=dict(cells=[code('x=1\n',1,'recorded first'), code('x=1\n',2,'recorded second'), code('y=1\n',3,'old code')])
    path=tmp_path/'chapter.ipynb'
    path.write_text(json.dumps(old))
    new=dict(cells=[dict(cell_type='markdown',source=['New teaching prose'],metadata={}),
                    code('x=1\n'),code('x=1\n'),code('y=2\n')])
    write_notebook(path,new)
    saved=json.loads(path.read_text())['cells']
    assert saved[1]['outputs'][0]['text']=='recorded first'
    assert saved[2]['outputs'][0]['text']=='recorded second'
    assert saved[3]['outputs']==[] and saved[3]['execution_count'] is None
    assert new['cells'][1]['outputs']==[]  # Does not mutate the builder source.


@pytest.fixture
def review_root(tmp_path):
    packet=json.loads((ROOT/'tutorial/sql_review_distinct.json').read_text())
    target=tmp_path/'tutorial'
    target.mkdir()
    (target/'sql_review_distinct.json').write_text(json.dumps(packet))
    (target/'semantic_reviews.json').write_text(json.dumps({'reviews': []}))
    return tmp_path


def entry(root):
    first=json.loads((root/'tutorial/sql_review_distinct.json').read_text())['outputs'][0]
    return dict(case_id=first['case_id'],sql_sha256=first['sql_sha256'],reviewer='Fixture reviewer',
                reviewed_at='2026-11-09',correct=False,rationale='Test fixture: an attributed verdict is independent of row match.')


def test_supplied_review_is_recorded_without_altering_immutable_measurements(review_root):
    module=script('review_sql')
    item=entry(review_root)
    result=module.record(item,review_root)
    assert result['reviewed_outputs']==1 and result['correct_outputs']==0
    assert result['unreviewed_outputs']==60
    before=(review_root/'tutorial/semantic_reviews.json').read_bytes()
    with pytest.raises(ValueError,match='Duplicate'):
        module.record(item,review_root)
    assert (review_root/'tutorial/semantic_reviews.json').read_bytes()==before


@pytest.mark.parametrize('delta', [dict(correct='true'),dict(reviewed_at='yesterday'),dict(reviewer='  '),
    dict(rationale=''),dict(sql_sha256='wrong'),dict(sql_sha256='a'*64)])
def test_invalid_or_unknown_human_review_never_partially_updates_file(review_root,delta):
    module=script('review_sql')
    before=(review_root/'tutorial/semantic_reviews.json').read_bytes()
    with pytest.raises(ValueError):
        module.record({**entry(review_root),**delta},review_root)
    assert (review_root/'tutorial/semantic_reviews.json').read_bytes()==before


def test_packet_regeneration_merges_supplied_review_and_keeps_source_bytes(review_root):
    module=script('build_sql_review_packet')
    for name in ('notebooks','experiment-tracker','evals'):
        (review_root/name).symlink_to(ROOT/name,target_is_directory=True)
    (review_root/'tutorial/recorded_runs.json').write_bytes((ROOT/'tutorial/recorded_runs.json').read_bytes())
    supplied=entry(review_root)
    script('review_sql').record(supplied,review_root)
    packet=module.build(review_root)
    reviewed=next(e for e in packet['outputs'] if (e['case_id'],e['sql_sha256'])==(supplied['case_id'],supplied['sql_sha256']))
    assert reviewed['correct'] is False and reviewed['rationale']==supplied['rationale']
    assert packet['reviewed_outputs']==1
    assert '**Human judgment:** False' in module.render(packet)
    record=dict(case_id=supplied['case_id'],attempts=[dict(sql=reviewed['sql'])],scores={'result_equivalence': True})
    assert semantic_scores([record],[supplied])==dict(correct=0,reviewed=1,total=1)


@pytest.mark.parametrize('path',['tutorial/config/vllm-qwen.local.json','experiment-tracker/mvp4/workflow/server.log',
    'copy.zip','.huggingface/token','__MACOSX/x','models/model.safetensors','.git/config'])
def test_clean_export_rejects_tracked_local_or_unreviewed_artifacts(tmp_path,path):
    module=script('export_tracked')
    repo=tmp_path/'repo'
    subprocess.run(['git','init','-q',str(repo)],check=True)
    file=repo/path
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text('fixture')
    if path=='.git/config':
        # A genuine .git/config cannot be tracked: verify policy directly.
        assert module.forbidden(path)
        return
    subprocess.run(['git','add','--',path],cwd=repo,check=True)
    subprocess.run(['git','-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture'],cwd=repo,check=True)
    destination=tmp_path/'export.zip'
    with pytest.raises(ValueError,match='excluded artifacts'):
        module.export(repo,destination)
    assert not destination.exists()


def test_static_notebook_check_starts_no_kernel_or_process(monkeypatch):
    module=script('validate_notebooks')
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k: pytest.fail('static check launched a process'))
    assert module.main(['--check-only'])==0


def test_response_inspection_shows_supplied_review_without_rewriting_recorded_score(tmp_path):
    import hashlib
    from types import SimpleNamespace
    from clinical_trial_mvp.tutorial.teaching import response_example
    sql='SELECT 1'
    case=dict(id='fixture',question='fixture question',reference_sql='SELECT 1')
    (tmp_path/'evals').mkdir()
    (tmp_path/'evals/evaluations.json').write_text(json.dumps([case]))
    (tmp_path/'tutorial').mkdir()
    review=dict(case_id='fixture',sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
                reviewer='Fixture reviewer',reviewed_at='2026-11-09',correct=False,
                rationale='Fixture: identical rows need not satisfy the intended definition.')
    (tmp_path/'tutorial/semantic_reviews.json').write_text(json.dumps({'reviews':[review]}))
    result=dict(records={'quality':[dict(case_id='fixture',attempts=[dict(sql=sql)],scores={'result_equivalence':True})]})
    study=SimpleNamespace(root=tmp_path,mode='live',results={'baseline':result})
    response=response_example(study,'baseline','fixture')
    assert response['human_review']==review
    assert response['semantic_review'].startswith('human review: incorrect')
    assert response['recorded_comparison']['result_equivalence'] is True
