"""Explicit edge stages: reuse attested export if requested, always generate afresh."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import uuid

from clinical_trial_mvp.tutorial.harness import ROOT


def template(model):
    return dict(mode='live', edge_python='.venv-edge/bin/python', download_models=False,
        reuse_export=True, artifact_dir='models/edge-qwen3-4b',
        nemotron_config='models/edge-nemotron-4b/source/config.json')


def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class EdgeStudy:
    def __init__(self, model, *, root=ROOT, mode=None, config=None):
        if model not in ('qwen','nemotron'):raise ValueError('Choose qwen or nemotron')
        self.root=Path(root);self.model=model
        cfg=template(model);path=self.root/f'tutorial/config/edge-{model}.local.json'
        if config is not None:cfg.update(config)
        elif path.exists():cfg.update(json.loads(path.read_text()))
        self.mode=mode if mode is not None else os.environ.get('LOCAL_AI_LAB_MODE',cfg['mode'])
        if self.mode not in ('live','plan','replay'):raise ValueError('MODE must be live, plan or replay')
        if 'execute' in cfg:raise ValueError('Remove execute; MODE controls execution')
        for key in ('download_models','reuse_export'):
            if type(cfg[key]) is not bool:raise ValueError(key+' must be boolean')
        cfg['mode']=self.mode
        self.config=cfg;self.session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10]
        self.results={};self.python=self.path(cfg['edge_python']);self.folder=self.path(cfg['artifact_dir'])
        self.output_dir=self.root/'experiment-tracker/tutorial/edge/sessions'/self.session_id
        self._reuse_checked=False

    def path(self,value):
        p=Path(value).expanduser();return p if p.is_absolute() else self.root/p

    def preflight(self):
        if self.mode!='live':return dict(status='not_required',mode=self.mode,session_id=self.session_id)
        if not self.python.is_file():raise ValueError('Set edge_python to an installed edge environment; see tutorial/EXECUTORCH_MAC.md')
        try:
            versions=json.loads(subprocess.check_output([str(self.python),'-c',
                "import json,importlib.metadata as m; print(json.dumps({p:m.version(p) for p in ['executorch','torch','transformers','omegaconf']}))"],text=True,timeout=60))
        except (OSError,subprocess.SubprocessError) as exc:raise ValueError('Install requirements-edge.txt in edge_python first; no automatic installation') from exc
        if self.model=='qwen':
            if self.config['reuse_export'] and not (self.folder/'artifact-lineage.json').is_file():
                raise ValueError('reuse_export=true needs artifact-lineage.json; restore the validated export or set reuse_export=false to build it')
            if not (self.folder/'source/config.json').is_file() and not self.config['download_models']:
                raise ValueError('Stage pinned Qwen source weights or opt in to download_models')
        elif not self.path(self.config['nemotron_config']).is_file() and not self.config['download_models']:
            raise ValueError('Stage the pinned Nemotron config.json or opt in to download_models (config only)')
        return dict(status='ready',mode=self.mode,session_id=self.session_id,versions=versions,
                    reuse_export=self.config['reuse_export'],runtime_python=str(self.python))

    def _save(self,name,result):
        result=dict(result,session_id=self.session_id,mode=self.mode,
                    measured_at=datetime.now(timezone.utc).isoformat())
        if self.mode=='live':
            self.output_dir.mkdir(parents=True,exist_ok=True)
            p=self.output_dir/(name+'-'+uuid.uuid4().hex[:8]+'.json')
            result['session_evidence_path']=str(p.relative_to(self.root))
            with p.open('x') as stream:json.dump(result,stream,indent=2,default=str);stream.write('\n')
        self.results[name]=result;return result

    def _reuse(self):
        if self._reuse_checked:return
        lineage=json.loads((self.folder/'artifact-lineage.json').read_text())
        identity=json.loads((self.folder/'source-identity.json').read_text())
        if identity.get('model_id')!='Qwen/Qwen3-4B' or identity.get('revision')!='1cfa9a7208912126459214e8b04321603b3df60c':
            raise ValueError('Export source identity must match the pinned Qwen base revision')
        if lineage.get('model_id')!=identity['model_id'] or lineage.get('model_revision')!=identity['revision']:
            raise ValueError('Export lineage differs from source identity')
        for name,digest in lineage['files_sha256'].items():
            if Path(name).name!=name:raise ValueError('Invalid lineage artifact path')
            if _sha(self.folder/name)!=digest:raise ValueError('Reused export artifact changed: '+name)
        if not json.loads((self.folder/'validation.json').read_text()).get('passed'):
            raise ValueError('Reused conversion validation failed')
        self._reuse_checked=True

    def stage(self,name):
        if self.model!='qwen':raise ValueError('Nemotron supports a configuration probe only')
        if name not in ('download','prepare','reference','validate','export','run','sql_check'):raise ValueError('Unknown edge stage')
        if self.mode=='replay':
            from clinical_trial_mvp.tutorial.edge_notebook import recorded_qwen_edge
            result=recorded_qwen_edge(self.root)
            if name=='sql_check':
                result=dict(result['sql_smoke'],status='completed',label=result['label'],source=result['source'])
            return self._save(name,result)
        if self.mode=='plan':return self._save(name,dict(status='plan_only',stage=name,reason='No edge process started'))
        print('Starting edge stage:',name,'| session',self.session_id,flush=True)
        try:
            if name=='download' and not self.config['download_models']:
                if not (self.folder/'source/config.json').is_file():raise ValueError('Pinned source checkpoint missing; download_models is false')
                return self._save(name,dict(status='reused_artifact',stage=name,reason='Download disabled; using explicit local source'))
            if name in ('prepare','reference','validate','export') and self.config['reuse_export']:
                self._reuse()
                return self._save(name,dict(status='reused_artifact',stage=name,
                    reason='Explicit reuse of hash-verified export and conversion validation; no new export claimed',
                    artifact_lineage=json.loads((self.folder/'artifact-lineage.json').read_text())))
            if name=='sql_check':
                current=self.results.get('run') or {}
                if current.get('session_id')!=self.session_id or current.get('status')!='completed':
                    raise ValueError('Run fresh native generation in this session before SQL checking')
                script=self.root/'scripts/tutorial/check_edge_sql.py'
                spec=importlib.util.spec_from_file_location('edge_sql_check',script)
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                result=module.check(self.root/current['session_evidence_path'],self.folder/'source')
                return self._save(name,dict(result,status='completed'))
            self.output_dir.mkdir(parents=True,exist_ok=True)
            raw=self.output_dir/(name+'-raw-'+uuid.uuid4().hex[:8]+'.json')
            command=[str(self.python),str(self.root/'scripts/tutorial/run_qwen_edge.py'),
                     '--stage',name,'--folder',str(self.folder),'--output',str(raw)]
            completed=subprocess.run(command,cwd=self.root,text=True,capture_output=True,
                env={**os.environ,'HF_HUB_OFFLINE':'0' if name=='download' else '1',
                     'TRANSFORMERS_OFFLINE':'0' if name=='download' else '1'})
            result=json.loads(raw.read_text()) if raw.exists() else dict(status='failed',reason=completed.stderr[-4000:])
            result.update(command=command,raw_evidence_path=str(raw.relative_to(self.root)))
            saved=self._save(name,result)
            if completed.returncode or saved['status']!='completed':raise RuntimeError(saved.get('reason','Edge stage failed; inspect session evidence'))
            return saved
        except BaseException as exc:
            self._save(name,dict(status='failed',stage=name,reason=str(exc)))
            raise

    def probe(self):
        if self.mode=='replay':
            from clinical_trial_mvp.tutorial.nemotron_edge_notebook import recorded_probe
            return self._save('probe',recorded_probe(self.root))
        if self.mode=='plan':return self._save('probe',dict(status='plan_only',native_inference_run=False))
        self.output_dir.mkdir(parents=True,exist_ok=True)
        raw=self.output_dir/('probe-raw-'+uuid.uuid4().hex[:8]+'.json')
        command=[str(self.python),str(self.root/'scripts/tutorial/probe_nemotron_edge.py'),'--output',str(raw)]
        config=self.path(self.config['nemotron_config'])
        if config.is_file():
            if _sha(config)!='fde9241f66cd414458df444a80eb535f53aef2b4b240a91f092e651fa6f27219':
                raise ValueError('Nemotron config differs from the pinned revision')
            command+=['--config',str(config)]
        elif not self.config['download_models']:raise ValueError('Stage Nemotron config.json; downloads are disabled')
        try:
            completed=subprocess.run(command,cwd=self.root,text=True,capture_output=True)
            result=json.loads(raw.read_text()) if raw.exists() else dict(status='failed',reason=completed.stderr[-4000:])
            result.update(command=command,raw_evidence_path=str(raw.relative_to(self.root)))
            saved=self._save('probe',result)
            if completed.returncode:raise RuntimeError(saved.get('reason','Compatibility probe failed'))
            return saved
        except BaseException as exc:
            self._save('probe',dict(status='failed',reason=str(exc),native_inference_run=False));raise

    def report(self):
        lines=[f'**Mode: {self.mode} · Session: {self.session_id}**','',
               '| Stage | Status | Evidence |','|---|---|---|']
        for name,result in self.results.items():
            if result.get('session_id')!=self.session_id:continue
            lines.append(f'| {name} | {result["status"]} | {result.get("session_evidence_path",result.get("source","unavailable"))} |')
        return '\n'.join(lines)

    def close(self):
        return 'Edge stages run in foreground child processes; no serving process is retained.'
