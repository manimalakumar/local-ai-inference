"""Pinned Qwen3-4B → official ExecuTorch Qwen converter → XNNPACK CPU.

Stages are explicit and resumable. Weights, PTEs and raw logs stay ignored. This is
an edge portability/short SQL smoke, not the vLLM AWQ W4A16 experiment or full agent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'models/edge-qwen3-4b'
SOURCE = FOLDER / 'source'
MODEL = 'Qwen/Qwen3-4B'
REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
CONTEXT = 4096
THREADS = 4
TOKENS = ([1, 456, 78, 123, 42], [234, 56, 345])
QUESTION = 'How many subjects are Enrolled or Randomized by treatment arm?'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def params(config):
    if config.get('model_type') != 'qwen3' or config.get('num_hidden_layers') != 36 or config.get('hidden_size') != 2560:
        raise ValueError('Expected Qwen3-4B, not a smaller substitute or another architecture')
    if config.get('quantization_config') or config.get('rope_scaling'):
        raise ValueError('Use the pinned unquantized base checkpoint')
    return dict(dim=config['hidden_size'], n_layers=config['num_hidden_layers'],
        n_heads=config['num_attention_heads'], n_kv_heads=config['num_key_value_heads'],
        head_dim=config['head_dim'], vocab_size=config['vocab_size'], hidden_dim=config['intermediate_size'],
        norm_eps=config['rms_norm_eps'], rope_theta=config['rope_theta'], use_hf_rope=True,
        use_qk_norm=True, qk_norm_before_rope=True, qk_norm_affine=True,
        bos_idx=config['bos_token_id'], eos_idx=config['eos_token_id'])


def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, revision=REVISION, local_dir=str(SOURCE), max_workers=2,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])
    write(FOLDER/'source-identity.json', dict(model_id=MODEL,revision=REVISION))


def prepare():
    import torch
    from executorch.examples.models.qwen3.convert_weights import load_checkpoint, qwen_3_tune_to_meta
    identity=json.loads((FOLDER/'source-identity.json').read_text())
    if identity['model_id'] != MODEL or identity['revision'] != REVISION:
        raise ValueError('Source identity differs from the pinned Dell base model')
    config=json.loads((SOURCE/'config.json').read_text())
    model_params=params(config)
    weights=load_checkpoint(str(SOURCE))
    converted=qwen_3_tune_to_meta(weights)
    torch.save(converted,FOLDER/'qwen3-4b.pth')
    write(FOLDER/'params.json',model_params)
    write(FOLDER/'conversion.json',dict(model_id=MODEL,revision=REVISION,source_keys=len(weights),
        converted_keys=len(converted),converter='executorch.examples.models.qwen3.convert_weights.qwen_3_tune_to_meta',
        converter_sha256=sha(Path(sys.modules[qwen_3_tune_to_meta.__module__].__file__)),
        note='Official Qwen3 mapping preserves Q/K normalization and tied output embeddings.'))
    print('Official Qwen3 weight conversion completed',flush=True)


def reference():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(THREADS)
    model=AutoModelForCausalLM.from_pretrained(SOURCE,local_files_only=True,
        dtype=torch.float32,attn_implementation='eager').eval()
    outputs=[]
    with torch.inference_mode():
        for tokens in TOKENS:
            outputs.append(model(torch.tensor([tokens]),use_cache=False).logits[:,-1,:].clone())
    torch.save(outputs,FOLDER/'reference-logits.pth')
    print('Saved two HF FP32 reference logits; process exits before loading converted model',flush=True)


def validate():
    import torch
    from executorch.extension.llm.export.config.llm_config import LlmConfig
    from executorch.examples.models.llama.model import Llama2Model
    torch.set_num_threads(THREADS)
    cfg=LlmConfig()
    cfg.base.checkpoint=str(FOLDER/'qwen3-4b.pth')
    cfg.base.params=str(FOLDER/'params.json')
    cfg.export.max_seq_length=cfg.export.max_context_length=128
    model=Llama2Model(cfg).get_eager_model().float().eval()
    expected=torch.load(FOLDER/'reference-logits.pth',weights_only=True)
    errors=[]
    with torch.inference_mode():
        for tokens,target in zip(TOKENS,expected):
            actual=model(torch.tensor([tokens]))
            torch.testing.assert_close(actual,target,atol=1e-3,rtol=1e-3)
            errors.append((actual-target).abs().max().item())
    write(FOLDER/'validation.json',dict(passed=True,input_sequences=len(TOKENS),dtype='float32',
        max_abs_logit_error=max(errors),atol=1e-3,rtol=1e-3,
        scope='Unquantized mapped model vs original HF; does not validate quantized clinical task quality.'))
    print('Qwen conversion reference checks passed:',max(errors),flush=True)


def export_config():
    if json.loads((FOLDER/'validation.json').read_text()).get('passed') is not True:
        raise ValueError('Reference conversion validation must pass before export')
    config=dict(base=dict(model_class='qwen3_4b',checkpoint=str(FOLDER/'qwen3-4b.pth'),params=str(FOLDER/'params.json'),
                         metadata=json.dumps(dict(get_bos_id=151644,get_eos_ids=[151645]))),
        model=dict(use_kv_cache=True,use_sdpa_with_kv_cache=True,dtype_override='fp32'),
        quantization=dict(qmode='8da4w',embedding_quantize='8,0'),
        export=dict(max_seq_length=CONTEXT,max_context_length=CONTEXT,output_name=str(FOLDER/'qwen3-4b-xnnpack.pte')),
        backend=dict(xnnpack=dict(enabled=True,extended_ops=True)),
        hydra=dict(run=dict(dir=str(ROOT/'.local-serving/qwen3-edge-export'))))
    write(FOLDER/'export.yaml',config)  # JSON is accepted YAML.
    print('Export configuration written:',FOLDER/'export.yaml',flush=True)


def attest_export():
    config=json.loads((FOLDER/'export.yaml').read_text())
    if config['base']['model_class']!='qwen3_4b' or not json.loads((FOLDER/'validation.json').read_text())['passed']:
        raise ValueError('Missing validated Qwen3-4B export contract')
    files=['qwen3-4b.pth','params.json','validation.json','export.yaml','qwen3-4b-xnnpack.pte']
    write(FOLDER/'artifact-lineage.json',dict(model_id=MODEL,model_revision=REVISION,
        context=CONTEXT,qmode=config['quantization']['qmode'],
        files_sha256={name:sha(FOLDER/name) for name in files}))


def _native_generation():
    import torch
    from transformers import AutoTokenizer
    from executorch.extension.llm.custom_ops import custom_ops  # Registers native LLM kernels.
    from executorch.kernels import quantized  # Registers INT8 embedding kernels used by the official recipe.
    from executorch.extension.llm.runner import TextLLMRunner, GenerationConfig
    lineage=json.loads((FOLDER/'artifact-lineage.json').read_text())
    if (lineage['model_id']!=MODEL or lineage['model_revision']!=REVISION or
        lineage['files_sha256']['qwen3-4b-xnnpack.pte']!=sha(FOLDER/'qwen3-4b-xnnpack.pte')):
        raise ValueError('PTE differs from the validated Qwen export; no silent model substitution')
    torch.set_num_threads(THREADS)
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(SOURCE,local_files_only=True)
    messages=[dict(role='user',content="Write only DuckDB SQL, without Markdown or explanation.\n"
        "Table subjects(subject_id VARCHAR PRIMARY KEY, treatment_arm VARCHAR, screening_status VARCHAR).\n"+QUESTION)]
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    runner=TextLLMRunner(str(FOLDER/'qwen3-4b-xnnpack.pte'),str(SOURCE/'tokenizer.json'))
    pieces,reports=[],[]
    cfg=GenerationConfig(temperature=0,seq_len=CONTEXT,max_new_tokens=192,echo=False)
    runner.generate(prompt,cfg,token_callback=pieces.append,
        stats_callback=lambda stats: reports.append(json.loads(stats.to_json_string())))
    if not reports or reports[-1].get('generated_tokens',0)<1 or not ''.join(pieces).strip():
        raise RuntimeError('No generated text or native execution statistics')
    observer=reports[-1]
    generation=dict(question=QUESTION,prompt=prompt,text=''.join(pieces),observer=observer,
        num_prompt_tokens=observer.get('prompt_tokens'),num_generated_tokens=observer.get('generated_tokens'))
    write(FOLDER/'native-output.json',generation)
    print('Native ExecuTorch generated:',repr(''.join(pieces)),flush=True)
    def duration(start,end):
        if start not in observer or end not in observer:return None
        delta=(observer[end]-observer[start])/1000
        if delta<0:raise ValueError('Invalid native observer timing order')
        return delta
    pte=FOLDER/'qwen3-4b-xnnpack.pte'
    return dict(generation=generation,native_observer=observer,artifact_lineage=lineage,
        startup_seconds=duration('model_load_start_ms','model_load_end_ms'),
        single_request_seconds=duration('inference_start_ms','inference_end_ms'),
        footprint_bytes=pte.stat().st_size+(SOURCE/'tokenizer.json').stat().st_size,
        model_sha256=sha(pte),tokenizer_sha256=sha(SOURCE/'tokenizer.json'),
        checkpoint_validation=json.loads((FOLDER/'validation.json').read_text()),
        conversion=json.loads((FOLDER/'conversion.json').read_text()),quality=None,serving=None,
        operator_fallback_count=None,
        limitation='One short-schema direct SQL portability smoke; not the full plan/repair workload, not a latency benchmark; file checks warm the filesystem cache. Native tokens retained as reported; operator fallback unreviewed.')


def run_native():
    import psutil
    import threading
    process=psutil.Process()
    stopped=threading.Event()
    values,errors=[],[]
    def sample():
        try:values.append(process.memory_info().rss)
        except psutil.Error as exc:errors.append(type(exc).__name__)
    def monitor():
        while not stopped.wait(.1):sample()
    sample()
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    try:
        result=_native_generation()
    finally:
        stopped.set()
        thread.join(timeout=2)
        sample()
    result['memory']=dict(peak_bytes=max(values) if values and not errors else None,
        scope='whole runner-process RSS, including Python/runtime imports; not GPU memory',
        sample_count=len(values),sampling_interval_seconds=.1,error=errors[0] if errors else None)
    return result


def execute(stage):
    if stage=='export':
        export_config()
        subprocess.run([sys.executable,'-m','executorch.extension.llm.export.export_llm',
            '--config',str(FOLDER/'export.yaml')],cwd=ROOT,check=True,
            env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':str(THREADS)})
        attest_export()
    else:
        return dict(download=download,prepare=prepare,reference=reference,validate=validate,run=run_native)[stage]()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['download','prepare','reference','validate','export','run'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--folder', type=Path, default=FOLDER)
    args=parser.parse_args()
    FOLDER=args.folder.resolve(); SOURCE=FOLDER/'source'
    if args.output and args.output.exists(): parser.error('Choose a new output path')
    FOLDER.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    outcome=dict(model_id=MODEL,model_name=MODEL,model_revision=REVISION,stage=args.stage,
                 hardware=f'{platform.system()} {platform.machine()} / CPU XNNPACK',dtype='INT4 linear weights / dynamic INT8 activations',
                 measured_at=datetime.now(timezone.utc).isoformat(),runtime='executorch',
                 status='failed',evidence='none',backend='XNNPACK / CPU',
                 quantization='8da4w: dynamic INT8 activations / INT4 linear weights; INT8 embeddings',
                 context=CONTEXT,threads=THREADS,
                 versions={n:importlib.metadata.version(n) for n in ('executorch','torch','torchao','transformers')})
    try:
        extra=execute(args.stage)
        if extra:outcome.update(extra)
        outcome.update(status='completed',evidence='hardware' if args.stage=='run' else 'local_artifact')
    except Exception as exc:
        outcome['reason']=(f'{type(exc).__name__}: {exc}').replace(str(ROOT),'.').replace(str(Path.home()),'~')
        print(outcome['reason'],flush=True)
        raise
    finally:
        outcome['wall_seconds']=time.perf_counter()-started
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output=args.output or ROOT/f'experiment-tracker/tutorial/edge/qwen3-4b-{args.stage}-{stamp}.json'
        write(output,outcome)
        print('Saved:',output.relative_to(ROOT),flush=True)
