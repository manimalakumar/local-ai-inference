"""CPU checks for the separate MVP4 study; no model downloads or GPU imports."""

import builtins
import importlib.util
import json
import shlex
import sys
import types
from pathlib import Path

import pytest

from clinical_trial_mvp.tutorial import optimization as opt
from clinical_trial_mvp.tutorial.harness import ROOT


@pytest.fixture
def checkpoints(tmp_path):
    config = {"model_type": "qwen", "architectures": ["QwenForCausalLM"],
              "hidden_size": 16, "num_hidden_layers": 2, "vocab_size": 100}
    fp16, awq = tmp_path / "fp16", tmp_path / "awq"
    for path in (fp16, awq):
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
        (path / "tokenizer.json").write_text('{"fixture": true}')
        (path / "tokenizer_config.json").write_text('{"chat_template": "same"}')
        (path / "model.safetensors").write_bytes(path.name.encode())
    config["quantization_config"] = {"quant_method": "awq", "bits": 4, "group_size": 128}
    (awq / "config.json").write_text(json.dumps(config))
    return fp16, awq


def test_matrix_holds_workload_and_fp16_activations_fixed():
    specs = opt.configurations()
    assert tuple(specs) == opt.VARIANTS
    assert {(s["quantization"], s["prefix_cache"]) for s in specs.values()} == {
        (None, False), (None, True), ("awq", False), ("awq", True)}
    for spec in specs.values():
        assert spec["dtype"] == "float16" and spec["concurrency"] == 1
        assert spec["profile"] == "full" and spec["warmups"] == 1
        assert spec["repeats"] == 5
    with pytest.raises(ValueError):
        opt.configurations(run_tag="../escape")


def test_preview_selects_fp16_and_awq_without_executing():
    from clinical_trial_mvp.serving.launch import launch_command
    for spec in opt.configurations().values():
        command = shlex.split(opt.launch_preview(spec, "models/checkpoint with spaces"))
        assert command[command.index("--snapshot") + 1] == "models/checkpoint with spaces"
        assert command[command.index("--dtype") + 1] == "float16"
        assert ("--prefix-cache" in command) == spec["prefix_cache"]
        assert ("--quantization" in command) == bool(spec["quantization"])
        engine = launch_command("vllm", "fixture", dtype=spec["dtype"],
                                prefix_cache=spec["prefix_cache"], quantization=spec["quantization"])
        assert ("--enable-prefix-caching" if spec["prefix_cache"] else "--no-enable-prefix-caching") in engine
        assert engine[engine.index("--dtype") + 1] == "float16"


def test_checkpoint_pair_records_distinct_weights_and_matching_tokenizers(checkpoints):
    pair = opt.checkpoint_pair(*checkpoints, base_revision="base@commit; awq@commit; recipe")
    assert pair["fp16_sha256"] != pair["awq_sha256"]
    assert len(pair["tokenizer_sha256"]) == 64 and pair["awq_config"]["bits"] == 4


@pytest.mark.parametrize("mutation", ["bits", "method", "architecture", "tokenizer", "template", "baseline", "weights"])
def test_invalid_checkpoint_pair_is_rejected(checkpoints, mutation):
    fp16, awq = checkpoints
    config = json.loads((awq / "config.json").read_text())
    if mutation == "bits":
        config["quantization_config"]["bits"] = 8
    elif mutation == "method":
        config["quantization_config"]["quant_method"] = "gptq"
    elif mutation == "architecture":
        config["hidden_size"] = 32
    elif mutation == "tokenizer":
        (awq / "tokenizer.json").write_text("changed")
    elif mutation == "template":
        (awq / "chat_template.jinja").write_text("changed")
    elif mutation == "baseline":
        (fp16 / "config.json").write_text(json.dumps(config))
    elif mutation == "weights":
        (awq / "model.safetensors").unlink()
    (awq / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        opt.checkpoint_pair(fp16, awq, base_revision="base@commit; awq@commit")


def test_default_plan_uses_existing_workload_without_server_access(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Plan tried to access a server")
    monkeypatch.setattr("clinical_trial_mvp.tutorial.harness.serving_adapter", unexpected)
    result = opt.run_variant(opt.configurations()[opt.VARIANTS[0]], root=ROOT)
    assert result["status"] == "plan_only" and result["evidence"] == "none"
    assert result["quality"] is None and result["serving"] is None
    assert result["inputs"]["case_ids"] == ["eval_001", "eval_002", "eval_005", "eval_007"]
    assert result["memory"]["peak_bytes"] is None
    assert result["cache_study"] is None


def test_live_requires_validated_pair_and_matching_server_before_harness(tmp_path, monkeypatch):
    spec = opt.configurations()[opt.VARIANTS[0]]
    with pytest.raises(ValueError, match="checkpoint_pair"):
        opt.run_variant(spec, execute=True, root=tmp_path)
    path = tmp_path / spec["provenance"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"runtime": "vllm", "model_sha256": "a" * 64,
        "engine_version": "fixture", "hardware": "fixture",
        "launch_settings": {k: spec[k] for k in ("context", "dtype", "quantization", "prefix_cache")}}))
    monkeypatch.setattr(opt, "serve_and_benchmark", lambda *a, **kw: pytest.fail("Should fail before workload"))
    with pytest.raises(ValueError, match="fingerprint"):
        opt.run_variant(spec, {"fp16_sha256": "b" * 64}, execute=True, root=tmp_path)


def record(name):
    spec = opt.configurations()[name]
    return {"experiment": spec["experiment"], "configuration": spec,
        "status": "completed", "evidence": "hardware", "inputs": {"prompts_sha256": "fixed"},
        "database_sha256": "db", "evaluation_date": "2026-10-30",
        "checkpoint_pair": {"base_revision_attestation": "base@commit; awq@commit", "tokenizer_sha256": "same",
                            "fp16_sha256": "a" * 64, "awq_sha256": "b" * 64},
        "server": {"runtime": "vllm", "model_sha256": ("b" if spec["quantization"] else "a") * 64,
                   "engine_version": "pinned", "hardware": "GPU", "gpu_uuid": "GPU-fixture",
                   "torch_version": "pinned", "cuda_version": "pinned", "cuda_visible_devices": "0",
                   "launch_settings": {k: spec[k] for k in ("dtype", "context", "prefix_cache", "quantization")}},
        "serving": {"requests": 20, "failed": 0, "ttft_p50_seconds": 2, "e2e_p50_seconds": 4},
        "quality": {"result_equivalent": 4, "reference_scored": 4},
        "cache_study": {"shared_prefix_sha256": "same", "phases": {
            phase: {"failed": 0, "ttft_p50_seconds": 2, "e2e_p50_seconds": 4,
                    "result_equivalent": 20, "reference_scored": 20}
            for phase in ("cold", "warm", "cache_off")}}}


@pytest.mark.parametrize("left,right,factor", [
    (opt.VARIANTS[0], opt.VARIANTS[1], "prefix_cache"),
    (opt.VARIANTS[0], opt.VARIANTS[2], "quantization"),
    (opt.VARIANTS[2], opt.VARIANTS[3], "prefix_cache"),
    (opt.VARIANTS[1], opt.VARIANTS[3], "quantization"),
])
def test_controlled_comparisons_allow_different_weights_only_for_quantization(left, right, factor):
    a, b = record(left), record(right)
    b["cache_study"]["phases"]["warm" if b["configuration"]["prefix_cache"] else "cache_off"]["ttft_p50_seconds"] = 1
    result = opt.compare_pair(a, b, factor=factor)
    assert result["ttft_speedup_left_over_right"] == 2
    assert result["result_matches_left_right"] == [4, 4]


@pytest.mark.parametrize("path,value", [
    (("status",), "failed"), (("evidence",), "test_or_unverified"),
    (("inputs", "prompts_sha256"), "other"), (("evaluation_date",), "tomorrow"),
    (("database_sha256",), "other"), (("configuration", "concurrency"), 4),
    (("server", "engine_version"), "other"), (("server", "gpu_uuid"), None),
    (("server", "model_sha256"), "c" * 64),
    (("server", "launch_settings", "context"), 1024),
    (("checkpoint_pair", "tokenizer_sha256"), "other"), (("serving", "failed"), 1),
])
def test_confounded_or_failed_comparisons_rejected(path, value):
    a, b = record(opt.VARIANTS[0]), record(opt.VARIANTS[1])
    target = b
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        opt.compare_pair(a, b, factor="prefix_cache")


@pytest.mark.parametrize('model,filename', [
    ('nemotron', 'local AI optimization - Nemotron.ipynb'),
    ('qwen', 'local AI optimization - Qwen.ipynb'),
])
def test_notebook_matches_builder_and_every_default_cell_runs_offline(monkeypatch, model, filename):
    path = ROOT / "scripts/tutorial/build_optimization_notebook.py"
    spec = importlib.util.spec_from_file_location("optimization_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    document = json.loads((ROOT / 'notebooks' / filename).read_text())
    assert document == builder.notebook_document(model)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "vllm", "sglang", "transformers"}
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    display_module = types.ModuleType("IPython.display")
    display_module.Markdown = lambda text: text
    display_module.display = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    monkeypatch.chdir(ROOT / "notebooks")
    namespace = {}
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            exec(compile("".join(cell["source"]), "mvp4-cell", "exec"), namespace)
    assert namespace["current"]["status"] == "plan_only"
    assert namespace["pair"] is None
    assert not namespace["RUN_LIVE"] and not namespace["LOAD_SAVED"]
    selected = builder.MODELS[model]
    assert namespace['MODEL_ID'] == selected['model_id']
    assert namespace['FP16_SNAPSHOT'] == ROOT / selected['fp16_snapshot']
    assert namespace['AWQ_SNAPSHOT'] == ROOT / selected['awq_snapshot']
    assert namespace['AWQ_SOURCE'] == selected['awq_source']
    assert namespace['RUN_TAG'] == model + '-run02'
    assert set(namespace['results']) == set(namespace['VARIANTS'])
    assert all(r['status'] == 'plan_only' for r in namespace['results'].values())



def test_notebook_scopes_keep_qwen_strategy_and_make_nemotron_awq_optional():
    nemotron = json.loads((ROOT / 'notebooks/local AI optimization - Nemotron.ipynb').read_text())
    qwen = json.loads((ROOT / 'notebooks/local AI optimization - Qwen.ipynb').read_text())
    ntext = '\n'.join(''.join(c['source']) for c in nemotron['cells'])
    qtext = '\n'.join(''.join(c['source']) for c in qwen['cells'])
    for name in ('fp16_baseline', 'schema_retrieved', 'concurrency_2', 'concurrency_4',
                 'batch_capacity_1', 'reasoning_on', 'output_256', 'output_2048'):
        assert name in ntext
    assert 'RUN_AWQ = False' in ntext
    assert 'RUN_LIVE = False' in ntext and 'RUN_LIVE = False' in qtext
    assert 'AWQ is optional stretch work and never gates core runs' in ntext
    assert 'awq_smoke(smoke_spec' in qtext  # Retain the Qwen compatibility gate.
    assert "('cold', 'warm')" in qtext


def test_qwen_notebook_is_preserved_from_the_recorded_source():
    import hashlib
    path = Path('notebooks/local AI optimization - Qwen.ipynb')
    lineage = json.loads((ROOT / 'SOURCE_LINEAGE.json').read_text())
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == lineage['inherited_files_sha256'][str(path)]
