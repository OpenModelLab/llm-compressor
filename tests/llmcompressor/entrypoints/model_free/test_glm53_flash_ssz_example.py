"""Real three-stage GLM SSZ control and CT export contract."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_ssz_control_keeps_mixed_schemes_and_floating_exceptions(
    tmp_path, monkeypatch
):
    folder = Path(__file__).resolve().parents[4] / "examples/quantization_w4a8_int8"
    monkeypatch.syspath_prepend(str(folder))
    example = _load(folder / "glm53_flash_w4a8c8_ssz_example.py", "glm_ssz_control")
    helpers = _load(
        Path(__file__).with_name("test_glm53_flash_gptq_example.py"), "glm_helpers"
    )
    model = helpers.tiny_model(tmp_path, with_mtp=True)
    # Transformers fuses KDA convs on load and splits them on save. Compare the
    # actual source checkpoint namespace, not that temporary in-memory layout.
    source_tensors = {}
    for path in (tmp_path / "source").glob("*.safetensors"):
        source_tensors.update(load_file(path))
    untouched = {
        name: value
        for name, value in source_tensors.items()
        if name.startswith("model.visual.")
        or ".mlp." not in name
        or ".mlp.gate." in name
    }
    result = example.quantize_mlp(model, helpers.tiny_tokenizer(), step=3)
    assert result["projection_counts"] == dict(routed=6, dense=3, shared=3)
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is not None:
            routed = ".experts." in name
            assert scheme.weights.num_bits == (4 if routed else 8)
            assert scheme.weights.observer == ("ssz" if routed else "memoryless_minmax")
    output = tmp_path / "quantized"
    model.save_pretrained(output, save_compressed=True)
    shards = {path: path.read_bytes() for path in output.glob("*.safetensors")}
    audit = example.audit_config(output, repair=True)
    assert audit["projections"] == dict(dense=3, routed=6, shared=3)
    assert all(path.read_bytes() == content for path, content in shards.items())
    assert example.audit_config(output)["hf_and_vllm_targets"] == "passed"
    tensors = {}
    for path in shards:
        tensors.update(load_file(path))
    for name, original in untouched.items():
        torch.testing.assert_close(tensors[name], original, rtol=0, atol=0)
    assert sum(name.endswith("weight_packed") for name in tensors) == 6
    assert (
        sum(
            t.dtype == torch.int8 and name.endswith(".weight")
            for name, t in tensors.items()
        )
        == 6
    )
    assert tensors[
        "model.language_model.layers.2.mlp.experts.0.up_proj.weight"
    ].is_floating_point()
    # A header audit of only existing tensors cannot detect an entirely missing
    # projection. Check the saved namespace against the independent source scope.
    removed = set()
    for path in shards:
        content = load_file(path)
        keys = [
            key
            for key in content
            if key.startswith("model.language_model.layers.1.mlp.experts.0.gate_proj.")
        ]
        if keys:
            removed.update(keys)
            for key in keys:
                del content[key]
            save_file(content, path, metadata={"format": "pt"})
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for key in removed:
            del index["weight_map"][key]
        index_path.write_text(json.dumps(index))
    assert removed
    with pytest.raises(ValueError, match="Exported MLP scope differs"):
        example.validate_export(output, result)


def test_cli_source_preflight_and_completed_export(tmp_path, monkeypatch, capsys):
    from transformers import AutoProcessor, GenerationConfig

    folder = Path(__file__).resolve().parents[4] / "examples/quantization_w4a8_int8"
    monkeypatch.syspath_prepend(str(folder))
    example = _load(folder / "glm53_flash_w4a8c8_ssz_example.py", "glm_ssz_cli")
    helpers = _load(
        Path(__file__).with_name("test_glm53_flash_gptq_example.py"), "glm_cli_helpers"
    )
    helpers.tiny_model(tmp_path, with_mtp=True)
    source, output = tmp_path / "source", tmp_path / "output"
    original_generation = json.dumps(dict(do_sample=False, top_p=0.95, temperature=1.0))
    (source / "generation_config.json").write_text(original_generation)
    # The tiny checkpoint has no real processor assets. Only that asset boundary
    # is substituted; load_model, all oneshot phases and compressed saving run.
    monkeypatch.setattr(
        AutoProcessor, "from_pretrained", lambda *a, **kw: helpers.tiny_tokenizer()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(folder / "glm53_flash_w4a8c8_ssz_example.py"),
            "--model",
            str(source),
            "--plan-only",
        ],
    )
    capsys.readouterr()
    example.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["assumed_45_layer_plan"] is False
    assert not any(
        re.match(target[3:], "model.language_model.layers.2.mlp.experts.0.up_proj")
        for group in plan["config_groups"].values()
        for target in group["targets"]
    )
    expected = example.expected_mlp_projections(
        json.loads((source / "config.json").read_text())
    )
    assert len(expected) == 12
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(folder / "glm53_flash_w4a8c8_ssz_example.py"),
            "--model",
            str(source),
            "--save-dir",
            str(output),
            "--device",
            "cpu",
            "--step",
            "3",
            "--linearize-after-load",
        ],
    )
    example.main()
    report = json.loads((output / "calibration_report.json").read_text())
    assert report["completion_status"] == "complete"
    assert report["full_reference_recipe"] is False
    assert report["projection_counts"] == dict(dense=3, routed=6, shared=3)
    assert report["source_files_sha256"]["generation_config.json"]
    assert report["environment"]["imported_modules"]["llmcompressor"]
    assert report["load_path"] == "linearize-after-load"
    assert (source / "generation_config.json").read_text() == original_generation
    assert GenerationConfig.from_pretrained(output).do_sample is True
    assert not (output / "calibration_report.json.tmp").exists()
    assert example.audit_config(output)["hf_and_vllm_targets"] == "passed"


def test_missing_expert_projection_fails_before_quantization(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[4] / "examples/quantization_w4a8_int8"
    monkeypatch.syspath_prepend(str(folder))
    example = _load(folder / "glm53_flash_w4a8c8_ssz_example.py", "glm_ssz_scope")
    helpers = _load(
        Path(__file__).with_name("test_glm53_flash_gptq_example.py"),
        "glm_scope_helpers",
    )
    model = helpers.tiny_model(tmp_path)
    del model.model.language_model.layers[1].mlp.experts[1].up_proj
    with pytest.raises(ValueError, match="Incomplete MLP scope"):
        example.quantize_mlp(model, helpers.tiny_tokenizer(), step=3)
    assert all(
        getattr(module, "quantization_scheme", None) is None
        for module in model.modules()
    )


def test_transformed_source_is_rejected_before_loading(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[4] / "examples/quantization_w4a8_int8"
    monkeypatch.syspath_prepend(str(folder))
    example = _load(folder / "glm53_flash_w4a8c8_ssz_example.py", "glm_ssz_transformed")
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(dict(quarot_config=dict(status="complete")))
    )
    monkeypatch.setattr(
        example, "load_model", lambda *a, **kw: pytest.fail("Must not load weights")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            str(source),
            "--save-dir",
            str(tmp_path / "output"),
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()
