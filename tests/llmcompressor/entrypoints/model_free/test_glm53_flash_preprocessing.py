"""Numerical and real-lifecycle checks for the example-local GLM53 adapters."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


@pytest.fixture
def example_modules(monkeypatch):
    folder = Path(__file__).resolve().parents[4] / "examples/quantization_w4a8_int8"
    monkeypatch.syspath_prepend(str(folder))
    import glm53_flash_preprocessing as preprocessing
    import glm53_flash_w4a8c8_quarot_flex_ssz_example as example

    return preprocessing, example


def tiny_glm(tmp_path):
    from transformers import Glm5NextForConditionalGeneration

    from llmcompressor.modeling.moe.linearize import linearize_moe

    path = Path(__file__).with_name("test_glm53_flash_gptq_example.py")
    spec = importlib.util.spec_from_file_location("glm_preprocess_fixtures", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    old = helper.tiny_model(tmp_path)
    config = old.config
    config.text_config.q_lora_rank = 32  # native shifted Hadamard has 32-wide blocks
    config.text_config.index_topk = 16
    config.text_config.num_nextn_predict_layers = 0
    model = linearize_moe(Glm5NextForConditionalGeneration(config).float()).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 1 and name.endswith("norm.weight"):
                parameter.uniform_(0.5, 1.5)
    return model, helper


def test_rotated_text_and_image_logits_and_float_reload(tmp_path, example_modules):
    from transformers import Glm5NextForConditionalGeneration

    from llmcompressor import oneshot

    preprocessing, _ = example_modules
    model, helper = tiny_glm(tmp_path)
    text = dict(input_ids=torch.tensor([[4, 5, 6, 7]]), attention_mask=torch.ones(1, 4))
    image = next(iter(helper.tiny_batches()))
    with torch.no_grad():
        expected = [model(**batch).logits.clone() for batch in (text, image)]
    bare_keys = {
        name
        for name, _ in model.named_parameters()
        if name.endswith((".fn", "compress_gate"))
    }
    gate_before = model.model.language_model.layers[
        1
    ].self_attn.indexer.index_kpool_compress_gate.clone()
    modifier = preprocessing.Glm53QuaRotModifier()
    oneshot(
        model=model,
        processor=helper.tiny_tokenizer(),
        recipe=modifier,
        pipeline="datafree",
        tie_word_embeddings=False,
    )
    assert model.config.quarot_config["status"] == "applied"
    assert bare_keys <= set(dict(model.named_parameters()))
    assert not torch.equal(
        gate_before,
        model.model.language_model.layers[
            1
        ].self_attn.indexer.index_kpool_compress_gate,
    )
    with torch.no_grad():
        for batch, reference in zip((text, image), expected):
            torch.testing.assert_close(
                model(**batch).logits, reference, rtol=5e-4, atol=5e-5
            )
    output = tmp_path / "rotated-float"
    model.save_pretrained(output)
    restored = Glm5NextForConditionalGeneration.from_pretrained(output).eval()
    with torch.no_grad():
        for batch, reference in zip((text, image), expected):
            torch.testing.assert_close(
                restored(**batch).logits, reference, rtol=5e-4, atol=5e-5
            )


def test_real_quarot_flex_ssz_lifecycles_and_ct_export(tmp_path, example_modules):
    preprocessing, example = example_modules
    from glm53_flash_w4a8c8_ssz_example import quantize_mlp, validate_export

    model, helper = tiny_glm(tmp_path)
    batch = next(iter(helper.tiny_batches()))
    with torch.no_grad():
        reference = model(**batch).logits.clone()
    matrices, diagnostics, _ = example.preprocess(
        model, helper.tiny_tokenizer(), helper.tiny_batches()
    )
    assert set(matrices) == {"rot", "rot_b_proj", "rot_uv", "rot_kv_b_proj"}
    assert len(diagnostics) == 3  # KDA input, DSA input, DSA q_a norm
    assert all(details["tokens_used"] == 8 for details in diagnostics.values())
    assert all(
        len(details["alpha_losses"]) == len(details["beta_losses"]) == 21
        for details in diagnostics.values()
    )
    assert model.config.flex_smooth_config["status"] == "applied"
    assert not any(module._forward_pre_hooks for module in model.modules())
    with torch.no_grad():
        torch.testing.assert_close(
            model(**batch).logits, reference, rtol=8e-4, atol=8e-5
        )
    scope = quantize_mlp(model, helper.tiny_tokenizer(), step=3)
    output = tmp_path / "ct"
    model.save_pretrained(output, save_compressed=True)
    audit = validate_export(output, scope, repair=True)
    assert audit["projections"] == dict(dense=3, routed=6, shared=3)
    tensors = {}
    for shard in output.glob("model*.safetensors"):
        tensors.update(load_file(shard))
    assert sum(name.endswith("weight_packed") for name in tensors) == 6
    assert sum(value.dtype == torch.int8 for value in tensors.values()) == 6
    assert any(name.endswith("index_kpool_compress_gate") for name in tensors)
    assert not any(
        name.endswith("index_kpool_compress_gate.weight") for name in tensors
    )


def test_mtp_is_rejected_before_any_transform(tmp_path, example_modules, monkeypatch):
    preprocessing, example = example_modules
    model, _ = tiny_glm(tmp_path)
    model.config.text_config.num_nextn_predict_layers = 1
    original = model.model.language_model.embed_tokens.weight.clone()
    with pytest.raises(ValueError, match="MTP explicitly disabled"):
        preprocessing.build_glm53_plans(model)
    torch.testing.assert_close(
        model.model.language_model.embed_tokens.weight, original, rtol=0, atol=0
    )
    source = tmp_path / "source"
    config = json.loads((source / "config.json").read_text())
    config["text_config"]["num_nextn_predict_layers"] = 1
    (source / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["example", "--model", str(source), "--plan-only"])
    with pytest.raises(SystemExit) as failure:
        example.main()
    assert failure.value.code == 2


def test_raw_calibration_preserves_order_and_rejects_messages(
    tmp_path, example_modules
):
    _, example = example_modules
    path = tmp_path / "mix.jsonl"
    path.write_text('"second"\n"first"\n')
    assert example.read_raw_calibration(path) == ["second", "first"]
    path.write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match="JSON strings"):
        example.read_raw_calibration(path)


def test_actual_cli_writes_five_stage_manifest(tmp_path, example_modules, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoProcessor, GenerationConfig, PreTrainedTokenizerFast

    _, example = example_modules
    model, helper = tiny_glm(tmp_path)
    source, output = tmp_path / "cli-source", tmp_path / "cli-output"
    model.save_pretrained(source)
    (source / "generation_config.json").write_text(
        json.dumps(dict(do_sample=False, top_p=0.95, temperature=1.0))
    )
    calibration = tmp_path / "mix.jsonl"
    calibration.write_text('"one two three four"\n')
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "one": 4, "two": 5, "three": 6, "four": 7}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]"
    )
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            str(source),
            "--save-dir",
            str(output),
            "--calibration-file",
            str(calibration),
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
    assert report["algorithm_chain_complete"] is True
    assert report["full_modelslim_recipe"] is False
    assert report["mtp"] == "disabled"
    assert report["calibration"]["samples"] == 1
    assert len(report["flex_smooth"]["results"]) == 3
    assert report["projection_counts"] == dict(dense=3, routed=6, shared=3)
    assert GenerationConfig.from_pretrained(output).do_sample is True
    assert (
        json.loads((source / "generation_config.json").read_text())["do_sample"]
        is False
    )
    assert set(load_file(output / "glm53_rotation.safetensors")) == {
        "rot",
        "rot_b_proj",
        "rot_uv",
        "rot_kv_b_proj",
    }
    assert len(load_file(output / "glm53_flex_scales.safetensors")) == 3


def test_streamed_search_keeps_global_statistics_and_native_grid(example_modules):
    from llmcompressor.modifiers.transform.flex_smooth.search import search_alpha_beta

    preprocessing, _ = example_modules
    generator = torch.Generator().manual_seed(92)
    activations = torch.randn(13, 8, generator=generator)
    activations[-1, -1] *= 50
    weights = torch.randn(12, 8, generator=generator)
    expected = search_alpha_beta(activations, weights)
    actual, amax = preprocessing.streamed_flex_search(
        [activations[:3], activations[3:8], activations[8:]], weights, 4
    )
    torch.testing.assert_close(amax, activations.abs().amax(0), rtol=0, atol=0)
    torch.testing.assert_close(
        torch.tensor(actual.alpha_losses),
        torch.tensor(expected.alpha_losses),
        rtol=2e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.tensor(actual.beta_losses),
        torch.tensor(expected.beta_losses),
        rtol=2e-5,
        atol=1e-7,
    )
    assert (actual.alpha, actual.beta) == (expected.alpha, expected.beta)
