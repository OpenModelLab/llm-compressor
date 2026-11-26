"""Calibration graph, MTP input construction and independent checkpoint copies."""

import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.fixture
def modules(monkeypatch):
    folder = Path(__file__).resolve().parents[4] / "examples/quantization_w4a8_int8"
    monkeypatch.syspath_prepend(str(folder))
    import glm53_flash_calibration as calibration

    path = Path(__file__).with_name("test_glm53_flash_preprocessing.py")
    spec = importlib.util.spec_from_file_location("glm53_calibration_fixture", path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    return calibration, fixtures


def test_shift_preserves_source_zero_and_no_zero_semantics(modules):
    calibration, _ = modules
    tokens = torch.tensor([[0, 4, 5, 0], [4, 0, 5, 6], [4, 5, 6, 7]])
    expected = torch.tensor([[4, 5, 0, 0], [4, 5, 6, 0], [5, 6, 7, 0]])
    torch.testing.assert_close(
        calibration.shift_input_ids(tokens), expected, rtol=0, atol=0
    )


def test_image_features_reach_backbone_and_mtp(tmp_path, modules):
    calibration, fixtures = modules
    model, helper = fixtures.tiny_glm(tmp_path, with_mtp=True)
    batch = next(iter(helper.tiny_batches()))
    captured = {"backbone": [], "mtp": []}

    def capture(name):
        def hook(_module, args):
            captured[name].append(args[0].detach().clone())

        return hook

    layers = model.model.language_model.layers
    handles = [
        layers[0].register_forward_pre_hook(capture("backbone")),
        layers[2].register_forward_pre_hook(capture("mtp")),
    ]
    try:
        with torch.no_grad():
            embedded = model.model.language_model.embed_tokens(batch["input_ids"])
            visual = model.model.get_image_features(
                batch["pixel_values"], batch["image_grid_thw"], return_dict=True
            )
            features = torch.cat(visual.pooler_output).to(embedded)
            mask, _ = model.model.get_placeholder_mask(
                batch["input_ids"], embedded, image_features=features
            )
            expected = embedded.masked_scatter(mask, features)
            with calibration.calibration_graph(model):
                model(**batch)
                model(
                    **{**batch, "pixel_values": torch.zeros_like(batch["pixel_values"])}
                )
        torch.testing.assert_close(
            captured["backbone"][0],
            expected.unsqueeze(2).expand(-1, -1, model.config.text_config.hc_mult, -1),
            rtol=0,
            atol=0,
        )
        for name in captured:
            assert len(captured[name]) == 2
            assert not torch.equal(captured[name][0], captured[name][1]), name
    finally:
        for handle in handles:
            handle.remove()


def test_calibration_attention_uses_token_zero_and_restores_on_failure(
    tmp_path, modules
):
    calibration, fixtures = modules
    model, _ = fixtures.tiny_glm(tmp_path)
    attention = model.model.language_model.layers[1].self_attn
    original = attention.forward.__func__
    hidden = torch.randn(1, 4, 32)
    # In the source placeholder graph every query selects only token zero.
    with calibration.calibration_graph(model), torch.no_grad():
        actual = attention(hidden, attention_mask=torch.ones(1, 4))[0]
        latent = attention.kv_a_proj_with_mqa(hidden[:, :1])
        value = attention.kv_b_proj(attention.kv_a_layernorm(latent))
        value = value.reshape(1, 1, 2, 32)[..., 16:].reshape(1, 1, -1)
        expected = attention.o_proj(value.expand(1, 4, -1))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
    assert attention.forward.__func__ is original
    assert "forward" not in attention.__dict__
    with pytest.raises(RuntimeError, match="fixture"):
        with calibration.calibration_graph(model):
            raise RuntimeError("fixture")
    assert attention.forward.__func__ is original
    assert "forward" not in model.__dict__


@pytest.mark.parametrize("linearize_after_load", [False, True])
def test_mtp_load_uses_independent_backbone_copies_and_rejects_missing_weights(
    tmp_path, modules, linearize_after_load
):
    if not linearize_after_load and importlib.util.find_spec("torchvision") is None:
        pytest.skip(
            "Native Transformers patch loader requires torchvision "
            "in this CPU environment"
        )
    calibration, fixtures = modules
    model, _ = fixtures.tiny_glm(tmp_path, with_mtp=True)
    source = tmp_path / "mtp-source"
    model.save_pretrained(source)
    shard = source / "model.safetensors"
    tensors = load_file(shard)
    prefix = "model.language_model.layers.2"
    for suffix in ("embed_tokens.weight", "shared_head.head.weight"):
        tensors.pop(f"{prefix}.{suffix}")
    save_file(tensors, shard, metadata={"format": "pt"})
    restored = calibration.load_recipe_model(
        source,
        torch.float32,
        device_map={"": "cpu"},
        linearize_after_load=linearize_after_load,
    )
    layer = restored.model.language_model.layers[2]
    for duplicate, original in (
        (layer.embed_tokens, restored.model.language_model.embed_tokens),
        (layer.shared_head.head, restored.lm_head),
    ):
        torch.testing.assert_close(duplicate.weight, original.weight, rtol=0, atol=0)
        assert duplicate.weight.data_ptr() != original.weight.data_ptr()
    torch.testing.assert_close(
        layer.eh_proj.weight, tensors[f"{prefix}.eh_proj.weight"], rtol=0, atol=0
    )
    # MTP-specific missing parameters must never be silently randomized.
    tensors.pop(f"{prefix}.eh_proj.weight")
    save_file(tensors, shard, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Incomplete recipe checkpoint"):
        calibration.load_recipe_model(
            source,
            torch.float32,
            device_map={"": "cpu"},
            linearize_after_load=linearize_after_load,
        )


def test_mtp_rotated_float_roundtrip_and_input_construction(tmp_path, modules):
    calibration, fixtures = modules
    from glm53_flash_preprocessing import Glm53QuaRotModifier

    from llmcompressor import oneshot

    model, helper = fixtures.tiny_glm(tmp_path, with_mtp=True)
    # Use the same head/embed aliases as a production source checkpoint.
    layer = model.model.language_model.layers[2]
    layer.embed_tokens.weight.data.copy_(model.model.language_model.embed_tokens.weight)
    layer.shared_head.head.weight.data.copy_(model.lm_head.weight)
    tokens = torch.tensor([[4, 5, 6, 7]])
    batch = dict(input_ids=tokens, attention_mask=torch.ones_like(tokens))

    def head_output():
        with calibration.calibration_graph(model), torch.no_grad():
            hidden = model(**batch)
            return layer.shared_head.head(layer.shared_head.norm(hidden))

    expected = head_output()
    gain = model.model.language_model.norm.weight.detach().clone()
    residual = torch.randn(3, 32)
    oneshot(
        model=model,
        processor=helper.tiny_tokenizer(),
        recipe=Glm53QuaRotModifier(strict_recipe=True),
        pipeline="datafree",
        tie_word_embeddings=False,
    )
    torch.testing.assert_close(head_output(), expected, rtol=5e-4, atol=5e-5)
    from llmcompressor.modifiers.transform.quarot.rotation import make_hadamard_rotation

    basis = make_hadamard_rotation(32, seed=1234).to_dense()
    torch.testing.assert_close(
        model.rot(residual @ basis), (residual * gain) @ basis, rtol=1e-5, atol=1e-6
    )
    assert torch.equal(layer.enorm.weight, torch.ones_like(layer.enorm.weight))
    assert torch.equal(layer.hnorm.weight, torch.ones_like(layer.hnorm.weight))
    assert "mtp" not in model._modules
    output = tmp_path / "rotated-mtp"
    model.save_pretrained(output)
    # Loading transformed float weights must preserve the separately folded MTP
    # head/embed, instead of copying the backbone again as for a raw source.
    restored = calibration.Glm53RecipeModel.from_pretrained(output).eval()
    with calibration.calibration_graph(restored), torch.no_grad():
        hidden = restored(**batch)
        mtp = restored.model.language_model.layers[2]
        actual = mtp.shared_head.head(mtp.shared_head.norm(hidden))
    torch.testing.assert_close(actual, head_output(), rtol=5e-4, atol=5e-5)
