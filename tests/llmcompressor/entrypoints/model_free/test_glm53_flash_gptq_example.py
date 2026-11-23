"""Local GLM calibration, explicit smoothing topology and CT export checks."""

import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file


def example_module():
    path = Path(__file__).resolve().parents[4] / (
        "examples/quantization_w4a8_int8/glm53_flash_w4a8c8_gptq_example.py"
    )
    spec = importlib.util.spec_from_file_location("glm_gptq_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_keeps_mlp_scope_and_excludes_unloaded_mtp():
    groups = example_module().make_config_groups(2)
    for group in groups.values():
        assert group["weights"]["strategy"] == "channel"
        assert group["weights"]["actorder"] is None
        assert group["input_activations"]["strategy"] == "token"
        assert group["input_activations"]["dynamic"] is True
    patterns = [p[3:] for group in groups.values() for p in group["targets"]]
    for suffix in ("self_attn.q_proj", "mlp.gate", "hc_attn_fn"):
        name = f"model.language_model.layers.1.{suffix}"
        assert not any(re.fullmatch(p, name) for p in patterns)
    for name in (
        "model.language_model.layers.2.mlp.experts.0.up_proj",
        "mtp.layers.0.mlp.experts.0.up_proj",
    ):
        assert not any(re.fullmatch(p, name) for p in patterns)


def test_calibration_loader_keeps_processor_tensor_layout(tmp_path):
    example = example_module()
    (tmp_path / "image.png").write_bytes(b"test-image-placeholder")
    path = tmp_path / "calibration.jsonl"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "Describe it"},
            ],
        }
    ]
    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    records = example.load_records(path, 1, 42)
    assert Path(records[0][0][0]["content"][0]["image"]).is_absolute()
    tensors = dict(
        input_ids=torch.ones(1, 8, dtype=torch.long),
        attention_mask=torch.ones(1, 8),
        pixel_values=torch.ones(4, 48),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )

    class Processor:
        def apply_chat_template(self, _messages, **kwargs):
            assert kwargs["truncation"] is False
            assert kwargs["return_tensors"] == "pt"
            return tensors

    batch = next(iter(example.make_dataloader(records, Processor(), 8)))
    for key, value in tensors.items():
        assert batch[key] is value
    with pytest.raises(ValueError, match="exceeds"):
        next(iter(example.make_dataloader(records, Processor(), 7)))


def tiny_model(tmp_path, *, with_mtp=False):
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration

    torch.manual_seed(43)
    config = Glm5NextConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            n_routed_experts=2,
            num_experts_per_tok=1,
            n_shared_experts=1,
            n_group=1,
            topk_group=1,
            q_lora_rank=16,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=0,
            v_head_dim=16,
            index_n_heads=2,
            index_head_dim=8,
            index_topk=4,
            index_kpool=2,
            layer_types=["linear_attention", "indexed_attention"],
            linear_head_dim=16,
            linear_num_heads=2,
            linear_conv_kernel_dim=2,
            mlp_layer_types=["dense", "sparse"],
            indexer_types=["full"] * 2,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            pad_token_id=0,
            use_cache=False,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            num_heads=2,
            intermediate_size=64,
            out_hidden_size=32,
            projection_intermediate_size=64,
            patch_size=4,
            spatial_merge_size=1,
            temporal_patch_size=1,
        ),
        image_token_id=60,
        image_start_token_id=61,
        image_end_token_id=62,
        video_token_id=63,
    )
    source = tmp_path / "source"
    Glm5NextForConditionalGeneration(config).save_pretrained(source)
    if with_mtp:
        config_path = source / "config.json"
        source_config = json.loads(config_path.read_text())
        source_config["text_config"]["num_nextn_predict_layers"] = 1
        config_path.write_text(json.dumps(source_config))
        shard = source / "model.safetensors"
        tensors = load_file(shard)
        tensors["model.language_model.layers.2.mlp.experts.0.up_proj.weight"] = (
            torch.full((16, 32), 0.25)
        )
        save_file(tensors, shard, metadata={"format": "pt"})
    return example_module().load_model(
        source,
        torch.float32,
        linearize_after_load=True,
        device_map={"": "cpu"},
    )


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[UNK]",
    )


def tiny_batches():
    generator = torch.Generator().manual_seed(44)
    batch = dict(
        input_ids=torch.tensor([[61, 60, 60, 60, 60, 62, 4, 5]]),
        attention_mask=torch.ones(1, 8, dtype=torch.long),
        pixel_values=torch.randn(4, 48, generator=generator),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        mm_token_type_ids=torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]]),
    )
    return torch.utils.data.DataLoader([batch], batch_size=None)


def test_smoothquant_preserves_float_mlp_and_router(tmp_path):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    example = example_module()
    model = tiny_model(tmp_path).eval()
    mappings = example.make_smoothquant_mappings(model)
    assert len(mappings) == 2
    assert "model.language_model.layers.1.mlp.gate" in mappings[1][0]
    assert len(mappings[1][0]) == 7
    probes = torch.randn(1, 8, 32) * torch.linspace(0.1, 10, 32)
    layers = model.model.language_model.layers
    with torch.no_grad():
        before = [
            layer.mlp(layer.post_attention_layernorm(probes)).clone()
            for layer in layers
        ]
        norm_before = layers[1].post_attention_layernorm.weight.clone()
    oneshot(
        model=model,
        processor=tiny_tokenizer(),
        dataset=tiny_batches(),
        recipe=SmoothQuantModifier(mappings=mappings, smoothing_strength=0.5),
        pipeline="sequential",
        sequential_targets=["Glm5NextTextDecoderLayer"],
        tie_word_embeddings=False,
    )
    with torch.no_grad():
        for layer, expected in zip(layers, before):
            actual = layer.mlp(layer.post_attention_layernorm(probes))
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert not torch.equal(layers[1].post_attention_layernorm.weight, norm_before)
    destination = tmp_path / "smoothed_float"
    model.save_pretrained(destination, save_compressed=False, max_shard_size="4GB")
    from transformers import Glm5NextForConditionalGeneration

    restored = Glm5NextForConditionalGeneration.from_pretrained(
        destination, dtype=torch.float32
    ).eval()
    with torch.no_grad():
        for layer, expected in zip(restored.model.language_model.layers, before):
            actual = layer.mlp(layer.post_attention_layernorm(probes))
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "smoothquant,routed_method", [(False, "gptq"), (True, "gptq"), (True, "minmax")]
)
def test_real_glm_image_calibration_and_packed_export(
    tmp_path, smoothquant, routed_method
):
    example = example_module()
    model = tiny_model(tmp_path)
    groups = example.make_config_groups(2)
    untouched = (
        model.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()
    )
    coverage = example.calibrate(
        model,
        tiny_batches(),
        groups,
        processor=tiny_tokenizer(),
        smoothquant=smoothquant,
        routed_method=routed_method,
    )
    assert len(coverage) == 6
    assert all(count > 0 for count in coverage.values())
    destination = tmp_path / "quantized"
    model.save_pretrained(destination, save_compressed=True)
    saved = {}
    for shard in destination.glob("*.safetensors"):
        saved.update(load_file(shard))
    packed = {
        key: value for key, value in saved.items() if key.endswith(".weight_packed")
    }
    assert len(packed) == 6
    assert all(
        ".mlp.experts." in key and value.dtype == torch.int32
        for key, value in packed.items()
    )
    w8 = {
        key: value
        for key, value in saved.items()
        if key.endswith(".weight") and value.dtype == torch.int8
    }
    assert len(w8) == 6
    assert all(".mlp." in key and ".experts." not in key for key in w8)
    key = "model.language_model.layers.0.self_attn.q_proj.weight"
    torch.testing.assert_close(saved[key], untouched, rtol=0, atol=0)
    config = json.loads((destination / "config.json").read_text())
    quantization = config.get(
        "quantization_config", config.get("text_config", {}).get("quantization_config")
    )
    assert quantization["format"] == "mixed-precision"
    for group in quantization["config_groups"].values():
        assert group["weights"]["strategy"] == "channel"
        assert group["input_activations"]["dynamic"] is True
    assert not any(key.endswith(".input_scale") for key in saved)


def test_unloaded_mtp_is_copied_as_float_and_not_targeted(tmp_path):
    example = example_module()
    model = tiny_model(tmp_path, with_mtp=True)
    assert not hasattr(model, "mtp")
    example.calibrate(
        model,
        tiny_batches(),
        example.make_config_groups(2),
        processor=tiny_tokenizer(),
    )
    destination = tmp_path / "with_mtp"
    model.save_pretrained(destination, save_compressed=True, max_shard_size="4GB")
    key = "model.language_model.layers.2.mlp.experts.0.up_proj.weight"
    found = {}
    for shard in destination.glob("*.safetensors"):
        found.update(load_file(shard))
    torch.testing.assert_close(found[key], torch.full((16, 32), 0.25), rtol=0, atol=0)
    assert key.removesuffix(".weight") + ".weight_scale" not in found
