"""Real tiny INT PTQ/export regression for the current Ascend serving contract."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from safetensors.torch import load_file, save_file

from llmcompressor import model_free_ptq


def int_example():
    path = Path(__file__).resolve().parents[4] / (
        "examples/quantization_w4a8_int8/glm53_flash_w4a8c8_example.py"
    )
    spec = importlib.util.spec_from_file_location("glm_int_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("algorithm", ["minmax", "mse"])
def test_tiny_real_export_preserves_mhc_and_quantizes_routed_shared_other(
    tmp_path, algorithm
):
    example = int_example()
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    shapes = {
        "model.layers.3.mlp.experts.gate_up_proj": (2, 32, 32),
        "model.layers.3.mlp.experts.down_proj": (2, 32, 16),
        "model.layers.3.mlp.shared_experts.up_proj.weight": (16, 32),
        "model.layers.3.mlp.gate.weight": (2, 32),
        "model.layers.3.self_attn.q_proj.weight": (32, 32),
        "model.layers.3.self_attn.kv_b_proj.weight": (32, 32),
        "model.layers.3.self_attn.indexer.index_kpool_compress_gate": (16, 32),
        "model.layers.3.hc_attn_fn": (24, 128),
        "model.layers.3.hc_ffn_fn": (24, 128),
        "model.layers.4.attn_hc.fn": (24, 128),
        "model.layers.4.ffn_hc.fn": (24, 128),
        "model.layers.5.attn_hc.fn.weight": (24, 128),
        "model.layers.5.hc_ffn_fn.weight": (24, 128),
        "model.visual.blocks.0.attn.qkv.weight": (96, 32),
        "model.visual.downsample.weight": (32, 32),
        "model.layers.3.input_layernorm.weight": (32,),
        "model.layers.3.hc_attn_scale": (3,),
        "model.layers.3.mlp.e_score_correction_bias": (2,),
        "model.embed_tokens.weight": (64, 32),
        "lm_head.weight": (64, 32),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": (16, 32),
        "mtp.layers.0.eh_proj.weight": (32, 64),
    }
    generator = torch.Generator().manual_seed(20261117)
    tensors = {
        key: torch.randn(shape, generator=generator).bfloat16()
        for key, shape in shapes.items()
    }
    save_file(tensors, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in tensors}})
    )
    model_free_ptq(
        source,
        output,
        config=QuantizationConfig(
            config_groups={
                key: QuantizationScheme(**value)
                for key, value in example.make_config_groups(algorithm).items()
            },
            ignore=example.IGNORE,
        ),
        converter=example.BareProjectionAliases(),
        device="cpu",
        max_workers=1,
    )
    saved = load_file(output / "model.safetensors")
    config = json.loads((output / "config.json").read_text())["quantization_config"]
    assert config["format"] == "mixed-precision"
    assert config["config_groups"]["routed"]["format"] == "pack-quantized"
    assert config["config_groups"]["shared_and_other"]["format"] == "int-quantized"
    for group in config["config_groups"].values():
        assert group["input_activations"]["dynamic"] is True
        assert group["input_activations"]["num_bits"] == 8
    # No static cache or activation calibration is fabricated by this example.
    assert config.get("kv_cache_scheme") is None
    assert not any(key.endswith(".input_scale") for key in saved)
    for key in tensors:
        if (
            any(part in key for part in ("hc_", "attn_hc", "ffn_hc"))
            or any(part in key for part in ("downsample", "layernorm", "embed_tokens"))
            or key.endswith("e_score_correction_bias")
        ):
            torch.testing.assert_close(saved[key], tensors[key], rtol=0, atol=0)
            assert key + ".weight_scale" not in saved
    routed = "model.layers.3.mlp.experts.0.gate_proj"
    mtp = "mtp.layers.0.mlp.experts.0.up_proj"
    for name in (routed, mtp):
        shape = tuple(saved[name + ".weight_shape"].tolist())
        codes = unpack_from_int32(saved[name + ".weight_packed"], 4, torch.Size(shape))
        assert codes.dtype == torch.int8
        assert codes.min() >= -8 and codes.max() <= 7
        assert saved[name + ".weight_scale"].shape == (shape[0], 1)
    for name in (
        "model.layers.3.mlp.shared_experts.up_proj",
        "model.layers.3.mlp.gate",
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.kv_b_proj",
        "model.layers.3.self_attn.indexer.index_kpool_compress_gate",
        "model.visual.blocks.0.attn.qkv",
        "lm_head",
        "mtp.layers.0.eh_proj",
    ):
        assert saved[name + ".weight"].dtype == torch.int8
        assert saved[name + ".weight_scale"].is_floating_point()


def test_only_kpool_alias_is_renamed_and_collision_is_rejected():
    converter = int_example().BareProjectionAliases()
    value = torch.empty((2, 4))
    key = "model.layers.3.self_attn.indexer.index_kpool_compress_gate"
    assert set(converter.process({key: value, "other.fn": value})) == {
        key + ".weight",
        "other.fn",
    }
    with pytest.raises(ValueError, match="Alias collision"):
        converter.process({key: value, key + ".weight": value})


def test_plan_only_does_not_require_npu_or_patch_memory_functions():
    example = int_example()
    before = torch.accelerator.memory.get_memory_info
    result = subprocess.run(
        [sys.executable, example.__file__, "--plan-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    config = json.loads(result.stdout)
    assert config["config_groups"]["routed"]["weights"]["num_bits"] == 4
    assert torch.accelerator.memory.get_memory_info is before
