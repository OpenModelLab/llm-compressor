"""Real tiny INT PTQ/export regression for the current Ascend serving contract."""

import importlib.util
import json
import re
import subprocess
import sys
from fnmatch import fnmatchcase
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
@pytest.mark.parametrize("prefix", ["model.language_model.layers", "model.layers"])
def test_tiny_real_export_matches_modelslim_mlp_scope(tmp_path, algorithm, prefix):
    example = int_example()
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    shapes = {
        "model.layers.3.mlp.experts.gate_up_proj": (2, 32, 32),
        "model.layers.3.mlp.experts.down_proj": (2, 32, 16),
        "model.layers.3.mlp.shared_experts.up_proj.weight": (16, 32),
        "model.layers.3.mlp.shared_experts.gate_proj.weight": (16, 32),
        "model.layers.3.mlp.shared_experts.down_proj.weight": (32, 16),
        "model.layers.0.mlp.gate_proj.weight": (16, 32),
        "model.layers.0.mlp.up_proj.weight": (16, 32),
        "model.layers.0.mlp.down_proj.weight": (32, 16),
        "model.layers.3.mlp.gate.weight": (2, 32),
        "model.layers.3.self_attn.q_proj.weight": (32, 32),
        "model.layers.3.self_attn.kv_b_proj.weight": (32, 32),
        "model.layers.3.self_attn.indexer.wk.weight": (16, 32),
        "model.layers.3.self_attn.indexer.wq_b.weight": (16, 32),
        "model.layers.3.self_attn.k_conv1d.weight": (32, 1, 4),
        "model.layers.3.self_attn.A_log": (4,),
        "model.layers.3.self_attn.dt_bias": (4,),
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
        "model.layers.45.mlp.experts.0.up_proj.weight": (16, 32),
        "model.layers.45.mlp.shared_experts.down_proj.weight": (32, 16),
        "model.layers.45.eh_proj.weight": (32, 64),
    }
    shapes = {
        key.replace("model.layers", prefix): shape for key, shape in shapes.items()
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
    assert set(config["config_groups"]) == {"routed", "dense", "shared"}
    assert config["config_groups"]["dense"]["format"] == "int-quantized"
    assert config["config_groups"]["shared"]["format"] == "int-quantized"
    for group in config["config_groups"].values():
        assert group["input_activations"]["dynamic"] is True
        assert group["input_activations"]["num_bits"] == 8
    # No static cache or activation calibration is fabricated by this example.
    assert config.get("kv_cache_scheme") is None
    assert not any(key.endswith(".input_scale") for key in saved)
    routed = {
        f"{prefix}.3.mlp.experts.{expert}.{projection}_proj"
        for expert in (0, 1)
        for projection in ("gate", "up", "down")
    } | {
        "mtp.layers.0.mlp.experts.0.up_proj",
        f"{prefix}.45.mlp.experts.0.up_proj",
    }
    w8 = (
        {
            f"{prefix}.{layer}.mlp.{projection}_proj"
            for layer in (0,)
            for projection in ("gate", "up", "down")
        }
        | {
            f"{prefix}.3.mlp.shared_experts.{projection}_proj"
            for projection in ("gate", "up", "down")
        }
        | {f"{prefix}.45.mlp.shared_experts.down_proj"}
    )
    for name in routed:
        shape = tuple(saved[name + ".weight_shape"].tolist())
        codes = unpack_from_int32(saved[name + ".weight_packed"], 4, torch.Size(shape))
        assert codes.dtype == torch.int8
        assert codes.min() >= -8 and codes.max() <= 7
        assert saved[name + ".weight_scale"].shape == (shape[0], 1)
    for name in w8:
        assert saved[name + ".weight"].dtype == torch.int8
        assert saved[name + ".weight_scale"].is_floating_point()
    assert {
        key.removesuffix(".weight_scale")
        for key in saved
        if key.endswith(".weight_scale")
    } == routed | w8
    for key, original in tensors.items():
        if key.removesuffix(".weight") in routed | w8 or (
            original.ndim == 3 and ".mlp.experts." in key
        ):
            continue
        saved_key = (
            key + ".weight" if key.endswith(".index_kpool_compress_gate") else key
        )
        assert saved[saved_key].dtype == original.dtype
        torch.testing.assert_close(saved[saved_key], original, rtol=0, atol=0)


def test_targets_match_modelslim_rules_for_glm_projections():
    example = int_example()
    groups = example.make_config_groups()
    # Same include/exclude rules as the supplied YAML, on adapter module paths.
    modelslim = [
        (8, ["*mlp*"], ["*mlp.gate", "*mlp.experts*", "*shared_experts*", "*visual*"]),
        (4, ["*language_model.layers*.mlp.experts*"], ["*shared_experts*"]),
        (8, ["*language_model.layers*.mlp.shared_experts*"], []),
    ]
    names = [
        f"model.language_model.layers.{layer}.{suffix}"
        for layer in (0, 3, 45)
        for suffix in (
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp.gate",
            "mlp.experts.0.gate_proj",
            "mlp.experts.287.down_proj",
            "mlp.shared_experts.up_proj",
            "self_attn.q_proj",
            "self_attn.kv_b_proj",
            "self_attn.indexer.index_kpool_compress_gate",
            "self_attn.indexer.wk",
            "eh_proj",
            "shared_head.head",
        )
    ] + ["lm_head", "model.visual.blocks.0.attn.qkv", "model.visual.mlp.gate_proj"]
    for name in names:
        expected = [
            bits
            for bits, include, exclude in modelslim
            if any(fnmatchcase(name, pattern) for pattern in include)
            and not any(fnmatchcase(name, pattern) for pattern in exclude)
        ]
        actual = [
            group["weights"]["num_bits"]
            for group in groups.values()
            if any(re.match(pattern[3:], name) for pattern in group["targets"])
            and not any(re.match(pattern[3:], name) for pattern in example.IGNORE)
        ]
        assert actual == expected, name


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
    assert set(config["config_groups"]) == {"routed", "dense", "shared"}
    assert torch.accelerator.memory.get_memory_info is before


@pytest.mark.parametrize("nested", [False, True])
def test_cli_rejects_changing_scope_on_quantized_input(tmp_path, nested):
    config = {"quantization_config": {"quant_method": "compressed-tensors"}}
    if nested:
        config = {"text_config": config}
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable,
            int_example().__file__,
            "--model",
            str(source),
            "--save-dir",
            str(tmp_path / "output"),
            "--device",
            "cpu",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "original floating-point checkpoint" in result.stderr
    assert not (tmp_path / "output").exists()
