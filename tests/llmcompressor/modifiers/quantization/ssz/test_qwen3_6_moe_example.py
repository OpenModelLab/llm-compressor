"""Exercise SSZ on real, locally constructed Qwen MoE experts."""

import json
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    CompressedTensorsConfig,
    PreTrainedTokenizerFast,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import linearize_moe

pytestmark = pytest.mark.unit

EXAMPLE = (
    Path(__file__).resolve().parents[5]
    / "examples/quantization_w4a8_int8/qwen3_6_moe_ssz_example.py"
)


@lru_cache
def _load_example(path=EXAMPLE):
    spec = spec_from_file_location("ssz_moe_example", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("expert_format", ["pack-quantized", "int-quantized"])
def test_moe_checkpoint_and_reload(tmp_path, monkeypatch, expert_format):
    config = Qwen3_5MoeConfig(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=32,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            num_position_embeddings=16,
            patch_size=2,
        ),
    )
    torch.manual_seed(42)
    model = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
    for module in model.modules():
        if isinstance(module, Qwen3_5MoeRMSNorm):
            with torch.no_grad():
                module.weight.fill_(0.003)
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    norms = {
        name + ".weight"
        for name, module in model.named_modules()
        if isinstance(module, Qwen3_5MoeRMSNorm)
    }
    source = tmp_path / "source"
    output = tmp_path / "quantized"
    model.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)

    example = _load_example()
    monkeypatch.setattr(example, "MODEL_ID", str(source))
    monkeypatch.setattr(example, "SAVE_DIR", str(output))
    monkeypatch.setattr(example, "STEP", 3)
    monkeypatch.setattr(example, "CALIBRATION_DEVICE", "cpu")
    monkeypatch.setattr(example, "CALIBRATION_BATCH_SIZE", 5)
    monkeypatch.setattr(example, "EXPERT_FORMAT", expert_format)
    monkeypatch.setattr(example, "find_spec", lambda name: None)
    monkeypatch.setattr(
        example.AutoProcessor, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    # A machine with an available accelerator must still honor explicit CPU
    # placement. An informational device message alone would not establish this.
    monkeypatch.setattr(
        example,
        "get_main_device",
        lambda: pytest.fail("Explicit CPU calibration must not auto-select a device"),
    )
    monkeypatch.setattr(
        "llmcompressor.pipelines.data_free.pipeline.get_main_device",
        lambda: torch.device("cpu"),
    )
    reference = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
    reference.load_state_dict(original, strict=True)
    reference.config._name_or_path = str(source)
    linearize_moe(reference).to(torch.bfloat16)
    oneshot(
        model=reference,
        processor=tokenizer,
        recipe=example.create_recipe(),
        pipeline="datafree",
    )
    ModelCompressor.from_pretrained_model(reference).compress_model(reference)
    reference_state = reference.state_dict()
    example.main()

    quant_config = json.loads((output / "config.json").read_text())[
        "quantization_config"
    ]
    packed_experts = expert_format == "pack-quantized"
    assert quant_config["format"] == (
        "mixed-precision" if packed_experts else "int-quantized"
    )
    assert quant_config["quantization_status"] == "compressed"
    groups = list(quant_config["config_groups"].values())
    assert {group["weights"]["num_bits"] for group in groups} == {4, 8}
    for group in groups:
        assert group["weights"]["strategy"] == "channel"
        assert group["weights"]["observer"] == "ssz"
        assert group["input_activations"]["num_bits"] == 8
        assert group["input_activations"]["strategy"] == "token"
        assert group["input_activations"]["dynamic"] is True

    dequantized_experts = {}
    with safe_open(output / "model.safetensors", framework="pt") as saved:
        keys = set(saved.keys())
        # Batching native events must preserve all quantized tensors exactly:
        # scales, INT4 packed codes/shape, and W8 integer weights.
        for name in keys:
            prefix = name.rsplit(".", 1)[0]
            if prefix + ".weight_scale" in keys:
                assert torch.equal(saved.get_tensor(name), reference_state[name]), name
        weight_suffix = ".weight_packed" if packed_experts else ".weight"
        expert_weights = sorted(
            key
            for key in keys
            if ".mlp.experts." in key and key.endswith(weight_suffix)
        )
        # All experts are quantized without routing/calibration data, not only top-k.
        assert len(expert_weights) == 2 * 4 * 3
        for name in expert_weights:
            prefix = name.removesuffix(weight_suffix)
            if packed_experts:
                packed = saved.get_tensor(name)
                shape = saved.get_tensor(prefix + ".weight_shape").tolist()
                assert packed.dtype == torch.int32
                codes = unpack_from_int32(packed, 4, shape)
                assert packed.nbytes == codes.numel() // 2
            else:
                codes = saved.get_tensor(name)
            scale = saved.get_tensor(prefix + ".weight_scale")
            assert codes.dtype == torch.int8
            assert codes.min() >= -8 and codes.max() <= 7
            assert scale.dtype == torch.bfloat16
            assert scale.shape == (codes.shape[0], 1)
            assert torch.isfinite(scale).all() and (scale > 0).all()
            dequantized_experts[prefix + ".weight"] = codes.to(scale.dtype) * scale
        for name, value in original.items():
            if (
                name.startswith("model.visual.")
                or name in norms
                or ".mlp.gate." in name
                or ".shared_expert_gate." in name
                or ".linear_attn.in_proj_a." in name
                or ".linear_attn.in_proj_b." in name
                or ".linear_attn.out_proj." in name
                or name.endswith("embed_tokens.weight")
                or name == "lm_head.weight"
            ):
                assert torch.equal(saved.get_tensor(name), value), name
        for suffix in (
            "0.linear_attn.in_proj_qkv",
            "0.linear_attn.in_proj_z",
            "1.self_attn.q_proj",
            "1.self_attn.o_proj",
            "0.mlp.shared_expert.gate_proj",
        ):
            prefix = "model.language_model.layers." + suffix
            assert saved.get_tensor(prefix + ".weight").dtype == torch.int8
            assert prefix + ".weight_scale" in keys
        assert all(
            ".mlp.experts." in key for key in keys if key.endswith("weight_packed")
        )
        assert not any(key.endswith("input_scale") for key in keys)

    # Reload through the native CT layout into a fresh linearized model. HF's
    # default fused-expert converter assumes packed non-FP8 weights, whereas the
    # target Ascend loader consumes per-expert, unpacked INT4 codes.
    reloaded = Qwen3_5MoeForConditionalGeneration(config).to(torch.bfloat16)
    linearize_moe(reloaded).to(torch.bfloat16)
    compressor = ModelCompressor.from_compression_config(
        CompressedTensorsConfig(**quant_config)
    )
    apply_quantization_config(
        reloaded, compressor.quantization_config, run_compressed=False
    )
    compressor.compress_model(reloaded)
    reloaded.load_state_dict(load_file(output / "model.safetensors"), strict=True)
    compressor.decompress_model(reloaded)
    state = reloaded.state_dict()
    for name, expected in dequantized_experts.items():
        assert torch.equal(state[name], expected), name
    for name in norms:
        assert torch.equal(state[name], original[name]), name

    if not packed_experts:
        # Existing large checkpoints can be repacked without running SSZ again.
        # Reverse conversion must recover every tensor exactly, not just logits.
        converter = _load_example(EXAMPLE.with_name("repack_ssz_checkpoint.py"))
        packed_dir = tmp_path / "packed"
        restored_dir = tmp_path / "restored"
        converter.repack_checkpoint(output, packed_dir)
        packed_state = load_file(packed_dir / "model.safetensors")
        expected_state = load_file(output / "model.safetensors")
        for name, tensor in expected_state.items():
            if ".mlp.experts." in name and name.endswith(".weight"):
                prefix = name.removesuffix(".weight")
                codes = unpack_from_int32(
                    packed_state[prefix + ".weight_packed"], 4, tensor.shape
                )
                assert torch.equal(codes, tensor), name
            else:
                assert torch.equal(packed_state[name], tensor), name
        converter.repack_checkpoint(packed_dir, restored_dir, "int-quantized")
        restored = load_file(restored_dir / "model.safetensors")
        assert restored.keys() == expected_state.keys()
        for name, tensor in expected_state.items():
            assert restored[name].dtype == tensor.dtype, name
            assert torch.equal(restored[name], tensor), name
        restored_q = json.loads((restored_dir / "config.json").read_text())[
            "quantization_config"
        ]
        assert restored_q["format"] == "int-quantized"
        assert restored_q["config_groups"] == quant_config["config_groups"]


def test_rejects_overwriting_source(monkeypatch):
    example = _load_example()
    monkeypatch.setattr(example, "SAVE_DIR", example.MODEL_ID)
    with pytest.raises(ValueError, match="must differ"):
        example.main()


def test_rejects_unknown_storage_before_loading(monkeypatch):
    example = _load_example()
    monkeypatch.setattr(example, "EXPERT_FORMAT", "packed")
    with pytest.raises(ValueError, match="EXPERT_FORMAT"):
        example.main()


def test_cpu_pipeline_receives_cpu_weights(monkeypatch):
    example = _load_example()
    monkeypatch.setattr(example, "CALIBRATION_DEVICE", "cpu")
    monkeypatch.setattr(example, "CALIBRATION_BATCH_SIZE", 1)
    model = torch.nn.Linear(19, 7, bias=False).to(torch.bfloat16)
    model.weight_observer = torch.nn.Identity()
    observed_devices = []
    monkeypatch.setattr(example.LifecycleCallbacks, "calibration_start", lambda: None)
    monkeypatch.setattr(example.LifecycleCallbacks, "calibration_end", lambda: None)
    monkeypatch.setattr(
        example.LifecycleCallbacks,
        "sequential_epoch_end",
        lambda modules: observed_devices.extend(m.weight.device for m in modules),
    )
    example.SSZDataFreePipeline()(model, None, None)
    assert observed_devices == [torch.device("cpu")]


def test_explicit_device_and_auto_selection(monkeypatch):
    example = _load_example()
    monkeypatch.setattr(example, "get_main_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(example, "CALIBRATION_DEVICE", "cpu")
    assert example.get_calibration_device() == torch.device("cpu")
    monkeypatch.setattr(example, "CALIBRATION_DEVICE", "auto")
    assert example.get_calibration_device() == torch.device("cuda:0")


def _storage_config(format):
    return QuantizationConfig(
        format=format,
        quantization_status="compressed",
        config_groups={
            "experts": {
                "targets": ["Linear"],
                "format": format,
                "weights": {"num_bits": 4, "strategy": "channel", "observer": "ssz"},
                "input_activations": {
                    "num_bits": 8,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
    )


def test_storage_preserves_signed_codes_and_tail_padding():
    converter = _load_example(EXAMPLE.with_name("repack_ssz_checkpoint.py"))
    codes = (torch.arange(57).reshape(3, 19) % 16 - 8).to(torch.int8)
    tensors = {"expert.weight": codes, "expert.weight_scale": torch.ones(3, 1)}
    pack = converter.SSZIntegerStorageConverter(
        _storage_config("int-quantized"), tensors, "pack-quantized"
    )
    packed = pack.process(tensors)
    assert packed["expert.weight_packed"].is_contiguous()
    unpack = converter.SSZIntegerStorageConverter(
        _storage_config("pack-quantized"), packed, "int-quantized"
    )
    restored = unpack.process(packed)
    assert torch.equal(restored["expert.weight"], codes)
    assert restored["expert.weight_scale"] is tensors["expert.weight_scale"]
    assert torch.equal(tensors["expert.weight"], codes)  # Input was not mutated.
    for invalid in (-9, 8):
        with pytest.raises(ValueError, match="outside"):
            pack.process({**tensors, "expert.weight": torch.full_like(codes, invalid)})


def test_unpack_resolves_shape_in_another_shard(tmp_path):
    converter = _load_example(EXAMPLE.with_name("repack_ssz_checkpoint.py"))
    source, output = tmp_path / "packed", tmp_path / "unpacked"
    source.mkdir()
    config = _storage_config("pack-quantized")
    (source / "config.json").write_text(
        json.dumps({"quantization_config": config.model_dump(mode="json")})
    )
    codes = (torch.arange(57).reshape(3, 19) % 16 - 8).to(torch.int8)
    save_file(
        {"expert.weight_packed": pack_to_int32(codes, 4).contiguous()},
        source / "model-00001-of-00002.safetensors",
    )
    save_file(
        {
            "expert.weight_shape": torch.tensor(codes.shape),
            "expert.weight_scale": torch.ones(3, 1, dtype=torch.bfloat16),
        },
        source / "model-00002-of-00002.safetensors",
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "expert.weight_packed": "model-00001-of-00002.safetensors",
                    "expert.weight_shape": "model-00002-of-00002.safetensors",
                    "expert.weight_scale": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    converter.repack_checkpoint(source, output, "int-quantized")
    saved = {}
    for shard in output.glob("*.safetensors"):
        saved.update(load_file(shard))
    assert torch.equal(saved["expert.weight"], codes)
    assert "expert.weight_shape" not in saved
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(saved)
    assert index["metadata"]["total_size"] == sum(v.nbytes for v in saved.values())
