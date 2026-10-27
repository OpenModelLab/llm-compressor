import json

import pytest
import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs, quantize
from pydantic import ValidationError
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    AutoModelForCausalLM,
    CompressedTensorsConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from llmcompressor import oneshot
from llmcompressor.core import State
from llmcompressor.modifiers import ModifierFactory
from llmcompressor.modifiers.quantization import SSZModifier as ExportedSSZModifier
from llmcompressor.modifiers.quantization.ssz import SSZModifier
from llmcompressor.modifiers.quantization.ssz.observer import SSZObserver
from llmcompressor.observers import Observer
from llmcompressor.recipe import Recipe

pytestmark = pytest.mark.unit


def test_default_scheme_and_public_export():
    assert ExportedSSZModifier is SSZModifier
    modifier = SSZModifier(step=17)
    group = modifier.resolved_config.config_groups["group_0"]
    assert group.weights.num_bits == 8
    assert group.weights.strategy == "channel"
    assert group.weights.observer == "ssz"
    assert group.weights.observer_kwargs == {"step": 17}
    assert group.input_activations.dynamic is True
    assert not modifier.requires_calibration_data


@pytest.mark.parametrize("step", [0, -1, True, 1.5, "10"])
def test_invalid_step(step):
    with pytest.raises(ValidationError):
        SSZModifier(step=step)


@pytest.mark.parametrize("scheme", ["W4A16", "W8A16", "FP8"])
def test_unsupported_preset(scheme):
    with pytest.raises(ValueError, match="ssz"):
        SSZModifier(scheme=scheme)


def test_explicit_groups_and_static_activation_calibration():
    group = {
        "targets": ["Linear"],
        "weights": {"num_bits": 4, "strategy": "channel"},
        "input_activations": {"num_bits": 8, "strategy": "tensor"},
    }
    modifier = SSZModifier(config_groups={"group_0": group}, step=10)
    resolved = modifier.resolved_config.config_groups["group_0"]
    assert modifier.scheme is None
    assert resolved.weights.observer == "ssz"
    assert resolved.weights.observer_kwargs == {"step": 10}
    assert resolved.input_activations.observer == "minmax"
    assert modifier.requires_calibration_data


@pytest.mark.parametrize(
    "kwargs",
    [{"weight_observer": "minmax"}, {"observer": {"weights": "mse"}}],
)
def test_conflicting_observer_override(kwargs):
    with pytest.raises(ValueError, match="ssz weight observer"):
        SSZModifier(**kwargs)


def test_conflicting_iteration_budget():
    with pytest.raises(ValueError, match="SSZModifier\\(step"):
        SSZModifier(
            step=10,
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "strategy": "channel",
                        "observer_kwargs": {"step": 20},
                    },
                }
            },
        )


def test_activation_only_group_rejected():
    with pytest.raises(ValueError, match="requires weight quantization"):
        SSZModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": {"num_bits": 8, "dynamic": True},
                },
            },
        )


def test_training_schedule_rejected_before_model_mutation():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    original = model[0].weight.detach().clone()
    with pytest.raises(ValueError, match="one-shot"):
        SSZModifier(start=0).initialize(State(model=model))
    assert torch.equal(model[0].weight, original)
    assert not hasattr(model[0], "weight_scale")


@pytest.mark.usefixtures("setup_modifier_factory")
def test_factory_and_python_recipe_roundtrip():
    modifier = ModifierFactory.create(
        type_="SSZModifier",
        allow_registered=True,
        allow_experimental=False,
        step=17,
        ignore=["lm_head"],
    )
    recipe = Recipe.create_instance([modifier])
    restored = Recipe.from_dict(recipe.to_dict()).modifiers[0]
    assert isinstance(restored, SSZModifier)
    assert restored.step == 17
    assert restored.ignore == ["lm_head"]
    assert restored.resolved_config == modifier.resolved_config


def _args(bits=4, **kwargs):
    return QuantizationArgs(
        num_bits=bits, strategy="channel", symmetric=True, observer="ssz", **kwargs
    )


def _observe(weight, bits=4, **kwargs):
    return SSZObserver(base_name="weight", args=_args(bits, **kwargs))(
        weight
    ).get_qparams()


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dynamic_activations", [False, True])
@torch.no_grad()
def test_oneshot_python_recipe_packing_and_reload(
    tmp_path, bits, dtype, dynamic_activations
):
    # Entirely local model/tokenizer: exercise Python modifier composition and oneshot,
    # native packed artifacts, reload without SSZ runtime hooks, and logits.
    torch.manual_seed(4)
    source = tmp_path / "source"
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
                tie_word_embeddings=False,
            )
        )
        .to(dtype)
        .eval()
    )
    model.save_pretrained(source)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "test": 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype).eval()
    original = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    group = {
        "targets": ["Linear"],
        "weights": {"num_bits": bits, "strategy": "channel", "symmetric": True},
    }
    if dynamic_activations:
        group["input_activations"] = {
            "num_bits": 8,
            "strategy": "token",
            "symmetric": True,
            "dynamic": True,
        }
    recipe = [
        SSZModifier(ignore=["lm_head"], step=10, config_groups={"group_0": group})
    ]
    oneshot(model=model, processor=tokenizer, recipe=recipe, pipeline="datafree")
    layer = model.model.layers[0].self_attn.q_proj
    expected = _observe(original, bits, observer_kwargs={"step": 10})
    torch.testing.assert_close(layer.weight_scale, expected["scale"])
    assert torch.equal(layer.weight, original)  # SSZ only solves qparams.
    assert not any(isinstance(module, Observer) for module in model.modules())
    ids = torch.tensor([[1, 2, 3, 4]])
    expected_logits = model(ids).logits
    expected_codes = quantize(
        original,
        layer.weight_scale,
        layer.weight_zero_point,
        layer.quantization_scheme.weights,
        dtype=torch.int8,
    )
    output = tmp_path / "quantized"
    model.save_pretrained(output, save_compressed=True)
    config = json.loads((output / "config.json").read_text())
    weights_config = config["quantization_config"]["config_groups"]["group_0"][
        "weights"
    ]
    assert weights_config["observer"] == "ssz"
    assert weights_config["num_bits"] == bits
    with safe_open(output / "model.safetensors", framework="pt") as saved:
        prefix = "model.layers.0.self_attn.q_proj."
        torch.testing.assert_close(
            saved.get_tensor(prefix + "weight_scale"), expected["scale"]
        )
        if prefix + "weight_packed" in saved.keys():
            packed = saved.get_tensor(prefix + "weight_packed")
            assert packed.dtype == torch.int32
            assert torch.equal(
                unpack_from_int32(packed, bits, original.shape), expected_codes
            )
        else:
            assert saved.get_tensor(prefix + "weight").dtype == torch.int8
            assert torch.equal(saved.get_tensor(prefix + "weight"), expected_codes)
    reloaded = AutoModelForCausalLM.from_pretrained(
        output,
        dtype=dtype,
        quantization_config=CompressedTensorsConfig(dequantize=not dynamic_activations),
    ).eval()
    torch.testing.assert_close(
        reloaded(ids).logits,
        expected_logits,
        atol=1e-6 if dtype == torch.float32 else 2e-3,
        rtol=1e-5 if dtype == torch.float32 else 2e-2,
    )
