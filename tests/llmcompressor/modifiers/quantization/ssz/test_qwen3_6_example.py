"""Verify example composition without downloading the production model."""

from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmcompressor.modifiers.quantization.ssz import SSZModifier

pytestmark = pytest.mark.unit

EXAMPLE = (
    Path(__file__).resolve().parents[5]
    / "examples/quantization_w8a8_int8/qwen3_6_ssz_example.py"
)


def _load_example():
    spec = spec_from_file_location("ssz_example", EXAMPLE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_python_recipe_and_loading(monkeypatch):
    example = _load_example()
    calls = []
    saved = []
    model = SimpleNamespace(
        eval=lambda: None,
        named_modules=lambda: [],
        save_pretrained=lambda *args, **kwargs: saved.append((args, kwargs)),
    )
    processor = SimpleNamespace(
        save_pretrained=lambda path: saved.append((path, "processor"))
    )
    monkeypatch.setattr(example, "find_spec", lambda name: None)
    monkeypatch.setattr(example, "load_context", lambda cls: nullcontext())
    monkeypatch.setattr(
        example,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
    )
    monkeypatch.setattr(
        example,
        "Qwen3_5ForConditionalGeneration",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    monkeypatch.setattr(example, "oneshot", lambda **kwargs: calls.append(kwargs))
    example.main()
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] is model
    assert call["processor"] is processor
    assert "dataset" not in call
    assert call["pipeline"] == "independent"
    assert "output_dir" not in call
    assert saved == [
        ((example.SAVE_DIR,), {"save_compressed": True}),
        (example.SAVE_DIR, "processor"),
    ]
    (modifier,) = call["recipe"]
    assert isinstance(modifier, SSZModifier)
    assert modifier.scheme == "W8A8"
    assert modifier.step == example.STEP
    assert len(modifier.targets) == 5
    weights = modifier.resolved_config.config_groups["group_0"].weights
    assert weights.observer == "ssz"


def test_rejects_overwriting_source(monkeypatch):
    example = _load_example()
    monkeypatch.setattr(example, "SAVE_DIR", example.MODEL_ID)
    with pytest.raises(ValueError, match="must differ"):
        example.main()


def test_example_preserves_qwen_offset_norms_in_checkpoint(tmp_path, monkeypatch):
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    config_module = pytest.importorskip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    )
    model_module = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    config = config_module.Qwen3_5Config(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["full_attention"],
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
    model = model_module.Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    original_norms = {}
    for name, module in model.named_modules():
        if type(module).__name__ == "Qwen3_5RMSNorm":
            with torch.no_grad():
                module.weight.fill_(0.003)
            original_norms[name + ".weight"] = module.weight.detach().clone()
    assert original_norms
    source = tmp_path / "source"
    model.save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(source)

    example = _load_example()
    output = tmp_path / "quantized"
    monkeypatch.setattr(example, "MODEL_ID", str(source))
    monkeypatch.setattr(example, "SAVE_DIR", str(output))
    monkeypatch.setattr(example, "STEP", 3)
    monkeypatch.setattr(example, "find_spec", lambda name: None)
    monkeypatch.setattr(
        example.AutoProcessor, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    example.main()

    with safe_open(output / "model.safetensors", framework="pt") as saved:
        for name, original in original_norms.items():
            assert torch.equal(saved.get_tensor(name), original), name
        prefix = "model.language_model.layers.0.self_attn.q_proj."
        assert saved.get_tensor(prefix + "weight").dtype == torch.int8
        assert torch.isfinite(saved.get_tensor(prefix + "weight_scale")).all()
