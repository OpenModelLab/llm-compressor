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
    model = SimpleNamespace(eval=lambda: None)
    processor = object()
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
    assert call["save_compressed"] is True
    assert call["output_dir"] == example.SAVE_DIR
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
