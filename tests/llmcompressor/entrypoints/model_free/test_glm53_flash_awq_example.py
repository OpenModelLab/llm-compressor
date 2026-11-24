"""Native GLM AWQ experiment B; CPU tests skip only accelerator pinning."""

import importlib.util
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


def helpers_module():
    path = Path(__file__).with_name("test_glm53_flash_gptq_example.py")
    spec = importlib.util.spec_from_file_location("glm_awq_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example(monkeypatch):
    folder = Path(__file__).resolve().parents[4] / ("examples/quantization_w4a8_int8")
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "glm_awq_example", folder / "glm53_flash_w4a8c8_awq_example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cpu_pinning(monkeypatch):
    from llmcompressor.pipelines.cache import IntermediatesCache

    # The CPU-only Torch build cannot pin memory. No calibration, quantization,
    # scale search, routing or export computation is substituted in these tests.
    monkeypatch.setattr(IntermediatesCache, "pin_memory", lambda *args, **kwargs: None)


def test_awq_mapping_keeps_all_moe_consumers(example, tmp_path):
    model = helpers_module().tiny_model(tmp_path)
    mappings = example.make_awq_mappings(model)
    assert len(mappings) == 1
    assert mappings[0].smooth_layer.endswith("layers.1.post_attention_layernorm")
    assert len(mappings[0].balance_layers) == 7
    assert "model.language_model.layers.1.mlp.gate" in mappings[0].balance_layers
    assert all("layers.1.mlp." in name for name in mappings[0].balance_layers)


@pytest.mark.parametrize("duo_scaling", [False, True])
def test_real_awq_mixed_export_without_mixed_search(
    example, cpu_pinning, monkeypatch, tmp_path, duo_scaling
):
    import llmcompressor.modifiers.transform.awq.base as awq_base

    helpers = helpers_module()
    model = helpers.tiny_model(tmp_path, with_mtp=True)
    attention = model.model.language_model.layers[0].self_attn.q_proj.weight.clone()
    seen_bits = []
    native_forward_quantize = awq_base.forward_quantize

    def record_quantization(module, tensor, base_name, args, **kwargs):
        assert args.model_dump() == module.quantization_scheme.weights.model_dump()
        seen_bits.append(args.num_bits)
        return native_forward_quantize(module, tensor, base_name, args, **kwargs)

    monkeypatch.setattr(awq_base, "forward_quantize", record_quantization)
    result = example.calibrate_awq(
        model,
        helpers.tiny_batches(),
        processor=helpers.tiny_tokenizer(),
        n_grid=4,
        duo_scaling=duo_scaling,
    )
    assert seen_bits and set(seen_bits) == {4}
    assert len(result["per_mapping_errors"]) == 1
    assert all(n > 0 for n in result["forward_token_counts_including_replay"].values())
    assert len(result["forward_token_counts_including_replay"]) == 6
    metric = result["per_mapping_errors"][0]
    assert metric["best_error"] <= metric["initial_error"] + 1e-8
    output = tmp_path / "output"
    model.save_pretrained(output, save_compressed=True)
    audit = example.audit_config(output)
    assert audit["projections"] == dict(dense=3, routed=6, shared=3)
    saved = {}
    for shard in output.glob("*.safetensors"):
        saved.update(load_file(shard))
    assert sum(name.endswith(".weight_packed") for name in saved) == 6
    assert (
        sum(
            name.endswith(".weight") and t.dtype == torch.int8
            for name, t in saved.items()
        )
        == 6
    )
    torch.testing.assert_close(
        saved["model.language_model.layers.0.self_attn.q_proj.weight"],
        attention,
        rtol=0,
        atol=0,
    )
    assert saved["model.language_model.layers.1.mlp.gate.weight"].is_floating_point()
    assert saved[
        "model.language_model.layers.2.mlp.experts.0.up_proj.weight"
    ].is_floating_point()
    torch.testing.assert_close(
        saved["model.language_model.layers.2.mlp.experts.0.up_proj.weight"],
        torch.full((16, 32), 0.25),
        rtol=0,
        atol=0,
    )


def test_awq_native_folding_preserves_float_moe_and_router(
    example, cpu_pinning, monkeypatch, tmp_path
):
    from compressed_tensors.quantization import apply_quantization_config
    from compressed_tensors.utils import update_offload_parameter

    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from llmcompressor.utils.helpers import DisableQuantization

    model = helpers_module().tiny_model(tmp_path).eval()
    layer = model.model.language_model.layers[1]
    update_offload_parameter(
        layer.mlp.gate, "weight", torch.randn_like(layer.mlp.gate.weight) * 0.05
    )
    groups = example.make_config_groups(2)
    apply_quantization_config(
        model,
        QuantizationModifier(
            config_groups={"routed": groups["routed"]}
        ).resolve_quantization_config(),
    )
    awq = AWQModifier(
        mappings=example.make_awq_mappings(model),
        duo_scaling=False,
        offload_device=torch.device("cpu"),
    )
    awq._set_resolved_mappings(model)
    awq._setup_activation_cache_hooks()
    scales = torch.linspace(0.5, 2, 32)
    # This test isolates native scale folding with a known scale; the tests
    # above separately exercise the real AWQ grid search and quantization.
    monkeypatch.setattr(AWQModifier, "_compute_best_scale", lambda *args: scales)
    probes = torch.randn(1, 8, 32)
    with torch.no_grad(), DisableQuantization(model):
        normalized = layer.post_attention_layernorm(probes)
        expected = layer.mlp(normalized)
        router = layer.mlp.gate(normalized.reshape(-1, 32))
        try:
            awq._apply_smoothing(model)
        finally:
            awq.remove_hooks()
        updated = layer.post_attention_layernorm(probes)
        actual = layer.mlp(updated)
        router_after = layer.mlp.gate(updated.reshape(-1, 32))
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(router_after, router, rtol=1e-4, atol=1e-5)
    assert not hasattr(layer.mlp.gate, "weight_scale")


def test_invalid_search_grid_rejected(example, tmp_path):
    model = helpers_module().tiny_model(tmp_path)
    with pytest.raises(ValueError, match="n_grid"):
        example.calibrate_awq(model, [], processor=None, n_grid=2)
