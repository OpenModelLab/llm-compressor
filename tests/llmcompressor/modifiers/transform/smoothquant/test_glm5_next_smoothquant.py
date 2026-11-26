"""Tests for GLM-5.3-Flash SmoothQuant: the dense/MoE-aware FFN mapping and the
FlexSmooth formula alignment that reproduce msmodelslim's per-channel fold
``s = absmax(a)^alpha / absmax(w)^beta``.

These tests are unit-level: they mock the ``Glm5NextForConditionalGeneration``
config (``SimpleNamespace``) and drive ``build_glm5_next_smoothquant_mappings`` /
``SmoothQuantModifier._calculate_smoothing_scales`` / ``_apply_smoothing``
directly, so they need no HF checkpoint, no weights, and no accelerator.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import LayerNorm, Linear

from llmcompressor.modifiers.transform.smoothquant.base import (
    SmoothQuantMapping,
    SmoothQuantModifier,
    SmoothQuantScale,
)
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY,
    build_glm5_next_smoothquant_mappings,
    get_layer_mappings_from_model,
)


def _make_glm5next_mock(mlp_layer_types, num_hidden_layers):
    model = type("Glm5NextForConditionalGeneration", (), {})()
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            mlp_layer_types=mlp_layer_types,
            num_hidden_layers=num_hidden_layers,
        )
    )
    return model


# --- mapping builder ---------------------------------------------------------


@pytest.mark.unit
def test_build_glm5_next_smoothquant_mappings_dense_moe_split():
    model = _make_glm5next_mock(
        mlp_layer_types=["dense", "dense", "sparse", "sparse", "sparse"],
        num_hidden_layers=5,
    )

    mappings = build_glm5_next_smoothquant_mappings(model)
    assert len(mappings) == 2

    dense, moe = mappings
    assert dense.smooth_layers == "re:.*layers\\.(0|1)\\.post_attention_layernorm$"
    assert dense.balance_layers == [
        "re:.*layers\\.(0|1)\\.mlp\\.gate_proj$",
        "re:.*layers\\.(0|1)\\.mlp\\.up_proj$",
    ]

    # MoE balances routed experts + shared experts + the router (mlp.gate).
    assert moe.smooth_layers == "re:.*layers\\.(2|3|4)\\.post_attention_layernorm$"
    assert moe.balance_layers == [
        "re:.*layers\\.(2|3|4)\\.mlp\\.experts\\..*\\.gate_proj$",
        "re:.*layers\\.(2|3|4)\\.mlp\\.experts\\..*\\.up_proj$",
        "re:.*layers\\.(2|3|4)\\.mlp\\.shared_experts\\.gate_proj$",
        "re:.*layers\\.(2|3|4)\\.mlp\\.shared_experts\\.up_proj$",
        "re:.*layers\\.(2|3|4)\\.mlp\\.gate$",
    ]


@pytest.mark.unit
def test_build_glm5_next_smoothquant_mappings_all_moe():
    model = _make_glm5next_mock(
        mlp_layer_types=["sparse", "sparse"], num_hidden_layers=2
    )

    mappings = build_glm5_next_smoothquant_mappings(model)
    # No dense layer -> a single MoE-only mapping (with the router balanced).
    assert len(mappings) == 1
    assert mappings[0].smooth_layers == "re:.*layers\\.(0|1)\\.post_attention_layernorm$"
    assert any("mlp\\.gate$" in p for p in mappings[0].balance_layers)


@pytest.mark.unit
def test_build_glm5_next_smoothquant_mappings_all_dense():
    model = _make_glm5next_mock(
        mlp_layer_types=["dense", "dense"], num_hidden_layers=2
    )

    mappings = build_glm5_next_smoothquant_mappings(model)
    # No MoE layer -> dense-only mapping, no router in the balance layers.
    assert len(mappings) == 1
    assert mappings[0].smooth_layers == "re:.*layers\\.(0|1)\\.post_attention_layernorm$"
    assert mappings[0].balance_layers == [
        "re:.*layers\\.(0|1)\\.mlp\\.gate_proj$",
        "re:.*layers\\.(0|1)\\.mlp\\.up_proj$",
    ]
    assert not any("mlp\\.gate$" in p for p in mappings[0].balance_layers)


@pytest.mark.unit
def test_build_glm5_next_smoothquant_mappings_out_of_range_defaults_to_sparse():
    # MTP layers sit beyond mlp_layer_types; they default to sparse and must be
    # included as MoE layers (not dropped).
    model = _make_glm5next_mock(
        mlp_layer_types=["dense", "sparse"], num_hidden_layers=3
    )

    mappings = build_glm5_next_smoothquant_mappings(model)
    moe = mappings[1]
    assert moe.smooth_layers == "re:.*layers\\.(1|2)\\.post_attention_layernorm$"


@pytest.mark.unit
def test_build_glm5_next_smoothquant_mappings_requires_metadata():
    model = type("Glm5NextForConditionalGeneration", (), {})()
    model.config = SimpleNamespace(text_config=SimpleNamespace())

    with pytest.raises(ValueError, match="mlp_layer_types"):
        build_glm5_next_smoothquant_mappings(model)


@pytest.mark.unit
def test_glm5_next_uses_dynamic_registry():
    model = _make_glm5next_mock(
        mlp_layer_types=["dense", "sparse"], num_hidden_layers=2
    )

    assert "Glm5NextForConditionalGeneration" in SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY
    mappings = get_layer_mappings_from_model(model)
    assert len(mappings) == 2
    assert mappings[0].smooth_layers == "re:.*layers\\.(0)\\.post_attention_layernorm$"


# --- FlexSmooth formula alignment -------------------------------------------


@pytest.mark.unit
def test_smoothing_scales_flexsmooth_formula():
    """Reproduce msmodelslim ``s = absmax(a)^alpha / absmax(w)^beta`` exactly:

    * ``weight_scale_multiplier=1.0`` drops the original 2.0 coefficient;
    * independent ``smoothing_beta`` replaces ``beta = 1 - alpha``.
    """
    linear = Linear(4, 8)
    linear.weight.data.fill_(1.0)  # per-input-channel absmax = 1

    sq = SmoothQuantModifier(
        smoothing_strength=0.5,
        smoothing_beta=0.5,
        weight_scale_multiplier=1.0,
    )
    activation_scales = torch.tensor([1.0, 2.0, 3.0, 4.0])

    scales = sq._calculate_smoothing_scales([linear], activation_scales)

    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).pow(0.5) / torch.tensor(
        [1.0, 1.0, 1.0, 1.0]
    ).pow(0.5)
    torch.testing.assert_close(scales, expected)


@pytest.mark.unit
def test_smoothing_scales_default_formula_unchanged():
    """Without the FlexSmooth knobs, the original SmoothQuant formulation is
    preserved (``s = dynamic_range(a)^alpha / (2 * absmax(w))^(1-alpha)``."""
    linear = Linear(4, 8)
    linear.weight.data.fill_(1.0)

    sq = SmoothQuantModifier(smoothing_strength=0.5)  # beta=None -> 1-alpha
    activation_scales = torch.tensor([1.0, 2.0, 3.0, 4.0])

    scales = sq._calculate_smoothing_scales([linear], activation_scales)

    # weight term = 2.0 * absmax(w) = 2.0; beta = 1 - 0.5 = 0.5
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).pow(0.5) / torch.tensor(
        [2.0, 2.0, 2.0, 2.0]
    ).pow(0.5)
    torch.testing.assert_close(scales, expected)


# --- fold direction + absmax statistics -------------------------------------


@pytest.mark.unit
def test_apply_smoothing_absmax_fold_direction():
    """With ``activation_absmax=True`` the scale uses ``max(|min|,|max|)``, and
    the fold pushes it into ``norm /= s`` + ``linear *= s`` (the msmodelslim
    norm-linear convention, not the up-down pairing)."""
    model = torch.nn.Module()
    model.norm = LayerNorm(4, elementwise_affine=True)
    model.norm.weight.data.fill_(1.0)
    model.norm.bias.data.zero_()
    model.linear = Linear(4, 8, bias=False)
    model.linear.weight.data.fill_(1.0)

    sq = SmoothQuantModifier(
        smoothing_strength=0.5,
        smoothing_beta=0.5,
        activation_absmax=True,
        weight_scale_multiplier=1.0,
    )
    sq.scales_ = {
        "norm": SmoothQuantScale(
            min_channel_vals=torch.tensor([-1.0, -2.0, -3.0, -4.0]),
            max_channel_vals=torch.tensor([1.0, 1.0, 1.0, 2.0]),
        )
    }
    sq.resolved_mappings_ = [SmoothQuantMapping("norm", model.norm, [model.linear])]

    sq._apply_smoothing(model)

    # absmax = max(|min|, |max|) = [1, 2, 3, 4]; linear absmax = [1, 1, 1, 1].
    expected_s = torch.tensor([1.0, 2.0, 3.0, 4.0]).pow(0.5)

    # linear weight *= s (per input channel, broadcast over output rows).
    torch.testing.assert_close(model.linear.weight[0], expected_s)
    torch.testing.assert_close(model.linear.weight[3], expected_s)
    # norm weight /= s.
    torch.testing.assert_close(model.norm.weight, 1.0 / expected_s)