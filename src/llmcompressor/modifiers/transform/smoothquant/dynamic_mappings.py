"""
Dynamic SmoothQuant mapping builders for architectures that need model-aware logic.
"""

from collections.abc import Callable

from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.transform.smoothquant.utils import (
    DEFAULT_SMOOTHQUANT_MAPPINGS,
    MAPPINGS_REGISTRY,
    LayerMap,
)
from llmcompressor.modifiers.transform.utils.hybrid_attention import (
    _get_config_metadata,
    get_config_layer_types,
)

__all__ = ["SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY", "get_layer_mappings_from_model"]


def get_layer_mappings_from_model(model: Module) -> list[LayerMap]:
    """
    Infer SmoothQuant mappings from a model.

    Checks the dynamic mapping registry first for model-aware builders, then falls back
    to the static architecture registry, then to the default mappings.

    :param model: model instance used to infer mappings
    :return: list of SmoothQuant LayerMap entries for the model
    """
    architecture = model.__class__.__name__

    if architecture in SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY:
        return SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY[architecture](model)

    if architecture in MAPPINGS_REGISTRY:
        return MAPPINGS_REGISTRY[architecture]

    logger.info(
        f"Architecture {architecture} not found in mappings. "
        f"Using default mappings: {DEFAULT_SMOOTHQUANT_MAPPINGS}"
    )
    return DEFAULT_SMOOTHQUANT_MAPPINGS


def _build_qwen3_5_smoothquant_mappings(
    model: Module, mlp_balance_layers: list[str]
) -> list[LayerMap]:
    """
    Shared SmoothQuant mapping builder for Qwen3.5 hybrid-attention models.

    Restricts the attention input_layernorm regex to the indices of
    full-attention layers (linear-attention layers do not expose q/k/v
    projections), and uses the caller-provided MLP balance layers for the
    post_attention_layernorm mapping.
    """
    layer_types = get_config_layer_types(model)
    if layer_types is None:
        raise ValueError(
            "Qwen3.5 SmoothQuant mappings require model.config.text_config."
            "layer_types (or model.config.layer_types) to identify full_attention "
            "layers."
        )

    full_attn_indices = [
        str(index)
        for index, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]
    if not full_attn_indices:
        raise ValueError(
            "Qwen3.5 SmoothQuant mappings require at least one full_attention "
            "layer in layer_types."
        )

    joined = "|".join(full_attn_indices)
    smooth_pattern = rf"re:.*layers\.({joined})\.input_layernorm$"

    return [
        LayerMap(
            balance_layers=[
                "re:.*self_attn\\.q_proj$",
                "re:.*self_attn\\.k_proj$",
                "re:.*self_attn\\.v_proj$",
            ],
            smooth_layers=smooth_pattern,
        ),
        LayerMap(
            balance_layers=mlp_balance_layers,
            smooth_layers="re:.*post_attention_layernorm$",
        ),
    ]


def build_qwen3_5_moe_smoothquant_mappings(model: Module) -> list[LayerMap]:
    """
    Build SmoothQuant mappings for Qwen3.5 MoE hybrid-attention models.

    Only full-attention layers expose self_attn q/k/v projections, so the input
    layernorm regex must be restricted to those layer indices. The shared expert MLP
    remains safe to smooth with the standard post-attention layernorm mapping.
    """
    return _build_qwen3_5_smoothquant_mappings(
        model,
        mlp_balance_layers=[
            "re:.*mlp\\.shared_expert\\.gate_proj$",
            "re:.*mlp\\.shared_expert\\.up_proj$",
        ],
    )


def build_qwen3_5_dense_smoothquant_mappings(model: Module) -> list[LayerMap]:
    """
    Build SmoothQuant mappings for dense Qwen3.5 hybrid-attention models.

    Dense Qwen3.5 variants expose a regular ``mlp.gate_proj``/``mlp.up_proj``
    pair instead of the MoE ``shared_expert`` submodule.
    """
    return _build_qwen3_5_smoothquant_mappings(
        model,
        mlp_balance_layers=[
            "re:.*mlp\\.gate_proj$",
            "re:.*mlp\\.up_proj$",
        ],
    )


def build_glm5_next_smoothquant_mappings(model: Module) -> list[LayerMap]:
    """
    Build SmoothQuant mappings for GLM-5.3-Flash (``Glm5Next*``) models.

    FFN-side only (``post_attention_layernorm`` -> MLP projections), matching the
    msmodelslim ``get_ln_fuse_map`` targets that produce the leading ~2.5pp
    accuracy gap between llm-compressor and ms W4A4C8 on GLM-5.3-Flash:

    * dense layers balance ``mlp.gate_proj``/``mlp.up_proj`` (no router);
    * MoE layers balance every routed expert's ``gate_proj``/``up_proj``, the
      shared experts' ``gate_proj``/``up_proj``, and the ``mlp.gate`` router.

    The dense/MoE split is driven by ``config.mlp_layer_types`` (read through
    ``text_config`` first, like the Qwen3.5 builders). The router is scoped to
    MoE layers only because dense layers have no ``mlp.gate`` module -- mixing
    it into a single unscoped mapping would collapse the per-layer grouping in
    ``match_modules_set`` (the same collapse fixed for GlmMoeDsa in #2923).
    """
    mlp_layer_types = _get_config_metadata(model, "mlp_layer_types")
    num_layers = _get_config_metadata(model, "num_hidden_layers")
    if mlp_layer_types is None or num_layers is None:
        raise ValueError(
            "GLM-5.3 SmoothQuant mappings require config.mlp_layer_types and "
            "config.num_hidden_layers (checked on text_config as well)."
        )

    dense_indices: list[str] = []
    moe_indices: list[str] = []
    for layer_idx in range(num_layers):
        mlp_type = (
            mlp_layer_types[layer_idx]
            if layer_idx < len(mlp_layer_types)
            else "sparse"
        )
        (dense_indices if mlp_type == "dense" else moe_indices).append(str(layer_idx))

    mappings: list[LayerMap] = []
    if dense_indices:
        joined = "|".join(dense_indices)
        mappings.append(
            LayerMap(
                balance_layers=[
                    rf"re:.*layers\.({joined})\.mlp\.gate_proj$",
                    rf"re:.*layers\.({joined})\.mlp\.up_proj$",
                ],
                smooth_layers=rf"re:.*layers\.({joined})\.post_attention_layernorm$",
            )
        )
    if moe_indices:
        joined = "|".join(moe_indices)
        mappings.append(
            LayerMap(
                balance_layers=[
                    rf"re:.*layers\.({joined})\.mlp\.experts\..*\.gate_proj$",
                    rf"re:.*layers\.({joined})\.mlp\.experts\..*\.up_proj$",
                    rf"re:.*layers\.({joined})\.mlp\.shared_experts\.gate_proj$",
                    rf"re:.*layers\.({joined})\.mlp\.shared_experts\.up_proj$",
                    rf"re:.*layers\.({joined})\.mlp\.gate$",
                ],
                smooth_layers=rf"re:.*layers\.({joined})\.post_attention_layernorm$",
            )
        )
    return mappings


SMOOTHQUANT_DYNAMIC_MAPPING_REGISTRY: dict[str, Callable[[Module], list[LayerMap]]] = {
    "Qwen3_5MoeForConditionalGeneration": build_qwen3_5_moe_smoothquant_mappings,
    "Qwen3_5MoeForCausalLM": build_qwen3_5_moe_smoothquant_mappings,
    "Qwen3_5ForCausalLM": build_qwen3_5_dense_smoothquant_mappings,
    "Qwen3_5ForConditionalGeneration": build_qwen3_5_dense_smoothquant_mappings,
    "Glm5NextForConditionalGeneration": build_glm5_next_smoothquant_mappings,
}
