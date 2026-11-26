"""Example-local calibration graph and checkpoint-compatible single-step MTP.

Only the calibration context uses the fixed-index attention graph. Nothing is
installed in the serving model, framework source, or serialized Python config.
"""

from contextlib import contextmanager
from copy import deepcopy
from types import MethodType

import torch
from compressed_tensors.offload import (
    align_module_device,
    unwrap_offload_forward,
    update_offload_parameter,
)
from torch import nn
from transformers import Glm5NextForConditionalGeneration, PreTrainedModel
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextHyperConnection,
    Glm5NextTextLinearAttention,
    Glm5NextTextMLP,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
    Glm5NextTextTopkRouter,
)

from llmcompressor.modeling.moe.linear_experts import ExpertMLPWithGate


def mtp_count(config):
    count = getattr(config, "num_nextn_predict_layers", 0) or 0
    if count not in (0, 1):
        raise ValueError("This recipe supports zero or one next-token predictor")
    return count


class Glm53MTPDecoder(nn.Module):
    """DSA + sparse MLP, with ordinary residuals and no hyper connections."""

    def __init__(self, config, index):
        super().__init__()
        config = deepcopy(config)
        config.layer_types = [*config.layer_types, "indexed_attention"]
        config.mlp_layer_types = [*config.mlp_layer_types, "sparse"]
        config.indexer_types = [*config.indexer_types, "full"]
        self.block_type = "indexed_attention"
        self.self_attn = Glm5NextTextAttention(config, index)
        self.mlp = Glm5NextTextMoE(config)
        for key in ("input_layernorm", "post_attention_layernorm", "enorm", "hnorm"):
            setattr(
                self, key, Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
            )
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.shared_head = nn.Module()
        self.shared_head.norm = Glm5NextTextRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.shared_head.head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        residual = hidden_states
        hidden_states, _, _ = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask=attention_mask, **kwargs
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, None


class Glm53RecipeModel(Glm5NextForConditionalGeneration):
    """Register MTP at its checkpoint layer index before the native loader runs."""

    def __init__(self, config):
        super().__init__(config)
        text = config.get_text_config()
        text.num_nextn_predict_layers = mtp_count(text)
        if mtp_count(text):
            self.model.language_model.layers.append(
                Glm53MTPDecoder(text, text.num_hidden_layers)
            )
            self.post_init()
            if (getattr(config, "quarot_config", None) or {}).get(
                "mtp_input_correction"
            ):
                self.rot = nn.Linear(text.hidden_size, text.hidden_size, bias=False)
        # The calibration model stores ordinary convolution/gate parameters in
        # the requested weight dtype. The router correction bias stays FP32.
        self._keep_in_fp32_modules_strict = {"e_score_correction_bias"}

    @property
    def mtp(self):
        # Non-registering view: native save sees loaded MTP and does not append
        # untouched source tensors over the transformed/quantized checkpoint.
        text = self.config.get_text_config()
        return tuple(self.model.language_model.layers[text.num_hidden_layers :])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Native offload/MoE wrappers require a model return, not (model, info).
        # Collect loading diagnostics inside their boundary.
        model, info = PreTrainedModel.from_pretrained.__func__(
            cls, *args, **kwargs, output_loading_info=True
        )
        model._recipe_loading_info = info
        return model


def load_recipe_model(
    source,
    dtype,
    *,
    offload_folder=None,
    device_map="auto_offload",
    linearize_after_load=False,
):
    """Use native MoE conversion/offloading; fail on incomplete MTP checkpoints."""
    from compressed_tensors.offload import load_offloaded_model

    from llmcompressor.modeling.moe.linearize import linearize_moe
    from llmcompressor.utils import load_context

    context = (
        load_offloaded_model(Glm53RecipeModel)
        if linearize_after_load
        else load_context(Glm53RecipeModel, load_mtp=False)
    )
    with context:
        model = Glm53RecipeModel.from_pretrained(
            source,
            dtype=dtype,
            device_map=device_map,
            offload_folder=offload_folder,
        )
    if linearize_after_load:
        model = linearize_moe(model)
    info = model.__dict__.pop("_recipe_loading_info")
    text = model.config.get_text_config()
    aliases = {}
    if mtp_count(text):
        prefix = f"model.language_model.layers.{text.num_hidden_layers}"
        aliases = {
            f"{prefix}.embed_tokens.weight": "model.language_model.embed_tokens",
            f"{prefix}.shared_head.head.weight": "lm_head",
        }
    missing = set(info.get("missing_keys", ())) - aliases.keys()
    if missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"Incomplete recipe checkpoint load: {info}")
    # These are independent copies, as their norms will be folded separately.
    # Source extras use the backbone embedding/head even when duplicate keys exist.
    for target, source_name in aliases.items():
        source_module = model.get_submodule(source_name)
        owner, _, key = target.rpartition(".")
        with align_module_device(source_module):
            update_offload_parameter(
                model.get_submodule(owner), key, source_module.weight.detach().clone()
            )
    return model


def shift_input_ids(input_ids):
    """Remove the first zero; when absent remove column zero; append zero."""
    rows, width = input_ids.shape
    first_zero = (input_ids == 0).int().argmax(1)
    columns = torch.arange(width - 1, device=input_ids.device).expand(rows, -1)
    indices = columns + (columns >= first_zero[:, None]).long()
    return torch.cat(
        (input_ids.gather(1, indices), torch.zeros_like(input_ids[:, :1])), 1
    )


def fixed_index_attention(self, hidden_states, attention_mask=None, **kwargs):
    """Calibration-only zero-index sparse selection and FP32 QK/softmax."""
    batch, length = hidden_states.shape[:2]
    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    query = (
        self.q_b_proj(q_resid).view(batch, length, -1, self.qk_head_dim).transpose(1, 2)
    )
    latent = self.kv_a_proj_with_mqa(hidden_states)
    latent, positional = latent.split((self.kv_lora_rank, self.qk_rope_head_dim), -1)
    projected = self.kv_b_proj(self.kv_a_layernorm(latent))
    projected = projected.view(
        batch, length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    key, value = projected.split((self.qk_nope_head_dim, self.v_head_dim), -1)
    positional = positional.view(batch, 1, length, self.qk_rope_head_dim)
    key = torch.cat((key, positional.expand(*key.shape[:-1], -1)), -1)
    # Repeated zero indices select only key 0. Keep the source's finite-min mask
    # and softmax arithmetic instead of shortcutting the attention to value[0].
    mask = query.new_full((batch, 1, length, length), torch.finfo(query.dtype).min)
    mask[..., 0] = 0
    scores = (query.float() @ key.transpose(-2, -1).float()) * self.scaling
    probabilities = (scores + mask).softmax(-1).to(query.dtype)
    output = (
        (probabilities @ value).transpose(1, 2).contiguous().reshape(batch, length, -1)
    )
    return self.o_proj(output), None, None


def calibration_linear_attention(self, hidden_states, **kwargs):
    """Calibration's simplified KDA: convolution, head attention, output gate.

    No convolution activation, causal mask, L2 normalization or recurrent delta
    update. Forget/beta projections execute, but do not enter this graph's output.
    """
    batch, length = hidden_states.shape[:2]
    projected = torch.cat(
        (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        ),
        -1,
    ).transpose(1, 2)
    convolved = self.conv1d(projected)[..., :length].transpose(1, 2)
    query, key, value = [
        part.reshape(batch, length, -1, self.head_dim)
        for part in convolved.split(self.qkv_dim, -1)
    ]
    self.forget_gate(hidden_states)
    torch.sigmoid(self.b_proj(hidden_states))
    query = query * (self.head_dim**-0.5)
    scores = query.float() @ key.transpose(-2, -1).float()
    probabilities = scores.softmax(-1).to(query.dtype)
    output = probabilities @ value
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(
        batch, length, -1, self.head_dim
    )
    return self.o_proj(self.o_norm(output, gate).reshape(batch, length, -1))


def calibration_mlp(self, hidden_states):
    """Unclipped FP32 SiLU/product/down projection, returning the input dtype.

    Explicitly promote the down weight at this mixed-dtype boundary: ordinary
    CPU Linear rejects FP32 activation with BF16 weights. No parameter is changed.
    """
    gate = self.gate_proj(hidden_states).float()
    up = self.up_proj(hidden_states).float()
    activated = torch.nn.functional.silu(gate) * up
    output = torch.nn.functional.linear(activated, self.down_proj.weight.float())
    return output.to(hidden_states.dtype)


def calibration_router(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    logits = torch.nn.functional.linear(hidden_states.float(), self.weight.float())
    probabilities = logits.sigmoid()
    choices = probabilities + self.e_score_correction_bias
    if self.num_group > 1:
        choices = choices.view(hidden_states.shape[0], self.num_group, -1)
        grouped = choices.topk(2, -1)[0].sum(-1)
        keep = grouped.topk(self.topk_group, -1)[1]
        rejected = torch.ones_like(grouped, dtype=torch.bool).scatter_(1, keep, False)
        choices = choices.masked_fill(rejected[..., None], float("-inf")).flatten(1)
    indices = choices.topk(self.top_k, -1)[1]
    weights = probabilities.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True)
    return logits, weights * self.routed_scaling_factor, indices


def forward(
    self,
    input_ids,
    attention_mask=None,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
    position_ids=None,
    token_type_ids=None,
):
    """Calibration with native image embeddings and single-step MTP inputs.

    MTP keeps the token-shift/enorm contract. Its conditioning hidden states
    include the image features inserted into the backbone, without rerunning ViT.
    """
    language = self.model.language_model
    embeddings = language.embed_tokens(input_ids)
    if pixel_values is not None:
        vision = self.model.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeddings = torch.cat(vision.pooler_output, dim=0).to(embeddings)
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids, inputs_embeds=embeddings, image_features=image_embeddings
        )
        embeddings = embeddings.masked_scatter(image_mask, image_embeddings)
    hidden = (
        embeddings.unsqueeze(2)
        .expand(-1, -1, self.config.text_config.hc_mult, -1)
        .contiguous()
    )
    for layer in language.layers[: self.config.text_config.num_hidden_layers]:
        hidden, _ = layer(hidden, attention_mask=attention_mask)
    for layer in language.layers[self.config.text_config.num_hidden_layers :]:
        hidden = hidden.mean(dim=2)
        # Native calibration disables lm_head.forward by returning meta logits.
        # MTP needs a real prediction; consume its normally onloaded weight.
        logits = torch.nn.functional.linear(
            language.norm(hidden), self.lm_head.weight
        ).float()
        tokens = torch.cat(
            (
                shift_input_ids(input_ids)[:, :-1],
                logits[:, -1].argmax(-1, keepdim=True),
            ),
            -1,
        )
        embedded = layer.enorm(layer.embed_tokens(tokens))
        normalized = layer.hnorm(hidden)
        hidden = layer.eh_proj(torch.cat((embedded, normalized), -1))
        hidden, _ = layer(hidden, attention_mask=attention_mask)
    return hidden


calibration_forward = forward


@contextmanager
def calibration_graph(model):
    """Restore every forward on success or failure; never change serving config."""
    replacements = [(model, calibration_forward)]
    replacements += [
        (module, fixed_index_attention)
        for module in model.modules()
        if isinstance(module, Glm5NextTextAttention)
    ]
    replacements += [
        (module, calibration_linear_attention)
        for module in model.modules()
        if isinstance(module, Glm5NextTextLinearAttention)
    ]
    replacements += [
        (module, calibration_mlp)
        for module in model.modules()
        if isinstance(module, (Glm5NextTextMLP, ExpertMLPWithGate))
    ]
    replacements += [
        (module, calibration_router)
        for module in model.modules()
        if isinstance(module, Glm5NextTextTopkRouter)
    ]
    epsilon_values = [
        (module.input_norm, module.input_norm.eps)
        for module in model.modules()
        if isinstance(module, Glm5NextTextHyperConnection)
    ]
    originals = []
    try:
        for module in model.modules():
            if isinstance(module, Glm5NextTextHyperConnection):
                module.input_norm.eps = module.hc_eps
        for module, forward in replacements:
            local = "forward" in module.__dict__
            with unwrap_offload_forward(module):
                originals.append((module, module.forward.__func__, local))
                module.forward = MethodType(forward, module)
            if hasattr(module, "_original_forward_func"):
                # Native tracing unwraps functools metadata as well as the
                # offload dispatch slot. Both must describe the same function.
                module.forward.__func__.__wrapped__ = forward
        yield
    finally:
        for norm, epsilon in epsilon_values:
            norm.eps = epsilon
        for module, original, local in reversed(originals):
            with unwrap_offload_forward(module):
                module.forward = MethodType(original, module)
            if hasattr(module, "_original_forward_func"):
                module.forward.__func__.__wrapped__ = original
            if not local and not hasattr(module, "_original_forward_func"):
                module.__dict__.pop("forward", None)
