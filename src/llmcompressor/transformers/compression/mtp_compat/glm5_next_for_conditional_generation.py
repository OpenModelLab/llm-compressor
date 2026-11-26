"""MTP adaptations for the Glm5NextForConditionalGeneration architecture.

GLM-5.3-Flash stores its MTP draft layer (checkpoint key ``layers.45``) with
DSA attention + MoE but *without* the mHC hyper-connection weights carried by
the backbone decoder layers. Transformers' ``Glm5NextTextDecoderLayer`` builds
``attn_hc``/``ffn_hc`` unconditionally and its ``forward`` dereferences them,
so the draft layer needs a dedicated no-HC subclass. The checkpoint keys also
live under the VLM ``model.language_model.*`` prefix (not a text model's
``model.*``), which ``GlmMoeDsaMtpModel`` does not handle. This adapter pins
the draft layer class, the norm class, and the VLM prefix.
"""

from copy import deepcopy

import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.core_model_loading import PrefixChange, WeightRenaming
from transformers.modeling_layers import MtpLayer, MtpModel
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextDecoderLayer,
    Glm5NextTextLinearAttention,
    Glm5NextTextMLP,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
)


class Glm5NextMtpDecoderLayer(Glm5NextTextDecoderLayer):
    """Glm5NextTextDecoderLayer minus the mHC hyper-connection streams."""

    def __init__(self, config, layer_idx):
        # Do not call Glm5NextTextDecoderLayer.__init__: it constructs
        # attn_hc/ffn_hc unconditionally (the draft layer has no such weights
        # in the checkpoint). Reproduce the rest field-for-field.
        nn.Module.__init__(self)
        self.block_type = config.layer_types[layer_idx]
        self.hidden_size = config.hidden_size
        self.self_attn = (
            Glm5NextTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextTextAttention(config, layer_idx)
        )
        self.mlp = (
            Glm5NextTextMoE(config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else Glm5NextTextMLP(config)
        )
        self.input_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        prev_topk_indices=None,
        **kwargs,
    ):
        # Plain pre-norm residual path (no hyper-connection streams). Returns a
        # single tensor to match MtpLayer.forward, which assigns the block
        # output directly to `hidden_states` before post-norm.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                prev_topk_indices=prev_topk_indices,
                **kwargs,
            )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Glm5NextMtpModel(MtpModel):
    @staticmethod
    def get_mtp_config(model: PreTrainedModel) -> PretrainedConfig:
        text_config = model.config.get_text_config()
        num_layers = getattr(text_config, "num_nextn_predict_layers", None)
        if num_layers is None:
            num_layers = getattr(text_config, "num_mtp_layers", None)
        if num_layers != 1:
            raise ValueError("GLM-5.3-Flash MTP requires one prediction layer")
        config = deepcopy(text_config)
        config.num_mtp_layers = config.num_hidden_layers = num_layers
        # The draft layer is always MoE with its own DSA indexer, even though
        # the backbone is hybrid KDA/DSA. first_k_dense_replace is a harmless
        # no-op on Glm5Next (the field was removed from Glm5NextTextConfig).
        config.first_k_dense_replace = 0
        config.mlp_layer_types = ["sparse"]
        config.layer_types = ["indexed_attention"]
        config.indexer_types = ["full"]
        return config

    @staticmethod
    def get_mtp_weight_conversions():
        # Order matters: strip the VLM prefix before the anchored renamings
        # match the now-prefix-free ``layers.0.mtp_block.*`` names.
        return [
            PrefixChange(prefix_to_remove="model.language_model"),
            WeightRenaming(r"^layers\.0\.mtp_block\.enorm\.", "layers.0.enorm."),
            WeightRenaming(r"^layers\.0\.mtp_block\.hnorm\.", "layers.0.hnorm."),
            WeightRenaming(r"^layers\.0\.mtp_block\.eh_proj\.", "layers.0.eh_proj."),
            WeightRenaming(
                r"^layers\.0\.mtp_block\.shared_head\.norm\.",
                "layers.0.post_norm.",
            ),
        ]

    def __init__(self, main_model: PreTrainedModel, num_mtp_layers: int):
        # Reimplement MtpModel.__init__ so we can pin the draft decoder layer
        # class (no-HC) and the norm class. The base implementation infers
        # layer_cls = type(backbone.layers[-1]) -- the HC backbone layer -- and
        # norm_cls by name from the same layer; both are wrong for the draft.
        PreTrainedModel.__init__(self, main_model.config.get_mtp_config())
        self.loss_type = "ForCausalLM"
        self.num_mtp_layers = num_mtp_layers
        self.use_post_norm = True
        self.use_shared_post_norm = False
        if hasattr(self.config, "chain_hidden_post_norm"):
            self.use_post_norm = False
            self.use_shared_post_norm = self.config.chain_hidden_post_norm

        self.layers = nn.ModuleList(
            [
                MtpLayer(
                    self.config,
                    Glm5NextMtpDecoderLayer,
                    Glm5NextTextRMSNorm,
                    k,
                    self.use_post_norm,
                )
                for k in range(num_mtp_layers)
            ]
        )
        if self.use_shared_post_norm:
            self.shared_post_norm = Glm5NextTextRMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps
            )

        self.tie_with_main_model(main_model)
        # GLM-5.3-Flash uses a flat (axial) RoPE with no per-layer RoPE parameter
        # tables, so there is nothing to pass as `layer_type` to the rotary
        # embedding. Keep the attribute (MtpModel.forward_layer reads it), but it
        # is always False. Do NOT copy the upstream `nested_rope_parameter_keys(...)`
        # lookup here: that came from a Transformers where `Glm5NextTextConfig`
        # still had `rope_parameters`; the modular build deletes it
        # (`rope_parameters = AttributeError()`), so touching `self.config.rope_parameters`
        # raises AttributeError.
        self.has_per_layer_rope = False
        self.post_init()

    def get_weight_conversions(self, main_model: PreTrainedModel):
        num_hidden_layers = main_model.config.get_text_config().num_hidden_layers
        return [
            WeightRenaming(rf"layers\.{num_hidden_layers}\.", "layers.0.mtp_block."),
            *self.get_mtp_weight_conversions(),
            *main_model._weight_conversions,
        ]
