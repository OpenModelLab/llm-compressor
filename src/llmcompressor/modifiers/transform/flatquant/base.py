"""
FlatQuant: learnable Kronecker-factored activation transforms, as described
in https://arxiv.org/abs/2410.09426 (ported from msmodelslim).
"""

import inspect
from dataclasses import dataclass, field

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.offload.dist_utils import is_distributed
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformScheme,
    apply_transform_config,
)
from compressed_tensors.utils import (
    align_modules,
    find_unique_name,
    get_execution_device,
    match_modules_set,
    update_offload_parameter,
)
from loguru import logger
from pydantic import ConfigDict, Field, PrivateAttr, field_validator
from torch.nn import Module, Parameter
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.core import Event, State, active_session
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization import QuantizationMixin
from llmcompressor.modifiers.transform.awq.base import (
    get_lowest_common_ancestor_with_avoid,
)
from llmcompressor.modifiers.transform.flatquant.components import (
    ActivationQuantizer,
    GeneralMatrixTrans,
    WeightQuantizer,
    get_decompose_dim,
    get_init_scale,
)
from llmcompressor.modifiers.transform.flatquant.mappings import (
    ATTN_LINEAR_LINEAR,
    NORM_LINEAR,
    FlatQuantMapping,
    ResolvedFlatQuantMapping,
    get_layer_mappings_from_model,
)
from llmcompressor.modifiers.transform.flatquant.patch import (
    FLATQUANT_STATE_ATTR,
    MODE_CALIB,
    MODE_EVAL,
    MODE_ORG,
    FlatQuantState,
    get_patched_modules,
    patch_flatquant_linear,
    register_flatquant_trans,
    set_flatquant_mode,
    unpatch_flatquant_module,
)
from llmcompressor.modifiers.transform.flatquant.runtime import FlatQuantInputTransform
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.modifiers.utils.pytorch_helpers import is_moe_model
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.sentinel import Sentinel
from llmcompressor.utils.helpers import calibration_forward_context
from llmcompressor.utils.pytorch.module import get_module_to_name_dict

__all__ = ["FlatQuantModifier"]

# parameter-name substrings of the transform / quantizer parameters, used to
# collect trainable parameters (mirrors msmodelslim's get_trainable_parameters)
BASE_LR_KEYS = ("linear_u", "linear_v", "trans_linear", "linear_diag", "diag_scale")
CLIP_LR_KEYS = ("clip_factor",)
TRAINABLE_KEYS = BASE_LR_KEYS + CLIP_LR_KEYS


def _is_ancestor_name(ancestor: str, descendant: str) -> bool:
    """
    Return True if the module name ``ancestor`` refers to an ancestor of (or
    the same module as) the module named ``descendant``.
    """
    return (
        not ancestor or descendant == ancestor or descendant.startswith(ancestor + ".")
    )


@dataclass
class TrainingUnit:
    """
    A group of structure-pair mappings which are trained together.

    Mappings are first grouped by the lowest common ancestor of all modules
    they involve (source + targets), so that the ancestor's forward replay
    covers every transform of the group. Groups whose ancestors are nested
    (e.g. ``self_attn`` inside a decoder layer) are then merged into the
    outermost ancestor, yielding one unit per transformer layer, which is
    the granularity at which the msmodelslim blueprint trains.

    :param parent: module whose forward is replayed to extract fp teacher
        outputs and to train the transforms
    :param parent_name: name of the parent module
    :param mappings: resolved mappings merged into this unit
    :param modules: all patched modules below parent, refreshed after
        patching (linears carrying a FlatQuantState)
    """

    parent: Module
    parent_name: str
    mappings: list[ResolvedFlatQuantMapping] = field(default_factory=list)
    modules: list[Module] = field(default_factory=list)


class FlatQuantModifier(Modifier):
    """
    Implements the FlatQuant algorithm, as described in
    https://arxiv.org/abs/2410.09426, ported from msmodelslim.

    For each supported structure pair (norm -> linears, v_proj -> o_proj,
    up_proj -> down_proj), a learnable row-vector transform
    ``A = diag(d) @ kron(L, R)`` is applied to the target Linear inputs,
    while ``A^-T`` is applied to the target weights. The transform parameters
    (orthogonal Kronecker factors, learnable diagonal, activation/weight clip
    factors) are optimized with a per-layer MSE loss against the fp outputs of
    the layer, using cached calibration data. After training, weight transforms
    are absorbed on CPU in FP64. Input transforms, including the diagonal, are
    exported through compressed-tensors TransformConfig. Source norms are not
    patched or modified, so shared consumers and temporary offset-norm restore
    are safe. Exact floating-point equivalence only applies with quantization
    and clipping disabled; learned clipping is intentionally lossy.

    Like AWQModifier, this is a transform-based modifier: it does not
    quantize anything by itself and must be applied before a modifier which
    inherits from QuantizationMixin, which will then observe the transformed
    weights. During training, fake-quantizers matching the intended
    downstream scheme (``w_bits``/``a_bits``/symmetry/lwc/lac) are used to
    shape the transforms. The training patches are removed after optimization;
    native input hooks transform and clip activations before the downstream
    quantized forward. Use load_flatquant_model for explicit checkpoint reload
    on Transformers versions which do not reconstruct transform_config.

    Example recipe:
    ```yaml
    - FlatQuantModifier:
        epochs: 15
        lr: 5.0e-3
    - QuantizationModifier:
        scheme: W4A4
        targets: Linear
    ```

    Lifecycle:

    - on_initialize
        - set unresolved mappings if not set by user, based on architecture
    - on_calibration_start
        - resolve mappings against the model, constructing transforms
        - patch the forward of involved linears (org mode)
        - freeze model parameters inside the training units
        - attach forward-arg cache hooks on each unit parent and input-max
            statistics hooks on each target linear
    - on_sequential_epoch_end
        - for each unit whose parent ran during this subgraph pass:
            - extract fp teacher outputs by replaying cached parent args
            - initialize learnable diagonals from weight/activation maxima
            - train all transforms of the unit jointly (one step per sample)
            - absorb the transforms into the involved weights and freeze
    - on_calibration_end
        - raise if any cached data was left unconsumed
        - remove hooks (falling back any never-executed units to original)
    - on_finalize
        - clear training state; native serializable input transforms remain

    Deviations from the msmodelslim blueprint, all noted for maintainers:

    - training patches are replaced with native input transform hooks for eval
    - diagonals remain online, rather than modifying shared source parameters
    - quantization scales/ranges use compressed-tensors numerics; learnable
        activation clipping is an explicit clamp before native dynamic QDQ
    - the fp teacher outputs are computed in the model's native dtype rather
        than casting each layer to fp32
    - the learnable diagonal is initialized from the elementwise max of the
        input-max statistics over all targets of the pair (the blueprint leaks
        a loop variable and uses the last target's statistics)
    - distributed execution is not supported (parameters would diverge across
        ranks without gradient synchronization)

    :param mappings: list of structure pairs to transform. Each entry pairs a
        source module (norm or linear) with the target linears whose inputs it
        feeds, plus the pair kind. If regex is used, modules match with the
        largest overlap in module name. If unset, mappings are inferred from
        the model architecture
    :param epochs: number of epochs over the calibration data per unit
    :param lr: base learning rate for the transform parameters (AdamW). Clip
        factors use 10x this value
    :param diag_alpha: interpolation between weight (0.0) and activation
        (1.0) magnitudes used to initialize the learnable diagonal
    :param add_diag: include a learnable diagonal in the transforms of
        norm_linear and mlp_linear_linear pairs (the v_proj -> o_proj pair
        never has one)
    :param diag_relu: keep SVD singular values positive via softplus
    :param tran_type: parameterization of the Kronecker factors, "svd"
        (orthogonal via Cayley) or "inv" (directly learnable invertible)
    :param warmup: prepend a short linear lr warmup to the cosine schedule
    :param deactive_amp: disable autocast during training (train in the
        model's native dtype instead)
    :param amp_dtype: dtype used by autocast during training
    :param seed: seed applied before transform construction, for
        reproducible orthogonal initializations
    :param w_bits: weight bit-width assumed by the fake weight quantizer
        used during training
    :param a_bits: activation bit-width assumed by the fake activation
        quantizer used during training
    :param w_sym: symmetric fake weight quantization during training
    :param a_sym: symmetric fake activation quantization during training
    :param lwc: learnable weight clipping during training
    :param lac: learnable activation clipping during training
    :param offload_device: device to which cached parent args are offloaded,
        which reduces memory requirements but requires more time to move
        data. Defaults to None (no offloading), or "cpu" for MoE models
    """

    requires_calibration_data: bool = True

    # Allow arbitrary types because ResolvedFlatQuantMapping has Module fields
    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    mappings: list[FlatQuantMapping] | None = None
    epochs: int = Field(default=15, ge=1)
    lr: float = Field(default=5e-3, gt=0, allow_inf_nan=False)
    diag_alpha: float = Field(default=0.3, ge=0, le=1)
    add_diag: bool = True
    diag_relu: bool = True
    tran_type: str = "svd"
    warmup: bool = False
    deactive_amp: bool = False
    amp_dtype: str = "bfloat16"
    seed: int = 0
    w_bits: int = 4
    a_bits: int = 4
    w_sym: bool = True
    a_sym: bool = True
    lwc: bool = True
    lac: bool = True
    offload_device: torch.device | None | Sentinel = Sentinel("not_provided")

    # Private vars set during calibration, cleared during finalization
    _resolved_mappings: list[ResolvedFlatQuantMapping] = PrivateAttr(
        default_factory=list
    )
    _training_units: list[TrainingUnit] = PrivateAttr(default_factory=list)
    # cache of forward args for each unit parent, one dict per batch
    _parent_args_cache: dict[Module, IntermediatesCache] = PrivateAttr(
        default_factory=dict
    )
    # per-target running max of |input|, used to initialize the diagonals
    _act_stats: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)

    @field_validator("w_bits", "a_bits")
    @classmethod
    def validate_bits(cls, value: int) -> int:
        if value not in (*range(2, 9), 16):
            raise ValueError("FlatQuant bit widths must be 2..8 or 16")
        return value

    @field_validator("amp_dtype")
    @classmethod
    def validate_amp_dtype(cls, value: str) -> str:
        if value not in ("bfloat16", "float16"):
            raise ValueError(f"amp_dtype must be 'bfloat16' or 'float16', got {value}")
        return value

    @field_validator("tran_type")
    @classmethod
    def validate_tran_type(cls, value: str) -> str:
        if value not in ("svd", "inv"):
            raise ValueError(f"tran_type must be 'svd' or 'inv', got {value}")
        return value

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Start FlatQuant on the given state. This runs before quantization
        config has been applied.

        - infer unresolved mappings based on model architecture, if not set
        - set default offload device

        :param state: state to run FlatQuant on
        :return: True on a successful run, False otherwise
        """
        if self.mappings is None:
            logger.info("No FlatQuantModifier.mappings provided, inferring from model")
            self.mappings = get_layer_mappings_from_model(state.model)

        if self.offload_device == Sentinel("not_provided"):
            if is_moe_model(state.model):
                self.offload_device = torch.device("cpu")
                logger.info(
                    "MoE model detected: setting offload_device to 'cpu' by default "
                    "to reduce memory usage. You can override this by explicitly "
                    "setting offload_device in your recipe."
                )
            else:
                self.offload_device = None

        return True

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        """
        Start FlatQuant calibration. This runs after quantization config has
        been applied.

        - resolve mappings and construct transforms
        - validate downstream quantization schemes
        - build training units and patch all involved modules
        - attach parent-args cache hooks and input statistics hooks
        """
        if is_distributed():
            raise NotImplementedError(
                "FlatQuantModifier does not support distributed execution: the "
                "transform parameters would be trained independently on each "
                "rank without gradient synchronization"
            )

        recipe = active_session().lifecycle.recipe
        if any(
            isinstance(modifier, QuantizationMixin)
            for modifier in getattr(recipe, "modifiers", [])
        ):
            raise ValueError(
                "FlatQuant requires pipeline='independent' so downstream observers "
                "and GPTQ Hessians see the trained input transforms"
            )

        torch.manual_seed(self.seed)

        self._resolve_mappings(state.model)
        self._validate_downstream_schemes()
        self._build_training_units()
        self._patch_units()
        self._setup_hooks()

    def on_sequential_epoch_end(self, state: State, event: Event, **kwargs):
        """
        Train and absorb the transforms of every unit whose parent module
        executed during this sequential pass. Units of other subgraphs are
        identified by their (still empty) args cache and skipped.
        """
        try:
            for unit in self._training_units:
                cache = self._parent_args_cache.get(unit.parent)
                if cache is not None and len(cache) > 0:
                    self._train_unit(state.model, unit)
        except Exception:
            self.remove_hooks()
            for unit in self._training_units:
                if unit.parent in self._parent_args_cache:
                    self._unpatch_unit(unit)
            raise

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibrating by verifying all cached data was consumed, falling
        any never-executed unit back to its original forward, and removing
        hooks.
        """
        try:
            for cache in self._parent_args_cache.values():
                if len(cache) > 0:
                    raise RuntimeError("Some cached activations were not used")
            for unit in self._training_units:
                if unit.parent in self._parent_args_cache:
                    self._unpatch_unit(unit)
        finally:
            for unit in self._training_units:
                if unit.parent in self._parent_args_cache:
                    self._unpatch_unit(unit)
            self.remove_hooks()

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up all runtime state. The absorbed weights and frozen transform
        buffers stay attached to the model and are persisted with its
        state_dict.

        :param state: unused
        :return: True
        """
        self.remove_hooks()
        self._parent_args_cache.clear()
        self._act_stats.clear()
        self._training_units.clear()
        self._resolved_mappings.clear()

        return True

    ###############################
    # Mapping and unit resolution #
    ###############################

    def _resolve_mappings(self, model: Module) -> None:
        """
        Resolve mappings against the model, constructing the transform of
        each pair. Mappings with incompatible shapes or untrainable
        dimensions are skipped with a warning (e.g. hidden sizes which admit
        no Kronecker decomposition).
        """
        module_to_name = get_module_to_name_dict(model)
        resolved: list[ResolvedFlatQuantMapping] = []
        num_skipped = 0

        for mapping in self.mappings:
            for source_modules, *nested_target_modules in match_modules_set(
                model, (mapping.source_layer, *mapping.target_layers)
            ):
                if len(source_modules) > 1:
                    raise ValueError(
                        "FlatQuant needs to match a single source module for each "
                        f"mapping but got {len(source_modules)} for mapping: "
                        f"{mapping}"
                    )
                source = source_modules[0]
                source_name = module_to_name[source]

                targets = tree_leaves(nested_target_modules)
                target_names = [module_to_name[target] for target in targets]

                if len(targets) == 0:
                    logger.warning(
                        f"Skipping {source_name} for mapping {mapping} because no "
                        "target layers were found."
                    )
                    continue

                if not any(
                    hasattr(module, "quantization_scheme") for module in targets
                ):
                    logger.warning(
                        f"Skipping {source_name} for mapping {mapping} because no "
                        "modules are targeted for quantization. FlatQuant must be "
                        "run with a subsequent quantization modifier."
                    )
                    continue

                trans = self._build_trans(model, mapping, source, targets)
                if trans is None:
                    num_skipped += 1
                    continue

                if mapping.pair_kind == NORM_LINEAR:
                    # The source remains unchanged, so ignored consumers need
                    # no compensation and must not receive fake quantization.
                    selected = [
                        (name, target)
                        for name, target in zip(target_names, targets)
                        if hasattr(target, "quantization_scheme")
                    ]
                    target_names = [name for name, _ in selected]
                    targets = [target for _, target in selected]

                ancestor_name, ancestor = get_lowest_common_ancestor_with_avoid(
                    [source_name, *target_names], model, torch.nn.ModuleList
                )

                resolved.append(
                    ResolvedFlatQuantMapping(
                        pair_kind=mapping.pair_kind,
                        source_name=source_name,
                        source_module=source,
                        target_names=target_names,
                        target_modules=targets,
                        parent=ancestor,
                        parent_name=ancestor_name,
                        trans=trans,
                    )
                )

        if num_skipped > 0:
            logger.warning(
                f"{num_skipped} mappings were skipped due to incompatible shapes "
                "or unsupported dimensions. Set logging to DEBUG to view the "
                "full list."
            )
        if len(resolved) == 0:
            raise ValueError(
                "No FlatQuant mappings were resolved against the model. Check "
                f"the mappings argument: {self.mappings}"
            )

        input_targets = set()
        output_sources = set()
        for mapping in resolved:
            for target in mapping.target_modules:
                if target in input_targets:
                    raise ValueError(
                        "Multiple FlatQuant input transforms target the same Linear"
                    )
                input_targets.add(target)
            if mapping.pair_kind == ATTN_LINEAR_LINEAR:
                if mapping.source_module in output_sources:
                    raise ValueError(
                        "Multiple FlatQuant output transforms target the same Linear"
                    )
                output_sources.add(mapping.source_module)
        self._resolved_mappings = resolved

    def _build_trans(
        self, model: Module, mapping: FlatQuantMapping, source: Module, targets: list
    ) -> GeneralMatrixTrans | None:
        """
        Construct the GeneralMatrixTrans for a resolved structure pair,
        validating shapes. Returns None (with a debug log) if the pair is not
        trainable, e.g. because a dimension admits no Kronecker decomposition.
        """
        for target in targets:
            if not isinstance(target, torch.nn.Linear):
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: target "
                    f"{type(target)} is not a Linear"
                )
                return None

        if mapping.pair_kind == NORM_LINEAR:
            if source.weight.dim() != 1:
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: source {type(source)} "
                    "does not have a 1D weight"
                )
                return None
            if any(target.in_features != source.weight.numel() for target in targets):
                raise ValueError(
                    f"FlatQuant norm mapping has mismatched input dimensions: {mapping}"
                )
            dims = get_decompose_dim(source.weight.shape[0])
            if dims is None:
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: hidden size "
                    f"{source.weight.shape[0]} admits no Kronecker decomposition "
                    "(== 2 mod 4)"
                )
                return None
            return GeneralMatrixTrans(
                *dims,
                add_diag=self.add_diag,
                diag_relu=self.diag_relu,
                tran_type=self.tran_type,
            )

        if mapping.pair_kind == ATTN_LINEAR_LINEAR:
            if not isinstance(source, torch.nn.Linear):
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: source "
                    f"{type(source)} is not a Linear"
                )
                return None
            if any(target.in_features != targets[0].in_features for target in targets):
                raise ValueError("Attention targets must have the same input dimension")
            # MSMODELSLIM-COMPAT: a^2 - b^2 decomposition, matching
            # vllm-ascend's native solve_kronecker_decompose so the exported
            # left_trans / right_trans shapes match the parameters the native
            # scheme pre-allocates.  The head_dim split differs for models
            # where it is not the minimal a^2 - b^2 solution (e.g. 4096).
            dims = get_decompose_dim(targets[0].in_features)
            if dims is None:
                logger.debug(
                    f"Skipping FlatQuant mapping {mapping}: attention input "
                    f"{targets[0].in_features} admits no Kronecker decomposition"
                )
                return None
            return GeneralMatrixTrans(
                *dims,
                add_diag=False,
                diag_relu=self.diag_relu,
                tran_type=self.tran_type,
            )

        # MLP_LINEAR_LINEAR
        if not isinstance(source, torch.nn.Linear):
            logger.debug(
                f"Skipping FlatQuant mapping {mapping}: source {type(source)} "
                "is not a Linear"
            )
            return None
        if any(target.in_features != source.out_features for target in targets):
            raise ValueError(
                f"FlatQuant MLP mapping has mismatched dimensions: {mapping}"
            )
        dims = get_decompose_dim(source.weight.shape[0])
        if dims is None:
            logger.debug(
                f"Skipping FlatQuant mapping {mapping}: source output size "
                f"{source.weight.shape[0]} admits no Kronecker decomposition "
                "(== 2 mod 4)"
            )
            return None
        return GeneralMatrixTrans(
            *dims,
            add_diag=self.add_diag,
            diag_relu=self.diag_relu,
            tran_type=self.tran_type,
        )

    def _head_dim(self, model: Module, source: Module, target: Module) -> int | None:
        """
        Resolve the attention head dimension for a v_proj -> o_proj pair,
        following the msmodelslim blueprint: the model config's head_dim if
        present, else hidden_size // num_attention_heads.
        """
        config = getattr(model, "config", None)
        config = getattr(config, "text_config", config)
        if getattr(config, "attn_output_gate", False) or getattr(
            config, "output_gate", False
        ):
            raise ValueError(
                "FlatQuant v_proj -> o_proj cannot cross attention output gating. "
                "Omit this pair for Qwen3.5/Qwen3.6 gated attention."
            )
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(config, "hidden_size", None)
            num_heads = getattr(config, "num_attention_heads", None)
            if hidden_size is not None and num_heads:
                head_dim = hidden_size // num_heads

        if head_dim is None:
            logger.debug(
                "Skipping FlatQuant v_proj -> o_proj mapping: could not resolve "
                "head_dim from the model config"
            )
            return None
        if source.out_features % head_dim or target.in_features % head_dim:
            logger.debug(
                "Skipping FlatQuant v_proj -> o_proj mapping: head_dim "
                f"{head_dim} does not divide v_proj out_features "
                f"{source.out_features} / o_proj in_features {target.in_features}"
            )
            return None
        return head_dim

    def _validate_downstream_schemes(self) -> None:
        """Reject unsupported numerics before patching or modifying parameters."""
        checked = set()
        for mapping in self._resolved_mappings:
            for module in (mapping.source_module, *mapping.target_modules):
                if module in checked or not isinstance(module, torch.nn.Linear):
                    continue
                checked.add(module)
                scheme = getattr(module, "quantization_scheme", None)
                if scheme is None:
                    continue
                weights = scheme.weights
                activations = scheme.input_activations
                if weights is not None and (
                    weights.type != "int"
                    or weights.strategy != "channel"
                    or weights.dynamic is not False
                    or weights.num_bits != self.w_bits
                    or weights.symmetric != self.w_sym
                    or weights.scale_dtype is not None
                ):
                    raise ValueError(
                        "FlatQuant training requires matching per-channel static INT "
                        f"weights (w_bits={self.w_bits}, w_sym={self.w_sym})"
                    )
                if activations is not None and (
                    activations.type != "int"
                    or activations.strategy != "token"
                    or activations.dynamic is not True
                    or activations.num_bits != self.a_bits
                    or activations.symmetric != self.a_sym
                    or activations.scale_dtype is not None
                ):
                    raise ValueError(
                        "FlatQuant training requires matching dynamic per-token INT "
                        f"activations (a_bits={self.a_bits}, a_sym={self.a_sym})"
                    )
                if scheme.output_activations is not None:
                    raise ValueError(
                        "FlatQuant output activation quantization is unsupported"
                    )

    def _build_training_units(self) -> None:
        """
        Group resolved mappings into training units: first by the lowest
        common ancestor of all involved modules, then merging groups whose
        ancestors are nested into the outermost ancestor (e.g. the
        self_attn group of a v_proj -> o_proj pair merges into the decoder
        layer group which also holds the norm -> q/k/v pair), yielding one
        unit per transformer layer as in the msmodelslim blueprint.
        """
        units: dict[Module, TrainingUnit] = {}
        for mapping in self._resolved_mappings:
            unit = units.setdefault(
                mapping.parent, TrainingUnit(mapping.parent, mapping.parent_name)
            )
            unit.mappings.append(mapping)

        # merge units whose parents are ancestors of one another, repeating
        # until no parent is nested inside another unit's parent
        merged = True
        while merged:
            merged = False
            for inner_parent, inner_unit in units.items():
                for outer_parent, outer_unit in units.items():
                    if outer_parent is inner_parent:
                        continue
                    if _is_ancestor_name(
                        outer_unit.parent_name, inner_unit.parent_name
                    ):
                        outer_unit.mappings.extend(inner_unit.mappings)
                        del units[inner_parent]
                        merged = True
                        break
                if merged:
                    break

        self._training_units = list(units.values())
        logger.info(
            f"FlatQuant resolved {len(self._resolved_mappings)} mappings into "
            f"{len(self._training_units)} training units"
        )

    ################
    # Patching and #
    # hook set-up  #
    ################

    def _patch_units(self) -> None:
        """
        Patch target linears and supported attention sources, register
        per-target fake quantizers (sharing one lac clip factor per pair) and
        the transform homes, then freeze all parameters inside the units
        (trainable parameters are re-enabled at training time).
        """
        for unit in self._training_units:
            for mapping in unit.mappings:
                trans = mapping.trans
                if mapping.pair_kind == NORM_LINEAR:
                    # Transform each consumer, not the norm output. Other
                    # consumers (routers, residuals, gated attention branches)
                    # stay unchanged, including after temporary norm restore.
                    for index, target in enumerate(mapping.target_modules):
                        self._patch_linear(
                            target, act_in_trans=trans, weight_in_trans=trans
                        )
                        if index == 0:
                            register_flatquant_trans(target, trans)
                elif mapping.pair_kind == ATTN_LINEAR_LINEAR:
                    # the right factor is applied to the v_proj output (and
                    # absorbed into its weight), the left factor stays online
                    # at the o_proj input, and the full transform is absorbed
                    # into the o_proj weight
                    self._patch_linear(
                        mapping.source_module, weight_out_trans=trans.right_trans
                    )
                    for target in mapping.target_modules:
                        self._patch_linear(
                            target,
                            act_in_trans=trans.left_trans,
                            weight_in_trans=trans,
                        )
                        register_flatquant_trans(target, trans)
                else:  # MLP_LINEAR_LINEAR
                    for target in mapping.target_modules:
                        self._patch_linear(
                            target, act_in_trans=trans, weight_in_trans=trans
                        )
                        register_flatquant_trans(target, trans)

                if self.lac:
                    # one shared clip factor per pair (the blueprint
                    # overrides the quantizer's default 4.0 init with 1.0)
                    clip_factor = Parameter(torch.ones(1) * 1.0, requires_grad=True)
                    for target in mapping.target_modules:
                        state = getattr(target, FLATQUANT_STATE_ATTR)
                        if state.act_quantizer.lac:
                            state.act_quantizer.clip_factor = clip_factor

            # freeze all parameters inside the unit; the transforms are
            # re-enabled by name at training time
            for param in unit.parent.parameters():
                param.requires_grad = False

            unit.modules = get_patched_modules(unit.parent)

    def _patch_linear(self, module: Module, **transforms) -> FlatQuantState:
        """
        Patch a linear module, or extend the FlatQuantState of an already
        patched one (e.g. v_proj is patched as a norm-pair target first and
        gains a weight output transform from the v_proj -> o_proj pair).
        """
        state = getattr(module, FLATQUANT_STATE_ATTR, None)
        if state is None:
            scheme = getattr(module, "quantization_scheme", None)
            weights = getattr(scheme, "weights", None)
            activations = getattr(scheme, "input_activations", None)
            state = patch_flatquant_linear(
                module,
                act_quantizer=ActivationQuantizer(
                    self.a_bits if activations is not None else 16,
                    sym=self.a_sym,
                    lac=self.lac and activations is not None,
                ),
                weight_quantizer=WeightQuantizer(
                    in_size=module.weight.shape[1],
                    out_size=module.weight.shape[0],
                    bits=self.w_bits if weights is not None else 16,
                    sym=self.w_sym,
                    lwc=self.lwc and weights is not None,
                ),
                **transforms,
            )
        else:
            for attr, trans in transforms.items():
                if trans is not None:
                    setattr(state, attr, trans)
        return state

    def _setup_hooks(self) -> None:
        """
        Attach a forward_pre hook on each unit parent caching the kwargs
        needed to replay its forward, and a forward hook on each target linear
        accumulating per-input-channel |input| maxima (pre-initialized to
        1e-5, as in the blueprint).
        """

        def cache_parent_kwargs_hook(
            module: Module,
            args: tuple,
            kwargs: dict,
        ):
            # bind() nests **kwargs contents under a 'kwargs' key; flatten
            # them so the result can be replayed as module(**cached)
            arguments = (
                inspect.signature(module.forward).bind(*args, **kwargs).arguments
            )
            arguments |= arguments.pop("kwargs", {})

            from compressed_tensors.modeling.kvcache import QuantizedKVCache

            for key, value in arguments.items():
                if isinstance(value, QuantizedKVCache):
                    arguments[key] = None

            # setdefault for robustness: the cache entry is deleted after a
            # unit is trained (to free memory), and no forward of its parent
            # is expected afterwards; if one happens anyway, it re-creates
            # the entry and on_calibration_end raises a descriptive error
            # instead of a KeyError mid-forward
            self._parent_args_cache.setdefault(
                module, IntermediatesCache(None, self.offload_device)
            ).append(arguments)

        def make_stat_hook(name: str):
            def stat_input_hook(_module: Module, args: tuple, _output):
                stats = self._act_stats.get(name)
                if stats is None:
                    # unit was already trained and its stats consumed
                    return
                x = args[0] if isinstance(args, tuple) and args else None
                if not isinstance(x, torch.Tensor):
                    return
                x_max = x.reshape(-1, x.shape[-1]).abs().amax(dim=0)
                torch.maximum(stats, x_max.to(stats.dtype).cpu(), out=stats)

            return stat_input_hook

        for unit in self._training_units:
            if unit.parent not in self._parent_args_cache:
                self._parent_args_cache[unit.parent] = IntermediatesCache(
                    None, self.offload_device
                )
                self.register_hook(
                    unit.parent,
                    cache_parent_kwargs_hook,
                    "forward_pre",
                    with_kwargs=True,
                )

            for mapping in unit.mappings:
                for name, target in zip(mapping.target_names, mapping.target_modules):
                    self._act_stats[name] = torch.full(
                        [target.weight.shape[1]], 1e-5, dtype=target.weight.dtype
                    )
                    self.register_hook(target, make_stat_hook(name), "forward")

    ##################
    # Training units #
    ##################

    def _train_unit(self, model: Module, unit: TrainingUnit) -> None:
        """
        Train all transforms of one unit against fp teacher outputs, then
        absorb them into the model weights.

        1. replay the cached parent args in org mode to extract fp teacher
            outputs (falling back to the original forward if the data is
            empty or non-finite)
        2. initialize the learnable diagonals from weight and activation maxima
        3. train the transforms jointly: one optimizer over all transform and
            clip-factor parameters of the unit, one step per calibration
            sample, MSE loss normalized by its (detached) value
        4. absorb weights on CPU in FP64, retain the diagonal online, then
            replace training patches with native serializable input hooks
        """
        parent = unit.parent
        cache = self._parent_args_cache[parent]

        # pin memory for faster onloading during replay
        if get_execution_device(parent).type != "cpu":
            for batch_index in range(len(cache)):
                cache.pin_memory(batch_index)

        involved = [parent]
        for mapping in unit.mappings:
            involved.append(mapping.source_module)
            involved.extend(mapping.target_modules)

        with (
            align_modules(involved),
            HooksMixin.disable_hooks(),
        ):
            # 1. fp teacher outputs
            set_flatquant_mode(unit.modules, MODE_ORG)
            with calibration_forward_context(model):
                fp_outs = self._run_samples(parent)
            if len(fp_outs) == 0 or all(f.numel() == 0 for f in fp_outs):
                logger.info(
                    f"Skipping FlatQuant unit {unit.parent_name}: no activations "
                    "found to transform. This can occasionally occur in MoE models "
                    "when no calibration data reaches the unit."
                )
                self._unpatch_unit(unit)
                return
            if not all(torch.isfinite(f).all() for f in fp_outs):
                logger.warning(
                    f"Skipping FlatQuant unit {unit.parent_name}: NaN or inf "
                    "outputs found in the fp teacher forward pass. The model may "
                    "be generating NaN outputs with the provided calibration data, "
                    "or the mappings are set incorrectly."
                )
                self._unpatch_unit(unit)
                return
            fp_outs = [f.cpu() for f in fp_outs]

            # 2. diagonal initialization
            self._init_diag_scales(unit)

            # 3. training
            set_flatquant_mode(unit.modules, MODE_CALIB)
            try:
                self._train_loop(unit, fp_outs)
            except Exception:
                self.remove_hooks()
                self._unpatch_unit(unit)
                raise

            # DIAG-CHECK: print diagonal stats after training, before absorption
            for mapping in unit.mappings:
                trans = mapping.trans
                if trans.diag_trans is not None and trans.diag_trans.diag_scale is not None:
                    d = trans.diag_trans.diag_scale.data
                    logger.info(
                        f"DIAG-CHECK {unit.parent_name}: "
                        f"min={d.min():.6f} max={d.max():.6f} "
                        f"mean={d.mean():.6f} extreme={(d.abs() > 20).sum().item()}"
                    )

            # 4. absorption
            with torch.no_grad():
                self._absorb_unit(unit)
            set_flatquant_mode(unit.modules, MODE_EVAL)
            self._export_unit(model, unit)

        # remove caches and stats belonging to this unit
        del self._parent_args_cache[parent]
        for mapping in unit.mappings:
            for name in mapping.target_names:
                self._act_stats.pop(name, None)

    def _run_samples(self, module: Module) -> list[torch.Tensor]:
        """
        Replay the cached forward args of a module, one forward pass per
        batch, returning the first output of each pass.
        """
        cache = self._parent_args_cache[module]
        batch_iter = cache.iter_prefetch()
        outputs = []
        for batch in batch_iter:
            output = module(**batch)
            output = output[0] if isinstance(output, tuple) else output
            outputs.append(output.detach().cpu())
        return outputs

    def _init_diag_scales(self, unit: TrainingUnit) -> None:
        """
        Initialize the learnable diagonal of each pair transform from the
        per-input-channel maxima of the target weights and of the target
        input activations. Unlike the blueprint (which leaks a loop variable
        and uses only the last target's statistics), the activation maxima of
        all targets of the pair are combined elementwise.
        """
        for mapping in unit.mappings:
            trans = mapping.trans
            if trans.diag_trans is None:
                continue

            weight_max = (
                torch.cat(
                    [target.weight.data.float() for target in mapping.target_modules],
                    dim=0,
                )
                .abs()
                .max(dim=0)[0]
            )
            input_max = torch.stack(
                [self._act_stats[name].float() for name in mapping.target_names],
                dim=0,
            ).max(dim=0)[0]

            diag_scale = trans.diag_trans.diag_scale
            diag_scale.data = get_init_scale(
                weight_max.to(diag_scale), input_max.to(diag_scale), self.diag_alpha
            )

    def _train_loop(self, unit: TrainingUnit, fp_outs: list[torch.Tensor]) -> None:
        """
        Jointly train all transforms of a unit: an AdamW optimizer over the
        transform parameters (base lr) and clip factors (10x lr) of the unit,
        with a cosine annealing schedule over epochs * num_samples steps (and
        an optional linear warmup), one step per calibration sample, under
        autocast. The MSE loss against the fp teacher output is normalized by
        its detached value, so the gradient direction is scale-invariant.
        """
        param_groups = self._get_trainable_parameters(unit.parent, self.lr)
        if not param_groups:
            logger.debug(
                f"FlatQuant unit {unit.parent_name} has no trainable parameters, "
                "skipping training"
            )
            return

        optimizer = torch.optim.AdamW(param_groups)
        num_samples = len(fp_outs)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs * num_samples, eta_min=self.lr * 1e-3
        )
        if self.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=16
            )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [scheduler_warmup, scheduler_main]
            )
        else:
            scheduler = scheduler_main

        loss_fn = torch.nn.MSELoss()
        device = get_execution_device(unit.parent)
        cache = self._parent_args_cache[unit.parent]
        amp_dtype = torch.bfloat16 if self.amp_dtype == "bfloat16" else torch.float16

        pbar = tqdm(
            total=self.epochs * num_samples, desc=f"FlatQuant {unit.parent_name}"
        )
        # the sequential pipeline invokes this hook inside a no_grad context
        with torch.enable_grad(), pbar:
            for _epoch in range(self.epochs):
                epoch_mse = 0.0
                batch_iter = cache.iter_prefetch()
                for sample_idx, batch in enumerate(batch_iter):
                    with torch.amp.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=not self.deactive_amp,
                    ):
                        output = unit.parent(**batch)
                        quant_output = (
                            output[0] if isinstance(output, tuple) else output
                        )
                        loss = loss_fn(
                            quant_output.float(),
                            fp_outs[sample_idx].to(quant_output.device).float(),
                        )

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite FlatQuant loss in {unit.parent_name}"
                        )
                    epoch_mse += loss.detach().float().item()
                    if loss.detach().item() == 0:
                        pbar.update(1)
                        continue
                    loss = loss / loss.detach().clamp_min(torch.finfo(loss.dtype).tiny)
                    optimizer.zero_grad()
                    loss.backward()
                    if any(
                        param.grad is not None and not torch.isfinite(param.grad).all()
                        for group in param_groups
                        for param in group["params"]
                    ):
                        raise RuntimeError(
                            f"Non-finite FlatQuant gradients in {unit.parent_name}"
                        )
                    # V6: gradient clipping removed — fp32 params (23-bit
                    # mantissa) allow diagonal to freely adjust from any
                    # value, so early-step spikes self-correct.
                    optimizer.step()
                    scheduler.step()
                    pbar.update(1)

                pbar.set_postfix({"mse": f"{epoch_mse / max(num_samples, 1):.3e}"})
                logger.debug(
                    f"FlatQuant unit {unit.parent_name} epoch {_epoch}: "
                    f"MSE loss {epoch_mse:.6e}, lr {scheduler.get_last_lr()[0]:.8f}"
                )

    def _get_trainable_parameters(self, parent: Module, base_lr: float) -> list[dict]:
        """
        Collect the transform and clip-factor parameters within a unit,
        grouped by name for the optimizer (clip factors train at 10x lr),
        mirroring msmodelslim's get_trainable_parameters. Parameters are
        deduplicated: the lac clip factor is shared across the targets of a
        pair (registered under each target's act quantizer), and passing the
        same parameter to the optimizer multiple times would step it once per
        occurrence.
        """
        params: dict[str, list[Parameter]] = {key: [] for key in TRAINABLE_KEYS}
        seen: set[int] = set()
        owned = set()
        for module in get_patched_modules(parent):
            state = getattr(module, FLATQUANT_STATE_ATTR)
            for part in (
                state.act_in_trans,
                state.weight_in_trans,
                state.weight_out_trans,
                state.act_quantizer,
                state.weight_quantizer,
            ):
                if part is not None:
                    owned.update(id(parameter) for parameter in part.parameters())
        for name, param in parent.named_parameters():
            if id(param) not in owned:
                continue
            for key in TRAINABLE_KEYS:
                if key in name:
                    param.requires_grad = True
                    if id(param) not in seen:
                        seen.add(id(param))
                        params[key].append(param)
                    break

        # Diagonal trains at the same lr as Kronecker factors. The clamp in
        # DiagonalTransMatrix.forward (max=10.0) and gradient clipping
        # (max_norm=1.0) prevent runaway values without slowing convergence.
        param_groups = [
            {"params": params[key], "lr": base_lr}
            for key in BASE_LR_KEYS
            if params[key]
        ]
        if params["clip_factor"]:
            param_groups.append({"params": params["clip_factor"], "lr": base_lr * 10})

        return param_groups

    ##############
    # Absorption #
    ##############

    def _absorb_unit(self, unit: TrainingUnit) -> None:
        """Freeze transforms and absorb target weights on CPU in FP64.

        The diagonal stays in the input transform. Source norms and MLP
        projections are untouched; no offset-gamma restore/writeback is needed.
        Learned weight clipping is intentionally lossy when enabled.
        """
        # MSMODELSLIM-COMPAT: absorb the learnable diagonal into the source
        # module so the exported checkpoint carries only left/right/clip_ratio.
        self._absorb_diagonal(unit)

        for mapping in unit.mappings:
            mapping.trans.to_eval_mode()
            mapping.trans.to(device="cpu", dtype=torch.float64)

        for module in unit.modules:
            if isinstance(module, torch.nn.Linear):
                self._absorb_linear(module)

        # MSMODELSLIM-COMPAT: defer diagonal removal — the target weights have
        # now absorbed the full (D R L) transform, so D must not be applied
        # online nor exported.
        for mapping in unit.mappings:
            if getattr(mapping.trans, "diag_trans", None) is not None:
                mapping.trans.diag_trans = None

        for module in unit.modules:
            state = getattr(module, FLATQUANT_STATE_ATTR)
            if state.act_quantizer is not None:
                state.act_quantizer.enable = False
            if state.weight_quantizer is not None:
                state.weight_quantizer.enable = False
                state.weight_quantizer.enable_find = False

    def _absorb_diagonal(self, unit: TrainingUnit) -> None:
        """MSMODELSLIM-COMPAT: fold diag_scale into the source module.

        llm-compressor keeps the diagonal online at the target input.  The
        native vllm-ascend FlatQuant scheme has no diagonal parameter, so the
        scaling must be pushed upstream:

          norm  : (x_norm * g) * d == x_norm * (g * d)   -> norm.weight *= d
          linear: (x @ W^T) * d   == x @ (W * d)^T      -> weight *= d[:, None]

        The diagonal is then detached from the transform so it is neither
        applied online nor written to the checkpoint.
        """
        for mapping in unit.mappings:
            trans = mapping.trans
            diag = getattr(trans, "diag_trans", None)
            if diag is None or diag.diag_scale is None:
                continue
            d = diag.diag_scale.detach().to(torch.float64)
            source = mapping.source_module
            weight = source.weight.detach()
            original_device = weight.device
            original_dtype = weight.dtype
            w = weight.to(device="cpu", dtype=torch.float64)
            if w.dim() == 1:
                if w.numel() != d.numel():
                    raise RuntimeError(
                        f"diagonal size {d.numel()} != norm weight {w.numel()} "
                        f"for {mapping}"
                    )
                w = w * d
            elif w.dim() == 2:
                if w.shape[0] != d.numel():
                    raise RuntimeError(
                        f"diagonal size {d.numel()} != linear out_features "
                        f"{w.shape[0]} for {mapping}"
                    )
                w = w * d.unsqueeze(1)
            else:
                raise RuntimeError(f"unsupported source weight dim {w.dim()}")
            if not torch.isfinite(w).all():
                raise RuntimeError("non-finite source weight after diagonal absorb")
            update_offload_parameter(
                source,
                "weight",
                w.to(dtype=original_dtype).to(original_device),
            )
            # MSMODELSLIM-COMPAT: defer diagonal removal until AFTER the
            # target weights absorb the full (D R L) transform.  Dropping it
            # here would lose the D^{-T} factor and corrupt the weights.

    def _absorb_linear(self, module: Module) -> None:
        """
        Absorb the transforms attached to a patched linear into its weight
        (and bias), in fp64 as in the blueprint, then reparameterize its fake
        quantizers. Idempotent: modules with no weight transforms left (or
        none to begin with) only reparameterize.
        """
        state = getattr(module, FLATQUANT_STATE_ATTR)
        weight_in_trans = state.weight_in_trans
        weight_out_trans = state.weight_out_trans

        if weight_in_trans is not None or weight_out_trans is not None:
            weight = module.weight.detach()
            ori_dtype = weight.dtype
            original_device = weight.device
            weight = weight.to(device="cpu", dtype=torch.float64)

            if weight_in_trans is not None:
                weight = weight_in_trans(weight, inv_t=True)
            if state.weight_quantizer.lwc:
                weight = state.weight_quantizer.apply_wclip(weight)
            if weight_out_trans is not None:
                weight = weight_out_trans(weight.T).T
                if module.bias is not None:
                    # the blueprint absorbs the bias in its original dtype
                    update_offload_parameter(
                        module,
                        "bias",
                        weight_out_trans(
                            module.bias.detach().to(device="cpu", dtype=torch.float64)
                        ).to(device=module.bias.device, dtype=module.bias.dtype),
                    )

            if not torch.isfinite(weight).all():
                raise RuntimeError("Non-finite FlatQuant weights after absorption")
            converted = weight.to(dtype=ori_dtype)
            if not torch.isfinite(converted).all():
                raise RuntimeError("FlatQuant weights overflowed the model dtype")
            update_offload_parameter(module, "weight", converted.to(original_device))
            state.weight_in_trans = None
            state.weight_out_trans = None

        state.act_quantizer.reparameterize()
        state.weight_quantizer.reparameterize()

    def _export_unit(self, model: Module, unit: TrainingUnit) -> None:
        """Replace training patches with native, serializable input hooks.

        Only INPUT applications are exported: the checkpoint already contains
        transformed weights, so reloading must never reapply weight absorption.
        """
        previous = getattr(model, TRANSFORM_CONFIG_NAME, None)
        names = set(previous.config_groups) if previous is not None else set()
        groups = {}
        payloads = []
        module_names = get_module_to_name_dict(model)
        for module in unit.modules:
            state = getattr(module, FLATQUANT_STATE_ATTR)
            trans = state.act_in_trans
            quantizer = state.act_quantizer
            clipped = quantizer is not None and quantizer.lac and quantizer.bits < 16
            if trans is None and not clipped:
                continue
            key = find_unique_name("flatquant", names)
            names.add(key)
            if trans is None:
                left, right = get_decompose_dim(module.in_features) or (
                    1,
                    module.in_features,
                )
                left_matrix, right_matrix = torch.eye(left), torch.eye(right)
                diagonal = torch.ones(module.in_features)
            elif isinstance(trans, GeneralMatrixTrans):
                left_matrix = trans.left_trans.get_matrix().detach()
                right_matrix = trans.right_trans.get_matrix().detach()
                diagonal = (
                    trans.diag_trans.diag_scale.detach()
                    if trans.diag_trans is not None
                    else torch.ones(module.in_features)
                )
            else:
                # The across-head factor of a supported v_proj -> o_proj pair.
                left_matrix = trans.get_matrix().detach()
                right_matrix = torch.eye(module.in_features // trans.size)
                diagonal = torch.ones(module.in_features)
            ratio = quantizer.get_clip_ratio().detach() if clipped else torch.ones(1)
            groups[key] = TransformScheme(
                type="flatquant" if quantizer.sym else "flatquant-asymmetric",
                head_dim=right_matrix.shape[0],
                precision=torch.float32,
                apply=[TransformArgs(targets=[module_names[module]], location="input")],
            )
            payloads.append(
                (
                    module,
                    f"{key}_input",
                    {
                        "left": left_matrix,
                        "right": right_matrix,
                        "clip_ratio": ratio,
                    },
                )
            )

        for module in unit.modules:
            unpatch_flatquant_module(module)
        config = TransformConfig(config_groups=groups)
        apply_transform_config(model, config)
        for module, name, parameters in payloads:
            runtime = getattr(module, name)
            assert isinstance(runtime, FlatQuantInputTransform)
            for parameter, value in parameters.items():
                update_offload_parameter(
                    runtime, parameter, value.to(dtype=torch.float32)
                )
        if previous is not None:
            previous.merge(config)
            setattr(model, TRANSFORM_CONFIG_NAME, previous)

    def _unpatch_unit(self, unit: TrainingUnit) -> None:
        """
        Restore the original forward of all modules of a unit which was
        skipped (no data, or non-finite teacher outputs), removing the
        transforms and fake quantizers so the modules keep their original
        semantics and state_dict.
        """
        for module in get_patched_modules(unit.parent):
            unpatch_flatquant_module(module)
        self._parent_args_cache.pop(unit.parent, None)
        for mapping in unit.mappings:
            for name in mapping.target_names:
                self._act_stats.pop(name, None)
        logger.info(f"Fell FlatQuant unit {unit.parent_name} back to original forward")