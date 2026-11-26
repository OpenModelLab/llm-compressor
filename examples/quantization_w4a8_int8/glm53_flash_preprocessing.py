"""Example-local GLM53 adapters for native QuaRot and FlexSmooth arithmetic.

GLM53 backbone geometry; MTP requires a separate topology adapter.
"""

import math
from dataclasses import dataclass
from time import perf_counter

import torch
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import (
    OffloadCache,
    align_module_device,
    get_execution_device,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr
from torch import nn

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.transform.flex_smooth import FlexSmoothModifier
from llmcompressor.modifiers.transform.flex_smooth.mappings import SmoothMapping
from llmcompressor.modifiers.transform.flex_smooth.search import (
    _row_int8,
    _search_candidates,
    search_alpha_beta,
    smooth_scale,
)
from llmcompressor.modifiers.transform.quarot.mappings import (
    NormFusion,
    RotationPlan,
    RotationSpace,
    WeightRotation,
)
from llmcompressor.modifiers.transform.quarot.rotation import (
    make_hadamard_rotation,
    rotate_axis,
)


@dataclass(frozen=True)
class ParameterRef:
    owner: nn.Module
    key: str


def parameter_ref(model, path):
    """Resolve a normal weight or an actual bare Parameter without wrapping it."""
    try:
        module = model.get_submodule(path)
    except AttributeError:
        stem, _, key = path.rpartition(".")
        module = model.get_submodule(stem)
    else:
        key = "weight"
    if key not in module._parameters:
        raise ValueError(f"Missing GLM53 transform parameter: {path}")
    value = getattr(module, key)
    if value is None or not value.is_floating_point():
        raise ValueError(f"Expected floating parameter: {path}")
    if value.is_meta and not isinstance(module._parameters, OffloadCache):
        raise ValueError(f"Unbacked meta parameter: {path}")
    return ParameterRef(module, key)


def build_glm53_plans(model):
    """Validate the full backbone before any norm/rotation parameter is changed."""
    if is_distributed():
        raise ValueError("This GLM53 example adapter is single-process")
    config = model.config.get_text_config()
    if getattr(config, "num_nextn_predict_layers", 0):
        raise ValueError("GLM53 preprocessing requires MTP explicitly disabled")
    if config.qk_rope_head_dim != 0:
        raise ValueError("This GLM53 adapter requires the legal NoPE geometry")
    prefix = "model.language_model"
    layers = model.get_submodule(f"{prefix}.layers")
    if len(layers) != config.num_hidden_layers:
        raise ValueError("Unexpected decoder count")
    hidden, query, kv = config.hidden_size, config.q_lora_rank, config.kv_lora_rank
    value, key, heads = (
        config.v_head_dim,
        config.qk_nope_head_dim,
        config.num_attention_heads,
    )
    spaces = (
        RotationSpace("rot", hidden),
        RotationSpace("rot_b_proj", query, shifted=True),
        RotationSpace("rot_uv", value),
        RotationSpace("rot_kv_b_proj", kv),
    )
    sizes = {space.name: space.size for space in spaces}
    rotations, fusions, smoothing = [], [], []

    def weight(path, inputs=None, outputs=None):
        ref = parameter_ref(model, path)
        tensor = getattr(ref.owner, ref.key)
        if tensor.ndim != 2 or tensor.layout != torch.strided:
            raise ValueError(f"Expected dense 2D projection: {path}")
        if (inputs is not None and tensor.shape[1] != inputs) or (
            outputs is not None and tensor.shape[0] != outputs
        ):
            raise ValueError(f"Incompatible GLM53 projection geometry: {path}")
        return tensor

    def rotate(path, space, axis, stride=None, offset=0):
        tensor = weight(path)
        size = sizes[space]
        stride = size if stride is None else stride
        if tensor.shape[axis] % stride or offset + size > stride:
            raise ValueError(f"Invalid rotation segment: {path}")
        rotations.append(WeightRotation(path, space, axis, stride, offset))

    def fuse(path, consumers, width, *, smooth=False):
        norm = model.get_submodule(path)
        if type(norm).__name__ != "Glm5NextTextRMSNorm":
            raise ValueError(f"Expected ordinary GLM53 RMSNorm: {path}")
        if norm.weight.shape != (width,) or getattr(norm, "bias", None) is not None:
            raise ValueError(f"Incompatible norm: {path}")
        for consumer in consumers:
            weight(consumer, inputs=width)
        fusions.append(NormFusion(path, tuple(consumers)))
        if smooth:
            smoothing.append(SmoothMapping("norm-linear", path, tuple(consumers)))

    embedding = f"{prefix}.embed_tokens"
    weight(embedding, inputs=hidden)
    weight("lm_head", inputs=hidden)
    rotate("lm_head", "rot", 1)
    for index, layer in enumerate(layers):
        base = f"{prefix}.layers.{index}"
        attention = f"{base}.self_attn"
        if layer.block_type == "linear_attention":
            inputs = [
                f"{attention}.{name}"
                for name in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "b_proj",
                    "forget_gate.f_a_proj",
                    "g_a_proj",
                )
            ]
        elif layer.block_type == "indexed_attention":
            qa, qb = f"{attention}.q_a_proj", f"{attention}.q_b_proj"
            kva, kvb = f"{attention}.kv_a_proj_with_mqa", f"{attention}.kv_b_proj"
            weight(qa, hidden, query)
            weight(qb, query, heads * key)
            weight(kva, hidden, kv)
            weight(kvb, kv, heads * (key + value))
            weight(f"{attention}.o_proj", heads * value, hidden)
            inputs, query_consumers = [qa, kva], [qb]
            if layer.self_attn.indexer is not None:
                inputs += [
                    f"{attention}.indexer.wk",
                    f"{attention}.indexer.weights_proj",
                    f"{attention}.indexer.index_kpool_compress_gate",
                ]
                query_consumers.append(f"{attention}.indexer.wq_b")
            fuse(f"{attention}.q_a_layernorm", query_consumers, query, smooth=True)
            fuse(f"{attention}.kv_a_layernorm", [kvb], kv)
            rotate(qa, "rot_b_proj", 0)
            for name in query_consumers:
                rotate(name, "rot_b_proj", 1)
            rotate(kva, "rot_kv_b_proj", 0)
            rotate(kvb, "rot_kv_b_proj", 1)
            rotate(kvb, "rot_uv", 0, key + value, key)
            rotate(f"{attention}.o_proj", "rot_uv", 1)
        else:
            raise ValueError(f"Unsupported GLM53 attention: {layer.block_type}")
        fuse(f"{base}.input_layernorm", inputs, hidden, smooth=True)
        for name in inputs:
            rotate(name, "rot", 1)
        weight(f"{attention}.o_proj", outputs=hidden)
        rotate(f"{attention}.o_proj", "rot", 0)
        for connection in ("attn_hc", "ffn_hc"):
            weight(f"{base}.{connection}.fn", inputs=hidden * config.hc_mult)
            rotate(f"{base}.{connection}.fn", "rot", 1)
        mlp = f"{base}.mlp"
        if config.mlp_layer_types[index] == "dense":
            experts = [mlp]
            consumers = []
        elif config.mlp_layer_types[index] == "sparse":
            experts = [f"{mlp}.experts.{i}" for i in range(config.n_routed_experts)]
            experts.append(f"{mlp}.shared_experts")
            consumers = [f"{mlp}.gate"]
        else:
            raise ValueError("Unsupported MLP type")
        for expert in experts:
            consumers += [f"{expert}.gate_proj", f"{expert}.up_proj"]
            weight(f"{expert}.down_proj", outputs=hidden)
            rotate(f"{expert}.down_proj", "rot", 0)
        fuse(f"{base}.post_attention_layernorm", consumers, hidden)
        for name in consumers:
            rotate(name, "rot", 1)
    fuse(f"{prefix}.norm", ["lm_head"], hidden)
    # The visual tower stays unquantized. Its final output must nevertheless
    # enter the same residual basis as the rotated text embeddings.
    merger = "model.visual.merger.down_proj"
    weight(merger, outputs=hidden)
    rotate(merger, "rot", 0)
    # Keep reference phase order, including low-precision writeback between
    # spaces: hidden -> query latent -> per-head value -> KV latent.
    order = {space.name: index for index, space in enumerate(spaces)}
    rotations.sort(key=lambda operation: order[operation.space])
    smoothing.sort(
        key=lambda item: (
            int(item.source.split(".layers.")[1].split(".")[0]),
            not item.source.endswith(".input_layernorm"),
        )
    )
    plan = RotationPlan(
        spaces,
        tuple(fusions),
        WeightRotation(embedding, "rot", 1, hidden),
        tuple(rotations),
    )
    targets = {op.target for op in (plan.embedding, *plan.rotations)}
    targets.update(f.norm for f in fusions)
    identities = {}
    for name in sorted(targets):
        ref = parameter_ref(model, name)
        tensor = getattr(ref.owner, ref.key)
        identity = id(tensor)
        if identity in identities:
            raise ValueError(f"Untie transform targets before loading: {name}")
        identities[identity] = name
        if getattr(ref.owner, "quantization_scheme", None) is not None:
            raise ValueError("Run preprocessing before quantization")
    return plan, tuple(smoothing)


@torch.no_grad()
def _multiply_parameter(model, path, factor):
    ref = parameter_ref(model, path)
    with align_module_device(ref.owner):
        tensor = getattr(ref.owner, ref.key)
        updated = (tensor.float() * factor.to(tensor.device).float()).to(tensor.dtype)
        if not torch.isfinite(updated).all():
            raise ValueError(f"Nonfinite transformed parameter: {path}")
        update_offload_parameter(ref.owner, ref.key, updated)


@torch.no_grad()
def streamed_flex_search(cached, weight, chunk_tokens):
    """All-token native grid/proxy with bounded token-by-output temporaries.

    Global moments follow native distributed-search dtype handling. Reduction
    order can differ from the full-tensor reference; this is not bit-exact.
    """
    count = sum(values.shape[0] for values in cached)
    if count < 1 or chunk_tokens < 1:
        raise ValueError("FlexSmooth needs tokens and a positive chunk size")

    def chunks():
        for values in cached:
            for block in values.split(chunk_tokens):
                yield block.to(device=weight.device, dtype=weight.dtype)

    amax = weight.new_zeros(weight.shape[1])
    for block in chunks():
        amax = torch.maximum(amax, block.abs().amax(0))
    if not torch.isfinite(amax).all() or not torch.isfinite(weight).all():
        raise ValueError("Nonfinite FlexSmooth search input")
    if count <= chunk_tokens:
        return search_alpha_beta(torch.cat(tuple(chunks())), weight), amax
    wmax = weight.abs().amax(0)
    accumulation = torch.float64 if weight.dtype == torch.float64 else torch.float32
    numel = count * weight.shape[0]

    def evaluate(alpha, beta):
        scale = amax.pow(alpha) * wmax.pow(-beta)
        quantized_weight = _row_int8(weight * scale)
        moments = torch.zeros(2, device=weight.device, dtype=accumulation)
        for block in chunks():
            golden = block @ weight.T
            reconstructed = _row_int8(block / scale) @ quantized_weight.T
            moments[0] += (
                (reconstructed - golden).abs().square().to(accumulation) / numel
            ).sum()
            moments[1] += (golden.square().to(accumulation) / numel).sum()
        means = moments.to(weight.dtype)
        return (means[0].sqrt() / means[1].sqrt()).item()

    return _search_candidates(evaluate), amax


class Glm53QuaRotModifier(Modifier):
    """Native structured rotations with an example-local GLM53 backbone plan."""

    _plan: RotationPlan | None = PrivateAttr(default=None)
    _matrices: dict = PrivateAttr(default_factory=dict)
    _model: object = PrivateAttr(default=None)
    _applied: bool = PrivateAttr(default=False)
    strict_recipe: bool = False

    @property
    def matrices(self):
        return self._matrices

    def on_initialize(self, state, **kwargs):
        model = state.model
        if getattr(model.config, "quarot_config", None) is not None:
            raise ValueError("Start from original, untransformed BF16 weights")
        with OffloadCache.disable_onloading():
            self._plan, _ = build_glm53_plans(model)
        # Construct every basis before mutation. Native full-dimensional
        # Hadamard and shifted 32-block construction match the imported kernel.
        self._matrices = {
            space.name: make_hadamard_rotation(
                space.size, shifted=space.shifted, seed=1234, dtype=torch.float32
            )
            for space in self._plan.spaces
        }
        self._model = model
        return True

    @torch.no_grad()
    def on_calibration_start(self, state, event, **kwargs):
        model = state.model
        if model is not self._model:
            raise ValueError("QuaRot lifecycle model changed")
        if self._applied:
            return
        metadata = dict(
            status="in_progress",
            topology="glm53_backbone",
            precision="float32",
            seed=1234,
            block_size=None,
            online=False,
            mtp="disabled",
        )
        model.config.quarot_config = metadata
        started = perf_counter()
        try:
            for fusion in self._plan.fusions:
                norm = model.get_submodule(fusion.norm)
                with align_module_device(norm):
                    gain = norm.weight.detach().float().cpu().clone()
                for name in fusion.consumers:
                    _multiply_parameter(model, name, gain)
                update_offload_parameter(norm, "weight", torch.ones_like(gain))
            operations = (self._plan.embedding, *self._plan.rotations)
            for index, op in enumerate(operations, 1):
                ref = parameter_ref(model, op.target)
                with align_module_device(ref.owner):
                    options = dict(
                        axis=op.axis,
                        stride=op.stride,
                        offset=op.offset,
                        precision=torch.float32,
                    )
                    matrix = self._matrices[op.space]
                    if self.strict_recipe:
                        matrix = matrix.to_dense()
                    updated = rotate_axis(
                        getattr(ref.owner, ref.key), matrix, **options
                    )
                    update_offload_parameter(ref.owner, ref.key, updated)
                    if op.axis == 0 and getattr(ref.owner, "bias", None) is not None:
                        updated = rotate_axis(ref.owner.bias, matrix, **options)
                        update_offload_parameter(ref.owner, "bias", updated)
                if index % 100 == 0 or index == len(operations):
                    logger.info(
                        "GLM53 QuaRot {}/{} ({:.1f}s)",
                        index,
                        len(operations),
                        perf_counter() - started,
                    )
        except Exception:
            metadata["status"] = "failed"
            raise
        metadata["status"] = "applied"
        self._applied = True

    def on_finalize(self, state, **kwargs):
        self._model = None
        if not self._applied:
            raise ValueError("QuaRot did not complete")
        return True


class Glm53FlexSmoothModifier(FlexSmoothModifier):
    """Reuse native collection/search with GLM53 norm-linear consumers."""

    subgraphs: tuple = ("norm-linear",)
    loss_chunk_tokens: int = Field(default=4096, strict=True, gt=0)
    strict_recipe: bool = False

    def _preflight(self, model):
        if self.subgraphs != ("norm-linear",) or self.max_tokens is not None:
            raise ValueError(
                "Reference GLM53 mode requires norm-linear, uncapped tokens"
            )
        if self.alpha is not None or self.beta is not None:
            raise ValueError("Reference GLM53 mode searches both alpha and beta")
        with OffloadCache.disable_onloading():
            return build_glm53_plans(model)[1]

    @torch.no_grad()
    def _smooth(self, model, item):
        first = parameter_ref(model, item.consumers[0])
        device = get_execution_device(first.owner)
        weights = []
        for name in item.consumers:
            ref = parameter_ref(model, name)
            with align_module_device(ref.owner, device):
                weights.append(getattr(ref.owner, ref.key).detach().clone())
        weight = torch.cat(weights)
        del weights
        if self.strict_recipe:
            # Match the reference's full cat/GEMM/mean, including BF16 rounding.
            # Do not silently fall back to chunked moments on OOM.
            activation = torch.cat(
                [
                    value.to(device=weight.device, dtype=weight.dtype)
                    for value in self._cache[item.source]
                ]
            )
            result = search_alpha_beta(activation, weight)
            activation_max = activation.abs().amax(0)
            del activation
        else:
            result, activation_max = streamed_flex_search(
                self._cache[item.source], weight, self.loss_chunk_tokens
            )
        scale = smooth_scale(
            activation_max, weight.abs().amax(0), result.alpha, result.beta
        )
        # The bare kpool gate is updated at its original parameter key. No
        # temporary Linear survives into forward execution or serialization.
        for name in item.consumers:
            _multiply_parameter(model, name, scale)
        _multiply_parameter(model, item.source, scale.reciprocal())
        details = dict(
            alpha=result.alpha,
            beta=result.beta,
            loss=result.loss,
            fallback=None,
            tokens_seen=self._seen[item.source],
            tokens_used=self._used[item.source],
            loss_chunk_tokens=None if self.strict_recipe else self.loss_chunk_tokens,
            search_execution="full-tensor" if self.strict_recipe else "streamed",
            alpha_losses=list(result.alpha_losses),
            beta_losses=list(result.beta_losses),
        )
        model.config.flex_smooth_config["results"][item.source] = {
            key: value
            for key, value in details.items()
            if key not in ("alpha_losses", "beta_losses")
        }
        model.config.flex_smooth_config["results"][item.source][
            "nonfinite_candidates"
        ] = sum(
            not math.isfinite(value)
            for value in (
                *result.alpha_losses,
                *result.beta_losses,
            )
        )
        self._diagnostics[item.source] = {**details, "scale": scale.detach().cpu()}
        self._processed.add(item.source)
        self._cache.pop(item.source, None)
        logger.info(
            "GLM53 FlexSmooth {} alpha={} beta={} tokens={}",
            item.source,
            result.alpha,
            result.beta,
            details["tokens_used"],
        )

    def on_sequential_epoch_end(self, state, event, modules=None, **kwargs):
        model = state.model
        self._check_model(model)
        if not self._collecting:
            return
        members = set(model.modules() if modules is None else modules)
        try:
            for item in self._plan:
                if item.source in self._processed or item.source not in self._cache:
                    continue
                participants = {
                    parameter_ref(model, path).owner for path in item.targets
                }
                if not participants & members:
                    continue
                if not participants <= members:
                    raise ValueError(
                        f"Sequential partition splits GLM53 norm: {item.source}"
                    )
                self._smooth(model, item)
        except Exception:
            self._fail(model)
            raise
