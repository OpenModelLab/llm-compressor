# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compute static HiFloat8 (HiF8) DSA scales for GLM-5.3-Flash (sparse MLA).

The DSA (sparse attention) path on Ascend 950 quantizes *two* per-tensor HiF8
latents, both exported as a static FLOAT32 per-tensor scale by ModelSlim's
``QK_HIF8_PER_TENSOR`` recipe (``fa_q`` / ``fa_k``):

* ``fa_k.scale`` -- the 512-dim NoPE KV latent (``kv_a_layernorm`` output,
  ``kv_lora_rank`` channels), stored as the DSA KV cache.
* ``fa_q.scale`` -- the 1536-dim absorbed query latent (``q_a_layernorm``
  output, ``q_lora_rank`` channels), quantized on the fly before the
  ``aclnnQuantSparseFlashMla`` operator.

The underlying HiF8 quantize operator only supports a per-tensor scale (no
per-tile/per-group), so a single FLOAT32 scale governs each whole latent row.

Convention (matches ``msmodelslim.ir.api.impl.hif8_quantization`` and the
vllm-ascend ``AscendHIF8KVCacheAttentionMethod``): the checkpoint stores the
*descale* value,

    scale = absmax / HIF8_MAX   (HIF8_MAX = 32768), FLOAT32, per tensor.

At inference vllm-ascend reciprocates it and casts on the descale domain:

    q = cast_hiF8(x * (1 / scale))   dequant = q * scale.

This module only computes and writes those two FLOAT32 scales; the HiF8
element cast itself (BF16 -> HiFloat8) happens on the inference side.

Two entry points:

* :class:`DsaHif8ScaleCollector` -- attaches forward hooks to each DSA layer's
  ``q_a_layernorm`` / ``kv_a_layernorm`` (the RMSNorms applied to the two
  latents right before they are consumed) so the scales are accumulated
  *during* an existing calibration forward pass (the llm-compressor
  ``oneshot`` pipeline). This avoids a second full forward pass and is
  memory-safe because it reuses the sequential pipeline's subgraph offloading.

* :func:`compute_dsa_hif8_kv_scales` -- a standalone helper that runs its own
  forward pass over a calibration dataloader and returns the scales dict. Useful
  for a data-dependent workflow outside ``oneshot`` (``model_free_ptq`` is
  data-free and cannot produce activation scales).

MTP draft layers (checkpoint ``layers.N`` with ``N >= num_hidden_layers``,
e.g. GLM-5.3-Flash's ``layers.45``) are never forwarded by the calibration
pipelines -- the draft lives outside every traced subgraph, so its hooks stay
silent. msmodelslim instead drives the draft through its layer-wise pass with a
synthesized input (``enorm(embed(ids))`` + ``hnorm(backbone final hidden)``
-> ``eh_proj``). :class:`MtpCalibrationCapture` +
:func:`replay_mtp_draft_calibration` mirror that here: the backbone pass is
captured, then the draft's DSA latent path is replayed over the captured
inputs while the collector hooks are still attached, so the draft layer
receives ``fa_q.scale`` / ``fa_k.scale`` like every backbone DSA layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from compressed_tensors.offload.module import (
    subgraph_offload_modules,
    subgraph_onload_modules,
)
from loguru import logger
from torch.fx._symbolic_trace import is_fx_tracing
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm

__all__ = [
    "DsaHif8ScaleCollector",
    "compute_dsa_hif8_kv_scales",
    "MtpCalibrationCapture",
    "attach_mtp_calibration_capture",
    "replay_mtp_draft_calibration",
    "has_dsa_layers",
    "DSA_HIF8_MAX",
    "DSA_KV_NOPE_DIM",
    "DSA_Q_NOPE_DIM",
    "FA_Q_SCALE",
    "FA_K_SCALE",
]

# HiF8 (HiFloat8) largest finite magnitude (msmodelslim ``HIF8_MAX``). A static
# scale maps the observed absmax onto this value so the full HiF8 dynamic range
# is used instead of collapsing everything into [0, 1].
DSA_HIF8_MAX = 32768.0

# NoPE latent dimensions of the two HiF8-quantized tensors (GLM-5.3-Flash).
# ``kv_a_layernorm`` normalizes the leading ``kv_lora_rank`` NoPE KV channels;
# ``q_a_layernorm`` normalizes the ``q_lora_rank`` absorbed query channels.
DSA_KV_NOPE_DIM = 512
DSA_Q_NOPE_DIM = 1536

# Submodule names registered on each DSA attention module. Each is a plain
# ``nn.Module`` holding a single FLOAT32 ``scale`` buffer, so ``save_pretrained``
# serializes ``<attention_prefix>.fa_q.scale`` / ``<attention_prefix>.fa_k.scale``
# -- exactly the names vllm-ascend's ``AscendHIF8KVCacheAttentionMethod`` loads.
FA_Q_SCALE = "fa_q"
FA_K_SCALE = "fa_k"
SCALE_BUFFER = "scale"

# The MLA projection whose output feeds the cached DSA NoPE KV latent. The KV
# latent this module ultimately produces is the *RMSNorm-ed* NoPE slice (see
# ``kv_a_layernorm`` below), so the projection is only the layer anchor used to
# locate the layernorms whose outputs are actually measured.
_DSA_LATENT_PROJECTION_SUFFIXES = ("kv_a_proj_with_mqa", "kv_a_proj")

# The RMSNorm applied to the KV NoPE slice right before it is cached. The cached
# (and therefore HiF8-quantized) KV latent is the *layernorm output*.
_DSA_KV_LAYERNORM_SUFFIX = "kv_a_layernorm"

# The RMSNorm applied to the absorbed query latent right before it is used by
# the sparse-MLA operator. The HiF8-quantized query latent is its output.
_DSA_Q_LAYERNORM_SUFFIX = "q_a_layernorm"


def _layer_prefix(module_name: str) -> str:
    """Return the attention prefix for a hooked MLA projection.

    ``model.layers.5.self_attn.kv_a_proj_with_mqa`` ->
    ``model.layers.5.self_attn``.
    """
    for suffix in _DSA_LATENT_PROJECTION_SUFFIXES:
        if module_name.endswith(suffix):
            return module_name[: -len(suffix)].rstrip(".")
    return module_name


def _register_scale_submodule(attention_module: Module, name: str, scale: torch.Tensor) -> None:
    """Register ``scale`` as ``<attention_module>.<name>.scale`` (a buffer).

    Keeps the scales as persistent FLOAT32 *buffers* (not parameters) so the
    weight-quantization lifecycle (``session.finalize``) leaves them untouched.
    Registering them as nested ``nn.Module`` submodules makes ``save_pretrained``
    emit the ``fa_q.scale`` / ``fa_k.scale`` tensor names.
    """
    holder = getattr(attention_module, name, None)
    if not isinstance(holder, Module):
        holder = Module()
        attention_module.add_module(name, holder)
    if SCALE_BUFFER in holder._buffers:
        # Re-registering over a previous run: update the offloaded buffer.
        holder._buffers[SCALE_BUFFER] = scale
    else:
        holder.register_buffer(SCALE_BUFFER, scale)


class DsaHif8ScaleCollector:
    """Accumulate per-tensor HiF8 DSA Q/K scales during calibration forwards.

    Locates every DSA layer via its ``kv_a_proj_with_mqa`` (or ``kv_a_proj``)
    projection, then hooks the sibling ``q_a_layernorm`` and ``kv_a_layernorm``
    *outputs* and reduces a running per-tensor absmax for each. The cached KV
    latent and the absorbed query latent are the RMSNorm-ed slices of their
    projections, so the scales must match those domains exactly; measuring the
    raw projection outputs would produce scales in the wrong magnitude range.
    It is meant to be attached before the llm-compressor calibration pipeline
    runs and read after it completes, so the scales are collected without an
    extra forward pass. Forward hooks are plain ``register_forward_hook`` hooks
    (not ``HooksMixin`` hooks), so they are guarded against FX tracing and
    meta/proxy tensors rather than being disabled by ``HooksMixin.disable_hooks``.
    """

    def __init__(
        self,
        model: Module,
        kv_nope_dim: int = DSA_KV_NOPE_DIM,
        q_nope_dim: int = DSA_Q_NOPE_DIM,
        hook_suffixes: tuple[str, ...] = _DSA_LATENT_PROJECTION_SUFFIXES,
    ):
        self.model = model
        self.kv_nope_dim = kv_nope_dim
        self.q_nope_dim = q_nope_dim
        self.hook_suffixes = hook_suffixes

        # (attention_prefix, attention_module, q_source, kv_source) --
        # ``q_source`` is the ``q_a_layernorm`` whose output is measured (or
        # ``None`` when the model has no absorbed query); ``kv_source`` is the
        # ``kv_a_layernorm`` whose output is measured, or the projection itself
        # when no explicit layernorm exists.
        self._targets: list[tuple[str, Module, Module | None, Module]] = []
        self._handles: list[RemovableHandle] = []
        # ``{attention_prefix: {"q": absmax | None, "k": absmax}}`` over all
        # forward calls, each scalar float32.
        self._accum: dict[str, dict[str, torch.Tensor | None]] = {}
        self._find_targets()

    @property
    def has_dsa_layers(self) -> bool:
        """Whether any sparse-MLA latent projection was found."""
        return bool(self._targets)

    def _find_targets(self) -> None:
        seen: set[str] = set()
        for name, module in self.model.named_modules():
            if not name.endswith(self.hook_suffixes):
                continue

            prefix = _layer_prefix(name)
            if prefix == name or prefix in seen:
                continue
            seen.add(prefix)

            attention_module = self.model.get_submodule(prefix) if prefix else self.model
            # The KV latent is the RMSNorm-ed slice of the projection output
            # (``kv = kv_a_layernorm(kv_a_proj_with_mqa(x)[..., :kv_lora_rank])``),
            # so the per-tensor absmax must be measured on the *layernorm
            # output*. Hook the layernorm when it exists; otherwise fall back to
            # the projection (a model without an explicit KV layernorm caches the
            # raw latent, where the projection output is already the right domain).
            kv_layernorm = getattr(attention_module, _DSA_KV_LAYERNORM_SUFFIX, None)
            kv_source = kv_layernorm if isinstance(kv_layernorm, Module) else module
            # The absorbed query latent (``q_a_layernorm`` output) has no
            # projection fallback: a model without it does not quantize Q.
            q_layernorm = getattr(attention_module, _DSA_Q_LAYERNORM_SUFFIX, None)
            q_source = q_layernorm if isinstance(q_layernorm, Module) else None
            self._targets.append((prefix, attention_module, q_source, kv_source))

    def _make_hook(self, prefix: str, kind: str, nope_dim: int):
        def hook(_module, _inputs, output):
            # Skip FX tracing (proxy outputs) and meta/placeholder tensors.
            if is_fx_tracing():
                return
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor) or output.is_meta:
                return

            # The hooked module is a layernorm whose output *is* the latent that
            # gets HiF8-quantized (already [..., nope_dim]). Slicing [..., :nope_dim]
            # is a no-op there and only trims the RoPE tail in the fallback case
            # (a hooked ``kv_a_proj_with_mqa`` emitting [..., nope_dim + rope_dim]).
            if output.ndim < 2 or output.shape[-1] < nope_dim:
                logger.warning(
                    "DSA {} latent output dim {} is smaller than "
                    "nope_dim {}; skipping scale accumulation for {}.",
                    kind.upper(),
                    output.shape[-1],
                    nope_dim,
                    prefix,
                )
                return

            latent = output.detach()[..., :nope_dim]  # [..., nope_dim]
            # Per-tensor absmax over all observed tokens of this layer. Reduce in
            # float32 so the scale (absmax / 32768) is not rounded to the 8-bit
            # bf16 mantissa before it is stored.
            absmax = latent.to(torch.float32).abs().amax()

            slot = self._accum.setdefault(prefix, {"q": None, "k": None})
            prev = slot[kind]
            slot[kind] = absmax if prev is None else torch.maximum(prev, absmax)

        return hook

    def attach(self) -> "DsaHif8ScaleCollector":
        """Register forward hooks on every DSA layer's measured source modules."""
        if self._handles or not self._targets:
            return self

        for prefix, _attention, q_source, kv_source in self._targets:
            self._handles.append(
                kv_source.register_forward_hook(self._make_hook(prefix, "k", self.kv_nope_dim))
            )
            if q_source is not None:
                self._handles.append(
                    q_source.register_forward_hook(self._make_hook(prefix, "q", self.q_nope_dim))
                )

        logger.info(
            "Computing HiF8 DSA Q/K scales: hooked {} layers "
            "(q_dim={}, kv_dim={}, per-tensor)",
            len(self._targets),
            self.q_nope_dim,
            self.kv_nope_dim,
        )
        return self

    def detach(self) -> "DsaHif8ScaleCollector":
        """Remove all registered forward hooks (safe to call repeatedly)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return self

    def scales(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return ``{attention_prefix: {"q": tensor, "k": tensor}}`` on CPU.

        Each scale is ``absmax / DSA_HIF8_MAX``, the static per-tensor *descale*
        factor the inference side reciprocates when casting to HiFloat8. The
        ``"q"`` entry is omitted when the model has no absorbed query latent.
        """
        tiny = torch.finfo(torch.float32).tiny
        scales: dict[str, dict[str, torch.Tensor]] = {}
        for prefix, slot in self._accum.items():
            entry: dict[str, torch.Tensor] = {}
            # atleast_1d keeps the checkpoint tensor 1-D (shape (1,)) so the
            # vllm-ascend loader's ``flatten()`` path stays uniform.
            if slot["k"] is not None:
                entry["k"] = torch.atleast_1d(
                    (slot["k"] / DSA_HIF8_MAX).clamp_min(tiny)
                ).cpu().contiguous()
            if slot["q"] is not None:
                entry["q"] = torch.atleast_1d(
                    (slot["q"] / DSA_HIF8_MAX).clamp_min(tiny)
                ).cpu().contiguous()
            scales[prefix] = entry
        return scales

    def has_scales_for(self, prefixes: Iterable[str]) -> bool:
        """Whether Q/K statistics were accumulated for any of the prefixes.

        Used after :func:`replay_mtp_draft_calibration` to verify the draft
        layer's hooks actually fired (its layernorms ran and were observed),
        rather than silently leaving the draft checkpoint without scales.
        """
        return any(
            slot.get("k") is not None or slot.get("q") is not None
            for prefix in prefixes
            if (slot := self._accum.get(prefix)) is not None
        )

    def write_scales(self) -> dict[str, dict[str, torch.Tensor]]:
        """Register each scale as a persistent FLOAT32 ``fa_q.scale``/``fa_k.scale``.

        The scales are registered as nested ``nn.Module`` submodules on the
        attention module owning each latent projection, so ``save_pretrained``
        serializes them as ``<attention_prefix>.fa_q.scale`` and
        ``<attention_prefix>.fa_k.scale``. Calling this before ``session.finalize``
        keeps the plain FLOAT32 buffers untouched by the weight-quantization
        lifecycle.

        :return: the same mapping as :meth:`scales`.
        """
        scales = self.scales()
        if not scales:
            logger.warning(
                "No HiF8 DSA Q/K scale was collected (no calibration forward pass "
                "observed). Use `compute_dsa_kv_scale=True` with a data-dependent "
                "pipeline (not `datafree`)."
            )
            return scales

        attention_by_prefix = {prefix: attn for prefix, attn, _q, _k in self._targets}
        for prefix, entry in scales.items():
            attention_module = attention_by_prefix[prefix]
            if "k" in entry:
                _register_scale_submodule(attention_module, FA_K_SCALE, entry["k"])
            if "q" in entry:
                _register_scale_submodule(attention_module, FA_Q_SCALE, entry["q"])

        logger.info("Wrote HiF8 DSA Q/K scales for {} layers", len(scales))
        return scales

    def __enter__(self) -> "DsaHif8ScaleCollector":
        return self.attach()

    def __exit__(self, *_exc) -> None:
        self.detach()


def has_dsa_layers(model: Module) -> bool:
    """Return whether ``model`` contains a sparse-MLA (DSA) latent projection."""
    return any(
        name.endswith(_DSA_LATENT_PROJECTION_SUFFIXES)
        for name, _ in model.named_modules()
    )


def compute_dsa_hif8_kv_scales(
    model: torch.nn.Module,
    calibration_dataloader: Iterable[Mapping],
    kv_nope_dim: int = DSA_KV_NOPE_DIM,
    q_nope_dim: int = DSA_Q_NOPE_DIM,
    hook_suffixes: tuple[str, ...] = _DSA_LATENT_PROJECTION_SUFFIXES,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute the static per-tensor FLOAT32 Q/K scales for each DSA layer.

    Runs its own forward pass over the calibration dataloader. Prefer
    :class:`DsaHif8ScaleCollector` when a calibration forward pass is already
    happening (e.g. inside ``oneshot``).

    :param model: the GLM-5.3 model with sparse-MLA (DSA) layers.
    :param calibration_dataloader: iterable of input batches accepted by
        ``model(**batch)`` (dict/``Mapping``) or ``model(*batch)`` (tuple/list).
        Only the forward pass is run; gradients are off.
    :param kv_nope_dim: NoPE KV latent dimension (``kv_lora_rank``, 512 for GLM-5.3).
    :param q_nope_dim: NoPE query latent dimension (``q_lora_rank``, 1536 for GLM-5.3).
    :param hook_suffixes: projection names to treat as the KV latent source.
    :return: mapping ``{attention_prefix: {"q": tensor, "k": tensor}}`` where
        each scale is ``absmax / DSA_HIF8_MAX`` (per-tensor, FLOAT32).
    """
    collector = DsaHif8ScaleCollector(
        model,
        kv_nope_dim=kv_nope_dim,
        q_nope_dim=q_nope_dim,
        hook_suffixes=hook_suffixes,
    )
    if not collector.has_dsa_layers:
        raise ValueError(
            "No DSA latent projection found. The model must expose "
            f"{', '.join(hook_suffixes)} submodules (GLM-5.3 sparse-MLA/DSA layers)."
        )

    collector.attach()
    model.eval()
    try:
        with torch.no_grad():
            for batch in calibration_dataloader:
                if isinstance(batch, Mapping):
                    model(**batch)
                elif isinstance(batch, (tuple, list)):
                    model(*batch)
                else:
                    model(batch)
    finally:
        collector.detach()

    scales = collector.scales()
    logger.info("Computed HiF8 DSA Q/K scales for {} layers", len(scales))
    return scales


# ---------------------------------------------------------------------------
# MTP draft calibration (msmodelslim parity)
# ---------------------------------------------------------------------------
#
# The calibration pipelines execute only the traced backbone forward, so an
# attached MTP draft layer (``model.mtp``, checkpoint ``layers.N`` with
# ``N >= num_hidden_layers`` -- GLM-5.3-Flash's ``layers.45``) never runs and
# its DSA hooks stay silent: the checkpoint gets fa scales for every backbone
# DSA layer but none for the draft. msmodelslim instead drives the draft
# during its layer-wise pass with a synthesized input built from the backbone
# output (``mtp_preprocess``: ``eh_proj(cat(enorm(embed(ids_shifted)),
# hnorm(backbone_hidden)))``) and collects draft scales like any other layer.
#
# ``MtpCalibrationCapture`` + ``replay_mtp_draft_calibration`` reproduce that
# within ``oneshot``: the backbone pass is captured, then the draft's DSA
# latent path is replayed over the captured inputs while the collector hooks
# are still attached. Only the path up to ``q_a_layernorm`` /
# ``kv_a_layernorm`` is replayed -- both latents are pure functions of the
# draft block input, so the replayed tensors equal a full draft forward's.


def _is_mtp_subtree(name: str) -> bool:
    """Whether a module name belongs to the attached MTP draft (``mtp.``)."""
    return name.split(".", 1)[0] == "mtp"


def _find_text_final_norm(model: Module) -> Module | None:
    """Locate the text decoder's final norm (the module feeding ``lm_head``).

    The final norm is the ``norm``-named module whose parent owns the decoder
    ``layers`` stack, excluding the MTP draft subtree and the vision tower.
    The final norm's *input* is the pre-norm hidden state the draft's ``hnorm``
    consumes (msmodelslim's ``mtp_preprocess`` applies the norm only to compute
    the next-token prediction, not for the draft input).
    """
    for name, module in model.named_modules():
        if not name or _is_mtp_subtree(name):
            continue
        if "visual" in name or "vision" in name:
            continue
        if not name.endswith(".norm"):
            continue
        parent_path = name[: -len(".norm")]
        parent = model.get_submodule(parent_path) if parent_path else model
        if hasattr(parent, "layers"):
            return module
    return None


def _find_lm_head(model: Module) -> Module | None:
    """Locate the language-model head (a ``lm_head``-named module)."""
    for name, module in model.named_modules():
        if not name or _is_mtp_subtree(name):
            continue
        if "visual" in name or "vision" in name:
            continue
        if name == "lm_head" or name.endswith(".lm_head"):
            return module
    return None


def _module_name(model: Module, module: Module) -> str | None:
    """Return ``module``'s qualified name in ``model`` (identity match)."""
    for name, candidate in model.named_modules():
        if candidate is module:
            return name
    return None


class MtpCalibrationCapture:
    """Capture the MTP draft layer's inputs during the backbone calibration.

    An MTP draft layer is never executed by the calibration pipelines (it sits
    outside every traced subgraph), so its DSA Q/K scales cannot be observed
    in place. This recorder stores, for every calibration batch, the two
    tensors the draft consumes at inference:

    * ``input_ids`` -- captured by a pre-hook on the (shared) ``embed_tokens``;
    * the backbone's pre-final-norm hidden states -- captured by a pre-hook on
      the text model's final norm.

    Hooks are plain ``register_forward_pre_hook`` hooks guarded against FX
    tracing and meta/proxy tensors, matching :class:`DsaHif8ScaleCollector`.
    Captured tensors are moved to CPU so NPU memory stays flat; the replay
    moves each batch back on demand. Both hooks fire once per forward, so the
    two lists stay index-aligned even across pipeline retries (duplicated
    batches only repeat absmax observations, which is harmless).
    """

    def __init__(
        self,
        model: Module,
        mtp_module: Module,
        embed_module: Module,
        final_norm: Module,
    ):
        self.model = model
        self.mtp_module = mtp_module
        self.embed_module = embed_module
        self.final_norm = final_norm
        self.input_ids: list[torch.Tensor] = []
        self.hidden_states: list[torch.Tensor] = []
        self._handles: list[RemovableHandle] = []

    def _make_ids_hook(self):
        def hook(_module, args):
            if is_fx_tracing() or not args:
                return
            ids = args[0]
            if not isinstance(ids, torch.Tensor) or ids.is_meta:
                return
            self.input_ids.append(ids.detach().to("cpu"))

        return hook

    def _make_hidden_hook(self):
        def hook(_module, args):
            if is_fx_tracing() or not args:
                return
            hidden = args[0]
            if not isinstance(hidden, torch.Tensor) or hidden.is_meta:
                return
            # A hyper-connection backbone emits 4D hidden (batch, seq, streams,
            # hidden); the draft consumes the stream mean, exactly like
            # msmodelslim's ``mtp_preprocess``.
            if hidden.dim() == 4:
                hidden = hidden.mean(dim=2)
            self.hidden_states.append(hidden.detach().to("cpu"))

        return hook

    def attach(self) -> "MtpCalibrationCapture":
        """Register the capture hooks (idempotent)."""
        if self._handles:
            return self
        self._handles.append(
            self.embed_module.register_forward_pre_hook(self._make_ids_hook())
        )
        self._handles.append(
            self.final_norm.register_forward_pre_hook(self._make_hidden_hook())
        )
        logger.info(
            "Capturing MTP draft inputs from the backbone calibration pass "
            "(embed_tokens and final-norm pre-hooks)"
        )
        return self

    def detach(self) -> "MtpCalibrationCapture":
        """Remove the capture hooks (safe to call repeatedly)."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return self

    def __enter__(self) -> "MtpCalibrationCapture":
        return self.attach()

    def __exit__(self, *_exc) -> None:
        self.detach()


def attach_mtp_calibration_capture(model: Module) -> MtpCalibrationCapture | None:
    """Attach input capture for the MTP draft layer, when one is loaded.

    Returns ``None`` (with a log) when the model carries no MTP draft or the
    required anchors cannot be located, so callers fail open: the backbone
    scales are still produced, only the draft scales are skipped.
    """
    mtp_module = getattr(model, "mtp", None)
    if not isinstance(mtp_module, Module) or not getattr(mtp_module, "layers", None):
        logger.debug("No MTP draft model attached; skipping draft KV-scale capture.")
        return None

    final_norm = _find_text_final_norm(model)
    if final_norm is None:
        logger.warning(
            "MTP draft present but the text model's final norm was not found; "
            "the draft layer will not receive fa_q/fa_k scales."
        )
        return None

    # Hook the BACKBONE's embedding, not ``mtp.embed_tokens``: the MTP model
    # may hold a value-copied embedding that is never executed during the
    # backbone calibration pass, so a hook on it would stay silent and the
    # replay would be skipped. The backbone's own embedding is the module the
    # calibration forward actually calls, and it carries identical weights
    # (embed_tokens is excluded from both smoothing and quantization), so
    # replaying through it is equivalent.
    embed_module = (
        model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    )
    if not isinstance(embed_module, Module):
        embed_module = getattr(mtp_module, "embed_tokens", None)
    if not isinstance(embed_module, Module):
        logger.warning(
            "MTP draft present but no shared embedding was found; "
            "the draft layer will not receive fa_q/fa_k scales."
        )
        return None

    capture = MtpCalibrationCapture(model, mtp_module, embed_module, final_norm)
    return capture.attach()


@dataclass
class _DraftReplayTarget:
    """Modules of one MTP draft layer needed to rebuild its DSA latents."""

    prefix: str
    enorm: Module
    hnorm: Module
    eh_proj: Module
    input_layernorm: Module
    q_source: Module | None
    q_proj: Module | None
    kv_source: Module
    kv_proj: Module
    kv_rank: int


def replay_mtp_draft_calibration(
    model: Module,
    collector: DsaHif8ScaleCollector,
    capture: MtpCalibrationCapture,
) -> list[str]:
    """Replay the MTP draft's DSA latent path over the captured calibration.

    Mirrors msmodelslim's ``mtp_preprocess``: for every captured batch, the
    draft input is rebuilt as
    ``eh_proj(cat(enorm(embed(ids_shifted)), hnorm(backbone_hidden)))`` where
    ``ids_shifted`` is the input ids shifted left by one position with the
    last slot holding the argmax next-token prediction
    (``lm_head(norm(hidden))``, computed on the last position only). Each
    draft block then runs up to its ``q_a_layernorm`` / ``kv_a_layernorm`` --
    the exact tensors the HiF8 scales govern -- so the still-attached
    collector hooks accumulate the draft's Q/K absmax exactly like a full
    draft forward would.

    ``lm_head``'s *forward* patching from the pipeline's calibration context
    has already exited by replay time; the next-token prediction is therefore
    computed as a direct matmul against the (onloaded) ``lm_head`` weight on
    the last position only -- cheaper than a full ``[batch, seq, vocab]``
    projection and independent of any forward patching. Only the modules on
    the replayed path are onloaded here and
    re-offloaded afterwards; the MTP subtree itself stays in whatever state
    the pipeline left it in (the sequential pipeline onloads untraced modules
    for the whole calibration).

    :param model: the model being compressed, with ``model.mtp`` attached
    :param collector: the still-attached HiF8 DSA scale collector
    :param capture: inputs captured from the backbone calibration pass
    :return: draft attention prefixes that now carry Q/K statistics
    """
    if not capture.input_ids or not capture.hidden_states:
        logger.warning(
            "MTP draft replay skipped: no backbone inputs were captured "
            "(no calibration forward reached the final norm)."
        )
        return []

    num_batches = min(len(capture.input_ids), len(capture.hidden_states))
    if len(capture.input_ids) != len(capture.hidden_states):
        logger.warning(
            "Captured MTP inputs are misaligned ({} id batches vs {} hidden "
            "batches); replaying the first {}.",
            len(capture.input_ids),
            len(capture.hidden_states),
            num_batches,
        )

    # Draft targets are the collector's hook targets inside the MTP subtree
    # (``collector._targets`` is shared private state with this module).
    draft_targets = [
        (prefix, attention, q_source, kv_source)
        for prefix, attention, q_source, kv_source in collector._targets
        if _is_mtp_subtree(prefix)
    ]
    if not draft_targets:
        logger.warning(
            "No DSA latent hooks are attached under the MTP draft; the draft "
            "layer will not receive fa_q/fa_k scales."
        )
        return []

    # Modules the replay executes, keyed by their model names so the
    # on/offload bookkeeping matches the sequential pipeline's.
    modules: dict[str, Module] = {}

    embed_name = _module_name(model, capture.embed_module)
    if embed_name is None:
        logger.warning("Shared embedding is not registered on the model; MTP replay skipped.")
        return []
    modules[embed_name] = capture.embed_module

    norm_name = _module_name(model, capture.final_norm)
    if norm_name is None:
        logger.warning("Final norm is not registered on the model; MTP replay skipped.")
        return []
    modules[norm_name] = capture.final_norm

    lm_head = _find_lm_head(model)
    if lm_head is not None:
        lm_head_name = _module_name(model, lm_head)
        modules[lm_head_name if lm_head_name is not None else "lm_head"] = lm_head

    usable: list[_DraftReplayTarget] = []
    for prefix, attention, q_source, kv_source in draft_targets:
        block_path = prefix.rsplit(".", 1)[0]
        block = model.get_submodule(block_path)
        layer_path = block_path
        layer = block
        if not hasattr(layer, "eh_proj"):
            layer_path = block_path.rsplit(".", 1)[0]
            layer = model.get_submodule(layer_path)

        enorm = getattr(layer, "enorm", None)
        hnorm = getattr(layer, "hnorm", None)
        eh_proj = getattr(layer, "eh_proj", None)
        input_layernorm = getattr(block, "input_layernorm", None)
        q_proj = getattr(attention, "q_a_proj", None)
        kv_proj = None
        kv_proj_name = _DSA_LATENT_PROJECTION_SUFFIXES[0]
        for candidate_name in _DSA_LATENT_PROJECTION_SUFFIXES:
            candidate = getattr(attention, candidate_name, None)
            if isinstance(candidate, Module):
                kv_proj = candidate
                kv_proj_name = candidate_name
                break
        kv_rank = getattr(attention, "kv_lora_rank", None) or collector.kv_nope_dim

        if not all(
            isinstance(m, Module) for m in (enorm, hnorm, eh_proj, input_layernorm, kv_proj)
        ):
            logger.warning(
                "MTP draft layer at {} does not expose the expected DSA input "
                "path; skipping its fa scales.",
                prefix,
            )
            continue
        if q_source is not None and not isinstance(q_proj, Module):
            logger.warning(
                "MTP draft layer at {} has a query layernorm but no q_a_proj; "
                "collecting only the KV scale.",
                prefix,
            )
            q_proj = None

        modules[f"{layer_path}.enorm"] = enorm
        modules[f"{layer_path}.hnorm"] = hnorm
        modules[f"{layer_path}.eh_proj"] = eh_proj
        modules[f"{block_path}.input_layernorm"] = input_layernorm
        if q_source is not None and q_proj is not None:
            modules[f"{prefix}.q_a_proj"] = q_proj
            modules[f"{prefix}.q_a_layernorm"] = q_source
        modules[f"{prefix}.{kv_proj_name}"] = kv_proj
        if kv_source is not kv_proj:
            modules[f"{prefix}.{_DSA_KV_LAYERNORM_SUFFIX}"] = kv_source

        usable.append(
            _DraftReplayTarget(
                prefix=prefix,
                enorm=enorm,
                hnorm=hnorm,
                eh_proj=eh_proj,
                input_layernorm=input_layernorm,
                q_source=q_source,
                q_proj=q_proj,
                kv_source=kv_source,
                kv_proj=kv_proj,
                kv_rank=kv_rank,
            )
        )

    if not usable:
        return []

    # The capture hooks must not re-record the replay's own module calls.
    capture.detach()

    onload_kwargs = subgraph_onload_modules(modules)
    try:
        embed_param = next(capture.embed_module.parameters(), None)
        if embed_param is None:
            logger.warning("Shared embedding has no parameters; MTP replay skipped.")
            return []
        device = embed_param.device
        dtype = embed_param.dtype

        logger.info(
            "Replaying MTP draft calibration for {} draft layer(s) over {} "
            "captured batch(es)",
            len(usable),
            num_batches,
        )

        with torch.no_grad():
            for batch_idx, (ids, hidden) in enumerate(
                tqdm(
                    zip(
                        capture.input_ids[:num_batches],
                        capture.hidden_states[:num_batches],
                    ),
                    total=num_batches,
                    desc="MTP draft KV-scale replay",
                )
            ):
                if ids.shape[-1] < 2:
                    continue
                ids = ids.to(device)
                hidden = hidden.to(device=device, dtype=dtype)

                # Next-token prediction from the final position, mirroring
                # msmodelslim's ``mtp_preprocess``: norm -> lm_head -> argmax.
                normed = capture.final_norm(hidden)
                predicted = None
                if lm_head is not None:
                    weight = lm_head.weight
                    logits = torch.nn.functional.linear(normed[:, -1].to(weight.dtype), weight)
                    predicted = logits.argmax(dim=-1).to(ids.dtype)
                if predicted is None:
                    predicted = torch.zeros(ids.shape[0], dtype=ids.dtype, device=device)

                # Draft input ids: shifted left by one position, with the last
                # slot holding the predicted token.
                ids_shifted = torch.cat([ids[:, 1:], predicted.unsqueeze(1)], dim=1)
                embeds = capture.embed_module(ids_shifted)

                for target in usable:
                    enorm_out = target.enorm(embeds)
                    hnorm_out = target.hnorm(hidden)
                    eh_input = torch.cat([enorm_out, hnorm_out], dim=-1)
                    block_input = target.eh_proj(eh_input)
                    x = target.input_layernorm(block_input)

                    # Fire the collector-hooked layernorms with the same
                    # tensors a full draft forward would produce.
                    if target.q_source is not None and target.q_proj is not None:
                        q_out = target.q_proj(x)
                        target.q_source(q_out)
                    kv_out = target.kv_proj(x)
                    if target.kv_source is not target.kv_proj:
                        target.kv_source(kv_out[..., : target.kv_rank])

        with_scales = [
            target.prefix for target in usable if collector.has_scales_for([target.prefix])
        ]
        if with_scales:
            logger.info("MTP draft replay produced Q/K statistics for: {}", with_scales)
        else:
            logger.warning(
                "MTP draft replay produced no Q/K statistics; the draft layer "
                "will not receive fa_q/fa_k scales."
            )
        return with_scales
    finally:
        # Offload exactly what this replay onloaded; modules the pipeline had
        # already onloaded (the untraced MTP subtree) are left untouched.
        onloaded = {name: module for name, module in modules.items() if name in onload_kwargs}
        if onloaded:
            subgraph_offload_modules(onloaded, onload_kwargs)
