"""Quantize GLM-5.3-Flash to W4A4C8 (HiF8) static quantization on Ascend NPU (A5).

Ascend NPU, single-card variant of ``glm53_flash_w4a4c8.py``. Same recipe, same
checkpoint contract (per 需求描述.txt), adapted for an offline NPU machine:

  2.1 MLP linear  : Router experts -> W4A4 (MXFP4); Shared experts -> W8A8 (MXFP8)
  2.2 KVCache     : DSA -> HiF8 (static per-tensor FLOAT32 scales fa_q/fa_k = absmax/32768,
                    produced here via ``compute_dsa_kv_scale=True``); LI -> FP8 is
                    runtime-only (vllm-ascend ``indexer_kv_dtype=fp8``), NOT
                    produced here.
  2.3 Other linear: W8A8 (MXFP8)

Differences from the CUDA/CPU version (``glm53_flash_w4a4c8.py``):

  * Runs on ONE NPU card (default: physical card 3) selected via
    ``ASCEND_RT_VISIBLE_DEVICES``. The variable is read by torch_npu at import
    time, so this script sets it from ``NPU_VISIBLE_DEVICES`` (default ``"3"``)
    *before* importing torch_npu. It can also be exported by the launcher.
  * Single process: no torchrun, no HCCL process group. The sequential
    pipeline runs with world size 1 (all weights stay on host memory via
    ``auto_offload``; one decoder layer is onloaded to the NPU at a time).
    Multi-card would need ``torchrun`` plus an ``npu -> hccl`` backend branch
    in ``compressed_tensors.distributed.init_dist`` (upstream only knows
    nccl/xccl/gloo).
  * Weights and calibration data come from LOCAL paths (offline machine):

      GLM53_FLASH_MODEL   BF16 checkpoint dir  (default
                          /mnt/share/y00506514/GLM-5.3-Flash)
      GLM53_FLASH_OUTPUT  output dir          (default
                          /mnt/share/y00506514/GLM-5.3-Flash-W4A4C8-v2)
      GLM53_FLASH_CALIB   calibration data     (default
                          /mnt/share/y00506514/W4A4C8_HiF8/calibration_data)
                         A data file (.jsonl/.json/.csv/.parquet) with a ``text``
                         column, or a directory which is scanned (recursively)
                         for one -- train-named files win, then the largest.
                         Use a few hundred diverse samples, e.g. the same mix
                         as ``perfectblend`` (mlabonne/open-perfectblend).

NPU compatibility notes (verified against llm-compressor / compressed-tensors):

  * ``get_main_device()`` resolves the accelerator through ``torch.accelerator``;
    with torch_npu registered it returns ``npu:0``. The script fails fast if the
    installed torch_npu does not register with ``torch.accelerator`` -- otherwise
    llm-compressor would silently calibrate on CPU.
  * compressed-tensors quantization/packing kernels are Triton-accelerated on
    CUDA only; on NPU the ``ImplBackend`` registry automatically falls back to
    the pure-PyTorch implementations (slower, same numerics).
  * The HiF8 element quantization itself runs at inference time on the
    vllm-ascend side (``npu_quantize``); this script only produces the static
    ``fa_q.scale`` / ``fa_k.scale`` buffers that side consumes.

Run (physical card 3, plain python, no torchrun):

    python examples/quantizing_moe/glm53_flash_w4a4c8_npu.py
"""

import glob
import os
import shutil

# ASCEND_RT_VISIBLE_DEVICES is read by torch_npu at import time; it must be set
# before ``import torch_npu``. "3" = physical card #3 (remapped to npu:0).
os.environ.setdefault(
    "ASCEND_RT_VISIBLE_DEVICES", os.environ.get("NPU_VISIBLE_DEVICES", "3")
)

import torch
import torch_npu  # noqa: F401  (registers the "npu" device; must follow `import torch`)

from compressed_tensors.quantization.quant_scheme import (
    MXFP4,
    MXFP8,
    QuantizationScheme,
)
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.utils import load_context

# --- NPU pre-flight: fail fast instead of silently calibrating on CPU -------
if not torch.npu.is_available():
    raise RuntimeError(
        "torch_npu reports no available NPU. Check that "
        f"ASCEND_RT_VISIBLE_DEVICES={os.environ['ASCEND_RT_VISIBLE_DEVICES']} "
        "points at a usable card and that the CANN environment is sourced."
    )
_accel = torch.accelerator.current_accelerator()
if _accel is None or _accel.type != "npu":
    raise RuntimeError(
        f"torch.accelerator.current_accelerator() is {_accel!r}, expected npu. "
        "llm-compressor resolves its main device through torch.accelerator, so "
        "without npu registered there it would silently compress on CPU. "
        "Upgrade torch_npu/CANN to a build that registers with torch.accelerator."
    )
DEVICE = torch.device("npu", 0)  # the single visible (physical card 3) device
print(
    f"[glm53-flash-npu] calibrating on {DEVICE} "
    f"(physical card {os.environ['ASCEND_RT_VISIBLE_DEVICES']})"
)

# --- Inputs: local model + calibration data (offline NPU machine) -----------
MODEL_PATH = os.environ.get("GLM53_FLASH_MODEL", "/mnt/share/y00506514/GLM-5.3-Flash")
OUTPUT_PATH = os.environ.get(
    "GLM53_FLASH_OUTPUT", "/mnt/share/y00506514/GLM-5.3-Flash-W4A4C8-v2"
)
CALIB_PATH = os.environ.get(
    "GLM53_FLASH_CALIB", "/mnt/share/y00506514/W4A4C8_HiF8/calibration_data"
)
if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint dir not found: {MODEL_PATH}. Set GLM53_FLASH_MODEL to "
        "the dequantized BF16 GLM-5.3-Flash checkpoint (see "
        "examples/model_free_ptq/decompress_glm53_flash.py)."
    )


def _resolve_calibration_file(path: str) -> str:
    """Resolve the calibration source to ONE data file.

    The resolved file is passed to ``oneshot`` as ``dataset``; llm-compressor
    hands it straight to ``datasets.load_dataset``, which accepts local data
    files (inferring the builder from the extension) and exposes them as the
    "train" split. A directory is scanned recursively and the best candidate
    is picked: files named like ``train`` first, then the largest file.
    """
    calib_exts = (".jsonl", ".json", ".csv", ".parquet")

    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Calibration data not found: {path}. Set GLM53_FLASH_CALIB to a "
            "data file (.jsonl/.json/.csv/.parquet, with a 'text' column) or to "
            "a directory containing one."
        )

    candidates = [
        os.path.join(root, name)
        for root, _, files in os.walk(path)
        for name in files
        if name.lower().endswith(calib_exts)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No calibration data file ({'/'.join(calib_exts)}) found under "
            f"{path}. Set GLM53_FLASH_CALIB to a data file or a directory "
            "containing one."
        )
    train_named = [c for c in candidates if "train" in os.path.basename(c).lower()]
    pool = train_named or candidates
    chosen = max(pool, key=os.path.getsize)
    print(f"[glm53-flash-npu] calibration data: {chosen}")
    ignored = sorted(set(candidates) - {chosen})
    if ignored:
        print(f"[glm53-flash-npu]   (ignoring {len(ignored)} other file(s): {ignored})")
    return chosen


CALIB_FILE = _resolve_calibration_file(CALIB_PATH)

# llmcompressor's oneshot resolves a *string* `dataset` through the
# TextGenerationDataset registry (c4/custom/...), so a local file path raises
# KeyError. Load the file into a `datasets.Dataset` object here and pass that
# instead: non-string datasets are routed to the "custom" dataset manager and
# go straight to tokenization.
from datasets import load_dataset as _hf_load_dataset

calibration_dataset = _hf_load_dataset("json", data_files=CALIB_FILE, split="train")
print(
    f"[glm53-flash-npu] loaded {len(calibration_dataset)} calibration samples "
    f"from {CALIB_FILE}"
)

with load_context(Glm5NextForConditionalGeneration):
    # GLM-5.3-Flash is a vision-language MoE model, so it must be loaded with its
    # `Glm5NextForConditionalGeneration` class (not `AutoModelForCausalLM`) in
    # order to keep the vision tower. `auto_offload` keeps all weights on host
    # memory; the sequential pipeline onloads one decoder layer at a time.
    model = Glm5NextForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto_offload",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Configure the mixed quantization recipe (identical to the CUDA version, plus
# the pre-quantization FFN smoothing below). SmoothQuantModifier reproduces the
# msmodelslim FlexSmooth per-channel fold (post_attention_layernorm -> [gate,
# up] + router on MoE layers), the source of ms's ~2.5pp accuracy lead on
# GLM-5.3-Flash. `mappings=None` infers the Glm5Next dynamic mapping (dense/MoE
# split driven by config.mlp_layer_types). `activation_absmax=True` +
# `weight_scale_multiplier=1.0` + independent `smoothing_beta` reproduce ms's
# `s = absmax(a)^alpha / absmax(w)^beta` exactly (alpha/beta fixed for now; ms
# searches them per layer, to be added next). The transform is folded into the
# norm + projection weights, so the saved checkpoint is self-contained and the
# vllm-ascend runtime needs no change.
recipe = [
    SmoothQuantModifier(
        algorithm="smoothquant",
        smoothing_strength=0.5,
        smoothing_beta=0.5,
        activation_absmax=True,
        weight_scale_multiplier=1.0,
    ),
    QuantizationModifier(
        config_groups={
            # Router experts -> W4A4 (MXFP4). Listed first so its name-based regex
            # targets take priority over the `Linear` catch-all below (mixing the
            # two groups preserves the deliberate first-match-wins ordering).
            "routed_experts": QuantizationScheme(
                targets=[r"re:.*mlp\.experts\..*(gate|up|down)_proj$"],
                **MXFP4,
            ),
            # Shared experts + attention projections + dense MLP + KDA linear
            # attention -> W8A8 (MXFP8). `Linear` matches every nn.Linear module
            # not already captured by the routed-experts group above.
            "shared_and_other": QuantizationScheme(
                targets=["Linear"],
                **MXFP8,
            ),
        },
        ignore=[
            r"re:.*visual.*",  # vision tower stays full precision
            r"re:.*lm_head",
            r"re:.*mlp\.gate$",  # MoE router
            r"re:.*embed_tokens$",
            r"re:.*eh_proj$",  # MTP input projection
            # --- scope aligned with the msmodelslim W4A4C8 best-practice recipe ---
            # ms leaves the KDA (linear-attention) projections in BF16; only
            # o_proj is quantized there. Keep q/k/v/b/g projections full precision.
            r"re:.*self_attn\.(q_proj|k_proj|v_proj|b_proj|g_a_proj|g_b_proj)$",
            # KDA forget-gate projections are NESTED under the `forget_gate`
            # container module (`self_attn.forget_gate.f_a_proj`), so the regex
            # must include the intermediate path -- `.*self_attn\.f_a_proj$`
            # silently matches nothing and the projections get quantized anyway.
            r"re:.*self_attn\.forget_gate\.(f_a_proj|f_b_proj)$",
            # ms quantizes only q_a_proj/q_b_proj/kv_a_proj_with_mqa/o_proj on DSA
            # layers; the up-projection kv_b_proj and the rest stay BF16.
            r"re:.*self_attn\.kv_b_proj$",
            # indexer: ms quantizes only wq_b (falls through to the Linear
            # catch-all below as W8A8); wk and weights_proj stay BF16.
            r"re:.*self_attn\.indexer\.(wk|weights_proj)$",
        ],
    ),
]

# Apply the recipe. `compute_dsa_kv_scale=True` hooks the `q_a_layernorm` /
# `kv_a_layernorm` outputs during calibration and writes per-tensor FLOAT32
# `fa_q.scale` / `fa_k.scale` buffers (shape (1,), value absmax/32768) into each
# DSA/MLA layer of the saved checkpoint. The HiF8 element quantization itself and
# its runtime consumption remain on the vllm-ascend inference side.
#
# The MTP draft layer (layers.45) is never forwarded by the calibration
# pipelines (it sits outside every traced subgraph), so its backbone-pass
# inputs (input_ids + pre-final-norm hidden) are captured via pre-hooks and
# its DSA latent path is replayed right after the backbone pass while the
# collector hooks are still attached -- the msmodelslim mtp_preprocess
# equivalent. Without this, layers.45 would ship without fa scales and the
# vllm-ascend HIF8 loader would fall back to a BF16 draft KV cache.
#
# `pipeline="sequential"` is REQUIRED for the DSA KV scales (and for the
# `sequential_targets` below to have any effect): the default `independent`
# pipeline infers `DataFreePipeline` for this recipe (all activations are
# dynamic, weights are static RTN, so `requires_calibration_data=False`) and
# runs NO calibration forward pass at all -- every hook would stay silent and
# `fa_q.scale` / `fa_k.scale` would never be collected.
oneshot(
    model=model,
    processor=tokenizer,
    recipe=recipe,
    # Pre-loaded `datasets.Dataset` object (see above). A raw path string would
    # be looked up in the dataset registry and fail with KeyError.
    dataset=calibration_dataset,
    text_column="text",
    max_seq_length=2048,
    num_calibration_samples=512,
    batch_size=1,
    shuffle_calibration_samples=True,
    # Force the sequential (data-dependent) pipeline for DSA KV scale
    # collection; see the comment block above.
    pipeline="sequential",
    # Large MoE: load one decoder layer at a time onto the NPU. A fixed
    # subgraph size of 1 also skips the accelerator free-memory probe, which is
    # the safest configuration for NPU where llm-c has not been exercised yet.
    sequential_targets=["Glm5NextTextDecoderLayer"],
    sequential_targets_per_subgraph=1,
    sequential_offload_device="cpu",  # activations wait on host between layers
    propagate_error=False,
    compute_dsa_kv_scale=True,
)

# Save to disk compressed. The MTP draft layer (layers.45) is now loaded and
# quantized via the dedicated glm5_next_text adapter (Glm5NextMtpModel, see
# mtp_compat/glm5_next_for_conditional_generation.py): it reconstructs the draft
# as a no-HC DSA+MoE layer and renames the VLM `model.language_model.*` keys, so
# the draft weights enter the same MXFP4/MXFP8 pipeline as the backbone. The
# `compute_dsa_kv_scale=True` MTP capture/replay (see
# llmcompressor.entrypoints.hif8_kv) additionally drives the draft's DSA latent
# path over the captured backbone outputs, so layers.45 also exports
# `self_attn.fa_q.scale` / `fa_k.scale` like the backbone DSA layers (ms parity).
# The save utility only falls back to copying MTP tensors unchanged (BF16) when
# the adapter could not load them (e.g. config.json lacks num_nextn_predict_layers).
model.generation_config.top_p = None
SAVE_DIR = OUTPUT_PATH
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

# --- Copy processor-side files (save_pretrained does not write them) --------
# save_pretrained(save_compressed=True) writes weights + model config, and
# tokenizer.save_pretrained writes tokenizer/chat-template files. For a VL
# model (vision tower stays BF16 above), vllm serve additionally needs the
# processor / image / video preprocessor configs and any custom remote-code
# modules -- none of which are emitted here. Copy them from the source
# checkpoint so the output is directly serveable without manual patching.
_PROCESSOR_SIDE_FILES = (
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",  # present on some GLM-5 video-capable repos
    "chat_template.jinja",
)
for fn in _PROCESSOR_SIDE_FILES:
    src = os.path.join(MODEL_PATH, fn)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(SAVE_DIR, fn))

# Custom remote-code modules (e.g. image_processing_glm5_next.py) referenced by
# the processor's `auto_map`; copy them too so trust_remote_code / processor
# auto_map resolution works on the saved checkpoint.
for src in glob.glob(os.path.join(MODEL_PATH, "*.py")):
    shutil.copy2(src, os.path.join(SAVE_DIR, os.path.basename(src)))

print(f"[glm53-flash-npu] saved quantized model to {SAVE_DIR}")