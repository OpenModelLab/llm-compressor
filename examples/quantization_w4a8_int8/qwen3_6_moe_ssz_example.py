"""SSZ for Qwen3.6-35B-A3B: W4A8 routed experts and W8A8 text projections.

Vision, routers, embeddings, and sensitive linear-attention projections keep
their original floating-point weights. No calibration dataset is needed.
Packed experts reduce checkpoint size without changing SSZ scales or codes.
Packed output requires a loader supporting packed W4A8 experts alongside INT8
text weights. EXPERT_FORMAT="int-quantized" remains available for legacy loaders.
Calibration selects the available compute device by default. Set
CALIBRATION_DEVICE="cpu" to bypass accelerator execution after a native crash.
"""

import faulthandler
import re
from importlib.util import find_spec
from pathlib import Path

import torch
from compressed_tensors.distributed import is_distributed
from compressed_tensors.offload import set_onload_device, update_offload_parameter
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

from llmcompressor import oneshot
from llmcompressor.core.session_functions import LifecycleCallbacks
from llmcompressor.modifiers.quantization.ssz import SSZModifier
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.utils import load_context
from llmcompressor.utils.dev import get_main_device

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SAVE_DIR = "Qwen3.6_35B_A3B_SSZ_W4A8_packed"
STEP = 50
EXPERT_FORMAT = "pack-quantized"
CALIBRATION_DEVICE = "auto"
CALIBRATION_BATCH_SIZE = 32

EXPERT_TARGET = r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"


def get_calibration_device():
    return (
        get_main_device()
        if CALIBRATION_DEVICE == "auto"
        else torch.device(CALIBRATION_DEVICE)
    )


@CalibrationPipeline.register("qwen3_6_ssz_datafree")
class SSZDataFreePipeline(CalibrationPipeline):
    """Single-process, per-channel SSZ with explicit placement and progress."""

    @staticmethod
    def __call__(model, dataloader, dataset_args):
        if is_distributed():
            raise ValueError("This example supports single-process calibration only")
        if CALIBRATION_BATCH_SIZE < 1:
            raise ValueError("CALIBRATION_BATCH_SIZE must be positive")
        device = get_calibration_device()
        set_onload_device(model, device)
        LifecycleCallbacks.calibration_start()
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if getattr(module, "weight_observer", None) is not None
        ]
        if modules:
            weight = modules[0][1].weight
            print(
                f"SSZ actual input: {modules[0][0]}, device={weight.device}, "
                f"dtype={weight.dtype}, shape={tuple(weight.shape)}",
                flush=True,
            )
            del weight
        with tqdm(
            total=len(modules), desc=f"SSZ fitting ({device})", unit="matrix"
        ) as progress:
            for start in range(0, len(modules), CALIBRATION_BATCH_SIZE):
                batch = modules[start : start + CALIBRATION_BATCH_SIZE]
                progress.set_postfix_str(batch[0][0], refresh=False)
                # Use the same native callback as DataFreePipeline. All weights
                # are independent CHANNEL observers, so batch boundaries do not
                # change the scale search or integer assignments.
                LifecycleCallbacks.sequential_epoch_end([m for _, m in batch])
                progress.update(len(batch))
        LifecycleCallbacks.calibration_end()


def create_recipe():
    if EXPERT_FORMAT not in ("int-quantized", "pack-quantized"):
        raise ValueError("EXPERT_FORMAT must be int-quantized or pack-quantized")
    return [
        SSZModifier(
            config_groups={
                "experts_w4a8": {
                    "targets": [EXPERT_TARGET],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                    "format": EXPERT_FORMAT,
                },
                "text_w8a8": {
                    "targets": [
                        r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
                        r"re:.*mlp\.shared_expert\.(gate_proj|up_proj|down_proj)$",
                        r"re:.*linear_attn\.(in_proj_qkv|in_proj_z)$",
                    ],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                    "format": "int-quantized",
                },
            },
            step=STEP,
        )
    ]


def main():
    if Path(SAVE_DIR).resolve() == Path(MODEL_ID).resolve():
        raise ValueError("SAVE_DIR must differ from the original model directory")
    recipe = create_recipe()
    if EXPERT_FORMAT == "pack-quantized":
        print(
            "Saving packed INT4 experts and INT8 text weights (mixed-precision). "
            "Load directly with a backend supporting these per-group formats."
        )

    if CALIBRATION_DEVICE != "cpu" and find_spec("torch_npu") is not None:
        import torch_npu  # noqa: F401

    device = get_calibration_device()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # This context also unfolds fused 3D experts into quantizable 2D Linears.
    with load_context(Qwen3_5MoeForConditionalGeneration):
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
        )
    model.eval()

    # Detect missing/incompatible MoE linearization before changing any weights.
    expert_count = sum(
        isinstance(module, torch.nn.Linear)
        and re.fullmatch(EXPERT_TARGET.removeprefix("re:"), name) is not None
        for name, module in model.named_modules()
    )
    text_config = model.config.text_config
    expected_count = text_config.num_hidden_layers * text_config.num_experts * 3
    if expert_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} expert Linear projections, got {expert_count}. "
            "Use a llm-compressor/transformers version with Qwen3.5 MoE linearization."
        )

    # Preserve offset norms across the native BF16 gamma conversion, as in the
    # dense SSZ example. Restore offload storage before serializing the model.
    norm_weights = {
        name: module.weight.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, Qwen3_5MoeRMSNorm)
    }
    print(f"Preparing SSZ calibration on {device}.", flush=True)
    # Enable fatal-signal traces without periodic "Timeout" diagnostics. SIGSEGV
    # cannot be recovered by a Python try/except or the Norm-restoration finally.
    faulthandler.enable()
    try:
        print(
            f"Starting SSZ on {device}: {expert_count} expert projections, "
            f"up to {STEP} iterations per matrix, plus selected W8 text projections.",
            flush=True,
        )
        oneshot(
            model=model,
            processor=processor,
            recipe=recipe,
            pipeline="qwen3_6_ssz_datafree",
        )
    finally:
        for name, weight in norm_weights.items():
            update_offload_parameter(model.get_submodule(name), "weight", weight)

    # Keep final quantization and packing on CPU even when fitting was explicitly
    # run on an accelerator. No full-model accelerator allocation is needed.
    set_onload_device(model, "cpu")
    print("SSZ calibration finished; compressing and saving weights.", flush=True)
    # Respect per-group formats: only INT4 experts are packed. A global override
    # would also pack W8 weights, changing their loader contract without savings.
    model.save_pretrained(
        SAVE_DIR,
        save_compressed=True,
        max_shard_size="4GB",
    )
    processor.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
