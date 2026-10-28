"""SSZ + dynamic W8A8 for dense Qwen3.6-27B.

Set the model and output paths below before running this example.
SSZ optimizes weight scales without a calibration dataset.
"""

from importlib.util import find_spec
from pathlib import Path

import torch
from compressed_tensors.offload import update_offload_parameter
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.ssz import SSZModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.6-27B"
SAVE_DIR = "Qwen3.6_27B_SSZ_W8A8"
STEP = 50


def main():
    if Path(SAVE_DIR).resolve() == Path(MODEL_ID).resolve():
        raise ValueError("SAVE_DIR must differ from the original model directory")

    # Quantize text projections and selected linear-attention and vision layers.
    recipe = [
        SSZModifier(
            targets=[
                r"re:.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
                r"re:.*mlp\.(gate_proj|up_proj|down_proj)$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z)$",
                r"re:.*visual\.blocks\.\d+\.attn\.(qkv|proj)$",
                r"re:.*visual\.blocks\.\d+\.mlp\.linear_fc1$",
            ],
            scheme="W8A8",
            step=STEP,
        ),
    ]

    # Register the Ascend backend when installed; CUDA needs no extra import.
    if find_spec("torch_npu") is not None:
        import torch_npu  # noqa: F401

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    # Keep weights on CPU; the native lifecycle onloads them for quantization.
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
        )
    model.eval()

    # SSZ only fits Linear weight scales. Preserve offset norms across the native
    # calibration context's BF16 gamma conversion; no smoothing is performed here.
    norm_weights = {
        name: module.weight.detach().cpu().clone()
        for name, module in model.named_modules()
        if isinstance(module, Qwen3_5RMSNorm)
    }
    try:
        oneshot(
            model=model,
            processor=processor,
            recipe=recipe,
            pipeline="independent",
        )
    finally:
        for name, weight in norm_weights.items():
            update_offload_parameter(model.get_submodule(name), "weight", weight)

    # Save only after restoring the original norms to persistent offload storage.
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    processor.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    main()
