"""Forward-calibrated GLM-5.3-Flash INT W4A8, with optional MLP SmoothQuant.

Uses native load_context, SmoothQuantModifier, GPTQModifier and oneshot.
Only backbone MLP projections are quantized. MTP is preserved in floating point;
KV cache remains runtime-dynamic: serve with --kv-cache-dtype int8.
Calibration JSONL contains {"messages": [...]} using the HF image/text schema.
"""

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path


def make_config_groups(num_layers):
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    layers = "(?:" + "|".join(str(i) for i in range(num_layers)) + ")"
    prefix = rf"re:^model\.language_model\.layers\.{layers}\.mlp"
    groups = {}
    for name, bits, suffix in (
        ("routed", 4, r"\.experts\.\d+\.(gate|up|down)_proj$"),
        ("dense", 8, r"\.(gate|up|down)_proj$"),
        ("shared", 8, r"\.shared_experts\.(gate|up|down)_proj$"),
    ):
        groups[name] = dict(
            targets=[prefix + suffix],
            weights=dict(
                num_bits=bits,
                type="int",
                strategy="channel",
                symmetric=True,
                dynamic=False,
                observer="memoryless_minmax",
                actorder=None,
            ),
            input_activations=dict(
                num_bits=8, type="int", strategy="token", symmetric=True, dynamic=True
            ),
            format="pack-quantized" if bits == 4 else "int-quantized",
        )
    return groups


def quantized_modules(model, groups):
    import torch

    patterns = [re.compile(t[3:]) for g in groups.values() for t in g["targets"]]
    modules = {
        name: module
        for name, module in model.named_modules()
        if any(p.fullmatch(name) for p in patterns)
    }
    if not modules or not any(".experts." in name for name in modules):
        raise ValueError("No linearized GLM backbone experts matched the recipe")
    if any(not isinstance(module, torch.nn.Linear) for module in modules.values()):
        raise ValueError("Quantized targets must be per-expert 2D Linear modules")
    return modules


def make_smoothquant_mappings(model):
    """Compensate every consumer of the backbone post-attention norm output."""
    import torch

    mappings = []
    for name, layer in model.named_modules():
        if not re.fullmatch(r"model\.language_model\.layers\.\d+", name):
            continue
        if type(layer).__name__ != "Glm5NextTextDecoderLayer":
            raise ValueError(f"Unsupported decoder topology at {name}")
        norm = layer.post_attention_layernorm
        if type(norm).__name__ != "Glm5NextTextRMSNorm":
            raise ValueError(f"Unsupported norm at {name}")
        mlp = layer.mlp
        consumers = []
        if hasattr(mlp, "experts"):
            if not isinstance(mlp.experts, torch.nn.ModuleList):
                raise ValueError("Use load_context to linearize experts first")
            if type(mlp.gate).__name__ != "Glm5NextTextTopkRouter":
                raise ValueError(f"Unsupported router topology at {name}")
            for index in range(mlp.config.n_routed_experts):
                for projection in ("gate_proj", "up_proj"):
                    consumers.append(f"{name}.mlp.experts.{index}.{projection}")
            consumers += [
                f"{name}.mlp.shared_experts.gate_proj",
                f"{name}.mlp.shared_experts.up_proj",
                f"{name}.mlp.gate",
            ]
        else:
            if type(mlp).__name__ != "Glm5NextTextMLP":
                raise ValueError(f"Unsupported MLP topology at {name}")
            consumers += [f"{name}.mlp.gate_proj", f"{name}.mlp.up_proj"]
        for consumer in consumers:
            weight = model.get_submodule(consumer).weight
            if weight.ndim != 2 or weight.shape[1] != norm.weight.numel():
                raise ValueError(f"Invalid SmoothQuant consumer {consumer}")
        mappings.append((consumers, f"{name}.post_attention_layernorm"))
    if not mappings:
        raise ValueError("No supported GLM MLP norm-linear mappings found")
    return mappings


def load_records(path, num_samples, seed, allow_text_only=False):
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Line {line_number}: expected non-empty messages")
        has_image = False
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                continue
            if not isinstance(content, list) or not content:
                raise ValueError(f"Line {line_number}: invalid message content")
            for block in content:
                if block.get("type") == "image":
                    has_image = True
                    image = block.get("image")
                    if not isinstance(image, str) or not image:
                        raise ValueError(f"Line {line_number}: image path is required")
                    if "://" not in image and not image.startswith("data:"):
                        image_path = Path(image)
                        if not image_path.is_absolute():
                            image_path = path.parent / image_path
                        if not image_path.is_file():
                            raise ValueError(f"Missing calibration image: {image_path}")
                        block["image"] = str(image_path.resolve())
                elif block.get("type") != "text":
                    raise ValueError(
                        "Use HF text/image content blocks for this example"
                    )
        records.append((messages, has_image, line_number))
    if not records:
        raise ValueError("Calibration JSONL is empty")
    random.Random(seed).shuffle(records)
    records = records[:num_samples]
    modalities = {item[1] for item in records}
    if len(modalities) != 1:
        raise ValueError(
            "Use an all-image or all-text calibration file: mixed forward branches "
            "are not safe with a single sequential trace"
        )
    if modalities == {False} and not allow_text_only:
        raise ValueError(
            "MMMU calibration needs images; use --allow-text-only for probes"
        )
    return records


def make_dataloader(records, processor, max_seq_length):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class CalibrationDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            messages, _, line = records[index]
            encoded = dict(
                processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    truncation=False,
                )
            )
            if encoded["input_ids"].shape[-1] > max_seq_length:
                raise ValueError(
                    f"Line {line} exceeds --max-seq-length; shorten the sample or "
                    "adjust image resolution. Do not slice image token sequences."
                )
            if not all(isinstance(value, torch.Tensor) for value in encoded.values()):
                raise ValueError("Processor must return tensor-only model inputs")
            if records[index][1] and not (
                "pixel_values" in encoded and "image_grid_thw" in encoded
            ):
                raise ValueError("Image sample lost pixel_values/image_grid_thw")
            return encoded

    # Each processor result already includes the text batch dimension and packed
    # image/grid structure. Do not stack or squeeze these tensors again.
    return DataLoader(CalibrationDataset(), batch_size=None, num_workers=0)


def load_model(
    source,
    dtype,
    *,
    linearize_after_load=False,
    offload_folder=None,
    device_map="auto_offload",
):
    from transformers import Glm5NextForConditionalGeneration

    if linearize_after_load:
        from compressed_tensors.offload import load_offloaded_model

        from llmcompressor.modeling.moe.linearize import linearize_moe

        with load_offloaded_model(Glm5NextForConditionalGeneration):
            model = Glm5NextForConditionalGeneration.from_pretrained(
                source,
                dtype=dtype,
                device_map=device_map,
                offload_folder=offload_folder,
            )
        return linearize_moe(model)
    from llmcompressor.utils import load_context

    with load_context(Glm5NextForConditionalGeneration, load_mtp=False):
        return Glm5NextForConditionalGeneration.from_pretrained(
            source,
            dtype=dtype,
            device_map=device_map,
            offload_folder=offload_folder,
        )


def calibrate(
    model,
    loader,
    groups,
    *,
    processor=None,
    smoothquant=False,
    alpha=0.5,
    targets_per_graph=1,
    dampening_frac=0.01,
    routed_method="gptq",
):
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    if routed_method not in ("gptq", "minmax"):
        raise ValueError("routed-method must be gptq or minmax")
    if routed_method == "minmax" and not smoothquant:
        raise ValueError("The calibrated MinMax candidate requires --smoothquant")
    model.eval()
    model.config.get_text_config().use_cache = False
    modules = quantized_modules(model, {"routed": groups["routed"]})
    coverage = {name: 0 for name in modules}
    handles = []

    def counter(name):
        def hook(_module, inputs):
            coverage[name] += inputs[0].numel() // inputs[0].shape[-1]

        return hook

    calibration_args = dict(
        model=model,
        processor=processor,
        dataset=loader,
        pipeline="sequential",
        sequential_targets=[
            "Glm5NextTextDecoderLayer",
            "Glm5NextVisionBlock",
            "Glm5NextVisionPatchMerger",
        ],
        sequential_offload_device="cpu",
        moe_calibrate_all_experts=True,
        tie_word_embeddings=False,
    )
    gptq = None
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(counter(name)))
        if smoothquant:
            # Finish the float preprocessing pass before initializing quantizers.
            oneshot(
                **calibration_args,
                recipe=SmoothQuantModifier(
                    mappings=make_smoothquant_mappings(model), smoothing_strength=alpha
                ),
                sequential_targets_per_subgraph=1,
            )
        if routed_method == "gptq":
            gptq = GPTQModifier(
                config_groups={"routed": groups["routed"]},
                actorder=None,
                block_size=128,
                dampening_frac=dampening_frac,
                batched_quantization=None,
            )
            oneshot(
                **calibration_args,
                recipe=gptq,
                sequential_targets_per_subgraph=targets_per_graph,
            )
    finally:
        for handle in handles:
            handle.remove()
    missing = [name for name, count in coverage.items() if count == 0]
    if missing:
        raise ValueError(f"Uncalibrated projections: {missing[:10]}")
    fallback = list(gptq._rtn_fallback_module_names) if gptq is not None else []
    if fallback:
        raise ValueError(
            f"GPTQ Hessian factorization fell back to RTN: {fallback[:10]}"
        )
    # SQ+MinMax allocates no Hessians. GPTQ leaves only the W8 groups for MinMax.
    minmax_groups = (
        groups
        if routed_method == "minmax"
        else {key: groups[key] for key in ("dense", "shared")}
    )
    oneshot(
        model=model,
        processor=processor,
        recipe=QuantizationModifier(config_groups=minmax_groups),
        tie_word_embeddings=False,
    )
    return coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash-BF16")
    parser.add_argument(
        "--revision", default="f9fed3bbc82291073373d67452e9f25317ef7577"
    )
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--save-dir", type=Path, default=Path("GLM53-Flash-W4A8-GPTQ"))
    parser.add_argument("--device", choices=("cpu", "npu:0", "cuda:0"), default="npu:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoothquant", action="store_true")
    parser.add_argument("--routed-method", choices=("gptq", "minmax"), default="gptq")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--targets-per-graph", type=int, default=1)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--allow-text-only", action="store_true")
    parser.add_argument("--linearize-after-load", action="store_true")
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.routed_method == "minmax" and not args.smoothquant:
        parser.error("--routed-method minmax requires --smoothquant")
    if (
        min(args.num_calibration_samples, args.max_seq_length, args.targets_per_graph)
        < 1
    ):
        parser.error(
            "sample count, sequence length and targets-per-graph must be positive"
        )
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must be between zero and one")
    if not 0 < args.dampening_frac < 1:
        parser.error("dampening-frac must be between zero and one")
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    config_groups=make_config_groups(45),
                    smoothquant=args.smoothquant,
                    routed_method=args.routed_method,
                    mtp="float",
                    cache="runtime-int8",
                ),
                indent=2,
            )
        )
        return
    if not args.calibration_data:
        parser.error("--calibration-data is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error(
            "This example is single-process; inference TP is configured separately"
        )
    records = load_records(
        args.calibration_data,
        args.num_calibration_samples,
        args.seed,
        args.allow_text_only,
    )
    source = Path(args.model)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(args.model, revision=args.revision))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config") or config.get("text_config", {}).get(
        "quantization_config"
    ):
        raise ValueError(
            "Start from the original BF16 checkpoint, not quantized weights"
        )
    output = args.save_dir.resolve()
    if source.resolve().is_relative_to(output) or output.is_relative_to(
        source.resolve()
    ):
        raise ValueError("Keep source and output in separate directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh empty save-dir")
    import torch

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    from transformers import AutoProcessor, set_seed

    if args.device == "cpu" and torch.accelerator.is_available():
        raise ValueError(
            "CPU probes require a CPU-only runtime; hide accelerators first"
        )
    set_seed(args.seed)
    offload = (args.offload_dir or output.with_name(output.name + "-offload")).resolve()
    if (
        offload.is_relative_to(source.resolve())
        or source.resolve().is_relative_to(offload)
        or offload.is_relative_to(output)
        or output.is_relative_to(offload)
    ):
        raise ValueError("Keep offload-dir separate from source and save-dir")
    offload.mkdir(parents=True, exist_ok=True)
    model = load_model(
        source,
        getattr(torch, args.dtype),
        linearize_after_load=args.linearize_after_load,
        offload_folder=str(offload),
    )
    processor = AutoProcessor.from_pretrained(source)
    groups = make_config_groups(model.config.get_text_config().num_hidden_layers)
    loader = make_dataloader(records, processor, args.max_seq_length)
    coverage = calibrate(
        model,
        loader,
        groups,
        processor=processor,
        smoothquant=args.smoothquant,
        alpha=args.alpha,
        targets_per_graph=args.targets_per_graph,
        dampening_frac=args.dampening_frac,
        routed_method=args.routed_method,
    )
    model.save_pretrained(output, save_compressed=True, max_shard_size="4GB")
    processor.save_pretrained(output)
    report = dict(
        algorithm=("smoothquant+" if args.smoothquant else "") + args.routed_method,
        config_groups=groups,
        mtp="preserved-float-not-calibrated",
        cache="runtime-dynamic-int8",
        calibration_file=str(args.calibration_data.resolve()),
        calibration_sha256=hashlib.sha256(
            args.calibration_data.read_bytes()
        ).hexdigest(),
        selected_lines=[record[2] for record in records],
        image_samples=sum(record[1] for record in records),
        forward_token_counts_including_replay=coverage,
        alpha=args.alpha if args.smoothquant else None,
        seed=args.seed,
        dampening_frac=args.dampening_frac,
        source_model=str(source.resolve()),
        source_config_sha256=hashlib.sha256(
            (source / "config.json").read_bytes()
        ).hexdigest(),
        example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dtype=args.dtype,
        device=args.device,
        max_seq_length=args.max_seq_length,
        targets_per_graph=args.targets_per_graph,
    )
    (output / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
