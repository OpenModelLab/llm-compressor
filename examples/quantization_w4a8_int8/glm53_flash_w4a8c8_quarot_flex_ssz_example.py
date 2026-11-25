"""GLM53 backbone: offline QuaRot -> FlexSmooth -> MinMax/SSZ/MinMax.

Reuses native llm-compressor oneshot, Hadamard, FlexSmooth search, SSZ and CT
export. GLM53 topology glue lives beside this example, not in framework src/.
MTP is NOT yet adapted; an MTP source requires explicit --disable-mtp. This is
an algorithm-chain-aligned backbone experiment, not full ModelSlim MTP parity.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter


def read_raw_calibration(path, limit=None):
    """ModelSlim-compatible raw strings, original file order, no chat template."""
    text = path.read_text(encoding="utf-8")
    records = (
        json.loads(text)
        if path.suffix.lower() == ".json"
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, str) or not record.strip() for record in records)
    ):
        raise ValueError("Raw calibration must contain nonempty JSON strings")
    return records if limit is None else records[:limit]


def raw_dataloader(records, tokenizer, max_seq_length):
    from torch.utils.data import DataLoader, Dataset

    class RawDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            encoded = dict(tokenizer(records[index], return_tensors="pt", padding=True))
            if encoded["input_ids"].shape[-1] > max_seq_length:
                raise ValueError(
                    f"Raw calibration record {index + 1} exceeds length limit"
                )
            return encoded

    return DataLoader(RawDataset(), batch_size=None, num_workers=0)


def preprocess(model, processor, loader, *, loss_chunk_tokens=4096):
    """Separate lifecycles guarantee collection uses the rotated model."""
    from glm53_flash_preprocessing import (
        Glm53FlexSmoothModifier,
        Glm53QuaRotModifier,
        build_glm53_plans,
    )

    from llmcompressor import oneshot

    build_glm53_plans(model)  # complete geometry check before the first mutation
    rotation = Glm53QuaRotModifier()
    started = perf_counter()
    oneshot(
        model=model,
        processor=processor,
        recipe=rotation,
        pipeline="datafree",
        tie_word_embeddings=False,
    )
    timings = {"quarot_seconds": perf_counter() - started}
    matrices = {
        name: matrix.to_dense().cpu() for name, matrix in rotation.matrices.items()
    }
    smoothing = Glm53FlexSmoothModifier(loss_chunk_tokens=loss_chunk_tokens)
    started = perf_counter()
    oneshot(
        model=model,
        processor=processor,
        recipe=smoothing,
        dataset=loader,
        pipeline="sequential",
        sequential_targets=["Glm5NextTextDecoderLayer"],
        sequential_targets_per_subgraph=1,
        sequential_offload_device="cpu",
        tie_word_embeddings=False,
    )
    timings["flex_smooth_seconds"] = perf_counter() - started
    return matrices, smoothing.diagnostics, timings


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--num-calibration-samples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--flex-loss-chunk-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "npu:0"), default="npu:0")
    parser.add_argument("--linearize-after-load", action="store_true")
    parser.add_argument("--disable-mtp", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This example is single-process")
    if (
        args.step < 1
        or args.flex_loss_chunk_tokens < 1
        or args.max_seq_length < 1
        or (
            args.num_calibration_samples is not None
            and args.num_calibration_samples < 1
        )
    ):
        parser.error("step, length and sample limit must be positive")
    source = args.model.resolve()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    for owner in (config, text_config):
        if any(
            owner.get(key) is not None
            for key in (
                "quantization_config",
                "quarot_config",
                "flex_smooth_config",
            )
        ):
            parser.error("Start from original, unquantized and untransformed BF16")
    mtp = text_config.get("num_nextn_predict_layers", 0) or 0
    if mtp and not args.disable_mtp:
        parser.error(
            "MTP is not adapted. Use --disable-mtp for a backbone experiment, "
            "or the original ModelSlim recipe for MTP parity"
        )
    from glm53_flash_w4a8c8_ssz_example import expected_mlp_projections, make_ssz_groups

    expected_mlp_projections(config)
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    stages=[
                        "offline_quarot",
                        "flex_smooth_quant:norm-linear:auto-alpha-beta",
                        "dense:minmax:W8A8",
                        "routed:ssz:W4A8",
                        "shared:minmax:W8A8",
                    ],
                    rotation_seed=1234,
                    rotation_precision="float32",
                    block_size=None,
                    online=False,
                    down_proj_online_layers=[],
                    config_groups=make_ssz_groups(
                        text_config["num_hidden_layers"], args.step
                    ),
                    calibration_protocol="raw strings -> tokenizer; no chat template",
                    full_modelslim_recipe=False,
                    alignment_scope="backbone algorithms",
                    mtp="disabled" if mtp else "absent",
                ),
                indent=2,
            )
        )
        return
    if not args.save_dir or not args.calibration_file:
        parser.error("--save-dir and --calibration-file are required")
    records = read_raw_calibration(args.calibration_file, args.num_calibration_samples)
    output = args.save_dir.resolve()
    offload = (args.offload_dir or output.with_name(output.name + "-offload")).resolve()
    folders = (source, output, offload)
    for index, left in enumerate(folders):
        for right in folders[index + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                parser.error("Keep source/output/offload directories separate")
    for folder in (output, offload):
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            parser.error(f"Use a fresh empty output/offload directory: {folder}")

    import torch
    from safetensors.torch import save_file
    from transformers import AutoProcessor, GenerationConfig, set_seed

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    elif torch.accelerator.is_available():
        parser.error(
            "CPU mode requires accelerators hidden; avoid implicit device selection"
        )
    from glm53_flash_w4a8c8_gptq_example import load_model
    from glm53_flash_w4a8c8_ssz_example import (
        build_modifiers,
        environment_report,
        prepare_generation_config,
        quantize_mlp,
        validate_export,
    )

    build_modifiers(
        make_ssz_groups(text_config["num_hidden_layers"], args.step), args.step
    )
    if (source / "generation_config.json").exists():
        prepare_generation_config(GenerationConfig.from_pretrained(source))
    set_seed(42)
    processor = AutoProcessor.from_pretrained(source)
    tokenizer = getattr(processor, "tokenizer", processor)
    loader = raw_dataloader(records, tokenizer, args.max_seq_length)
    # Validate all input lengths/tokenization before modifying model weights.
    token_hash = hashlib.sha256()
    for batch in loader:
        for key in ("input_ids", "attention_mask"):
            token_hash.update(key.encode())
            token_hash.update(batch[key].contiguous().numpy().tobytes())
    offload.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    model = load_model(
        source,
        torch.bfloat16,
        linearize_after_load=args.linearize_after_load,
        offload_folder=str(offload),
        device_map={"": "cpu"} if args.device == "cpu" else "auto_offload",
    )
    load_seconds = perf_counter() - started
    prepare_generation_config(model.generation_config)
    model.eval()
    model.config.get_text_config().use_cache = False
    model.config.get_text_config().num_nextn_predict_layers = 0
    matrices, diagnostics, timings = preprocess(
        model, processor, loader, loss_chunk_tokens=args.flex_loss_chunk_tokens
    )
    result = quantize_mlp(model, processor, args.step)
    started = perf_counter()
    model.save_pretrained(output, save_compressed=True, max_shard_size="4GB")
    audit = validate_export(output, result, repair=True)
    processor.save_pretrained(output)
    # These are reproducibility assets, not additional checkpoint model keys.
    # Backbone transforms are fully folded into ordinary weights and norm gains.
    save_file(matrices, output / "glm53_rotation.safetensors")
    scales = {name: details["scale"] for name, details in diagnostics.items()}
    save_file(scales, output / "glm53_flex_scales.safetensors")
    report = dict(
        completion_status="complete",
        full_modelslim_recipe=False,
        algorithm_chain_complete=True,
        alignment_scope="GLM53 backbone",
        algorithm="offline QuaRot -> FlexSmooth -> MinMax/SSZ/MinMax",
        mtp="disabled",
        source_mtp_layers=mtp,
        runtime_cache="dynamic-int8",
        rotation=dict(seed=1234, precision="float32", online=False, block_size=None),
        flex_smooth=dict(
            subgraphs=["norm-linear"],
            max_tokens=None,
            loss_chunk_tokens=args.flex_loss_chunk_tokens,
            results={
                name: {k: v for k, v in detail.items() if k != "scale"}
                for name, detail in diagnostics.items()
            },
        ),
        calibration=dict(
            file=str(args.calibration_file.resolve()),
            sha256=hashlib.sha256(args.calibration_file.read_bytes()).hexdigest(),
            samples=len(records),
            max_seq_length=args.max_seq_length,
            input_ids_and_mask_sha256=token_hash.hexdigest(),
            protocol="raw strings; tokenizer; original order; no template",
        ),
        source_model=str(source),
        source_files_sha256={
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
            )
            if (source / name).exists()
        },
        source_weight_shards_content_hashed=False,
        environment=environment_report(),
        load_path="linearize-after-load"
        if args.linearize_after_load
        else "load-context",
        load_seconds=load_seconds,
        save_and_audit_seconds=perf_counter() - started,
        export_audit=audit,
        **timings,
        **result,
        script_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                Path(__file__).name,
                "glm53_flash_preprocessing.py",
                "glm53_flash_w4a8c8_ssz_example.py",
                "glm53_flash_w4a8c8_gptq_example.py",
                "glm53_flash_ct_config.py",
            )
        },
    )
    pending = output / "calibration_report.json.tmp"
    pending.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False), encoding="utf-8"
    )
    pending.replace(output / "calibration_report.json")


if __name__ == "__main__":
    main()
