"""Experiment B: native routed-only AWQ + MinMax for GLM-5.3-Flash CT W4A8.

No SmoothQuant, GPTQ, rotation, framework patches or runtime monkey-patches.
Keep the GPTQ example and CT config auditor beside this file: their proven
GLM loader, data handling, scope and export checks are shared unchanged.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path

from glm53_flash_ct_config import audit_config
from glm53_flash_w4a8c8_gptq_example import (
    load_model,
    load_records,
    make_config_groups,
    make_dataloader,
    make_smoothquant_mappings,
    quantized_modules,
)


def make_awq_mappings(model):
    """Reuse topology validation only; no SmoothQuantModifier is instantiated.

    Every MoE norm consumer is compensated, including floating shared/router.
    Only routed experts receive a quantization scheme during the AWQ pass.
    """
    from llmcompressor.modifiers.transform.awq import AWQMapping

    return [
        AWQMapping(smooth_layer=norm, balance_layers=consumers)
        for consumers, norm in make_smoothquant_mappings(model)
        if any(".experts." in name for name in consumers)
    ]


def calibrate_awq(model, loader, *, processor, n_grid=20, duo_scaling=False):
    """Use two native oneshot calls, keeping AWQ's search uniformly INT4."""
    from loguru import logger

    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    if n_grid < 4 or (duo_scaling == "both" and n_grid % 2):
        raise ValueError("n_grid must be >= 4, and even for duo_scaling='both'")
    if any(getattr(m, "quantization_scheme", None) for m in model.modules()):
        raise ValueError("Start AWQ from the original floating model")
    model.eval()
    model.config.get_text_config().use_cache = False
    groups = make_config_groups(model.config.get_text_config().num_hidden_layers)
    mappings = make_awq_mappings(model)
    if not mappings:
        raise ValueError("No backbone MoE mappings found")
    modules = quantized_modules(model, {"routed": groups["routed"]})
    coverage = {name: 0 for name in modules}
    handles = []
    native_metrics = []

    def counter(name):
        def hook(_module, inputs):
            coverage[name] += inputs[0].numel() // inputs[0].shape[-1]

        return hook

    # Native AWQ clears its private metrics on finalization. Capture its existing
    # debug summary without subclassing or replacing the search implementation.
    def capture_metrics(message):
        prefix = "AWQ per-mapping error metrics: "
        text = message.record["message"]
        if text.startswith(prefix):
            native_metrics.extend(ast.literal_eval(text[len(prefix) :])["metrics"])

    sink = logger.add(capture_metrics, level="DEBUG", catch=False)
    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(counter(name)))
        oneshot(
            model=model,
            processor=processor,
            dataset=loader,
            recipe=[
                AWQModifier(
                    mappings=mappings,
                    duo_scaling=duo_scaling,
                    n_grid=n_grid,
                    offload_device="cpu",
                ),
                QuantizationModifier(config_groups={"routed": groups["routed"]}),
            ],
            pipeline="sequential",
            sequential_targets=[
                "Glm5NextTextDecoderLayer",
                "Glm5NextVisionBlock",
                "Glm5NextVisionPatchMerger",
            ],
            sequential_targets_per_subgraph=1,
            sequential_offload_device="cpu",
            moe_calibrate_all_experts=True,
            tie_word_embeddings=False,
        )
    finally:
        logger.remove(sink)
        for handle in handles:
            handle.remove()
    missing = [name for name, count in coverage.items() if count == 0]
    if missing:
        raise ValueError(f"Uncalibrated routed projections: {missing[:10]}")
    expected = {mapping.smooth_layer for mapping in mappings}
    searched = [metric["layer_name"] for metric in native_metrics]
    if set(searched) != expected or len(searched) != len(expected):
        raise ValueError("Native AWQ did not report exactly one search per MoE norm")
    if any(
        not math.isfinite(metric[key])
        for metric in native_metrics
        for key in ("initial_error", "best_error")
    ):
        raise ValueError("Non-finite AWQ reconstruction errors")

    # Shared/router were floating while AWQ evaluated each MoE. Their weights
    # have already received the same equivalence compensation as the experts.
    # Only now initialize W8 scales from the compensated dense/shared weights.
    oneshot(
        model=model,
        processor=processor,
        recipe=QuantizationModifier(
            config_groups={name: groups[name] for name in ("dense", "shared")}
        ),
        tie_word_embeddings=False,
    )
    return dict(
        awq_scope="routed-MoE norm-to-gate/up; shared/router compensated",
        search_quantized_bits=[4],
        n_grid=n_grid,
        duo_scaling=duo_scaling,
        per_mapping_errors=native_metrics,
        forward_token_counts_including_replay=coverage,
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="zai-org/GLM-5.3-Flash-BF16")
    parser.add_argument(
        "--revision", default="f9fed3bbc82291073373d67452e9f25317ef7577"
    )
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument(
        "--save-dir", type=Path, default=Path("GLM53-Flash-CT-AWQ-V256")
    )
    parser.add_argument("--device", choices=("npu:0", "cuda:0"), default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--num-calibration-samples", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument(
        "--duo-scaling", choices=("false", "true", "both"), default="false"
    )
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--linearize-after-load", action="store_true")
    parser.add_argument("--allow-text-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if min(args.num_calibration_samples, args.max_seq_length) < 1:
        parser.error("sample count and sequence length must be positive")
    if args.n_grid < 4 or (args.duo_scaling == "both" and args.n_grid % 2):
        parser.error("n-grid must be >= 4, and even for duo-scaling=both")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Calibration is single-process; inference TP is separate")
    if args.plan_only:
        source_config = Path(args.model) / "config.json"
        config = (
            json.loads(source_config.read_text(encoding="utf-8"))
            if source_config.is_file()
            else {}
        )
        num_layers = config.get("text_config", config).get("num_hidden_layers", 45)
        print(
            json.dumps(
                dict(
                    experiment="B",
                    algorithm="awq+minmax",
                    smoothquant=False,
                    config_groups=make_config_groups(num_layers),
                    stage1="AWQ+routed W4A8; shared/router floating compensation",
                    stage2="MinMax dense/shared W8A8",
                    mtp="preserved-float",
                    cache="runtime-dynamic-int8",
                    n_grid=args.n_grid,
                    duo_scaling=args.duo_scaling,
                ),
                indent=2,
            )
        )
        return
    if not args.calibration_data:
        parser.error("--calibration-data is required")
    source = Path(args.model)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(snapshot_download(args.model, revision=args.revision))
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config") or config.get("text_config", {}).get(
        "quantization_config"
    ):
        parser.error(
            "Start from original BF16 weights, not an existing SQ/quantized output"
        )
    records = load_records(
        args.calibration_data,
        args.num_calibration_samples,
        args.seed,
        args.allow_text_only,
    )
    if len(records) != args.num_calibration_samples:
        parser.error("Calibration JSONL has fewer samples than requested")
    from transformers import AutoProcessor, set_seed

    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(source)
    loader = make_dataloader(records, processor, args.max_seq_length)
    # Check real image expansion before loading hundreds of GB of model weights.
    lengths = [int(batch["input_ids"].shape[-1]) for batch in loader]
    preflight = dict(
        samples=len(records),
        image_samples=sum(record[1] for record in records),
        min_processed_tokens=min(lengths),
        max_processed_tokens=max(lengths),
        total_processed_tokens=sum(lengths),
        selected_lines=[record[2] for record in records],
        calibration_sha256=_sha256(args.calibration_data),
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    output = args.save_dir.resolve()
    offload = (args.offload_dir or output.with_name(output.name + "-offload")).resolve()
    folders = [source.resolve(), output, offload]
    for i, left in enumerate(folders):
        for right in folders[i + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                parser.error("Keep BF16 source, save-dir and offload-dir separate")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh empty save-dir")
    import torch

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    set_seed(args.seed)
    offload.mkdir(parents=True, exist_ok=True)
    model = load_model(
        source,
        getattr(torch, args.dtype),
        linearize_after_load=args.linearize_after_load,
        offload_folder=str(offload),
    )
    duo_scaling = {"false": False, "true": True, "both": "both"}[args.duo_scaling]
    search = calibrate_awq(
        model, loader, processor=processor, n_grid=args.n_grid, duo_scaling=duo_scaling
    )
    # GLM can load sampling fields without do_sample and warn, but saving
    # validates them strictly. Preserve those values by enabling sampling.
    gc = model.generation_config
    if (gc.top_p is not None or gc.temperature is not None) and not gc.do_sample:
        gc.do_sample = True
    model.save_pretrained(output, save_compressed=True, max_shard_size="4GB")
    export_audit = audit_config(output)
    processor.save_pretrained(output)
    report = dict(
        experiment="B",
        algorithm="awq+minmax",
        smoothquant=False,
        calibration_file=str(args.calibration_data.resolve()),
        **preflight,
        **search,
        export_audit=export_audit,
        source_model=str(source.resolve()),
        source_config_sha256=_sha256(source / "config.json"),
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        max_seq_length=args.max_seq_length,
        mtp="preserved-float-not-calibrated",
        cache="runtime-dynamic-int8",
        script_sha256={
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                Path(__file__).name,
                "glm53_flash_w4a8c8_gptq_example.py",
                "glm53_flash_ct_config.py",
            )
        },
    )
    (output / "calibration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
