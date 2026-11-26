"""GLM-5.3-Flash SSZ control: routed W4A8, dense/shared W8A8 MinMax.

This is the quantization-only control for the reference recipe, not its full
QuaRot/FlexSmooth reproduction. No activation samples or forward passes are used.
Keep the GPTQ loader and CT metadata auditor in this directory.
For the preprocessing chain, see glm53_flash_w4a8c8_quarot_flex_ssz_example.py;
that separate entrypoint includes QuaRot and forward-calibrated FlexSmooth.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from time import perf_counter

from glm53_flash_ct_config import AUDIT_CONTRACT_VERSION, audit_config
from glm53_flash_w4a8c8_gptq_example import (
    load_model,
    make_config_groups,
    quantized_modules,
)


def make_ssz_groups(num_layers, step):
    groups = make_config_groups(num_layers)
    groups["routed"]["weights"].update(observer="ssz", observer_kwargs={"step": step})
    return groups


def expected_mlp_projections(config):
    """Derive the complete backbone scope independently of regex matching."""
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("Expected a GLM text config")
    layers = text.get("num_hidden_layers")
    types = text.get("mlp_layer_types")
    if (
        not isinstance(layers, int)
        or isinstance(layers, bool)
        or layers < 1
        or not isinstance(types, (list, tuple))
        or len(types) != layers
        or any(kind not in ("dense", "sparse") for kind in types)
    ):
        raise ValueError("Expected one dense/sparse mlp_layer_types entry per layer")
    experts, shared = text.get("n_routed_experts"), text.get("n_shared_experts")
    if "sparse" in types and any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in (experts, shared)
    ):
        raise ValueError("Sparse GLM layers require positive routed/shared counts")
    expected = {}
    for index, kind in enumerate(types):
        prefix = f"model.language_model.layers.{index}.mlp"
        stems = (
            [(prefix, "dense")]
            if kind == "dense"
            else [
                *[(f"{prefix}.experts.{i}", "routed") for i in range(experts)],
                (f"{prefix}.shared_experts", "shared"),
            ]
        )
        for stem, family in stems:
            for projection in ("gate", "up", "down"):
                expected[f"{stem}.{projection}_proj"] = family
    return expected


def build_modifiers(groups, step):
    from llmcompressor.modifiers.quantization import QuantizationModifier, SSZModifier

    return {
        family: SSZModifier(config_groups={family: groups[family]}, step=step)
        if family == "routed"
        else QuantizationModifier(config_groups={family: groups[family]})
        for family in ("dense", "routed", "shared")
    }


def prepare_generation_config(config):
    # Preserve GLM sampling fields while making strict save validation succeed.
    if (
        config.top_p is not None or config.temperature is not None
    ) and not config.do_sample:
        config.do_sample = True
    config.validate(strict=True)


def environment_report():
    versions = {}
    for package in (
        "llmcompressor",
        "compressed-tensors",
        "transformers",
        "torch",
        "torch-npu",
        "torchvision",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(
        python=sys.version,
        packages=versions,
        imported_modules={
            name: str(importlib.import_module(name).__file__)
            for name in ("llmcompressor", "compressed_tensors", "transformers", "torch")
        },
    )


def validate_export(output, scope, *, repair=False):
    audit = audit_config(output, repair=repair)
    if (
        any(
            audit["projections"].get(name, 0) != count
            for name, count in scope["projection_counts"].items()
        )
        or audit.get("projection_names_sha256") != scope["projection_names_sha256"]
    ):
        raise ValueError("Exported MLP scope differs from the validated source scope")
    return audit


def quantize_mlp(model, processor, step=50, *, strict_recipe=False):
    """Apply the YAML's three weight quantizers through native data-free oneshot."""
    from compressed_tensors.offload import OffloadCache

    from llmcompressor import oneshot

    if step < 1:
        raise ValueError("step must be positive")
    if any(getattr(m, "quantization_scheme", None) for m in model.modules()):
        raise ValueError("Start from an unquantized floating model")
    groups = make_ssz_groups(model.config.get_text_config().num_hidden_layers, step)
    if strict_recipe:
        from glm53_flash_quantization import strict_groups

        groups = strict_groups(model.config.get_text_config().num_hidden_layers, step)
    expected = expected_mlp_projections(model.config.to_dict())
    modules = quantized_modules(model, groups)
    missing, extra = (
        sorted(expected.keys() - modules.keys()),
        sorted(modules.keys() - expected.keys()),
    )
    if missing or extra:
        raise ValueError(
            f"Incomplete MLP scope: missing={missing[:5]}, extra={extra[:5]}"
        )
    with OffloadCache.disable_onloading():
        for name, module in modules.items():
            if module.weight.ndim != 2 or not module.weight.is_floating_point():
                raise ValueError(f"Expected floating 2D source projection: {name}")
    counts = {name: 0 for name in groups}
    for name in modules:
        family = (
            "routed"
            if ".experts." in name
            else "shared"
            if ".shared_experts." in name
            else "dense"
        )
        counts[family] += 1
    timings = {}
    if strict_recipe:
        from llmcompressor.modifiers.quantization import QuantizationModifier

        modifiers = {
            family: QuantizationModifier(config_groups={family: groups[family]})
            for family in groups
        }
    else:
        modifiers = build_modifiers(groups, step)
    model.eval()
    for family in ("dense", "routed", "shared"):
        if not counts[family]:
            timings[family] = 0.0
            continue
        started = perf_counter()
        print(f"Starting {family}: {counts[family]} projections", flush=True)
        oneshot(
            model=model,
            processor=processor,
            recipe=modifiers[family],
            pipeline="datafree",
            tie_word_embeddings=False,
        )
        timings[family] = perf_counter() - started
        print(f"Finished {family}: {timings[family]:.1f}s", flush=True)
    return dict(
        projection_counts=counts,
        quantization_seconds=timings,
        projection_names_sha256=hashlib.sha256(
            "\n".join(sorted(expected)).encode()
        ).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--offload-dir", type=Path)
    parser.add_argument("--device", choices=("npu:0", "cuda:0", "cpu"), default="npu:0")
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linearize-after-load", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if AUDIT_CONTRACT_VERSION < 2:
        parser.error("Update glm53_flash_ct_config.py with the SSZ example")
    if args.step < 1:
        parser.error("step must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This example is single-process; inference TP is separate")
    if args.plan_only:
        assumed = args.model is None
        if args.model is not None:
            config = json.loads(
                (args.model / "config.json").read_text(encoding="utf-8")
            )
            expected_mlp_projections(config)
            layers = config.get("text_config", config)["num_hidden_layers"]
        else:
            layers = 45
        print(
            json.dumps(
                dict(
                    algorithm="SSZ-only control; no QuaRot/FlexSmooth",
                    config_groups=make_ssz_groups(layers, args.step),
                    assumed_45_layer_plan=assumed,
                    mtp="preserved-float",
                    cache="runtime-dynamic-int8",
                    calibration_data_required=False,
                ),
                indent=2,
            )
        )
        return
    if not args.model or not args.model.is_dir() or not args.save_dir:
        parser.error("--model must be a local BF16 directory; --save-dir is required")
    source, output = args.model.resolve(), args.save_dir.resolve()
    offload = (args.offload_dir or output.with_name(output.name + "-offload")).resolve()
    folders = (source, output, offload)
    for i, left in enumerate(folders):
        for right in folders[i + 1 :]:
            if left.is_relative_to(right) or right.is_relative_to(left):
                parser.error("Keep source, output and offload directories separate")
    for folder in (output, offload):
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            parser.error(f"Use a fresh empty output/offload directory: {folder}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config") or config.get("text_config", {}).get(
        "quantization_config"
    ):
        parser.error("Use original BF16 weights, not an existing quantized model")
    if any(
        owner.get(key) is not None
        for owner in (config, config.get("text_config", {}))
        for key in ("quarot_config", "flex_smooth_config")
    ):
        parser.error("The SSZ-only control requires untransformed BF16 weights")
    expected_mlp_projections(config)

    import torch
    from transformers import AutoProcessor, GenerationConfig, set_seed

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    elif args.device.startswith("cuda"):
        torch.accelerator.set_device_index(args.device)
    set_seed(args.seed)

    # Fail on missing SSZ support or invalid recipes before loading all weights.
    build_modifiers(
        make_ssz_groups(
            config.get("text_config", config)["num_hidden_layers"], args.step
        ),
        args.step,
    )
    if (source / "generation_config.json").exists():
        prepare_generation_config(GenerationConfig.from_pretrained(source))
    source_files = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        )
        if (source / name).exists()
    }
    runtime = environment_report()

    processor = AutoProcessor.from_pretrained(source)
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
    result = quantize_mlp(model, processor, args.step)
    started = perf_counter()
    model.save_pretrained(output, save_compressed=True, max_shard_size="4GB")
    # The reused builder can emit HF-only targets. Repair metadata explicitly;
    # preserve all integer codes/scales and validate HF + runtime coverage.
    export_audit = validate_export(output, result, repair=True)
    processor.save_pretrained(output)
    report = dict(
        completion_status="complete",
        algorithm="SSZ routed + MinMax dense/shared",
        full_reference_recipe=False,
        quarot=False,
        flex_smooth=False,
        calibration_data_required=False,
        step=args.step,
        seed=args.seed,
        source_model=str(source),
        load_path="linearize-after-load"
        if args.linearize_after_load
        else "load-context",
        source_files_sha256=source_files,
        source_weight_shards_content_hashed=False,
        environment=runtime,
        mtp="preserved-float-not-calibrated",
        cache="runtime-dynamic-int8",
        load_seconds=load_seconds,
        save_and_audit_seconds=perf_counter() - started,
        export_audit=export_audit,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "glm53_flash_w4a8c8_gptq_example.py",
                "glm53_flash_ct_config.py",
            )
        },
        **result,
    )
    pending_report = output / "calibration_report.json.tmp"
    pending_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pending_report.replace(output / "calibration_report.json")


if __name__ == "__main__":
    main()
