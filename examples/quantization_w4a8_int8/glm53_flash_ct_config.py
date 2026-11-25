"""Audit/repair GLM backbone MLP CT metadata without loading weight tensors.

Use with the forward-calibrated example: routed INT4, dense/shared INT8,
floating MTP. Requires safetensors; no NPU or llmcompressor import is needed.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_PROJECTION = re.compile(
    r"^(?:model\.language_model|language_model\.model)\.layers\.(\d+)\.mlp\."
    r"(?:(experts\.\d+|shared_experts)\.)?(gate|up|down)_proj$"
)
_FLOAT = {"BF16", "F16", "F32", "F64"}
AUDIT_CONTRACT_VERSION = 2


def _family(stem):
    kind = _PROJECTION.fullmatch(stem)[2]
    if kind is None:
        return "dense"
    return "shared" if kind == "shared_experts" else "routed"


def _aliases(name):
    hf = name.replace("language_model.model.", "model.language_model.", 1)
    return hf, hf.replace("model.language_model.", "language_model.model.", 1)


def _matches(name, targets):
    return any(
        re.match(target[3:], name) if target.startswith("re:") else target == name
        for target in targets
    )


def _metadata(folder):
    from safetensors import safe_open

    index_path = folder / "model.safetensors.index.json"
    index = None
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted({folder / shard for shard in index.values()})
    else:
        shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise ValueError("No safetensors checkpoint found")
    tensors = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in tensors:
                    raise ValueError(f"Duplicate checkpoint key: {name}")
                if index is not None and index.get(name) != shard.name:
                    raise ValueError(f"Index/shard mismatch: {name}")
                view = handle.get_slice(name)
                tensors[name] = (view.get_dtype(), tuple(view.get_shape()))
    if index is not None and set(index) != set(tensors):
        raise ValueError("Safetensors index and shard keys differ")
    return tensors


def _check_scheme(group, bits):
    weight = group.get("weights") or {}
    activation = group.get("input_activations") or {}
    expected_format = "pack-quantized" if bits == 4 else "int-quantized"
    if not (
        group.get("format") == expected_format
        and weight.get("num_bits") == bits
        and weight.get("type") == activation.get("type") == "int"
        and weight.get("strategy") == "channel"
        and weight.get("symmetric", True)
        and not weight.get("dynamic", False)
        and activation.get("num_bits") == 8
        and activation.get("strategy") == "token"
        and activation.get("symmetric", True)
        and activation.get("dynamic") is True
    ):
        raise ValueError(f"Expected per-channel INT{bits}/dynamic-token INT8")


def _projections(tensors, num_layers):
    projections = {}
    for name, metadata in tensors.items():
        stem, _, suffix = name.rpartition(".")
        match = _PROJECTION.fullmatch(stem)
        if match is None:
            if suffix in {
                "weight_packed",
                "weight_shape",
                "weight_scale",
                "weight_scale_inv",
            } or (suffix == "weight" and metadata[0] not in _FLOAT):
                raise ValueError(f"This repair requires MLP-only quantization: {name}")
            continue
        if int(match[1]) >= num_layers:
            if suffix != "weight" or metadata[0] not in _FLOAT:
                raise ValueError(
                    f"This backbone-only recipe requires floating MTP: {name}"
                )
            continue
        if suffix in {"weight", "weight_packed", "weight_shape", "weight_scale"}:
            projections.setdefault(stem, {})[suffix] = metadata
    counts = Counter()
    for stem, entries in projections.items():
        family = _family(stem)
        dtype = "I32" if family == "routed" else "I8"
        data_key = "weight_packed" if family == "routed" else "weight"
        data = entries.get(data_key)
        scale = entries.get("weight_scale")
        if data is None or data[0] != dtype or len(data[1]) != 2:
            raise ValueError(f"Missing/invalid {dtype} {data_key}: {stem}")
        if scale is None or scale[0] not in _FLOAT or scale[1] != (data[1][0], 1):
            raise ValueError(f"Missing/invalid per-channel weight_scale: {stem}")
        if family == "routed" and (
            entries.get("weight_shape") != ("I64", (2,)) or "weight" in entries
        ):
            raise ValueError(
                f"Expected packed-only routed weights with shape metadata: {stem}"
            )
        counts[family] += 1
    if not projections or not counts["routed"]:
        raise ValueError("No backbone routed packed projections found")
    return projections, counts


def _repair_groups(quantization, projections, num_layers):
    old_groups = list(quantization["config_groups"].values())
    layers = "(?:" + "|".join(map(str, range(num_layers))) + ")"
    prefix = (
        rf"re:^(?:model\.language_model|language_model\.model)\.layers\.{layers}\.mlp"
    )
    result = {}
    for family, bits, suffix in (
        ("dense", 8, r"\.(gate|up|down)_proj$"),
        ("routed", 4, r"\.experts\.\d+\.(gate|up|down)_proj$"),
        ("shared", 8, r"\.shared_experts\.(gate|up|down)_proj$"),
    ):
        candidates = [
            g for g in old_groups if (g.get("weights") or {}).get("num_bits") == bits
        ]
        family_stems = [stem for stem in projections if _family(stem) == family]
        matched = [
            g
            for g in candidates
            if any(
                _matches(alias, g["targets"])
                for stem in family_stems
                for alias in _aliases(stem)
            )
        ]
        candidates = matched or candidates
        payloads = {
            json.dumps({k: v for k, v in g.items() if k != "targets"}, sort_keys=True)
            for g in candidates
        }
        if len(payloads) != 1:
            raise ValueError(f"Cannot infer an unambiguous {family} scheme")
        group = copy.deepcopy(candidates[0])
        _check_scheme(group, bits)
        group["targets"] = [prefix + suffix]
        result[family] = group
    return result


def audit_config(folder, *, repair=False):
    """Check header dtype/scale and HF/runtime target coverage; optionally fix config.

    No weight values are read or written. This does not verify NPU kernels,
    INT4 code ordering, numeric scale values or task accuracy.
    """
    folder = Path(folder)
    path = folder / "config.json"
    original = path.read_bytes()
    config = json.loads(original)
    owner = config if "quantization_config" in config else config.get("text_config", {})
    quantization = owner.get("quantization_config")
    if not quantization or quantization.get("quant_method") != "compressed-tensors":
        raise ValueError("Expected compressed-tensors quantization_config")
    if "quantization_config" in config and "quantization_config" in config.get(
        "text_config", {}
    ):
        raise ValueError("Duplicate top-level/text_config quantization_config")
    if quantization.get("format") != "mixed-precision":
        raise ValueError("Expected mixed-precision checkpoint")
    text = config.get("text_config", config)
    num_layers = text.get("num_hidden_layers", 0)
    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("Missing positive num_hidden_layers")
    tensors = _metadata(folder)
    projections, counts = _projections(tensors, num_layers)
    if repair:
        quantization["config_groups"] = _repair_groups(
            quantization, projections, num_layers
        )
    groups = list(quantization["config_groups"].values())
    for name in tensors:
        stem, _, suffix = name.rpartition(".")
        match = _PROJECTION.fullmatch(stem)
        if match and int(match[1]) >= num_layers and suffix == "weight":
            for alias in _aliases(stem):
                if not _matches(alias, quantization.get("ignore", [])) and any(
                    _matches(alias, group["targets"]) for group in groups
                ):
                    raise ValueError(f"Floating MTP projection is targeted: {alias}")
    for stem in projections:
        bits = 4 if ".experts." in stem else 8
        for alias in _aliases(stem):
            if _matches(alias, quantization.get("ignore", [])):
                raise ValueError(f"Quantized checkpoint projection is ignored: {alias}")
            matched = [g for g in groups if _matches(alias, g["targets"])]
            if len(matched) != 1:
                raise ValueError(
                    f"Expected one target match, got {len(matched)}: {alias}"
                )
            _check_scheme(matched[0], bits)
    changed = repair and json.loads(original) != config
    backup = None
    if changed:
        if path.read_bytes() != original:
            raise ValueError(
                "config.json changed during the audit; retry on the new config"
            )
        original_mode = stat.S_IMODE(path.stat().st_mode)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(f"config.json.bak_ct_metadata.{stamp}")
        with backup.open("xb") as handle:
            handle.write(original)
        fd, temporary = tempfile.mkstemp(prefix="config-", suffix=".json", dir=folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.chmod(temporary, original_mode)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return dict(
        projections=dict(counts),
        projection_names_sha256=hashlib.sha256(
            "\n".join(sorted(_aliases(stem)[0] for stem in projections)).encode()
        ).hexdigest(),
        hf_and_vllm_targets="passed",
        changed=changed,
        backup=str(backup) if backup else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--repair", action="store_true", help="Back up and fix config.json"
    )
    args = parser.parse_args()
    print(json.dumps(audit_config(args.model, repair=args.repair), indent=2))


if __name__ == "__main__":
    main()
