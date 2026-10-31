"""Losslessly pack/unpack an existing SSZ INT4 checkpoint on CPU.

Only integer storage changes; scales, activation schemes and float tensors are
preserved. Packed output requires a compatible inference loader; use
--format int-quantized only when an unpacked checkpoint is needed.
"""

import argparse
import json
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import (
    pack_to_int32,
    unpack_from_int32,
)
from compressed_tensors.entrypoints.convert import Converter, convert_checkpoint
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils import match_quantizable_tensors
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)


class SSZIntegerStorageConverter(Converter):
    def __init__(self, config, weight_names, target_format):
        if target_format not in ("int-quantized", "pack-quantized"):
            raise ValueError("Unsupported target format")
        self.target_format = target_format
        self.modules = set()
        names = dict.fromkeys(weight_names)
        self.groups = set()
        for name, scheme in config.config_groups.items():
            w = scheme.weights
            if w is None or w.num_bits != 4:
                continue
            if (
                w.type != "int"
                or w.strategy != "channel"
                or not w.symmetric
                or w.dynamic
                or w.observer != "ssz"
            ):
                raise ValueError("Expected static, symmetric, per-channel SSZ INT4")
            source_format = scheme.format or config.format
            if source_format not in ("int-quantized", "pack-quantized"):
                raise ValueError(f"Unsupported source format: {source_format}")
            if source_format == target_format:
                raise ValueError("INT4 weights already use the requested format")
            self.groups.add(name)
            self.modules.update(
                module
                for module, _ in match_quantizable_tensors(
                    names,
                    config.ignore,
                    scheme.targets,
                    param_targets=("weight_scale",),
                )
            )
        if not self.modules:
            raise ValueError("No calibrated SSZ INT4 modules found")
        for module in self.modules:
            param = "weight" if target_format == "pack-quantized" else "weight_packed"
            if module + "." + param not in names:
                raise ValueError(f"Missing {param} for {module}")
            if (
                target_format == "int-quantized"
                and module + ".weight_shape" not in names
            ):
                raise ValueError(f"Missing weight_shape for {module}")

    def get_dependencies(self, weight_name):
        module, _, param = weight_name.rpartition(".")
        if module in self.modules and param == "weight_packed":
            return {module + ".weight_shape"}
        return set()

    def validate(self, tensors):
        return self._convert(tensors, meta=True)

    def process(self, tensors):
        return self._convert(tensors, meta=False)

    def _convert(self, tensors, meta):
        output = dict(tensors)
        for name in list(tensors):
            module, _, param = name.rpartition(".")
            if module not in self.modules:
                continue
            prefix = module + "."
            if self.target_format == "pack-quantized" and param == "weight":
                weight = output.pop(name)
                if weight.dtype != torch.int8 or weight.ndim != 2:
                    raise ValueError(f"Expected 2D INT8 integer codes: {name}")
                if not meta and ((weight < -8) | (weight > 7)).any():
                    raise ValueError(f"INT4 code outside [-8, 7]: {name}")
                if meta:
                    packed = torch.empty(
                        (weight.shape[0], (weight.shape[1] + 7) // 8),
                        dtype=torch.int32,
                        device="meta",
                    )
                else:
                    packed = pack_to_int32(weight, 4).contiguous()
                output[prefix + "weight_packed"] = packed
                output[prefix + "weight_shape"] = torch.tensor(weight.shape)
            elif self.target_format == "int-quantized" and param == "weight_packed":
                packed = output.pop(name)
                shape = output.pop(prefix + "weight_shape")
                if packed.dtype != torch.int32 or packed.ndim != 2:
                    raise ValueError(f"Expected 2D INT32 packed weights: {name}")
                if meta:
                    weight = torch.empty(
                        (packed.shape[0], packed.shape[1] * 8),
                        dtype=torch.int8,
                        device="meta",
                    )
                else:
                    weight = unpack_from_int32(packed, 4, shape.tolist()).contiguous()
                output[prefix + "weight"] = weight
        return output

    def update_config(self, config):
        config = config.model_copy(deep=True)
        for name in self.groups:
            config.config_groups[name].format = self.target_format
        formats = {s.format or config.format for s in config.config_groups.values()}
        config.format = next(iter(formats)) if len(formats) == 1 else "mixed-precision"
        return config


def repack_checkpoint(source, output, target_format="pack-quantized"):
    source = Path(source).resolve(strict=True)
    output = Path(output).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("Output must be outside the source checkpoint directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory")
    data = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config = QuantizationConfig.model_validate(data["quantization_config"])
    names = get_weight_map(get_checkpoint_files(source))
    converter = SSZIntegerStorageConverter(config, names, target_format)
    convert_checkpoint(source, output, converter, max_workers=1, device="cpu")
    print(f"Converted {len(converter.modules)} SSZ INT4 modules to {target_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--format",
        choices=("pack-quantized", "int-quantized"),
        default="pack-quantized",
    )
    args = parser.parse_args()
    repack_checkpoint(args.source, args.output, args.format)
