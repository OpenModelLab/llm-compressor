"""Serializable FlatQuant input transforms using compressed-tensors hooks.

Import this package before loading a checkpoint with ``type=flatquant`` transforms.
The native loader constructs these modules before loading their parameters; no
training state or instance-level Linear forward patches are needed at inference.
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    apply_transform_config,
)
from compressed_tensors.transform.factory.base import TransformBase, TransformFactory
from compressed_tensors.utils import match_named_modules, update_offload_parameter
from compressed_tensors.utils.offload import get_offloaded_device
from safetensors import safe_open
from torch.nn import Module, Parameter

from .components import clip_activation, get_decompose_dim


@TransformFactory.register("flatquant", alias=["flatquant-asymmetric"])
class FlatQuantTransformFactory(TransformFactory):
    def create_transform(self, module: Module, args: TransformArgs):
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("FlatQuant runtime transforms require Linear targets")
        if args.location != "input" or args.inverse:
            raise ValueError("FlatQuant runtime supports forward input transforms only")
        size = module.in_features
        if self.scheme.head_dim is None:
            dims = get_decompose_dim(size)
        else:
            right_size = self.scheme.head_dim
            dims = (size // right_size, right_size) if size % right_size == 0 else None
        if dims is None:
            raise ValueError(f"Unsupported FlatQuant transform dimension: {size}")
        return FlatQuantInputTransform(
            dims, self.scheme, args, get_offloaded_device(module)
        )


class FlatQuantInputTransform(TransformBase):
    """Frozen ``x D (L ⊗ R)`` followed by optional token-wise clipping.

    The diagonal deliberately stays online. This avoids modifying offset norms
    or shared upstream consumers, and preserves the training quantization order.
    """

    _dynamic_tied_weights_keys: list[str] = []

    def __init__(self, dims, scheme, args, device):
        super().__init__()
        self.args = args
        self.symmetric = scheme.type == "flatquant"
        left_size, right_size = dims
        # Keep learned transform parameters FP32 even for a BF16 model. The
        # forward casts to the input dtype, matching calibration application.
        self.left = Parameter(
            torch.eye(left_size, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        self.right = Parameter(
            torch.eye(right_size, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        # MSMODELSLIM-COMPAT: no diagonal — it is absorbed into the source
        # module during training and is not part of the native vllm-ascend
        # W4A4_FLATQUANT_DYNAMIC parameter set.
        self.clip_ratio = Parameter(
            torch.ones(1, device=device, dtype=torch.float32), requires_grad=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        left, right = self.left.to(value), self.right.to(value)
        result = value.reshape(-1, left.shape[0], right.shape[0]) @ right
        result = left.T @ result
        result = result.reshape(shape)
        return clip_activation(result, self.clip_ratio.to(value), self.symmetric)


def load_flatquant_model(model_path, model_cls=None, **kwargs):
    """Load a local compressed checkpoint and restore every FlatQuant transform.

    Some Transformers versions do not apply ``transform_config`` at load time.
    This explicit loader handles those versions without patching model classes:
    it lets Transformers load the base/quantized weights, then installs native
    input hooks and loads the frozen transform parameters from the checkpoint.
    Loading missing transforms is a hard error, never an identity fallback.

    Pass ``model_cls=Qwen3_5ForConditionalGeneration`` for Qwen3.6. With
    ``output_loading_info=True``, return the model and remaining diagnostics.
    Only keys actually restored here are removed from unexpected_keys.
    """
    from transformers import AutoModelForCausalLM

    root = Path(model_path).expanduser().resolve(strict=True)
    document = json.loads((root / "config.json").read_text(encoding="utf-8"))
    raw = document.get("quantization_config", {}).get("transform_config")
    if raw is None:
        raw = document.get(TRANSFORM_CONFIG_NAME)
    if not raw:
        raise ValueError(
            "Checkpoint has no serialized FlatQuant transform configuration"
        )
    config = TransformConfig.model_validate(raw)
    if not any(
        scheme.type.startswith("flatquant") for scheme in config.config_groups.values()
    ):
        raise ValueError("Checkpoint has no FlatQuant transform groups")
    return_info = kwargs.pop("output_loading_info", False)
    model, info = (model_cls or AutoModelForCausalLM).from_pretrained(
        str(root), output_loading_info=True, **kwargs
    )
    declared = []
    for group_name, scheme in config.config_groups.items():
        if not scheme.type.startswith("flatquant"):
            continue
        for application in scheme.apply:
            targets = list(
                match_named_modules(model, application.targets, application.ignore)
            )
            if not targets:
                raise ValueError(
                    f"FlatQuant targets were not found: {application.targets}"
                )
            declared.extend(
                f"{name}.{group_name}_{application.location}" for name, _ in targets
            )
    modules = dict(model.named_modules())
    existing = [
        name
        for name in declared
        if isinstance(modules.get(name), FlatQuantInputTransform)
    ]
    if existing and len(existing) != len(declared):
        raise ValueError(
            "Checkpoint loader restored only part of its FlatQuant transforms"
        )
    if not existing:
        apply_transform_config(model, config)
    else:
        setattr(model, TRANSFORM_CONFIG_NAME, config)

    expected = {}
    modules = dict(model.named_modules())
    for name in declared:
        module = modules.get(name)
        if not isinstance(module, FlatQuantInputTransform):
            raise ValueError(f"FlatQuant transform was not constructed: {name}")
        for parameter in ("left", "right", "clip_ratio"):
            expected[f"{name}.{parameter}"] = (module, parameter)
    index = root / "model.safetensors.index.json"
    files = defaultdict(list)
    if index.exists():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        for key in expected:
            if key not in mapping:
                raise ValueError(f"Missing FlatQuant checkpoint parameter: {key}")
            files[mapping[key]].append(key)
    else:
        files["model.safetensors"] = list(expected)
    restored = set()
    for filename, keys in files.items():
        with safe_open(root / filename, framework="pt", device="cpu") as archive:
            available = set(archive.keys())
            for key in keys:
                if key not in available:
                    raise ValueError(f"Missing FlatQuant checkpoint parameter: {key}")
                module, parameter = expected[key]
                value = archive.get_tensor(key)
                if (
                    value.shape != getattr(module, parameter).shape
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid FlatQuant checkpoint parameter: {key}")
                update_offload_parameter(module, parameter, value.to(torch.float32))
                restored.add(key)
    info["unexpected_keys"] = [
        key for key in info.get("unexpected_keys", []) if key not in restored
    ]
    info["missing_keys"] = [
        key for key in info.get("missing_keys", []) if key not in restored
    ]
    return (model, info) if return_info else model