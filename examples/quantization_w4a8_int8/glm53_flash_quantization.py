"""Strict symmetric MinMax and SSZ arithmetic for the GLM53 recipe.

This example plugin uses the native observer lifecycle and CT serialization.
Weight fitting preserves storage-dtype products and reductions; reconstruction
errors are evaluated after storage-dtype dequantization, in FP32.
"""

import torch
from compressed_tensors.quantization import QuantizationStrategy

from llmcompressor.observers.base import Observer


@torch.no_grad()
def weight_qparams(weight, bits, method, step=50):
    if (
        weight.ndim != 2
        or weight.numel() == 0
        or weight.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or bits not in (4, 8)
        or method not in ("minmax", "ssz")
        or step < 1
    ):
        raise ValueError("Expected finite floating 2D INT4/INT8 weight settings")
    if not torch.isfinite(weight).all():
        raise ValueError("Nonfinite source weights")
    upper = 2 ** (bits - 1) - 1
    eps = weight.new_tensor(torch.finfo(torch.float32).eps)
    scale = (weight.abs().amax(1, keepdim=True) / upper).clamp_min(eps)

    def encode(value):
        return (weight / value).round().clamp(-upper - 1, upper).to(torch.int8)

    def mse(codes, value):
        # INT8 -> storage dtype before multiplication is part of the contract.
        reconstructed = (codes.to(value.dtype) * value).float()
        return (weight.float() - reconstructed).abs().square().mean(1, keepdim=True)

    codes = encode(scale)
    if method == "ssz":
        best_error = mse(codes, scale)
        for _ in range(step):
            assignment = codes.to(weight.dtype)
            fitted = (weight * assignment).sum(1, keepdim=True) / (
                assignment.square().sum(1, keepdim=True).clamp_min(1e-30)
            )
            # Zero rows retain the seeded positive scale. Do not accept NaN/Inf
            # into a checkpoint when storage-dtype reductions overflow.
            valid = torch.isfinite(fitted) & (fitted > 0)
            candidate = torch.where(valid, fitted, scale)
            codes = encode(candidate)
            error = mse(codes, candidate)
            gain = best_error - error
            progressing = (gain / best_error.clamp_min(1e-4) >= 1e-10) & (
                gain.abs() >= 1e-10
            )
            if not progressing.any():
                break
            better = valid & (error < best_error)
            scale = torch.where(better, candidate, scale)
            best_error = torch.where(better, error, best_error)
    return scale, encode(scale)


class _StrictWeightObserver(Observer):
    _stats_attrs = ["scale", "zero_point"]
    method = "minmax"

    def update_statistics_from_observed(self, observed):
        if (
            self.base_name != "weight"
            or self.args.strategy != QuantizationStrategy.CHANNEL
            or self.args.type != "int"
            or self.args.num_bits not in (4, 8)
            or not self.args.symmetric
            or self.args.dynamic
        ):
            raise ValueError(
                "Strict observer requires symmetric per-channel INT weights"
            )
        if self.args.observer_kwargs.keys() - {"step"}:
            raise ValueError("Only the SSZ step option is supported")
        self.scale, _ = weight_qparams(
            observed[0, :, 0, :],
            self.args.num_bits,
            self.method,
            self.args.observer_kwargs.get("step", 50),
        )
        self.zero_point = torch.zeros_like(self.scale, dtype=self.args.zp_dtype)

    def get_qparams(self):
        if not self.has_statistics:
            raise ValueError("Observe weights before requesting qparams")
        result = dict(scale=self.scale, zero_point=self.zero_point, global_scale=None)
        self.delete_statistics()
        return result


@Observer.register("glm53_strict_minmax")
class StrictMinMaxObserver(_StrictWeightObserver):
    """Integer symmetric endpoint convention: /127 for INT8."""


@Observer.register("glm53_strict_ssz")
class StrictSSZObserver(_StrictWeightObserver):
    """Storage-dtype least-squares scale fitting with greedy per-row selection."""

    method = "ssz"


def strict_groups(num_layers, step):
    from glm53_flash_w4a8c8_gptq_example import make_config_groups

    groups = make_config_groups(num_layers)
    for name, group in groups.items():
        group["weights"]["observer"] = (
            "glm53_strict_ssz" if name == "routed" else "glm53_strict_minmax"
        )
        if name == "routed":
            group["weights"]["observer_kwargs"] = {"step": step}
    return groups
