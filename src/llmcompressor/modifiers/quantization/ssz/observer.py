"""Internal Scan-Scale-Zero observer for SSZModifier.

Algorithm reference: msmodelslim/core/quantizer/impl/ssz.py at
a18feb70d980b3e119f295d7e6accd524234e9c8 (Mulan PSL v2).
This implementation uses the native compressed-tensors quantization contract.
"""

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    fake_quantize,
    quantize,
)

from llmcompressor.observers.base import Observer, QParamsDict

__all__ = ["SSZObserver"]


@Observer.register("ssz")
class SSZObserver(Observer):
    """Optimize symmetric INT4/INT8 scales independently per weight row.

    Used by ``SSZModifier(step=50)`` during native weight calibration.
    No activations are collected.
    Search arithmetic is FP32; candidates are evaluated in the observed weight
    dtype, which is also the dtype used to store scales by the native lifecycle.
    """

    _stats_attrs = ["scale", "zero_point"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _validate_args(self.base_name, self.args)
        options = self.args.observer_kwargs
        if options.keys() - {"step"}:
            raise ValueError("ssz only accepts the observer option 'step'")
        self.step = options.get("step", 50)
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step <= 0
        ):
            raise ValueError("ssz step must be a positive integer")

    @torch.no_grad()
    def forward(self, observed: torch.Tensor) -> "SSZObserver":
        if observed.ndim != 2 or observed.numel() == 0:
            raise ValueError("ssz requires a nonempty 2D weight tensor")
        return super().forward(observed)

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        # Native CHANNEL flattening: (1, out_features, 1, in_features).
        weight = observed[0, :, 0, :]
        self.scale, self.zero_point = _ssz_qparams(weight, self.args, self.step)

    @torch.no_grad()
    def get_qparams(self) -> QParamsDict:
        assert (
            self.has_statistics
        ), "No statistics available. Call observer(value) first."
        result = {
            "scale": self.scale,
            "zero_point": self.zero_point,
            "global_scale": None,
        }
        self.delete_statistics()
        return result


def _validate_args(base_name: str, args: QuantizationArgs) -> None:
    if (
        base_name != "weight"
        or args.type != QuantizationType.INT
        or args.num_bits not in (4, 8)
        or not args.symmetric
        or args.strategy != QuantizationStrategy.CHANNEL
        or args.dynamic
    ):
        raise ValueError(
            "ssz requires static symmetric INT4/INT8 per-channel weight quantization"
        )
    if args.scale_dtype is not None:
        raise ValueError("ssz does not support a quantized scale_dtype")


@torch.no_grad()
def _ssz_qparams(
    weight: torch.Tensor, args: QuantizationArgs, step: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alternate integer assignment and least-squares scale fitting.

    Keep the last assignment for the next iteration, and retain the best scale
    separately per row, as in the source. Stop *before* accepting the candidate
    if no channel improves beyond both source convergence thresholds.
    """
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("ssz supports FP16, BF16 and FP32 weights")
    values = weight.detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("ssz weights must be finite")

    # Preserve the source initialization (absmax / 7 or 127), rather than
    # compressed-tensors MinMax's absmax / 7.5 or 127.5.
    qmax = 2 ** (args.num_bits - 1) - 1
    best_scale = (
        (values.abs().amax(dim=1, keepdim=True) / qmax)
        .clamp(min=torch.finfo(torch.float32).eps)
        .to(weight.dtype)
    )
    zero_point = torch.zeros_like(best_scale, dtype=args.zp_dtype)
    codes = quantize(weight, best_scale, zero_point, args).float()
    best_mse = (
        (values - fake_quantize(weight, best_scale, zero_point, args).float())
        .square()
        .mean(dim=1, keepdim=True)
    )
    if not torch.isfinite(best_mse).all():
        raise ValueError("ssz weight magnitude exceeds FP32 MSE search range")

    for _ in range(step):
        fitted = (values * codes).sum(dim=1, keepdim=True) / codes.square().sum(
            dim=1, keepdim=True
        ).clamp(min=1e-30)
        candidate = fitted.to(weight.dtype)
        valid = torch.isfinite(candidate) & (candidate > 0)
        # Zero channels (and unrepresentable candidates) retain their valid scale.
        candidate = torch.where(valid, candidate, best_scale)
        codes = quantize(weight, candidate, zero_point, args).float()
        error = (
            (values - fake_quantize(weight, candidate, zero_point, args).float())
            .square()
            .mean(dim=1, keepdim=True)
        )
        improvement = best_mse - error
        progressing = (improvement / best_mse.clamp(min=1e-4) >= 1e-10) & (
            improvement.abs() >= 1e-10
        )
        if not progressing.any():
            break
        better = valid & (error < best_mse)
        best_scale = torch.where(better, candidate, best_scale)
        best_mse = torch.where(better, error, best_mse)

    return best_scale, zero_point
