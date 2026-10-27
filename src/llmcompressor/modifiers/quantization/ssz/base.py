"""SSZ weight quantization through the native quantization lifecycle."""

from compressed_tensors.quantization import QuantizationConfig
from pydantic import Field, model_validator

from llmcompressor.core import State
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.ssz.observer import SSZObserver

__all__ = ["SSZModifier"]


class SSZModifier(QuantizationModifier):
    """Fit symmetric INT4/INT8 per-channel weight scales using SSZ.

    Accepts the native ``scheme``, ``config_groups``, ``targets`` and ``ignore``
    arguments. The default scheme is dynamic W8A8. ``step`` limits the number
    of least-squares iterations and defaults to 50.

    SSZ requires no activation samples. Static activation quantization still
    requires calibration data, as determined by the inherited lifecycle.
    Use this modifier as the quantization stage after any preprocessing.
    """

    step: int = Field(default=50, strict=True, gt=0)

    @model_validator(mode="before")
    @classmethod
    def default_scheme(cls, value):
        if isinstance(value, dict):
            if value.get("scheme") is None and value.get("config_groups") is None:
                value = {**value, "scheme": "W8A8"}
        return value

    def resolve_quantization_config(self) -> QuantizationConfig:
        if self.weight_observer not in (None, "ssz") or (
            self.observer is not None
            and self.observer.get("weights") not in (None, "ssz")
        ):
            raise ValueError("SSZModifier requires the ssz weight observer")
        config = super().resolve_quantization_config().model_copy(deep=True)
        for scheme in config.config_groups.values():
            weights = scheme.weights
            if weights is None:
                raise ValueError(
                    "SSZModifier requires weight quantization in every group"
                )
            options = weights.observer_kwargs
            if "step" in options and options["step"] != self.step:
                raise ValueError("Configure SSZ iterations with SSZModifier(step=...)")
            weights.observer = "ssz"
            weights.observer_kwargs = {**options, "step": self.step}
            # Validate the complete weight configuration before model initialization.
            SSZObserver(base_name="weight", args=weights)
        return config

    def on_initialize(self, state: State, **kwargs) -> bool:
        if self.start not in (None, -1) or self.end not in (None, -1):
            raise ValueError("SSZModifier only supports one-shot compression")
        return super().on_initialize(state, **kwargs)
