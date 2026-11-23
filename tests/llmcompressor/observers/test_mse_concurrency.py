"""MSE candidate QDQ must not change a scheme used by another shard."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

from llmcompressor.entrypoints.model_free.lifecycle import (
    validate_weight_for_quantization,
)
from llmcompressor.observers import grid_search
from llmcompressor.observers.helpers import flatten_for_calibration


@pytest.mark.parametrize("strategy", ["channel", "group", "tensor"])
def test_mse_qdq_keeps_shared_weight_scheme_valid(monkeypatch, strategy):
    weights = QuantizationArgs(
        num_bits=4,
        strategy=strategy,
        group_size=8 if strategy == "group" else None,
        symmetric=True,
        observer="memoryless_mse",
    )
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=weights,
        input_activations=QuantizationArgs(
            num_bits=8, strategy="token", symmetric=True, dynamic=True
        ),
    )
    weights = scheme.weights
    tensor = torch.linspace(-10, 10, 64).reshape(4, 16)
    observed = flatten_for_calibration(tensor, "weight", weights)
    scales = torch.ones(observed.shape[1:-1])
    zero_points = torch.zeros_like(scales)
    entered, release = Event(), Event()
    original_qdq = grid_search.fake_quantize

    def paused_qdq(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=10):
            raise TimeoutError("MSE concurrency test did not release QDQ")
        return original_qdq(*args, **kwargs)

    monkeypatch.setattr(grid_search, "fake_quantize", paused_qdq)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            grid_search._fake_quantize_with_observer_qparams,
            observed,
            weights,
            scales,
            zero_points,
        )
        try:
            assert entered.wait(timeout=10), "MSE did not enter candidate QDQ"
            # A different shard validates the same config while QDQ is active.
            validate_weight_for_quantization(tensor, scheme, "other.weight")
            assert weights.strategy == strategy
        finally:
            release.set()
        actual = future.result(timeout=10)

    torch.testing.assert_close(actual, observed.round().clamp(-8, 7))
    assert weights.strategy == strategy
    assert scheme.input_activations.strategy == "token"
