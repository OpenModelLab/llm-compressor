import pytest
import torch
from compressed_tensors.quantization import QuantizationArgs, fake_quantize

from llmcompressor.modifiers.quantization.ssz.observer import SSZObserver
from llmcompressor.observers import Observer


def _args(bits=4, **kwargs):
    return QuantizationArgs(
        num_bits=bits, strategy="channel", symmetric=True, observer="ssz", **kwargs
    )


def _observe(weight, bits=4, **kwargs):
    return SSZObserver(base_name="weight", args=_args(bits, **kwargs))(
        weight
    ).get_qparams()


def test_analytical_least_squares_and_channel_axis():
    # Row 0 assigns integer codes [7, 1, 3]; its least-squares scale is
    # dot([1, .2, .4], [7, 1, 3]) / dot([7, 1, 3], [7, 1, 3]) = 8.4/59.
    weight = torch.tensor([[1.0, 0.2, 0.4], [-2.0, -0.4, -0.8]])
    result = _observe(weight, observer_kwargs={"step": 1})
    torch.testing.assert_close(result["scale"], torch.tensor([[8.4 / 59], [16.8 / 59]]))
    assert result["scale"].shape == (2, 1)
    assert torch.equal(result["zero_point"], torch.zeros(2, 1, dtype=torch.int8))


def test_source_initialization_and_stop_before_update():
    # INT4 source initialization is 1/7, not native MinMax's 1/7.5.
    # Rescaling the preceding example makes the absolute gain < 1e-10;
    # source convergence exits before accepting the candidate.
    weight = torch.tensor([[1.0, 0.2, 0.4]]) * 1e-4
    result = _observe(weight)
    torch.testing.assert_close(result["scale"], torch.tensor([[1e-4 / 7]]))


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_no_worse_than_source_minmax_and_monotone_budget(bits, dtype):
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(12, 97, generator=generator).to(dtype)
    weight[0] = 0
    weight[1] = -weight[1].abs()
    weight = weight.T.contiguous().T  # Include non-contiguous Linear weights.
    args = _args(bits)
    initial = weight.float().abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1)
    initial = initial.clamp(min=torch.finfo(torch.float32).eps).to(dtype)

    def loss(scale):
        dequant = fake_quantize(weight, scale, torch.zeros_like(scale), args)
        return (weight.float() - dequant.float()).square().mean(1)

    previous = loss(initial)
    for step in (1, 10, 50):
        result = _observe(weight, bits, observer_kwargs={"step": step})
        assert result["scale"].dtype == dtype
        assert (result["scale"] > 0).all()
        current = loss(result["scale"])
        assert (current <= previous + 1e-12).all()
        previous = current
        assert current[0] == 0


@pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1)])
def test_singleton_shapes(shape):
    weight = torch.ones(shape)
    result = _observe(weight)
    assert result["scale"].shape == (shape[0], 1)
    torch.testing.assert_close(
        fake_quantize(weight, result["scale"], result["zero_point"], _args()), weight
    )


def test_registry_statistics_lifecycle():
    observer = Observer.load_from_registry("ssz", base_name="weight", args=_args())
    assert isinstance(observer, SSZObserver)
    weight = torch.tensor([[1.0, 0.2, 0.4]])
    observer(weight)
    original = observer.scale.clone()
    observer(weight * 2)  # Weights are observed only once until qparams are consumed.
    assert torch.equal(observer.scale, original)
    result = observer.get_qparams()
    assert not observer.has_statistics
    assert result["global_scale"] is None
    assert not list(observer._forward_hooks)
    with pytest.raises(AssertionError, match="No statistics"):
        observer.get_qparams()
    observer(weight * 2)
    torch.testing.assert_close(observer.get_qparams()["scale"], original * 2)


@pytest.mark.parametrize("step", [0, -1, 1.5, True, "10", None])
def test_invalid_step(step):
    with pytest.raises(ValueError, match="positive integer"):
        SSZObserver(base_name="weight", args=_args(observer_kwargs={"step": step}))


@pytest.mark.parametrize(
    "changes",
    [
        {"symmetric": False},
        {"strategy": "tensor"},
        {"strategy": "group", "group_size": 4},
        {"num_bits": 2},
        {"type": "float"},
        {"strategy": "tensor", "dynamic": True},
        {"scale_dtype": torch.float16},
    ],
)
def test_unsupported_schemes(changes):
    fields = _args().model_dump()
    fields.update(changes)
    with pytest.raises(ValueError, match="ssz"):
        SSZObserver(base_name="weight", args=QuantizationArgs(**fields))


def test_activation_and_unknown_options_rejected():
    with pytest.raises(ValueError, match="weight"):
        SSZObserver(base_name="input", args=_args())
    with pytest.raises(ValueError, match="only accepts"):
        SSZObserver(base_name="weight", args=_args(observer_kwargs={"steps": 10}))


@pytest.mark.parametrize(
    "weight",
    [
        torch.empty(0, 2),
        torch.empty(2, 0),
        torch.ones(2, 3, 4),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
        torch.ones(2, 3, dtype=torch.int32),
    ],
)
def test_invalid_weights(weight):
    with pytest.raises(ValueError, match="ssz"):
        _observe(weight)
