import json
from types import SimpleNamespace

import pytest
import torch
from compressed_tensors.transform import TransformArgs
from compressed_tensors.transform.factory.base import TransformFactory
from torch.nn import Linear
from torch.testing import assert_close

from llmcompressor.modifiers.transform.flatquant.components import get_decompose_dim
from llmcompressor.modifiers.transform.flatquant.runtime import (
    FlatQuantInputTransform,
    FlatQuantTransformFactory,
    load_flatquant_model,
)


def _scheme(head_dim=None, symmetric=True):
    return SimpleNamespace(
        type="flatquant" if symmetric else "flatquant-asymmetric",
        head_dim=head_dim,
        precision=torch.float32,
    )


def _args(location="input", inverse=False):
    return TransformArgs(targets=["re:.*"], location=location, inverse=inverse)


def _make_transform(size, head_dim=None, symmetric=True):
    scheme = _scheme(head_dim, symmetric)
    factory = FlatQuantTransformFactory(scheme)
    return factory.create_transform(Linear(size, size), _args())


def test_factory_is_registered():
    names = TransformFactory.registered_names()
    assert "flatquant" in names
    assert "flatquant-asymmetric" in names


def test_factory_creates_flatquant_input_transform():
    transform = _make_transform(64)
    assert isinstance(transform, FlatQuantInputTransform)


def test_factory_uses_decompose_dim_without_head_dim():
    size = 64
    transform = _make_transform(size)
    expected = get_decompose_dim(size)
    assert (transform.left.shape[0], transform.right.shape[0]) == expected


def test_factory_uses_head_dim_when_provided():
    size, head_dim = 64, 16
    transform = _make_transform(size, head_dim=head_dim)
    assert transform.right.shape[0] == head_dim
    assert transform.left.shape[0] == size // head_dim


def test_factory_rejects_non_linear_target():
    factory = FlatQuantTransformFactory(_scheme())
    with pytest.raises(ValueError, match="require Linear targets"):
        factory.create_transform(torch.nn.LayerNorm(16), _args())


def test_factory_rejects_non_input_location():
    factory = FlatQuantTransformFactory(_scheme())
    with pytest.raises(ValueError, match="forward input transforms only"):
        factory.create_transform(Linear(16, 16), _args(location="output"))


def test_factory_rejects_inverse_transform():
    factory = FlatQuantTransformFactory(_scheme())
    with pytest.raises(ValueError, match="forward input transforms only"):
        factory.create_transform(Linear(16, 16), _args(inverse=True))


def test_factory_rejects_unsupported_dimension():
    # head_dim does not divide the input size, and the dimension has no
    # a^2 - b^2 decomposition either (n = 2 mod 4)
    factory = FlatQuantTransformFactory(_scheme(head_dim=3))
    with pytest.raises(ValueError, match="Unsupported FlatQuant transform dimension"):
        factory.create_transform(Linear(64, 64), _args())


def test_parameters_are_fp32_and_frozen():
    transform = _make_transform(64)
    for name in ("left", "right", "clip_ratio"):
        parameter = getattr(transform, name)
        assert parameter.dtype == torch.float32
        assert parameter.requires_grad is False


def test_identity_transform_is_a_noop():
    size = 64
    transform = _make_transform(size)
    x = torch.randn(2, 3, size)
    assert_close(transform(x), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_matches_manual_kron(dtype):
    left_size, right_size = 8, 8
    size = left_size * right_size
    transform = _make_transform(size)
    torch.manual_seed(0)
    with torch.no_grad():
        transform.left.copy_(torch.randn(left_size, left_size))
        transform.right.copy_(torch.randn(right_size, right_size))

    x = torch.randn(4, size, dtype=dtype)
    expected = (
        torch.einsum(
            "ij,njk->nik",
            transform.left.to(dtype).T,
            x.reshape(-1, left_size, right_size)
            @ transform.right.to(dtype),
        )
        .reshape(x.shape)
    )
    assert_close(transform(x), expected, atol=1e-2, rtol=1e-2)


def test_forward_preserves_input_shape():
    transform = _make_transform(64)
    for shape in [(64,), (2, 64), (2, 3, 64)]:
        x = torch.randn(*shape)
        assert transform(x).shape == x.shape


def test_forward_handles_empty_input():
    transform = _make_transform(64)
    x = torch.empty(0, 64)
    assert transform(x).numel() == 0


def test_clip_ratio_scales_the_activation_bound():
    size = 64
    transform = _make_transform(size)
    x = torch.randn(4, size)
    # ratio 1.0 clips to the per-token absmax, so values are preserved
    assert_close(transform(x), x)
    # ratio 0.5 halves the bound and clips the outliers
    with torch.no_grad():
        transform.clip_ratio.fill_(0.5)
    clipped = transform(x)
    assert clipped.abs().amax(dim=-1).max() <= x.abs().amax(dim=-1).max()
    assert (clipped.abs() <= x.abs() + 1e-6).all()


def test_asymmetric_scheme_keeps_sign_of_bounds():
    size = 64
    transform = _make_transform(size, symmetric=False)
    assert transform.symmetric is False
    x = torch.randn(4, size)
    assert_close(transform(x), x)


def test_load_flatquant_model_rejects_checkpoint_without_transform_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="no serialized FlatQuant transform"):
        load_flatquant_model(tmp_path)


def test_load_flatquant_model_rejects_non_flatquant_groups(tmp_path):
    config = {
        "quantization_config": {
            "transform_config": {
                "config_groups": {
                    "hadamard_0": {
                        "type": "hadamard",
                        "apply": [{"targets": ["re:.*"], "location": "input"}],
                    }
                }
            }
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="no FlatQuant transform groups"):
        load_flatquant_model(tmp_path)


def test_load_flatquant_model_rejects_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_flatquant_model("/nonexistent/flatquant/checkpoint")