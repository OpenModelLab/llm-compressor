"""Compare FP32 SSZ against source functions without importing the source SDK.

Usage: python tests/llmcompressor/modifiers/quantization/ssz/validate_source.py SOURCE
Only the numerical functions are executed; IR containers are lightweight adapters.
This does not exercise the original framework's processor or serialization.
"""

import argparse
import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)
from compressed_tensors.quantization import (
    quantize as target_quantize,
)

from llmcompressor.modifiers.quantization.ssz.observer import SSZObserver


class QDType(Enum):
    FLOAT = "float"
    INT4 = "int4"
    INT8 = "int8"


@dataclass
class QStorage:
    dtype: object
    value: torch.Tensor

    def same_like(self, value):
        return QStorage(self.dtype, value)

    def to(self, dtype):
        self.dtype = dtype
        self.value = self.value.to(
            torch.float32 if dtype == QDType.FLOAT else torch.int8
        )
        return self


class QParam:
    def __init__(self, scheme, ext):
        self.scheme, self.ext = scheme, ext


env = dict(
    torch=torch,
    QStorage=QStorage,
    QParam=QParam,
    QDType=QDType,
    QScheme=SimpleNamespace,
    SpecError=ValueError,
    SCALE_SEARCH_ITER_NUM=50,
    SCALE_SEARCH_CONVERGE_THRESHOLD=1e-10,
    SCALE_SEARCH_MIN_SCALE=1e-30,
    EXT_SCALE_NAME="scale",
    EXT_OFFSET_NAME="offset",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_repo", type=Path)
root = parser.parse_args().source_repo / "msmodelslim"
for path, names in (
    (
        "ir/api/impl/int_quantization.py",
        {"int_quantize", "int_dequantize", "calculate_int_qparam"},
    ),
    (
        "core/quantizer/impl/ssz.py",
        {
            "set_ext_scale",
            "set_ext_offset",
            "get_ext_scale",
            "get_ext_offset",
            "ssz_calculate_qparam",
        },
    ),
):
    nodes = []
    for node in ast.parse((root / path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(root / path), "exec"), env)
env["quantize"] = env["int_quantize"]
env["fake_quantize"] = lambda weight, param: env["int_dequantize"](
    env["int_quantize"](weight, param), param
)

max_error = 0.0
count = 0
for seed in range(10):
    generator = torch.Generator().manual_seed(seed)
    for shape in ((1, 1), (1, 37), (9, 97)):
        weight = torch.randn(*shape, generator=generator)
        for bits in (4, 8):
            for step in (1, 10, 50):
                ref = env["calculate_int_qparam"](
                    weight.amin(1),
                    weight.amax(1),
                    QDType("int" + str(bits)),
                    "per_channel",
                    True,
                )
                ref = env["ssz_calculate_qparam"](
                    QStorage(QDType.FLOAT, weight.T),
                    ref,
                    SimpleNamespace(ext={"step": step}),
                )
                args = QuantizationArgs(
                    num_bits=bits,
                    strategy="channel",
                    symmetric=True,
                    observer="ssz",
                    observer_kwargs={"step": step},
                )
                result = SSZObserver(base_name="weight", args=args)(
                    weight
                ).get_qparams()
                difference = (
                    (result["scale"].flatten() - ref.ext["scale"].flatten())
                    .abs()
                    .max()
                    .item()
                )
                max_error = max(max_error, difference)
                torch.testing.assert_close(
                    result["scale"].flatten(),
                    ref.ext["scale"].flatten(),
                    rtol=3e-6,
                    atol=1e-8,
                )
                expected_codes = env["int_quantize"](
                    QStorage(QDType.FLOAT, weight.T), ref
                ).value.T
                actual_codes = target_quantize(
                    weight,
                    result["scale"],
                    result["zero_point"],
                    args,
                    dtype=torch.int8,
                )
                assert torch.equal(actual_codes, expected_codes)
                count += 1
print("source comparisons", count, "max abs scale difference", max_error)

for weight in (torch.zeros(2, 7), torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.4]])):
    ref = env["calculate_int_qparam"](
        weight.amin(1), weight.amax(1), QDType.INT4, "per_channel", True
    )
    ref = env["ssz_calculate_qparam"](QStorage(QDType.FLOAT, weight.T), ref)
    print("zero-channel source final scale", ref.ext["scale"])
