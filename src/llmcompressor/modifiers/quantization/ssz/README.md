# SSZ

`SSZModifier` fits symmetric INT4/INT8 per-channel weight scales using
Scan-Scale-Zero. It runs through the native quantization lifecycle and requires
no calibration data for weight-only or dynamic activation quantization.

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.ssz import SSZModifier

oneshot(
    model=model,
    processor=tokenizer,
    recipe=[
        SSZModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"], step=50),
    ],
    pipeline="independent",
    output_dir="model-SSZ-W8A8",
)
```

The default scheme is dynamic `W8A8`. For weight-only INT4 or INT8, configure the
per-channel weights explicitly (set `num_bits` to 4 or 8). The standard `W4A16`
and `W8A16` presets use group quantization and are rejected:

```python
modifier = SSZModifier(
    step=50,
    ignore=["lm_head"],
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "symmetric": True, "strategy": "channel"},
        },
    },
)
```

SSZ is the quantization stage, so it replaces `QuantizationModifier` in a Python
recipe. Preprocessing modifiers can precede it with `pipeline="independent"`.
Static activation quantization still needs a dataset and its normal calibration
pipeline. See the [Qwen3.6 example](../../../../../examples/quantization_w8a8_int8/qwen3_6_ssz_example.py)
for the IterSmooth-style loading pattern, with targets covering text projections
and selected linear-attention and vision layers.

The Qwen3.6 SSZ-only example snapshots the original offset RMSNorm weights and
restores them through the offload API before saving. This avoids rounding changes
from the shared calibration context's BF16 gamma conversion. Saving is deliberately
separate from `oneshot`; passing an output directory there would save before the
restoration. This safeguard belongs to that example and must not be used to undo
intentional norm changes in a smoothing recipe. Other direct `oneshot` entrypoints
must account for the shared context separately.

## Algorithm and compatibility

The source is `msmodelslim/core/quantizer/impl/ssz.py`, commit
`a18feb70d980b3e119f295d7e6accd524234e9c8` (Mulan PSL v2). The target baseline is
`d7f4e7bcd3eeb314fef94050190ca231815a655b`.

For each Linear weight row, SSZ initializes `s = max(absmax(W) / qmax, eps_fp32)`,
where `qmax` is 7 or 127. It alternates integer assignment
`q = clamp(round(W / s), -qmax - 1, qmax)` with least-squares fitting
`s_new = sum(W * q) / max(sum(q * q), 1e-30)`, retaining the best scale per row.
The default iteration budget is 50. Source convergence thresholds and the check
before accepting the final candidate are preserved. Initialization deliberately
uses the source denominator, rather than native MinMax's 7.5 or 127.5.

The observer performs FP32 fitting and error reduction, evaluates candidates in
the weight dtype, and retains a valid scale for zero or unrepresentable candidates.
Low-precision results are not guaranteed to match the source bit for bit.
Unsupported schemes, invalid iteration budgets, and quantized scale dtypes are
rejected before calibration. Non-finite weights are rejected during scale fitting.

Only static symmetric per-channel INT4/INT8 weights in FP32, FP16 or BF16 are
supported. The source's public registry exposes the same bitwidth/symmetry/scope
combinations. Native observers, offload updates, integer packing and loading are
reused; no extra inference hooks or model classes are required. Three-dimensional
expert weights, tensor/expert parallelism, and custom fused layouts are unverified.

## Validation

```bash
python -m pytest tests/llmcompressor/modifiers/quantization/ssz -q
python tests/llmcompressor/modifiers/quantization/ssz/validate_source.py /path/to/msmodelslim
```

Tests cover analytical scales, edge cases, factory registration, Python recipe
serialization, and local Llama quantization, packing and reload for weight-only
and dynamic activation schemes. The source comparison executes extracted source
numerical functions with lightweight IR adapters; it does not run the source SDK.
CPU validation passed 58 SSZ tests and 32 neighboring MinMax/FusionHandler tests
(90 total). All 180 FP32 source comparisons matched scales and integer codes exactly.
GPU/NPU execution, production-model accuracy, throughput, and distributed runs
remain unverified.
