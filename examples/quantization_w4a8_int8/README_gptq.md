# GLM-5.3-Flash native oneshot W4A8 calibration

`glm53_flash_w4a8c8_gptq_example.py` uses existing LLM Compressor algorithms:

1. Optionally run native SmoothQuant on backbone MLP norm-linear subgraphs.
2. Calibrate routed experts with native GPTQ using image/text forward inputs,
   or use MinMax after SmoothQuant for a lower-memory candidate.
3. Quantize dense/shared MLP with native MinMax, preserving the reference's W8
   algorithm. Save all three groups as mixed-precision compressed-tensors.

The passes use separate `oneshot` calls. Smoothing completes before GPTQ's
observers are created. GPTQ and W8 targets are disjoint; W8 activation quantizers
are installed after the routed GPTQ pass. This is a first candidate, not joint
optimization of every quantizer in the final inference graph.

Three candidates are provided: G (GPTQ), SG (SmoothQuant + GPTQ), and SM
(SmoothQuant + MinMax). On the current A2 64GB cards, start with SM: it gathers
activation statistics during smoothing without allocating expert Hessians.
Use a larger-memory calibration node to compare G and SG. SM requires
`--smoothquant`; it does not claim activation-aware GPTQ weight optimization.

| Reference ModelSlim stage | This example |
| --- | --- |
| Offline QuaRot | Not enabled; no validated full GLM rotation mapping/backend contract |
| FlexSmoothQuant, norm-linear | Optional native SmoothQuant, restricted to backbone MLP input norms |
| Routed INT4 SSZ | Activation-aware GPTQ; or MinMax after smoothing for SM. Neither is SSZ |
| Dense/shared INT8 MinMax | Native MinMax |
| INT8 per-token activations | Dynamic INT8 per-token |
| AscendV1 saver | CT pack-quantized W4 and int-quantized W8 |

All weights are symmetric per-channel; groupwise INT4 and activation ordering
are not introduced. Attention, indexer, router, vision, heads and mHC remain
floating. Optional smoothing changes norm and router floating values to preserve
the computation; these modules are still not quantized.

## Calibration data

Use a separate training/calibration collection, not MMMU-Pro evaluation items.
For example, store one HF-style conversation per JSONL line:

```json
{"messages":[{"role":"user","content":[{"type":"image","image":"images/chart.png"},{"type":"text","text":"Explain the trend in this chart."}]},{"role":"assistant","content":"The value rises steadily over time."}]}
```

Relative image paths are resolved against the JSONL's directory. The processor
loads the images and preserves image token/grid information. The example uses
one processor-formed batch at a time; it does not re-stack or squeeze pixels.
It rejects samples longer than `--max-seq-length` rather than cutting image
tokens without adjusting image metadata. Match the serving image-resolution
policy in the source processor configuration.

Use all-image conversations for MMMU-Pro-oriented calibration. Do not mix
image and text-only forward branches in one file: a single sequential trace
may not cover both. `--allow-text-only` supports separate text probes.
Start with 256 samples from varied chart/OCR/math/science tasks, then expand
coverage as needed. All-expert calibration is enabled explicitly; its coverage
does not represent natural routed-token frequencies.

## Run

Use the original BF16 checkpoint and a fresh output directory. The current
implementation is single-process with CPU/disk offload; it is not an inference
TP8 quantization job. Choose the physical NPU through the visibility variable;
the logical device in this process is `npu:0`.

This is a new forward-calibration entrypoint, not `model_free_ptq`. Do not pass
the old example's `--algorithm` or `--max-workers` options to it.

```bash
# SM: lower-memory starting point for A2 64GB.
ASCEND_RT_VISIBLE_DEVICES=0 \
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_gptq_example.py \
  --model /models/GLM-5.3-Flash-BF16 \
  --calibration-data /datasets/glm_calibration/mix_calib.jsonl \
  --save-dir /models/GLM53-Flash-CT-SQ-MinMax \
  --device npu:0 --num-calibration-samples 256 --max-seq-length 4096 \
  --smoothquant --alpha 0.5 --routed-method minmax
```

For G, omit `--smoothquant --alpha 0.5 --routed-method minmax`; the default routed
method is GPTQ. For SG, keep `--smoothquant --alpha 0.5` and use
`--routed-method gptq`. Always choose a new output directory for each candidate.
Smoothing transforms the dense and MoE MLP inputs, compensating all routed
gate/up, shared gate/up and the floating router. It does not reuse a generic
Llama mapping or add online activation hooks to the saved model. A common
MoE smoothing scale can still hurt task accuracy; compare the candidates with
GPTQ alone before selecting it.

`--targets-per-graph 1` limits each subgraph to one decoder/vision block and
defaults to one. Per-expert tracing did not own all routed weight observers in
the tested version, so this example deliberately uses the decoder boundary.
For 288 experts, H=4096, I=2048, FP32 Hessians alone need approximately
`288 * (2*4096**2 + 2048**2) * 4 = 40.5 GiB` per MoE layer, plus approximately
13.5 GiB of BF16 expert weights and runtime buffers. A 64GB NPU is tight;
prefer a larger-memory calibration node for the first full-model run. Multiple
visible NPUs do not automatically shard this single-process job. Start with
shorter samples to validate memory, then increase sequence length.

SmoothQuant uses one decoder/vision block without storing expert Hessians.
SM therefore avoids that 40.5 GiB allocation, but still needs the floating
block weights, activation buffers and quantization workspace; full-model NPU
memory consumption has not been measured. The forward passes cache
intermediate activations on CPU; ensure RAM and disk/offload
space are sufficient. Native GPTQ compression batching is disabled, but that
does not remove the layer's simultaneously collected Hessians. `block_size=128`
is the GPTQ processing tile, not a change to per-channel quantization.

`--offload-dir` defaults to a sibling `<save-dir>-offload` directory and must be
separate from the BF16 source and output. The default `load_context` path loads
linearized experts directly. `--linearize-after-load` uses the native
`load_offloaded_model` + `linearize_moe` path when direct Transformers patching
is unavailable; it can increase loading time and temporary memory.

## MTP and serving

MTP forward calibration is not supported by the current native pipeline. The
example deliberately does not attach MTP, restricts targets to backbone layer
IDs and lets the native saver preserve source MTP tensors in floating point
when the source config declares them. It does not claim MTP W4A8 calibration.
Keep speculative decoding disabled for the first comparison, and check source
MTP declaration/completeness when copying those weights is required.

The cache contract is unchanged: request runtime dynamic C8 using
`vllm serve ... --kv-cache-dtype int8` on the adapted INT8 cache backend.
No static cache scales are generated. Reuse the existing Ascend CT packed
loader; this recipe does not make stock vLLM-Ascend v0.32 understand packed W4.

The output includes `calibration_report.json`, the selected input line numbers,
source JSONL hash, algorithm options and routed-module forward token counts.
Counts include trace/replay and forced all-expert execution; they are coverage
diagnostics, not natural routing probabilities or independent token counts.
The example rejects uncovered targets or an observed GPTQ-to-RTN fallback.

## Validation

The local regression constructs a random tiny GLM with a vision tower, KDA,
DSA, dense and routed/shared experts. It exercises real image tensors in the
forward path, floating smoothing equivalence including save/reload, and saved
CT dtype/scope checks for all three candidates, plus floating MTP preservation.
It uses the native post-load linearization path in a CPU-only environment;
it does not replace calibration or quantization with mocks. Data preparation
has a separate layout test with a minimal processor stub.

```bash
python -m pytest --noconftest -q \
  tests/llmcompressor/entrypoints/model_free/test_glm53_flash_gptq_example.py
```

The full BF16 checkpoint, real processor/image pipeline, direct-load path,
NPU calibration, serving kernels and MMMU-Pro gains require deployment-side
verification. GPTQ models input sensitivity; neither it nor SmoothQuant
guarantees a score improvement.
