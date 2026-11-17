# GLM-5.3-Flash INT: W4A8/W8A8 weight conversion

The single Python example uses native `model_free_ptq`. It has no local utility
imports. The two recipe groups are routed experts (W4A8) and shared/other
projections (W8A8). Norms, embeddings, convolutions/downsample, state parameters
and mHC fn projections are excluded. Weight conversion can run on CPU or Ascend
A2/A3/A5; this does not establish serving-kernel or cache compatibility.

```bash
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_example.py \
  --model /models/GLM-5.3-Flash-BF16 \
  --save-dir /models/GLM-5.3-Flash-A3-W4A8-CT \
  --device npu:0 --max-workers 1
```

The default HF ID is `zai-org/GLM-5.3-Flash-BF16`, pinned to
`f9fed3bbc82291073373d67452e9f25317ef7577`. Use an unquantized local snapshot
to avoid downloading the model implicitly. `--plan-only` prints the recipe
without loading weights or frameworks; it does not audit checkpoint headers.
`--algorithm mse` changes only the routed W4 observer (`norm=2.0`). Each
candidate must use a fresh output directory and the same original BF16 input.

Native PTQ splits 3D experts, computes qparams, compresses weights, and copies
model assets. Static weights use per-output-channel symmetric INT4 or INT8;
activations use dynamic per-token symmetric INT8. W4 uses `pack-quantized`,
W8 uses `int-quantized`.

The only model-specific Converter renames the raw indexer
`.index_kpool_compress_gate` parameter to `.weight` for native PTQ. The serving
loader must support this alias. mHC `hc_attn_fn`/`hc_ffn_fn` and `attn_hc.fn`/
`ffn_hc.fn` retain their original names and floating values: the current Ascend
mHC path consumes floating parameters. Optional Linear-style mHC weights are
also excluded, instead of silently widening the quantized projection set.

Use the paired compressed-tensors checkout containing the native NPU memory
query and worker device-context adaptation in
`src/compressed_tensors/entrypoints/convert/memory.py`. The example only imports
`torch_npu` to register the backend; it does not replace PyTorch memory-query
functions globally. Install the paired checkouts without replacing the image's
matched Torch/torch_npu packages:

```bash
python -m pip install --no-deps -e /path/to/compressed-tensors
python -m pip install --no-deps -e /path/to/llm-compressor
```

`--no-deps` preserves those packages but does not satisfy missing dependencies;
verify the environment before converting a full checkpoint.

Checkpoint completeness, unknown matrices, missing MTP and input dtype checks
are separate from the example. The delivery repository provides
`vllm-RP4/tools/audit_glm53_flash_checkpoint.py`; run it before full-model PTQ.
The delivery workflow's `a3_shell/01_quantize.sh` does this automatically.
The example itself has no dependency on that tool or repository.

Compared with `examples/quantizing_moe/glm53_flash_example.py`, this example
also quantizes shared experts, router, attention/indexer, head, vision and
MTP tensors. The existing NVFP4 example quantizes routed experts only and
copies unbuilt MTP tensors unchanged. Those examples have different scopes.

There is no offline C8 observer here. For the adapted A2/A3 INT8 DSA path,
request C8 at serving time with `vllm serve ... --kv-cache-dtype int8` and verify
the actual cache dtype. A5 cache support must be verified separately; running
weight conversion on A5 does not enable INT8 DSA cache on that device.
The current serving adaptation restores quantized `kv_b_proj` to BF16 for MLA
absorption. This and floating mHC are explicit execution exceptions. Full-model
NPU/loader/kernel and accuracy validation are still required. Existing converted
weights do not need regeneration solely for the NPU scheduler compatibility fix
or an inference-side A5 NZ layout correction.
