# GLM-5.3-Flash A3: W4A8/W8A8 + INT8 C8

The single Python example uses native `model_free_ptq`. It has no local utility
imports. The two recipe groups are routed experts (W4A8) and shared/other
projections (W8A8). Norms, embeddings and convolutions are excluded.

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

The only model-specific Converter renames raw mHC `.fn` and indexer
`.index_kpool_compress_gate` parameters to `.weight`. This lets native PTQ
quantize those projections. The serving loader must support these aliases.

Checkpoint completeness, unknown matrices, missing MTP and input dtype checks
are separate from the example. The delivery repository provides
`vllm-RP4/tools/audit_glm53_flash_checkpoint.py`; run it before full-model PTQ.
The delivery workflow's `a3_shell/01_quantize.sh` does this automatically.
The example itself has no dependency on that tool or repository.

Compared with `examples/quantizing_moe/glm53_flash_example.py`, this example
also quantizes shared experts, router, attention/indexer, mHC, head, vision and
MTP tensors. The existing NVFP4 example quantizes routed experts only and
copies unbuilt MTP tensors unchanged. Those examples have different scopes.

Enable DSA C8 at serving time: `vllm serve ... --kv-cache-dtype int8`.
There is no offline C8 observer here. Full-model NPU/loader/kernel validation
is still required for the intended A3 deployment.
