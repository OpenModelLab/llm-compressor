# GLM-5.3-Flash INT: ModelSlim-aligned MLP quantization scope

The single Python example uses native `model_free_ptq`. It has no local utility
imports. Three disjoint recipe groups quantize only MLP projections:

| Group | Projections | Weights / activations |
| --- | --- | --- |
| `routed` | `mlp.experts.N.(gate/up/down)_proj` | INT4 / INT8 |
| `dense` | `mlp.(gate/up/down)_proj` | INT8 / INT8 |
| `shared` | `mlp.shared_experts.(gate/up/down)_proj` | INT8 / INT8 |

Router, attention/MLA/KDA/indexer, vision, LM head, embeddings, norms,
convolutions, state coefficients and mHC projections retain floating weights.
MTP MLP weights follow the same rules; its other projections remain floating.
This matches the supplied ModelSlim best-practice YAML's GLM module coverage.
It does not reproduce its QuaRot/FlexSmoothQuant/SSZ transformations or export
AscendV1 weights. The existing MinMax/MSE observer options remain unchanged.
Weight conversion can run on CPU or Ascend
A2/A3/A5; this does not establish serving-kernel or cache compatibility.

```bash
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_example.py \
  --model /models/GLM-5.3-Flash-BF16 \
  --save-dir /models/GLM-5.3-Flash-MLP-W4A8-CT \
  --device npu:0 --max-workers 1
```

The default HF ID is `zai-org/GLM-5.3-Flash-BF16`, pinned to
`f9fed3bbc82291073373d67452e9f25317ef7577`. Use an unquantized local snapshot
to avoid downloading the model implicitly. `--plan-only` prints the recipe
without loading weights or frameworks; it does not audit checkpoint headers.
`--algorithm mse` changes only the routed W4 observer (`norm=2.0`). Each
candidate must use a fresh output directory and the same original BF16 input.
The CLI rejects a source with a quantization_config. Existing broad-scope INT
checkpoints cannot acquire floating attention/router/head weights by editing
their config; generate this candidate from the original BF16 model.

Native PTQ splits 3D experts, computes qparams, compresses weights, and copies
model assets. Static weights use per-output-channel symmetric INT4 or INT8;
activations use dynamic per-token symmetric INT8. W4 uses `pack-quantized`,
W8 uses `int-quantized`.

The only model-specific Converter retains the serving name alias from the raw
indexer `.index_kpool_compress_gate` parameter to `.weight`. This tensor is
outside the quantized targets: its floating dtype and values are preserved.
The current Ascend CT loader still requires this alias for its Linear wrapper.
mHC `hc_attn_fn`/`hc_ffn_fn` and `attn_hc.fn`/
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
also quantizes dense/shared MLP and MTP MLP tensors. The existing NVFP4 example
quantizes routed experts only and copies unbuilt MTP tensors unchanged.

KV cache quantization is dynamic at serving time and transparent to the model.
This weight recipe does not calibrate caches, add cache schemes or export cache
scales. For the adapted A2/A3 INT8 DSA path,
request C8 at serving time with `vllm serve ... --kv-cache-dtype int8` and verify
the actual cache dtype. A5 cache support must be verified separately; running
weight conversion on A5 does not enable INT8 DSA cache on that device.
New MLP-only checkpoints already contain floating `kv_b_proj`; the legacy
INT8-to-BF16 absorption restore is only needed by older broad-scope checkpoints.
Full-model NPU/loader/kernel and accuracy validation are still required.
