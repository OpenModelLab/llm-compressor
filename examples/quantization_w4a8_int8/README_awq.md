# GLM-5.3-Flash experiment B: AWQ + MinMax

Use native `AWQModifier`, `QuantizationModifier`, `oneshot` and the existing
GLM loader. No changes under `src/`, no algorithm copies, no runtime patches.
The experiment compares against SQ+MinMax; it applies no SmoothQuant,
GPTQ or rotation. MMMU-Pro gains remain to be measured.

## Scope and staged recipe

| Group | Final weight | Final activation | Format |
| --- | --- | --- | --- |
| Backbone routed gate/up/down | symmetric INT4 per-channel | dynamic INT8 per-token | pack-quantized |
| Backbone dense/shared gate/up/down | symmetric INT8 per-channel | dynamic INT8 per-token | int-quantized |
| Attention, vision, router, mHC, LM head, MTP | floating | original execution | original |

Runtime cache quantization continues to use `--kv-cache-dtype int8`.
The floating MoE router and shared gate/up projections are rescaled as
consumers of the same norm output; floating does not mean exempt from
equivalence compensation. Dense MLPs receive only INT8 MinMax, without AWQ.

1. Construct explicit MoE `post_attention_layernorm -> gate/up` mappings.
   Include every routed gate/up, shared gate/up and the router. The helper
   named `make_smoothquant_mappings` is reused solely to validate topology;
   no SmoothQuant modifier runs. Do not use the generic attention/down mappings.
2. One sequential `oneshot` runs `[AWQModifier, QuantizationModifier]` with
   **only routed W4A8** configured. AWQ searches gate/up scaling; routed down
   projections are also quantized by native MinMax. Shared/router stay floating
   during this phase and are compensated with the selected common scaling.
3. A second native data-free `oneshot` adds dense/shared W8A8 MinMax, using
   their weights after the compensation above.
4. Native compressed save, header/target audit, processor save, calibration report.

This ordering avoids mixed W4/W8 consumers in a single AWQ search. The current
native search reuses the last weight argument object across consumers; all
quantized consumers of each mapping here have identical W4 arguments. W8 schemes
are initialized only after AWQ finishes, so no framework fix is required.
The tradeoff is that W8 quantization error is not present during AWQ search or
the first pass's propagation. The search is not a jointly optimized W4/A8/W8
objective. Do not change the first phase to include W8 schemes without review.

## Scripts and commands

Keep these three files together under `examples/quantization_w4a8_int8/`:

- `glm53_flash_w4a8c8_awq_example.py` — new entrypoint.
- `glm53_flash_w4a8c8_gptq_example.py` — reused loader/data/scope helpers.
- `glm53_flash_ct_config.py` — reused export audit.

From the llm-compressor repository:

```bash
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_awq_example.py --plan-only

# Processor preflight, without loading BF16 weights or requiring an NPU.
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_awq_example.py \
  --model /mnt/share/w30015592/models/GLM-5.3-Flash-BF16 \
  --calibration-data /mnt/share/w30015592/datasets/glm53_calibration_v1/data/vision_256.jsonl \
  --num-calibration-samples 256 --max-seq-length 4096 --seed 42 \
  --preflight-only

# Full experiment B; start from original BF16, not the SQMM checkpoint.
ASCEND_RT_VISIBLE_DEVICES=0 \
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_awq_example.py \
  --model /mnt/share/w30015592/models/GLM-5.3-Flash-BF16 \
  --calibration-data /mnt/share/w30015592/datasets/glm53_calibration_v1/data/vision_256.jsonl \
  --save-dir /mnt/share/w30015592/models/GLM-5.3-Flash-AWQ-V256 \
  --device npu:0 --dtype bfloat16 --seed 42 \
  --num-calibration-samples 256 --max-seq-length 4096 \
  --n-grid 20 --duo-scaling false
```

The normal run also checks processor-expanded lengths before loading weights.
No image token sequences are truncated. Keep processor settings and selected
records identical to the SQ+MinMax baseline. CPU-only production quantization
is not offered: native AWQ unconditionally pins its activation cache.

`--duo-scaling false` searches activation-only scales, including identity.
This is the initial B candidate; `true` adds weight statistics and `both` splits
the search grid between both methods. Treat changes as separate candidates.
`--n-grid` defaults to 20; use 4 with a small sample set only as a deployment
smoke test, writing to a different directory. `--linearize-after-load` enables
the existing native fallback loader if direct load-context support is unavailable.

## Memory, export and validation

Calibration is one process on one visible accelerator. Inference TP8 does not
shard it automatically. AWQ avoids GPTQ Hessians, but clones mapped weights and
replays MoEs for each grid candidate. On A2 64GB, first measure a small-sample
full-model smoke run. Record NPU/RAM peak and time; do not assume a fixed runtime
or guaranteed fit. Cache offload uses CPU and whole-decoder subgraphs.

`calibration_report.json` records input hashes/selected lines, processor lengths,
seed, n_grid, duo_scaling, all routed forward counts, native AWQ error metrics,
script hashes and export audit. Counts include forced all-expert execution and
replay, not natural routing frequencies. The entrypoint rejects skipped mappings,
non-finite reported errors and uncovered routed projections. Native error metrics
are captured from its debug summary because finalization clears private metrics;
updating llm-compressor requires revalidating this summary format.

Check the exported mixed-precision contract before serving:

```bash
python examples/quantization_w4a8_int8/glm53_flash_ct_config.py \
  --model /mnt/share/w30015592/models/GLM-5.3-Flash-AWQ-V256
```

The saved weight observer still reads `memoryless_minmax`: AWQ scaling is
folded into norm/consumer weights, while MinMax supplies the final qparams.
Use the report's per-mapping search metrics to confirm that AWQ ran, not the
observer name alone.

Use the already adapted Ascend CT packed loader; pass `--quantization
compressed-tensors`, rather than the separate AWQ checkpoint loader. Keep
floating MTP and disable speculative decoding for the initial comparison.

Regression command:

```bash
python -m pytest --noconftest -q \
  tests/llmcompressor/entrypoints/model_free/test_glm53_flash_awq_example.py \
  tests/llmcompressor/entrypoints/model_free/test_glm53_flash_gptq_example.py \
  tests/llmcompressor/entrypoints/model_free/test_glm53_flash_ct_config.py
```

The CPU tests skip only activation-cache page pinning, which requires an
accelerator-enabled runtime; the real GLM image forward, AWQ search, MinMax and
save path execute. A separate known-scale test isolates native equivalence
folding and router preservation. Full-model NPU execution and task accuracy
are deployment validations, not established by these CPU tests.
