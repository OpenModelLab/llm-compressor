# GLM53 主干 QuaRot + FlexSmooth + SSZ

`glm53_flash_w4a8c8_ssz_example.py` 是无前向的 SSZ 消融对照，不能代替 ModelSlim 完整配方。本目录新增 `glm53_flash_w4a8c8_quarot_flex_ssz_example.py`，实际运行以下五个阶段：

1. offline QuaRot：全维 hidden/KV/value Hadamard、query latent 的 shifted 32-block Hadamard，seed=1234，FP32 变换，无 online down_proj。
2. 前向 FlexSmooth：仅 norm-linear，alpha/beta 自动搜索，各 21 个候选；保留全部校准 token。
3. dense MLP W8A8：原生 MinMax，W per-channel / A per-token dynamic。
4. routed experts W4A8：原生 SSZ，step=50，CT pack-quantized。
5. shared experts W8A8：原生 MinMax，CT int-quantized。

**当前支持边界是主干算法链。MTP 尚未适配，不是原 YAML 包含 MTP 的完整复现。** 如果源配置声明 MTP，必须显式传 `--disable-mtp`，导出配置将关闭 MTP；不应在推理端启用 speculative/MTP。需要原 YAML 连同 MTP 的严格参考时，继续使用 ModelSlim 原始配方。

## 为什么新增 helper

已有原生 QuaRot/FlexSmooth 映射面向 GLM52。`glm53_flash_preprocessing.py` 只补 GLM53 的模型映射和裸参数更新：复用原生 Hadamard/rotate_axis、FlexSmooth 采样生命周期和搜索、oneshot、SSZ/MinMax、offload 与 CT 保存，不修改 `src/`，不做全局 monkey patch。

主干映射覆盖 KDA 六个 norm 消费者、DSA 的 q/KV 低秩投影及 indexer、裸 kpool gate、dense/routed/shared/router、mHC 每个 stream，以及 visual merger 最后投影。视觉主体和 attention 不量化；merger 旋转是保持 residual basis 一致所需的浮点变换。

所有旋转和平滑都折叠到原有权重/norm。裸 `fn` 和 `index_kpool_compress_gate` 不改成 Linear，保存键名不变。仍需现有 vLLM-Ascend CT packed 加载支持。

FlexSmooth 默认按 `--flex-loss-chunk-tokens 4096` 分块计算重建误差，避免一次创建全部 token×输出通道的巨型临时矩阵。保留全量 token、全局通道最大值、原生两阶段候选顺序和 INT8 重建代理，不按块各选一套 scale，也不截取前 4096 个 token。跨块归约沿用原生分布式搜索的累积 dtype；归约/GEMM 顺序可能带来浮点差异，不承诺与整块计算逐 bit 相同。

全部候选非有限时停止，不静默使用 identity。文本校准的原生 sequential pipeline 可能提示视觉模块位于子图外并在校准期驻留，实际 NPU 峰值还需包含视觉塔、一个 decoder 层及 CPU 激活缓存，不能只计算搜索分块的内存。

## 校准数据

为对齐本地 ModelSlim adapter，输入为 JSON 字符串列表，或每行一个 JSON 字符串的 JSONL，例如：

```json
"A training example in its original textual format."
```

逐条直接调用 tokenizer，保留原顺序，不额外套 chat template，不接受 HF messages/image 对象。必须使用实际 `mix_calib.jsonl` 才能比较原参考结果；其他文本文件是新的校准实验。不要使用正式评测题校准。

超过 `--max-seq-length` 会在加载模型前报错，不静默截断。默认使用文件全部样本；`--num-calibration-samples` 仅用于明确标记的 smoke/数据量消融。

## 量化命令

依赖同目录的 preprocessing helper、SSZ 示例、GPTQ loader 和 CT auditor v2。保持当前本地已包含原生 SSZ、QuaRot/FlexSmooth 数学核的 llm-compressor 与配套 compressed-tensors，不使用未包含这些能力的任意 PyPI 包替代。

```bash
set -euo pipefail
cd /vllm-workspace/llm-compressor
SOURCE=/mnt/share/w30015592/models/GLM-5.3-Flash-BF16
CALIB=/mnt/share/w30015592/datasets/original_modelslim/mix_calib.jsonl
OUTPUT=/mnt/share/w30015592/models/GLM53-CT-QuaRot-Flex-SSZ-main
mkdir -p /mnt/share/w30015592/tmp
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_quarot_flex_ssz_example.py \
  --model "$SOURCE" --disable-mtp --plan-only
ASCEND_RT_VISIBLE_DEVICES=0 \
python examples/quantization_w4a8_int8/glm53_flash_w4a8c8_quarot_flex_ssz_example.py \
  --model "$SOURCE" --save-dir "$OUTPUT" \
  --offload-dir "${OUTPUT}-offload" \
  --calibration-file "$CALIB" --max-seq-length 4096 \
  --device npu:0 --step 50 --disable-mtp \
  2>&1 | tee /mnt/share/w30015592/tmp/glm53_quarot_flex_ssz_main.log
```

源、输出、offload 目录必须独立，输出/offload 使用新目录。默认 load-context 与已有 GPTQ loader 一致；目标环境遇到 Transformers 可选模块兼容问题时，可明确加 `--linearize-after-load`，不能将本地 CPU 的回退验收视为 NPU 默认路径通过。

## 产物和验收

- `calibration_report.json`：只有所有保存/审计成功才写入；包含阶段计时、输入文件/token IDs/mask 哈希、搜索结果、环境及脚本指纹。
- `glm53_rotation.safetensors`：四个旋转空间的矩阵，用于复核与复现，不加入模型权重索引。
- `glm53_flex_scales.safetensors`：每个 norm-linear 子图的平滑 scale。
- 配置中的 `quarot_config`、`flex_smooth_config`：阶段状态；主干权重已吸收变换，不依赖持久 runtime hook。
- `algorithm_chain_complete=true` 表示五阶段已执行，`full_modelslim_recipe=false` 明确保留 MTP 与部署验收差异，不能改成 true 来跳过验收。

本地 CPU 小模型需验证：旋转前后文本/图片 logits、浮点保存重载、实际 sequential FlexSmooth 校准及 hook 清理、原生三组量化/CT 导出、完整 CLI 和 generation_config 严格保存。实际全模型 NPU、vLLM eager/graph、TP4/TP8 和正式 MMMU-Pro/GPQA 仍须目标机器验证。

保存兼容逻辑会在 top_p/temperature 非 None 且 do_sample=False 时设置 do_sample=True，保留原采样字段；这也覆盖中性默认值。评测端必须显式固定采样参数，不能依赖保存后的默认采样行为。
