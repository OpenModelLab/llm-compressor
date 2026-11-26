# GLM53 QuaRot + FlexSmooth + SSZ W4A8

完整入口：`glm53_flash_w4a8c8_quarot_flex_ssz_example.py`。使用已有 oneshot、Hadamard、FlexSmooth 搜索、observer 生命周期和 CT 保存；拓扑适配和严格数值 observer 放在同目录，不修改框架源码，不依赖其他量化框架。

五阶段顺序固定：

1. 离线 QuaRot，seed=1234，FP32 构造和 dense 应用；无在线 down_proj 旋转。
2. norm-linear FlexSmooth，alpha/beta 各 21 个候选，完整激活张量搜索。
3. dense MLP：W8A8 MinMax，权重 per-channel，动态 per-token INT8 激活。
4. routed experts：W4A8 SSZ，step 默认 50，权重 per-channel，CT pack-quantized。
5. shared experts：W8A8 MinMax。

严格 observer 为 `glm53_strict_minmax`、`glm53_strict_ssz`。INT8 对称端点使用 /127；SSZ 的乘积与最小二乘归约保留权重存储 dtype，重建后以 FP32 评估误差，不擅自改成全 FP32 搜索。保留零行的正 scale，拒绝非有限输入。

FlexSmooth 的逐层统计使用 `propagate_error=False`：保留采集时的层输出给下一层，
平滑写回后不重放该层更新统计轨迹。报告明确记录这一传播约定，避免 BF16 舍入
导致后续层统计与预定流程不同；这不关闭平滑，也不改变五阶段顺序。

量化范围仅 backbone MLP，attention、router、indexer、视觉主体、mHC 不量化。为保持 hidden basis 一致，视觉 merger 最后投影、router、mHC、attention 的相关浮点权重仍参与必要旋转与补偿。所有 norm 消费者必须共同处理，裸 Parameter 不改成 Linear 键名。

MTP 尚未适配，有 MTP 的源模型必须显式 `--disable-mtp`；报告 `full_reference_recipe=false`，不能仅改标记宣称全图对齐。单进程量化与推理 TP 策略独立。

## 执行

```bash
set -euo pipefail
cd /vllm-workspace/llm-compressor
SOURCE=/mnt/share/w30015592/models/GLM-5.3-Flash-BF16
CALIB=/mnt/share/w30015592/datasets/mix_calib_raw.jsonl
OUTPUT=/mnt/share/w30015592/models/GLM53-CT-QuaRot-Flex-SSZ-strict-v2
ENTRY=examples/quantization_w4a8_int8/glm53_flash_w4a8c8_quarot_flex_ssz_example.py
python "$ENTRY" --model "$SOURCE" --disable-mtp --plan-only
ASCEND_RT_VISIBLE_DEVICES=0 python "$ENTRY" \
  --model "$SOURCE" --save-dir "$OUTPUT" --offload-dir "${OUTPUT}-offload" \
  --calibration-file "$CALIB" --max-seq-length 4096 \
  --device npu:0 --step 50 --disable-mtp
```

输入为 JSON 字符串列表或每行一个 JSON 字符串的 JSONL；逐条直接 tokenize，保留文件顺序，不加 chat template。不接受 messages/image 对象，不静默截断超长样本。`--num-calibration-samples` 改变样本范围，应作为独立实验记录。不要使用正式评测题校准。

以上为默认 `--calibration-format raw`。图文校准显式使用
`--calibration-format messages --num-calibration-samples 256`，输入为已有的
messages JSONL，图片路径相对 JSONL 所在目录。复用 GPTQ 示例的 processor
loader，每个样本都须包含真实图片，不接受纯文本与图文混合前向分支。
通过 AutoProcessor chat template 生成完整图文 tensor，不丢弃像素或图像 grid。

正式量化前，在同一命令末尾加 `--calibration-check-only`：只运行 processor
检查全部样本、展开后的长度及输入哈希，不加载模型权重。图像展开后超长直接
报错，不切断图片 token。报告记录 image_samples、选中行号、实际 token 数及
包含像素 tensor 的全输入哈希。messages 模式使用 seed=42 的确定性排序；这是
独立于原 raw 文本的校准实验，五阶段算法不变。

源、输出、offload 目录须相互独立，后两者使用新目录。同步入口时也须同步 `glm53_flash_preprocessing.py`、`glm53_flash_quantization.py`、SSZ helper、GPTQ loader 和 CT config auditor。

不传 `--flex-loss-chunk-tokens`：完整入口已切换严格 full-tensor 搜索，显式分块参数会报错。相对旧分块实现，峰值内存可能增加；OOM 不会自动改为近似搜索或丢弃 token。`--linearize-after-load` 是显式加载兼容选项，不是默认路径的验收证明。

## 保存与验收

- `calibration_report.json` 仅在保存和审计全部成功后写入，记录 `numeric_contract`、脚本及校准输入哈希、实际算法、覆盖范围和计时。
- `glm53_rotation.safetensors` 保存四个旋转空间，`glm53_flex_scales.safetensors` 保存各子图平滑尺度；它们不加入 checkpoint 权重索引。
- 离线变换已折叠进正常权重和 norm gain，运行时不读取旋转 sidecar。KV 由服务动态量化，保留 `--kv-cache-dtype int8` 及相应 indexer INT8 配置。
- sampling 保存 guard 在 top_p/temperature 非 None 且 do_sample=false 时设 true，保留原值，也覆盖中性默认值。评测请求必须显式指定采样设置。

本地验证覆盖文本/图像浮点 logits、保存重载、真实 sequential 校准、完整 CLI、FP32/BF16 权重经 oneshot/CT 导出后 scale/整数码逐值检查。真实全模型 NPU、稀疏检索、SFA 吸收、动态 C8、TP/EP、graph 与任务精度仍须目标机器验收，不能从 shape 或 sidecar 正交性直接推断通过。
