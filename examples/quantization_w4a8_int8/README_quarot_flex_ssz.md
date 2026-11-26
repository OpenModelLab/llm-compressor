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

量化范围为 backbone 与单层 MTP 的 MLP，attention、router、indexer、视觉主体、mHC、eh_proj 不量化。为保持 hidden basis 一致，视觉 merger 最后投影、router、mHC、attention 的相关浮点权重仍参与必要旋转与补偿。所有 norm 消费者必须共同处理，裸 Parameter 不改成 Linear 键名。

MTP 保持源层号，支持一个 next-token predictor：无 HC 的 DSA + MoE；enorm/hnorm 按拼接次序折叠到 eh_proj，两段输入右旋、输出左旋，独立 embedding/head 完成相应补偿。MTP 专家也参加 SSZ/MinMax 和 CT 范围审计；缺失主体权重直接报错。不要传 `--disable-mtp`。单进程量化与推理 TP 策略独立。

校准使用专用图：固定零索引的稀疏注意力、简化 KDA 前向（执行 forget/beta 投影但不参与其注意力输出）、仅计算路由命中专家、逐层保留平滑前输出。MTP 输入由主干多流均值、最终 norm/head 预测、删除第一个零并左移 token、预测值补尾、enorm/hnorm/eh_proj 构造。图文模式先用原生视觉编码器和 placeholder 检查/替换构造主干 embedding；图像条件隐藏状态进入 MTP，MTP 的另一半输入继续使用移位 token 的独立 embedding/enorm，不重复执行视觉塔。该图只在校准上下文存在，成功或异常均恢复所有 forward 与 offload 元数据。推理仍使用真实 KDA/Indexer，不保存校准用的简化算子。

校准 MLP 去掉 HF 的 clamp，使用 FP32 SiLU/乘积/down projection 后转回输入 dtype；仅临时提升 down 权重参与运算，不改持久权重。已核查参考 Python 路径：加载 BF16 权重，却将 FP32 乘积直接传入 Linear；校准调用链无显式 autocast 或权重提升。原始定义在 CPU BF16 下报 dtype 不匹配。本示例的 down 权重提升是明确的兼容处理，不是已核实的原路径转换；报告以 `calibration_precision_equivalence_verified=false` 记录。不能把 CPU FP32 对照当作完整 BF16 图逐位等价，也不据此擅自把乘积截回 BF16 或全局开启 autocast。

## 执行

```bash
set -euo pipefail
cd /vllm-workspace/llm-compressor
SOURCE=/mnt/share/w30015592/models/GLM-5.3-Flash-BF16
CALIB=/mnt/share/w30015592/datasets/mix_calib.jsonl
OUTPUT=/mnt/share/w30015592/models/GLM53-CT-QuaRot-Flex-SSZ-reference-mtp-v3
ENTRY=examples/quantization_w4a8_int8/glm53_flash_w4a8c8_quarot_flex_ssz_example.py
python "$ENTRY" --model "$SOURCE" --plan-only
ASCEND_RT_VISIBLE_DEVICES=0 python "$ENTRY" \
  --model "$SOURCE" --save-dir "$OUTPUT" --offload-dir "${OUTPUT}-offload" \
  --calibration-file "$CALIB" --max-seq-length 4096 \
  --calibration-format raw --num-calibration-samples 48 --device npu:0 --step 50
```

输入为 JSON 字符串列表、每行一个 JSON 字符串，或含 `inputs_pretokenized` 字段的 JSONL；逐条直接 tokenize，保留文件顺序，不加 chat template。不静默截断超长样本。对标时使用原始48条文本，样本范围或文本变更须作为独立实验记录。不要使用正式评测题校准。

以上为原48条文本的同输入对照。256图文校准已支持单层 MTP，算法链保持不变，属于数据扩展，不能称同输入对标。图文校准显式使用
`--calibration-format messages --num-calibration-samples 256`，输入为已有的
messages JSONL，图片路径相对 JSONL 所在目录。复用 GPTQ 示例的 processor
loader，每个样本都须包含真实图片，不接受纯文本与图文混合前向分支。
通过 AutoProcessor chat template 生成完整图文 tensor，不丢弃像素或图像 grid。

正式量化前，在同一命令末尾加 `--calibration-check-only`：只运行 processor
检查全部样本、展开后的长度及输入哈希，不加载模型权重。图像展开后超长直接
报错，不切断图片 token。报告记录 image_samples、选中行号、实际 token 数及
包含像素 tensor 的全输入哈希。messages 模式使用 seed=42 的确定性排序；这是
独立于原 raw 文本的校准实验，五阶段算法不变。

源、输出、offload 目录须相互独立，后两者使用新目录。同步入口时也须同步 `glm53_flash_calibration.py`、`glm53_flash_preprocessing.py`、`glm53_flash_quantization.py`、SSZ helper、GPTQ loader 和 CT config auditor。

不传 `--flex-loss-chunk-tokens`：完整入口已切换严格 full-tensor 搜索，显式分块参数会报错。相对旧分块实现，峰值内存可能增加；OOM 不会自动改为近似搜索或丢弃 token。`--linearize-after-load` 是显式加载兼容选项，不是默认路径的验收证明。

## 保存与验收

- `calibration_report.json` 仅在保存和审计全部成功后写入，记录 `numeric_contract`、脚本及校准输入哈希、实际算法、覆盖范围和计时。
- `glm53_rotation.safetensors` 保存四个旋转空间，`glm53_flex_scales.safetensors` 保存各子图平滑尺度；它们不加入 checkpoint 权重索引。
- 离线变换已折叠进正常权重和 norm gain，运行时不读取旋转 sidecar。KV 由服务动态量化，保留 `--kv-cache-dtype int8` 及相应 indexer INT8 配置。
- sampling 保存 guard 在 top_p/temperature 非 None 且 do_sample=false 时设 true，保留原值，也覆盖中性默认值。评测请求必须显式指定采样设置。
- 输出 config.architectures 恢复为标准 `Glm5NextForConditionalGeneration`，不要求推理端导入本示例模型类；MTP 位于 `model.language_model.layers.<num_hidden_layers>`，没有重复 `mtp.*` 权重子树。
- 有 MTP 时，真实 checkpoint 包含 `rot.weight = Qᵀ diag(原最终norm gamma) Q`，config 的 `is_rot_used=true` 供后端在 MTP hnorm 前应用；它与四个诊断旋转矩阵不是同一资产。

本地验证覆盖文本/图像浮点 logits、保存重载、真实 sequential 校准、完整 CLI、FP32/BF16 权重经 oneshot/CT 导出后 scale/整数码逐值检查。真实全模型 NPU、稀疏检索、SFA 吸收、动态 C8、TP/EP、graph 与任务精度仍须目标机器验收，不能从 shape 或 sidecar 正交性直接推断通过。
