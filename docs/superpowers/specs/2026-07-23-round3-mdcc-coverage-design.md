# 点心杯粤语 ASR Round 3：MDCC 数据覆盖实验设计

日期：2026-07-23  
状态：待用户书面审阅  
范围：只改变训练 Manifest 的外部数据覆盖量；不改变模型、优化器、解码或验证边界

## 1. 目标

在 Round 2 的 `external73` 获得平台分数 `62.69` 后，验证继续增加合格
MDCC publisher-train 样本能否进一步提升泛化能力，同时保持官方固定验证集性能。

本轮不以平台分数选模。平台只抽取 200 条样本，且抽样种子由权重哈希派生，
不同提交之间存在抽样噪声。所有 checkpoint 仍只按固定官方 validation 的
`sentence_accuracy_tol2` 选择，CER 是硬保护指标。

## 2. 不变边界

- 基座始终为 `openai/whisper-small`，进行 Full SFT。
- 保持 Whisper-small 架构、Tokenizer、词表、GELU、层数和 attention heads 不变。
- 训练标签继续使用官方“粤语原文”、Common Voice `sentence`、MDCC `transcript`。
- 官方 validation 固定为 702 条，不加入训练。
- 公开 `template_pre` 及其 1,900 条映射样本不加入训练或选模。
- Common Voice dev/test 与 MDCC validation/test 共 27,177 条继续作为固定 OOD，
  不加入训练或选模。
- 继续执行受保护文本重叠、音频 SHA-256、重复 ID、损坏音频、空标签和 30 秒
  时长限制等现有 QC。
- 不加入 LoRA、蒸馏、伪标签、模型集成、数据增强或解码搜索。

## 3. 数据事实与实验组

仓库回归测试记录的已验收 publisher-train 池为：

- 官方 train：6,292 条。
- Common Voice 26.0 `zh-HK` train：8,451 条。
- MDCC train：64,779 条。

服务器执行前必须从
`artifacts/manifests/external/v1/data_report.json` 和各 JSONL 实际读取数量；
若与上述记录不一致，使用服务器 Manifest 的实际数量并在 `mix_report.json`
记录差异，不静默沿用硬编码数量。

四张 GPU 分别从原始 Whisper-small 训练下列配置：

| Trial | 官方 | Common Voice | MDCC | 预计总行数 |
|---|---:|---:|---:|---:|
| `external73-control` | 6,292 | 8,451 | 8,561 | 23,304 |
| `external80-mdcc` | 6,292 | 8,451 | 16,717 | 31,460 |
| `external85-mdcc` | 6,292 | 8,451 | 27,204 | 41,947 |
| `all-train-mdcc-max` | 6,292 | 8,451 | 64,779 | 79,522 |

前三组沿用现有确定性采样规则：先在 CV/MDCC 间按行数交替取样，CV 用尽后
继续从 MDCC 无放回抽样。第四组保留全部合格 publisher-train 行，不重复任何样本。
每组都保留全部官方训练行。

`mix_report.json` 必须记录每组的实际行数、各来源行数、音频小时数、实际外部比例、
seed 和 Manifest SHA-256。构建失败或样本不足时必须终止，不允许有放回补足目标比例。

## 4. 训练配置

四组只改变 train Manifest，其他设置完全一致：

- Full SFT、AdamW。
- learning rate：`2e-5`。
- cosine scheduler。
- weight decay：`0.01`。
- warmup ratio：`0.05`。
- micro-batch：`8`。
- gradient accumulation：`2`，effective batch `16`。
- epochs：`3`。
- gradient clipping：`1.0`。
- bf16、TF32、seed `42`。
- generation max length：`225`。

每个 epoch 保存 checkpoint，并评测固定 validation、train-probe 和 2,000 条固定
OOD panel。训练 3 epochs 不代表选择最后 epoch；每个 trial 都按验证规则选择
checkpoint。

## 5. 选模与诊断

checkpoint 必须先满足：

- `sentence_accuracy_tol2 >= 0.8219`。
- `CER <= 0.1163`。

合格 checkpoint 按以下顺序排序：

1. validation `sentence_accuracy_tol2` 更高；
2. validation CER 更低；
3. validation teacher-forced loss 更低；
4. epoch 更早。

train-probe、OOD、公开测试和平台隐藏分只用于诊断，禁止参与 checkpoint 或 recipe
选择。Round 3 的最佳 recipe 与当前提交模型比较时，要求固定 validation 准确率
至少不降低；提升达到 0.5 个百分点才视为明确改进。若 validation 持平但完整 OOD
明显提升，可保留为平台候选，但必须单独标注为“泛化候选”，不得替代验证集冠军。

## 6. 自动报告

沿用现有可复用报告脚本，每个 trial 和全轮结束后生成：

- `metrics.jsonl`、resolved config、代码/Manifest/环境哈希；
- training/validation loss、accuracy、CER、LR、gradient norm 曲线；
- 吞吐、运行时间、GPU 峰值显存；
- train-probe/validation/OOD panel 的预测、错误样例、场景指标和字符混淆；
- 四组数据来源与时长分布；
- 四组 checkpoint 对比 PNG、CSV、JSON 和 HTML。

Round 3 全轮报告写入独立目录，不能覆盖 Round 2 证据。

## 7. 测试与失败处理

本地测试必须覆盖：

- `73%`、`80%`、`85%` 和全部外部数据的确定性构建；
- CV 用尽后只从 MDCC 无放回补充；
- 全量组恰好包含所有官方/CV/MDCC train ID；
- 四组不含任何固定 validation、公开测试或 OOD ID；
- 重跑生成相同 SHA-256；
- Slurm array 的四个 task 映射到四个独立输出目录。

服务器启动训练前必须核对磁盘、四张 GPU、Manifest 行数和哈希。单个 trial
OOM、NaN/Inf、数据读取失败或 checkpoint 评测失败时保留日志并只修复该 trial；
不得删除或重启其他正常作业。

## 8. 交付与平台提交

只有满足验证/CER 护栏的 Round 3 候选才执行完整 27,177 条 OOD 复评和离线打包。
提交包继续保持根目录平铺、恰好一个 `model.safetensors`、断网可加载并兼容官方
`predict.py` 协议。

Codex 负责生成并验证包，随后向用户提供 Mac 本地上传路径、文件大小和 SHA-256；
用户手动上传平台。平台结果只作为最终确认和下一轮诊断。
