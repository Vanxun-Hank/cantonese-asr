# 点心杯粤语 ASR 算法交接（2026-07-26）

本文面向新接手项目的算法、数据和工程成员，记录当前已经完成的工作、关键实验、失败路线、已确认结论、可复现实验入口，以及下一阶段的唯一推荐动作。

## 1. 项目目标与规则边界

比赛任务是在 `openai/whisper-small` 基座上提升粤语自动语音识别能力。当前实现遵守以下约束：

- 最终模型保持 Whisper-small 架构和 tokenizer，不改变层数、hidden size、attention head 或 GELU。
- Full SFT 更新全部可训练参数；LoRA 仅作为已审计但未采用的候选路线。
- 最终提交是一个 Whisper-small student 权重，不做模型集成。
- SenseVoice 只用于外部数据声学质量诊断；Goal 1 尚未开展知识蒸馏。
- 训练标签始终是粤语原文，普通话翻译、粤拼和场景只作为元数据。
- 官方公开自测数据不得进入训练或 checkpoint 选择。
- checkpoint 只按固定 official validation 选择；公开测试、OOD 和平台结果只用于诊断或最终确认。

### 官方一致指标

评测流程为：

1. 仅将 prediction 由繁体转换为简体；
2. 去除参考与预测中的中英文标点和空白；
3. 计算字符 Levenshtein 编辑距离；
4. 编辑距离不超过 2 的句子计为正确；
5. 同时记录 CER。

公式：

```text
sentence_accuracy_tol2
= count(edit_distance(reference, prediction) <= 2) / N

CER
= sum(edit_distance(reference, prediction))
  / sum(number_of_reference_characters)
```

`sentence_accuracy_tol2` 是主要选模指标，CER 是保护指标。训练 loss 是 token cross-entropy；accuracy 和 CER 不参与反向传播。

## 2. 当前系统架构

```text
官方/外部音频与文本
        │
        ▼
数据审计、去重、隔离泄漏、生成 JSONL manifest
        │
        ▼
WhisperFeatureExtractor（16 kHz、log-Mel）
        │
        ▼
openai/whisper-small Full SFT
AdamW + cosine/linear scheduler + bf16
        │
        ▼
每 epoch official validation
        │
        ├── sentence_accuracy_tol2 / CER / validation loss
        ├── train-probe loss / learning rate / throughput
        └── 固定 OOD 诊断（不参与选模）
        │
        ▼
单权重离线 submission.zip
```

训练环境的目标版本为 Python 3.11、PyTorch 2.6、CUDA 12.4、Transformers 4.57.6。正式训练在 Slurm 管理的四张 RTX 4090 上进行；四卡主要用于四个独立单 GPU trial 并行，而不是把一个 trial 做成四卡数据并行。

## 3. 已完成的工程工作

### 3.1 数据整理

已实现：

- 官方 `index.csv` 与音频的唯一匹配；
- 16 kHz 单声道读取，不覆盖原始音频；
- Unicode NFKC、异常空白与可控标点处理；
- 缺失、重复、损坏、空标签及超 30 秒样本隔离；
- 固定 seed 的 train/validation 划分；
- public test 与 validation 隔离；
- 外部数据来源、许可证、split、哈希和处理历史记录；
- 文本、压缩文件字节、解码 PCM 等多层去重；
- protected validation/test 文本或音频重叠隔离；
- 固定 OOD 面板和来源分栏诊断。

### 3.2 训练

`train.py` 已支持：

- 本地离线加载 Whisper processor/model；
- Full SFT；
- AdamW、weight decay、gradient clipping；
- linear/cosine learning-rate scheduler 与 warmup；
- bf16、gradient checkpointing；
- 动态 padding，label padding 转 `-100`；
- SpecAugment；
- encoder 层冻结实验；
- LoRA target module 配置；
- checkpoint 保存、恢复、early stopping；
- per-epoch JSONL/TensorBoard 日志；
- validation accuracy、CER、loss 与 train-probe 诊断。

### 3.3 推理、评测和打包

已实现：

- `predict.py --audio_dir --test_list --output_jsonl`；
- JSONL/CSV 测试清单兼容；
- 输入顺序和原始 `audio_path` 保留；
- generation max length 统一为 225；
- 离线加载和单 `model.safetensors` 平铺 ZIP；
- 输出数量、顺序、路径一致性验证；
- 官方指标、字符混淆和错误样例；
- Round 级 PNG/CSV/JSON/HTML 报告。

历史上曾出现提交包中的 `predict.py` 导入仓库模块、平台环境缺少该模块而产生 `ModuleNotFoundError`。后续提交必须在仓库目录之外的干净临时目录中执行完全离线验证，不能依赖本地 `PYTHONPATH`。

## 4. 数据资产

### 4.1 官方数据

内部固定训练子集为 6,292 条。训练标签采用粤语原文。公开自测集为 1,900 条，仅作阶段性复核，不用于训练和超参数选择。

### 4.2 Common Voice zh-HK

Round 2 / EXP004 使用 8,451 条。该来源更接近日常口语和香港粤语，但文本规范、口音和录音条件更分散。

### 4.3 MDCC

Round 2 / EXP004 使用 8,561 条。该来源整体更正式、更接近朗读或规范表达，在 MDCC OOD 上稳定，但单独提高其权重会造成明显域专化。

### 4.4 Common Voice 26 Cantonese（yue）

已完成 CPU/单 GPU 数据准入审计，没有启动新训练：

- candidate train：7,420 条；
- 通过 QC：7,418 条；
- quarantine：2 条，原因是 protected text overlap；
- 清洗后 protected text/audio overlap：0；
- 新增时长：8.29 小时；
- 相对 EXP004 新增说话人：87；
- Top 10 说话人占比：48.11%；
- 固定声学抽样：229 条；
- SenseVoice 标记为粤语：228 条，粤语率 99.56%；
- 平均规范化诊断 CER：8.81%，中位数 5.88%；
- 精确音频重叠：0。

当前准入结论是 **reject / 暂不直接加入 EXP004**。数据本身质量不差，但没有达到事先锁定的新增说话人门槛 100，且说话人集中度接近 50% 上限。不得在看到 87 后事后把门槛改成 80。

## 5. 各阶段实验复盘

## 5.1 Goal 1 基础 Full SFT

首先完成了：

- 原始 Whisper-small 零样本评测；
- 16–32 条样本过拟合 smoke test；
- 4090 micro-batch 探测；
- learning rate 与 scheduler 四组筛选；
- effective batch 16/32 对照；
- checkpoint、恢复、推理和离线打包验证。

最终稳定起点为：

```text
optimizer: AdamW
learning rate: 2e-5
scheduler: cosine
weight decay: 0.01
effective batch: 16
warmup ratio: 0.05
gradient clipping: 1.0
precision: bf16
```

## 5.2 Round 2：当前核心基线

数据组成：

```text
Official:      6,292
Common Voice:  8,451
MDCC:          8,561
Total:        23,304
External:      73.0%
```

训练配置：

```text
start model: openai/whisper-small
method: Full SFT
optimizer: AdamW
learning rate: 2e-5
scheduler: cosine
epochs: 3
effective batch: 16
weight decay: 0.01
SpecAugment: off
```

结果：

```text
selected epoch: 2（历史 Round 2）
validation accuracy: 84.19%
validation CER: 9.21%
public accuracy: 87.58%
public CER: 9.12%
fixed combined OOD accuracy: 86.75%
fixed combined OOD CER: 10.49%
platform: 62.69
```

Round 8 中用相同数据和训练设置重新训练得到 EXP004：

```text
selected epoch: 3
validation accuracy: 83.90%
validation CER: 9.33%
CV OOD accuracy: 85.60%
MDCC OOD accuracy: 88.20%
combined OOD accuracy: 86.90%
platform: 63.02
```

因此，**EXP004 / Round 2 exact 是当前唯一基线和平台最佳模型**。

## 5.3 Round 5：官方速度三视图

数据：

```text
Official 0.9x: 6,292
Official 1.0x: 6,292
Official 1.1x: 6,292
Total:         18,876
External:      0
```

配置主要为 Full SFT、LR `2e-5`、linear、3 epochs、effective batch 16，无 SpecAugment。

本地 public 指标没有明显退步，但固定 OOD 严重下降：

```text
public accuracy: 87.42%
public CER: 8.60%
combined OOD accuracy: 52.20%
combined OOD CER: 23.53%
```

结论：同域 public 指标无法代表隐藏域；移除外部数据会显著破坏跨来源泛化。该轮同时改变了外部数据、速度视图和 scheduler，不能单独认定 0.9/1.1 速度扰动有害。

## 5.4 Round 6：速度三视图 + 原外部数据 + SpecAugment

数据：

```text
Official speed views: 18,876
Common Voice:          8,451
MDCC:                  8,561
Total:                35,888
External fraction:     47.4%
```

运行了从原始 Whisper-small 开始和从 Round 2 checkpoint 继续训练两种方式，并比较 SpecAugment on/off。

最佳本地结果：

| 版本 | Validation Acc | Validation CER | Public Acc | Public CER | 平台 |
|---|---:|---:|---:|---:|---:|
| Fresh + SpecAugment | 86.61% | 8.27% | 89.84% | 7.98% | 用户记录约 56 |
| Round 2 continue + SpecAugment | 85.47% | 8.39% | 90.26% | 7.76% | 用户记录约 56 |

固定 CV/MDCC OOD 并未整体崩溃，所以不能简单说模型“全面变差”。最可疑的机制是固定三速度视图把每条官方录音重复三次，使外部数据权重从 73% 降到 47.4%，并配合更长训练产生置信度过拟合。

SpecAugment 在同数据成对比较中改善了本地指标，因此现有证据不支持把 SpecAugment 判为主要退步原因。Fresh 与 continue 都退步，也不支持把初始化方式判为主因。

## 5.5 Round 7：LoRA 路线

测试了不同 LoRA target module：

- `q_proj,v_proj`；
- 扩展 attention targets；
- `q/k/v/out + fc1/fc2`。

早期 LoRA 版本出现严重的 225-token runaway repetition、异常字符和大规模插入：

```text
q/v validation accuracy: 4.56%
q/v validation CER: 811.80%
mean worst prediction length: about 221 characters
```

扩展 target 的 adapter 仍明显低于 base；active adapter 与 merged model 也未达到严格逐样本等价，parity audit 未通过。因此：

- Round 7 LoRA 全部 rejected；
- 没有作为平台候选；
- 当前不应继续投入 GPU；
- 若未来重启，必须先解决 label/decoder prompt/merge parity，再做训练。

## 5.6 Round 8：外部数据来源消融

四个 trial 保持 Whisper-small Full SFT、LR `2e-5`、cosine、effective batch 16、无 SpecAugment，主要变量是数据组成：

| EXP | 数据 | Train rows | Val Acc | Val CER | CV OOD Acc | MDCC OOD Acc | 平台 |
|---|---|---:|---:|---:|---:|---:|---:|
| EXP001 | Official + MDCC 73% | 23,304 | 82.76% | 9.31% | 57.0% | 92.0% | 未提交 |
| EXP002 | Official + MDCC 57% | 14,743 | 82.62% | 9.03% | 55.3% | 89.3% | 53.66 |
| EXP003 | Official + CV 57% | 14,743 | 83.05% | 9.33% | 84.8% | 71.7% | 未提交 |
| EXP004 | Official + CV + MDCC | 23,304 | 83.90% | 9.33% | 85.6% | 88.2% | **63.02** |

这是目前最强的因果证据：

- MDCC-only 让 MDCC OOD 变好，但 CV OOD 大幅下降；
- CV-only 让 CV OOD 较好，但 MDCC OOD 下降；
- CV + MDCC 不是简单“数据更多”，而是提供互补域覆盖；
- official validation 对平台结果的解释力有限；
- 隐藏集需要跨域平衡，而不是单一来源极致优化。

## 6. 已确认结论

### 6.1 当前最佳

`EXP004 / Round 2 exact`，平台 `63.02`。

### 6.2 最可能决定平台表现的因素

1. 外部来源是否互补；
2. 不同来源和说话人的采样权重；
3. 平台隐藏样本的域构成；
4. 后期置信度过拟合；
5. 少量 runaway repetition。

### 6.3 暂时没有证据支持的解释

- “数据越多必然越差”；
- “SpecAugment 导致退步”；
- “从 checkpoint 继续训练一定更差”；
- “学习率是 Round 6 的共同主因”；
- “速度 0.9x/1.1x 本身一定有害”。

### 6.4 平台分数的统计限制

平台日志显示每个权重通过 hash 派生抽样 seed，并只评测 200 条。不同权重不一定面对相同 200 条，因此单次平台结果不是严格配对 A/B。平台分应当用于最终确认，不能反向承担全部调参和因果归因。

## 7. 当前所处阶段

截至 2026-07-26：

- 没有新的 ASR 训练作业正在运行；
- Round 8 来源消融已完成；
- EXP004 以 63.02 保持最佳；
- Common Voice 26 yue 数据准入审计已完成；
- 该审计启动训练作业 0 次、平台提交 0 次；
- 当前应暂停盲目扩数据和多变量组合；
- 下一步应先做低成本、单变量的解码重复保护实验。

## 8. 下一步唯一推荐实验

### EXP004 权重不变，仅增加重复生成保护

假设：EXP004 公开评测中观察到的 11 个 225-token 循环属于独立解码故障。

唯一改动：

```text
no_repeat_ngram_size = 2
```

保持不变：

- 模型权重；
- 数据；
- language/task prompt；
- beam；
- max length 225；
- 文本规范化；
- 评测脚本；
- 提交包结构。

预期信号：

- runaway repetition 从 11 降至 0；
- public CER 明显下降；
- official validation accuracy 下降不超过 0.3 个百分点；
- validation CER 不退化。

失败信号：

- 正常粤语重复词被错误抑制；
- validation accuracy 下降超过 0.3 个百分点；
- CER 变差。

失败即放弃，不提交平台。该实验无需训练权重，信息收益高、成本最低。

## 9. GPU 实验 backlog

只有完成上一项验收后，才按顺序考虑：

1. EXP004 + 解码重复保护；
2. EXP004 + 经 speaker-balanced 抽样的 CV26 yue 小子集；
3. EXP004 仅加入在线随机速度扰动，而不是固定三视图；
4. 保持数据不变，只缩短训练/加强 validation-loss guardrail；
5. 保持数据不变，测试分层学习率或冻结底部 encoder 层。

不得并行把新数据、速度增强、SpecAugment、scheduler 和初始化同时改掉。

## 10. 暂时不要做

- 不继续增加 epochs；
- 不从 Round 6 checkpoint 继续长跑；
- 不把官方 validation/public test 放入训练；
- 不把 Common Voice 与 MDCC 合成无法追踪的统一 external 标签；
- 不未经准入审计加入外部数据；
- 不做模型集成；
- 不立即开展 SenseVoice 蒸馏；
- 不重启 LoRA，除非 parity 问题先被解决；
- 不根据单次平台 200 条结果反复猜 generation 参数。

## 11. 交接后的运行入口

关键入口：

```text
train.py
predict.py
scripts/prepare_manifest.py
scripts/build_external_training_mixes.py
scripts/build_mdcc_source_isolation_manifest.py
scripts/build_round8_source_ablation_manifests.py
scripts/evaluate_checkpoints.py
scripts/package_submission.py
scripts/verify_submission.py
scripts/audit_common_voice_yue.py
scripts/audit_yue_acoustic_sample.py
```

服务器作业：

```text
slurm/gpu_preflight.slurm
slurm/smoke.slurm
slurm/grid_search.slurm
slurm/round8_mdcc_source_isolation.slurm
slurm/round8_source_ablation.slurm
slurm/audit_common_voice_yue.slurm
slurm/audit_common_voice_yue_acoustic.slurm
```

每轮必须输出并回答：

1. 发生了什么；
2. 为什么；
3. 证据是什么；
4. 下一轮只推荐哪一个实验。

## 12. 报告入口

- `reports/round_8/report.html`：Round 8 四组 canonical report；
- `reports/round_8/diagnosis/report.html`：Round 8 来源消融与因果诊断；
- `reports/round2-round5-round6-diagnosis/diagnosis.md`：Round 2/5/6 退步复盘；
- `reports/common_voice_26_yue_admission_audit/report.html`：CV26 yue 数据准入结论；
- `reports/common_voice_26_yue_admission_audit/evidence/acoustic_summary.json`：固定声学样本汇总。

公开仓库不包含原始音频、数据集、模型权重、checkpoint、submission ZIP、服务器凭据或临时下载地址。

## 13. 仍缺少的证据

- Round 5 的精确平台分数；
- Round 2 实际上传 ZIP 的 SHA-256；
- 两个 Round 6 平台作品各自的精确分数；
- 更接近平台方言和录音条件、且不来自训练源的独立 OOD；
- 按音频时长、英文混合、数字和粤语片区分桶的完整预测结果；
- 解码重复保护在固定 validation/public 上的配对结果。

接手者应优先补证据，不要用新组合掩盖旧实验无法解释的问题。
