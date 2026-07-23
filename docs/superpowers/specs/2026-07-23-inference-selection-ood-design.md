# Whisper 推理一致性、选模护栏与 OOD 诊断设计

日期：2026-07-23  
状态：设计已由用户批准，待规格复核后实施  
范围：修复推理/验证协议一致性，强化 checkpoint 选择，建立外部 OOD 诊断；不重新训练现有权重，不改变固定官方 validation

## 1. 背景与目标

外部数据 Round 1 已完成四组 Full SFT。训练时验证显式使用
`generation_max_length=225`，而离线 `predict.py` 未设置最大生成长度。控制组
epoch 3 因少数长幻觉样本出现训练时 CER `0.1072`、离线 CER `0.1242`
的不一致；临时加入 `max_length=225` 后，离线 CER 精确恢复为 `0.1072`。

本设计实现以下目标：

1. 训练验证、checkpoint 复评、离线提交统一使用 225-token 最大生成长度。
2. checkpoint 选择强制满足 `CER <= 0.1163`，并使用确定性并列规则。
3. 将 Common Voice 与 MDCC publisher validation/test 建成只读 OOD 诊断面板。
4. 在不重新训练现有权重的情况下，按修复后的统一协议重评当前四组模型。

## 2. 数据边界

### 2.1 固定官方 validation

固定官方 validation 共 702 条，是唯一允许用于以下决策的数据：

- checkpoint 与最佳 epoch 选择；
- early stopping；
- trial/recipe 比较；
- `sentence_accuracy_tol2 >= 0.8219` 与 `CER <= 0.1163` 验收。

不得改变其样本、split、文本规范化或官方指标实现。

### 2.2 外部 OOD 数据

OOD 原始候选由以下 publisher split 构成：

| 数据源 | Split | 原始行数 | 角色 |
|---|---|---:|---|
| Common Voice 26.0 zh-HK | dev | 5,604 | OOD only |
| Common Voice 26.0 zh-HK | test | 5,604 | OOD only |
| MDCC | validation | 5,663 | OOD only |
| MDCC | test | 12,492 | OOD only |
| 合计 |  | 29,363 | QC 前 |

OOD 数据不得进入训练、early stopping、checkpoint 选择或 recipe 选择。公开测试与
平台隐藏集边界保持不变。

### 2.3 两级 OOD 评测

- **固定快速面板**：每个上述 source/split 在 QC 后用 seed 42 无放回抽取最多
  500 条，最多 2,000 条；每个 epoch checkpoint 都评测。
- **完整 OOD**：最终候选对 QC 后全部可用 OOD 数据评测一次；结果仅用于最终泛化
  审计，不得推翻官方 validation 的选模结果。

固定面板保存样本 ID、`audio_path`、source、publisher split、duration、文本、
音频 SHA-256、Manifest SHA-256 和抽样 seed，保证跨实验完全一致。

## 3. 推理协议

`predict.py` 新增：

```text
--generation-max-length 225
```

约束如下：

- 参数必须为正整数；默认 225。
- 调用 `model.generate()` 时显式传入 `max_length`。
- 训练默认值、独立 checkpoint 评测、OOD 评测、离线包和平台提交均使用 225。
- 保留 `num_beams=1`、`language="zh"`、`task="transcribe"` 和现有文本输出协议。
- 不修改权重、架构、Tokenizer 或 vocabulary。

由此消除长幻觉导致的训练/离线 CER 不一致，同时使提交行为可复现。

## 4. Checkpoint 选择

`scripts/select_best_checkpoint.py` 增加显式硬护栏：

```text
--max-cer 0.1163
--min-sentence-accuracy 0.8219
```

选择流程：

1. 只读取固定官方 validation 的独立离线复评结果。
2. 过滤掉 CER 大于 `0.1163` 或 tol2 准确率低于 `0.8219` 的 checkpoint。
3. 在合格 checkpoint 中依次比较：
   1. `sentence_accuracy_tol2` 更高；
   2. CER 更低；
   3. validation loss 更低；
   4. epoch/step 更早。
4. 若没有 checkpoint 通过护栏，脚本失败并输出全部候选及失败原因，禁止打包。

准确率与 CER 必须来自统一 225-token 的独立离线生成。validation loss 从该 run 的
`metrics.jsonl` 按 checkpoint `global_step` 精确关联；`best_model` 先通过
`trainer_state.json.best_model_checkpoint` 解析回来源 step。缺失、重复或无法唯一关联
loss 时，该候选不得用静默默认值参与并列选择，选模器必须失败并报告关联错误。

不得使用 train-probe、OOD、公开测试或平台隐藏成绩替代官方 validation 选模。

## 5. OOD 数据准备

新增独立 OOD 准备脚本，复用现有外部数据 QC 与文本规则，但输出目录和字段明确标记
`role="ood_diagnostic"`：

- Common Voice 从 `dev.tsv`、`test.tsv` 读取 `sentence`；只解压对应 MP3。
- MDCC 从 validation/test Parquet 读取 `transcript` 并原子物化音频。
- 隔离空标签、缺失/损坏音频、非正时长、超过 30 秒、重复 source ID、重复音频及
  溯源不清样本。
- OOD 与训练、固定官方 validation、公开测试做音频 SHA-256 和受保护规范化文本检查；
  重叠样本进入 quarantine。
- 输出按 source/split 分开的完整 Manifest、固定 2,000 条以内 panel Manifest、
  quarantine 和 `data_report.json`。

OOD 音频与 Manifest 仅保存在服务器 `/home/bolin/cantonese-asr`，不复制到 Mac。

## 6. 评测与报告数据流

```text
checkpoint
   ├── 固定官方 validation ──> 唯一选模指标与 CER 护栏
   ├── train-probe ──────────> 过拟合诊断，不选模
   └── 固定 OOD panel ───────> 跨域诊断，不选模

最终 validation 选中的 checkpoint
   └── 完整 OOD ─────────────> 最终泛化审计，不重新选模
```

每个 OOD split 保存 loss、tol2/exact accuracy、CER、总编辑距离、预测、参考、
场景/来源指标、字符 substitutions/insertions/deletions 和错误样例。报告脚本增加 OOD
面板曲线和 source/split 对比，但 scoreboard 的模型排名仍只读取官方 validation。

生成指标与 loss 使用两个明确但共享 Manifest 的评测路径：

- `predict.py` 负责 225-token 生成，再由官方指标实现计算准确率、CER 与错误分析。
- 独立 teacher-forced loss 评测器使用同一 processor、checkpoint、audio/text Manifest
  和标签 padding 规则执行只读 forward；不反向传播、不更新权重，输出样本数、token
  数与加权平均 loss。

两条路径均记录 checkpoint、Manifest SHA-256、processor、dtype、batch size 和代码
SHA-256。样本数或 Manifest hash 不一致时，OOD/validation 报告视为无效。

## 7. 当前四组重评

修复完成后，不重训下列权重：

1. official-only control；
2. source-balanced 25% external；
3. source-balanced 50% external；
4. external pre-adaptation followed by official Full SFT。

对每个保存的 epoch checkpoint 重新运行统一 225-token 官方 validation、train-probe
和固定 OOD panel；随后用新选模器生成选择报告并刷新 PNG/CSV/JSON/HTML 报告。

## 8. 测试与验收

必须新增或更新以下自动测试：

- `predict.py` 默认和显式 generation max length 均传入 `model.generate()`，非法值失败。
- 训练配置与提交推理默认生成长度均为 225。
- checkpoint step 与训练 validation loss 必须一一关联；缺失或歧义时选模失败。
- CER 超限 checkpoint 无资格入选。
- 并列时按准确率、CER、validation loss、较早 epoch 的顺序选择。
- 无合格 checkpoint 时选模器明确失败。
- OOD panel 固定 seed 重跑字节一致，每 source/split 不超过 500 条。
- OOD 与训练/官方 validation/公开测试无 ID、音频哈希及受保护文本泄漏。
- OOD 字段、publisher split、许可证与 provenance 完整。
- OOD teacher-forced loss 只做 forward，样本数/Manifest hash 与生成评测完全一致。
- 报告中的 OOD 指标不影响 scoreboard 排名。

服务器测试全部通过后，才允许重新评测、生成候选提交包和平台确认。

## 9. 错误处理与可恢复性

- OOD 某个 split 缺文件或无法完成 QC 时，整体准备失败，不以较小样本静默替代。
- 已原子物化且哈希正确的音频允许断点复用；原始 archive/Parquet 永不覆盖。
- 评测失败保留日志和已完成 split，重跑时可覆盖派生产物，不改 checkpoint。
- Qwen 作业 958/959 继续保持 Slurm hold；需要恢复时执行
  `scontrol release 958 959`，不修改 watchdog 或服务脚本。

## 10. 非目标

- 不重新训练当前四组权重。
- 不修改 Whisper-small 架构或 Tokenizer。
- 不加入 LoRA、SenseVoice、蒸馏、伪标签或模型集成。
- 不用 OOD、公开测试或平台隐藏成绩选择 checkpoint。
