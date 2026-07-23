# 分轮粤语 ASR 实验报告设计

## 目标

将实验报告按 Round 隔离，避免将 smoke、显存探测、不同数据配方和不同 scheduler 混在同一张曲线图中。每份 Round 报告应能独立解释模型、数据、训练参数、最佳 checkpoint、权重、评测结果及下一轮决策。

## 输出结构

每次报告运行写入：

```text
reports/<round>/
  report.html
  experiment_summary.json
  selected_model.json
  artifacts.json
  figures/
  appendix/
```

`selected_model.json` 记录最佳 trial、计划/完成/选中 epoch、checkpoint、验证指标及选模理由。`artifacts.json` 记录模型权重和 submission ZIP 的路径、大小、SHA-256；不复制权重。

## Report 主体

1. Executive summary：本轮目标、最佳 trial、Full SFT/LoRA 状态、计划/完成/选中 epoch、validation、OOD 与平台结果。
2. Data contract：官方/外部样本数、时长、隔离边界、QC。
3. Model and training configuration：Whisper-small、Full SFT 参数范围、LoRA 参数（未使用时明确 N/A）、loss、optimizer、scheduler、batch、precision、seed。
4. Experiment matrix：每个可比 trial 的数据配方、全部关键超参数、epoch 与结果。
5. Main figures：trial accuracy/CER 比较、最佳模型 2x2 训练曲线、最小样本数过滤后的场景表现；外部数据轮额外放 OOD 对称归一化指标。
6. Artifacts：可复现权重、checkpoint、submission、配置与哈希。
7. Decision：本轮结论、未选模型及下一轮单变量。

## 图表分层

主报告只保留决策图。梯度范数、学习率、显存、吞吐、train-probe、generalization gap、teacher-forced OOD loss、confusion 与错误样例进入 appendix。每张图由 HTML caption 明确显示：指标名称、公式、用途、是否用于选模。

## 公式

- Token cross-entropy: `L = -(1/N) Σ log p(y_t | y_<t, x)`。
- Tolerance-2 accuracy: `A_≤2 = (1/M) Σ 1[D(norm(r_i), norm(h_i)) ≤ 2]`。
- CER: `CER = Σ D(norm(r_i), norm(h_i)) / Σ max(1, |norm(r_i)|)`。
- Generalization gap: `G = A_train-probe - A_validation`。
- Gradient L2 norm: `||∇θL||₂ = sqrt(Σ_j (∂L/∂θ_j)²)`。
- Source count: `n_k = Σ_i 1[source_i = k]`；source hours: `H_k = Σ duration_i / 3600`。

## Round 边界

Round 1 仅包含本轮 baseline、grid、batch 与官方数据对照 trial。Round 2 仅包含外部数据配方及其诊断。跨轮仅在 `summary/` 使用两轮最佳 checkpoint 进行简洁比较；不得混合训练曲线。

## 验收

- 同一个 Round 不出现其他 Round 的 trial。
- 每份报告有 `selected_model.json` 与 `artifacts.json`。
- 主图无 smoke/memory 诊断 trial。
- 每张嵌入图都具备公式、用途与选模状态。
- 数据缺失时报告可生成，并显示可理解的缺失说明。
