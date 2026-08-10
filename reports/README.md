# Experiment Reports

将经过复核、适合公开的实验结果按轮次放在独立目录中。每个 `roundN` 目录包含该轮最终确认的指标、图表和简短说明。失败、未完成或仅供内部诊断的运行结果不应提交到这里。

Organize verified, publication-ready experiment results by round in separate directories. Each `roundN` directory should contain the final confirmed metrics, plots, and a brief summary for that round. Failed, incomplete, or internal-only diagnostic runs should not be committed here.

---

## Directory Structure

```text
reports/
├── README.md
├── round1/           # Baseline Whisper-small SFT
│   ├── summary.md    # Round summary & key findings
│   ├── metrics.json  # Structured metrics dump
│   ├── cer_curve.png
│   ├── loss_curve.png
│   └── confusion_matrix.png
├── round2/
│   └── ...
└── roundN/
    └── ...
```

## Round Summary Template

每个 `roundN/summary.md` 建议包含以下内容：

```markdown
# Round N: <实验标题>

**日期**: YYYY-MM-DD
**基座模型**: whisper-small (checkpoint-XXXXX)
**训练数据**: 官方 + <外部数据来源>
**训练配置**: batch_size=X, lr=Xe-X, epochs=X

## 核心结果

| 指标 | 验证集 | OOD |
|------|--------|-----|
| sentence_accuracy_tol2 | XX.X% | XX.X% |
| CER | X.XX% | X.XX% |

## 关键发现

- 发现 1
- 发现 2

## 结论与下一步

- 结论
- 下一步计划
```

## Adding a New Round

```bash
mkdir -p reports/roundN
python scripts/plot_experiments.py \
  --results-dir outputs/run_XX \
  --output-dir reports/roundN
# 手动编写 summary.md 并提交
```
