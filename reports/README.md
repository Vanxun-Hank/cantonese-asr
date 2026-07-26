# Experiment reports

将经过复核、适合公开的实验结果按轮次放在独立目录中：

```text
reports/
  round1/
  round2/
  round3/
  ...
```

每个 `roundN` 目录应包含该轮最终确认的指标、图表和简短说明。失败、未完成或仅供内部诊断的运行结果不应提交到这里。

当前公开报告：

- [`round_8/report.html`](round_8/report.html)：Round 8 四组来源消融实验；
- [`round_8/diagnosis/report.html`](round_8/diagnosis/report.html)：平台结果与跨来源 OOD 因果诊断；
- [`round2-round5-round6-diagnosis/diagnosis.md`](round2-round5-round6-diagnosis/diagnosis.md)：Round 2、5、6 退步复盘；
- [`common_voice_26_yue_admission_audit/report.html`](common_voice_26_yue_admission_audit/report.html)：Common Voice 26 yue 准入审计。
