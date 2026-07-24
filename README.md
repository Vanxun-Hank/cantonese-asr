# Cantonese ASR with Whisper-small

本项目提供一套基于 `openai/whisper-small` 的粤语自动语音识别训练、评测和离线打包流程。模型保持 Whisper-small 架构与 tokenizer 不变，通过监督微调适配粤语音频—文本数据。

## 功能

- 整理官方及外部粤语音频—文本数据，并生成可复现的 JSONL manifest。
- 使用 Full SFT、AdamW、学习率调度和 Slurm 训练 Whisper-small。
- 记录 loss、整句准确率、CER、学习率、显存和吞吐等实验指标。
- 生成 PNG、CSV、JSON 和 HTML 实验报告。
- 根据固定验证集选择 checkpoint，并使用独立 OOD 数据诊断泛化能力。
- 生成根目录平铺、只包含一个 `model.safetensors` 的离线提交包。

## 数据格式

训练 manifest 的核心字段如下：

```json
{
  "id": "00001",
  "audio_path": "artifacts/data/train_raw/00001.wav",
  "text": "你好！",
  "duration_s": 1.24,
  "source": "official",
  "split": "train"
}
```

其中 `text` 是监督微调标签。普通话翻译、粤拼和场景等字段可以保留为元数据，但不替代粤语原文。

## 基本流程

```bash
# 安装环境和依赖
bash scripts/server_bootstrap.sh

# 下载、解压并整理官方数据
python scripts/download_assets.py
python scripts/extract_assets.py
python scripts/prepare_manifest.py \
  --index-csv artifacts/datasets/official/index.csv \
  --audio-root artifacts/data/train_raw \
  --output-dir artifacts/manifests \
  --exclude-test-list artifacts/datasets/official/template_pre.jsonl

# 训练前检查和基线
sbatch slurm/gpu_preflight.slurm
sbatch slurm/zero_shot.slurm
sbatch slurm/smoke.slurm
sbatch slurm/memory_probe.slurm

# 启动超参数实验
BATCH_SIZE=4 GRAD_ACCUM=4 bash scripts/submit_grid.sh
```

服务器目录、Python 环境和远程主机可通过 `PROJECT_DIR`、`ENV_DIR`、`PYTHON`、`REMOTE_HOST` 与 `REMOTE_DIR` 环境变量配置。

## 评测与报告

主指标是 `sentence_accuracy_tol2`：预测文本与参考文本经过统一规范化后，字符编辑距离不超过 2 的句子计为正确。项目同时记录 CER，并提供 checkpoint 对比、错误样例、字符混淆、场景指标和 OOD 诊断。

每轮可复用 `scripts/plot_experiments.py` 生成报告。适合公开的实验结果统一放在 `reports/roundN/`，每个目录只包含对应轮次的最终确认结果；失败或未完成的实验不发布。

## 离线提交

`scripts/package_submission.py` 会创建平铺 ZIP，并校验其中恰好包含一个 `model.safetensors`。`scripts/verify_submission.py` 可在断网模式下检查模型加载、预测输出数量、顺序和 `audio_path` 一致性。
