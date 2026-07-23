# 点心杯粤语 ASR：Whisper-small Full SFT

本项目实现“Mac 本地开发 + `/home/bolin` Slurm 训练”的初赛 Goal 1。基座固定为 `openai/whisper-small`，不修改架构或 tokenizer，不使用 SenseVoice、蒸馏、模型融合或未经批准的外部数据。

## 核心规则

- 训练标签只使用 `index.csv` 的“粤语原文”。
- 官方主指标：预测繁转简并去标点/空白后，字符编辑距离 ≤ 2 的整句计为正确；CER 为保护指标。
- `train_probe` 只诊断过拟合；checkpoint 和模型选择只使用 internal validation。
- `test_audio.zip` 与 `template_pre.jsonl` 不进入训练和超参数选择。
- 正式提交是根目录平铺、只有一个 `model.safetensors` 的离线 ZIP。

## 快速入口

完整命令和每一阶段的 gate 见 [服务器运行手册](docs/RUNBOOK.md)。

```bash
# Mac：同步代码（排除数据、模型和输出）
bash scripts/sync_server.sh

# Server login node：环境、下载、解压、manifest
bash scripts/server_bootstrap.sh
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/download_assets.py
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/extract_assets.py
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/prepare_manifest.py \
  --index-csv artifacts/datasets/official/index.csv \
  --audio-root artifacts/data/train_raw \
  --output-dir artifacts/manifests \
  --exclude-test-list artifacts/datasets/official/template_pre.jsonl

# Gate 顺序：GPU → zero-shot → smoke → 显存探测 → 四卡三轮
sbatch slurm/gpu_preflight.slurm
sbatch slurm/zero_shot.slurm
sbatch slurm/smoke.slurm
sbatch slurm/memory_probe.slurm
BATCH_SIZE=4 GRAD_ACCUM=4 bash scripts/submit_grid.sh
```

每轮结束固定调用 `scripts/plot_experiments.py`，生成 PNG、CSV、JSON 和 HTML；无需让 Agent 重写绘图逻辑。

## 本地验证

```bash
conda run -n dl python -m pytest -q
python3 -m py_compile train.py predict.py cantonese_asr/*.py scripts/*.py
bash -n scripts/*.sh slurm/*.sh slurm/*.slurm
```

教学 Notebook：[fine_tune_whisper_cantonese.ipynb](notebooks/fine_tune_whisper_cantonese.ipynb)。它用于数据抽查和理解训练流程，不承担正式长训练。
