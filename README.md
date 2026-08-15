# Cantonese ASR with Whisper-small

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

粤语自动语音识别（ASR）系统，基于 `openai/whisper-small` 进行 source-alternating adaptation。WenetSpeech-Yue 的 source-pure optimizer step 只更新 Encoder，交替出现的 task-provided step 更新完整模型。

A Cantonese Automatic Speech Recognition system adapted from
`openai/whisper-small`. Source-pure WenetSpeech-Yue optimizer steps update the
encoder, while alternating task-provided steps update the full model.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Training](#training)
- [Evaluation & Reports](#evaluation--reports)
- [Offline Submission](#offline-submission)
- [Config System](#config-system)
- [Scripts Reference](#scripts-reference)
- [Slurm Jobs](#slurm-jobs)
- [Technical Report and Model Release](#technical-report-and-model-release)

---

## Project Structure

```text
cantonese-asr/
├── cantonese_asr/          # Core library (io, metrics, etc.)
│   ├── __init__.py
│   ├── io.py               # Manifest & artifact I/O
│   └── metrics.py          # CER, sentence accuracy, normalization
├── configs/
│   └── rounds/             # Per-round experiment configs (JSON)
│       ├── w500_extension_40_to_100.json
│       └── short_domain_encoder_adapt.json
├── scripts/                # Data prep, training, evaluation, analysis
│   ├── prepare_*.py        # Manifest preparation
│   ├── download_*.py       # Asset downloaders
│   ├── evaluate_*.py       # Evaluation & scoring
│   ├── predict_*.py        # Inference scripts
│   ├── train_*.py          # Specialized training variants
│   ├── analyze_*.py        # Analysis & diagnostics
│   ├── audit_*.py          # Data quality audits
│   ├── build_*.py          # Training mix builders
│   └── ...                 # See [Scripts Reference](#scripts-reference)
├── slurm/                  # Slurm job definitions
├── tests/                  # Unit & integration tests
├── reports/                # Experiment reports (by round)
├── train.py                # Main training entry point
├── predict.py              # Main prediction entry point
├── predict_guarded.py      # Guarded prediction with safety checks
├── requirements*.txt       # Dependency specs
└── README.md
```

---

## Features

- **Data Pipeline** — 整理官方及外部粤语音频—文本数据，生成可复现的 JSONL manifest。
- **Full SFT Training** — AdamW 优化器 + 学习率调度，支持 Slurm 集群训练。
- **Comprehensive Metrics** — 记录 loss、整句准确率 (sentence accuracy)、CER、学习率、显存和吞吐。
- **Rich Reporting** — 自动生成 PNG、CSV、JSON 和 HTML 实验报告。
- **Checkpoint Selection** — 根据固定验证集选择最优 checkpoint，并用独立 OOD 数据诊断泛化能力。
- **Offline Packaging** — 生成根目录平铺、只包含 `model.safetensors` 的离线提交包。
- **Advanced Decoding** — 支持 beam search、n-best rescoring、character-level LM 集成。
- **Distillation** — 支持 teacher model（如 SenseVoice、FunASR）蒸馏训练。
- **Domain Adaptation** — 支持 domain-specific encoder 和定向训练。

---

## Requirements

### Base

```bash
pip install -r requirements.txt
```

### Server / Training

```bash
pip install -r requirements-server-torch.txt
```

### Submission Packaging

```bash
pip install -r requirements-submission.txt
```

Core dependencies: `torch`, `transformers`, `datasets`, `librosa`, `jiwer`, `editdistance`.

---

## Quick Start

```bash
# 1. 安装环境与依赖
bash scripts/server_bootstrap.sh

# 2. 下载、解压并整理官方数据
python scripts/download_assets.py
python scripts/extract_assets.py
python scripts/prepare_manifest.py \
  --index-csv artifacts/datasets/official/index.csv \
  --audio-root artifacts/data/train_raw \
  --output-dir artifacts/manifests \
  --exclude-test-list artifacts/datasets/official/template_pre.jsonl

# 3. 训练前检查与基线
sbatch slurm/gpu_preflight.slurm
sbatch slurm/zero_shot.slurm
sbatch slurm/smoke.slurm
sbatch slurm/memory_probe.slurm

# 4. 启动超参数实验
BATCH_SIZE=4 GRAD_ACCUM=4 bash scripts/submit_grid.sh
```

服务器目录、Python 环境和远程主机可通过环境变量配置：

| Variable | Purpose |
|---|---|
| `PROJECT_DIR` | 项目根目录 |
| `ENV_DIR` | Python 虚拟环境路径 |
| `PYTHON` | Python 解释器路径 |
| `REMOTE_HOST` | 远程训练主机 |
| `REMOTE_DIR` | 远程项目目录 |

---

## Data Format

训练 manifest 为 JSONL 格式，每行一个样本：

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

- `text` — 粤语原文（监督微调标签），普通话翻译、粤拼、场景等可作为元数据保留，但不替代粤语原文。
- `audio_path` — 16kHz 单声道 WAV。
- `source` — 数据来源 (`official` / `external`)。
- `split` — 数据集划分 (`train` / `val` / `test` / `ood`)。

---

## Training

### 基本训练 (Main Entry)

```bash
python train.py \
  --manifest artifacts/manifests/train.jsonl \
  --val-manifest artifacts/manifests/val.jsonl \
  --output-dir outputs/run_01 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-5 \
  --epochs 5
```

### 定向 / 扩展训练

```bash
# 领域定向训练
python scripts/train_directional.py --config configs/rounds/w500_extension_40_to_100.json

# W500 扩展训练
python scripts/train_w500_extension.py --config configs/rounds/w500_extension_40_to_100.json

# 蒸馏训练
python scripts/train_clean_char_lm.py --teacher funasr_nano --output-dir outputs/distill
```

### Guarded Prediction

`predict_guarded.py` 提供带安全检查的推理，适用于生产环境：

```bash
python predict_guarded.py \
  --checkpoint outputs/run_01/checkpoint-5000 \
  --manifest artifacts/manifests/test.jsonl \
  --output predictions.jsonl
```

---

## Evaluation & Reports

### 核心指标

| Metric | Description |
|---|---|
| `sentence_accuracy_tol2` | 字符编辑距离 ≤ 2 的句子比例（主指标） |
| `cer` | 字符错误率 (Character Error Rate) |
| `loss` | 交叉熵损失 |

### 生成报告

```bash
# 单轮实验报告
python scripts/plot_experiments.py --results-dir outputs/run_01 --output-dir reports/roundN

# Checkpoint 对比
python scripts/evaluate_checkpoints.py --checkpoint-dir outputs/run_01

# OOD 泛化诊断
python scripts/evaluate_predictions.py --predictions pred.jsonl --reference ref.jsonl

# N-best Oracle 分析
python scripts/analyze_nbest_oracle.py --nbest nbest.jsonl
```

报告包含：checkpoint 对比曲线、错误样例、字符混淆矩阵、场景分解指标和 OOD 诊断。

每轮最终确认结果放入 `reports/roundN/`，失败或未完成的实验不发布。

---

## Offline Submission

```bash
# 打包提交
python scripts/package_submission.py \
  --checkpoint outputs/best/checkpoint-5000 \
  --output submission.zip

# 离线验证
python scripts/verify_submission.py --submission submission.zip
```

`package_submission.py` 创建平铺 ZIP，校验恰好包含一个 `model.safetensors`。
`verify_submission.py` 在断网模式下检查模型加载、预测输出数量、顺序和 `audio_path` 一致性。

---

## Config System

实验配置通过 JSON 文件管理，放置在 `configs/rounds/` 下。每个配置文件定义一轮实验的参数：

```json
{
  "round": "w500_extension_40_to_100",
  "base_checkpoint": "outputs/w500_40/checkpoint-20398",
  "training": {
    "epochs": 60,
    "batch_size": 4,
    "grad_accum": 4,
    "lr": 2.5e-5
  },
  "data": {
    "train_manifest": "artifacts/manifests/w500_extension_train.jsonl",
    "val_manifest": "artifacts/manifests/val.jsonl"
  }
}
```

---

## Scripts Reference

### Data Preparation

| Script | Purpose |
|---|---|
| `scripts/prepare_manifest.py` | 生成标准训练/验证 manifest |
| `scripts/prepare_ood_manifest.py` | 生成 OOD 测试 manifest |
| `scripts/prepare_external_manifest.py` | 整合外部数据源 |
| `scripts/download_assets.py` | 下载官方数据集 |
| `scripts/download_mdc_dataset.py` | 下载 MDC 数据集 |
| `scripts/download_teacher_model.py` | 下载 teacher 模型权重 |
| `scripts/extract_assets.py` | 解压并整理原始数据 |
| `scripts/build_external_training_mixes.py` | 构建外部数据混合训练集 |

### Training

| Script | Purpose |
|---|---|
| `train.py` | 主训练入口 (Full SFT) |
| `scripts/train_directional.py` | 定向训练（领域适配） |
| `scripts/train_w500_extension.py` | W500 扩展训练 |
| `scripts/train_clean_char_lm.py` | 字符级 LM 训练 |
| `scripts/interpolate_checkpoints.py` | Checkpoint 插值合并 |
| `scripts/merge_checkpoint_regions.py` | 区域级 checkpoint 合并 |
| `scripts/compose_decoder_layers.py` | Decoder 层组合 |

### Prediction & Inference

| Script | Purpose |
|---|---|
| `predict.py` | 标准预测 |
| `predict_guarded.py` | 带安全检查的预测 |
| `scripts/predict_local_guarded.py` | 本地 guarded 预测 |
| `scripts/predict_beam2_norepeat4.py` | Beam search (beam=2, no repeat trigrams) |
| `scripts/predict_true_nbest_clean_lm.py` | N-best + clean LM rescoring |
| `scripts/predict_sensevoice_manifest.py` | SenseVoice teacher 预测 |
| `scripts/predict_funasr_nano_manifest.py` | FunASR nano teacher 预测 |

### Evaluation & Analysis

| Script | Purpose |
|---|---|
| `scripts/evaluate_predictions.py` | 预测结果评测 |
| `scripts/evaluate_checkpoints.py` | Checkpoint 对比评测 |
| `scripts/evaluate_manifest_loss.py` | Manifest 级 loss 评测 |
| `scripts/evaluate_nbest_char_lm.py` | N-best + char LM 评测 |
| `scripts/evaluate_closed_set_text_retrieval.py` | 闭集文本检索评测 |
| `scripts/plot_experiments.py` | 实验报告可视化 |
| `scripts/select_best_checkpoint.py` | 最优 checkpoint 选择 |
| `scripts/select_global_candidate.py` | 全局候选模型选择 |
| `scripts/compare_predictions_by_scene.py` | 按场景对比预测 |
| `scripts/analyze_nbest_oracle.py` | N-best oracle 上限分析 |
| `scripts/analyze_w500_domain_coverage.py` | W500 领域覆盖分析 |
| `scripts/analyze_closed_set_hard_subsets.py` | 闭集难样本分析 |
| `scripts/analyze_teacher_pairwise.py` | Teacher 模型 pairwise 对比 |

### Auditing & Quality

| Script | Purpose |
|---|---|
| `scripts/audit_lm_validation_overlap.py` | 检查 LM 训练/验证重叠 |
| `scripts/audit_closed_set_candidates.py` | 闭集候选质量审计 |
| `scripts/audit_official_audio_duplicates.py` | 官方音频重复检测 |
| `scripts/verify_cv_audio_origin.py` | CommonVoice 音频来源验证 |
| `scripts/score_cv_origin_labels.py` | CV 来源标签评分 |
| `scripts/check_metric_parity.py` | 指标一致性检查 |
| `scripts/validate_prediction_order.py` | 预测顺序校验 |

### Submission & Packaging

| Script | Purpose |
|---|---|
| `scripts/package_submission.py` | 创建离线提交包 |
| `scripts/verify_submission.py` | 离线提交包验证 |
| `scripts/check_submission.py` | 提交前检查 |

### Misc Builders

| Script | Purpose |
|---|---|
| `scripts/build_loss_filtered_manifests.py` | 基于 loss 的样本过滤 |
| `scripts/build_distillation_manifest.py` | 蒸馏训练 manifest 构建 |
| `scripts/build_wenet_teacher_challenge.py` | Wenet teacher 挑战集构建 |
| `scripts/build_short_domain_encoder_stream.py` | 短领域编码器数据流 |
| `scripts/retrieve_closed_set_formal.py` | 闭集正式文本检索 |
| `scripts/attribute_formal_text_domain.py` | 正式文本领域归因 |

---

## Slurm Jobs

所有 Slurm 作业脚本位于 `slurm/` 目录，覆盖：

- **Training** — `train_*.slurm`, `user_*.slurm`
- **Evaluation** — `evaluate_*.slurm`
- **Verification** — `verify_*.slurm`
- **Analysis** — `analyze_*.slurm`
- **Preparation** — `prepare_*.slurm`
- **Generation** — `generate_*.slurm`

使用方式：

```bash
sbatch slurm/train_w500_extension.slurm
```

---

## License

The released model weights are available under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). This model
license does not replace the separate terms attached to upstream datasets,
third-party models, or individual software dependencies.

---

## Technical Report and Model Release

The released source-alternating Whisper-small system uses two update scopes in
one fixed cycle:

- source-pure WenetSpeech-Yue steps update the encoder;
- task-provided Cantonese steps update the full learnable model.

The released checkpoint is the final 50 h milestone of this curriculum. Its
fixed inference configuration uses `language=zh`, `task=transcribe`, one beam,
`max_length=225`, `no_repeat_ngram_size=4`, and
`repetition_penalty=1.05`.

### Evaluation

All accuracy and CER values are percentages. The fixed validation set controls
checkpoint selection. The task-provided local test set and OOD set provide
additional evaluation and are not used to choose checkpoints, language-model
weights, or fusion methods.

| Evaluation surface | Items | tol2 accuracy (%) | CER (%) |
|---|---:|---:|---:|
| Fixed validation set | 702 | 85.47 | 8.59 |
| Task-provided local test set | 1,900 | 89.42 | 8.13 |
| OOD set | 2,000 | 33.15 | 34.04 |

The OOD result motivates robustness-oriented work across a wider range of
speakers, acoustic conditions, and transcription conventions.

### P0-P2 evidence

P0 and paired-seed P1 record the completed decoding, task-matched recentering,
augmentation, and checkpoint-selection decisions. P2 is a matched-budget
early-adaptation study: six Small/Medium/Large-v2 Full-SFT/LoRA arms each use
150 optimization steps, 2,400 ordered training examples, global batch 16, and
a single seed. It measures early adaptation and resource trade-offs under that
fixed budget. The released source-alternating Whisper-small checkpoint remains
the public system.

### Data provenance

The task-provided Cantonese materials were distributed for the preliminary ASR
task of the [AI Dimsum
Cup](https://www.aicompetition-pz.com/topic_detail/19), whose task page points to
the [Cantonese Life Scenarios
Corpus](https://huggingface.co/datasets/leeduckgo/cantonese-life-scenarios-corpus).
The current dataset page documents provenance; it is not presented as the
exact training-time revision. Training data is not redistributed with the
model, and users must follow each source's terms. Information identifying the
exact training-time task-data snapshot is available from the corresponding
author by email; the contact address will be added with the final author
record.

### Public artifacts

- [Technical Report](paper/manuscript.md)
- [Bibliography](paper/references.bib)
- [Hugging Face model](https://huggingface.co/cantonese-asr-lab/whisper-small-cantonese-w500-adaptive)
- [Source-alternating method description](docs/W500_ADAPTIVE_METHOD.md)

The released `model.safetensors` SHA-256 is
`a0f29a5a011213d5e4de34c40a02d42247255645f06e649e092d2dc495094370`.
A fixed 32-item offline audit found identical text and token outputs across the
raw checkpoint, extracted release package, and bundled inference path.

### Scope

The report evaluates one task-defined character protocol and one OOD surface.
P2 covers fixed-budget early adaptation rather than convergence. The final
author record, corresponding-author email, and AI-use disclosure remain to be
confirmed.
