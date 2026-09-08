<div align="center">

# 粤语语音识别：Whisper-small 微调

**面向粤语的 `whisper-small` 全量监督微调，附一条诚实、可复现的实验轨迹。**
不改架构、不改 tokenizer——所有增益都来自数据筛选与训练课程，且每一项结论都先在留出集上验证过才被称为结果。

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-whisper--small--cantonese--w500--adaptive-yellow.svg)](https://huggingface.co/Vanxun-Hank/whisper-small-cantonese-w500-adaptive)

[English](README.md) · **简体中文**

</div>

---

粤语自动语音识别（ASR）系统，基于 `openai/whisper-small` 微调。保持 Whisper-small 模型架构与 tokenizer 不变，通过监督微调（Full SFT）适配粤语音频—文本数据。

A Cantonese Automatic Speech Recognition system fine-tuned from `openai/whisper-small`. The model architecture and tokenizer are kept unchanged; adaptation is done via full supervised fine-tuning on Cantonese audio-text pairs.

---

## 目录

- [项目结构](#项目结构)
- [特性](#特性)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
- [数据格式](#数据格式)
- [训练](#训练)
- [评测与报告](#评测与报告)
- [离线提交](#离线提交)
- [配置系统](#配置系统)
- [脚本索引](#脚本索引)
- [Slurm 作业](#slurm-作业)
- [许可](#许可)
- [当前公开最佳方法](#当前公开最佳方法)
- [更新 2026-09-08：P2 容量矩阵已收敛](#更新-2026-09-08p2-容量矩阵已收敛)
- [当前分数与榜单竞争力](#当前分数与榜单竞争力)
- [改进路线图：从 69.49 到更强榜单表现](#改进路线图从-6949-到更强榜单表现)

---

## 项目结构

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

## 特性

- **Data Pipeline** — 整理官方及外部粤语音频—文本数据，生成可复现的 JSONL manifest。
- **Full SFT Training** — AdamW 优化器 + 学习率调度，支持 Slurm 集群训练。
- **Comprehensive Metrics** — 记录 loss、整句准确率 (sentence accuracy)、CER、学习率、显存和吞吐。
- **Rich Reporting** — 自动生成 PNG、CSV、JSON 和 HTML 实验报告。
- **Checkpoint Selection** — 根据固定验证集选择最优 checkpoint，并用独立 OOD 数据诊断泛化能力。<sub>注意（2026-09-08）：OOD 面的参考是繁体，而打分器会把预测强制转简，所以它的原始数字更多在测字形而不是识别 —— 见收敛报告。</sub>
- **Offline Packaging** — 生成根目录平铺、只包含 `model.safetensors` 的离线提交包。
- **Advanced Decoding** — 支持 beam search、n-best rescoring、character-level LM 集成。
- **Distillation** — 支持 teacher model（如 SenseVoice、FunASR）蒸馏训练。
- **Domain Adaptation** — 支持 domain-specific encoder 和定向训练。

---

## 环境依赖

### 基础

```bash
pip install -r requirements.txt
```

### 服务器 / 训练

```bash
pip install -r requirements-server-torch.txt
```

### 提交打包

```bash
pip install -r requirements-submission.txt
```

Core dependencies: `torch`, `transformers`, `datasets`, `librosa`, `jiwer`, `editdistance`.

---

## 快速开始

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

## 数据格式

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

## 训练

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

### 带护栏的推理

`predict_guarded.py` 提供带安全检查的推理，适用于生产环境：

```bash
python predict_guarded.py \
  --checkpoint outputs/run_01/checkpoint-5000 \
  --manifest artifacts/manifests/test.jsonl \
  --output predictions.jsonl
```

---

## 评测与报告

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

## 离线提交

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

## 配置系统

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

## 脚本索引

### 数据准备

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

### 训练

| Script | Purpose |
|---|---|
| `train.py` | 主训练入口 (Full SFT) |
| `scripts/train_directional.py` | 定向训练（领域适配） |
| `scripts/train_w500_extension.py` | W500 扩展训练 |
| `scripts/train_clean_char_lm.py` | 字符级 LM 训练 |
| `scripts/interpolate_checkpoints.py` | Checkpoint 插值合并 |
| `scripts/merge_checkpoint_regions.py` | 区域级 checkpoint 合并 |
| `scripts/compose_decoder_layers.py` | Decoder 层组合 |

### 预测与推理

| Script | Purpose |
|---|---|
| `predict.py` | 标准预测 |
| `predict_guarded.py` | 带安全检查的预测 |
| `scripts/predict_local_guarded.py` | 本地 guarded 预测 |
| `scripts/predict_beam2_norepeat4.py` | Beam search (beam=2, no repeat trigrams) |
| `scripts/predict_true_nbest_clean_lm.py` | N-best + clean LM rescoring |
| `scripts/predict_sensevoice_manifest.py` | SenseVoice teacher 预测 |
| `scripts/predict_funasr_nano_manifest.py` | FunASR nano teacher 预测 |

### 评测与分析

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

### 审计与质量

| Script | Purpose |
|---|---|
| `scripts/audit_lm_validation_overlap.py` | 检查 LM 训练/验证重叠 |
| `scripts/audit_closed_set_candidates.py` | 闭集候选质量审计 |
| `scripts/audit_official_audio_duplicates.py` | 官方音频重复检测 |
| `scripts/verify_cv_audio_origin.py` | CommonVoice 音频来源验证 |
| `scripts/score_cv_origin_labels.py` | CV 来源标签评分 |
| `scripts/check_metric_parity.py` | 指标一致性检查 |
| `scripts/validate_prediction_order.py` | 预测顺序校验 |

### 提交与打包

| Script | Purpose |
|---|---|
| `scripts/package_submission.py` | 创建离线提交包 |
| `scripts/verify_submission.py` | 离线提交包验证 |
| `scripts/check_submission.py` | 提交前检查 |

### 其他构建脚本

| Script | Purpose |
|---|---|
| `scripts/build_loss_filtered_manifests.py` | 基于 loss 的样本过滤 |
| `scripts/build_distillation_manifest.py` | 蒸馏训练 manifest 构建 |
| `scripts/build_wenet_teacher_challenge.py` | Wenet teacher 挑战集构建 |
| `scripts/build_short_domain_encoder_stream.py` | 短领域编码器数据流 |
| `scripts/retrieve_closed_set_formal.py` | 闭集正式文本检索 |
| `scripts/attribute_formal_text_domain.py` | 正式文本领域归因 |

---

## Slurm 作业

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

## 许可

MIT，见 [`LICENSE`](LICENSE)。

~~`scripts/package_submission.py` 会创建平铺 ZIP，并校验其中恰好包含一个 `model.safetensors`。`scripts/verify_submission.py` 可在断网模式下检查模型加载、预测输出数量、顺序和 `audio_path` 一致性。~~
<sub>划掉：这段与[离线提交](#离线提交)一字不差重复，本来就不是许可证内容。</sub>

## 当前公开最佳方法

当前公开的最佳方法是 W500 adaptive curriculum：WenetSpeech-Yue batch
只更新 Whisper Encoder，Official batch 则解除冻结并更新整个模型。两类
optimizer step 交错进行，因此外部数据主要扩展粤语声学覆盖，而 Official
短句持续校正粤语用字、插入、重复和 EOS 行为。

平台得分 69.49 的精确模型发布在
[Hugging Face](https://huggingface.co/Vanxun-Hank/whisper-small-cantonese-w500-adaptive)。
训练逻辑、选择护栏、推理配置和复现范围见
[`docs/W500_ADAPTIVE_METHOD.md`](docs/W500_ADAPTIVE_METHOD.md)。

## 更新 2026-09-08：P2 容量矩阵已收敛

下文若干划掉的说法已被
[`reports/raw_winner_p2_full_converged.md`](reports/raw_winner_p2_full_converged.md) 解决。
12 个基线 arm 全部跑满 3 epoch、两个 seed，前四个 arm 另有 Public 与 OOD。

- **容量已经量出来了，不再是假设。** Large-v2 Full 的 validation tol2 `0.8946` / CER `0.0685`，
  对比 RAW_WINNER 的 `0.8547` / `0.0859`。但 Small→Medium 是 +0.063 tol2，
  Medium→Large-v2 参数量翻倍只换来 **+0.005**。
- **LoRA 不是 validation 上看起来的那个弱选项。** 它在 validation 落后 Full 0.020 tol2，
  但在 Public **反超**（`0.9516` vs `0.9468`，CER 低 11%，两个 seed 一致），
  而只训练 0.254% 的参数。
- **OOD 面从来就没在测识别。** 它的参考是繁体，而打分器强制把预测转简，
  所以记录在案的 OOD 错误有 73~81% 是字形转换。归一化后 Large-v2 Full 的
  OOD CER 是 `0.063`，和它的 Public CER 只差 0.006。
- **剩下的错误不在容量上。** Large-v2 Full 残余编辑量的 89.7% 在 Small Full 上同样错，
  70.8% 是替换，只有 3.1% 像是数据错配。

以上都没有改变已发布的 69.49 模型，它仍然是 W500 adaptive 那个 checkpoint。

## 当前分数与榜单竞争力

本仓库公开的 W500_Adaptive_RAW_WINNER 在平台隐藏测试集上的结果为：

| 指标 | 结果 |
| --- | ---: |
| 平台分数 | 69.49 |
| CER | 0.2473053892215569 |
| sentence accuracy (edit distance <= 2) | 0.3400 |
| 隐藏评测样本数 | 200 |

仓库没有保存完整的榜单快照或其他参赛方案分数，因此不能据此诚实地给出当前名次。69.49 也不应被解读为通用粤语 ASR 基准；它是一次竞赛特定隐藏测试的结果。下面的判断解释为什么这个分数还不足以称为稳定的高榜单成绩。

### 主要原因

1. **绝对错误率仍然偏高。** CER 为 24.73%，而容错 2 字的整句准确率只有 34%。换句话说，隐藏集约 66% 的句子没有达到“最多错 2 个字符”的标准。这个误差水平本身就会限制榜单位置，即使训练流程很严谨。

2. **模型容量和 tokenizer 没有升级。** 发布模型仍是约 241.7M 参数的 Whisper-small，架构和 tokenizer 保持不变。~~训练重点放在数据和优化策略，而不是更大模型、粤语专用 tokenizer 或更强的语言建模能力；这会形成可见的上限。~~ 容量此后已经测过：Medium 和 Large-v2 都越过了 RAW_WINNER，但 Medium→Large-v2 只有 +0.005 tol2，所以这个上限主要不是容量上限。tokenizer 与语言建模仍未验证。

3. **外部数据与竞赛语料存在分布差异。** WenetSpeech-Yue 音频通常更长，且可能包含伪标签或不同的转写习惯；竞赛语音则更接近短句和特定标注风格。W500 curriculum 已经在缓解这个问题，但它不能消除说话人、噪声、时长、词汇和用字分布的差异。

4. **Wenet 数据只更新 Encoder。** 这是一个有意识的稳定性选择：Wenet batch 主要扩展声学覆盖，Decoder 和 proj_out 保持冻结；只有 Official batch 才做 Full SFT。这样可以减少粤语用字、插入、重复和 EOS 行为被外部数据带偏，但也意味着大量外部语音没有直接训练 Decoder 的词汇和语言模型能力，榜单所需的文本侧收益受到限制。

5. **推理仍是保守的单束解码。** 官方复现实验固定使用 num_beams=1、max_length=225，并依赖固定的 repetition penalty 与 suppression 配置。没有加入 beam-search 对比、语言模型重排序、候选融合、上下文 bias 或 test-time augmentation；这些可能带来榜单收益，但尚未被验证，不能混入当前 69.49 的复现结果。

6. **选择目标与隐藏榜单并不完全相同。** checkpoint 只按固定 validation 排名，Public/OOD 只作为稳定性 veto，不参与排序。这是更可靠、避免过拟合榜单的实验设计，但也意味着固定验证集的最优点不一定是隐藏榜单的最优点。

7. **增强策略仍较克制。** 当前配置显式关闭 SpecAugment。对于真实粤语录音中的噪声、通道变化、语速变化和截断问题，模型可能仍有泛化空间；这需要通过独立实验确认，而不是从当前结果直接推断。

### 如何读这个结果

当前工作更像是一个**可审计、可复现的竞赛提交基线**，而不是已经完成榜单优化的最终系统。要提高榜单竞争力，下一轮实验应分别记录：

- ~~逐句错误类型：插入、重复、替换、截断和 EOS 失败；~~ 已做 —— 替换占 70.8%，且误差是平坦长尾；
- 不同解码配置的增益，且与当前 num_beams=1 基线分开报告；
- Decoder-focused fine-tuning、~~模型容量和~~粤语 tokenizer 的独立影响；
- 竞赛语料与 Wenet/Official 数据在时长、说话人、场景和用字上的分布差异；
- ~~validation、public、OOD 与最终隐藏榜单之间的相关性。~~ 部分有答案 —— validation 与 Public 在 Full/LoRA 上给出相反排序；OOD 要先重建才谈得上相关性。

任何新方案都应继续通过固定验证排序、Public/OOD guardrails、离线推理 parity 和 SHA-256 产物校验；否则更高的单次分数无法证明它是更好的模型。


## 改进路线图：从 69.49 到更强榜单表现

~~下一轮不应把“训练更久”当作默认答案，而应把主要变量拆开，按成本和风险逐级验证。下面是一条可复现、可回退的路线；其中~~“通过”表示在固定验证集上有收益，同时不能触发 Public/OOD 或离线推理一致性护栏。

RAW_WINNER 69.49 获胜训练链，以及 P0–P2 已执行实验的完整技术配置、数据与权重 provenance、八卡拓扑、指标语义、解码矩阵、Official-only 回正、单轴增强、Small/Medium/Large-v2 Full SFT/LoRA、Tokenizer/Unicode、字符 LM、融合、资源成本、失败恢复和 SHA-256，统一见：

**[`docs/RAW_WINNER_P0_P1_P2_TECHNICAL_REPORT.md`](docs/RAW_WINNER_P0_P1_P2_TECHNICAL_REPORT.md)**

这是推荐的唯一详细入口。P2 的自动生成简版见
[`reports/raw_winner_p2_structural_probe.md`](reports/raw_winner_p2_structural_probe.md)，论文式实验与限制说明见
[`paper/manuscript.md`](paper/manuscript.md)。~~P2 结果只支持 2,400 样本、150-step matched-budget 适配效率结论，不代表充分收敛后的架构排名~~ —— 2026-09-08 已被
[`reports/raw_winner_p2_full_converged.md`](reports/raw_winner_p2_full_converged.md)
的收敛矩阵取代，该报告同时推翻了探针的排序。这些都没有混入 69.49 平台提交。

### P0：先建立可比较的误差基线

1. **固定评测协议。** 锁定当前 checkpoint、manifest、文本规范化、音频顺序和提交校验，保存一份不可变的 69.49 基线。
2. **做逐句错误分桶。** 统计插入、重复、替换、截断、EOS 失败，并按时长、说话人、噪声/通道、数据来源和词汇类型切片。目标是先回答“分数低主要低在哪里”，而不是只看一个总 CER。
3. **检查指标相关性。** 同时记录 validation、Public、OOD 和最终隐藏集结果；如果验证集最优点与隐藏集不一致，后续选模必须把这种不一致作为风险，而不是事后解释。

### P1：低风险、快速验证的改动

1. **解码消融。** 对同一个原始模型只改变 num_beams、max_length、repetition penalty 和候选重排序，先做 beam 1/2/4/5 的小矩阵。每个配置单独报告 CER、tol2、截断率和推理成本，不能把解码收益误写成训练收益。
2. **数据/域对齐。** 比较竞赛音频与 Wenet/Official 的时长、说话人、噪声、采样率和用字分布；优先构造不泄漏测试集的短句、说话人隔离验证集和目标域采样策略。
3. **Decoder 文本侧短实验。** 在保持 Encoder 方案不变的前提下，先做 Official-only 的 Decoder-focused fine-tuning，观察用字、插入、重复和 EOS 是否改善；不要第一轮就同时改变 Wenet 更新规则和 Decoder 学习率。
4. **单轴增强。** 逐一测试 SpecAugment、速度扰动、噪声/混响和通道扰动，每次只开启一个主要变量，避免无法归因。

### P2：中高成本的结构性改动

1. ~~**模型容量。** 在基线和数据对齐结论稳定后，再比较 Whisper-medium/large、参数高效微调与 Full SFT 的收益、显存、吞吐和延迟。~~ **已完成（2026-09-08）。** Small/Medium/Large-v2 × Full-SFT/LoRA，两个 seed，3 epoch。Medium 以上收益急剧递减；LoRA 峰值显存 6.58 GB，但**并不更快**（Large-v2 上 4:07 vs Full 的 3:53）。
2. **粤语文本建模。** 评估粤语专用 tokenizer、词表扩展或外部语言模型重排序；必须同时检查 Unicode、繁简/异体字规范化和离线提交兼容性。
3. **多模型/多候选融合。** 只有在单模型解码和数据策略的收益被确认后，才评估 checkpoint ensemble、TTA 或候选融合，避免用复杂度掩盖单模型问题。

### 最小实验矩阵

| ID | 只改变的变量 | 固定内容 | 主要观察指标 | 通过条件 |
| --- | --- | --- | --- | --- |
| A | 解码配置 | 原始 RAW_WINNER、同一音频顺序 | CER、tol2、截断率、时延 | 验证集改善且 Public/OOD 不恶化 |
| B | 目标域采样/短句策略 | 模型、解码和总更新预算 | 各时长/来源切片、整体 CER | 目标域切片与整体均有稳定收益 |
| C | Decoder-focused SFT | Encoder 更新规则和数据版本 | 用字、插入、重复、EOS | 文本侧错误下降且无明显声学回退 |
| D | 单一增强轴 | 数据采样、模型和解码 | 噪声/通道/时速切片 | OOD 改善且验证集不掉点 |
| E | 更大模型或 tokenizer | 数据与评测协议 | 分数、参数量、显存、时延 | ~~增益足以覆盖新增成本~~ **模型部分 2026-09-08 已做：Medium 以上不划算。tokenizer 未测。** |

### 选择与发布规则

- 每个实验只改变一个主要变量，配置、manifest、checkpoint 和输出包都记录 SHA-256。
- 先用固定 validation 排序，再用 Public/OOD 做 veto；隐藏榜单只作为最终外部验证，不反向调参。
- 任何新结果都与 69.49 的 RAW_WINNER 分开命名、分开报告；不覆盖当前基线，也不把未经验证的假设写成结论。
- 先完成 A→B→C→D，再决定是否投入 E；如果 P0 的错误分桶显示主要是标注规范或解码问题，就暂停扩大模型规模。
