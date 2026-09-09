<div align="center">

# 基于 Whisper 的粤语语音识别

**Source-alternating Whisper-small 训练、收敛后的容量/PEFT 实验与可审计评测。**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Repository License](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)

[English](README.md) · **简体中文**

</div>

本仓库提供粤语 ASR 的训练、评测、打包和验证代码。论文方法系统是一个
source-alternating Whisper-small checkpoint：WenetSpeech-Yue step 只更新 encoder，
与之交替的任务数据 step 更新完整模型。独立的 P2 follow-up 比较 Small、Medium、
Large-v2 的 Full SFT 与 LoRA，并公开 Large-v2 LoRA adapter。该 adapter 独立训练，
不是 source-alternating curriculum 的输出。

## 公开模型与报告

- [Source-alternating Whisper-small](https://huggingface.co/cantonese-asr-lab/whisper-small-cantonese-w500-adaptive)
- [独立 Large-v2 LoRA adapters](https://huggingface.co/cantonese-asr-lab/whisper-large-v2-cantonese-p2-lora)
- [收敛 P2 报告](reports/raw_winner_p2_full_converged.md)

下表 accuracy 与 CER 均为百分数。

### Source-alternating Whisper-small

| 评测面 | tol2 accuracy (%) | CER (%) | 选择角色 |
|---|---:|---:|---|
| 固定 Validation | 85.47 | 8.59 | 选择方法 checkpoint |
| 任务提供的 local evaluation | 89.42 | 8.13 | 补充评测 |
| OOD，冻结的不对称 scorer | 33.15 | 34.04 | 历史任务协议结果 |
| OOD，对称字形审计 | 87.45 | 8.84 | 控制字形差异后的诊断 |

### Large-v2 LoRA follow-up

| Adapter | Local-eval tol2 (%) | Local-eval CER (%) | 对称 OOD tol2 (%) | 对称 OOD CER (%) |
|---|---:|---:|---:|---:|
| Seed 42 | 95.16 | 5.07 | 91.40 | 6.38 |
| Seed 43（仓库根目录） | 95.53 | 5.10 | 90.80 | 6.66 |

两个 seed 均已公开。Seed 43 因 local-evaluation tol2 更高而放在模型仓库根目录；
seed 42 的 CER 略低。因此 local evaluation 是默认 adapter 的已披露选择面，不能再称为
完全未参与选择的 test set。OOD 不参与模型、语言模型或融合方法选择。

Large-v2 发布物是约 15.8 MB 的 LoRA adapter，必须与
`openai/whisper-large-v2` 基座组合使用；它不是独立的 15.5 亿参数全量模型文件。
本实验中它更新组合参数的 0.254%，峰值 allocated memory 为 6.58 GiB。

## OOD 评分说明

冻结 scorer 会把 prediction 从繁体转为简体，却不转换 reference。这个差异对以简体为主的
Validation 和 local evaluation 影响很小，但会显著影响以繁体为主的 OOD reference。
对两侧执行相同转换后，八个可逐句审计的收敛 endpoint 有 72.6%–80.6% 的 raw OOD
编辑量被消除。Raw 结果用于复现历史协议；讨论控制字形差异后的识别表现时，应同时查看
对称结果。

## 数据来源

任务提供的粤语材料来自 [AI Dimsum Cup 初赛 ASR
任务](https://www.aicompetition-pz.com/topic_detail/19)。任务页指向 [Cantonese Life
Scenarios Corpus](https://huggingface.co/datasets/leeduckgo/cantonese-life-scenarios-corpus)。
当前数据页面只用于 provenance，不冒充训练时的精确 snapshot。WenetSpeech-Yue、
Common Voice zh-HK 和 MDCC 只用于明确标注这些来源的实验。本仓库不重新分发训练数据。

## 仓库结构

```text
cantonese-asr/
├── cantonese_asr/          # Manifest I/O 与字符级指标
├── configs/rounds/         # 版本化实验配置
├── scripts/                # 数据、训练、评测与审计脚本
├── slurm/                  # 集群作业定义
├── tests/                  # 单元与集成测试
├── reports/                # 紧凑实验报告与 receipts
├── train.py                # 主训练入口
├── predict.py              # 标准推理
└── predict_guarded.py      # 带生成健康检查的推理
```

## 安装

```bash
pip install -r requirements.txt
```

服务器训练或离线打包可安装对应的固定依赖：

```bash
pip install -r requirements-server-torch.txt
pip install -r requirements-submission.txt
```

## 数据格式

训练 manifest 使用 JSONL，每行一条语音：

```json
{
  "id": "00001",
  "audio_path": "artifacts/data/train_raw/00001.wav",
  "text": "你好！",
  "duration_s": 1.24,
  "source": "task_provided",
  "split": "train"
}
```

`text` 是监督文本。翻译、粤拼和场景标签可以作为 metadata 保留，但不能替代 label。
除非实验配置另有声明，音频应为 16 kHz mono WAV。

## 训练与评测

```bash
python train.py \
  --manifest artifacts/manifests/train.jsonl \
  --val-manifest artifacts/manifests/val.jsonl \
  --output-dir outputs/run_01 \
  --batch-size 4 \
  --grad-accum 4 \
  --lr 1e-5 \
  --epochs 5

python scripts/evaluate_checkpoints.py --checkpoint-dir outputs/run_01
python scripts/evaluate_predictions.py \
  --predictions predictions.jsonl \
  --reference references.jsonl
```

核心指标是字符错误率 `cer` 和字符编辑距离不超过 2 的句子比例
`sentence_accuracy_tol2`。选择规则、文本规范化、解码参数、样本顺序与模型身份会共同
记录，避免指标脱离具体协议。

## 打包与验证

```bash
python scripts/package_submission.py \
  --checkpoint outputs/best/checkpoint-5000 \
  --output model-package.zip

python scripts/verify_submission.py --submission model-package.zip
```

验证器检查离线加载、预测数量与顺序、音频路径及包结构。PEFT checkpoint 应按 model
card 与基座组合；Hugging Face 上的 adapter 文件不会被 merged weights 覆盖。

## 证据边界

- Source-alternating curriculum 由完整 recipe 支持；update scope、学习率、初始化、
  总 exposure 和交替频率尚未在一个 factorial design 中全部拆开。
- Validation 与 local evaluation 对六个收敛模型 family 的排序不同；两 seed 平均 tol2
  的 Spearman 相关为 0.60。这提示 selection risk，但没有证明任一 split 本身错误。
- Full SFT 与 LoRA 的相对结果取决于评测面；本仓库不声称其中一种普遍更好。
- Tokenizer、字符 LM 和融合结果属于特定 recipe 的诊断，不能外推成整个方法类别无效。
- 下载的证据归档有 134 份可逐句复算 receipt；另有 28 份后期 aggregate receipt 在该
  归档中没有逐句对应文件。

## 许可与数据访问

仓库代码采用 MIT License，见 [`LICENSE`](LICENSE)。Whisper-small 权重及 Whisper
Large-v2 基座使用 Apache License 2.0；公开 Large-v2 LoRA adapter 使用 MIT License。
上游数据仍受各自条款约束。训练时任务数据精确 snapshot 的识别信息可通过 email 向
corresponding author 询问；地址会在最终作者信息确认后补充。
