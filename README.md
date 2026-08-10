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

## 当前公开最佳方法

当前公开的最佳方法是 W500 adaptive curriculum：WenetSpeech-Yue batch
只更新 Whisper Encoder，Official batch 则解除冻结并更新整个模型。两类
optimizer step 交错进行，因此外部数据主要扩展粤语声学覆盖，而 Official
短句持续校正粤语用字、插入、重复和 EOS 行为。

平台得分 69.49 的精确模型发布在
[Hugging Face](https://huggingface.co/Vanxun-Hank/whisper-small-cantonese-w500-adaptive)。
训练逻辑、选择护栏、推理配置和复现范围见
[`docs/W500_ADAPTIVE_METHOD.md`](docs/W500_ADAPTIVE_METHOD.md)。

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

2. **模型容量和 tokenizer 没有升级。** 发布模型仍是约 241.7M 参数的 Whisper-small，架构和 tokenizer 保持不变。训练重点放在数据和优化策略，而不是更大模型、粤语专用 tokenizer 或更强的语言建模能力；这会形成可见的上限。

3. **外部数据与竞赛语料存在分布差异。** WenetSpeech-Yue 音频通常更长，且可能包含伪标签或不同的转写习惯；竞赛语音则更接近短句和特定标注风格。W500 curriculum 已经在缓解这个问题，但它不能消除说话人、噪声、时长、词汇和用字分布的差异。

4. **Wenet 数据只更新 Encoder。** 这是一个有意识的稳定性选择：Wenet batch 主要扩展声学覆盖，Decoder 和 proj_out 保持冻结；只有 Official batch 才做 Full SFT。这样可以减少粤语用字、插入、重复和 EOS 行为被外部数据带偏，但也意味着大量外部语音没有直接训练 Decoder 的词汇和语言模型能力，榜单所需的文本侧收益受到限制。

5. **推理仍是保守的单束解码。** 官方复现实验固定使用 num_beams=1、max_length=225，并依赖固定的 repetition penalty 与 suppression 配置。没有加入 beam-search 对比、语言模型重排序、候选融合、上下文 bias 或 test-time augmentation；这些可能带来榜单收益，但尚未被验证，不能混入当前 69.49 的复现结果。

6. **选择目标与隐藏榜单并不完全相同。** checkpoint 只按固定 validation 排名，Public/OOD 只作为稳定性 veto，不参与排序。这是更可靠、避免过拟合榜单的实验设计，但也意味着固定验证集的最优点不一定是隐藏榜单的最优点。

7. **增强策略仍较克制。** 当前配置显式关闭 SpecAugment。对于真实粤语录音中的噪声、通道变化、语速变化和截断问题，模型可能仍有泛化空间；这需要通过独立实验确认，而不是从当前结果直接推断。

### 如何读这个结果

当前工作更像是一个**可审计、可复现的竞赛提交基线**，而不是已经完成榜单优化的最终系统。要提高榜单竞争力，下一轮实验应分别记录：

- 逐句错误类型：插入、重复、替换、截断和 EOS 失败；
- 不同解码配置的增益，且与当前 num_beams=1 基线分开报告；
- Decoder-focused fine-tuning、模型容量和粤语 tokenizer 的独立影响；
- 竞赛语料与 Wenet/Official 数据在时长、说话人、场景和用字上的分布差异；
- validation、public、OOD 与最终隐藏榜单之间的相关性。

任何新方案都应继续通过固定验证排序、Public/OOD guardrails、离线推理 parity 和 SHA-256 产物校验；否则更高的单次分数无法证明它是更好的模型。


## 改进路线图：从 69.49 到更强榜单表现

下一轮不应把“训练更久”当作默认答案，而应把主要变量拆开，按成本和风险逐级验证。下面是一条可复现、可回退的路线；其中“通过”表示在固定验证集上有收益，同时不能触发 Public/OOD 或离线推理一致性护栏。

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

1. **模型容量。** 在基线和数据对齐结论稳定后，再比较 Whisper-medium/large、参数高效微调与 Full SFT 的收益、显存、吞吐和延迟。
2. **粤语文本建模。** 评估粤语专用 tokenizer、词表扩展或外部语言模型重排序；必须同时检查 Unicode、繁简/异体字规范化和离线提交兼容性。
3. **多模型/多候选融合。** 只有在单模型解码和数据策略的收益被确认后，才评估 checkpoint ensemble、TTA 或候选融合，避免用复杂度掩盖单模型问题。

### 最小实验矩阵

| ID | 只改变的变量 | 固定内容 | 主要观察指标 | 通过条件 |
| --- | --- | --- | --- | --- |
| A | 解码配置 | 原始 RAW_WINNER、同一音频顺序 | CER、tol2、截断率、时延 | 验证集改善且 Public/OOD 不恶化 |
| B | 目标域采样/短句策略 | 模型、解码和总更新预算 | 各时长/来源切片、整体 CER | 目标域切片与整体均有稳定收益 |
| C | Decoder-focused SFT | Encoder 更新规则和数据版本 | 用字、插入、重复、EOS | 文本侧错误下降且无明显声学回退 |
| D | 单一增强轴 | 数据采样、模型和解码 | 噪声/通道/时速切片 | OOD 改善且验证集不掉点 |
| E | 更大模型或 tokenizer | 数据与评测协议 | 分数、参数量、显存、时延 | 增益足以覆盖新增成本 |

### 选择与发布规则

- 每个实验只改变一个主要变量，配置、manifest、checkpoint 和输出包都记录 SHA-256。
- 先用固定 validation 排序，再用 Public/OOD 做 veto；隐藏榜单只作为最终外部验证，不反向调参。
- 任何新结果都与 69.49 的 RAW_WINNER 分开命名、分开报告；不覆盖当前基线，也不把未经验证的假设写成结论。
- 先完成 A→B→C→D，再决定是否投入 E；如果 P0 的错误分桶显示主要是标注规范或解码问题，就暂停扩大模型规模。
