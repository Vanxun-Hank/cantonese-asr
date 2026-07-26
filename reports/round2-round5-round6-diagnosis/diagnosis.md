# Round 2 / Round 5 / Round 6 粤语 ASR 退步诊断

生成日期：2026-07-25

## 结论先行

服务器证据补齐后，不能再把平台下降直接解释为“Round 6 模型整体退化”。
Round 6 在固定官方 validation、公开测试，以及固定 2,000 条 CV/MDCC OOD
面板上都不差于 Round 2。当前最可能的是：

1. **平台由权重 hash 决定抽样 seed、每次只评 200 条，轮次间不是固定配对 A/B**；
2. 平台隐藏数据包含现有 validation/public/CV/MDCC 面板没有覆盖的域，而
   Round 6 的来源权重变化和长训练可能只在这个未覆盖域上退化；
3. SpecAugment、fresh/continue 和 `predict.py` 都不是现有证据支持的首要原因。

Round 6 没有新增新的外部样本。它完整复用了 Round 2 的 17,012 条外部样本，
但把 6,292 条官方原始音频替换为 18,876 个固定速度视图。因此：

- Round 2：6,292 官方 + 17,012 外部，外部占 73.0%；
- Round 6：18,876 官方速度视图 + 17,012 外部，外部占 47.4%。

换言之，Round 6 的主要变化是把每条官方录音重复成三个高度相关的声学视图，
并将官方域的训练权重提高三倍。这个改变仍然是隐藏域退化的主要可疑机制，
但固定 OOD 结果说明它没有造成广义的粤语识别退化。

平台本身还有一个重要混杂因素：日志表明平台只抽 200 条，并由提交权重的 hash
派生抽样 seed。不同权重会面对不同的 200 条隐藏样本，所以 62.69 与约 56
不是严格的配对 A/B。若把句准确率近似看作二项比例，200 条在 31% 附近的
标准误约为 3.3 个百分点，95% 波动约为 ±6.4 个百分点。平台分数应作为最终
确认，不能单独承担因果归因。

## 对比表

| 轮次 | 数据 | 外部占比 | 初始化 | 训练设置 | SpecAug | 本地 validation | 公开测试 | 平台 |
|---|---|---:|---|---|---|---|---|---|
| Round 2 | 官方 6,292 + CV 8,451 + MDCC 8,561，共 23,304 | 73.0% | Whisper-small | LR 2e-5，cosine，3 epochs，AdamW，effective batch 16 | 否 | Acc 84.19%，CER 9.21%，loss 0.2486，epoch 2 | Acc 87.58%，CER 9.12% | Acc 86.75%，CER 10.49% | 62.69 |
| Round 5 | 官方 0.9/1.0/1.1 各 6,292，共 18,876 | 0% | Whisper-small | LR 2e-5，linear，3 epochs，AdamW，effective batch 16 | 否 | Acc 84.33%，CER 8.66%，loss 0.3016，epoch 3 | Acc 87.42%，CER 8.60% | Acc 52.20%，CER 23.53% | 精确值未保存 |
| Round 6 fresh | 官方速度 18,876 + 同一批 CV/MDCC 17,012，共 35,888 | 47.4% | Whisper-small | LR 2e-5，cosine，最多 8 epochs，patience 2 | 是 | Acc 86.61%，CER 8.27%，loss 0.3189，epoch 7 | Acc 89.84%，CER 7.98% | Acc 87.85%，CER 8.93% | 用户报告约 56 |
| Round 6 continue | 同上 | 47.4% | Round 2 checkpoint-2914 | LR 5e-6，cosine，最多 8 epochs，patience 2 | 是 | Acc 85.47%，CER 8.39%，loss 0.3015，epoch 6 | Acc 90.26%，CER 7.76% | Acc 87.70%，CER 9.79% | 用户报告约 56 |

OOD 数字使用 `metrics_symmetric_t2s.json`，即参考和预测都转简体后评估。官方
不对称规则只转 prediction，会把 CV/MDCC 的繁体参考系统性计成替换错误，不适合
判断跨语料泛化。

下载到 Mac 的 continued submission 文件名写的是 epoch 4，但模型
SHA-256 `281f25c...` 已精确匹配服务器 `checkpoint-13458`，即 epoch 6。

## 因果归因

### 1. 平台 200 条 hash 抽样与隐藏域覆盖缺口：最高概率

平台日志明确写明抽样 seed 由提交权重 hash 派生，且只评 200 条。不同权重
对应不同测试子样本，因此平台轮次不是同一批数据上的配对比较。

更关键的是，Round 6 fresh + Spec 在同一个固定 OOD 面板上比 Round 2 更好：

- Round 2：Acc 86.75%，CER 10.49%；
- Round 6 fresh + Spec：Acc 87.85%，CER 8.93%；
- Round 6 continue + Spec：Acc 87.70%，CER 9.79%。

因此平台约 56 不能证明 Round 6 在一般粤语上退化。它可能是 200 条抽样波动，
也可能是平台隐藏样本含有 CV/MDCC 与官方测试都没覆盖的方言、录音或内容域。

把固定 OOD 按来源拆开后，MDCC 明显比 Common Voice 稳定：

| 版本 | CV Acc / CER | MDCC Acc / CER |
|---|---:|---:|
| Round 2 | 83.90% / 14.64% | 89.60% / 7.05% |
| Round 5 | 44.70% / 29.01% | 59.70% / 18.98% |
| Round 6 fresh + Spec | 85.80% / 11.45% | 89.90% / 6.85% |
| Round 6 continue + Spec | 86.70% / 13.18% | 88.70% / 6.98% |

CV 的 CER 在每个相关版本中都显著高于 MDCC。这不能单独证明 CV 数据有错，
但与“标注口径、口音和文本规范更不一致”的假设一致，足以支持下一轮把两个
来源拆开，不再统称为一个 external pool。

### 2. 数据来源权重 / 域不匹配：中高概率的隐藏域机制

证据链：

1. 隐藏成绩最好的 Round 2 外部占比最高，为 73%。
2. Round 5 去掉全部外部数据后，本地公开 CER 并没有恶化，但平台反而低于
   Round 2，说明公开测试不足以代表隐藏域。
3. Round 5 的固定 OOD Acc 从 Round 2 的 86.75% 降至 52.20%，CER 从
   10.49% 升至 23.53%，直接证明去掉外部数据会严重破坏外部域能力。
4. Round 6 使用与 Round 2 完全相同的 17,012 条外部样本，但把官方样本权重
   提高三倍，使外部占比降至 47.4%。
5. Round 6 加回外部数据后固定 OOD 已恢复并超过 Round 2，所以问题若存在，
   更可能是平台独有域，而非 CV/MDCC 域。

因此，“加数据反而下降”不是新增外部信息量造成的；它可能是固定三视图改变
采样分布和每个 epoch 优化权重后，只对平台独有域产生的负迁移。

### 3. 训练过久造成置信度过拟合：中概率、Round 6 的放大因素

Round 2 的 validation loss 在 epoch 2 达到 0.2486，epoch 3 为 0.2501，
train-probe loss 从 0.199 降至 0.049，泛化差距仍较可控。

Round 5 的 validation loss 从 epoch 1 的 0.2775 持续升至 epoch 3 的 0.3016，
train-probe loss却从 0.0485 降到 0.00093。Round 6 fresh + Spec 到 epoch 7
时 train-probe loss 为 0.00045，而 validation loss 为 0.3189。这是明显的
置信度过拟合，但 Acc/CER 仍改善，且固定 OOD 没有退化。

`sentence_accuracy_tol2` 允许每句最多错 2 字。继续训练可能让更多同域句子从
3 个错误进入 2 个错误，从而提高句准确率，却同时使模型分布更尖锐、对 OOD
输入更脆弱。仅以该指标 early stopping，patience 2 仍可能选到较晚 checkpoint。

### 4. SpecAugment：现有证据不支持它是主因

Round 6 内部形成了同数据、同初始化、同超参数的成对对照：

- Fresh：Spec 将公开 Acc 从 86.05% 提到 89.84%，CER 从 9.80% 降到 7.98%；
- Continue：Spec 将公开 Acc 从 89.32% 提到 90.26%，CER 从 8.45% 降到 7.76%。

现有强度只有 `mask_time_prob=0.05`、`mask_time_length=10`、不做频率遮挡，
并不激进。两个平台提交恰好都是 Spec 版本，因此平台没有固定样本的
no-Spec 对照；但从本地成对结果看，Spec 更像有益或中性因素。

### 5. 从头训练还是继续训练：低概率成为主因

Fresh + Spec 与 Round2-continued + Spec 的公开 Acc 只差 0.42 个百分点，
validation 只差 1.14 个百分点，且用户报告两者平台都约 56。两种初始化得到
相似退步，说明共同的数据混合和训练时长比初始化更可疑。

### 6. 速度增强本身：仍未被单独识别

Round 5 同时改变了三个因素：删除外部数据、加入三倍速度视图、cosine 改为
linear。Round 6 又同时改变了外部比例、SpecAugment、epoch 上限和初始化。
因此现在不能断言 0.9/1.1 本身有害。

Round 5 的公开 Acc 只比 Round 2 低 0.16 个百分点，CER 还改善 0.51 个百分点，
说明速度音频没有表现出明显的同域声学损坏。更可疑的是“每条固定复制三份”
造成的权重变化，而不是 0.9/1.1 这个幅度本身。

### 7. 超参数 / scheduler：可能有次要影响，但不是首要解释

Round 5 使用 linear，Round 2/6 使用 cosine；Round 6 fresh 使用 2e-5，
continue 使用 5e-6。两种 Round 6 学习率和初始化都出现相似平台退步，
所以学习率不是共同主因。Round 6 的 8-epoch 总步数与 Round 2 的 3-epoch
调度完全不同，长训练和后期过拟合值得优先修正。

## 错误类型

三轮公开测试错误都以替换为主，不是大规模漏字、插字或生成崩溃：

| 版本 | 公开 Acc | CER | top-100 替换计数 | 删除 | 插入 | 最大保留错误距离 |
|---|---:|---:|---:|---:|---:|---:|
| Round 2 | 87.58% | 9.12% | 776 | 87 | 102 | 6 |
| Round 5 | 87.42% | 8.60% | 769 | 101 | 46 | 6 |
| Fresh no-Spec | 86.05% | 9.80% | 818 | 119 | 58 | 7 |
| Fresh Spec | 89.84% | 7.98% | 754 | 77 | 39 | 7 |
| Continue no-Spec | 89.32% | 8.45% | 740 | 78 | 68 | 6 |
| Continue Spec | 90.26% | 7.76% | 694 | 77 | 55 | 8 |

`top_confusions.csv` 每类只保留最多 100 个混淆项，所以这些是可比的截断计数，
不是总编辑操作数。

主要模式：

- 粤语助词/正字变体：`啦↔喇`、`㗎` 删除、`咁→噉`、`冇→无`；
- 同音或近音替换：`再→在`、`约→药/又`、`琴→寻`；
- 口语同义写法：`穿→着`、`选→拣`，在声学上可能正确但按字符指标计错；
- 少量漏助词：`唔`、`㗎`、`我`、`呢`；
- 少量插入助词：`啊` 最突出，continued no-Spec 中为 31 次；
- 英文夹杂只在截断混淆中出现少量单字母删除/插入，没有形成主要错误源；
- 没有数字相关的 top confusion；
- Round 2、Round 5 和四个 Round 6 公共错误集的前 100 条均无超长生成崩溃。

固定 OOD 上还有一个有用差异：

- Round 5 明显更容易漏字，top-100 deletion 计数为 244，Round 2 为 68；
- Round 6 fresh + Spec 的 deletion 为 79、insertion 为 47，恢复良好；
- Round 6 continued + Spec 的 insertion 为 241，常见插入包括
  `么/你/怎/这/早/就/我`，说明继续训练版本比 fresh 更容易插入短语；
- OOD substitution 大量是繁简字（如 `係→系`、`個→个`），应以对称繁简指标
  判断，不应把这些全部解释成声学错误。

这些 `error_examples.json` 只保留 100 条错误，而且公开样本参考长度约 10–17 字，
不能用来判断长音频是否崩溃。长音频、code-switching 和数字必须用固定 OOD
面板单独分桶。

Round 2、Round 5、Round 6 的 selected-checkpoint metrics、confusions、
error examples、曲线和 OOD 报告现已同步到本报告的 `source_evidence/`。

## 推理与评测一致性

强证据表明 Round 5 和两个 Round 6 提交使用了相同推理协议：

- `predict.py` SHA-256 均为
  `6a35676e50515a66af5d7868c95022ab831d5ab6722e230230d07acfd137e8f9`；
- `generation_config.json` SHA-256 均为
  `1d83934369f87a72058bbacaa854d7cbdac06dfaa8eca2aa420c14253ee338e1`；
- `predict.py` 的 `--generation-max-length` 默认是 225，实际传给
  `model.generate(max_length=225)`；
- `language="zh"`、`task="transcribe"`、`num_beams=1`、
  `forced_decoder_ids=None` 一致；
- ZIP 根目录平铺，且都只有一个 `model.safetensors`。

Round 2 训练/评估脚本也显式传入 `--generation-max-length 225`。目前保存的
Round 3 代理提交包使用同一 `predict.py` hash，但精确的 Round 2 平台 ZIP
没有保存在 Mac，所以还差最后一项“Round 2 实际上传包 hash”的闭环证据。

历史提交一致，不代表当前工作区仍可直接提交。当前 `predict.py` SHA-256 已变为
`f547587b...`，并新增：

```python
from cantonese_asr.model_loading import load_whisper_model
```

而 `scripts/package_submission.py` 仍只把单个 `predict.py` 复制进平铺 ZIP，
不包含 `cantonese_asr/`。因此这不会解释 Round 2/5/6 的旧平台分数，但会让
下一轮提交存在 `ModuleNotFoundError` 风险。训练前必须恢复独立的 submission
entrypoint，或把加载逻辑内联到 `predict.py`；离线验证还必须在仓库目录之外
运行，避免本地源码恰好出现在 `PYTHONPATH` 而造成假通过。

本地指标实现与官方公开逻辑一致：只对 prediction 做 OpenCC `t2s`，随后参考
和预测都去除中英文标点及空白，按字符 Levenshtein 距离计算 CER，并以
`distance <= 2` 计句正确。该不对称繁简规则虽然不理想，但必须保持，因为它
就是官方 evaluator 的行为。

## 三个最小下一步实验

三组必须共享同一训练预算和配方，不能再把来源差异与速度、SpecAugment、
epoch 或初始化混在一起：

- 都从原始 Whisper-small 开始；
- LR `2e-5`、cosine、3 epochs、effective batch 16、weight decay `0.01`；
- 不使用速度增强，不使用 SpecAugment；
- 每个 epoch 固定 23,304 次采样：官方 6,292（27%），外部 17,012（73%）；
- 只用 sampler 控制来源，不物理合并成无法追踪的 external pool；
- validation 只用于选模，CV/MDCC OOD 分开诊断。

训练前还有一个不计入三组实验的硬门禁：修复当前非独立 `predict.py`，并在
仓库外的完全离线临时目录验证 flat ZIP 能导入、加载和完成 32 条推理。

### 实验 1（下一轮最值得跑）：Official + MDCC

**假设**：MDCC 是更稳定、更接近目标粤语分布的主要外部来源。

**具体改动**：每 epoch 采官方 6,292、MDCC 17,012。MDCC 只有 8,561 条
唯一样本，因此用固定 seed 的循环/有放回 sampler 达到相同训练预算；每条保留
`source=mdcc`，不复制生成新的假 ID。

**预期信号**：官方 validation 不低于 Round 2 约 84.2%，MDCC OOD
维持约 89.6% / CER 7.05% 或更好；CV OOD 即使略降也应可控。

**失败信号**：validation 明显下降，或 MDCC OOD 比 Round 2 下降超过
0.5 个百分点 / CER 上升超过 0.5 个百分点。此时 MDCC 过采样可能导致过拟合，
应降低外部总占比而不是立刻加回 CV。

### 实验 2：Official + Common Voice

**假设**：Common Voice 的口音覆盖有价值，但标注/文本口径可能引入噪声。

**具体改动**：与实验 1 完全同预算，只把 17,012 次外部采样全部换成 CV；
CV 8,451 条唯一样本，同样使用固定 seed sampler。

**预期信号**：CV OOD 明显优于 MDCC-only，同时官方 validation 和 CER
不明显退化，说明 CV 可作为独立辅助来源。

**失败信号**：CV OOD 没明显提升，或官方 validation、MDCC OOD/平台表现变差；
此时下调 CV 权重，暂不把它与 MDCC 等权混合。

### 实验 3：Official + MDCC 主导 + 少量 CV

**假设**：少量 CV 能增加口音覆盖，但不能大到改变 MDCC 主导的文本与声学分布。

**具体改动**：每 epoch 固定 23,304 次采样：官方 6,292（27%）、
MDCC 14,682（63%）、CV 2,330（10%）。所有来源独立记录，不生成统一 external
标签。

**预期信号**：相对实验 1，CV OOD 改善，同时官方 validation 与 MDCC OOD
退化不超过 0.5 个百分点，CER 增幅不超过 0.5 个百分点。

**失败信号**：只加 10% CV 就让 validation、MDCC OOD 或平台结果下降；
下一轮把 CV 降到 5%，若仍下降则暂停使用。

## 暂时不要跑

- 不要继续把 0.9/1.0/1.1 三份固定样本全部堆进每个 epoch；
- 不要再把 epoch 从 8 延长，更不要从 Round 6 checkpoint 继续训；
- 不要同时改数据比例、SpecAugment、LR、scheduler 和初始化；
- 不要先做蒸馏、LoRA、冻结层或 beam search；这些会扩大混杂；
- 不要把 validation/public test 混入训练；
- 不要只凭一次 200 条 hash 抽样的平台分数判断因果。

三组完成后按 validation 选合格 checkpoint，再把 MDCC/CV OOD 分栏比较。
平台只提交 validation 合格且来源 OOD 没有明显退化的候选，不用单次平台抽样
反向选择数据比例。

## 证据缺口

1. Round 5 精确平台分数；
2. Round 2 实际上传 ZIP 的 SHA-256；
3. 两个 Round 6 平台作品各自的精确分数；
4. 与平台方言/录音条件更接近、且不来自训练源的真正独立 OOD 面板；
5. 按时长、code-switch、数字和方言片区分桶的完整预测结果。
