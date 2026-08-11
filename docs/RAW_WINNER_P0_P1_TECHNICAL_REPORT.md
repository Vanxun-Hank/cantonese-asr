# Cantonese ASR Lab：Whisper-small 粤语语音识别

> 本文档对应本地实验记录提交 `a32cf0d`，集中保存 RAW_WINNER 69.49 基线之后 P0–P1 轮次的完整技术细节；仓库安装、脚本索引和公开方法概览请返回 [`README.md`](../README.md)。

本仓库记录点心杯粤语 ASR 项目的训练、评测与离线提交流程。模型架构固定为
`openai/whisper-small`，不修改 tokenizer 或词表，不使用模型融合、外部语言模型、
动态 fallback 或 LoRA。

当前已验证的平台基线为 `W500_Adaptive_RAW_WINNER`：

| 指标 | 数值 |
|---|---:|
| Platform score | **69.49** |
| Hidden-200 CER | 0.247305 |
| Hidden-200 sentence accuracy（tol2） | 0.340000 |

本文完整记录以 69.49 RAW_WINNER 为起点的 **P0–P1 八卡改进轮次**：冻结误差
基线、静态解码消融、Official-only 短程回正、单轴增强、validation-only 选模和
离线提交验收。本轮没有继续增加 Wenet 训练时长，也没有重跑此前的 W500
Encoder-only 课程训练。

## 1. 实验原则

- 所有训练 arm 都从同一个 RAW_WINNER checkpoint 独立初始化。
- 不从其他 arm 的 checkpoint 继续训练。
- 只加载模型权重，不继承原 optimizer state；每个 arm 重新初始化 AdamW。
- checkpoint 排名只允许使用 fixed official validation。
- Public 1900 与 OOD 只能淘汰不稳定候选，不能改变 validation 排名。
- 多随机种子实验必须 seed 42 和 seed 43 方向一致，才认定配置有效。
- 每个 arm 使用独立输出目录，不覆盖已有产物。
- 不自动上传比赛平台。

## 2. 冻结的 RAW_WINNER 起点

### 2.1 Checkpoint 与哈希

```text
/home/bolin/cantonese-asr/
outputs/w500-adaptive-continuation-v1/stage50/
TOP2_CONSTANT_TO_50H/checkpoint-wenet-50h
```

```text
model.safetensors SHA256:
a0f29a5a011213d5e4de34c40a02d42247255645f06e649e092d2dc495094370

baseline submission ZIP SHA256:
b4fda8ac549d37d8cac9797950636d50ca28f74d8e5e76e2479c15de9b956bc5
```

基线提交包大小为 `895,743,248` bytes。关键推理资产：

| 文件 | SHA256 |
|---|---|
| `config.json` | `53b4eb5c1c63510e9541417174df6ed709cd59e0b052492446ca1287088ee023` |
| `generation_config.json` | `3f2ced827b5a4b0241c4f2f8883cf13da2b00ece23c341d663333c9b07b8de64` |
| `predict.py` | `075d465a775f4ebff1f817d86ab16d3f8da0977257cdcab1870312aaa9c78546` |
| `cantonese_asr/metrics.py` | `b3a3c536c71d6156a0dc4e3ddcd8d04f24453dcca0b8a9bde86d0f33ed4799b2` |

### 2.2 模型结构

```yaml
architecture: WhisperForConditionalGeneration
d_model: 768
encoder_layers: 12
decoder_layers: 12
encoder_attention_heads: 12
decoder_attention_heads: 12
encoder_ffn_dim: 3072
decoder_ffn_dim: 3072
vocab_size: 51865
total_parameters: 241734912
```

本轮没有修改模型结构、tokenizer、词表或权重格式。

### 2.3 不可变基线收据

实验开始前生成只读 `baseline_lock.json`，锁定 checkpoint、模型权重、提交 ZIP、
配置、推理脚本、metric normalizer、三个评测 manifest、音频顺序、基线 predictions、
token sidecar 与文件 SHA256。

Wave 1 的 `D0_CURRENT` 必须在 702/1900/2000 三个表面逐句复现这份基线，
否则整个实验停止。

## 3. 数据与隔离

### 3.1 Official train

```text
artifacts/manifests/train.jsonl
rows: 6292
SHA256: 0d3d1df4b90f89907443ad9706f13d9ff0ae5fe66d6b32c939af99645867cb81
```

Wave 2 和 Wave 3 每个训练 arm 的真实 source exposure：

```text
Official: 2400 samples
Wenet:    0
CV:       0
MDCC:     0
```

本轮测量的是 Official 短程校准与增强效果，不是新的 Wenet 训练收益。

### 3.2 Fixed validation

```text
artifacts/manifests/validation.jsonl
rows: 702
reference field: text
SHA256: 8c2310529ae8d257cb9612dc85797aff5952a5b2b6078cd146db486daf9eee4a
```

Validation 是唯一 checkpoint 排名表面。

### 3.3 Public 1900

```text
artifacts/datasets/official/template_pre.jsonl
rows: 1900
reference field: ref_text
SHA256: 95bb1f05b648c6a2ec2e3b23b6b725a02143da0259804af631f5afa6a4367c50
```

### 3.4 OOD panel

```text
artifacts/manifests/ood/v1/ood_panel.jsonl
rows: 2000
reference field: text
SHA256: 1ab5f1547bc7c93713e99bfb749e1359ed97e98dd8258bf448656da53abb6450
```

Public 与 OOD 均不进入训练和 checkpoint 排名。

### 3.5 音频与标签输入

音频由 `librosa.load` 解码为 16 kHz、单声道 `float32`，随后通过 Whisper feature
extractor 生成 input features 与 attention mask。训练增强完成后执行：

```python
np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
```

Official manifest 的 `text` 字段直接送入 Whisper tokenizer；训练 loader 不执行
粤语到普通话的词汇改写。

## 4. 指标与文本规范化

Reference 不做 OpenCC，只删除预定义中英文标点、空格、换行与可选的
`[RAW]`/`[NORM]` 前缀。Prediction 先执行 OpenCC `t2s`，再执行同样的标点和
空白清理。该非对称行为用于匹配比赛评测逻辑。

```text
CER = 全集字符编辑距离总和 / normalized reference 字符数总和

tol0 = edit distance == 0
tol1 = edit distance <= 1
tol2 = edit distance <= 2
```

字符编辑距离使用 Levenshtein distance，并分别统计 substitutions、deletions 和
insertions。严重错误定义为 `character edit distance > 5`；
`top_20_error_contribution` 是编辑距离最大的20条样本对全集总编辑距离的贡献。

每个评测表面同时记录：

- `repeated_runaway_count`、`token_cycle_count`、`long_text_repeat_count`；
- `no_eos_count`；
- replacement character `�` 数量；
- effective max-length count；
- token length mean/P50/P90/P95/P99/max。

`no_eos_count` 是 token-side 诊断，不等同于真实重复失控。真实 repeated runaway
要求出现 token/text 循环或大量 replacement characters。

Whisper 有4个 decoder prompt tokens，因此有效长度命中定义为：

```text
generated token count + 4 >= max_length
```

## 5. 环境与八卡拓扑

一份正式运行收据记录的主要环境：

```yaml
python: 3.11.15
torch: 2.6.0+cu124
cuda_runtime: 12.4
gpu: NVIDIA GeForce RTX 4090
precision: BF16
tf32: true
gradient_checkpointing: false
```

八张 GPU 来自两台四卡节点：

```text
gpu001: physical GPU 0,1,2,3
gpu002: physical GPU 0,1,2,3
```

八卡运行的是八个独立单卡实验，不是一个八卡 DDP 模型。每个训练 arm 的有效
batch 为：

```text
per-device batch 8 × gradient accumulation 2 × world size 1 = 16
```

八卡只用于并行搜索配置，不改变单个模型的优化轨迹。

## 6. Wave 1：静态解码消融

### 6.1 共同参数

```yaml
language: zh
task: transcribe
return_timestamps: false
do_sample: false
length_penalty: 1.0
max_new_tokens: null
```

长度控制实际使用 `max_length`，不是 `max_new_tokens`。

### 6.2 八个 arm

| Arm | beams | max_length | no-repeat n-gram | repetition penalty | 说明 |
|---|---:|---:|---:|---:|---|
| D0_CURRENT | 2 | 225 | 4 | 1.05 | 基线 |
| D1_BEAM1 | 1 | 225 | 4 | 1.05 | 单 beam 确定性解码 |
| D2_BEAM4 | 4 | 225 | 4 | 1.05 | Beam 4 |
| D3_BEAM5 | 5 | 225 | 4 | 1.05 | Beam 5 |
| D4_MAX128 | 2 | 128 | 4 | 1.05 | 只缩短长度上限 |
| D5_RP100 | 2 | 225 | 4 | 1.00 | 关闭额外重复惩罚 |
| D6_RP110 | 2 | 225 | 4 | 1.10 | 加强重复惩罚 |
| D7_MBR_B5 | 5 | 225 | 4 | 1.05 | 五候选字符级 MBR |

Beam 大于1时传入 `early_stopping=true`。部分 Transformers 版本会提示该 flag
可能被忽略，因此它不被视为独立收益来源；最终包仍执行 raw/ZIP 推理 parity。

### 6.3 D7 MBR

D7 返回5个候选。每个候选的风险是其与其余候选之间的平均 normalized character
edit distance。依次以 MBR risk、model sequence score、原始候选排名决胜。
D7 不使用 reference、外部 LM 或动态 fallback。D7 rank-0 强制与 D3 单输出
逐句一致并经过完整 parity 验证。

### 6.4 护栏

```text
Validation tol2 >= 0.8497008547
repeated_runaway_count == 0
replacement_character_count == 0
effective_max_length_count == 0
normal_long_sentence_truncated_count == 0

Public tol2 >= 0.8892105263
Public CER  <= 0.0833275793

OOD tol2 >= 0.3265
OOD CER  <= 0.3454290634
```

通过护栏后按 Validation CER、tol2、severe、运行时间排序。Public/OOD 只能否决，
不能重新排序。

### 6.5 结果

| Arm | Val tol2 | Val CER | Severe | Runtime/s |
|---|---:|---:|---:|---:|
| D3_BEAM5 | 0.858974 | **0.084972** | **12** | 60.66 |
| D7_MBR_B5 | **0.860399** | 0.085079 | 13 | 215.74 |
| D2_BEAM4 | 0.858974 | 0.085614 | 13 | 91.25 |
| D0_CURRENT | 0.854701 | 0.085934 | 14 | 85.40 |
| D1_BEAM1 | 0.854701 | 0.085934 | 14 | 84.20 |
| D4_MAX128 | 0.854701 | 0.085934 | 14 | 118.77 |
| D5_RP100 | 0.854701 | 0.085934 | 14 | 118.89 |
| D6_RP110 | 0.856125 | 0.085934 | 14 | 118.90 |

Wave 1 胜出配置为 `D3_BEAM5`：

| Surface | tol2 | CER |
|---|---:|---:|
| Validation | 0.858974 | 0.084972 |
| Public | 0.895789 | 0.080482 |
| OOD | 0.333000 | 0.339350 |

解码胜出配置不会被强制套到后续新模型。每个最终候选仍分别测试 D0 与 D3。

## 7. Wave 2：Official-only 短程回正

### 7.1 实验矩阵

```text
2种更新范围 × 2个学习率 × 2个随机种子 = 8 arms
```

| Mode | LR | Seeds |
|---|---:|---|
| Decoder-only | 5e-7 | 42, 43 |
| Decoder-only | 1e-6 | 42, 43 |
| Full SFT | 5e-7 | 42, 43 |
| Full SFT | 1e-6 | 42, 43 |

### 7.2 统一训练配置

```yaml
initial_checkpoint: RAW_WINNER checkpoint-wenet-50h
training_data: Official only

optimizer: adamw_torch
optimizer_state: reinitialized
scheduler: constant
warmup_steps: 0
warmup_ratio: 0
weight_decay: 0.01
max_grad_norm: 1.0

max_steps: 150
checkpoint_steps: [25, 50, 75, 100, 125, 150]
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
global_effective_batch_size: 16
expected_samples_seen: 2400

evaluation_strategy: steps
eval_batch_size: 4
logging_steps: 1
num_workers: 4
save_total_limit: 32

bf16: true
tf32: true
gradient_checkpointing: false
lora: false
early_stopping: false

generation_num_beams: 2
generation_max_length: 225
generation_no_repeat_ngram_size: 4
generation_repetition_penalty: 1.05
```

命令中的 `epochs=100` 只是高上限，实际停止条件是 `max_steps=150`。每个 arm 内
`seed=data_seed=sampling_seed`，分别使用42与43。seed 不重新生成 manifest。

Wave 2 所有增强均关闭：

```yaml
apply_spec_augment: false
mask_time_prob: 0
mask_feature_prob: 0
online_speed_perturbation: false
online_gaussian_noise: false
online_channel_bandlimit: false
```

### 7.3 Decoder-only 参数范围

冻结 `encoder.conv1`、`encoder.conv2`、`encoder.layers.0-11` 和
`encoder.layer_norm`，训练完整 Decoder 与 `proj_out`。`proj_out.weight` 与
Decoder token embedding 绑定，是同一份可训练参数。

```text
Total parameters:     241,734,912
Trainable:            153,580,800
Frozen:                88,154,112
```

### 7.4 Full SFT 参数范围

Full SFT 更新 Encoder、Decoder 与 `proj_out` 的全部可学习参数：

```text
Total parameters:     241,734,912
Trainable:            240,582,912
Frozen:                 1,152,000
```

固定的 `1,152,000` 参数是 `model.encoder.embed_positions.weight`。因此这里的
Full SFT 表示所有可学习参数开启，不是强行训练固定位置表。

### 7.5 Checkpoint 选择

每个 arm 只从六个预注册 checkpoint 中选择。先要求 Validation tol2 不低于
`0.8497008547`，通过后依次比较 CER、更高 tol2、更低 severe、更早 step。

Trainer 训练结束后重载内部 best model 生成的 `final_*` receipt 不参与选模。
配置族按 paired-seed mean CER，再按 paired-seed mean tol2 排名。

### 7.6 结果

下表为用于 checkpoint 选择的训练时预注册 Validation evaluation：

| Arm | Step | Val tol2 | Val CER | Severe | Insertions |
|---|---:|---:|---:|---:|---:|
| DECODER 5e-7 S42 | 100 | 0.861823 | 0.084972 | 14 | 100 |
| DECODER 5e-7 S43 | 75 | 0.863248 | 0.085079 | 13 | 93 |
| DECODER 1e-6 S42 | 75 | 0.864672 | 0.084117 | 13 | 97 |
| DECODER 1e-6 S43 | 125 | 0.868946 | 0.083903 | 12 | 95 |
| FULL 5e-7 S42 | 125 | 0.864672 | 0.084331 | 13 | 97 |
| FULL 5e-7 S43 | 125 | 0.863248 | 0.084438 | 12 | 93 |
| FULL 1e-6 S42 | 150 | 0.870370 | 0.083048 | 10 | 91 |
| FULL 1e-6 S43 | 125 | 0.868946 | **0.082086** | 10 | **88** |

| Family | Mean tol2 | Mean CER |
|---|---:|---:|
| Full SFT, LR 1e-6 | **0.869658** | **0.082567** |
| Decoder-only, LR 1e-6 | 0.866809 | 0.084010 |
| Full SFT, LR 5e-7 | 0.863960 | 0.084384 |
| Decoder-only, LR 5e-7 | 0.862536 | 0.085026 |

Wave 3 因此固定使用 Full SFT、LR `1e-6`。

## 8. Wave 3：单轴增强

Wave 3 保持 Official-only、Full SFT、LR `1e-6`、150 steps 与 seed 42/43，
只改变一个增强轴。验证预处理始终保持干净。

### 8.1 SpecAugment

使用 Transformers Whisper 原生实现：

```yaml
apply_spec_augment: true
mask_time_prob: 0.05
mask_time_length: 10
mask_time_min_masks: 1
mask_feature_prob: 0.05
mask_feature_length: 8
mask_feature_min_masks: 1
```

Whisper 原生 SpecAugment 没有额外全局启用概率；`mask_*_prob` 描述轴向覆盖比例。

### 8.2 Speed perturbation

每条样本从 `0.9/1.0/1.1` 等概率采样，非1.0样本使用
`librosa.effects.time_stretch`。

| Seed | 0.9 | 1.0 | 1.1 |
|---|---:|---:|---:|
| 42 | 789 | 805 | 806 |
| 43 | 810 | 785 | 805 |

### 8.3 Gaussian noise

```yaml
probability: 0.5
snr_db: Uniform(15, 25)
```

```text
signal_rms = sqrt(mean(audio^2))
noise_rms  = signal_rms / 10^(snr_db / 20)
noise      ~ Normal(0, noise_rms)
augmented  = audio + noise
```

只有 `signal_rms > 1e-8` 时才实际加入噪声；否则保留原音频并在 receipt 中记为
未命中。

| Seed | Applied | Total |
|---|---:|---:|
| 42 | 1212 | 2400 |
| 43 | 1188 | 2400 |

### 8.4 Telephone channel

每条样本以0.5概率执行四阶 Butterworth `300-3400 Hz` band-pass。实现使用
second-order sections 与 `scipy.signal.sosfilt`，随后以 `resample_poly` 从16 kHz
降采样至8 kHz、恢复至16 kHz，并 pad 或截断至原始长度。

| Seed | Applied | Total |
|---|---:|---:|
| 42 | 1203 | 2400 |
| 43 | 1221 | 2400 |

### 8.5 增强随机性与互斥

在线增强使用可复现、worker-local 的独立随机流：

```text
stable_seed(training_seed, worker_seed, augmentation_stream)
```

Speed、noise、channel 使用不同 stream。训练入口强制 SpecAugment、speed、noise、
channel 四个轴互斥。

### 8.6 配置族通过条件

增强族必须两个 seed 都有通过 tol2 护栏的 checkpoint，而且 seed42 和 seed43
的 CER 都优于相同 seed 的无增强 Full-SFT 控制。

### 8.7 结果

下表同样是用于 checkpoint 选择的训练时预注册 Validation evaluation：

| Arm | Step | Val tol2 | Val CER | 相对同seed控制 |
|---|---:|---:|---:|---|
| SPEC S42 | 125 | 0.873219 | 0.083155 | FAIL |
| SPEC S43 | 125 | 0.873219 | 0.083369 | FAIL |
| SPEED S42 | 50 | 0.864672 | 0.084972 | FAIL |
| SPEED S43 | 50 | 0.861823 | 0.085720 | FAIL |
| NOISE S42 | 75 | **0.874644** | 0.082407 | PASS |
| NOISE S43 | 150 | 0.873219 | **0.081979** | PASS |
| CHANNEL S42 | 100 | 0.871795 | 0.084010 | FAIL |
| CHANNEL S43 | 125 | 0.871795 | 0.083903 | FAIL |

唯一通过 paired-seed 规则的增强为 Gaussian noise：`p=0.5`、SNR 15–25 dB。

## 9. 标准化复评与本地 Top 2

训练日志中的 checkpoint evaluation 用于选模。选出的 checkpoint 随后使用独立
评测程序，在固定音频顺序上完整重跑 Validation/Public/OOD。训练时与独立复评
指标都保留，不互相覆盖。例如 NOISE_S43：

```text
training checkpoint CER: 0.081979
independent D0 CER:       0.082086
```

最终对比表使用独立全量复评数据。

### 9.1 Top 1：NOISE_S43

```text
/home/bolin/cantonese-asr/
outputs/raw-winner-p1-v1/wave3_augmentation/v1/
NOISE_S43/checkpoint-150

model.safetensors SHA256:
949eb4e9832482ef7a7fb2de2ac3ff2a80248ab670a20490dad9e7f7c1f91844

selected decoder: D0_CURRENT
```

| Surface | tol2 | tol1 | tol0 | CER | S/D/I | Severe |
|---|---:|---:|---:|---:|---|---:|
| Validation | 0.873219 | 0.725071 | 0.462963 | 0.082086 | 581/95/92 | 12 |
| Public | 0.892632 | 0.754211 | 0.454737 | 0.080393 | 1687/93/27 | 8 |
| OOD | 0.332000 | 0.157500 | 0.042500 | 0.340844 | 8098/68/48 | 515 |

三个表面均为 `repeated runaway=0`、`replacement characters=0`、
`effective max-length=0`。

### 9.2 Top 2：FULL_LR1E6_S43

```text
/home/bolin/cantonese-asr/
outputs/raw-winner-p1-v1/wave2_recenter/v1/
FULL_LR1E6_S43/checkpoint-125

model.safetensors SHA256:
aba4a24989a8a1925ffd2da73bce427241d1fe5f9a6e3805d664a2796b5ad832

selected decoder: D0_CURRENT
```

| Surface | tol2 | tol1 | tol0 | CER | S/D/I | Severe |
|---|---:|---:|---:|---:|---|---:|
| Validation | 0.867521 | 0.719373 | 0.462963 | 0.082407 | 580/102/89 | 11 |
| Public | **0.896316** | **0.756842** | **0.460526** | **0.079592** | 1670/90/29 | 8 |
| OOD | 0.329000 | 0.156000 | 0.042000 | 0.340927 | 8085/84/47 | 515 |

Top 2 的 Public 指标优于 Top 1，但本轮规则禁止使用 Public 重排，因此不能据此
取代 validation 排名更高的 Gaussian-noise family。

### 9.3 相对 RAW_WINNER

| Candidate | Val tol2 | Val CER | Public tol2 | Public CER | OOD tol2 | OOD CER |
|---|---:|---:|---:|---:|---:|---:|
| RAW_WINNER D0 | 0.854701 | 0.085934 | 0.894211 | 0.081328 | 0.331500 | 0.340429 |
| NOISE_S43 | **0.873219** | **0.082086** | 0.892632 | 0.080393 | **0.332000** | 0.340844 |
| FULL_LR1E6_S43 | 0.867521 | 0.082407 | **0.896316** | **0.079592** | 0.329000 | 0.340927 |

## 10. Submission 打包与验证

每个 submission ZIP 必须根目录平铺、只含一个 `model.safetensors`、完全离线
加载、使用冻结的静态 `predict.py`，并通过32条 smoke、原 checkpoint/ZIP 逐句
prediction parity 与权重 SHA256 parity。`outputs_pre` 保留在 ZIP 外部。

### 10.1 NOISE_S43

```text
RAW_WINNER_P1_NOISE_S43_D0_CURRENT_submission.zip
size: 895,748,148 bytes
SHA256: 7a4c39a5874603acdcf7dc5f37be8a30e1d8f40d8650c270c44a198fcc280f85
```

### 10.2 FULL_LR1E6_S43

```text
RAW_WINNER_P1_FULL_LR1E6_S43_D0_CURRENT_submission.zip
size: 895,750,991 bytes
SHA256: c4f845e631775452d325f0be8939b2920373a2878869309435990316e72a9189
```

两个包均通过：

```text
offline32: PASS
raw/ZIP prediction parity: PASS
flat single-weight structure: PASS
weight SHA256 parity: PASS
automatic platform upload: false
```

每个包旁的 `outputs_pre` 至少包含：

```text
metrics.json
top_confusions.csv
error_examples.json
generation.json
summary.json
```

## 11. 工程保障

本轮新增或强化：

- 配置驱动的静态解码 arm 与 `.generate()` 最终参数收据；
- `max_length`/`max_new_tokens` 语义核验；
- D3/D7 rank-0 parity 与确定性字符级 MBR；
- 独立 Gaussian-noise probability 和 telephone-channel augmentation；
- 多增强轴互斥、worker-local 与 stream-specific RNG；
- Validation 无增强、16 kHz/mono/finite-audio 守卫；
- Decoder-only trainable parameter receipt；
- Official/Wenet/CV/MDCC source exposure receipt；
- 预注册 checkpoint 与 end-of-run receipt 分离；
- Public/OOD veto-only 选模守卫；
- D0 基线逐句 parity；
- offline32、raw/ZIP parity 与完整 SHA256 清单。

每个新训练或解码路径先运行32样本 smoke。单个 arm 失败时只重试该 arm，不覆盖
其他成功产物。

## 12. 本轮结论

本轮最稳定的改进路径为：

```text
RAW_WINNER
→ Official-only Full SFT
→ reinitialized AdamW
→ constant LR 1e-6
→ 150步以内选checkpoint
→ paired seeds
→ 可选单轴 Gaussian noise（p=0.5, SNR 15-25 dB）
```

实验证据表明：

- Official-only 短程 Full SFT 明显降低本地 Validation/Public CER；
- LR `1e-6` 稳定优于 `5e-7`；
- Full SFT 稳定优于 Decoder-only；
- Gaussian noise 是唯一两个 seed 都改善 CER 的增强轴；
- SpecAugment 虽提高部分 tol2，但两个 seed 的 CER 均未超过对应无增强控制；
- speed perturbation 与 telephone channel 在本设置中没有收益；
- Beam5 对原 RAW_WINNER 有效，但没有稳定迁移到新训练 checkpoint；
- 每个新权重必须重新验证解码配置，不能假定全局解码 winner 自动泛化。

Top 1 和 Top 2 尚未自动提交平台，因此不能宣称已经超过平台 `69.49`。目前能确认
的是：本地 Validation CER 改善、Public CER 改善、真实 repeated runaway 仍为0，
OOD 基本持平，并且 Gaussian-noise 方向通过 paired-seed 复现条件。

## 13. 仓库入口

服务器运行与安全操作见 [docs/RUNBOOK.md](docs/RUNBOOK.md)。主要实现和审计入口：

- `train.py`：Full SFT、冻结策略、在线增强和 source receipt；
- `predict.py`：离线 Whisper 推理入口；
- `cantonese_asr/metrics.py`：官方指标、normalizer 和错误分析；
- `scripts/evaluate_raw_winner_decode.py`：Wave 1 解码与 MBR；
- `scripts/select_raw_winner_p1_decode.py`：解码 validation 选择；
- `scripts/select_raw_winner_p1_recenter.py`：Wave 2 paired-seed 选择；
- `scripts/select_raw_winner_p1_augmentation.py`：Wave 3 paired-seed 选择；
- `scripts/finalize_raw_winner_p1_improvement.py`：最终矩阵与 Top 2；
- `slurm/raw_winner_p1_*.slurm`：八卡运行链。

本地基础检查：

```bash
python -m pytest -q
python -m compileall -q cantonese_asr scripts train.py predict.py
for file in scripts/*.sh slurm/*.sh slurm/*.slurm; do bash -n "$file"; done
```
