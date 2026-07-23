# 点心杯粤语 ASR Goal 1：Whisper-small SFT 实施计划

日期：2026-07-22  
状态：待用户批准后执行  
范围：官方数据 Full SFT 主线；预留外部数据接口；不包含蒸馏

## 1. 目标与完成定义

在琶洲 Slurm 集群上部署一套可复现的粤语 ASR 工程，使用 `openai/whisper-small` 和官方 `cantonese-life-scenarios-corpus` 完成 Full SFT，通过四张 RTX 4090 并行比较训练配置，产出符合点心杯初赛离线评测协议的单一 Whisper-small 提交包。

完成必须同时满足：

1. 官方音频与 `index.csv` 通过一一匹配和数据质检；公开自测集未进入训练。
2. 原始 Whisper-small 零样本、单卡 smoke test、四卡 SFT 和阶段性自测均有可复核记录。
3. 内部验证集 `sentence_accuracy_tol2` 至少比零样本高 2 个百分点，且 CER 不退化。
4. 连续两轮实验的最佳提升均不足 0.5 个百分点，或已到截止前两天，结束调参。
5. `submission.zip` 完全离线加载成功，只有一个 Whisper-small 权重，输出行数、顺序及 `audio_path` 与测试列表完全一致。

## 2. 官方边界

- 基座固定为 `openai/whisper-small`；Goal 1 使用 Full SFT。
- 不修改 Transformer 层数、attention head、GELU、tokenizer 或词表。
- 不使用模型投票、融合、级联 ASR 或多个推理权重。
- `test_audio.zip` 与 `template_pre.jsonl` 仅用于阶段性自测。
- 外部数据可后续接入，但在确认来源、许可证和数据类型前不阻塞官方数据主线。
- SenseVoice 和蒸馏不属于 Goal 1；如团队决定使用，另建 Goal 2。
- 正式提交适配 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`，推理完全断网。

## 3. 本地与服务器职责

### Mac 本地

项目根目录：`/Users/zhangxun/Desktop/service/cantonese-asr`

保存代码、Notebook、实验配置、Slurm 文件、测试和小型报告；不保存完整数据、模型权重或 checkpoint。将现有 `fine_tune_whisper.ipynb` 重构为 `notebooks/fine_tune_whisper_cantonese.ipynb`，仅用于教学、数据抽查和短 smoke test。

### 服务器

项目根目录：`/home/bolin/cantonese-asr`

保存：

```text
artifacts/
  datasets/official/     原始仓库文件和 ZIP
  data/train_raw/        官方训练音频
  data/test_audio/       公开自测音频
  data/quarantine/       无法自动确认的样本
  manifests/             all/train/validation/quarantine JSONL
  reports/               数据和实验报告
  models/whisper-small/  离线基座
outputs/                 trial、checkpoint、最佳模型、提交包
logs/                    Slurm 日志
```

Python 环境固定为 `/home/bolin/envs/cantonese-asr-whisper`。服务器 `/home` 为 NFS；登录节点负责下载和调度，GPU 节点只读本地资产。

## 4. 阶段 A：服务器预检与部署

1. 用户连接 EasyConnect；Codex 验证 `ssh pavb`。
2. 只读检查登录节点、`/home/bolin` 空间、`sinfo`、`squeue`、GPU 节点状态。
3. 申请一张 GPU 做 `nvidia-smi` 和最小 PyTorch CUDA 测试；未通过则不进入训练。
4. 用 `rsync` 同步本地代码，排除 `artifacts/`、`outputs/`、`logs/` 和缓存。
5. 使用公共 Miniconda 创建 Python 3.11 环境；Slurm 运行时直接使用环境绝对路径，不依赖非交互 shell 中的 `module load`。
6. 安装并固定：PyTorch 2.6.0、Torchaudio 2.6.0、CUDA 12.4 wheel、Transformers 4.57.6、Datasets 3.6.0、Accelerate 1.12.0 及项目依赖。
7. 保存 `pip freeze`、Python、PyTorch、CUDA、GPU 和 Slurm 信息至部署报告。

验收：环境可在一张 4090 上加载 CUDA tensor，并能离线导入全部训练与推理依赖。

## 5. 阶段 B：下载与原始资产验证

1. 在登录节点设置 `HF_ENDPOINT=https://hf-mirror.com`。
2. 下载整个官方数据仓库到 `artifacts/datasets/official/`。
3. 下载 `openai/whisper-small` 到 `artifacts/models/whisper-small/`。
4. 对 `data.zip`、`test_audio.zip`、`index.csv`、模型权重记录文件大小和 SHA-256。
5. 先列出 ZIP 内容，再分别解压到 `train_raw/` 和 `test_audio/`；原始 ZIP 保留不变。
6. 下载失败时依次尝试 HF mirror 和 ModelScope；禁止在 GPU 作业中联网下载。

验收：数据仓库文件齐全、ZIP 可测试解压、Whisper processor/model 可在 `local_files_only=True` 下加载。

## 6. 阶段 C：官方数据整理

### 6.1 数据接口

重构 `prepare_manifest.py`，接口固定为：

```text
--index-csv
--audio-root
--output-dir
--exclude-test-list artifacts/datasets/official/template_pre.jsonl
--validation-ratio 0.1
--seed 42
```

`index.csv` 中的“粤语原文”是唯一 SFT 标签。普通话翻译和粤拼只作为元数据；不再从文件名猜完整转写。

### 6.2 匹配与质检

- 读取“序号”为字符串，规范化前导零和 CSV 数字格式。
- 递归扫描 WAV，提取文件名前导数字并与“序号”连接。
- 将 `template_pre.jsonl` 的 1,900 个公开自测 ID 写入独立 `public_excluded.jsonl`，禁止进入 train、validation 或 train-probe。
- 对缺失、一对多、多对一、空标签、重复 ID、不可读音频、异常时长、采样率和声道生成明确报告。
- 不可读、空音频或时长不大于 0 的样本直接隔离；超过 Whisper 30 秒输入窗口的样本先进入 quarantine 并报告，未经人工确认切分边界前不截断音频或复用整句标签。
- 不能唯一确认的样本写入 `quarantine.jsonl`，不静默丢弃或猜测。
- 原始 WAV 不覆盖；训练读取时转为 16 kHz 单声道。

### 6.3 文本规范

训练文本仅做 Unicode NFKC、首尾空白和异常空白清理；保留官方标点和粤语字词，不改写为普通话，不删除“唔、咗、嘅、㗎”等字，不进行破坏性繁简转换。

### 6.4 Manifest

每条至少包含：

```json
{
  "id": "00001",
  "audio_path": "artifacts/data/train_raw/.../00001.wav",
  "text": "你好！",
  "scene": "1问候场景",
  "mandarin": "你好！",
  "jyutping": "nei5 hou2！",
  "duration_s": 1.24,
  "split": "train",
  "source": "official"
}
```

生成 `all.jsonl`、`train.jsonl`、`validation.jsonl`、`quarantine.jsonl` 和 `data_report.json`。按规范化文本分组防泄漏，并尽量保持场景比例；固定 90/10、seed 42。`test_audio` 零训练引用。

验收：有效样本 100% 可读取、音频文本一一对应、无空标签、train/validation 无规范化文本重叠、公开自测集未进入 manifest。

## 7. 阶段 D：Notebook 和训练代码

### Notebook

删除 Hindi Common Voice、Hugging Face 登录、Hub 上传、Gradio 和 Hindi WER 主指标。加入本地 manifest、随机播放、标签/场景/时长展示、Log-Mel 可视化、`language="zh"`、官方指标和短 smoke test。Notebook 不承担正式长训练。

### 训练脚本

`train.py` 读取本地模型及 train/validation manifest，使用 `WhisperProcessor(language="zh", task="transcribe")`。训练与生成配置显式锁定 `language="zh"`、`task="transcribe"` 和 `forced_decoder_ids=None`；训练时关闭 `use_cache`，推理时恢复。Data collator 分别 padding 音频特征与标签，标签 padding 置为 `-100`。模型按 token cross-entropy 反向传播；AdamW 更新全部可训练参数。

每个 epoch 写标准 `metrics.jsonl`，字段包括 trial、epoch、train/validation loss、`sentence_accuracy_tol2`、CER、learning rate、runtime、峰值显存、checkpoint、Slurm job ID 和状态。支持 `resume_from_checkpoint`。

### 官方指标

训练期指标与官方 `evaluator_pre.py` 做回归对照：预测繁转简，参考与预测去中英文标点和空白，字符编辑距离不超过 2 判整句正确；CER 作为保护指标。最佳模型按 `sentence_accuracy_tol2` 选择。

## 8. 阶段 E：零样本与单卡 smoke test

1. 原始 Whisper-small 在 internal validation 和 `template_pre` 各运行一次，记录整句准确率、CER、耗时、峰值显存、字符混淆和错误样例。
2. 取 16–32 条训练样本在一张 4090 上尝试过拟合，验证 forward、loss、backward、optimizer、scheduler、保存、恢复和 `predict.py`。
3. Smoke test 必须观察到 loss 明显下降且模型逐渐记住样本；否则先修数据或代码，不启动四卡实验。

## 9. 阶段 F：四卡 Full SFT 第一轮

共同配置：Full SFT、AdamW、weight decay 0.01、warmup ratio 0.05、3 epochs、每卡 batch 4、gradient accumulation 4、effective batch 16、gradient clipping 1.0、bf16、gradient checkpointing、seed 42。

使用四个独立单卡 Slurm array trial：

| Trial | Learning rate | Scheduler |
|---|---:|---|
| 0 | 5e-6 | linear |
| 1 | 1e-5 | linear |
| 2 | 2e-5 | linear |
| 3 | 1e-5 | cosine |

每个 epoch 生成独立指标；汇总器输出 `scoreboard.csv/json`。Goal 发现新 epoch 或作业状态变化时报告四卡结果。Accuracy/CER 只用于评估和模型选择，不直接更新权重。

## 10. 阶段 G：实验迭代

1. 先按 `sentence_accuracy_tol2`，再以 CER 和稳定性选择最佳 trial。
2. 下一轮围绕最佳配置只改变一个变量；先局部 learning-rate 搜索，再比较 SpecAugment、语速扰动或音量扰动。
3. 验证和测试不做增强；没有清晰许可证的噪声数据不使用。
4. OOM 时 batch 减半、gradient accumulation 翻倍，保持 effective batch；中断从最后 checkpoint 恢复；单个 trial 失败不取消其他 trial。
5. 新一轮提升至少 0.5 个百分点才继续搜索；连续两轮不足 0.5 个百分点即收敛。
6. `template_pre` 只检查阶段性最佳模型，不用于每轮调参。
7. 每个 trial 固化 resolved config、随机种子、manifest SHA-256、代码快照 SHA-256、`pip freeze`、Slurm 脚本、日志和最佳 checkpoint 路径；在项目建立 Git 仓库后再补充 commit ID。

## 11. 可选外部数据扩展

Goal 1 不等待外部数据。训练接口预留 `source`、可选 `target_field` 和按数据源采样权重。

- 有标注“音频+粤语文本”：质检后写入独立 `external_labeled.jsonl`。
- 只有音频：不直接 SFT；留给 Goal 2 的教师伪标签路线。
- 只有文本：不进入 Whisper 声学微调。

任何外部数据必须先保存“比赛允许外部数据”的书面规则或主办方确认，再记录来源、版本、许可证、SHA-256 和处理历史，并进行与官方 validation/test 的文本及音频去重。官方 validation 永远不混入外部数据。找到合格数据后，用四卡对照：官方控制组、25% 外部采样、50% 外部采样、外部预适应后官方收尾；只有验证集提升才替换官方 baseline。

## 12. 阶段 H：推理与提交

`predict.py` 必须支持官方三个参数：`--audio_dir`、`--test_list`、`--output_jsonl`；支持官方 CSV 和公开 JSONL，自始至终保留输入行顺序及原始 `audio_path`，只从提交根目录离线加载模型。

`package_submission.py` 复制最佳模型和 processor/tokenizer 文件到平铺目录，拒绝多权重、嵌套目录、绝对本机路径和 SenseVoice 依赖。提交包至少包含 `predict.py`、一个 `model.safetensors`、模型/生成/预处理配置、tokenizer/vocab/merges 和必要的最小 requirements。

## 13. 测试矩阵与最终验收

### 单元测试

- 序号规范化和 WAV/CSV 一一匹配。
- 重复、缺失、空标签和损坏音频进入 quarantine。
- 官方文本规范化、编辑距离、整句准确率与 CER。
- test list CSV/JSONL 解析、路径解析和输出顺序。
- 打包器拒绝多个权重和嵌套目录。

### 集成测试

- 32 条样本完成预处理、训练、保存、恢复和推理。
- 四个 array 配置可独立解析并写入不同目录。
- `predict.py` 在无网络环境加载本地模型并输出合法 JSONL。
- 自定义指标与官方 `evaluator_pre.py` 在同一预测文件上结果一致。

### 提交验收

- `submission.zip` 根目录平铺且只有一个 ASR 权重。
- 在与官方相容的 PyTorch 2.6/CUDA 12.4 环境设置离线变量后加载成功。
- 对完整 `template_pre` 输出行数、顺序、`audio_path` 100% 一致，无缺失、重复或多余项。
- 官方自测脚本成功产出 metrics、混淆和错误样例报告。

## 14. 执行顺序与时间边界

1. 部署/预检与资产下载。
2. 数据整理和质检。
3. 零样本与 smoke test。
4. 四卡第一轮和局部搜索。
5. 必要的数据增强实验。
6. 阶段性 `template_pre` 复核。
7. 最终模型、离线测试和提交包。

所有调参最迟在官方截止前两天停止，最后两天只允许修复提交兼容性、复测和打包，不再引入新方法。

## 15. Goal 运行与阻塞策略

计划获批后才创建 Goal 1。Goal 持续执行部署、数据、训练、监控、恢复、比较和打包，并在每个新 epoch 或状态变化时汇报。它不会自动加入蒸馏、未批准外部数据或模型集成。

若 EasyConnect/SSH、下载源、数据唯一匹配、GPU 调度或官方关键规则构成阻塞，先完成仍可进行的本地工作并记录证据。相同阻塞连续三个 Goal 回合、且没有安全替代路径时，报告具体命令/日志、未满足 criterion 和解锁动作；不因单次失败标记 blocked。

## 16. 当前非执行事项

- 本文批准前不修改训练代码、不下载资产、不创建服务器环境、不提交 Slurm 作业。
- 当前目录不是 Git 仓库；不擅自 `git init`。本文保存到本地 docs，但无法提交 commit，等待团队确定版本库后再纳入版本控制。
