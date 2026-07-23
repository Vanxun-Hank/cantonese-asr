# Common Voice 与 MDCC 外部训练数据准备设计

日期：2026-07-23  
状态：已由用户指示“开始下一步”  
范围：数据解压、适配、质检和 Manifest；不启动训练、不改变官方 validation

## 目标

将已获许可的 Common Voice 26.0 `zh-HK` 和 MDCC 转换成现有
`train.py` 可读取的文件式音频 Manifest，同时确保任何与官方固定 validation
或公开自测参考文本重合的外部样本都不会进入训练候选集。

## 方案选择

采用“保留 publisher split + 物化 MDCC 音频”的方案：Common Voice 仅处理
`train.tsv` 对应的 MP3；MDCC 仅处理 `train-*.parquet`，把嵌入的 WAV bytes
原子写入独立目录。相比直接把 Parquet 接入随机访问 DataLoader，这一方案占用更多
磁盘，但训练读取稳定、支持断点重跑，也无需改变 Whisper 训练数据接口。

Common Voice dev/test 和 MDCC validation/test 保留作数据源自身的诊断边界，初始 SFT
不使用。官方 validation 仍是唯一选模依据。

## 文本与泄漏规则

- 训练标签只做 Unicode NFKC、异常空白清理；保留粤语字词和标点。
- 另生成“繁转简 + 去标点空白”的强规范化 key，仅用于去重和泄漏检查。
- 外部文本命中官方 validation 或公开 `template_pre` 的强规范化 key 时隔离。
- 同文本的不同真实录音可以保留；音频 SHA-256 完全相同的后续副本隔离。
- 空文本、缺失/损坏音频、非正时长、超过 30 秒和重复 source ID 均隔离。

## 输出

`artifacts/manifests/external/v1/`：

- `common_voice_train.jsonl`
- `mdcc_train.jsonl`
- `external_all.jsonl`
- `quarantine.jsonl`
- `data_report.json`

MDCC 物化音频写到 `artifacts/data/external/mdcc_train/`。完整外部 Manifest
不会直接与官方训练集拼接；后续实验单独生成官方控制组、外部 25%、外部 50%
和“外部预适应后官方收尾”配置，避免 MDCC 规模淹没官方数据。

## 验收

- 输出 Manifest 中所有文件存在、音频可读、时长在 `(0, 30]` 秒。
- 受保护文本 overlap 为 0，音频 SHA-256 无重复。
- 数据源、publisher split、原始文本、清洗文本、时长、采样率、声道和来源位置可追溯。
- 脚本可重复执行；已正确物化的 MDCC WAV 不重复写入。
