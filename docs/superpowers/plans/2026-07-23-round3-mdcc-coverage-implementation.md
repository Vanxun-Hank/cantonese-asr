# 点心杯粤语 ASR Round 3：MDCC 数据覆盖实施计划

日期：2026-07-23  
依据：`docs/superpowers/specs/2026-07-23-round3-mdcc-coverage-design.md`

## Task 1：扩展确定性 Manifest 构建器

文件：`scripts/build_external_training_mixes.py`、
`tests/test_build_external_training_mixes.py`

1. 修正 `--external-fraction` 的显式参数行为，使调用方提供比例时不再隐式保留
   默认的 25%/50%。
2. 增加嵌套无放回抽样：较小覆盖组的每个来源样本 ID 必须是较大覆盖组的子集。
3. 增加全量 publisher-train 组，恰好保留所有官方/CV/MDCC train ID。
4. 在 `mix_report.json` 记录实际来源数量、外部比例、嵌套策略、时长和 SHA-256。
5. 测试 73%/80%/85% 数量、CV 耗尽后的 MDCC 补充、嵌套关系、全量覆盖和重跑
   字节级确定性。

## Task 2：增加 Round 3 Manifest 准备作业

文件：`slurm/prepare_external_round3.slurm`

1. 从现有已 QC 的官方、CV train、MDCC train Manifest 构建 Round 3。
2. 生成 `external73.jsonl`、`external80.jsonl`、`external85.jsonl` 和
   `all_train.jsonl`。
3. 用服务器实际输入数量校验每组来源计数、ID 唯一性、嵌套关系和文件哈希。
4. 不读取或修改固定 validation、公开测试及 OOD Manifest。

## Task 3：增加四卡训练和自动评测

文件：`slurm/external_round3.slurm`

1. 建立四任务 Slurm array，分别映射到四个独立 Manifest 和输出目录。
2. 固定 Full SFT、AdamW、LR `2e-5`、cosine、3 epochs、batch 8、grad accumulation
   2、warmup 0.05、weight decay 0.01、bf16/TF32、seed 42。
3. 每个 epoch 保存 checkpoint；训练后统一评测 validation、train-probe 和固定
   2,000 条 OOD panel。
4. 使用准确率/CER/loss/较早 epoch 的现有硬护栏选模。
5. 单个 task 失败时保留其他 task 的结果和日志。

## Task 4：Round 3 报告与候选选择

文件：`slurm/report_external_round3.slurm`、现有报告/全局选模入口

1. 生成独立 Round 3 报告目录，不覆盖 Round 2。
2. 汇总四组数据量、训练曲线、validation/OOD 指标、速度和显存。
3. 只使用固定 validation 选择 Round 3 recipe；完整 OOD 只评测最终合格候选。
4. 将 Round 3 候选加入最终全局选择入口，但不覆盖历史证据。

## Task 5：本地与服务器验证

1. 运行相关单元测试、Shell 语法检查、`git diff --check` 和完整测试集。
2. 提交本地实现，使用现有同步脚本传到服务器。
3. 服务器重跑测试，核对外部池实际数量、磁盘和四张 GPU。
4. 先运行 Manifest 准备作业，检查 `mix_report.json` 后再启动四卡训练。
5. 每个 epoch 汇报 loss、validation accuracy/CER、OOD、显存、速度及 Slurm 状态。

## Task 6：交付

1. 满足 validation/CER 护栏的候选运行完整 27,177 条 OOD。
2. 若验证准确率提升至少 0.5 个百分点，或验证持平且完整 OOD 明显提升，生成新的
   单权重离线包。
3. 断网验证官方 `predict.py` 协议、输出顺序和单权重结构。
4. 向用户提供 Mac 本地提交包绝对路径、大小和 SHA-256，由用户手动上传平台。
