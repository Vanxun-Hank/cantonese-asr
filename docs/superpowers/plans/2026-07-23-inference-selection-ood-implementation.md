# 推理一致性、选模护栏与 OOD 诊断实施计划

日期：2026-07-23  
依据：`docs/superpowers/specs/2026-07-23-inference-selection-ood-design.md`

## Task 1：统一生成长度

文件：`predict.py`、`tests/test_predict.py`

1. 为 `predict.py` 增加 `--generation-max-length`，默认 225。
2. 拒绝非正值，并将参数显式传入 `model.generate(max_length=...)`。
3. 用 mock processor/model 测试默认值、显式值和非法值，不加载真实权重。

## Task 2：实现硬护栏和确定性选模

文件：`scripts/select_best_checkpoint.py`、`tests/test_select_best_checkpoint.py`

1. 增加默认 `--max-cer 0.1163` 与 `--min-sentence-accuracy 0.8219`。
2. 从独立 validation diagnostics 读取准确率/CER。
3. 从 run `metrics.jsonl` 按 checkpoint step 关联 validation loss；best_model 通过
   `trainer_state.json.best_model_checkpoint` 解析来源 step。
4. 去重同一来源 checkpoint，并按准确率、CER、loss、较早 step 排序。
5. 测试护栏、全部并列级别、loss 关联失败和无合格候选失败。

## Task 3：准备固定 OOD 面板与完整 Manifest

文件：`scripts/prepare_ood_manifest.py`、`tests/test_prepare_ood_manifest.py`

1. 读取 CV dev/test TSV 与 MDCC validation/test Parquet。
2. 物化/复用音频，执行可读性、时长、ID、音频 hash、受保护文本和 provenance QC。
3. 输出四个完整 split Manifest、合并完整 Manifest、固定 seed 42 的最多 2,000 条
   panel、quarantine 和 data report。
4. 测试确定性、每 split 上限、字段、哈希去重和泄漏隔离。

## Task 4：计算 teacher-forced loss 并扩展 checkpoint 评测

文件：`scripts/evaluate_manifest_loss.py`、`scripts/evaluate_checkpoints.py`、相关测试

1. 使用与训练一致的 feature/label padding 做无梯度 forward，计算 token 加权 loss。
2. 为 validation、train-probe、OOD panel 保存 loss、样本/token 数及 Manifest hash。
3. 扩展 checkpoint 评测调用统一 225-token 生成和 loss 评测。
4. 保证 OOD 仅写诊断，不进入选模输入。

## Task 5：报告与验收

文件：`scripts/plot_experiments.py`、`scripts/check_goal1.py`、测试

1. 增加 OOD source/split 汇总与曲线，但 scoreboard 排名只使用官方 validation。
2. 验证提交包中的 `predict.py` 默认 225-token 且单权重、无网络和绝对路径。
3. 运行服务器完整测试并同步代码。

## Task 6：服务器数据和当前四组重评

1. 准备并 QC OOD 全量/固定 panel。
2. 四卡并行重评四组每个 checkpoint 的 validation、train-probe、OOD panel。
3. 用硬护栏选择 checkpoint，刷新 PNG/CSV/JSON/HTML 报告。
4. 对 validation 选出的最终候选运行完整 OOD。
5. 构建、断网验证候选 submission.zip，并在平台确认前报告内部证据。

