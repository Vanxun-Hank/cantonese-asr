# Goal 1 服务器运行手册

所有命令默认在 `/home/bolin/cantonese-asr` 执行。现有非 ASR Slurm 作业不得取消或修改。

## 1. 同步与环境

Mac 连接 EasyConnect 后：

```bash
cd /Users/zhangxun/Desktop/service/cantonese-asr
bash scripts/sync_server.sh
ssh pavb
cd /home/bolin/cantonese-asr
bash scripts/server_bootstrap.sh
```

环境固定为 `/home/bolin/envs/cantonese-asr-whisper`。部署证据保存在 `artifacts/reports/deployment/`。

## 2. 下载、解压和数据质检

```bash
export HF_ENDPOINT=https://hf-mirror.com
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/download_assets.py
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/extract_assets.py
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/check_metric_parity.py \
  --official-evaluator artifacts/datasets/official/evaluator.py
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/prepare_manifest.py \
  --index-csv artifacts/datasets/official/index.csv \
  --audio-root artifacts/data/train_raw \
  --output-dir artifacts/manifests \
  --exclude-test-list artifacts/datasets/official/template_pre.jsonl \
  --validation-ratio 0.1 \
  --seed 42 \
  --train-probe-size 256 \
  --max-duration 30
```

继续前必须确认 `data_report.json`：train/validation 非空、两种 overlap 均为 0、quarantine 原因可解释。不得从文件名猜转写。

## 3. 三个训练前 gate

```bash
PRE=$(sbatch --parsable slurm/gpu_preflight.slurm)
ZERO=$(sbatch --parsable --dependency=afterok:$PRE slurm/zero_shot.slurm)
SMOKE=$(sbatch --parsable --dependency=afterok:$ZERO slurm/smoke.slurm)
MEM=$(sbatch --parsable --dependency=afterok:$SMOKE slurm/memory_probe.slurm)
printf '%s\n' "preflight=$PRE" "zero_shot=$ZERO" "smoke=$SMOKE" "memory_probe=$MEM"
```

- GPU gate：CUDA tensor 成功且型号为 RTX 4090。
- Zero-shot gate：internal validation 与公开 template 各有完整指标。
- Smoke gate：16–32 条样本的后期 loss 明显低于初始 loss，保存、恢复、推理成功。
- Memory gate：4/6/8 micro-batch 各跑 20 step；选择无 OOM、无 NaN 且峰值显存 ≤ 约 90% 的最大值。

## 4. 四卡第一轮

若显存探测选择 batch 8，则 accumulation 2；batch 6 对应 3；batch 4 对应 4：

```bash
BATCH_SIZE=4 GRAD_ACCUM=4 bash scripts/submit_grid.sh
squeue -u bolin
```

四个独立单卡 trial：`5e-6/linear`、`1e-5/linear`、`2e-5/linear`、`1e-5/cosine`。每个 epoch 写 `metrics.jsonl`；作业结束生成 checkpoint 级 validation/train-probe 场景、混淆和错误报告以及固定图表。

聚合报告：

```bash
/home/bolin/envs/cantonese-asr-whisper/bin/python scripts/plot_experiments.py \
  --outputs-root outputs \
  --data-report artifacts/manifests/data_report.json \
  --report-dir artifacts/reports/experiments/all
```

## 5. Batch 对照与最终长跑

从 `scoreboard.csv` 选 validation 整句准确率最高者，CER 用于同分和退化保护。Batch 对照示例：

```bash
sbatch --export=ALL,BEST_LR=1e-5,BEST_SCHEDULER=linear,BATCH_SIZE=4 \
  slurm/batch_ablation.slurm
```

锁定配置后从原始基座重新训练最多 8 epochs，patience 2：

```bash
sbatch --export=ALL,FINAL_LR=1e-5,FINAL_SCHEDULER=linear,BATCH_SIZE=4,GRAD_ACCUM=4 \
  slurm/final_train.slurm
```

不得根据 train-probe 或 `template_pre` 选 checkpoint。后续实验每轮只改变一个变量；连续两轮提升不足 0.5 个百分点停止该方向。

## 6. 打包与最终验收

```bash
PY=/home/bolin/envs/cantonese-asr-whisper/bin/python
$PY scripts/package_submission.py \
  --model-dir outputs/final/<trial>/best_model \
  --predict-py predict.py \
  --requirements requirements-submission.txt \
  --output-zip outputs/submission/submission.zip

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY predict.py \
  --model_dir outputs/final/<trial>/best_model \
  --audio_dir artifacts/data/train_raw/4856-10393 \
  --test_list artifacts/datasets/official/template_pre.jsonl \
  --output_jsonl outputs/submission/template_predictions.jsonl

$PY scripts/check_goal1.py \
  --submission-zip outputs/submission/submission.zip
```

官方仓库当前的 `template_pre.jsonl` 参考音频位于训练 ZIP 的
`4856-10393/` 目录，已通过 `public_excluded.jsonl` 从 SFT 数据排除；随机命名的
`test_audio.zip` 对应无参考答案的 `template.jsonl`，只用于正式路径协议检查。

只有 `check_goal1.py` 全部通过，且官方相容 PyTorch 2.6/CUDA 12.4 容器完成断网加载，Goal 1 才能标记完成。
