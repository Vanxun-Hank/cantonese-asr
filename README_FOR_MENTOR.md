# Mentor 代码共享快照

> 本文件在 2026-09-06 11:35 的刷新版中曾意外遗失，现依 09-04 快照恢复，并按当前
> 目录结构更正（旧版写的 `code/` 前缀与 `snapshot_manifest.json` 都已不适用）。

这是本地最新开发工作区的代码快照，包含未 commit 的 P2 实现，不是全部已验收的发布版。
训练、推理、配置、测试与现有技术文档均在**本目录根下**（`train.py`、`predict.py`、
`cantonese_asr/`、`configs/`、`scripts/`、`slurm/`、`tests/`、`docs/`）。
P2 文本分析、最终汇总与研究包整合仍在完善；请以 `docs/P2_FULL_V2_EXECUTION.md`
标注为准，不把入口存在当成实验已完成。

## 先读这两份

- `docs/P2_NAS_HANDOFF.md` —— 完整交接说明：来源与状态、所需外部资产（含逐个
  manifest 的 SHA-256 与行数）、主入口清单、依赖安装与验证步骤、安全边界。
  **外层那张 `P2_NAS_HANDOFF_*.txt` 只是摘要，不含资产 SHA，不要只看它。**
- `docs/P2_FULL_V2_EXECUTION.md` —— 逐条执行记录与作业编号，判断"哪些真的跑完了"。

## 来源

- 仓库 `Vanxun-Hank/cantonese-asr`，基线 commit `beb2396808dffdf2ffe275fd14a7777b9f99f550`
- 工作分支 `codex/raw-winner-p2-full`（打包时尚未推送；**2026-09-08 已推送到远端**，
  本文件其余内容仍按 09-06 打包时状态保留，不追溯改写）
- 轮次 `raw_winner_p2_full_v1`，配方 `p2_full_four_gpu_v2`

正在服务器执行的训练代码使用独立冻结目录：
`/home/bolin/cantonese-asr/worktrees/raw-winner-p2-full-v1-deterministic1`
本包可能含冻结之后新增的分析/调度代码，**不应直接覆盖该目录**。
打包时服务器已不可达，因此本包尚未与该冻结目录逐字节比对哈希——这一条只影响
归档溯源，不影响本包作为源码实现的可用性。

## 不包含什么

不包含数据集、模型权重、实验输出、虚拟环境、Git 对象历史、缓存或登录凭证。
原始数据与模型仍依赖服务器已有资产；本包不是可脱离资产直接运行的完整环境。
`SHA256SUMS` 记录逐文件 SHA-256（覆盖除它自身以外的全部文件），替代旧版的
`snapshot_manifest.json`。

未修改主线工作区、未 commit/push，也未改变正在执行的训练。
