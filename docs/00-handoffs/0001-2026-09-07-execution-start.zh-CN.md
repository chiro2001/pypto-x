# PyPTO-X 执行起点交接快照

归档序号：`0001`

归档日期：2026-09-07（Asia/Shanghai）

状态：`EXECUTION_APPROVED_C0`

## 冻结范围

- Tensor frontend 是跨架构 Core IR 的主要语义入口。
- `pypto_pro` 保留为 Ascend expert dialect；只有明确 portable subset 可进入公共层。
- 后端顺序：CPU scalar reference → AVX2 → AVX-512 → SVE256 → NVIDIA → AMD；不开发 NEON 优化后端。
- 首个模型固定为 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，只做纯文本 BF16 与 W8A8-linear；视觉编码器不在 MVP。
- 模型权重不由 smoke 或 subagent 下载；性能门槛在首批后端实测后冻结。

## 执行基线

五个 edge 仓库在 2026-09-07 再次与远端默认分支核对：

| 仓库 | commit |
|---|---|
| PyPTO | `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad` |
| PyPTO-Gym | `945a360e12592239a3549cb62d0db37af32bbc03` |
| PTO-ISA | `668248ec886447a83787200786fe6f461169b701` |
| community PyPTO | `9f657f37ed20ce148b46fb7229c267a152a0644e` |
| PTOAS | `dc15ee5b9e459c025eb4f714f2f892b535d93eb0` |

stable lock 仍为 `pending_cann_toolkit_and_npu_validation`。缺少实际 CANN toolkit/NPU 配套环境，不能把 edge 候选宣称为 stable。

## 治理迁移

- 根目录初始化为轻量控制 Git 仓；五个 `upstream/*` 以固定 commit 的 submodule/gitlink 登记。
- 现有 `upstream/pypto` linked worktree 必须原地保留，不重新 clone 或吸收其 Git 元数据。
- `upstream/PTOAS` 已知的 CRLF dirty 与遗留 `3rdparty/` 不清理、不重置。
- 实现继续在项目外的 `../worktrees/pypto-x/<task>` 中进行。

## 第一执行波次

W1 包含 `target-abi`、`core-ir`、`verification`。三者共同遵守
[W1 公共接口 RFC](../10-architecture/0001-2026-09-07-w1-core-target-contract.zh-CN.md)，并使用独立分支、worktree 和一次无模型 smoke。
