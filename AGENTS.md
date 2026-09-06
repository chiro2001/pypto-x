# PyPTO-X Agent Instructions

## 语言

- 除非用户明确要求其他语言，始终使用简体中文回复。

## 接手顺序

1. 从项目根目录 `/home/chiro/projects/pypto/pypto_x` 开始。
2. 完整阅读根目录 `HANDOFF.zh-CN.md`。
3. 再阅读 `README.md`、`docs/WORKTREE_AGENT_PLAN.zh-CN.md`、`docs/PROJECT_LAYOUT.zh-CN.md`、`docs/RESOURCE_MATRIX.zh-CN.md` 和 `docs/SMOKE_TEST_SPEC.zh-CN.md`。
4. 进入嵌套仓库后，还必须遵守该仓库中距离目标文件最近的 `AGENTS.md`。

## 当前阶段

- 当前状态是 `EXECUTION_W2_COMPLETE_W2B_READY`。
- 用户已批准按现有计划和 subagent 协议执行；W2 已冻结，下一步执行 portable bootstrap 门禁。

## Git 与目录

- `upstream/pypto` 是 PyPTO-X 的代码主仓，当前 edge 快照为 `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad`；stable lock 尚待 CANN release 配套验证。
- `upstream/*` 原则上只读；禁止直接在 `master`/`main` 开发。
- 实现必须使用 `scripts/worktree/create.sh` 创建的独立 worktree，默认位于 `../worktrees/pypto-x/<task>`。
- 根目录是轻量控制 Git 仓；五个 `upstream/*` 以固定 commit 的 submodule/gitlink 登记。
- 不清理或重置未知修改。`upstream/PTOAS/.codex/CLAUDE.md` 的 dirty 状态来自上游 CRLF/`.gitattributes` 不一致，不是人工改动。

## Subagent 协议

- 只有用户明确批准进入实现阶段后才启动 subagent。
- 每个 subagent 一个 task、一个分支、一个 worktree。
- 启动参数必须包含 `started_at=<ISO-8601 UTC>`、绝对 worktree、branch、`smoke_once=true`、`wait_timeout_seconds=3600`、`poll=false`。
- 启动时省略 `model`/`model_name`；使用默认模型。
- 不下载或加载 LLM 权重。每个 subagent 启动后只执行一次无模型冒烟测试。
- 派发后父 agent 使用一次 `wait_agent(timeout_ms=3600000)` 长等待，不做周期轮询。

## 工程边界

- `pypto` Tensor 与 `pypto_pro` Professional 是同一 wheel 内的并存前端；当前前端分层策略已重新打开，未对齐前不得假定 Pro 替代 classic。
- 公共层只表达 Tensor/Scalar/Shape/控制流/逻辑 Tile；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe 和 CCE 语义必须留在 Ascend target。
- CPU 共享 lowering，开发优先级为 x86 AVX2/AVX-512 → AArch64 SVE256（不开发 NEON 优化后端）→ NVIDIA → AMD。
- NVIDIA 与 AMD 共享 GPU IR/Runtime ABI，再分别下降到 CUDA/NVVM 和 HIP/ROCDL。
- 首个实际模型为固定 revision 的 `Qwen/Qwen3.5-0.8B`，首期只做纯文本 BF16 与 W8A8-linear；不做 W8A16、INT4、FP8、KV/GDR state 量化。
- 视觉编码器不纳入当前 MVP；性能门槛在首批后端完成后再根据实测冻结。
- 当前正式 CANN 许可证仍限制非华为处理器用途。维护方态度积极，但对外发布前必须取得新许可证、双许可证或明确书面例外。

## 资源安全

- RTX 5080 主机 `192.168.101.5` 的 SSH 默认进入 Windows `cmd`；Linux 命令必须通过 `wsl.exe -e bash -lc`。
- AMD 6750GRE 尚未接入，不得声称 HIP 已在真机运行。
- QEMU 只用于 AArch64/SVE 功能验证，不得用其数字作性能结论。
- 鲲鹏机器尚未到位；拿到后先做能力探测，再跑原生性能测试。
