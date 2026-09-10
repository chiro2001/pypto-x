# PyPTO-X Agent Instructions

## 语言

- 除非用户明确要求其他语言，始终使用简体中文回复。

## 接手顺序

1. 从项目根目录 `/home/chiro/projects/pypto/pypto_x` 开始。
2. 完整阅读根目录 `HANDOFF.zh-CN.md`。
3. 再阅读 `README.md`、`docs/WORKTREE_AGENT_PLAN.zh-CN.md`、`docs/PROJECT_LAYOUT.zh-CN.md`、`docs/RESOURCE_MATRIX.zh-CN.md` 和 `docs/SMOKE_TEST_SPEC.zh-CN.md`。
4. 进入嵌套仓库后，还必须遵守该仓库中距离目标文件最近的 `AGENTS.md`。

## 当前阶段

- 当前状态是 `EXECUTION_W8A_QWEN35_BF16_WEIGHTED_FORWARD_ALIGNED_PENDING_VERIFICATION_AMD_RUNTIME_STATIC_ONLY`。
- Qwen3.5-0.8B M0–M1K 已完成并冻结；M1I 是24层无权重 decoder，M1J 是3/320/48 external buffer binding，M1K 已让 scalar、AVX2/AVX-512、SVE256 与 CUDA 正确摄取 typed byte views。
- 2026-09-10 实现侧已跑通**真权重整网**并修复 GDR decay 门（缺 `exp(A_log)`，见 `docs/00-handoffs/ERRATA.zh-CN.md` ERR-0001）：prefill argmax 5/5、cosine 0.99991，decode 11751→13→198→760 与 gold 一致；**独立验收 agent 进行中，验收通过前不得写成已冻结结论**。图契约已升到 v3（4,550 ops）。
- 用户已于 2026-09-10 明确授权下载/加载固定 revision 的 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` 权重，用于纯文本 BF16 带权执行；权重位于 `../worktrees/_meta/pypto-x/assets/qwen35-0.8b/<revision>/`，未进入 Git。带权前向结果产生前，仍不得声称整网已运行。
- AMD `amd-igpu-gfx1036` 静态 C3 已覆盖 math、任意轴 reduction 与 rank-2/3/4 exact-batch matmul，Qwen 静态 lowering 为4,532/4,532 ops（graph v2 计数；v3 修复 GDR decay 门后为4,550，见 `docs/00-handoffs/ERRATA.zh-CN.md`）。WSL 运行态仍为 `BLOCKED_DEVICE`（只有 `/dev/dxg`，无 `/dev/kfd`/ROCm/HIP）；完整静态 lowering 不得表述为 HIP 真机 PASS。

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
- 命令较多的阶段验收也交给独立 subagent：从待验收 integration HEAD 创建专用验收 worktree，源码只读，只把日志/报告写入独占 `_meta` 目录；不得让验收 agent 与实现 agent 共用 worktree。
- 启动消息还必须包含 `resource_lock_root=/home/chiro/projects/.resource-locks`、`local_heavy_policy=locked`、`local_heavy_runner=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh`、`local_min_available_mib=8192`、`local_safety_floor_mib=4096`、`local_max_cpus=6`。
- full pytest、大 shape lowering/compile、并行构建或预计使用本机至少一半 CPU/4 GiB 内存的命令，必须由 `run_local_heavy.sh` 取得 `local` 锁并受 cgroup/运行时监控；禁止裸跑 heredoc 重任务。
- 全局锁返回 75（BUSY）或 69（资源不足）时必须等待，不得绕过包装器、抢占 owner、删除 `.pid/.guard` 或改成无锁执行。

## 工程边界

- `pypto` Tensor 与 `pypto_pro` Professional 是同一 wheel 内的并存前端；当前前端分层策略已重新打开，未对齐前不得假定 Pro 替代 classic。
- 公共层只表达 Tensor/Scalar/Shape/控制流/逻辑 Tile；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe 和 CCE 语义必须留在 Ascend target。
- CPU 共享 lowering，开发优先级为 x86 AVX2/AVX-512 → AArch64 SVE256（不开发 NEON 优化后端）→ NVIDIA → AMD。
- NVIDIA 与 AMD 共享 GPU IR/Runtime ABI，再分别下降到 CUDA/NVVM 和 HIP/ROCDL。
- 首个实际模型为固定 revision 的 `Qwen/Qwen3.5-0.8B`，首期只做纯文本 BF16 与 W8A8-linear；不做 W8A16、INT4、FP8、KV/GDR state 量化。
- 视觉编码器不纳入当前 MVP；性能门槛在首批后端完成后再根据实测冻结。
- 当前正式 CANN 许可证仍限制非华为处理器用途。维护方态度积极，但对外发布前必须取得新许可证、双许可证或明确书面例外。

## 资源安全

- 本机共享重任务遵守 `/home/chiro/projects/.resource-locks/README.md`；运行前以 `resource-lock run` 实际取得锁才算获准，`status` 只供观察。
- PyPTO-X 本机 heavy 命令统一经 `scripts/resource/run_local_heavy.sh`：默认启动至少 8 GiB `MemAvailable`、保留 4 GiB 系统余量、最多 6/12 CPU，使用动态 `MemoryHigh/MemoryMax`、`MemorySwapMax=0`、CPU quota/affinity，并持续记录内存、RSS、load 与 PSI；安全停止只作用于本任务进程组。
- RTX 5080 主机 `192.168.101.5` 已恢复且 GPU 由 PyPTO-X 独占；GPU-only probe/执行不申请 `gamepc`，大量远端 CPU/内存阶段才持锁。SSH 默认进入 Windows `cmd`；Linux 命令必须通过 `wsl.exe -e bash -lc`。
- AMD 6750GRE 安装失败并暂缓；临时目标为 GamePC `gfx1036` 核显。Windows OpenCL/Vulkan 可见不等于 HIP 可用；在 WSL 获得 `/dev/kfd`、ROCm/HIP runtime 与真机 kernel 证据前，不得声称 HIP 已运行。
- QEMU 只用于 AArch64/SVE 功能验证，不得用其数字作性能结论。
- 鲲鹏 920B ECS 已完成 native SVE256 功能与汇编验证：SVE=1、VL=32、SVE2=0；它是 2 vCPU KVM guest，尚未形成性能门槛。
