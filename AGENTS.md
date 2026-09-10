# PyPTO-X Agent Instructions

本文件是 **agent 的唯一入口**：状态、必读顺序、快速自检、协作协议与工程/资源边界都在这里。项目概览与文档索引见 [`README.md`](README.md)。

## 语言

- 除非用户明确要求其他语言，始终使用简体中文回复与写文档。

## 接手入口

### 0. 状态行

```text
EXECUTION_W8A_DECAY_FIX_VERIFIED_MULTIPROMPT_IN_PROGRESS_AMD_RUNTIME_STATIC_ONLY
```

```text
实现主仓     upstream/pypto @ 34475e0d（只读）
集成分支     port/pypto-x-integration @ 9aae4649e
工作副本     /home/chiro/projects/pypto/pypto_x（= 公开主仓 chiro2001/pypto-x 的克隆，origin 指向公开仓）
私有备份     ../pypto_x_private_bkp（archive 分支，含第三方离线副本）+ GitHub chiro2001/pypto-x-private
证据目录     ../worktrees/_meta/pypto-x/<task>/（不在仓内；权重亦在仓外）
```

### 1. 必读顺序

1. 本文件（入口与协议）。
2. [`HANDOFF.zh-CN.md`](HANDOFF.zh-CN.md)（滚动接手文档：决策、资源、已验证结果、待确认事项）。
3. [`docs/00-handoffs/ERRATA.zh-CN.md`](docs/00-handoffs/ERRATA.zh-CN.md)（**勘误总表**；现有冻结结论的更正都在这里，必须先读，避免引用作废数字）。
4. [`docs/20-planning/0003-2026-09-10-w8-parallel-roadmap.zh-CN.md`](docs/20-planning/0003-2026-09-10-w8-parallel-roadmap.zh-CN.md)（当前波次计划、并发上限、锁预算、验收口径）。
5. [`docs/WORKTREE_AGENT_PLAN.zh-CN.md`](docs/WORKTREE_AGENT_PLAN.zh-CN.md)、[`docs/PROJECT_LAYOUT.zh-CN.md`](docs/PROJECT_LAYOUT.zh-CN.md)、[`docs/RESOURCE_MATRIX.zh-CN.md`](docs/RESOURCE_MATRIX.zh-CN.md)、[`docs/SMOKE_TEST_SPEC.zh-CN.md`](docs/SMOKE_TEST_SPEC.zh-CN.md)、[`docs/LOCAL_RESOURCE_POLICY.zh-CN.md`](docs/LOCAL_RESOURCE_POLICY.zh-CN.md)。
6. 进入嵌套仓库后，遵守距目标文件最近的 `AGENTS.md`（例如 `upstream/pto-isa/AGENTS.md`）。

### 2. 当前阶段快照

**已冻结（可用）**

```text
Core IR / Target ABI / CPU scalar / AVX2 / AVX-512 / SVE256 / GPU common / CUDA C1–C2
Qwen3.5-0.8B M0–M1K（无权重 decoder、3/320/48 binding、CPU/CUDA external ingestion）
真权重 BF16 整网：AVX-512 上 prefill argmax 5/5、cosine 0.99991、峰值 RSS 3.9 GiB；decode 11751→13→198→760
  → 独立验收 PASS_WITH_BOUNDARIES；图契约 v3（4,550 ops / digest 66dd4077…）
GDR T=128：五后端 + 920B 原生 PASS（gdr_t128_not_validated 已关闭）
AMD gfx1036 静态 C1–C3（运行态判定架构性不可达，用户决定长期只保留静态证据）
基础设施：GamePC CUDA Toolkit（nvcc 13.3.73 + cuBLAS 13.6）、本机 CANN 9.2.0-beta.2 toolkit（cannsim/npusim）、
          PTO-ISA CPU_SIM 基线 125/125、CANN CA-model 最小用例跑通
```

**进行中 / 待办**

```text
W8A-A1  qwen35-weighted-multiprompt：T=5/T=8/T=18 三 prompt 的 prefill + 4 步 decode 覆盖
W8B     AVX2 packed 参数级算子 / SVE fallback 分类清零 / CUDA cuEvent 计时 + GEMM 基线 / 本机 L0-L1 性能
W8C     W8A8-linear 实现（契约已冻结；按后端 DAG C1→C8）
W8D     GDR fused WY、长序列 T=64/128 代价评估
W8E     CANN report（缺 plotly）、npusim record 复现性、IR→PTO 桥、Ascend hooks；910C 租用环境待到位
```

**硬边界（不得夸大）**

```text
- 静态 lowering ≠ 真机执行：AMD/ Ascend 的静态结论不得写成运行 PASS
- CPU 侧仍是逐 op 派发、零融合；本机是 KVM guest 无 cpufreq → 绝对性能门槛永久 UNGATED
- CUDA 无 cuEvent 计时前，kernel_seconds 必须为 null
- CANN CAModel 只适合指令级细看（单条 64×64 TADD 95 s / 7.3 GiB），不能做模型级评估
- 权重只在本机 _meta 资产目录，未进 Git；未经用户批准不得下载/分发
```

### 3. 快速自检

```bash
cd /home/chiro/projects/pypto/pypto_x

git -C upstream/pypto status --short && git -C upstream/pypto rev-parse HEAD
scripts/worktree/status.sh
bash -n scripts/worktree/*.sh scripts/smoke/*.sh scripts/resource/*.sh scripts/remote/*.sh
/home/chiro/projects/.resource-locks/resource-lock status     # 只观察；取得锁必须用 run

python3 -c "import yaml;[yaml.safe_load(open(p)) for p in ['configs/development_lock.yaml','configs/agent_tasks.yaml','configs/upstream_lock.yaml']];print('yaml ok')"

git -C /home/chiro/projects/pypto/worktrees/pypto-x/integration log --oneline -1
```

接手自检**不重跑** smoke、不下载依赖、不加载权重（除非用户当次明确要求）。

## Git 与目录

- `upstream/pypto` 是 PyPTO-X 的实现主仓，当前 edge 快照 `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad`；stable lock 仍待 CANN release 配套验证。
- `upstream/*` 原则上只读；禁止直接在 `master`/`main` 开发，禁止向 gitcode 上游推送。
- 实现必须使用 `scripts/worktree/create.sh` 创建的独立 worktree（默认 `../worktrees/pypto-x/<task>`）；跨仓库任务用各自仓库的 worktree，不要把两个仓混在一个 worktree。
- 根目录是轻量控制仓（docs/configs/scripts/patches + submodule gitlink）。**日常开发的推送目标是公开主仓
  `chiro2001/pypto-x`（origin）**；私有备份是本地 `../pypto_x_private_bkp`（`archive` 分支含 references/ 第三方离线副本）
  与远端 `chiro2001/pypto-x-private`，用 `scripts/remote/sync_private_backup.sh` 同步。
  公开仓历史中**不含**第三方离线副本；把上游源码、权重或证据推上公开仓、或改写公开历史，属发布动作，需用户明确指示。
- 许可与补丁：本仓自有内容为 Apache-2.0（`LICENSE`/`NOTICE`）；`patches/pypto-x/` 是**机器生成**的补丁集，
  改动实现后需重跑 `scripts/remote/export_patches.sh` 再提交，不要手工编辑补丁文件。
- 不清理或重置未知修改。`upstream/PTOAS/.codex/CLAUDE.md` 的 dirty 来自上游 CRLF/`.gitattributes` 不一致，不是人工改动。

## Subagent 协议

- 只有用户明确批准进入实现阶段后才启动 subagent。
- 每个 subagent 一个 task、一个分支、一个 worktree；启动参数必须包含：
  `started_at=<ISO-8601 UTC>`、绝对 worktree、branch、`smoke_once=true`、`wait_timeout_seconds=3600`、`poll=false`、
  `resource_lock_root=/home/chiro/projects/.resource-locks`、`local_heavy_policy=locked`、
  `local_heavy_runner=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh`、
  `local_min_available_mib=8192`、`local_safety_floor_mib=4096`、`local_max_cpus=6`。
- 启动时省略 `model`/`model_name`；每个 subagent 启动后只执行一次无模型冒烟测试。
- 父 agent 派发后做一次长等待（`wait_timeout_seconds=3600`），不周期轮询。
- **并发上限**：平台 4 个 agent 槽位含父 agent → 活动 subagent ≤3（推荐 1 个 local-heavy + 1 个 GamePC/远端 + 1 个轻任务）。
- **阶段验收必须独立**：从待验收 integration HEAD 建 `verify/<phase>` worktree，源码只读，只写独占 `_meta` 目录；验收失败回原实现 worktree 修，不由验收 agent 改 integration。
- 不下载、不加载 LLM 权重（用户已授权的固定 revision 除外）；不上传权重或证据到公网。

## 工程边界

- `pypto` Tensor 与 `pypto_pro` Professional 是同一 wheel 内的并存前端；未对齐前不得假定 Pro 替代 classic。
- 公共层只表达 Tensor/Scalar/Shape/控制流/逻辑 Tile；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe 与 CCE 语义必须留在 Ascend target。
- CPU 共享 lowering，优先级 x86 AVX2/AVX-512 → AArch64 SVE256（不开发 NEON 优化后端）→ NVIDIA → AMD。
- NVIDIA 与 AMD 共享 GPU IR/Runtime ABI，再分别下降到 CUDA/NVVM 与 HIP/ROCDL。
- 首个实际模型固定为 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期只做纯文本 BF16 与 W8A8-linear；不做 W8A16/INT4/FP8/KV-GDR state 量化。
- 视觉编码器不纳入当前 MVP；性能门槛按已冻结协议在实测后逐步冻结（本机 G4 永久 UNGATED）。
- 契约级改动（图结构、binding schema、lowering/artifact 版本）必须显式记录连锁影响并升版本号；旧 artifact fail-closed。
- 勘误纪律：发现已冻结结论有误时，**不覆盖历史快照**，改为追加勘误并登记到 `docs/00-handoffs/ERRATA.zh-CN.md`。

## 资源安全

- 本机共享重任务遵守 `/home/chiro/projects/.resource-locks/README.md`：只有 `resource-lock run` 实际取得锁才算获准，`status` 只供观察。
- heavy 命令统一经 `scripts/resource/run_local_heavy.sh`：启动要求 `MemAvailable ≥ 8192 MiB`、系统保留 4096 MiB、最多 6/12 CPU、动态 `MemoryHigh/MemoryMax`（默认 `min(启动时 MemAvailable−4096, 20480)` MiB）、`MemorySwapMax=0`，并持续记录内存/RSS/CPU/load/PSI。
- 返回 **75（BUSY）/69（准入不足）必须等待重试**；禁止绕过包装器、抢占 owner、删除 `.pid/.guard` 或改无锁执行；禁止用 heredoc 承载重任务。
- 每个重任务必须显式登记自己的 `min_available_mib`、`memory_max_mib`、超时与峰值余量；大内存任务（CAModel 7.3 GiB、整网 3.9 GiB）不要紧邻排。
- RTX 5080（`192.168.101.5`）GPU 由 PyPTO-X 独占：GPU-only 短探测不申请 `gamepc`；host-heavy 编译/数据准备必须持 `gamepc`。SSH 默认进入 Windows `cmd`，Linux 命令经 `wsl.exe -e bash -lc`。
- 鲲鹏 920B ECS：2 vCPU / 2.5 GiB，SVE=1/SVE2=0/VL=32，**约定串行**使用；QEMU 只作功能验证，不得用其数字作性能结论。
- AMD 核显 `gfx1036`：官方支持面不含该型号，WSL2 GPU-PV 下无 `/dev/kfd`；未经真机证据不得声称 HIP 已运行。
- 本机 CANN：`/usr/local/Ascend`（仅 toolkit，无驱动、无 950-ops）；CA-model 运行会吃 7.3 GiB，须在锁内并显式提高 `memory_max_mib`。
- 昇腾 A2（910B3）租用环境：容器 256 vCPU / 2 TB 内存 / 1×910B3（64 GB HBM）/ CANN 9.0.0 / 驱动 25.2.0。
  访问用 `scripts/remote/a2_910b.sh`（**隧道优先**，经 **<relay-ip>**:2222，用 IP 不用域名；平台 `jt_xxx:token` 短期有效、仅作引导，容器重建后用 `--install-tunnel` 重建隧道）。
  该机是租用共享资源 → **约定串行**；出网仅 HTTPS(gitcode/pypi) 可达，无 22 出网、无 TUN/NET_ADMIN（VPN 不可行）。
