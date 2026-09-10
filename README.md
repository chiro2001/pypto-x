# PyPTO-X：PyPTO 跨架构后端

PyPTO-X（PyPTO Cross-Architecture）在保留官方 PyPTO Tensor / Professional 双前端定位的前提下，抽取目标无关 Core IR 与 Target/Compiler/Runtime ABI，并逐步支持 x86_64 CPU（AVX2/AVX-512）、鲲鹏 AArch64（SVE256）、NVIDIA GPU（CUDA/NVVM）与 AMD GPU（HIP/ROCDSL 方向）。Ascend CCE/CANN 保留为一个 target plugin。

**Agent 的接手入口在 [`AGENTS.md`](AGENTS.md)** —— 状态行、必读顺序、快速自检和当前阶段都在那里；本文件只做项目概览与文档索引。

## 当前状态（2026-09-10）

状态行：`EXECUTION_W8A_DECAY_FIX_VERIFIED_MULTIPROMPT_IN_PROGRESS_AMD_RUNTIME_STATIC_ONLY`

```text
实现主仓        upstream/pypto @ 34475e0d（只读快照）
集成分支        port/pypto-x-integration @ 9aae4649e
控制仓远端      https://github.com/chiro2001/pypto-x（私有归档；含第三方离线副本）
公开镜像        https://github.com/chiro2001/pypto-x-public（公开；历史中不含第三方 PDF/HTML）
```

**首个真实模型已端到端跑通并对齐官方实现**：`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` 纯文本 BF16，真实权重，AVX-512 后端：

```text
prefill (en, T=5)   argmax 5/5 一致、cosine 0.9999136、max|Δlogits| 0.2338（gold 自身 fp32↔bf16 误差带 0.2352）
decode (4 步)       11751 → 13 → 198 → 760 逐步与官方 gold 一致
资源               峰值 RSS 3.9 GiB、单次前向约 132 s（6 CPU cgroup 内）
独立验收           PASS_WITH_BOUNDARIES（§6 口径见 W8 路线图）
```

对齐过程中发现并修复了一个**冻结契约级 bug**（ERR-0001）：GDR decay 门缺 `exp(A_log)`，导致 recurrent state 指数爆炸、整网 logits 完全错位。详见 [`docs/00-handoffs/ERRATA.zh-CN.md`](docs/00-handoffs/ERRATA.zh-CN.md)。

## 能力矩阵

| 目标 | 状态 | 证据入口 |
|---|---|---|
| CPU scalar / AVX2 / AVX-512 | 正确性冻结；AVX-512 已支持真权重整网（liveness + packed cast/transpose/embedding） | `configs/development_lock.yaml` W3/W8 |
| AVX2 packed 参数级算子 | 未实现（仍 host_reference + liveness） | 路线图 W8B/B1 |
| AArch64 SVE256 | 功能与汇编冻结（QEMU + 鲲鹏 920B 原生）；`iota/compare/broadcast/where` 仍有 host-reference fallback | W4/W6 + 路线图 B2 |
| NVIDIA CUDA | C1–C2 正确性冻结（Driver API + PTX JIT）；Toolkit（nvcc 13.3 + cuBLAS）已装，性能分层待做 | W4 + 路线图 B3a/B3b |
| AMD `gfx1036` | 静态 C1–C3（math/reduction/matmul，Qwen 静态 4,532/4,532，v3 为 4,550）；运行态判定**架构性不可达**，按用户决定长期只保留静态证据 | 审计 0004 |
| Ascend CCE/CANN | 仅有 adapter seam（不 import CANN，靠 `PYPTO_X_ASCEND_HOOKS` 注入）；真机回归仍 blocked；本机 CANN 9.2.0-beta.2 toolkit + CA-model 最小用例已验证 | 审计 0006/0007、路线图 W8E |
| W8A8-linear | 契约已冻结（D1–D12 用户批准），**实现未开始** | [`docs/20-planning/0002-*`](docs/20-planning/0002-2026-09-10-qwen35-w8a8-linear-contract.zh-CN.md) |

## 文档索引

**状态与交接**

- [`AGENTS.md`](AGENTS.md)：**agent 入口**（状态、必读顺序、自检、协议、边界）。
- [`HANDOFF.zh-CN.md`](HANDOFF.zh-CN.md)：滚动接手文档（决策、资源、已验证结果、待确认事项）。
- [`docs/00-handoffs/`](docs/00-handoffs/README.md)：阶段快照索引 + [`ERRATA`](docs/00-handoffs/ERRATA.zh-CN.md)（勘误总表）。
- [`configs/development_lock.yaml`](configs/development_lock.yaml)：integration 冻结点、task commit、known_limits。

**规划与契约**

- [Qwen3.5-0.8B BF16/W8A8 MVP](docs/20-planning/0001-2026-09-07-qwen35-08b-bf16-w8a8-mvp.zh-CN.md)与[首期 W8A8-linear 契约](docs/20-planning/0002-2026-09-10-qwen35-w8a8-linear-contract.zh-CN.md)。
- [W8 并行执行路线图（资源与并发）](docs/20-planning/0003-2026-09-10-w8-parallel-roadmap.zh-CN.md)：波次、锁占用预算、并发上限、验收冻结流程。
- W1–W3 契约：[Core/Target ABI](docs/10-architecture/0001-2026-09-07-w1-core-target-contract.zh-CN.md)、[bridge/scalar/Ascend](docs/10-architecture/0002-2026-09-07-w2-bridge-scalar-ascend-contract.zh-CN.md)、[portable bootstrap](docs/10-architecture/0003-2026-09-07-portable-bootstrap-contract.zh-CN.md)、[x86 vector](docs/10-architecture/0004-2026-09-07-w3-x86-vector-contract.zh-CN.md)。

**规范与流程**

- [Worktree/subagent 计划](docs/WORKTREE_AGENT_PLAN.zh-CN.md)、[资源矩阵](docs/RESOURCE_MATRIX.zh-CN.md)、[本机重任务锁策略](docs/LOCAL_RESOURCE_POLICY.zh-CN.md)、[无模型冒烟规范](docs/SMOKE_TEST_SPEC.zh-CN.md)、[性能测量协议](docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md)（提案态）。

**审计与研究**

- [生态审计 0001](research/audits/2026/0001-2026-09-06-pypto-ecosystem.zh-CN.md)、[Tensor/Pro 关系 0002](research/audits/2026/0002-2026-09-06-pypto-pypto-pro-relationship.zh-CN.md)、[算子可移植率 0003](research/audits/2026/0003-2026-09-07-operator-portability-ratio.zh-CN.md)、[AMD 运行态可行性 0004](research/audits/2026/0004-2026-09-10-amd-gfx1036-runtime-feasibility.zh-CN.md)、[CANN 资源评估 0005](research/audits/2026/0005-2026-09-10-cann-resource-evaluation.zh-CN.md)、[PTO-ISA CPU_SIM 基线 0006](research/audits/2026/0006-2026-09-10-pto-isa-cpu-sim-baseline.zh-CN.md)、[CANN CA-model 探针 0007](research/audits/2026/0007-2026-09-10-cann-camodel-minimal-probe.zh-CN.md)。
- [早期移植研究](research/PYPTO_PORTING_RESEARCH.zh-CN.md)、[源码清单](research/SOURCE_MANIFEST.md)、[离线参考资料](references/README.md)。

## 目录

```text
upstream/pypto            官方 PyPTO（Tensor pypto + Professional pypto_pro），固定 commit 的 submodule
upstream/pypto-gym        官方算子/模型接入（含 Qwen3.5-9B 案例）
upstream/pto-isa          PTO Tile ISA、设备实现、CPU_SIM 与 cost model
upstream/pypto-community  无共同祖先的 community implementation
upstream/PTOAS            PTO assembler/optimizer
configs/                  任务 DAG、版本锁、开发集成锁、模型目标、性能协议提案
docs/                     架构 RFC、规划、规范、阶段快照与勘误
research/                 调研报告与审计
scripts/                  worktree、smoke、资源锁包装器、远端接入
references/               外部公开资料离线副本
../worktrees/             linked worktree 与全部证据目录（_meta），**不在本仓**
```

## 公开镜像与发布

```text
私有归档  chiro2001/pypto-x         完整历史，含 references/ 的第三方讲稿离线副本
公开镜像  chiro2001/pypto-x-public  由脚本从私有仓生成：历史中剔除 references/*.pdf|*.html|*.txt，
                                    并在 references/README.md 标注"公开镜像不再分发"
生成脚本  scripts/remote/publish_public_mirror.sh [--repo <owner/name>] [--dry-run]
```

发布纪律：**推送到公开仓属于对外发布动作，需用户明确指示**；脚本默认只做 dry-run 之外的 force-push 到公开仓，绝不改私有仓历史。

## 本地路径约定

文档与配置里出现的 `/home/chiro/...` 是**维护者本机布局**（控制仓、`../worktrees/`、跨项目资源锁），不是可移植假设：

```text
脚本            scripts/worktree/*、scripts/smoke/* 由 BASH_SOURCE 推导项目根；不写死路径
锁包装器        scripts/resource/run_local_heavy.sh 支持 PYPTO_X_LOCK_TOOL / PYPTO_X_LOCK_ROOT 覆盖，
                否则按"项目上两级/.resource-locks"推导，最后回退到约定路径
驱动/工具       新增脚本应使用相对仓库或环境变量（如 --asset-dir、PYPTO_X_ASSET_DIR）而不是绝对路径
配置            configs/*.yaml 中的 lock_root / resource_lock_root 记录的是跨项目协议的约定路径，可用环境变量覆盖
证据            一律放在 ../worktrees/_meta/...（仓外），不引用本机绝对路径以外的机器
```

## 许可证与发布边界

本地快照的 CANN Open Software License 2.0 仍限制在华为 AI 处理器/软件场景；本仓只包含文档、配置与脚本（上游源码以 submodule 引用形式存在，权重与证据在仓外）。**对外发布非华为处理器衍生后端前，必须取得新许可证、双许可证或明确书面例外**；本文件不构成法律意见。
