# PyPTO-X W8 并行执行路线图（资源与并发版）

文档编号：`0003`

日期：2026-09-10（Asia/Shanghai）

状态：`DRAFT_FOR_EXTERNAL_REVIEW`

用途：在**已知系统资源**与**全局资源锁协议**下，给出 W8 及后续波次的可执行计划、并发上限、每个任务的锁占用预算与验收口径。本文档自包含，供外部模型/评审者直接检查与优化。

---

## 1. 评审者需要的背景（不看会话也能读）

PyPTO-X 的目标是把官方 PyPTO（同仓 `pypto` Tensor 前端 + `pypto_pro` Professional 前端）的可移植语义抽成目标无关 Core IR，并逐步支持：x86 CPU（AVX2/AVX-512）→ AArch64 SVE256 → NVIDIA CUDA → AMD HIP；Ascend CCE/CANN 保持为一个 target plugin。首个真实模型固定为 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期只做纯文本 BF16 与 W8A8-linear。

工作目录：

```text
控制仓        /home/chiro/projects/pypto/pypto_x           （docs/configs/scripts + 5 个 upstream submodule）
实现主仓      /home/chiro/projects/pypto/pypto_x/upstream/pypto @ 34475e0d（只读；所有实现走 worktree）
集成分支      port/pypto-x-integration → ../worktrees/pypto-x/integration
              current HEAD = 37b76b929
任务 worktree ../worktrees/pypto-x/<task>（每 subagent 一个）
证据目录      ../worktrees/_meta/pypto-x/<task>/
```

必须遵守的既有规范：`AGENTS.md`（协议与工程边界）、`docs/LOCAL_RESOURCE_POLICY.zh-CN.md`（重任务锁）、`docs/WORKTREE_AGENT_PLAN.zh-CN.md`（每任务一 worktree、独立验收 agent）、`docs/SMOKE_TEST_SPEC.zh-CN.md`（每 subagent 只跑一次 smoke）。

---

## 2. 当前基线（W7 结束时）

### 2.1 已冻结成果

| 项 | 状态 | 证据/提交 |
|---|---|---|
| Core IR / Target ABI / CPU scalar / AVX2 / AVX-512 / SVE256 / GPU common / CUDA C1–C2 | 完成并冻结 | `configs/development_lock.yaml` W1–W4 |
| Qwen3.5-0.8B M0–M1K（无权重 decoder、binding、CPU/CUDA ingestion） | 完成并冻结 | 同上 W6 |
| AMD `gfx1036` 静态 C1–C3（math/任意轴 reduction/rank-2..4 matmul；Qwen 静态 4532/4532） | 完成；运行态判定架构性不可达 | 审计 `0004` |
| 官方 gold 参考（3 prompt，prefill+4 步 greedy，逐层 hidden states） | 完成 | `52b7e3d7c`，`_meta/qwen35-bf16-reference` |
| 权重接入（safetensors reader + 320 参数映射 + 真实 packed layout） | 完成 | `a63105eef` |
| GDR T=128 验收（五后端 + 920B 原生） | 完成，已关闭 `gdr_t128_not_validated` | `dba3c9d2f` |
| W8A8-linear 契约（D1–D12 已获用户批准） | 冻结 | `docs/20-planning/0002-*` |
| 性能测量协议 + 门槛策略 | 冻结（提案未生效） | `docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` |
| CUDA Toolkit（nvcc 13.3.73 + cuBLAS 13.6，Driver/PTX 回归 PASS） | 完成 | GamePC WSL，`_meta/cuda-toolkit-wsl` |
| PTO-ISA CPU_SIM 基线（125/125 PASS） | 完成 | 审计 `0006` |
| 本机 CANN 9.2.0-beta.2 toolkit（cannsim/npusim 可用） | 完成 | `_meta/cann-toolkit-local-install` |
| CANN CA-model 最小用例（PTO-ISA a5 tadd 跑通） | 完成 | 审计 `0007` |
| CPU vector runtime liveness + AVX-512 packed cast/transpose/embedding | 完成（待合并） | `70af8ed2c` |

### 2.2 关键未决问题（W8A 正在收口）

1. **GDR decay 门公式错误**：图里写 `-A_log * softplus(a+dt_bias)`，官方（`transformers/models/qwen3_5/modeling_qwen3_5.py:619`）是 `-exp(A_log) * softplus(...)`。debug 副本证明修正后 prefill **argmax 5/5、cosine 0.99991、max_abs 0.234**（落在 gold 自身 dtype 误差带 0.2352 内）。修复正在落地。
2. **S3 decode 疑为对比行错位**：已验证 gold `decode_logits[0] ≡ prefill_next_logits`（max_abs=0.0），故正确的对齐是「prefill 末位 ↔ `decode_logits[0]`，第 k 步 ↔ `decode_logits[k+1]`」。按此口径重跑。
3. 真权重整网已能执行：T=5 prefill 峰值 RSS **3.79 GiB**、wall **134 s**（修复前模型值 7.66 GiB / 旧 list 路径 309.6 GiB）。

### 2.3 已知边界（计划必须绕开或显式处理）

```text
- CPU 侧仍是逐 op 派发、零融合；标量 matmul 是纯 Python 三重循环
- 本机是 KVM guest、无 cpufreq → 绝对性能门槛永久 UNGATED（见性能协议）
- GamePC WSL 无时钟锁定手段；空闲 SM 时钟约为峰值 1/7，测性能必须 warmup + 断言
- AMD 运行态不可达（用户已决定长期只保留静态证据）
- CANN CAModel：单条 64x64 TADD 95 s、峰值 7.3 GiB → 只能做指令级细看，不能做模型级
- npusim record 第二次运行卡死（复现性未解决）；report 后端缺 plotly
- IR→PTO / IR→CCE 的桥不存在；Ascend adapter 仍只是 seam（PYPTO_X_ASCEND_HOOKS）
- 920B ECS 只有 2 vCPU / 2.5 GiB / 34 GB，只能做 SVE 原生功能验收
```

---

## 3. 资源清单与实测基线

### 3.1 资源

| 资源 | 规格 | 用途 | 独占方式 |
|---|---|---|---|
| 本机 | 12 vCPU（KVM guest，无 cpufreq）、29 GiB 内存（可用 ~23 GiB）、磁盘剩 82 GB | Core IR/ABI、CPU 后端、编译、QEMU、PTO-ISA CPU_SIM、CANN 仿真、全量 pytest | 全局 `local` 锁（排他） |
| GamePC | Windows + WSL2：24 线程、WSL 30 GiB（宿主 61.4 GiB）、磁盘 865 GB；RTX 5080 16 GB（KMD 616.92 / CUDA UMD 13.4；nvcc 13.3.73 + cuBLAS 13.6 已装） | CUDA 正确性与性能、GamePC 上大量 CPU/内存阶段 | `gamepc` 锁（仅 host-heavy）；GPU-only 短探测不申请 |
| 鲲鹏 920B ECS | 2 vCPU、2.5 GiB、34 GB；SVE=1/SVE2=0/VL=32；按量计费（约 0.278 元/时 + 磁盘/EIP） | SVE256 原生功能/汇编验收 | 无正式锁：**约定串行**（一次只派一个任务） |
| QEMU AArch64 | 本机 qemu-aarch64 11.0.3 | SVE 功能验证（不作性能结论） | 随 `local` 锁 |
| CANN 9.2.0-beta.2 | 本机 `/usr/local/Ascend`（3.9 GB，仅 toolkit，无驱动/无 950-ops） | CA-model/CostModel 仿真、ccec/bisheng 编译 | 随 `local` 锁（CAModel 峰值 7.3 GiB，需 `--memory-max-mib 8192` 以上） |

### 3.2 锁的参数与语义（`scripts/resource/run_local_heavy.sh`）

```text
local 锁：全局排他；启动要求 MemAvailable ≥ 8192 MiB；系统保留 4096 MiB
          默认 cgroup：MemoryHigh/MemoryMax 动态、MemorySwapMax=0、CPUQuota=600%（最多 6/12 CPU）
          每 2 s 采样 MemAvailable / 任务树 RSS / CPU / load / PSI，日志在 _meta/pypto-x/resource-usage/
返回码    75 = 锁被占（等待重试）；69 = 资源不足（等待重试）。禁止绕过/抢占/删除 .pid/.guard
```

### 3.3 实测耗时与内存（用于排期估算，全部来自本轮证据）

| 活动 | 锁内时间 | 峰值 RSS | 备注 |
|---|---|---|---|
| 全量 `python/tests/ut/pypto_x` | ~170–206 s | 321–409 MiB | 668–608 passed |
| Qwen T=5 整网 lower+compile+execute（AVX-512，真权重） | ~134 s | 3.79 GiB | 6,710 ops |
| 同上（合成权重，仅 S1） | ~135 s | 3.82 GiB | — |
| GDR T=128 lowering（单后端） | 0.6–2.0 s | ≤184 MiB | 3,843 plans |
| GDR T=128 执行：scalar / avx2 / avx512 / SVE(QEMU) | 129 / 108 / 116 / 216 s | 5.6 GiB / 880 MiB / 915 MiB / 5.6 GiB | 每后端一次 launch |
| GDR T=128 920B 原生 | 39 s（另加传输） | — | 远端 |
| PTO-ISA CPU_SIM 构建 + 全量 125 用例 | 134 s + 1.75 s | 1.76 GiB | build dir 146 MB |
| CANN CAModel 单条 64×64 TADD | 95 s | 7.3 GiB | 需 >8 GiB cgroup 余量 |
| CANN toolkit 安装（下载+安装） | 下载 23 s + 安装 75 s | 159 MiB | 实测等锁 636 s（29 次重试） |

**结论**：除 CAModel 与真权重整网外，绝大多数任务在锁内是**秒到几分钟**级；真正的排期瓶颈是**全局锁的串行性**与**CAModel/整网的 GiB 级内存**。

---

## 4. 并发策略（本计划的核心约束）

### 4.1 并发上限（建议）

```text
同时运行的 subagent 总数          ≤ 4
其中持 local 锁的重任务           ≤ 1（硬约束：锁本身排他）
其中持 gamepc 锁的重任务          ≤ 1
920B 原生任务                     ≤ 1（约定串行）
其余必须是"轻任务"：写代码、跑 focused 单测、读文档、准备证据、写脚本
```

理由：全局锁是排他的，多派重任务只会让它们在锁上排队（实测 CANN 安装等了 636 s、主线一度被 CANN 探针饿死）。轻任务不占锁，与重任务并行才有意义。

### 4.2 每个任务的时间结构（用于错峰）

建议把一个任务显式拆成三段，只有中段占锁：

```text
① 轻段：读契约、写实现、跑 focused 单测（不占锁）           —— 可与其他任务的重段并行
② 重段：lower/compile/execute/全量 pytest（必须持锁）        —— 全局串行，按到达顺序
③ 收尾：写证据、commit、报告（不占锁）                        —— 可与其他任务的重段并行
```

调度规则：

1. **重段尽量短且可预期**：把能预编译/预生成的产物提前在轻段做好；重段只做"必须独占资源"的事。
2. **CAModel 与真权重整网不要同时排**：它们分别要 7.3 GiB 与 3.8 GiB，虽然锁会串行，但两者都接近 8 GiB cgroup 上限，连续运行时要留出 `MemAvailable ≥ 8 GiB` 的窗口。
3. **920B 任务先本地编译、后远端执行**：远端只有 2 vCPU/2.5 GiB，本地不占锁的编译阶段可先做完。
4. **验收 agent 单独占一个重段**：不要与实现 agent 共用 worktree，也不要与实现的重段叠在同一时间窗（否则互相等锁）。
5. **父 agent 不再自己跑重任务**：父只做合并、审查与派发（本轮父 agent 曾用重段做过两次 metadata 探针，虽短但会挤掉子任务；后续改为派给子任务）。

### 4.3 反模式（明确禁止）

```text
- 同一任务同时提交两个重命令（会自己排自己；实测 CANN 探针出现过）
- 多个重任务"同时派发"期待并行（实际在锁上排队，只会拉长关键路径）
- 父 agent 长占锁做自己的探针
- 在锁内跑与本任务验收无关的命令（如顺手全量 pytest）
- 绕过 runner、裸跑 heredoc 重任务
```

---

## 5. 波次计划

估计口径：`锁占用` 指该任务在 `local` 锁内的累计时间（含重试等待之外的纯执行时间）。

### W8A 主线收口（关键路径）

| ID | 任务 | 目标 | 资源 | 依赖 | 验收 | 锁占用估计 |
|---|---|---|---|---|---|---|
| A1 | `qwen35-gdr-decay-fix`（进行中） | 落地 `-exp(A_log)·softplus`；grep 全部 decay 点；修测试期望；记录 digest 连锁 | local | 无 | S2：argmax 5/5、cosine≥0.9999、max_abs≤0.24；S3 按正确对齐逐步一致；全量 pytest ≥668 passed | 3×~150 s + 210 s |
| A2 | `verify-w8a`（独立验收 agent） | 从 A1 的 integration HEAD 建只读验收 worktree，独立复跑 S1/S2/S3 + 全量 pytest，核对 digest 变更与 M1J 证据作废标注 | local | A1 合并 | 验收报告 + `validation.json`；发现的问题回 A1 修 | ~2×150 s + 210 s |
| A3 | `qwen35-weighted-multiprompt` | zh + chat 两个 prompt 的 prefill+decode 对齐（3/3 覆盖） | local | A2 | 两 prompt argmax/cosine 达标 | 2×~150 s |
| A4 | 冻结与文档 | 0035 交接快照、`development_lock.yaml` 关项、HANDOFF/README 更新 | 无锁 | A2/A3 | 文档提交 | 0 |

**出口标准**：`BF16 真权重前向 = 官方 gold（3/3 prompt，prefill+decode）`，且证据可由第三方复跑。

### W8B 硬化与补齐（无需新授权）

| ID | 任务 | 目标 | 资源 | 依赖 | 验收 | 锁占用估计 |
|---|---|---|---|---|---|---|
| B1 | `avx2-packed-kernels` | AVX2 补 packed cast/transpose/embedding（现仍 host_reference+liveness），与 AVX-512 同口径 | local | A1 合并 | 单测逐位一致 + focused/full pytest 不减少 | ~150 s + 210 s |
| B2 | `sve256-fallback-closure` | native `iota`/`compare`，消除 19 处 `to_values` host fallback；版本号+fail-closed | local（QEMU）+ 920B | A1 合并 | QEMU 差分 + 920B 原生执行与反汇编（含 whilelo/ld1w/p 寄存器、NEON=0） | local ≤300 s；920B 独立 |
| B3 | `cuda-perf-baseline`（L0） | 用新装 nvcc/cuBLAS 建 G3 相对门槛：GEMM/elementwise/reduce 对比 cuBLAS 与本实现 | GamePC（GPU-only，短探测不持锁） | 无 | 按性能协议出 `dispatch_seconds` 差值、CV/p95 门槛、证据 JSON | 0（远端） |
| B4 | `local-perf-baseline`（L0/L1） | 本机 AVX-512/scalar：算子 microbench + T=1 decode/T=5 prefill 模型级计时 | local | A1 合并 | 协议要求的统计与 INVALID_CLOCK/UNGATED 标注 | 3×~60 s + 210 s |
| B5 | `perf-freeze` | 依据 B3/B4 冻结 G1–G3 键与阈值（G4 本机不做），写入 `configs/perf_lock.yaml` | 无锁 | B3/B4 | 用户批准 + 配置提交 | 0 |

### W8C 模型能力扩展（需要用户决策）

| ID | 任务 | 目标 | 依赖 | 备注 |
|---|---|---|---|---|
| C1 | `w8a8-l0-core` | `QuantizedTensorDesc` + 4 个 opcode + scalar golden + Q1–Q6 边界（R1–R5） | A 完成 | 契约 §9 R1–R5 |
| C2 | `w8a8-l1-binding` | binding schema v2 + packed layout v2 + scale region（R7–R9） | C1 | 旧版 fail-closed |
| C3 | `w8a8-l2-single-layer` | 单层精度阶梯（cosine≥0.9995 / rel-L2≤1e-2） | C2 | 用已授权权重算 scale（D2） |
| C4 | `w8a8-l3-layer-ladder` | 1→6→24 层替换，报逐层误差 | C3 | — |
| C5 | `w8a8-l4-model` | 整网 logits 阈值（D5 在 BF16 基线后冻结） | C4 | 阈值需用户批准 |
| C6 | `gdr-fused-wy` | 关闭 `gdr_chunk_is_sequential_reference_not_fused_wy_kernel` | A 完成 | 中大型 |
| C7 | `long-prefill-eval` | T=64/128 静态展开代价实测（估 5.2 万 ops），决定是否 chunk/融合 | A 完成 | 先测量后决策 |

### W8D Ascend/CANN（需要用户定深度）

| ID | 任务 | 目标 | 资源 | 备注 |
|---|---|---|---|---|
| D1 | `cann-report-enable` | 装 plotly（证据目录 venv）→ 打通 `npusim report` 泳道/流水产物 | local（轻） | 低风险、低价值密度 |
| D2 | `npusim-record-reproducibility` | 定位"第二次 record 卡在 soc_ready"的根因 | local（重） | 现在不能把它当可自动化入口 |
| D3 | `pypto-version-alignment` | CANN 自带 `pypto 0.2.1` vs 我们 edge 快照 `34475e0d` 的差异审计（stable lock 输入） | 无锁 + CANN | 版本策略 pending 项 |
| D4 | `ir-to-pto-bridge` | Core IR → PTO C++ kernel codegen + external buffer→GlobalTensor/TASSIGN 绑定翻译 | local + CANN | **大**：让我们的图跑上 CA-model 的必经之路 |
| D5 | `ascend-adapter-hooks` | 实现 `PYPTO_X_ASCEND_HOOKS` 的真实 hook（编译器/运行时） | 依赖 D4 | 现在只是 seam |

### W8E AMD 与杂项

| ID | 任务 | 目标 | 备注 |
|---|---|---|---|
| E1 | `amd-evidence-register` | 登记第三方 wave32/2CU/Fast-F16 证据；修 `probe_amd_hip_runtime()` 对 ROCm-on-WSL 形态的假阴性门禁 | 低风险 |
| E2 | `disk-evidence-policy` | 证据目录保留策略：大件（venv/build/.run/权重）只留哈希与重生成脚本 | 见 §7 |

---

## 6. 关键路径与排期（建议的 4 并发节奏）

```text
时段 1   A1（唯一重任务，持锁）        + B3（GamePC GPU-only，不占 local 锁）
时段 2   A2 验收（持锁）               + B1 轻段（写代码/单测，不占锁）
时段 3   A3（持锁）                    + B1 重段（持锁，串在 A3 后）+ D3（无锁）
时段 4   A4 冻结（无锁）+ B4 重段      + B2 轻段（本地编译不占锁 / 920B 另排）
时段 5   B5 冻结                        + C1 轻段（W8A8 若获批准）+ D1
```

串行点（不可并行）：`local` 锁上的每一个重段、920B 原生执行、CAModel 运行、真权重整网运行、独立验收 agent 的重段。

---

## 7. 磁盘与证据策略

当前 `_meta/pypto-x` 主要占用：CANN 安装包 **2.1 GB**、权重资产 **1.7 GB**、gold 参考 venv **1.1 GB**、PTO-ISA 构建 **146 MB**、各任务证据 5–90 MB。磁盘剩 82 GB，短期无压力，但需要规则：

```text
保留（不可删）    权重资产（含 sha256 清单）、gold 参考 npz、各任务 validation.json/brief/raw 日志与脚本
可删（需批准）    安装包 .run（2.1 GB，有 URL+sha256 可重下）、参考 venv（1.1 GB，有 setup_venv.sh 可重建）、
                  build/ 构建目录（可重编）、临时 npz 中间产物
禁止              删除任何上游 checkout、任何 worktree、任何验证证据目录
```

---

## 8. 验收与冻结流程（每个任务一致）

```text
1. 任务在独立 worktree/branch 完成 → commit（task commit）
2. 父 agent cherry-pick 进 integration（integration commit）
3. 阶段级任务另派独立验收 agent：从该 integration HEAD 建 verify/<phase> worktree，源码只读，
   只把日志/报告写进独占 _meta 目录
4. 验收失败 → 回原实现 worktree 修，不由验收 agent 直接改 integration
5. 通过后更新 configs/development_lock.yaml（任务、commit、验收数字、known_limits 增删）
6. 阶段结束写 docs/00-handoffs/00NN-*.zh-CN.md 快照，并更新 HANDOFF.zh-CN.md 状态行
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 全局锁串行导致关键路径拉长 | 交付变慢 | §4.2 的三段式错峰；重段预算化（每任务明确锁内秒数） |
| CAModel/整网内存接近 8 GiB cgroup 上限 | OOM 或频繁 69 | 这两类任务显式 `--memory-max-mib 8192+`，且不与另一个大内存任务紧邻排 |
| 920B 只有 2 vCPU，编译放远端会很慢 | 验收超时 | 本地交叉编译、远端只跑 ELF |
| `npusim record` 不可复现 | CANN 线自动化受阻 | 先做 D2，未解决前不把 CANN 仿真放进关键路径 |
| 修复 decay 改变 graph digest | M1J/M1I 证据对新图作废，binding/layout digest 全变 | A1 显式记录连锁；A2 复核；文档加勘误 |
| KVM guest 无 cpufreq / WSL 无锁频 | 性能数字不可作绝对门槛 | 按协议：本机 G4 永久 UNGATED；GamePC 强制 warmup+时钟断言 |
| 并发的轻任务过多分散评审注意力 | 质量下降 | 并发总数 ≤4；每个任务验收后立即冻结，不长期挂起 |

---

## 10. 待决策（用户）

1. **W8C 先做哪个**：W8A8 实现（C1–C5，工作量大但直接对应"首期 W8A8-linear"目标）／长序列 T=64–128（C7）／GDR fused WY（C6）？
2. **W8D 走多深**：只做 D1+D2（打通 report、查复现性），还是启动 **D4 IR→PTO 桥**（让 Ascend 线真正跑起来，但属新波次，需要独立预算）？
3. **是否接受 §4.1 的并发上限（≤4，且 local 重任务同时只 1 个）** 与 §4.2 的"父 agent 不自己跑重任务"？
4. **是否批准 §7 的可删清单**（安装包 2.1 GB / 参考 venv 1.1 GB / build 目录）？

---

## 11. 请外部评审者重点检查的点

```text
a) 并发上限是否合理：在"全局排他锁 + 单机 12 vCPU/23 GiB"的前提下，≤4 是否过于保守或过于激进？
b) 关键路径排序：A→B→C/D 的顺序有没有更优解（例如把 B3/B4 提前以尽早拿到性能数字）？
c) 三段式（轻-重-轻）错峰是否足够；是否建议引入"锁内任务批处理"（把多个小重段合并成一次持锁）？
d) 验收 agent 的重段是否必要每次都跑全量 pytest，还是可以按风险分级（focused + 抽样全量）？
e) 资源预算是否遗漏：CAModel 7.3 GiB、整网 3.8 GiB、920B 2 vCPU、GamePC WSL 30 GiB 的限制是否被正确转成排期约束？
f) 风险清单是否漏项（例如上游 master 漂移、CANN 许可证、权重资产丢失后的可重建性）？
```
