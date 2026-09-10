# PyPTO-X W8 并行执行路线图（资源与并发版）v2

文档编号：`0003`

日期：2026-09-10（Asia/Shanghai）

状态：`REVISED_AFTER_EXTERNAL_REVIEW`

修订记录：v1（提交 `12cd196`）→ v2（本文）。v2 依据一次外部独立评审的 8 点意见修订，**8 点均已由父 agent 逐条核实为事实**，修订点见 §12。

用途：在**已知系统资源**与**全局资源锁协议**下，给出 W8 及后续波次的可执行计划、并发上限、每任务锁占用预算与验收口径。本文自包含，供外部模型/评审者直接检查。

---

## 1. 背景（不看会话也能读）

PyPTO-X 把官方 PyPTO（同仓 `pypto` Tensor 前端 + `pypto_pro` Professional 前端）的可移植语义抽成目标无关 Core IR，并逐步支持：x86 CPU（AVX2/AVX-512）→ AArch64 SVE256 → NVIDIA CUDA → AMD HIP；Ascend CCE/CANN 保持为 target plugin。首个真实模型固定 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期只做纯文本 BF16 与 W8A8-linear。

```text
控制仓        /home/chiro/projects/pypto/pypto_x（docs/configs/scripts + 5 个 upstream submodule）
实现主仓      upstream/pypto @ 34475e0d（只读；实现一律走 worktree）
集成分支      port/pypto-x-integration（当前 HEAD 9aae4649e）
任务 worktree ../worktrees/pypto-x/<task>；证据 ../worktrees/_meta/pypto-x/<task>/
强制规范      AGENTS.md、docs/LOCAL_RESOURCE_POLICY.zh-CN.md、docs/WORKTREE_AGENT_PLAN.zh-CN.md、docs/SMOKE_TEST_SPEC.zh-CN.md
```

---

## 2. 当前基线

### 2.1 已冻结

| 项 | 证据 |
|---|---|
| Core IR / Target ABI / CPU scalar / AVX2 / AVX-512 / SVE256 / GPU common / CUDA C1–C2 | `development_lock.yaml` W1–W4 |
| Qwen M0–M1K（无权重 decoder、binding、CPU/CUDA ingestion） | W6 |
| AMD `gfx1036` 静态 C1–C3；运行态判定架构性不可达（用户决定长期只保留静态证据） | 审计 `0004` |
| 官方 gold 参考（3 prompt、prefill+4 步 greedy、逐层 hidden states；含 fp32/bf16 两变体） | `52b7e3d7c` |
| 权重接入（safetensors reader、320 参数映射、真实 packed layout） | `a63105eef` |
| GDR T=128 验收（五后端 + 920B 原生），`gdr_t128_not_validated` 已关闭 | `dba3c9d2f` |
| W8A8-linear 契约（D1–D12 用户已批准） | `docs/20-planning/0002-*` |
| 性能测量协议（提案态） | `docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` |
| CUDA Toolkit（nvcc 13.3.73 + cuBLAS 13.6；Driver/PTX 回归 PASS） | `_meta/cuda-toolkit-wsl` |
| PTO-ISA CPU_SIM 基线（125/125 PASS）、CANN toolkit + CA-model 最小用例 | 审计 `0006`/`0007` |
| CPU vector runtime liveness + AVX-512 packed cast/transpose/embedding + **GDR decay 修复** | `9aae4649e` |

### 2.2 主线数值状态（实现侧完成，**独立验收进行中**）

```text
ERR-0001：图 v1/v2 的 GDR decay 门缺 exp(A_log)（官方为 -exp(A_log)·softplus），
          FP32 状态指数爆炸（|state|max 0.60→2.8e4→2.36e16）被 gated RMSNorm 掩盖。
修复后（真权重、AVX-512、公开图）：
  prefill en_continuation T=5   argmax 5/5、cosine 0.9999136、max_abs 0.2338（gold dtype 带 0.2352）
  decode（对齐口径）            11751 → 13 → 198 → 760 逐步与 gold 一致
  峰值 RSS / wall               3.905 GiB / 131.6 s（修复前模型值 7.66 GiB / 旧 list 路径 309.6 GiB）
图契约 v3：digest 66dd4077…、4,550 ops（(1,1,4096)）/ 6,728 ops（(1,5,0)）
全量回归                      674 passed, 7 skipped
```

### 2.3 已知边界

```text
- CPU 侧逐 op 派发、零融合；标量 matmul 是纯 Python 三重循环
- 本机 KVM guest 无 cpufreq → 绝对性能门槛永久 UNGATED；GamePC WSL 无锁频手段
- CUDA 无 cuEvent 计时（cuEvent* 在 python/pypto 下 0 命中）→ kernel_seconds 必须为 null
- CANN CAModel：单条 64x64 TADD 95 s / 峰值 7.3 GiB；npusim record 第二次卡死；report 缺 plotly
- IR→PTO / IR→CCE 桥不存在；Ascend adapter 仍是 seam
- 920B ECS：2 vCPU / 2.5 GiB / 34 GB，只能做 SVE 原生功能验收
- 上游默认分支已漂移风险：本仓仍锁 34475e0d（edge lock），stable lock 待 CANN 配套验证
```

---

## 3. 资源与锁（修正后的准确描述）

### 3.1 资源

| 资源 | 规格 | 独占方式 |
|---|---|---|
| 本机 | 12 vCPU（KVM guest，无 cpufreq）、29 GiB 内存（可用 ~23 GiB）、磁盘剩 82 GB | 全局 `local` 锁（排他） |
| GamePC | WSL2：24 线程、WSL 30 GiB（宿主 61.4 GiB）、磁盘 865 GB；RTX 5080 16 GB；nvcc 13.3.73 + cuBLAS 13.6 | `gamepc` 锁（host-heavy）；GPU-only 短探测不申请 |
| 鲲鹏 920B ECS | 2 vCPU、2.5 GiB、34 GB；SVE=1/SVE2=0/VL=32 | 无正式锁 → **约定串行** |
| QEMU | 本机 qemu-aarch64 11.0.3 | 随 `local` 锁 |
| CANN 9.2.0-beta.2 | 本机 `/usr/local/Ascend`（3.9 GB，仅 toolkit） | 随 `local` 锁 |

### 3.2 锁语义（**修正 v1 的表述错误**）

```text
local 锁：全局排他；启动要求 MemAvailable ≥ 8192 MiB（这是"启动门槛"，不是 cgroup 上限）
          系统安全余量 4096 MiB
          默认 MemoryHigh/MemoryMax = 动态，MemoryMax = min(启动时 MemAvailable - 4096 MiB, 20480 MiB)
          CPUQuota = 600%（6/12 CPU）、MemorySwapMax=0、affinity 限定、每 2 s 采样并落 resource-usage 日志
返回码    75 = 锁被占；69 = 准入不足 → 都必须等待重试，禁止绕过/抢占/删 .pid/.guard/无锁执行
```

**因此**：任务必须显式登记自己的 `min_available_mib`、`memory_max_mib`、超时与峰值余量（见 §5 每任务字段），不能笼统写"接近 8 GiB 上限"。

### 3.3 实测耗时/内存（排期依据）

| 活动 | 锁内时间 | 峰值 RSS |
|---|---|---|
| 全量 `python/tests/ut/pypto_x` | ~170–206 s | 321–409 MiB |
| Qwen T=5 整网（真权重，AVX-512，lower+compile+execute） | ~132 s | 3.9 GiB |
| Qwen T=1 decode 单步（含 state 回灌） | ~60–70 s | 3.9 GiB |
| GDR T=128 lowering（单后端） | 0.6–2.0 s | ≤184 MiB |
| GDR T=128 执行（scalar/avx2/avx512/SVE-QEMU） | 129 / 108 / 116 / 216 s | 5.6 GiB / 0.9 / 0.9 / 5.6 GiB |
| GDR T=128 920B 原生 | 39 s（另加传输） | — |
| PTO-ISA CPU_SIM 构建+125 用例 | 134 s + 1.75 s | 1.76 GiB |
| CANN CAModel 单条 64×64 TADD | 95 s | 7.3 GiB |

**外推（标注为估算）**：`ops(T) ≈ 6,728 + 378×(T−5)`（(1,T,0) 口径）→ T=8 ≈ 7,862 ops、T=18 ≈ 11,642 ops；wall 按 ops 近似线性，T=8 约 160–200 s、T=18 约 250–350 s（含绑定与编译开销）。

---

## 4. 并发策略（按评审意见修正）

### 4.1 槽位（修正）

```text
平台总活动 agent 槽位 = 4（含父 agent）
  → 活动 subagent ≤ 3
推荐组合：1 个 local-heavy  +  1 个 GamePC/远端  +  1 个轻任务  +  父 agent
```

理由：全局锁排他；重任务多派只会排队（实测：CANN 安装等锁 636 s / 29 次重试；主线曾被 CANN 探针饿死）。

### 4.2 三段式错峰（每个任务）

```text
① 轻段：读契约、写实现、聚焦单测（不占锁）
② 重段：lower/compile/execute/全量 pytest（持锁，全局串行）
③ 收尾：证据、commit、报告（不占锁）
```

调度规则：

1. 重段尽量短且可预算；能预生成的产物提前在轻段做好。
2. 大内存任务（CAModel 7.3 GiB、整网 3.9 GiB）显式登记 `memory_max_mib`，且两者不在同一时间窗紧邻排。
3. 920B：本地编译、远端只跑 ELF。
4. 验收 agent 的重段独立占一个时间窗，不与实现重段重叠（否则互相等锁）。
5. **父 agent 不自己跑重任务**（只做合并/审查/派发）。
6. 同一任务**禁止**同时提交两个重命令（会自己排自己）。

---

## 5. 波次计划（v2）

任务字段统一为：`资源 / 锁占用预算 / min_available_mib / memory_max_mib / 验收`。

### W8A 主线收口（关键路径，修正顺序）

| ID | 任务 | 目标 | 依赖 | 验收（冻结口径） | 锁预算 |
|---|---|---|---|---|---|
| A0 | `integration-truth-normalization` | 收口当前 runtime 任务（合并 `70af8ed2c`/`d27a2cf33`/`11eceeb4a`，**已完成**）；**归一化配置真值**：删 `agent_tasks.yaml` 重复条目、同步 GDR/CUDA/W7 状态 | 无 | `yaml.safe_load` 通过 + 无重名 + 每条状态与 `development_lock.yaml` 一致 | 0 |
| A1 | `qwen35-weighted-multiprompt` | 在**同一实现轮**内跑完 **T=5/T=8/T=18 三个 prompt 的 prefill + 4 步 decode**（共享一份 correction 后的图与 driver） | A0 | 每 prompt 每步：argmax 与 gold 一致；cosine ≥ 0.9995；max_abs ≤ 0.35；state 有限且不爆炸；产出逐层 hidden 对照（gold 有 25×T×1024） | 3×3 个 prompt×stage ≈ 6–8 个重段，累计 ~25–35 min |
| A2 | `verify-w8a`（独立验收） | 从 **A1 的最终 integration HEAD** 建只读 worktree，独立复跑 S1/S2/S3 + 全量 pytest + liveness A/B + packed 内核抽样 | A1 完成并合并 | 见 §6 验收细则 | ~13–18 min（S1+S2+S3+pytest） |
| A3 | 冻结与文档 | `development_lock` 关项、`0035` 快照、HANDOFF/README/ERRATA 同步 | A2 | 文档提交 + 链接检查 | 0 |

**出口标准（修正）**：三个 prompt 的 prefill 与 decode 均**达到冻结容差**（不是"等于 gold"）；逐层 hidden 与最终 logits 都有对照；独立验收 PASS。

**注意（评审点 2）**：decay 修复是在 `work/qwen35-vector-runtime-packed-liveness` 分支上、由父 agent 书面解除"不改 `qwen35.py`"约束后完成的。该事实必须在 `development_lock.yaml` 与 0035 快照里明确登记为**范围扩展**，不允许保留"任务名与所有权不一致"的模糊状态。

### W8B 硬化与性能（修正）

| ID | 任务 | 目标 | 验收 | 锁预算 |
|---|---|---|---|---|
| B1 | `avx2-packed-kernels` | AVX2 补 packed cast/transpose/embedding（现仍 host_reference+liveness） | 与 AVX-512 同口径逐位单测；focused+全量 pytest 收集数不减少 | ~150 s + 210 s |
| B2 | `sve256-fallback-closure` | native `iota`/`compare` **以及 `broadcast`/`where`**（那 19 次 `to_values` 的来源），逐类清零或明确降级 | QEMU 差分 + 920B 原生；**报告必须按类给出"已 native / 仍 host"清单**，不得承诺整体清零 | local ≤300 s；920B 独立 |
| B3a | `cuda-kernel-timing` | 实现 `cuEventCreate/Record/ElapsedTime` 的 kernel 计时，接进证据 schema；在此之前 CUDA 的 `kernel_seconds` 必须为 `null` | 计时器自证（同一 kernel 重复 R 次离散度）+ 协议字段齐全 | GamePC（host 编译期持 `gamepc`） |
| B3b | `cuda-gemm-baseline` | **仅 GEMM** 与 cuBLAS 比（G3）；前置：G1 正确性、G2 稳定性、且非 dispatch-dominated | 五元组 key + 冻结中位数 + 证据 JSON | GamePC（持 `gamepc`） |
| B3c | `local-perf-l0-l1` | 本机 AVX-512/scalar：L0 microbench + L1（T=1 decode / T=5 prefill） | elementwise/reduction 保持 `T2/T3`、标 `UNGATED`；报 `dispatch_seconds` 差值 | ~3×60 s + 210 s |
| B4 | `perf-freeze` | 依据 B3b/B3c 冻结 G1–G3 键与阈值（G4 本机不做） | 用户批准 + `configs/perf_lock.yaml` | 0 |

### W8C W8A8-linear 实现（按契约附录 B 补全后端 DAG）

| ID | 任务 | 目标 | 依赖 |
|---|---|---|---|
| C1 | `qwen35-w8a8-scheme` | `QuantizedTensorDesc` + 4 个 opcode + scalar golden（R1–R5） | A 完成 |
| C2 | `qwen35-w8a8-binding` | binding schema v2 + packed layout v2 + 覆盖率 manifest（R7–R9） | C1 |
| C3 | `qwen35-w8a8-cpu-avx2` | AVX2 widening 路径（R6） | C2 |
| C4 | `qwen35-w8a8-cpu-avx512` | 接通 VNNI（含 s8 符号补偿）或 `vpdpwssd`（R6） | C3 |
| C5 | `qwen35-w8a8-sve256` | widening fallback；SDOT 另附 HWCAP/反汇编证据（R6） | C4 |
| C6 | `qwen35-w8a8-cuda` / `qwen35-w8a8-amd-static` | 各自能力门禁下的正确性 kernel | C5 |
| C7 | `qwen35-w8a8-layer-ladder` | L2/L3 阶梯与报告（R10–R11） | 任一后端 |
| C8 | `qwen35-w8a8-model-validation` | L4–L6 整网对比与覆盖率报告（R12–R14）；阈值 D5 在 BF16 基线后冻结 | 全部 |

### W8D 其他能力扩展

| ID | 任务 | 目标 | 备注 |
|---|---|---|---|
| D1 | `gdr-fused-wy` | 关闭 `gdr_chunk_is_sequential_reference_not_fused_wy_kernel` | 中大型 |
| D2 | `long-prefill-eval` | T=64/128 静态展开代价实测（估 ~2.9 万/5.2 万 ops），再决定是否 chunk/融合 | 先测量后决策 |

### W8E Ascend / CANN

| ID | 任务 | 目标 | 备注 |
|---|---|---|---|
| E1 | `cann-report-enable` | 装 plotly（证据目录 venv）→ 打通 `npusim report` | 低风险 |
| E2 | `npusim-record-reproducibility` | 定位第二次 record 卡死 | 未解决前不把 CANN 仿真放关键路径 |
| E3 | `pypto-version-alignment` | CANN 自带 `pypto 0.2.1` vs 我们 edge `34475e0d` 差异审计 | stable lock 输入 |
| E4 | `ir-to-pto-bridge` | Core IR → PTO C++ codegen + external buffer→GlobalTensor/TASSIGN | **大**，需独立预算 |
| E5 | `ascend-adapter-hooks` | 实现 `PYPTO_X_ASCEND_HOOKS` 真实 hook | 依赖 E4 |

### W8F AMD / 杂项

| ID | 任务 | 目标 |
|---|---|---|
| F1 | `amd-evidence-register` | 登记第三方 wave32/2CU/Fast-F16 证据；修 `probe_amd_hip_runtime()` 假阴性门禁 |
| F2 | `disk-evidence-policy` | 大件保留/删除策略（见 §8） |

---

## 6. 验收细则（修正，可直接抄进任务书）

```text
通用
  退出码 0；pytest：收集数不减少、无新增非预期 skip、失败为 0（不写"≥N passed"这种软口径）
  证据：validation.json + brief.zh-CN.md + raw/ 原始日志 + runner 资源日志
  独立验收不得复用实现方的 JSON 结论，须自行取数

A1/A2 数值（真权重，逐 prompt）
  prefill：逐行 argmax 与 gold 一致；cosine ≥ 0.9995；max|Δlogits| ≤ 0.35（gold 自身 fp32↔bf16 带宽 0.2352）
  decode ：对齐口径 = prefill 末行 ↔ gold decode_logits[0]，第 k 步 ↔ decode_logits[k+1]；
           每步报「输入 token / argmax / cosine / max_abs」，逐步 argmax 与 gold 一致
  逐层   ：`hidden_states_layers`（25×T×1024）逐层 cosine ≥ 0.999（gold 已提供，不得只看 logits）
  state  ：prefill 结束后的 recurrent state 必须有限且 |state|max 在 O(1) 量级（防再次指数爆炸）
  资源   ：峰值 RSS 与 wall 必须记录；CC 上限按 §3.2 的实际 MemoryMax 计算余量

性能（B3*）
  G1 正确性 → G2 稳定性（3 轮 × R，中位数离散 ≤10%）→ G3 相对比值（仅 GEMM vs cuBLAS）
  elementwise / reduction / 非 GEMM 的 matmul：保持 T2/T3，`UNGATED`
  CUDA：cuEvent 计时未落地前 `kernel_seconds = null`，`gflops_scope = end_to_end`

冻结流水线（每个阶段收口都要走一遍）
  1. 实现 worktree commit → 父 agent cherry-pick 进 integration → 固定 exact HEAD
  2. 独立验收 worktree（源码只读）复跑并出 validation.json
  3. 更新 configs/development_lock.yaml（task/integration commit、验收数字、known_limits 增删）
  4. 写 docs/00-handoffs/00NN-*.zh-CN.md 快照 + 更新 HANDOFF.zh-CN.md 状态行 + ERRATA（如有勘误）
  5. 重跑 `scripts/remote/export_patches.sh` 刷新 `patches/`（机器生成，禁止手工编辑）
  6. `scripts/remote/publish_public_mirror.sh` 发布公开主仓（同样机器生成）
  7. 涉及许可/发布边界的改动同步 `LICENSE` / `NOTICE` / `patches/README.md`
```

---

## 7. 关键路径（修正后的推荐）

```text
A0 配置真值归一化 + 当前 runtime 任务收口（已完成合并，仅剩配置）
 → A1 decay 修复后一次跑齐 T5/T8/T18 的 prefill + 4 步 decode
 → A2 从最终 HEAD 独立验收（只验一次）
 → A3 development_lock / HANDOFF / 0035 冻结
 → B1 / B2 后端硬化
 → B3a → B3b / B3c → B4 性能分层与冻结
 → W8C 完整后端 DAG（W8A8）
 → W8D/D2 长序列评估、W8E Ascend（按用户决策）
```

4 槽位排期示例（父 agent 常驻）：

```text
时段1  A1 重段(持锁)            + 父 agent 合并/文档
时段2  A2 验收重段(持锁)        + B3a 轻段(GamePC 准备)
时段3  A3 冻结(无锁)            + B1 轻段 + E3(无锁)
时段4  B1 重段(持锁)            + B3b(GamePC,持 gamepc) + E1(轻)
时段5  B3c 重段(持锁)           + B2 轻段(本地编译) → 920B 另排
时段6  B4 冻结 + C1 轻段        + E2/F1
```

---

## 8. 磁盘与证据策略

```text
当前 _meta/pypto-x 主要占用：CANN 安装包 2.1 GB、权重资产 1.7 GB、gold 参考 venv 1.1 GB、
                            PTO-ISA 构建 146 MB、各任务证据 5–90 MB；磁盘剩 82 GB
保留（不可删）  权重资产（含 sha256 清单）、gold 参考 npz、全部 validation/brief/raw/脚本
可删（需批准）  .run 安装包、参考 venv、build/ 目录、临时 npz（都须留有 URL/脚本可重建）
禁止            删上游 checkout、worktree、证据目录
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 全局锁串行 | 关键路径变长 | 三段式错峰；重段预算化；父 agent 不占锁 |
| 大内存任务撞 cgroup 上限 | OOM/69 | 每任务显式 `memory_max_mib`（按实际 MemoryMax 公式算余量） |
| 验收 HEAD 被后续修改作废 | 重复验收 | A1 一次跑齐 3 prompt；A2 只验最终 HEAD（评审点 1） |
| 配置真值与 Git 实况漂移 | 计划失真 | A0 归一化 + 每阶段同步（评审点 2） |
| `npusim record` 不可复现 | CANN 线受阻 | E2 先修，未解决不进关键路径 |
| 上游 master 漂移 | stable lock 风险 | 继续锁 `34475e0d`；E3 做版本对齐审计 |
| 权重资产丢失 | 不可复现 | 已有 sha256 清单 + 下载脚本；列入"不可删" |
| CANN 许可证（非华为处理器限制） | 对外发布风险 | 发布前取得新许可证/双许可证/书面例外 |
| KVM/WSL 无锁频 | 性能数字不可作绝对门槛 | 本机 G4 永久 UNGATED；GamePC 强制 warmup + 时钟断言 |

---

## 10. 待决策（用户）

1. **W8C 起点**：是否按 C1→C8 完整后端 DAG 推进（推荐），还是先只做 C1/C2 打契约底座？
2. **W8D/E 深度**：D2 长序列评估、E4 IR→PTO 桥，是否现在启动？
3. **是否确认采用 ≤3 subagent + 父 agent 的 4 槽并发模型**（评审建议）？
4. **是否批准 §8 的可删清单**（安装包 / venv / build 目录）？

---

## 11. 请评审者重点检查

```text
a) A1 一次跑齐 3 prompt 是否确实是更优顺序（是否应把 T=18 拆成独立任务以缩短单次反馈环）？
b) B2 的"逐类清零或明确降级"是否可验收；`broadcast/where` 的 native 化成本是否被低估？
c) B3 分层（a 计时 / b GEMM / c 本机）是否还有遗漏的基线来源（例如 5080 上是否值得装 cuDNN/NCCL）？
d) A2 的容差（cosine ≥0.9995 / max_abs ≤0.35 / 逐层 ≥0.999）是否过松或过紧，是否应改用相对 gold-dtype 带宽的比值？
e) 4 槽并发在"验收也要占槽"的现实下是否仍够用；是否需要把 A2 与 B1 合并成一个 slot 顺序执行？
f) 是否有被忽略的更短路径：例如先用 CUDA 后端跑真权重（VRAM 16 GB 足够放 1.5 GB 参数 + 中间值）来交叉验证数值？
```

---

## 12. v1 → v2 修订清单（对应外部评审 8 点）

| # | 评审意见 | v2 处理 |
|---|---|---|
| 1 | W8A 验收顺序错误（A3 改动会让 A2 验收失效） | A1 一次跑齐 T5/T8/T18，A2 只验最终 HEAD，A3 冻结 |
| 2 | 任务边界与 Git 实况不一致（decay 修复混在 runtime 任务） | A0 收口 + 在 lock/快照中登记为**范围扩展** |
| 3 | 并发上限多算一个（含父 agent 共 4 槽） | §4.1 改为 subagent ≤3 |
| 4 | W8A8 路线缺后端节点 | W8C 按契约附录 B 补 C3–C6（AVX2/AVX-512/SVE256/CUDA/AMD） |
| 5 | 性能 G3 口径不成立（cuBLAS 不能做任意算子基线；cuEvent 缺失） | 拆 B3a/B3b/B3c；非 GEMM 保持 T2/T3 UNGATED；`kernel_seconds=null` |
| 6 | 内存描述事实错误（8 GiB 是启动门槛不是上限） | §3.2 按 `LOCAL_RESOURCE_POLICY` 更正并强制每任务登记内存参数 |
| 7 | 验收条件不冻结 | §6 给出逐项口径、预算与外推；pytest 改退出码口径；要求逐层 hidden/state |
| 8 | B2 范围不足（19 次 fallback 含 broadcast/where） | B2 扩范围并要求按类给出清单，禁止整体清零承诺 |
| + | 配置真值漂移（重复条目、状态滞后） | 新增 A0 配置归一化 |
