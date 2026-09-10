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
集成分支      port/pypto-x-integration（当前 HEAD dca302ef4）
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
| **ERR-0002 修复**（W8A driver cos/sin 布局，图契约零改动）+ near-tie 判据修订 | `5f479d1a1` + `verify/qwen35-t18-divergence-localization` |
| **W8A-C Ascend A2(910B3) 真机验收 PASS**（限定式关闭 `ascend_cann_bisheng_npu_regression_blocked`） | `0e5a51891` + `verify/qwen35-ascend-npu-acceptance` |
| **W8G 可移植性清理**完成 + 独立验收 PASS | `c464927fa` + `verify/qwen35-portability-cleanup` |
| **W8H/W8I A2 vllm-ascend E2E / profiling 基线**（原始事实；独立验收 PASS，6 条非阻断 discrepancy 已登记） | `9951a2fe7+a65a37cc3` / `cc1cc7178..dca302ef4` |

### 2.2 主线数值状态（**已收口，ERR-0002 修复后口径**）

```text
ERR-0001：图 v1/v2 的 GDR decay 门缺 exp(A_log)（官方为 -exp(A_log)·softplus），
          FP32 状态指数爆炸（|state|max 0.60→2.8e4→2.36e16）被 gated RMSNorm 掩盖。修复已验收。
ERR-0002：W8A 测试 driver 的 cos/sin 运行时输入按 position-major 展平（图契约是 head-major
          (batch,heads,steps,rotary_dim)），T≠heads 时位置表错位；此前 chat(T=18)"真实累积分歧"
          （3.78×band、row 11 起、首个不达标层 gold index 8）结论作废。修复 `5f479d1a1`，图契约零改动。
修复后独立验收（真权重、AVX-512、公开图 v3；2026-09-10 用户裁定 + near-tie 修订）：
  en_continuation T=5   argmax 5/5、cosine 0.99997235、max_abs 0.159756（0.68×band 0.235212）
  zh_continuation T=8   argmax 8/8、cosine 0.99993803、max_abs 0.235296（0.85×band 0.277682）
  chat_zh_user   T=18   严格 argmax 17/18；唯一 mismatch row 14 为 near-tie（gold fp32 margin
                        0.115007=0.18813×band ≤ 0.5×band，ours argmax=gold runner-up）
                        → 17/17 有效行 + 1 near-tie，按修订判据 PASS；严格 17/18 保留可见
  decode（对齐口径）    token 链三 prompt 全对：11751→13→198→760→6511 / 271→248068→271→248069→271 /
                        109266→6115→103724→1167→16451；T=1 布局 no-op，但端到端 logits 因 state 继承而变
  state                全部有限；recurrent |state|max en 0.55–13.27 / zh 0.82–13.29 / chat 0.69–14.14
图契约 v3：digest 66dd4077…、4,550 ops（(1,1,4096)）/ 13,748 ops（(1,18,0)）；ops(T)=6,728+540×(T−5)
全量回归                      694 collected / 687 passed / 7 skipped
```

### 2.3 已知边界

```text
- CPU 侧逐 op 派发、零融合；标量 matmul 是纯 Python 三重循环
- 本机 KVM guest 无 cpufreq → 绝对性能门槛永久 UNGATED；GamePC WSL 无锁频手段
- CUDA 无 cuEvent 计时（cuEvent* 在 python/pypto 下 0 命中）→ kernel_seconds 必须为 null
- CANN CAModel：单条 64x64 TADD 95 s / 峰值 7.3 GiB；npusim record 第二次卡死；report 缺 plotly
- IR→PTO / IR→CCE 桥不存在；Ascend adapter 仍是 seam（A2 真机 PASS 仅覆盖注入式最小 f32 add hook）
- A2 真机只解锁限定范围；stable CANN 9.2.0-beta.2 未在带卡环境验证
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

**实测（取代早先外推）**：`ops(T) = 6,728 + 540×(T−5)`（(1,T,0) 口径）→ T=8 = **8,348**、T=18 = **13,748**（T=5 的 6,728 一致）。wall 不随 ops 线性：T=5 prefill 132 s、T=8 约 377 s、T=18 约 932 s（含绑定/编译；单锁内 8 段重活合计约 46 min）。

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
| A1 | `qwen35-weighted-multiprompt` | 在**同一实现轮**内跑完 **T=5/T=8/T=18 三个 prompt 的 prefill + 4 步 decode**（共享一份 correction 后的图与 driver） | A0 | 按 §6 修订后判据（相对 gold dtype band + near-tie 例外）；state 有限且不爆炸；产出逐层 hidden 对照（gold 有 25×T×1024）。**注意：其历史 prefill 数字因 ERR-0002 作废，替换值以独立验收为准** | 3×3 个 prompt×stage ≈ 6–8 个重段，累计 ~25–35 min |
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

A1/A2 数值（真权重，逐 prompt）——判据已改为"相对 gold 自身 dtype 噪声底"（2026-09-10 用户裁定），
2026-09-10 同日追加 near-tie 例外：
  prefill：逐行 argmax 与 gold 一致（硬门槛）
  near-tie 例外（2026-09-10 用户裁定）：若 gold 该行 top-1/top-2 margin ≤ 0.5×band 且
      ours argmax == gold top-2 token，则该行记为 near-tie flip、不判失败，但必须单列计数；
      严格口径结果（如 17/18）必须同时可见。
  band = gold 自身 fp32↔bf16 的 max_abs 与 cosine，随 prompt 变化：
      en(T=5)  0.235212 / 0.999956
      zh(T=8)  0.277682 / 0.999929
      chat(T=18) 0.611310 / 0.999772
  判据：max|Δlogits| ≤ 2×band 且 cosine ≥ band_cosine − 1e-4
      （ERR-0002 修复后：en 0.159756=0.68×band/cos 0.99997235 PASS；
        zh 0.235296=0.85×band/cos 0.99993803 PASS；
        chat 0.416867=0.68×band/cos 0.99989976，严格 argmax 17/18，row14 near-tie → 修订判据 PASS）
  decode ：对齐口径 = prefill 末行 ↔ gold decode_logits[0]，第 k 步 ↔ decode_logits[k+1]；
           每步报「输入 token / argmax / cosine / max_abs」，逐步 argmax 与 gold 一致；
           注意 T=1 布局 no-op，但端到端 decode logits 因继承 prefill state 而变（不得写端到端无变化）
  逐层   ：`hidden_states_layers`（25×T×1024）逐层 cosine ≥ 0.999（gold 已提供，不得只看 logits；
           修复后实测逐层 min：en 0.99993098 / zh 0.99993980 / chat 0.99988217）
  state  ：prefill 结束后的 recurrent state 必须有限且 |state|max 在 O(1) 量级
           （修复后实测 en 0.55–13.27 / zh 0.82–13.29 / chat 0.69–14.14）
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
  6. `git push origin main` 推送公开主仓（origin = chiro2001/pypto-x）；再 `scripts/remote/sync_private_backup.sh`
     同步本地私有备份（archive 分支保留第三方离线副本）
  7. 涉及许可/发布边界的改动同步 `LICENSE` / `NOTICE` / `patches/README.md`
  8. 仅当需要从归档重建公开树时，才用 `scripts/remote/publish_public_mirror.sh`（应急路径）
```

---

## 7. 关键路径（修正后的推荐）

```text
状态（2026-09-11）：A0–A3 已完成；ERR-0002 修复 + near-tie 修订已收口；W8A-C + W8G 完成；
W8H/W8I 基线已产出并通过独立验收；下一阶段起点待用户决策（§10）。

A0 配置真值归一化 + 当前 runtime 任务收口（已完成）
 → A1 decay 修复后一次跑齐 T5/T8/T18 的 prefill + 4 步 decode（已完成；数字经 ERR-0002 更正）
 → A2 从最终 HEAD 独立验收（已完成：verify-qwen35-t18-divergence-localization）
 → A3 0036 勘误/快照/配置/补丁收口（本批）
 → W8A-C（已完成，限定式关闭）；W8H/W8I（验收 PASS）
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

> 状态更新 2026-09-11（收口于 0036 快照）：评审通过（v1→v2，见 §12）；W8A 的 A0–A3 已完成（0035）；
> ERR-0002（driver cos/sin 布局）已修复并重定义 chat 口径；**W8A-C A2 真机验收 PASS（限定式）**；
> **W8G 已完成**；**W8H/W8I vllm-ascend 基线已产出并通过独立验收 PASS**（A2 卡锁协议已落地并经运行验证）。
> 以下是**当前仍待用户决定**的项；新 agent 不要自行开工。

1. **W8B vs W8C 优先级**：先做 W8B 硬化（AVX2 packed / SVE fallback / CUDA cuEvent 与 GEMM 基线 /
   本机 L0–L1），还是先启动 W8C W8A8-linear（契约已冻结）？
2. **W8C 起点**：按 C1→C8 完整后端 DAG 推进（推荐），还是先只做 C1/C2 打契约底座？
3. **W8D/E 深度**：D2 长序列 T=64/128 代价评估、E4 Core IR→PTO/CCE codegen（A2 限定式关闭后的真实缺口），
   是否现在启动？
4. **是否要求 chat 严格 18/18**：若要求，需评估任务分支 `327b17158`（`--debug-f32-residual`，改 `qwen35.py`，
   未合入、未采纳）的收益/代价；当前按 near-tie 修订判据已 PASS。
5. **W8H/W8I 后续范围**：是否补做 ACL graph 口径精度复跑、并发 sweep、`logprobs=-1` 全词表往返？
6. **是否批准 §8 的可删清单**（安装包 / venv / build 目录；本机磁盘约 78 GB 可用）？

已决（2026-09-10，勿再开工）：并发采用 ≤3 subagent + 父 agent 四槽；验收口径 = 相对 gold dtype 带宽 +
near-tie 例外；W8A-C 以"限定式关闭 blocked"记账；A2 NPU 任务走 `/root/a2-npu-lock/` 卡锁。
若本节与 `HANDOFF.zh-CN.md` §0/§11 冲突，以 HANDOFF 为准。

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
