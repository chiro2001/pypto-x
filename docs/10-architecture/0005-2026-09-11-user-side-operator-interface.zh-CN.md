# 用户侧算子接口设计方案（trunk + knobs）

文档编号：`0005`

日期：2026-09-11（Asia/Shanghai）

状态：`DRAFT_FOR_EXTERNAL_REVIEW`

用途：本文把「一套代码适配多平台」与「每平台各自最优」这对矛盾，收敛成一套**可冻结、可验收、可扩展**的用户侧算子接口设计。**本文自包含**——读者无需本项目的会话上下文即可评审。

---

## 0. 给外部评审者的说明（先读这段）

- 本文的**第 6 节是不可改动的约束**（对应本项目已冻结的验收体系）；**第 3–5 节是设计方案**，欢迎重构；**第 8 节是需要裁决的决策点**，欢迎给出与"建议"相反的方案与理由；**第 10 节列出了我们特别希望被优化的地方**。
- 请把"性能"与"精度承诺"分开对待：本项目对外的位级/误差主张只对**可移植主干路径**成立；任何性能路径的引入都会改变可复现性的**来源**（从"数学必然"变成"版本钉死"）。
- 术语约定：**主干（trunk）** = 语义正确、任何平台都能跑的通用实现；**旋钮（knob）** = 只改变"怎么算"、不改变"算的是什么"的声明式选项；**覆盖（overlay）** = 平台给出的推荐旋钮值。

---

## 1. 背景（不看会话也能读）

PyPTO-X 把官方 PyPTO 的可移植语义抽成目标无关 Core IR，并为每个目标做 lowering/runtime，目标是**同一份模型代码适配多个平台**：

```text
x86_64 CPU: scalar reference / AVX2 / AVX-512(含 VNNI、BF16)
aarch64 CPU: SVE256（鲲鹏 920B，VL=32，有 svebf16/svei8mm）
NVIDIA GPU: GPU 公共层 → CUDA
AMD GPU: GPU 公共层 → HIP/ROCDL（当前运行态 BLOCKED_DEVICE，仅静态证据）
Ascend: 保持为一个 target plugin（CCE/CANN）
```

首个真实模型是 `Qwen/Qwen3.5-0.8B`（BF16 与 W8A8-linear 两种精度）。

**当前状态（2026-09-11）**：

- BF16 整网在真实权重上已达 en/zh/chat 三 prompt 的验收判据（chat 严格 argmax 17/18 + 1 near-tie）；
- W8A8 四条后端内核面已通（AVX2 widening / AVX-512 VNNI / SVE256 sunpklo+mla / CUDA dp4a）；
- 性能分解已完成：AVX-512 上 **T=5 prefill 的 39.9%、T=1 decode 的 49.3% 时间是"每 op 的运行时派发"**；op 时间内部 **91% 是 layout 类算子**（reshape/slice/transpose/concat/split，全部走 host_reference）；
- 已决定：**GEMM 一律使用各平台最优的 vendor 算子**（oneDNN / AOCL-LPGEMM / cuBLAS / hipBLAS / ACLNN 等），我们自己的内核保留为**可移植主干与契约基准**；
- 本机性能**永久 `UNGATED`**（12 vCPU KVM guest，无 cpufreq，窗口非静默）：只能报中位数与离散度，不得作为可比较的性能结论。

## 2. 设计目标与非目标

**目标**

1. 一套模型/算子代码在**所有平台都能跑通且语义一致**；
2. 在每个平台都能**逼近该平台的最优**（允许借 vendor 库、允许平台特化）；
3. 用户能**看得见、控得住、复现得了**：知道跑的是哪条路径、精度承诺是什么、性能花在哪里；
4. 旋钮必须**可枚举、可校验、可报告、有证据**；不允许存在"幻觉旋钮"。

**非目标**

- 不追求"一个实现打遍所有平台且最快"；
- 不承诺 vendor 路径与主干路径位级一致；
- 不把精度语义（量化方案、累加精度、舍入/饱和）开放为旋钮。

## 3. 现状接口面（真实清单，供评审判断改造成本）

### 3.1 面向用户的入口

目前**没有**面向用户的统一入口。现有的是：

- `pypto.framework`：torch 面向的桥接层（`register_torch_ops()` 注册 `torch.ops.pypto_x.*`；`linear` 失败开放、`linear_strict` 失败关闭；`pack_weight`/`linear_prepacked` 预打包路径）。**它只覆盖 linear，不覆盖模型级使用**；
- `pypto.portable`：Qwen3.5 的权重映射/布局/量化方案/覆盖率计算等模型管道；
- `pypto.compiler.targets.*` + `pypto.backends.*`：per-target 的编译与运行；
- 其余面向用户的操作目前都通过**内部 driver 脚本**（`tools/`、`scripts/`）完成。

### 3.2 散落的 35 个环境变量（全部实测枚举）

```text
构建/产物目录类：PYPTO_X_AVX2_BUILD_DIR  PYPTO_X_AVX2_ARTIFACT_DIR  PYPTO_X_AVX2_PROBE_DIR
                  PYPTO_X_AVX512_BUILD_DIR PYPTO_X_AVX512_ARTIFACT_DIR PYPTO_X_AVX512_PROBE_DIR
                  PYPTO_X_SVE256_BUILD_DIR PYPTO_X_SVE256_ARTIFACT_DIR
                  PYPTO_X_ASCEND_BUILD_DIR PYPTO_X_ASCEND_ARTIFACT_FORMAT
工具链/路径类：    PYPTO_X_AARCH64_SYSROOT  PYPTO_X_CANN_INSTALL_PATH  PYPTO_X_CANN_META_ROOT
                  PYPTO_X_PTO_ISA_ROOT  PYPTO_X_PYPTO_ROOT  PYPTO_X_HWHIAIUSER_HOME
                  PYPTO_X_TORCH_CACHE_DIR
性能相关类：      PYPTO_X_AVX512_GEMM_THREADS  PYPTO_X_AVX512_WEIGHT_CACHE  PYPTO_X_VLLM_GEMM_THREADS
运行/设备类：      PYPTO_X_ASCEND_DEVICE_ID  PYPTO_X_ASCEND_HOOKS  PYPTO_X_SVE_QEMU_CPU
                  PYPTO_X_SVE_TEST_MODE  PYPTO_X_PORTABLE_ONLY
模型资产类：      PYPTO_X_QWEN35_ASSET_DIR  PYPTO_X_QWEN35_ASSETS_DIR  PYPTO_X_QWEN35_WEIGHTED_WORKTREE
vLLM 集成类：      PYPTO_X_VLLM_ENABLE  PYPTO_X_VLLM_GATE_JSON  PYPTO_X_VLLM_LOG
                  PYPTO_X_TORCH_FALLBACK_WARNINGS
开发/资源类：      PYPTO_X_LOCAL_HEAVY_RUNNER  PYPTO_X_RESOURCE_LOCK_ROOT  PYPTO_X_RESOURCE_USAGE_DIR
```

**问题**：没有默认值说明、没有合法域、没有效果证据、没有优先级规则、没有弃用策略；用户无法"发现"它们，也无法判断改动是否安全。

### 3.3 已有的、必须沿用的不变式

1. **声明 = 执行**：artifact 元数据里声明的原生内核，必须就是运行时实际执行的内核（同一个谓词函数同时被元数据与运行时调用），因此不可能"声明了 A 却跑了 B"；
2. **fail-closed**：元数据/线格式/版本被篡改或过期（lowering v8 / runner v6 / W8A8 wire v6 / kernel v1 之类版本链）时必须拒绝执行，而不是回退到未声明路径；
3. **结构化错误**：已有 `PyPTOXBridgeError` / `PyPTOXBuildError` / `PyPTOXUnsupportedError` 等分类；
4. **版本指纹**：每次执行都能溯源到 lowering/runner/wire/kernel 版本；
5. **UNGATED 性能**：在不可信的测量环境里，性能数字不得被写成可比较结论。

## 4. 设计：分层与对象模型

```text
┌─ L0 语义（Core IR 冻结，不可旋钮）
│    C = A·B 的数学定义、dtype 合约、累加精度、舍入/饱和、边界语义
├─ L1 可移植主干（trunk，必存在）
│    我们的 vector/scalar reference：任何平台都能跑通；位级一致的唯一来源；无 vendor 库时的 fallback
├─ L2 旋钮（声明式、可枚举、可校验、带证据）
│    provider / layout_policy / threads / partition_axis / packing / block / isa_path …
├─ L3 目标覆盖（advisory overlay，只给建议，不改变语义）
│    每后端一张"按形状分档的推荐旋钮表"，来源=实测或 autotune
└─ L4 契约与报告（对用户的承诺）
     实际旋钮组合 × provider/库版本 × 精度类别 × 时间分解 × 降级原因
```

### 4.1 核心对象：`OpSpec`

每个算子声明一份自描述规格（这是"主干 + 旋钮"的载体）：

```yaml
op: matmul
semantics:                      # L0：冻结，旋钮不可触及
  contract: "C = A·B with declared accumulation and rounding; see numeric contract"
  accumulation: {bf16: fp32, int8: int32}
  rounding: rne
  saturation: declared_per_contract
  edge_cases: [k_tail, non_multiple_of_8_or_16, empty_axis, nan_inf]
trunk:                          # L1
  implementation: portable_reference
  guarantees: [bitwise_reproducible, always_available]
knobs:                          # L2
  - {name: provider, type: enum, values: [portable, vendor, auto], level: L1, default: auto}
  - {name: precision_path, type: enum, values: [strict_bitwise, vendor_default], level: L1, default: strict_bitwise}
  - {name: threads, type: int, range: [1, platform_max], level: L2, default: {portable: 1, vendor: platform_advisory}}
  - {name: partition_axis, type: enum, values: [m, n, k, batch], level: L2}
  - {name: packing, type: enum, values: [none, block_n, panel], level: L2}
  - {name: isa_path, type: enum, values: [auto, avx512bf16, vnni, s8s8, i8mm], level: L3}
advisories:                     # L3：平台建议（示例，数字来自本项目实测）
  - {platform: x86_zen4_avx512, shape_class: "m=1,k=1024,n>=5120,bf16", knobs: {provider: vendor, threads: 6, partition_axis: n}}
  - {platform: aarch64_sve256,  shape_class: "m=1, w8a8",                 knobs: {provider: vendor, isa_path: i8mm}}
report_fields: [provider, library, library_version, precision_class, time_dispatch, time_op, degraded]
```

### 4.2 旋钮元数据（每个旋钮必须齐备，缺一不可入库）

```yaml
name: partition_axis
scope: op:matmul
type: enum
values: [m, n, k, batch]
default: {portable: n, vendor: auto}
effect: 决定工作如何在线程间切分
interacts_with: [threads, blocking]
fails_when: ["threads==1 且按 k 切分无收益"]
determinism: does_not_change_numerics
evidence:                        # 没有证据的旋钮一律视为"幻觉旋钮"，不得进入 L1/L2
  measured: true
  artifact: <证据路径>
  finding: "m=1 时按 n 切分才有收益；按 m 切分在 m=1 时退化为单线程"
stability: L2_half_stable
```

### 4.3 解析优先级与 fail-closed

```text
用户显式指定  >  平台 advisory  >  算子默认值  >  主干默认值
```

- **非法组合 → 报错**（fail-closed），返回可读的诊断（哪个旋钮、为何非法、可替代的合法组合）；
- **唯一允许的自动降级**：请求的 `provider` 在目标平台不可用 → 退回 `portable`，但必须在报告里标 `degraded=true` 与原因；
- 任何自动降级都不得偷偷改变 `precision_class`：若精度类别发生了变化（例如从 `bitwise` 变成 `bounded_tolerance`），必须显式报告为**精度降级**，而不是静默通过。

### 4.4 报告 schema（对用户的承诺载体）

```yaml
execution_report:
  model: {id: ..., weights_digest: ...}
  targets: [{platform: x86_zen4_avx512, artifact_versions: {lowering: 8, runner: 6}}]
  ops:
    - {op: matmul, shape: [1, 1024, 7168], knobs_resolved: {...}, provider: oneDNN,
       library_version: ..., precision_class: bounded_tolerance, seconds: {...}, degraded: false}
  decomposition: {dispatch_seconds: ..., op_seconds: ..., wall_seconds: ...}
  gates: {performance_gated: false, reason: "host has no cpufreq; numbers are UNGATED"}
  precision: {class: ..., basis: <band/判据引用>}
```

## 5. 工作示例：matmul

### 5.1 旋钮清单（分档）

| 档 | 旋钮 | 取值 | 说明 |
|---|---|---|---|
| L1 选路 | `provider` | `portable` / `vendor` / `auto` | `auto` 允许降级但必须报告 |
| L1 选路 | `precision_path` | `strict_bitwise` / `vendor_default` | 决定能声称位级一致，还是"一致性来自版本钉死" |
| L1 选路 | `layout_policy` | `copy` / `view` | 对接 layout 算子原生化（视图别名默认关闭） |
| L2 策略 | `threads` | `1..platform_max` | 平台给上限 |
| L2 策略 | `partition_axis` | `m` / `n` / `k` / `batch` | **m=1 时按 n 切分才有效**（实测：按 m 切分在 m=1 时退化为单线程，甚至因每调用创建线程而更慢） |
| L2 策略 | `packing` | `none` / `block_n` / `panel` | 权重布局 |
| L2 策略 | `pack_cache` | `on` / `off` | 对应 vendor 的 reorder/pack 缓存 |
| L2 策略 | `blocking` | `auto` / `(M,N,K)` | 显式分块 |
| L3 内核 | `isa_path` | `auto` / `avx512bf16` / `vnni` / `s8s8` / `i8mm` | 研究/调优用 |
| L3 内核 | `microkernel` | 具体内核名 | 不承诺稳定性 |

### 5.2 平台组合示例（数字来自本项目实测，标注为 UNGATED）

| 场景 | 旋钮组合 | 可以声称什么 |
|---|---|---|
| x86 Zen4 decode（m=1,k=1024,n=7168,bf16） | `provider=vendor`, `threads=6`, `partition_axis=n` | 相对主干快 5–6×；精度=有界误差 |
| x86 W8A8（同形状） | `provider=vendor(AOCL/LPGEMM)`, `isa_path=s8s8`, `pack_cache=on` | **有望同时拿到速度与位级一致**（int32 累加精确、顺序无关） |
| aarch64 鲲鹏 W8A8 | `provider=vendor`, `isa_path=i8mm` | 同左；该平台 BF16:FP16:INT8 算力比约为 1:2:4，整数路径优势更明显 |
| 无 vendor 库 / 校验模式 | `provider=portable`, `threads=1` | **位级一致**（唯一可如此声称的路径） |

### 5.3 反例（必须能表达"不该用旋钮做的事"）

- 用户想让 `matmul` "不要饱和" → **拒绝**：饱和是 L0 语义；
- 用户想让 int8 用 fp32 累加 → **拒绝**：累加精度是 L0 语义；
- 用户想让 epilogue 少一次舍入 → **拒绝**：舍入次数属于契约。

## 6. 不可改动的约束（评审时请勿"优化掉"）

1. **位级/误差主张只对可移植主干路径成立**；vendor 路径的可复现性来自**版本钉死**，不是数学必然；
2. **"声明 = 执行"不变式**：任何 provider/旋钮解析结果都必须能被 artifact 元数据复现，运行期不得临时改变实现；
3. **fail-closed**：版本/指纹/元数据不符即拒绝；唯一允许的降级是 provider 不可用，且必须报告；
4. **性能数字的 UNGATED 规则**：在无 cpufreq / 非静默窗口上，性能只能报中位数与离散度，不得写成可比较结论；
5. **精度类别必须随旋钮一起报告**：不允许"换成 vendor 路径但精度声明不变"；
6. **证据要求**：任何旋钮都必须有实测证据；无证据的旋钮不得进入 L1/L2。

## 7. 与既有资产的接口

| 既有资产 | 与本设计的关系 |
|---|---|
| layout 算子原生化（进行中） | `layout_policy` 旋钮的落点；视图别名默认关闭，需安全判定 |
| vendor GEMM 决策（`0006`，已采纳） | `provider` 的实现层；artifact 元数据将新增 `provider/library/library_version/tolerance_class/accum_width/scale_granularity` |
| W8A8 精度收口（`0005` 之外的 C8 队列） | 报告里 `precision_class` 的判据来源（band/near-tie 规则） |
| 性能分解实测（B3c） | 报告里 `decomposition` 的数据来源；旋钮效果的证据来源 |
| `pypto.framework`（torch 桥） | 现有唯一的用户面向接口；本设计应在其上收敛而不是另起一套 |
| `PyPTOX_*` 环境变量（35 个） | 将被配置对象 + 旋钮取代；env 保留为**兼容覆盖层**，并提供弃用策略 |

## 8. 待裁决的决策点

| # | 决策 | 本文建议 | 备选 |
|---|---|---|---|
| D1 | 旋钮分档（L1/L2/L3）与默认暴露面 | 默认只暴露 L1 + 稳定 L2 子集；L3 需显式"调优模式" | 全量暴露（简单但易误用） |
| D2 | 稳定性与弃用策略 | L1 稳定（变更走勘误）；L2 半稳定（changelog）；L3 不承诺 | 全部稳定（维护成本高） |
| D3 | 非法组合 | fail-closed 报错 | "最近合法组合 + 警告"（用户友好但可能掩盖误解） |
| D4 | 旋钮合法域的定义权 | 算子（语义相关）+ 平台（能力相关）；**冲突时算子优先** | 平台优先（更易做平台特化但会破坏语义一致性） |
| D5 | 载体与命名 | 配置对象为一等公民；env 作兼容覆盖层；`doctor --knobs` 可发现 | 纯 env（现状，已证明不可维护） |
| D6 | 是否开放用户自定义内核/插件 | 先留接口、不开放实现 | 立即开放（灵活但契约风险大） |

## 9. 交付物与验收

1. **契约文档**：`OpSpec` schema、旋钮元数据 schema、解析优先级、fail-closed 规则、报告 schema（本文升级为冻结契约）；
2. **最小原型（以 matmul 为例）**：在 x86 上把 `portable` + `vendor(oneDNN)` + `vendor(AOCL/LPGEMM)` 三条路接进同一个算子，必须同时满足：
   - **可枚举**：`doctor --knobs op=matmul` 列出全部旋钮、合法域、默认值、证据链接；
   - **可校验**：非法组合被拒绝且诊断可读；
   - **可报告**：报告含 provider、库版本、精度类别、时间分解、降级原因；
   - **可覆盖**：用户显式指定优先于平台建议，且被记录。
3. **回归**：既有入口（`pypto.framework`、内部 driver）不得被破坏；35 个 env 变量的兼容行为要有测试。

## 10. 特别希望外部评审优化的地方

1. **旋钮归属**：本文按"算子声明合法域 + 平台给建议值"设计。是否存在更好的归属模型（例如"能力（capability）为中心"、"算子 - 平台 - 用户三角色分离"）？
2. **分档与发现机制**：L1/L2/L3 分档是否合理？`doctor --knobs` 是否够用，还是应该有旋钮注册表（registry）与版本化 schema？
3. **自动选择与用户控制的边界**：是否应引入成本模型（cost model）做自动选择？若引入，如何保证"可复现"与"可解释"（同样的形状为什么这次选了 vendor 那么下次选 portable）？
4. **精度的表达**：`precision_class` 的两档（`strict_bitwise` / `bounded_tolerance`）是否够？是否需要更细的"误差预算（error budget）"对象（例如按算子给出允许的 max_abs / cosine 下限）？
5. **失败与降级的语义**：`degraded=true` 是否足够？是否需要"降级链（provider ladder）"的显式声明与上限（例如最多降两级）？
6. **多算子与整网层面**：旋钮是 per-op 还是也允许 per-graph / per-region（例如"整网不允许 vendor"）？冲突如何裁决？
7. **可测试性**：如何设计一套"旋钮效果回归"，使每个旋钮的效果声明都可被自动验证（避免旋钮腐化）？
8. **迁移路径**：35 个现存环境变量如何在**不破坏现有用户**的前提下收敛？给出分阶段迁移与弃用时间表草案。
9. **是否有遗漏的控制维度**：例如内存预算（峰值 RSS 上限）、确定性要求（必须逐位可复现）、功耗/频率约束、编译时间预算、跨进程/跨设备的数据搬运策略——这些是否应当成为一等旋钮？
