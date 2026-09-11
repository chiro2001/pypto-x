# 跨平台算子用户接口指导（Contract + Policy + Capability）

文档编号：`0006`

日期：2026-09-12（Asia/Shanghai）

状态：`PROPOSED_FOR_IMPLEMENTATION`

关联文档：[`0005-2026-09-11-user-side-operator-interface.zh-CN.md`](0005-2026-09-11-user-side-operator-interface.zh-CN.md)

用途：在不牺牲 Core IR 语义、数值可解释性和 fail-closed 边界的前提下，给出面向用户的跨平台算子接口，以及后续实现、验证和迁移的统一依据。本文是对 0005 的优化方案：保留其中不可改动的约束，替换“把大量旋钮直接交给用户”的接口模型。

> 本文是设计指导，不代表接口已经实现，也不宣称本文中的示例 API 当前可直接导入。实现前应先把 schema、解析规则和报告格式冻结；本文不要求本次运行测试。

> **版本说明**：§1–§12 为外部评审稿原文；**§13 为 PyPTO-X 项目于 2026-09-12 追加的补充材料**（算子分类与各族的契约字段差异），作为下一轮评审输入。评审时请把 §13 视为待优化对象，其余章节的结论以原文为准。

---

## 1. 结论先行

跨平台算子接口不应以 `isa_path`、`blocking`、`microkernel` 等后端旋钮为中心。更稳定的方向是四个相互独立的对象：

```text
OpContract（算子语义契约）
        ↓ 由用户声明意图与约束
ExecutionPolicy（执行策略）
        ↓ 与目标/后端能力协商
Capability + Provider（能力与实现提供者）
        ↓ 在编译期解析并冻结
ExecutionPlan / Artifact（执行计划与制品）
        ↓ 运行时只验证并执行
ExecutionReport（可追溯报告）
```

推荐的用户体验是：

1. 用户只需要表达“我要什么”（可移植、确定性、误差预算、设备、资源上限和是否允许 vendor）；
2. 后端负责决定“怎么做”，并把候选、选择理由、降级原因和版本指纹写入计划；
3. 计划一旦编译完成就不可暗中改道，运行时只能执行已声明的实现；
4. 低层调优仍可存在，但必须放在版本化的 tuning profile 或 provider 插件内，而不是成为默认公共 API；
5. 不具备能力、契约或证据的路径应明确显示为 `BLOCKED_*` / `UNAVAILABLE`，不能伪装成通用支持。

这同时满足两个看似冲突的目标：同一份模型代码可跨平台运行；每个平台仍可使用 oneDNN、AOCL/LPGEMM、cuBLAS、hipBLAS、ACLNN 等最佳 vendor 实现。

---

## 2. 设计边界与必须继承的事实

### 2.1 继承 0005 的不可变约束

以下约束继续有效，任何实现都不得通过新接口绕过：

1. 位级或误差主张的来源必须可说明。可移植主干是跨平台基准；vendor 路径的可复现性来自版本、能力和制品指纹钉死，而不是数学必然。
2. **声明 = 执行**：artifact 中记录的 provider、内核类别和 ABI 必须与运行时实际调用一致；不得先声明 A、运行时偷偷改成 B。
3. 元数据、线格式、lowering、runner、kernel 或 provider 版本不匹配时必须拒绝执行。旧 artifact 不得“尽量运行”。
4. 唯一可以自动发生的降级是用户策略中已经允许、且仍满足数值/资源契约的候选替换；任何数值类别变化必须显式报告。
5. 无 cpufreq、非静默窗口或只具静态证据的环境中，性能只能按既有协议报告为 `UNGATED`，不能写成可比较结论。
6. 公共层只表达目标无关 Tensor/Scalar/Shape、控制流和逻辑 Tile；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe、CCE 等细节只能留在 target plugin。

### 2.2 本文解决的问题

现状的 35 个环境变量和内部 driver 把构建路径、设备选择、线程数、权重缓存、vendor 选择和开发资源混在一起。用户无法回答以下问题：

- 当前算子究竟走了 portable、哪一个 vendor，还是 host fallback？
- 这个选择是否改变了累加精度、舍入、确定性或误差承诺？
- 同一 shape 在另一台机器为什么选了不同路径，能否重放？
- 某个“旋钮”是合法能力、实验开关，还是根本没有效果的幻觉配置？

本文把这些问题分开：语义由 `OpContract` 冻结，用户控制由 `ExecutionPolicy` 表达，平台差异由 `Capability/Provider` 描述，实际选择由 `ExecutionPlan` 固化。

### 2.3 非目标

- 不把一个低层实现强行伪装成所有平台的最佳实现；
- 不让用户通过公共 API 修改算子的数学语义；
- 不在首期开放任意用户内核、任意 C ABI 或未经验证的自动代码注入；
- 不把在线 benchmark、频率波动或未经门禁的数字写成稳定性能承诺。

### 2.4 为什么不继续扩大“旋钮”

| 直接暴露的旋钮 | 长期问题 | 本文的替代物 |
|---|---|---|
| `isa_path` / `microkernel` | 把模型代码绑到一代 CPU/GPU，换平台即失效 | provider capability + 版本化 tuning profile |
| 每个 op 的 `threads` / `blocking` | 破坏 graph 级资源调度，容易制造线程风暴 | graph/region 资源上限，provider 内部调度 |
| `precision_path` 改累加/舍入 | 让“同一个算子”拥有不透明的数学语义 | 新的 `OpContract` 版本 + `NumericGuarantee` |
| 隐式 `auto` fallback | 用户不知道实际执行路径，结果难以复现 | 显式 `ExecutionPlan` + fallback chain |
| 纯环境变量配置 | 无 schema、无合法域、无法进入 artifact 指纹 | 配置对象/registry，env 只作兼容层 |

---

## 3. 四类一等对象（报告是输出，不另算一类）

### 3.1 `OpContract`：算子是什么

`OpContract` 是 Core IR 层的稳定身份。它描述输入输出、shape/layout、数值语义、别名和副作用，不描述某个 CPU 指令或 vendor 库。

建议 schema（示意）：

```yaml
op_contract:
  id: matmul
  contract_version: 1.0
  inputs:
    - {name: a, dtype: [bf16, f32, s8], rank: 2..4, layout: declared}
    - {name: b, dtype: [bf16, f32, s8], rank: 2..4, layout: declared}
  outputs:
    - {name: c, dtype: declared_by_contract, shape: broadcast_batch+m+n}
  shape_rules:
    - k_a == k_b
    - batch_dims_broadcastable
  layout_rules:
    view_allowed: false
    aliasing: no_output_input_alias
  numeric:
    bf16: {accumulation: f32, rounding: rne}
    s8: {accumulation: s32, zero_point: declared, saturation: declared}
  edge_cases: [k_zero, empty_axis, k_tail, nan_inf]
  effects: pure
  reference: portable_reference_v1
```

`accumulation`、`rounding`、`saturation`、zero point、scale 粒度、空维度和 NaN/Inf 行为属于 L0 语义，不能变成普通用户旋钮。若这些字段需要改变，应创建新的 contract/opcode 和版本，而不是复用同一个 `matmul` 名字。

> **注意**：上面这段 schema 只是 `matmul` 一个族的样板。本项目真实图里 matmul 只占约 7% 的算子数，**五个算子族各自的契约必备字段差异很大**（例如搬运类最关键的 alias/stride 规则、量化 primitive 的累加宽度与饱和策略，都没有出现在上面的样板里）。逐族清单见 **§13**。

### 3.2 `ExecutionPolicy`：用户要什么

用户接口应提供少量稳定的“意图/约束”，而不是暴露每一个实现参数。建议字段如下：

```yaml
execution_policy:
  schema_version: 1
  profile: auto_static             # portable | auto_static | frozen_tuning
  tuning_profile: null             # profile=frozen_tuning 时填写
  target: auto                     # cpu/cuda/hip/ascend 或完整 target triple
  numeric:
    requirement: portable          # portable | deterministic | bounded
    deterministic: true
    error_budget: null             # 或显式 ErrorBudget
  provider:
    mode: auto                     # portable | vendor | auto | required
    required: null                 # provider id；mode=required 时必须填写
    allow: []                       # 例如 [vendor:onednn]
    deny: []
  fallback:
    mode: portable_only             # deny | portable_only | declared
    max_steps: 1
    allow_numeric_downgrade: false
  resources:
    max_threads: auto
    max_workspace_bytes: null
    max_memory_bytes: null
    compile_time_budget_ms: null
    power_cap_watts: null
    frequency_policy: auto         # auto | locked | bounded
    thermal_policy: nominal
  placement:
    device: auto
    allow_copy: true
    transfer: auto
  layout:
    preferred: preserve_input
    allow_view: false
    allow_copy: true
  optimization:
    allow_fusion: true
    dispatch_budget: null       # 约束/诊断，不是性能承诺
  observability:
    report: summary                 # none | summary | full
    trace_resolution: false
```

字段分为三档，而不是把所有字段都当作同等稳定：

| 档位 | 面向谁 | 示例 | 稳定性 |
|---|---|---|---|
| `stable` | 普通用户、模型作者 | `profile`、`target`、`numeric.requirement`、`deterministic`、`max_workspace_bytes`、`fallback`、`allow_fusion` | 走兼容承诺和变更记录 |
| `tuning` | 性能工程师 | `profile_id`、固定线程上限、provider pin、已登记的 shape bucket | 必须有证据、版本和可重放制品 |
| `internal` | target/provider 实现者 | ISA、tile、microkernel、寄存器/共享内存布局、CCE Pipe 参数 | 不进入公共稳定 API |

`provider=vendor` 可以作为稳定的意图，但 `provider=oneDNN:某个 microkernel` 不应成为稳定接口。用户要固定某个库时，应固定 provider ID 和版本，而不是固定库内部符号。

### 3.3 `Capability` 与 `Provider`：平台能做什么

能力是事实，不是建议；provider 是实现，不是语义。二者都必须机器可读并带指纹。

```yaml
capability_snapshot:
  schema_version: 1
  target: {triple: x86_64-linux-gnu, cpu: zen4, features: [avx512f, avx512_vnni]}
  runtime: {status: available, driver: ..., device_id: ...}
  digest: sha256:...
  contracts:
    matmul: {version: 1.0, dtypes: [bf16, s8], layouts: [row_major]}
  providers:
    - id: portable
      version: portable_reference_v1
      numeric_modes: [portable, deterministic]
      workspace_bytes: 0
    - id: vendor:onednn
      version: 3.x.y
      numeric_modes: [bounded]
      deterministic_modes: [fixed_threads]
      layouts: [row_major, blocked]
      status: available
```

provider 必须至少提供以下接口语义：

```text
probe()       -> CapabilityFragment
enumerate()   -> Candidate[]
lower(plan)   -> LoweredArtifact
compile(...)  -> Artifact
launch(...)   -> RuntimeResult
explain(...)  -> ProviderExplanation
```

`status` 必须区分 `available`、`static_only`、`blocked_device`、`blocked_toolchain`、`unavailable`。静态 lowering 或代码生成证据不能冒充真机执行能力；例如 AMD gfx1036、Ascend 尚未覆盖的桥接路径必须保留其实际状态。

实现上，`CapabilitySnapshot` 是现有 `TargetSpec` / `CapabilitySet` 的可序列化、带证据的用户侧视图，不另造一套互不兼容的 target registry；快照可以包含 probe 时间和原始特征，但 digest 只纳入规范化字段。

### 3.4 `ExecutionPlan` / `Artifact`：最终怎么做

解析器根据 contract、policy 和 capability 生成不可变计划。计划至少包含：

```yaml
execution_plan:
  plan_schema_version: 1
  contract: {id: matmul, version: 1.0, digest: ...}
  request: {policy_digest: ..., shape: [1, 1024, 7168], dtype: bf16}
  target: {triple: ..., capability_digest: ...}
  selected:
    provider: vendor:onednn
    implementation: matmul_bf16_fixed_threads
    library: {name: oneDNN, version: ...}
  candidates:
    - {id: vendor:onednn, state: selected, score: ...}
    - {id: portable, state: declared_fallback, reason: contingency}
  numeric_guarantee: {...}
  fallback_chain: [vendor:onednn, portable]
  artifact: {format_version: ..., digest: ...}
```

制品 cache key 应至少包含：contract digest、policy digest、target triple、capability digest、provider/library 版本、shape/layout/dtype、lowering/runner/wire/kernel ABI 版本。任何一个影响执行的字段变化都不能复用旧制品。

`ExecutionPlan` 最终应映射到现有 compiler/runtime ABI：计划选择 `CompilerBackend`，生成 `Artifact`，运行时通过 `LaunchRequest` 交给 `RuntimeBackend`。本 RFC 增加的是解析和可观测层，不改变 W1 已冻结的 Core IR、`Artifact` 和 `RuntimeBackend.launch` 语义。

---

## 4. 面向用户的 API 形态

### 4.1 推荐的 Python 入口

公共入口继续收敛到现有 `pypto` / `pypto.framework`，不另起一套 parallel frontend。实现初期应从 `pypto.execution` 等子模块导入，避免为了实验 API 改动 `pypto.__init__`；以下是拟议形态：

```python
from pypto import ops
from pypto import execution as px

policy = px.ExecutionPolicy(
    profile="auto_static",
    target="auto",
    numeric=px.NumericPolicy(
        requirement="deterministic",
        deterministic=True,
    ),
    provider=px.ProviderPolicy(mode="auto"),
    fallback=px.FallbackPolicy(mode="portable_only"),
    resources=px.ResourcePolicy(max_threads="auto"),
)

with px.execution(policy):
    y = ops.matmul(a, b)
```

如果用户需要可移植基准：

```python
portable = px.ExecutionPolicy(
    profile="portable",
    numeric=px.NumericPolicy(requirement="portable", deterministic=True),
    provider=px.ProviderPolicy(mode="portable"),
    fallback=px.FallbackPolicy(mode="deny"),
)
```

只有在已知 provider 合同、版本和证据时才允许 pin：

```python
frozen = px.ExecutionPolicy(
    profile="frozen_tuning",
    provider=px.ProviderPolicy(required="vendor:onednn", version="3.x.y"),
    tuning_profile="matmul-m1n7168k1024-bf16-v3",
    fallback=px.FallbackPolicy(mode="deny"),
)
```

`isa_path="avx512bf16"`、`microkernel="..."`、`saturation="off"` 等写法应被拒绝或仅在 internal provider API 中可见。

### 4.2 三种作用域

策略支持 graph、region、op 三层，但不允许用局部覆盖破坏全局安全约束：

```python
with px.execution(graph_policy):
    # 默认适用于整张图
    with px.region_policy(region_policy):
        # 只收紧该 region 的约束
        y = ops.matmul(a, b, policy=op_policy)
```

合并规则不是简单的“后者覆盖前者”，而是分字段处理：

| 字段类型 | 合并规则 |
|---|---|
| 安全/数值要求 | 取交集或更严格者；冲突立即报错 |
| 允许/禁止 provider | `deny` 优先；`required` 必须满足，否则失败 |
| 资源上限 | 取更小上限 |
| 偏好（profile、advisory） | op > region > graph > deployment default |
| 设备/placement | 只能在父作用域允许的设备集合中收紧 |

因此，“整网不允许 vendor”可以在 graph 级表达；某个经过审计的 matmul 也可以在 op 级 `required=vendor:onednn`，但不能绕过 graph 的 `deny` 或数值上限。

### 4.3 Torch/vLLM 桥接

`pypto.framework.register_torch_ops()` 继续是 Torch 用户的适配点。策略应通过注册或执行上下文传入，而不是再增加一组算子专用环境变量：

```python
import pypto

pypto.framework.register_torch_ops(
    policy=policy,
    report="summary",
)
```

`linear`（开放失败）与 `linear_strict`（失败关闭）的现有语义应保留，并映射到 `fallback.mode`：新接口不能把 strict 变成静默回退。`pack_weight` / `linear_prepacked` 属于 ingestion/artifact 阶段，计划中应明确标为 host-side preparation，不伪装成 Core opcode。

### 4.4 算子作者与平台作者的注册接口

模型作者和算子实现者应使用不同的 API 面。算子作者提交的是“契约 + 主干 + 证据”，平台作者提交的是“能力 + provider”，两者都不能把 target 专属字段塞进用户模型代码。概念上的注册形态如下（仅为接口方向示例）：

```python
register_operator(
    contract=MatmulContract(version="1.0"),
    trunk=PortableImplementation(
        symbol="portable_reference_v1",
        guarantees=["portable_bitwise", "deterministic"],
    ),
    providers=["vendor:onednn", "vendor:cublas", "vendor:aclnn"],
    evidence="evidence://matmul/contract-v1",
)

register_provider(
    id="vendor:onednn",
    version="3.x.y",
    capabilities=...,             # 由 probe/manifest 产生
    candidates=...,               # 只返回满足 contract 的实现
)
```

注册时必须校验：

1. 每个公开算子都有可运行的 portable trunk，或显式标记为某些 target 上的 `unsupported`；
2. provider 的输入输出、累加和边界行为与 `OpContract` 对齐；
3. provider manifest 给出真实支持的 dtype/layout/shape、确定性模式、workspace 和证据状态；
4. 任何 fused provider 都要说明融合后的语义和舍入边界；只优化执行顺序而不改变语义的 fusion 应体现在 plan/report，不另造一个“假融合” opcode；
5. 注册表中的 contract、provider ABI 和 evidence ID 可独立版本化、撤回和审计。

这样，用户侧只写 `ops.matmul`，算子作者不必为每个 target 复制一套公共 API，平台作者也不必修改模型源码即可加入 vendor 实现。

### 4.5 W8A8-linear 的落地方式

当前项目的 W8A8-linear 是最适合验证这套接口的真实案例。用户侧只应看到已经冻结的量化 contract 和执行意图：

```python
y = ops.qlinear_w8a8(
    x,
    packed_weight,
    contract="w8a8-linear.v1",  # builder/composite 的语义 contract
    policy=px.ExecutionPolicy(
        profile="auto_static",
        numeric=px.NumericPolicy(requirement="deterministic", deterministic=True),
        provider=px.ProviderPolicy(mode="auto"),
    ),
)
```

其中 `s8/s8 → s32` 累加、zero-point、scale 粒度、RNE/饱和、尾块和 packed wire layout 都来自 `w8a8-linear.v1`，不能通过 policy 改写。`packed_weight` 是 ingestion/artifact 阶段的 host-side 对象，不应作为一个“可选优化 opcode”混入 Core 图。

`qlinear_w8a8`、QKV split、SwiGLU 等仍应优先作为 builder/composite；只有当某个 fused 名字拥有独立且冻结的语义契约、native symbol 和验证证据时，才允许升级为公共 opcode。否则 plan 可以报告“已融合的实现”，但 IR 不应增加假融合节点。

resolver 可以在能力快照允许时选择 AVX2 widening、AVX-512 VNNI、SVE256 widening、CUDA `dp4a` 或平台 vendor GEMM；但每条候选都必须登记 `packing_id`、`accum_width`、`scale_granularity`、kernel ABI 和数值证据。某个平台没有对应 packed layout 时，应在 plan 中显式记录一次转换/copy 或返回 `CAPABILITY_UNAVAILABLE`，不能悄悄把 W8A8 变成 BF16/W8A16。

---

## 5. 解析、自动选择与降级

### 5.1 解析顺序

建议的输入优先级为：

```text
op 显式约束 > region 约束 > graph 约束 > deployment profile
> 平台 advisory > 算子默认值
```

但最终解析采用“约束求交 + 偏好排序”，不是无条件覆盖。任一候选必须同时满足：

1. `OpContract` 版本和 shape/layout/dtype 规则；
2. capability 的真实支持范围；
3. numeric/determinism/error budget；
4. provider、placement、workspace、线程和编译时间上限；
5. fallback 和 provider allow/deny 规则。

不满足时返回结构化错误，至少包含 `field`、`requested`、`available`、`reason`、`suggested_policy` 和 `capability_digest`。

建议把错误码固定下来，并映射到现有 `PyPTOXBridgeError`、`PyPTOXBuildError`、`PyPTOXUnsupportedError` 等异常层次：

| 错误码 | 触发条件 | 默认动作 |
|---|---|---|
| `OP_CONTRACT_INVALID` | shape/dtype/layout/effect 不满足 contract | 立即失败 |
| `POLICY_CONFLICT` | graph/region/op 约束无交集 | 立即失败并给出冲突字段 |
| `CAPABILITY_UNAVAILABLE` | target 不具备所需特征或仅有静态证据 | 进入已声明 fallback，否则失败 |
| `PROVIDER_UNAVAILABLE` | provider、库或版本缺失 | 按 fallback chain，否则失败 |
| `NUMERIC_GUARANTEE_UNMET` | provider 不能满足 error budget/确定性 | 禁止静默替换 |
| `RESOURCE_BUDGET_EXCEEDED` | workspace、内存、线程或编译预算超限 | 选择满足上限的候选，否则失败 |
| `ARTIFACT_MISMATCH` | digest、ABI、版本或 capability snapshot 不一致 | fail-closed |
| `FALLBACK_DISALLOWED` | 请求的降级未被 policy 声明 | fail-closed |

### 5.2 自动选择模式

自动选择分为三种，避免“同样输入每次随机选路”：

| 模式 | 选择依据 | 是否在线测量 | 适用场景 |
|---|---|---:|---|
| `portable` | 只选 trunk | 否 | 参考、跨平台回归、精度基准 |
| `auto_static` | 版本化能力表 + 确定性 cost model | 否 | 默认生产路径 |
| `frozen_tuning` | 已生成并签名的 tuning profile | 否（运行时） | 固定硬件/shape 的性能部署 |

不建议首期提供“每次运行现场 benchmark 后自由切换”的模式。若将来加入 `auto_tuned`，必须把测量样本、环境指纹、候选集合、得分和选路结果写入 tuning artifact；再次运行只能重放该 artifact，不能因为频率、温度或负载变化静默改变计划。

cost model 的 tie-break 必须固定，例如：

```text
numeric requirement → determinism → no-copy → workspace → provider rank
→ estimated cost → provider id lexical order
```

性能估计只是选路依据，不自动构成性能承诺；最终报告仍须遵守 `UNGATED` 规则。

同一份用户 policy 可以在不同能力快照上得到不同 provider，但不应得到不同的模型源码或 Core IR。例如在 capability 均为 `available` 且 contract 兼容时，典型计划可能是：

| 能力快照 | 可能的 provider | 计划中必须留下的事实 |
|---|---|---|
| x86_64 + AVX-512 | oneDNN / AOCL-LPGEMM | CPU features、库版本、线程策略、数值类别 |
| AArch64 + SVE256/i8mm | 平台 vendor 或 portable widening | VL、指令能力、是否 deterministic、是否发生 copy |
| NVIDIA CUDA | cuBLAS 或 CUDA provider | driver/device、库版本、kernel 计时是否可用 |
| AMD HIP runtime 可用 | hipBLAS/HIP provider | runtime 状态、gfx 能力、静态证据与运行证据分开 |
| Ascend bridge 可用 | ACLNN/CCE provider | CANN/driver、target plugin、桥接版本和设备状态 |

表中是 resolver 的设计示例，不是对当前设备已经具备这些运行能力的声明；`doctor` 的真实 capability snapshot 才是选路依据。

### 5.3 显式降级链

`degraded: true` 一个布尔值不够。计划应保存完整的候选链和事件：

```yaml
fallback_chain:
  - {provider: vendor:onednn, state: rejected, reason: library_missing}
  - {provider: portable, state: selected, reason: policy_allows_portable_fallback}
fallback_events:
  - kind: provider_unavailable
    from: vendor:onednn
    to: portable
    numeric_class_changed: false
```

规则如下：

- 默认 `max_steps=1`；复杂链必须由 policy 明确声明；
- provider 不可用时，只有仍满足同一 numeric requirement 的 portable 候选可以自动接替；
- 不得自动把 `portable` 换成低精度、非确定性或更宽误差路径；
- `allow_numeric_downgrade=true` 只能由显式用户策略开启，并要求给出 `ErrorBudget`，否则 fail-closed；
- 编译时已冻结的 artifact 在运行时缺库时默认失败；只有 artifact 内含已声明的 fallback 且 policy 允许，才可执行回退；
- 每次降级都进入报告、日志和计数器，不得只打印一次容易丢失的 warning。

---

## 6. 数值、确定性与误差预算

### 6.1 从两档字符串升级为结构化保证

`strict_bitwise` / `bounded_tolerance` 适合作为报告摘要，但不足以指导跨算子和跨平台选择。内部应使用结构化 `NumericGuarantee`：

```yaml
numeric_guarantee:
  class: portable_bitwise       # exact | portable_bitwise | deterministic_bounded | best_effort
  reference: portable_reference_v1
  accumulation: f32
  rounding: rne
  deterministic: true
  error_budget:
    scope: op
    max_abs: null              # 由具体 op/graph contract 或验收证据填写
    max_rel: null
    cosine_min: null
    ulp_max: null
    argmax_preservation: required
  evidence:
    id: evidence://...
    method: gold_fp32_vs_bf16_band
```

建议类别：

| 类别 | 含义 | 可否作为默认跨平台承诺 |
|---|---|---:|
| `exact` | 整数/位级契约精确满足 | 可以，须明确输入和顺序 |
| `portable_bitwise` | 同一 portable 主干、同一 wire/ABI 下逐位可重放 | 可以，但不等于所有 vendor |
| `deterministic_bounded` | 固定 provider/版本/线程后确定，误差有预算 | 只能带版本和证据 |
| `best_effort` | 无稳定误差或确定性保证 | 只能显式 opt-in |

vendor GEMM 仍可作为各平台默认性能实现，但它必须声明对应类别和误差证据；不能因为算子名字相同就继承 portable 的位级主张。

`profile=portable` 的含义是“只使用可移植主干并以它作参考”，不是无条件承诺所有硬件、所有 reduction 顺序都逐位相同；是否达到 `portable_bitwise` 仍由 contract、wire/ABI 和实际报告共同证明。

### 6.2 误差预算的边界

- 算子级预算用于候选过滤和单算子回归；
- region/graph 级预算必须由独立验收定义，不能简单把每个 op 的 `max_abs` 相加后宣称整网通过；
- 用户可以要求比 contract 更严格的预算；要求更宽松不会改变 contract 的真实语义，只会改变是否接受某个 provider；
- `argmax`、near-tie、state 有限性等模型级条件仍按现有验收文档单独登记，不能压缩成一个 cosine 数字。

### 6.3 确定性是独立维度

`deterministic=true` 不等于 `portable_bitwise`。它表示在给定 target、provider、版本、线程和随机源下可重复。报告应同时列出：

```text
numeric_class / deterministic / accumulation / reduction_order / seed / thread_policy
```

---

## 7. Layout、放置和资源：作为约束，不作为微内核旋钮

跨平台性能常常由 layout、数据搬运和 dispatch 主导。它们应有一等对象，但仍然以“可接受的约束”表达。

### 7.1 Layout 与别名

建议使用：

```yaml
layout_policy:
  preferred: preserve_input
  allowed: [contiguous, blocked]
  allow_view: false
  allow_copy: true
  max_copy_bytes: null
```

`view` 只有在别名、stride、写后读和生命周期证明通过时才可解析为零拷贝；否则必须生成显式 copy 并在计划中可见。不能用 `layout_policy=view` 绕过 Core IR 的 alias/effect 契约。

### 7.2 Placement 与搬运

`placement` 应在 graph/region 级优先表达：设备、内存空间、是否允许 HtoD/DtoH/跨设备 copy、是否允许 host staging。算子 provider 只能在父级允许的 placement 中选路。这样可避免每个 op 分别“选 CUDA”而图中间隐式发生不可见搬运。

### 7.3 资源约束

`max_threads`、`max_workspace_bytes`、`max_memory_bytes` 和 `compile_time_budget_ms` 是上限，不是性能保证。线程策略默认由 graph runtime/provider 管理；只有 tuning profile 才固定到某个线程数。共享资源锁、heavy runner 和证据目录是开发基础设施配置，不应暴露成普通模型 API。

功耗上限、频率策略和 thermal policy 可以作为部署 profile 的可选约束，但前提是 target 能力快照确实报告并执行它们：

```yaml
resources:
  power_cap_watts: null
  frequency_policy: auto       # auto | locked | bounded；不支持时 fail-closed
  thermal_policy: nominal
```

这些字段只限制可选候选或运行条件，不能把“频率已锁定”推导成性能门禁通过；报告仍须记录实际 gate 状态。

`optimization.allow_fusion` 是一个安全的意图开关：调试或逐 op 对照时可以设为 `false`，生产 profile 通常设为 `true`。`dispatch_budget` 只能作为显式约束或诊断字段；如果 provider 无法满足，resolver 应报告原因或失败，不能把“少于 N 次 dispatch”当成未经测量的性能保证。实际融合边界、dispatch 数和是否仍有 `host_reference` fallback 都必须写入 plan/report。

---

## 8. 报告与可发现性

### 8.1 `ExecutionReport` 最小 schema

```yaml
execution_report:
  schema_version: 1
  request:
    op: matmul
    shape: [1, 1024, 7168]
    policy_digest: sha256:...
    requested_policy: {...}
  resolved:
    target: x86_64-linux-gnu/zen4
    provider: vendor:onednn
    implementation: matmul_bf16_fixed_threads
    library: {name: oneDNN, version: ...}
    profile: auto_static
    resolved_policy: {...}
  guarantees:
    numeric_class: deterministic_bounded
    precision_class: bounded_tolerance   # 兼容旧报告；由 numeric_class 展开
    deterministic: true
    error_budget: {...}
  resolution:
    candidates: [...]
    fallback_events: []
    degraded: false
  artifact:
    contract_version: 1.0
    lowering_version: ...
    runner_version: ...
    wire_version: ...
    kernel_abi: ...
    digest: sha256:...
  timing:
    dispatch_seconds: ...
    op_seconds: ...
    transfer_seconds: ...
    wall_seconds: ...
    kernel_seconds: null
  gates:
    performance_gated: false
    reason: host_no_cpufreq
  capability:
    snapshot_digest: sha256:...
```

`kernel_seconds` 在 CUDA cuEvent 计时尚未落地或计时范围不满足协议时必须为 `null`，不能用 wall time 冒充。报告应同时保留 `requested_policy` 和 `resolved_policy`，否则用户无法发现自动选择或降级。

### 8.2 三个发现命令

建议提供稳定 CLI（名称可在实现阶段微调，但语义应保持）：

```text
pypto doctor --capabilities [--target ...] [--op matmul]
pypto explain --op matmul --shape 1,1024,7168 --policy-file policy.yaml
pypto plan --graph graph.json --policy-file policy.yaml --emit plan.yaml
```

- `doctor` 只报告事实：能力、工具链、设备状态和证据状态；静态证据与运行证据分开；
- `explain` 展示候选、过滤原因、cost model 输入和最终选择，不执行大任务；
- `plan` 生成可审计、可签名、可重放的计划；执行时使用 `--plan` 可禁止重新解析。

---

## 9. 环境变量迁移

环境变量保留为兼容覆盖层，不再作为新的公共接口。优先级统一为：

```text
代码 API / 显式 CLI > 项目配置文件 > 兼容环境变量 > 平台默认值
```

同一字段由 API 和 env 同时设置时，API 生效并输出一次结构化 `CONFIG_ENV_IGNORED`；未知或拼写错误的变量在 `doctor` 中列出。影响 artifact 语义的 env 必须进入 policy digest，不能在运行时暗改。

### 9.1 迁移映射

| 现有变量 | 新配置归属 | 可见性 |
|---|---|---|
| `PYPTO_X_AVX2_BUILD_DIR`、`PYPTO_X_AVX512_BUILD_DIR`、`PYPTO_X_SVE256_BUILD_DIR`、`PYPTO_X_ASCEND_BUILD_DIR` | `build.targets.<target>.dir` | internal/tooling |
| `PYPTO_X_AVX2_ARTIFACT_DIR`、`PYPTO_X_AVX512_ARTIFACT_DIR`、`PYPTO_X_SVE256_ARTIFACT_DIR` | `artifact_store.targets.<target>.dir` | tooling |
| `PYPTO_X_AVX2_PROBE_DIR`、`PYPTO_X_AVX512_PROBE_DIR` | `capability.probe_cache` | tooling |
| `PYPTO_X_SVE256_ARTIFACT_DIR` | `artifact_store.targets.sve256.dir` | tooling |
| `PYPTO_X_ASCEND_ARTIFACT_FORMAT` | `artifact.format` + Ascend provider profile | stable/tuning |
| `PYPTO_X_AARCH64_SYSROOT` | `toolchain.aarch64.sysroot` | tooling |
| `PYPTO_X_CANN_INSTALL_PATH`、`PYPTO_X_CANN_META_ROOT`、`PYPTO_X_HWHIAIUSER_HOME` | `toolchain.ascend.*` / `evidence.ascend.*` | internal |
| `PYPTO_X_PTO_ISA_ROOT`、`PYPTO_X_PYPTO_ROOT` | `toolchain.*` | internal |
| `PYPTO_X_TORCH_CACHE_DIR` | `cache.torch_dir` | tooling |
| `PYPTO_X_AVX512_GEMM_THREADS`、`PYPTO_X_VLLM_GEMM_THREADS` | `execution.resources.max_threads` 或 provider tuning profile | stable/tuning |
| `PYPTO_X_AVX512_WEIGHT_CACHE` | `cache.weight_pack` | tooling |
| `PYPTO_X_ASCEND_DEVICE_ID` | `target.device_id` / `placement.device` | stable |
| `PYPTO_X_ASCEND_HOOKS` | Ascend provider/plugin 注册表 | internal |
| `PYPTO_X_SVE_QEMU_CPU` | `target.qemu.cpu` | tooling |
| `PYPTO_X_SVE_TEST_MODE` | `validation.mode` | internal |
| `PYPTO_X_PORTABLE_ONLY` | `execution.profile=portable` + `provider.mode=portable` | stable 兼容别名 |
| `PYPTO_X_QWEN35_ASSET_DIR`、`PYPTO_X_QWEN35_ASSETS_DIR`、`PYPTO_X_QWEN35_WEIGHTED_WORKTREE` | `assets.models.qwen35.*` | model/tooling |
| `PYPTO_X_VLLM_ENABLE`、`PYPTO_X_VLLM_GATE_JSON`、`PYPTO_X_VLLM_LOG` | `integration.vllm.*` | integration |
| `PYPTO_X_TORCH_FALLBACK_WARNINGS` | `observability.fallback_warnings` | stable |
| `PYPTO_X_LOCAL_HEAVY_RUNNER`、`PYPTO_X_RESOURCE_LOCK_ROOT`、`PYPTO_X_RESOURCE_USAGE_DIR` | `developer.resource.*` | internal；不进入模型制品 |

其中 `PYPTO_X_AVX2_PROBE_DIR` 等历史变量的精确别名关系需由实现时的 registry 登记；若一个旧变量存在多种含义，应拆成两个新键并在迁移警告中指出歧义，而不是继续扩大旧变量的语义。

### 9.2 分阶段时间表

1. **P0（登记）**：建立 `config_registry`，为 35 个变量记录类型、默认值、合法域、作用域、弃用版本和迁移键；不改变行为。
2. **P1（兼容层）**：加入 `ExecutionPolicy` / 配置文件；env 仍可用，但每次使用输出 `CONFIG_ENV_DEPRECATED`，并生成等价 policy 摘要。
3. **P2（默认切换）**：文档和示例只使用 API/配置文件；内部 driver 改为显式传 policy；env 不再创建新的功能。
4. **P3（移除）**：至少跨一个主版本、且有迁移检查器后，才移除纯内部变量；用户可见别名应保留到 major release，并在 changelog/ERRATA 登记。

---

## 10. 版本、插件和安全边界

### 10.1 版本指纹

每次计划和制品至少记录：

```text
op_contract_version
policy_schema_version
capability_schema_version + capability_digest
provider_abi_version + provider/library version
lowering_version
runner_version
wire_version
kernel_abi_version
artifact_format_version
```

契约级变化（图结构、binding schema、量化语义、layout/alias 规则、数值合约）必须升 contract/digest；实现优化但语义不变时也必须更新 provider/kernel 指纹。任何旧制品无法证明兼容时 fail-closed。

### 10.2 Provider 插件边界

provider 可以注册能力、候选和 lowering，但不能：

- 修改 Core `OpContract` 的数学定义；
- 绕过用户的 numeric、determinism、placement 或 fallback 约束；
- 以“支持某个 opcode”的名义提供实际未融合、未验证的假实现；
- 把静态生成、仿真或 host reference 证据标为设备执行 PASS。

自定义 kernel/插件首期只开放内部 SPI 和审计后的 provider manifest。公共插件 ABI 应在 plan/report schema 稳定后再单独立项。

### 10.3 `pypto` 与 `pypto_pro`

两者继续在同一 wheel 中并存：

- `pypto` 暴露本 RFC 的目标无关 contract、policy、plan 和公共算子；
- `pypto_pro` 可表达 Ascend/专家 dialect，但必须通过 provider/target 边界进入计划；
- 未完成语义对齐前，不把 Pro 当作 classic 的替代入口，也不把 Ascend 专属字段泄漏到公共 policy。

---

## 11. 后续开发路线与验收门槛

实现应按“先可解释、再自动化、最后调优”的顺序推进：

| 阶段 | 交付 | 退出条件 |
|---|---|---|
| P0 schema | `OpContract`、`ExecutionPolicy`、`Capability`、`Plan`、`Report` schema 与 registry | schema 有版本、合法域、默认值和错误码；不改现有入口 |
| P1 portable resolver | 以 `matmul` 为例接通 portable provider | `profile=portable` 可列举、可计划、可重放；计划与执行一致 |
| P2 vendor provider | 接入 oneDNN/AOCL/LPGEMM 中至少一条 x86 provider | provider 版本、数值类别、候选拒绝原因和 artifact 指纹完整；无库时按 policy 正确回退/失败 |
| P3 graph policy | graph/region/op 作用域、layout/placement/resource 约束 | 冲突按“收紧或 fail”处理；不会出现隐式跨设备搬运 |
| P4 observability | `doctor`、`explain`、`plan` 和 `ExecutionReport` | 用户可从报告回答“走了什么、为何走、能否复现、是否降级” |
| P5 env migration | 35 个 env 的兼容 registry、警告和文档迁移 | 旧用户行为保持；新功能不再增加 env；artifact 不受未登记 env 偷改 |
| P6 tuning | frozen tuning profile、shape bucket 和可重放 cost model | 在线不重新 benchmark；profile 过期或能力指纹变化时拒绝或重新计划 |

每个 provider/算子都应有以下回归类别（具体执行仍遵守资源锁和既有验收协议）：

1. contract 不变量：shape、边界、alias、NaN/Inf、空维度和累加语义；
2. capability 矩阵：可用、静态-only、toolchain/device blocked 的状态不能混淆；
3. resolver golden：相同输入和 capability digest 产生相同 plan；
4. plan replay：制品实际 provider 与报告一致，旧/篡改制品 fail-closed；
5. fallback：只沿 policy 声明链降级，且不偷偷降低 numeric guarantee；
6. 数值证据：portable、vendor、graph-level 误差分别报告；
7. 性能证据：dispatch/op/transfer/kernel 分解，遵守 `UNGATED` 和 `kernel_seconds=null` 规则；
8. 兼容性：`pypto.framework`、Torch bridge、vLLM integration 和现有 env 别名不回归。

---

## 12. 采用决策（替代 0005 的开放问题）

| 主题 | 本文建议 | 原因 |
|---|---|---|
| 旋钮归属 | 能力中心 + 用户意图；算子定义合法语义，provider 声明能力，resolver 负责选择 | 避免平台旋钮侵入模型代码 |
| L1/L2/L3 | 对用户改成 `stable/tuning/internal`；内部仍可保留 L0–L3 映射 | 稳定性比“旋钮数量”更容易理解 |
| 自动选择 | 默认 `auto_static`，cost model 必须可解释、可重放 | 防止运行时随机改道 |
| 非法组合 | fail-closed；只给出可执行的建议 policy | 错误尽早暴露，避免数值/性能误解 |
| 作用域 | graph > region > op 的约束收紧模型 | 同时支持整网禁 vendor 和局部特化 |
| 精度表达 | `NumericGuarantee + ErrorBudget`，摘要仍可输出两档兼容字段 | 能表达 BF16/W8A8、near-tie、argmax 和确定性 |
| 降级 | 显式 fallback chain，默认最多一步；数值降级必须 opt-in | `degraded=true` 单字段不足以审计 |
| 发现机制 | registry + `doctor`/`explain`/`plan` | 让旋钮和能力可发现、可验证、可复现 |
| 自定义内核 | 首期不开放公共实现，只留 provider SPI | 先稳定 contract、artifact 和报告边界 |
| 35 个 env | API/配置一等、env 兼容覆盖并分阶段弃用 | 不破坏旧用户，同时停止继续膨胀 |

如果本 RFC 获得批准，0005 的第 6 节继续作为硬约束；0005 的“trunk + knobs”示例和 D1–D6 开放问题由本文的 contract/policy/capability/plan 模型取代。真正冻结前，应把本文件中的“拟议 API”改写为版本化 schema，并在 `configs/development_lock.yaml` 登记其契约版本和迁移状态。

---

## 13. 算子分类与各族的契约字段差异（**项目补充，待评审**）

> 本节由 PyPTO-X 项目于 2026-09-12 追加，目的是把 §3 的四类对象落到**真实的算子族**上。所有数量与状态都来自本项目已验收或已冻结的实测，不是设计推测。

### 13.1 为什么必须有这一节

§3.1 的 `OpContract` 示例只写了 `matmul`。但本项目真实模型（Qwen3.5-0.8B，T=5 prefill，6728 个算子，AVX-512 后端）的构成是：

| 算子族 | 代表算子 | 算子数 | 占比 |
|---|---|---:|---:|
| 逐元素/转换 | cast 1484、mul 747、broadcast 494、add 398、silu/exp/where | ~3300 | **~49%** |
| 搬运/重排 | transpose 889、reshape 776、slice 696、concat 84、split 24 | ~2470 | **~37%** |
| 矩阵乘 | matmul | 469 | **~7%** |
| 规约 | reduce_sum / reduce_mean / reduce_max | ~130 | ~2% |
| 索引 | embedding / gather | 1（表为 248320×1024） | ~0% |
| 量化 primitive | qmatmul_s8s8_s32 / quantize_per_token_s8 / dequantize_s8 / dequantize_epilogue_bf16 | 未量化跑时为 0 | — |

**结论**：matmul 是"花算力"的，但不是"花时间/花算子数"的。若照 matmul 的样板硬套到别的族，最容易漏掉的是**搬运类的别名（aliasing）规则**——而这恰好是历史上最容易出正确性事故的地方（本项目为此专门把零拷贝视图别名默认关闭，并要求生命周期证明）。因此 `OpContract` 必须是**按族定义必备字段**的分类体系。

### 13.2 各族的契约必备字段

| 族 | `OpContract` 必备字段（L0，不可被 policy 改写） |
|---|---|
| **A 矩阵乘** | 输入输出 dtype 组合；`accumulation`；`rounding`；`saturation`；量化时的 zero-point / scale 粒度；K 尾与非 8/16 倍数处理；空轴；NaN/Inf；`aliasing: no_output_input_alias`；`effects: pure` |
| **B 搬运/重排** | 输入输出 shape 规则；**源 strides 必须行主连续**（本项目既有约束）；`permutation` 合法域；`slice_starts/slice_steps` 合法域（步长必须为正）；**view/alias 规则与写覆盖禁令**；空轴与退化 shape；dtype 覆盖范围（未覆盖的必须显式登记） |
| **C 逐元素/规约** | 广播规则；归约轴与 keepdims；**融合边界与融合后舍入次数**；NaN/Inf 传播规则；dtype 提升表；`effects` |
| **D 索引/查表** | 表布局与行宽；**越界索引语义（本项目为 fail-closed）**；表的只读性与别名；表生命周期与常驻策略；输出顺序 |
| **E 量化 primitive** | 码值区间（本项目为 `[-127,127]`，**拒绝 `-128`**）；舍入模式（RNE）；scale 的粒度与 dtype；**`accum_width = int32`**；饱和策略与饱和计数口径；packed wire layout 版本；NaN/Inf/下溢处理 |

### 13.3 各族的"允许用户表达什么"（policy 侧）

| 族 | 允许作为用户意图的表达 | 禁止进入 policy 的东西 |
|---|---|---|
| A 矩阵乘 | `provider.mode`（portable / vendor / auto / required）、`resources.max_threads` 上限、`numeric.requirement`、`error_budget`（只允许更严） | `isa_path`、`microkernel`、`blocking`、库内符号、逐 op 线程数 |
| B 搬运/重排 | `layout.allow_view`、`layout.allow_copy`、`max_copy_bytes` | 任何改变 shape/stride/别名语义的字段；`allow_view` 只能**收紧**，不能放宽契约 |
| C 逐元素/规约 | `optimization.allow_fusion`、dtype 相关约束、内存预算 | 改变舍入次数或归约顺序的开关（那是新 contract） |
| D 索引/查表 | 内存预算、表常驻策略、placement | 越界行为（属 L0） |
| E 量化 primitive | 只有：`provider.mode`（`portable` / `required`）与"必须位级一致"的声明 | **任何放宽**：不许改码制、scale 粒度、累加宽度、饱和策略 |

### 13.4 各族的报告必须回答的问题

| 族 | `ExecutionReport` 必须回答 |
|---|---|
| A 矩阵乘 | 用了哪个 provider / 库 / 版本；线程策略；精度类别；是否降级；dispatch 与 kernel 时间 |
| B 搬运/重排 | **是否零拷贝（视图）**；若拷贝，拷了多少字节；是否触发写覆盖保护；descriptor digest |
| C 逐元素/规约 | 是否融合、融合边界在哪；dispatch 次数；是否仍有 host fallback |
| D 索引/查表 | 是否零拷贝；表是否常驻；峰值内存 |
| E 量化 primitive | `packing_id`、`accum_width`、`scale_granularity`、kernel ABI、**饱和计数**、wire 版本、是否 fail-closed |

### 13.5 各族"最容易骗人"的地方（应成为报告检查项）

| 族 | 典型的"谎" |
|---|---|
| A | 偷偷换了 provider / 线程数变了但报告不说 |
| B | **声称是视图，实际在拷贝** |
| C | 融合后舍入次数变了，却仍挂着"语义未变" |
| D | 悄悄把零拷贝变成拷贝（大表时代价巨大） |
| E | 失败后回退成 BF16/W8A16 假装成功 |

### 13.6 各族"必须升契约版本"的触发条件

| 族 | 触发 |
|---|---|
| A | 累加精度、舍入、饱和、量化方案、K 尾处理改变 |
| B | 别名或 stride 规则、view 语义改变 |
| C | 融合边界或舍入次数、归约顺序语义改变 |
| D | 表布局、越界语义改变 |
| E | 码制、scale 粒度、累加宽度、饱和策略、wire layout 改变 |

### 13.7 落地顺序（按"收益 ÷ 风险"）

```text
竖切仍只做 matmul(A)：打通 policy → resolver → plan → report 一条线
但 schema 必须为五族留位（留空，不许用 matmul 的字段占死）

推进顺序：
  1) 搬运/重排(B)      —— 本项目已实测收益最大（原生后 10.5 ms/call → 0.022 ms/call）
  2) 量化 primitive(E) —— W8A8 主线；且必须位级一致，没有讨价空间
  3) 逐元素/规约(C)    —— 等单算子测试框架（0007）的实测数据再决定是否动融合
  4) 索引(D)           —— 只有 embedding 一个大表，单独立项
持续: A 由 vendor 选型（0006 主体 / Q8）推进
```

### 13.8 请评审者重点回答的问题

1. **分类是否遗漏或应合并**？（本项目的 cast/where 归入 C；是否需要单列"复合算子"一类，例如 `qlinear_w8a8`、QKV split、SwiGLU 这类 builder/composite？）
2. **13.3 的"允许意图"是否过多或过少**？特别是 B 族的 `allow_view`：应由 policy 表达，还是必须由契约强制、policy 只能收紧？
3. **五族的必备字段是否有遗漏**？请特别检查 B 族（别名/生命周期）与 E 族（饱和与累加）。
4. **`OpContract` 与既有 Core IR opcode 的映射如何登记**，才能避免"双份真源"（本项目要求单一真源，契约是既有 opcode 的规范视图，而不是第二套类型系统）？
5. **复合算子（builder/composite）**在四类对象模型里应处于什么位置——是 contract 的一种、还是 plan 的一种表达？
6. **跨族的一致性**：同一个 `ExecutionPolicy` 施加到五个族时，语义是否仍然自洽（例如"不允许 vendor"对 A 有意义、对 E 几乎无意义）？是否需要在 policy 里按族收敛字段？
