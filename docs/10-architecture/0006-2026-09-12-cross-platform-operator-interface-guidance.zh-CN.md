# 跨平台算子用户接口指导（Contract + Policy + Capability）

文档编号：`0006`

日期：2026-09-12（Asia/Shanghai）

状态：`PROPOSED_FOR_IMPLEMENTATION`

关联文档：[`0005-2026-09-11-user-side-operator-interface.zh-CN.md`](0005-2026-09-11-user-side-operator-interface.zh-CN.md)

用途：在不牺牲 Core IR 语义、数值可解释性和 fail-closed 边界的前提下，给出面向用户的跨平台算子接口，以及后续实现、验证和迁移的统一依据。本文是对 0005 的优化方案：保留其中不可改动的约束，替换“把大量旋钮直接交给用户”的接口模型。

> 本文是设计指导，不代表接口已经实现，也不宣称本文中的示例 API 当前可直接导入。实现前应先把 schema、解析规则和报告格式冻结；本文不要求本次运行测试。

> **版本说明**：§1–§12 为外部评审基线；本轮审理对 §3、§5、§6、§7、§8 和 §14.3 增加了交叉约束说明；**§13 现为 v2（外部 Agent 2026-09-12 审理后建议稿）**——
> v1 由项目方追加，v2 由外部 Agent 重写（五族 → P1–P8 primitive + Composite/State + artifact/target 层，并修正了 v1 的三处问题：
> `allow_view` 的授权语义、性能数字缺 `UNGATED`、分类不完备）。§13.9 是项目方对审理方 5 个待确认项的逐条回执。

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
    input_view: capability_checked
    output_aliasing: must_not_alias_input
    materialization: provider_or_policy_checked
  numeric:
    bf16: {accumulation: f32, rounding: rne}
    s8: {accumulation: s32, zero_point: declared, saturation: declared}
  edge_cases: [k_zero, empty_axis, k_tail, nan_inf]
  effects: pure
  reference: portable_reference_v1
```

`accumulation`、`rounding`、`saturation`、zero point、scale 粒度、空维度和 NaN/Inf 行为属于 L0 语义，不能变成普通用户旋钮。若这些字段需要改变，应创建新的 contract/opcode 和版本，而不是复用同一个 `matmul` 名字。

> **注意**：上面这段 schema 只是 `matmul` 一个族的样板。本项目真实图里 matmul 只占约 7% 的算子数，**不同 primitive 族的契约必备字段差异很大**（例如搬运类最关键的 alias/stride 规则、量化 primitive 的累加宽度与饱和策略，都没有出现在上面的样板里）。逐族清单见 **§13**。

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
    parallelism:                    # deployment-level；不进入 OpContract
      worker_count: 1
      worker_index: 0
      intra_op_threads: auto
      oversubscription_policy: reject
      cpu_affinity: auto
  placement:
    device: auto
    allow_transfer_copy: true
    transfer: auto
  layout:
    preferred: preserve_input
    view_mode: forbid          # stable: forbid | prefer；require 仅 internal
    allow_materialize_copy: true
    max_copy_bytes: null
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
| `stable` | 普通用户、模型作者 | `profile`、`target`、`numeric.requirement`、`deterministic`、`max_workspace_bytes`、`fallback`、`view_mode=forbid/prefer`、`allow_materialize_copy`、`allow_fusion` | 走兼容承诺和变更记录 |
| `tuning` | 性能工程师 | `profile_id`、固定线程上限、provider pin、已登记的 shape bucket | 必须有证据、版本和可重放制品 |
| `internal` | target/provider 实现者 | ISA、tile、microkernel、寄存器/共享内存布局、CCE Pipe 参数 | 不进入公共稳定 API |

`provider=vendor` 可以作为稳定的意图，但 `provider=oneDNN:某个 microkernel` 不应成为稳定接口。用户要固定某个库时，应固定 provider ID 和版本，而不是固定库内部符号。

`view_mode` 是约束/偏好，不是 alias 授权：contract 的 `view_obligation`、liveness、写覆盖和 owner/lifetime proof 优先于 policy。`prefer` 只影响候选排序；`require` 在 proof 未接入前仅供 internal 调试使用，无法证明时必须 fail-closed。旧的 `allow_view` 只作为兼容输入，不能把 `must_not_alias` 改成 `may_alias`。

同理，`allow_materialize_copy` 只是“允许候选考虑 layout 物化”的上限条件；contract 若要求保留 storage alias，或 `max_copy_bytes`/lifetime 不满足，resolver 仍必须拒绝 copy。`allow_transfer_copy` 只控制设备/内存空间搬运，不能替代 layout 的物化约束。

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

每个 provider 的 capability fragment 还必须声明适用前置条件和可解析的 policy 字段，至少包括：`stride_class`（如 `contiguous_only`）、`slice_step_domain`、`supported_view_modes`、`alias_proof` 状态、`policy_applicability`、以及 numeric `exactness/reduction_order_id`。这些是候选过滤事实，不是用户可以自行放宽的旋钮。

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
  policy_applicability: {status: validated, rejected_fields: []}
  layout_decision:
    requested_view_mode: not_applicable
    resolved_view_mode: not_applicable
    alias_proof_digest: null
    layout_copy_bytes: null
    transfer_bytes: null
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
4. effect、alias、state、owner/lifetime 和 synchronization 规则；
5. provider、placement、workspace、线程和编译时间上限；
6. fallback、provider allow/deny 和 policy applicability 规则。

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
| `POLICY_FIELD_NOT_APPLICABLE` | policy 字段不属于该 contract/族，或 provider 未声明其适用性 | fail-closed |
| `ALIAS_PROOF_UNAVAILABLE` | 请求 view/alias，但缺少 liveness、owner 或写覆盖证明 | fail-closed；不得偷偷改成 view |

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
- view→copy 不是普通 provider fallback：只有 contract 允许 materialize、`layout.allow_materialize_copy=true` 且 `max_copy_bytes` 满足时才能选择，并必须记录 `resolved_view_mode=materialized`、`layout_copy_bytes` 和 alias decision；`view_mode=require` 缺 proof 时直接失败；
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
  reduction_order_id: declared_order_v1
  exactness_basis: contract_and_order
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

`exact` 是逐 opcode、逐 contract 判断，不是整个算子族的标签。当前 W8A8 v1 的建议表达为：
`qmatmul_s8s8_s32` 在固定整数 contract 下 `exact`；`quantize_per_token_s8` 在固定
RNE/absmax/码值域/归约顺序下 `exact`；`dequantize_epilogue_bf16` 只有相对于声明的
“先形成一次 `s_a*s_w`，再与 accumulator 相乘”及单次 RNE 顺序才是 `exact`，换乘法顺序的 vendor 路径必须是
`deterministic_bounded`，不能继承位级主张。

`profile=portable` 的含义是“只使用可移植主干并以它作参考”，不是无条件承诺所有硬件、所有 reduction 顺序都逐位相同；是否达到 `portable_bitwise` 仍由 contract、wire/ABI 和实际报告共同证明。

### 6.2 误差预算的边界

- 算子级预算用于候选过滤和单算子回归；
- region/graph 级预算必须由独立验收定义，不能简单把每个 op 的 `max_abs` 相加后宣称整网通过；
- 用户可以要求比 contract 更严格的预算；要求更宽松不会改变 contract 的真实语义，只会改变是否接受某个 provider；
- `argmax`、near-tie、state 有限性等模型级条件仍按现有验收文档单独登记，不能压缩成一个 cosine 数字。

### 6.3 确定性是独立维度

`deterministic=true` 不等于 `portable_bitwise`。它表示在给定 target、provider、版本、线程和随机源下可重复。报告应同时列出：

```text
numeric_class / exactness_basis / reduction_order_id / deterministic / accumulation / seed / thread_policy
```

---

## 7. Layout、放置和资源：作为约束，不作为微内核旋钮

跨平台性能常常由 layout、数据搬运和 dispatch 主导。它们应有一等对象，但仍然以“可接受的约束”表达。

### 7.1 Layout 与别名

建议使用：

```yaml
layout:
  preferred: preserve_input
  allowed: [contiguous, blocked]
  view_mode: forbid          # stable: forbid | prefer；require 仅 internal
  allow_materialize_copy: true
  max_copy_bytes: null
```

`view` 只有在别名、stride、写后读和生命周期证明通过时才可解析为零拷贝；否则必须生成显式 copy 并在计划中可见。`view_mode=prefer` 只影响候选排序，`require` 在 proof 未接入前不属于 stable API。不能用 policy 绕过 Core IR 的 alias/effect 契约。

### 7.2 Placement 与搬运

`placement` 应在 graph/region 级优先表达：设备、内存空间、是否允许 HtoD/DtoH/跨设备 copy、是否允许 host staging。`allow_transfer_copy` 只控制这些搬运，不控制 layout materialization。算子 provider 只能在父级允许的 placement 中选路。这样可避免每个 op 分别“选 CUDA”而图中间隐式发生不可见搬运。

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
    scheme_digest: null
    exactness_basis: contract_and_order
    reduction_order_id: declared_order_v1
    deterministic: true
    error_budget: {...}
  policy_applicability:
    status: validated
    rejected_fields: []
  layout_decision:
    requested_view_mode: not_applicable
    resolved_view_mode: not_applicable
    alias_proof_digest: null
    layout_copy_bytes: null
    transfer_bytes: null
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
  resources:
    parallelism: {worker_count: 1, worker_index: 0, intra_op_threads: 1,
                  oversubscription_policy: reject, cpu_affinity: auto}
  coverage:
    logical_operation_counts: {...}
    provider_invocation_counts: {...}
    artifact_preparation_counts: {...}
    epilogue_counts: {...}
    counting_profile:
      precision: ...
      graph_digest: ...
      domains:
        logical_operation_counts: {included_in_graph_total: true, parent_opcode: null}
        provider_invocation_counts: {included_in_graph_total: false, parent_opcode: null}
        artifact_preparation_counts: {included_in_graph_total: false, parent_opcode: null}
        epilogue_counts: {included_in_graph_total: false, parent_opcode: null}
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

## 13. 算子族、复合层与契约字段（审理后建议稿 v2）

> 本节把 §3 的四类对象落到真实算子族，同时区分 primitive、composite、artifact 和 target-private 层。
> `[事实]` 表示已有图/代码/证据；其余规范性文字是设计要求。
> 本节的统计必须绑定 `graph_digest`、`precision profile`、integration HEAD 和 evidence 路径，不能只写一个脱离 profile 的总数。

### 13.1 统计口径与覆盖边界

[事实] Qwen3.5-0.8B 的 T=5 BF16 图为 6728 个 operation。按 canonical `by_operation` 重新归类后：

| primitive 族 | 代表操作 | 数量 | 占比 |
|---|---|---:|---:|
| P1 逐元素 / 类型 / 控制生成 | cast、add/sub/mul/div/neg、exp/rsqrt/sigmoid/silu/softplus、broadcast/where、compare/iota/constant | 3662 | 54.43% |
| P2 view / layout / materialize | reshape、slice、transpose、concat、split | 2469 | 36.70% |
| P3 收缩 / 线性代数 | matmul | 469 | 6.97% |
| P4 reduction / scan | reduce_sum、reduce_mean、reduce_max | 127 | 1.89% |
| P5 只读索引 | embedding（本图无 gather 调用） | 1 | 0.01% |
| **合计** |  | **6728** | **100%** |

证据：`graph_digest 456ab519c9b9bc7f281103b4a327f14b4112d6751acf30c00cd612f40d6d786e`（T=5 BF16）、
`verify-w8a-decay-fix/raw/s3_aligned.json` 的 `op_summary.by_operation`（五组之和精确等于 6728，无未归类项）。
W8A8 图必须使用独立 profile 统计：`qmatmul_s8s8_s32`、`quantize_per_token_s8`、
`dequantize_epilogue_bf16` 的数量不能混入 BF16 图的 matmul 或 P1 统计。

本节不把所有 PyPTO 操作声称为当前 MVP 已支持。scatter/update、sort/topk、
random、collective、distributed 和 target-private hardware op 先列为 reserved 或
unsupported，并在 capability 中明确状态。

### 13.2 Primitive 族与必备 contract 字段

#### P1：逐元素、类型转换、布尔/形状生成

包括 cast、broadcast、where、compare、iota、constant、基本算术和逐元素数学函数。

必备字段：

- 输入输出 shape 与 broadcast 规则；
- dtype promotion、scalar promotion 和输出 dtype；
- rounding、saturation、NaN/Inf、domain error；
- exp/rsqrt 等数学函数的有限性和误差类别；
- predicate/boolean 语义；
- constant/iota 的来源、范围、step 和溢出规则；
- `effects`。

#### P2：view、shape/layout 与 materialization

包括 reshape、view、slice、transpose、contiguous、split、concat。

必备字段：

- 逻辑 shape、axis/permutation、slice starts/stops/steps；
- byte offset、element/byte strides、base storage identity；
- `view_obligation: must_alias | may_alias | must_not_alias`；
- readonly、写覆盖和 byte-range overlap；
- owner/lifetime/completion contract；
- 是否允许 materialize，以及 materialize 后是否仍必须保留 alias；
- descriptor/wire version、copy bytes 和 workspace；
- provider 支持的 stride/layout 子集。

当前 CPU/provider 的 contiguous、正 step、rank 上限等限制应登记为 capability/precondition；
只有明确要冻结为公共语义时，才升为 contract 字段。

#### P3：收缩/线性代数

包括 matmul、linear、batched matmul 和带 `QuantizedTensorDesc` 的 qmatmul。

必备字段：

- batch、M/N/K、transpose、layout 和 broadcast 规则；
- accumulation dtype、reduction order、determinism；
- K tail、非对齐、empty/zero-work；
- overflow/headroom；
- bias/epilogue 和舍入边界；
- 输入输出 alias/effect；
- 对量化变体的 scheme reference。

`qmatmul_s8s8_s32` 属于 P3；量化描述符不把它改造成 E 族操作。

#### P4：reduction 与 scan

必备字段：

- axes 归一化、keepdims、输出 shape；
- empty identity；
- accumulation dtype、overflow/headroom；
- reduction order 与 deterministic mode；
- NaN 传播、max/min tie 规则；
- scan 的初值、方向和状态。

融合只能在 CompositeContract 中声明；primitive reduction 的 contract 不被一个
`allow_fusion` 开关重写。

#### P5：只读索引、查表与搜索

包括 gather、embedding，并为 sort/topk/search 预留字段。

必备字段：

- index dtype、shape、broadcast、negative index；
- 越界语义和检查时机；
- table layout、row width、readonly；
- table owner/lifetime/placement/residency；
- 输出顺序、稳定性和 tie 规则；
- 是否允许 zero-copy/view。

#### P6：索引写入与 effectful update（当前 MVP 预留）

包括 scatter、scatter_add、index_add、index_put、atomic update。

必备字段：

- effect、destination alias 和写入 byte range；
- duplicate index 的顺序、atomicity 和 reduction mode；
- bounds/failure 行为；
- 部分写入是否允许；
- synchronization、stream 和 completion lifetime。

P6 不得与只读 P5 共用“表只读”假设。

#### P7：窗口、stencil、卷积（当前 Qwen 以 composite/state 形式使用）

包括 conv1d、depthwise causal convolution 和未来的 2D/3D convolution。

必备字段：

- kernel、stride、dilation、padding、groups；
- 输入/权重 layout；
- boundary/padding value；
- accumulation、output dtype 和 epilogue；
- causal/state 输入输出及 reset/initial state；
- empty/short sequence 行为。

#### P8：量化、反量化与量化 epilogue

必备字段：

- `scheme_id`；
- signedness、bits、保留码值；
- scale/zero-point 的粒度、dtype、来源和关联；
- RNE、saturation、saturation count；
- NaN/Inf、zero row、scale underflow；
- accumulation width、K 上限和 overflow；
- logical quantized layout；
- logical binding schema reference；physical wire/artifact version 另行记录。

物理 VNNI/SDOT/dp4a/tensor-core packing、tile 和 kernel ABI 属于 provider/artifact，
不属于公共 Core IR 的量化语义。`pack_weight_s8` 是 ingestion/artifact 操作，不是
Core opcode。

### 13.3 State、composite 与 reserved contract

状态不是一个可由普通 policy 临时打开的优化选项。凡是涉及 KV cache、GDR state、
conv state 或流式 decode，都必须有 `StateContract`：

- state shape/dtype/layout；
- initial/reset/checkpoint；
- transition；
- owner/lifetime；
- read/write/alias；
- prefill/decode 边界；
- asynchronous completion。

`CompositeContract` 是 region-level contract，不是 primitive opcode，也不是
ExecutionPlan 的替代物。它至少包含：

- 输入输出语义；
- builder/expansion digest；
- required primitive opcode + contract digest；
- rounding/state/alias semantic barriers；
- 允许的 fusion/rewrite；
- region-level numeric guarantee。

RMSNorm、Softmax、RoPE、attention、GDR、SwiGLU、`qlinear_w8a8` 及其 split/gate
变体首先属于 CompositeContract。只有具备独立冻结语义、native symbol、portable
reference 和证据后，才可升级为公共 opcode。

random、collective、distributed、barrier 和 Ascend/CCE physical op 先保留为
effectful/reserved 或 target-private namespace，不伪装成 P1–P8 的 portable primitive。

### 13.4 Policy 适用域

跨族公共 policy 只包含：

- `numeric.requirement`、`error_budget`；
- `deterministic`；
- `provider.mode`、`fallback`；
- graph/region 资源上限；
- placement、transfer、observability。

族特定字段必须经过 contract applicability 校验：

| 族/层 | 可表达的意图 | 不可表达的内容 |
|---|---|---|
| P2 | stable `view_mode=forbid/prefer`、internal `require`、`max_copy_bytes` | 通过 policy 授权 alias、改 stride/shape/写覆盖语义 |
| P3 | provider、workspace、线程上限、numeric requirement | ISA、microkernel、accumulation/rounding 变更 |
| P4/P1 | deterministic、region fusion preference、资源上限 | 改 reduction order 或舍入次数 |
| P5/P6 | placement、residency preference、资源上限 | 越界、duplicate update、atomic 语义 |
| P8 | provider 选择和 exact/bounded requirement | 改码制、scale 粒度、累加宽度、饱和策略 |
| CompositeContract | allow/prefer/forbid fusion | 把假融合变成公共 opcode |

`view_mode=require` 是硬要求而不是授权；resolver 没有 alias/liveness/lifetime proof
时必须返回 `CAPABILITY_UNAVAILABLE` 或 `POLICY_CONFLICT`。不适用字段必须返回
`POLICY_FIELD_NOT_APPLICABLE`，不得静默忽略。

### 13.5 单一真源与 Core IR 映射

建立 canonical `OpDefinition` registry：

```text
OpDefinition
├── canonical opcode
├── contract version/digest
├── operand/result predicates
├── attribute schema
├── shape/effect/alias verifier
├── numeric contract reference
├── reference implementation
└── allowed provider references
```

Core IR 的 `CoreType`、SSA、Region、Effect 和 canonical serialization 仍是结构真源；
`Operation` 只引用 canonical opcode 和规范化 attributes。`OpContract` 是
`OpDefinition` 的用户/报告视图，不另维护一套手写类型系统。

provider capability、artifact、op-bench 和 composite expansion 都引用同一个
`opcode + contract_version + contract_digest`。未知公共 opcode、未知 contract
版本或 digest 不匹配时 fail-closed。

### 13.6 ExecutionReport 与反欺骗检查

所有族都必须报告：

- requested/resolved policy；
- opcode/composite contract id、version、digest；
- target/capability/provider/library 版本；
- artifact/lowering/runner/wire/kernel ABI；
- numeric guarantee、determinism 和 fallback events；
- dispatch/op/transfer/kernel 时间及 `UNGATED` 状态。

P2 额外报告：

- requested/resolved `view_mode`；
- storage alias 与 name alias 分开；
- source/destination storage、byte range；
- alias proof digest、owner/lifetime；
- copy bytes、descriptor digest、实际 kernel/mode。

P8 额外报告：

- scheme、packing_id（provider/artifact 层）；
- accum_width、scale_granularity；
- saturation count、reserved code rejection；
- logical wire version、kernel ABI；
- exact/bounded precision basis。

典型反欺骗检查：

- “view”但 source/destination pointer 不同；
- policy 要求 copy 但报告无 copy bytes；
- provider/library/thread 变化而 plan digest 不变；
- fusion 改变舍入次数却仍沿用 primitive contract；
- quantization 失败后静默回退 BF16/W8A16；
- static-only、host-reference 或 blocked-device 被报告为运行 PASS；
- `UNGATED` 数字被写成跨平台加速比。

### 13.7 版本与落地顺序

以下变化必须升 contract 或相关 wire/ABI 版本：

- P1：dtype promotion、domain、NaN/Inf、rounding；
- P2：shape/stride/offset、alias/view/materialization、owner/lifetime；
- P3：accumulation、reduction order、epilogue、tail/empty；
- P4：identity、axis、order、NaN/tie；
- P5/P6：bounds、duplicate update、effect/atomicity；
- P7：padding、window、state transition；
- P8：scheme、code domain、scale、accum width、saturation、logical binding schema；physical wire 属于 artifact/provider 版本；
- Composite：可观察的 decomposition、rounding/state barrier 或 required primitive digest。

物理 packing/kernel 变化但逻辑 contract 不变时，升 provider/kernel/artifact 版本，不改
公共 opcode 语义。binding/wire schema 也必须与 logical contract 分开版本。

落地顺序：

1. 用 exact profile-specific histogram 修正 §13.1；
2. 以现有 op-bench registry 建立 `OpDefinition` 单一真源；
3. 先完成 matmul 的 resolver/plan/report 竖切；
4. P2 layout 继续 hard-off view alias，先接入 proof schema，再考虑启用；
5. 拆分 pointwise/reduction/index-read/index-write；
6. 接入 W8A8 P8，逐 opcode 声明 exact/bounded；
7. 最后接入 composite/state/streaming contract；
8. 所有性能数字保留 host、commit、协议和 `UNGATED` 标记，不写成权威加速比。

### 13.8 对外部评审问题的最终裁决

1. 五族不作为最终平级分类；采用 P1–P8 primitive + CompositeContract + artifact/target 层。
2. B 的 view 不由 `allow_view` 授权；contract 规定 alias obligation，policy 只能禁止、偏好或要求并等待 proof。
3. B 增加 storage/offset/stride/overlap/owner/lifetime/completion；E 增加 scheme、underflow、overflow、scope 和 exactness 字段。
4. `OpDefinition` registry 是唯一语义真源；`OpContract` 是规范视图，Core IR 保留结构语义。
5. builder/composite 是 region-level contract；ExecutionPlan 只记录其解析后的实现和实际融合。
6. ExecutionPolicy 采用公共字段 + applicability 校验；不适用字段显式失败，不能静默忽略。

### 13.9 项目方回执：对审理方 5 个待确认项的逐条答复（2026-09-12）

| # | 问题 | 项目方答复 | 依据 |
|---|---|---|---|
| 1 | “源 strides 必须行主连续、slice step 必须为正”是公共 contract 还是 CPU provider 限制？ | **是 lowering/provider 前置条件，不是 L0 语义**。它属于既有 `LayoutPlan` 校验与 runtime descriptor 约束；跨平台公共契约里只登记 shape/permutation/slice 的语义合法域，contiguous/step/rank 上限记入 capability precondition | `lowering/cpu/vector/plan.py` 的 `LayoutPlan` 要求 `source_strides == 连续 strides`；`abi/descriptors.py` 的 contiguous 检查 |
| 2 | `sort/topk/random/collective` 是否纳入当前公共 API？ | **全部列为 reserved**：不在公共 API、不在 capability 的 supported 集合内；出现在图上必须 fail-closed。经典 PyPTO 审计中的这些条目属于"未来需要覆盖"的历史清单 | 当前 MVP 图只用 26 个 operation key（见 §13.1 的证据文件） |
| 3 | §13 证据基线用哪个 commit？ | **统计绑定三元组**：`graph_digest 456ab519…d786e` + precision profile `BF16/T=5` + evidence `verify-w8a-decay-fix/raw/s3_aligned.json`；文档修订对应 integration HEAD `c2ec98f3c`。W8A8 图另立 profile 统计，不与 BF16 混算 | 本次核对：五组之和精确 6728、无未归类项 |
| 4 | `view_mode=require` 是否作为公共 stable API？ | **暂不**：B6 实测视图别名 hard-off、liveness proof 未接入，此时 `require` 必然失败，公开它只会变成陷阱。现阶段公共 stable 只保留 `forbid` / `prefer`，`require` 记为 internal，待 proof schema 接入后再提级 | `verify-b6-layout-native` §5：`layout_view_alias_mode="disabled"`、3 个 env 开关无效、执行器总分配独立 destination |
| 5 | W8A8 各 primitive 的 exact/bounded 划分与 scale 归约顺序是否已冻结？ | **v1 方案已冻结**，逐 opcode 划分为：`qmatmul_s8s8_s32` = `exact`（int32 累加，且无饱和时与整数和恒等）；`quantize_per_token_s8` = `exact`（RNE + absmax + 码值域 `[-127,127]`，拒 `-128`）；`dequantize_epilogue_bf16` = **相对契约声明的单次 RNE 序为 exact、相对其它乘序为 bounded**；scale 归约顺序冻结为“先形成一次 `s_a*s_w`，再与 accumulator 相乘”。机器报告必须同时写 `scheme_digest`、`reduction_order_id`、`exactness_basis`；vendor 融合路径若为两次 accumulator-scale 乘法（如 AOCL `_sym_quant`）只能记 `deterministic_bounded`，不得继承位级主张 | 契约 `docs/20-planning/0002-…`（FROZEN）；C5/C6 验收的逐位证据；AOCL LPGEMM 源码级语义发现（AOCL `(acc×s_w)×s_a` 两次乘） |

**同时采纳的三处修正**（v1 → v2）：

1. `allow_view: true` 的"授权"语义删除，改为三态 `view_mode: forbid | prefer | require` + `max_copy_bytes`，并明确"policy 不得把 contract 的 `must_not_alias` 改成 `may_alias`"；
2. §13.7 的性能引用补上完整边界：**本机 12 vCPU KVM、无 cpufreq、非静默窗口 → 永久 `UNGATED`**，只能报中位与离散度，**不得写成跨平台加速比结论**；引用时必须带 host、commit、协议与证据路径；
3. 分类从五族平级改为 **P1–P8 primitive + CompositeContract + StateContract + artifact/target 层**，并为 P6/P7 与非 Qwen 族显式标注 reserved/unsupported，避免读者误以为已覆盖全部 PyPTO。

**仍待用户裁决的两项**（不属于本次审理范围）：

- `view_mode=require` 何时提级为公共 stable API（取决于 liveness proof 的接入时机与形态）；
- W8A8 v1 之外的量化方案（新 scheme/新码制/新 scale 粒度）是否立项，以及届时是否新建 contract 版本而非复用 `w8a8-linear.v1`。

---

## 14. 悬置项清单（供下一轮外部评审；项目方 2026-09-12 追加）

> 本节只列**尚未冻结**的事项，分两类：**(A) 需要项目方/用户裁决的口径**（外部评审可给建议但不能替项目决定）；**(B) 需要外部评审复核的设计悬置**（项目方已给倾向，等评审意见后再冻结）。

### 14.1 (A) 需要项目方/用户裁决的口径

| # | 事项 | 现状与影响 | 项目方倾向 |
|---|---|---|---|
| A-1 | **L5/L6 的语料与阈值** | 契约 §5.2 的 L4–L6 阈值是暂定值，`D5` 明确需用户确认；决定 Q6 能否启动，也决定能否对外声称"W8A8 精度达标" | 先只做 L4（logits）；L5 用现有 token 链做弱化版并显式标注覆盖不足；L6 待语料与阈值 |
| A-2 | **`view_mode=require` 何时提级为公共 stable** | 当前 view alias 为 hard-off（liveness proof 未接入），公开 `require` 必然失败 | 先保持 internal，待 proof schema 接入（与 A-4 相关）后再提级 |
| A-3 | **W8A8 v1 之外是否立项新量化方案** | 新 scheme/码制/scale 粒度按 §13.7 必须新建 contract 版本，不许复用 `w8a8-linear.v1` | 等单算子框架与真实工业框架对比之后再议（用户 2026-09-12） |
| A-4 | **图执行层：自研还是外包**（决定 N4 的存废） | 现状是我们自己的 Python 逐 op 循环；B6 后 prefill launch 约 24 s 中 op 约 4.5 s，残差约 19.4 s（约 81%）。这些数字来自本机 KVM、无 cpufreq、非静默窗口，**永久 `UNGATED`**，只能作为诊断线索，不能解释为与算子内容无关或跨平台结论。若外包给框架（PyTorch/vLLM 的 executor），残差变成框架的问题；若保留自研，必须先分解再优化 | 先做**三档计时分解**（低成本的诊断），用数据支撑"自研 vs 外包"；“每次 launch 重复 `from_dict`+`canonical_json`+`sha256`”目前是待 profile 证实的假设，确认后无论走哪条路都应修；证据来自 `verify-b6-layout-native/brief.zh-CN.md §7` |
| A-5 | **W8J 注入门政策** | 现行门是"必须追平 oneDNN 才允许注入"，导致注入被禁；vendor GEMM 决策后背景已变（W8J 转 W8A8） | 改为**按 provider/精度分层声明 + 显式 opt-in + 诚实标注倍数**，而不是一刀切禁止 |

### 14.2 (B) 需要外部评审复核的设计悬置

| # | 悬置项 | 项目方已给的处理 | 希望评审回答 |
|---|---|---|---|
| B-1 | §13.9 第 1 答："源 strides 连续 / slice step 为正"是 **provider 前置条件**而非 L0 语义 | 依据：`LayoutPlan` 校验 + `abi/descriptors.py` 的 contiguous 检查 | 这个划分是否成立？若成立，capability 里应如何表达"仅支持连续输入"？ |
| B-2 | §13.9 第 3 答：统计绑定 `graph_digest + profile + evidence path` | 已给出 BF16/T=5 的 6728 精确直方图 | W8A8 profile 的直方图应包含哪些计数（packed 前/后、含/不含 epilogue）？ |
| B-3 | §13.9 第 5 答：W8A8 的 exact/bounded 逐 opcode 划分 | `qmatmul`=exact（int32）、`quantize`=exact（RNE+码域）、`epilogue`=相对契约声明的单次 RNE 序为 exact、相对其它乘序为 bounded | 这个"相对性"表述是否可被机器校验？是否应引入显式的**归约顺序 ID**？ |
| B-4 | `OpDefinition` registry 与 op-bench（`0007`）共用 | 两者必须是**同一个 registry**；U1 首切片正在建它 | registry 的 schema 应由谁定义（Core IR 侧还是 execution 侧）？版本升级谁有否决权？ |
| B-5 | 模型级验收的挂接点 | 0006 是 op/graph 作用域，而我们的验收是**模型级**（L4/L5/L6、band、near-tie、token 链） | 是否应在 `ExecutionReport` 之上定义 `ModelReport`？两者字段如何避免重复与冲突？ |
| B-6 | 进程级并行（A3 的现实） | 0006 把线程收归 graph/region 资源上限 + provider 内部调度；但 A3 实测正确姿势是 **16–20 进程 × 每进程 8 线程**，超线程为负收益 | policy 是否需要表达"我是 N 个并行 worker 之一"以避免线程超订？放在哪个字段？ |

### 14.3 本轮外部审理增补（2026-09-12）

以下是对 §14.1/§14.2 的设计建议；不替用户关闭 A 类裁决项。

| 项 | 外部审理建议 | 应落到哪里 |
|---|---|---|
| A-4 图执行层 | 先做三档诊断：`plan/metadata`、`dispatch`、`provider/kernel/transfer`；每档保留 host、commit、协议和 `UNGATED`。元数据 `from_dict/canonical_json/sha256` 重建属于执行基础设施优化，不应写入某个算子 contract | `ExecutionReport.timing`、runtime profile；不改 `OpContract` |
| A-5 注入门 | 按 provider × precision × guarantee 分层，显式 opt-in，并保留倍率/边界；禁止把“未追平 vendor”写成“不允许观察” | provider registry、deployment policy、report |
| B-1 连续输入 | 在 capability 中使用结构化前置条件：`stride_class=contiguous_only`、`slice_step_domain=positive`、`rank_max`、`layout_descriptor_version`；resolver 以 capability 过滤，不能把它们写成所有平台 L0 | `CapabilityFragment.preconditions` |
| B-2 W8A8 统计 | 同时输出四个不相加的计数域：`logical_operation_counts`、`provider_invocation_counts`、`artifact_preparation_counts`、`epilogue_counts`；每项带 `included_in_graph_total` 与 `parent_opcode`，避免 packed/epilogue 双计数 | `ExecutionReport.coverage` / op-bench schema |
| B-3 exactness | 引入机器可比较的 `scheme_digest`、`reduction_order_id`、`exactness_basis`；provider 若顺序不同只能得到 `deterministic_bounded` | `NumericGuarantee`、provider manifest |
| B-4 registry 治理 | Core IR 侧拥有 canonical opcode、属性 schema、contract digest 和 verifier；execution 侧只注册 provider/cost model。contract-breaking 变更需架构锁 + 独立验收；新增 provider 不应升级 Core IR schema | `OpDefinition` registry governance |
| B-5 模型级验收 | 增加轻量 `ModelReport` 外壳，引用一个或多个 `ExecutionReport`，只放 L4–L6、gold digest、token chain、near-tie、state 有限性等模型字段；不复制 op/provider 字段 | `ModelReport {execution_report_refs, model_gates}` |
| B-6 进程并行 | 增加 deployment-level `parallelism`：`worker_count`、`worker_index`、`intra_op_threads`、`oversubscription_policy`、`cpu_affinity`；不放入 OpContract，也不让 per-op policy 改写它 | `ExecutionPolicy.resources.parallelism`、host report |

`view_mode=require`、新量化 scheme、L5/L6 阈值和自研/外包执行层仍保持 §14.1 的用户裁决状态；本节只冻结可审计的数据形状和边界，不替用户做产品选择。
