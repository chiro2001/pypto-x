# 0023 — U 线算子注册第三批（batch3）：8 个 Tier-2 opcode 契约 + 零默认变化 + 主验收指标 13→5

- 任务：`u-ops-registry-batch3`（batch 0051 切片 A）
- 实现基线：integration tip `9264deb93`（tree `ec93dfb19a9392cd930f787f1441f4a30800547c`，冻结）
- 分支：`work/u-ops-registry-batch3`（未 push）
- 实现 commit：round-1 `403633d97`（integration cherry-pick `a2373408d`）+ round-2
  `5c1d8dd4d9d47ae52fcdccd61d7b8b3745ef33d4`（父提交 `6ee250569`，含 A/B/C/D 四项；
  见 §11 契约修订记录）
- 所有证据在 round-2 实现树上重跑；零默认变化对照含父树 `9264deb93`
- 依赖：`0019`（batch1 的 schema v3 契约、通用入口、按 opcode 收窄的 capability 视图）、`0022`
  （batch2 的视图语义冻结与主验收指标口径）、`0021`（U5 只读 region 与 §5 真实图缺失契约频次表）
- 结论等级：**实现完成（静态 shape；portable provider 路径）**。未做性能声明；本机数字一律 **UNGATED**
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u-ops-registry-batch3/`

## 0. 摘要

本切片把执行层 opcode 契约从 16 个扩到 24 个，新增 **Tier 2 的 8 个 opcode**：

`reduce_mean`、`concat`、`constant`、`silu`、`neg`、`sub`、`sigmoid`、`softplus`

主验收指标（§8）：在集成 tip 上用 U5 `plan_region` 对三个真实 Qwen3.5 图各跑一次，
缺失契约从 **13 类降到 5 类**（只剩 `split/div/gather/identity/embedding`），
8 个 opcode 的缺失次数全部归零；已登记契约的成功规划数从 **4282 升到 4657**，
`unplannable` 归零（round-2 已允许 `constant` 的 rank-0 scalar-kind 结果）。

> **勘误（round-2，登记 ERR-0016）**：round-1 曾把 66 个 `constant` 的 `unplannable`
> 归因为“U5 region 预检的入口边界、**非**契约缺口”；U5 round-5 的最小复现证明该归因错误：
> 撤掉 region 预检后 66 次仍全部失败，`plan_operation("constant", [], dtype="float32",
> output_shape=[], attributes={"value": 1e-06})` → `rank_out_of_range`（旧契约只允许
> rank 1..4）。这是**契约缺口**。round-2 允许 rank-0 scalar-kind 输出后，region 侧
> （round-5 已就绪）无需再改，66 次全部可规划。

### 0.1 冻结兼容策略（关键设计决定）

1. **8 个新契约**沿用 registry 级 schema **v3**（`0019 §0.1`），`contract_version = "1.0"`、
   `allowed_providers = ("portable",)`、`effects = pure`、`verifier = generic`；
   16 个既有契约的 document / digest 一字节不改。
2. **capability 视图按批次冻结累积集合**：`operation_provider_capability(opcode)` 现在按 opcode
   返回冻结集合——batch1 请求 → 10 个 contract（与 0048-B 一致）、batch2 请求 → 16 个、
   batch3 请求 → 24 个。默认探测（`default_providers`）仍只声明 matmul/qmatmul 两个契约，
   `matmul`/`qmatmul` 路径不受影响。
3. **`concat` 引入 variable-arity 契约机制**：`arity=1`（最小操作数）、`max_arity=64`（含），
   契约文档额外带 `variable_arity=true` 与 `max_arity` 两个键；按请求生成的 operand 角色为
   `operand0..operandN-1`。批量 1 个输入是退化拼接（逐位复制），与 portable 参考 lowering
   的「至少 1 个输入」一致。固定 arity 契约的文档/路径逐字节不变。
4. **整数 `reduce_mean` 走 portable scalar 专属放宽**：`validate_program` 新增
   `allow_integer_reduce_mean=False` 关键字；只有 `lower_cpu_scalar`、`CpuScalarProgram`
   与执行层的 `verify_lowering_accepts` 对 `reduce_mean` 传 `True`。向量 planner
   （AVX2/AVX512/SVE）保持 0051 前的 float-only 准入，不会获得未实现的整数 mean 路径。
5. **`constant` 非有限浮点字面量的规范拼写**：Core IR 的 attribute 载体（canonical JSON）
   拒绝原始 NaN/infinity 浮点，因此契约把 `NaN/±inf` 冻结为字符串拼写 `nan`/`inf`/`-inf`，
   portable 参考在 materialize 前解码回 IEEE-754 非有限值。有限字面量保持数值原样。
6. **`constant` rank-0 = scalar-kind 结果（round-2 范围扩展）**：`output_shape=[]` 是
   scalar-kind 结果（单字面量），rank 1..4 是 tensor 结果；plan 的 `request` 对非 tensor
   结果**新增可选** `outputs=[{name, role, kind, dtype, shape}]`（`kind="scalar"`、
   `shape=[]`，与 U5 round-5 的 region 约定一致），tensor 结果的 request/digest 逐字节不变。
   `execute_operation` 对 scalar plan 返回原始标量（int/float/bool）且不接受 output buffer；
   provider 以 `CoreType.scalar` 建程序、`LaunchRequest.outputs=[]`，runtime 的 scalar return
   直接回传。`ProviderCapability.rank_range` 下限从 1 放宽到 0（只影响新 scalar 契约的
   收窄视图，旧 manifest/digest 不变）。
7. 不修改 `pypto/__init__.py`、`python/pypto/portable/qwen35.py`、`python/pypto/execution/region.py`
   （round-5 已就绪），不新增环境变量。

## 1. 八个新契约的逐算子摘要

所有 8 个契约：rank 域（张量操作数）1..4（`constant` 输出 rank 0..4：rank 0 = scalar-kind
单字面量、rank 1..4 = tensor；`reduce_mean` 输出 rank 0..4 的 tensor）；`attributes` 白名单
之外一律 `unknown_attribute`；序列型属性只接受有序 list/tuple，set/generator/mapping 一律
结构化拒绝。**reason 口径**：`constant` 的 value 容器用 `unordered_attribute_container`；
`reduce_mean` 的 `axes`/`axis` 走 batch1 的既有轴校验路径，无序容器报
`reduction_axis_invalid`（这是冻结的 batch1 错误码，本切片不改；见 §6 表）。

| opcode | 元数/操作数 | shape 规则 | dtype 域 | 结果 dtype | 属性白名单 | portable executor 符号 |
|---|---|---|---|---|---|---|
| `reduce_mean` | 1（input） | rank(in) 1..4，rank(out) 0..4；`shape(out)=reduced_shape(in, axes, keepdims)`；**任一被约简轴长度为 0 → 契约拒绝**（mean 无单位元） | 全部 portable dtype 去 bool | = input dtype | `axes`/`axis`（同现必须归一化一致）、`keepdims`(bool) | `pypto.backends.cpu.runtime._reduce_mean_value` |
| `concat` | 1..64（variable arity） | 所有输入同 rank 1..4、同 dtype；非 `axis` 维必须逐维相等；`shape(out)[axis]=Σ shape(input_i)[axis]` | 全部 portable dtype（含 bool） | = 输入共同 dtype | `axis`/`dim`（默认 0，负值归一化；同现必须一致） | `pypto.backends.cpu.runtime._concat_value` |
| `constant` | 0（无输入） | 零输入；**显式声明** rank 0..4 输出 shape（`[]` = scalar-kind 单字面量；rank 1..4 = tensor）；`element_count(output_shape) == 字面量个数`（无隐式填充/广播） | 输出 ∈ 全部 portable dtype | 显式声明的 output dtype（**绝不从字面量推断**） | `value`/`values`（恰一；标量或有序嵌套 list/tuple） | `pypto.backends.cpu.runtime._constant_value` |
| `silu` | 1 | 逐元素，`shape(out)=shape(in)`，rank 1..4 | `float16/bf16/float32/float64` | = input dtype | 无 | `pypto.backends.cpu.runtime._math_value` |
| `neg` | 1 | 逐元素，`shape(out)=shape(in)`，rank 1..4 | 全部 portable dtype 去 bool | = input dtype | 无 | `pypto.backends.cpu.runtime._neg_value` |
| `sub` | 2（left, right） | 尾轴对齐广播；scalar 为 rank-0 恒等（任一侧）；两个 scalar 拒绝 | 全部 portable dtype 去 bool | = 操作数共同 dtype | 无 | `pypto.backends.cpu.runtime._binary_value` |
| `sigmoid` | 1 | 逐元素，`shape(out)=shape(in)`，rank 1..4 | `float16/bf16/float32/float64` | = input dtype | 无 | `pypto.backends.cpu.runtime._math_value` |
| `softplus` | 1 | 逐元素，`shape(out)=shape(in)`，rank 1..4 | `float16/bf16/float32/float64` | = input dtype | 无 | `pypto.backends.cpu.runtime._math_value` |

广播的精确规则与 batch1/batch2 同源（执行层直接调用 `pypto.lowering.cpu._broadcast_shape`
后复核）：尾轴对齐；每维相等或其一为 1；`0` 只与 `0`/`1` 广播（结果为 0）；
其他组合一律 `broadcast_shape_mismatch`。

`sub` 复用 batch2 的 `_build_elementwise_binary_definition` 模板（整数 exact / 浮点 bounded、
尾轴广播、scalar admission、`build_binary_elementwise_program` 显式降 broadcast）；
`silu/sigmoid/softplus` 复用 `_math_value`/`_math_scalar`，不新增第二套参考实现。

`contract_digest`（实现树，`raw/batch3_record.json`）：

```text
concat     sha256:adf0aca3d6dd02c35c5218fb09f261221de1b906e14c469392af27ba721b750a
constant   sha256:cf33e1a612ffd4f0b526247a976e56acd6052a4abe08fc2ff9b286c06e73a4d8
neg        sha256:016922779127b2b82cc9cc93dcc7058b90e937410663028f69efb53546b377f3
reduce_mean sha256:3ab5ee5aaec518b3af33fb175fc68b144a3b3faa730c073a0d9866ed82acf9c4
sigmoid    sha256:abee314e247ed9aa7c59eaf9059a761c5ee83b4679699acfe99f5e87e05fff32
silu       sha256:50dede82914e16840b9e34f3107724fcc6342ff38651d46acd295d380e1f0162
softplus   sha256:ad362695e1498edd07b9647f77c2c9c2aa7f254c79a88d0a9fc6370b53b5f8da
sub        sha256:0a37e2a3da1c3c479437a9ae338a9211bc6ac17a10b504fe3cd6909410e63aa5
```

> round-2 契约修订（见 §11）：`reduce_mean`（示例更正）、`constant`（binary64 可解码性 +
> NaN 文本 + rank-0 scalar-kind 输出）的 digest 已更新（上表为修订后值），其余 6 个新契约与
> 16 个既有契约不变。


## 2. 精度声明（诚实口径）

| opcode / 路径 | 类别 | bound kind | 理由 |
|---|---|---|---|
| `neg` 整数 | **exact** | `exact_contract_bound`（proven，bound=0） | 每个元素为精确 Python 整数取负 + 一次带范围检查 cast；最小负值取负越界 → fail-closed，无回绕/饱和 |
| `neg` 浮点 | `deterministic_bounded` | `deterministic_serial_order_no_registered_analytic_bound` | IEEE-754 符号位翻转本身逐位精确；但执行层未注册 proven 包络，`require_proven_deviation_bound` 必须 fail-closed |
| `sub` 整数 | **exact** | `exact_contract_bound`（proven，bound=0） | 精确 Python 整数减 + 一次带范围检查 cast；越界 fail-closed |
| `sub` 浮点 | `deterministic_bounded` | 同上 | 每个元素一次 binary64 运算 + 一次 RNE cast；**无解析上界** |
| `reduce_mean` 整数 | **exact** | `exact_contract_bound`（proven，bound=0） | 精确 Python 整数和 + 向零截断除法 + 带范围检查 cast；除法语义与取整规则是本契约冻结的参考语义 |
| `reduce_mean` 浮点 | `deterministic_bounded` | 同上 | flat 行主序升序 binary64 累加（从整数 0 起）+ 一次 binary64 除法 + 一次 RNE cast，与 `reduce_sum` 同族；**无解析上界** |
| `silu`/`sigmoid`/`softplus` 全部（float only） | `deterministic_bounded` | 同上 | binary64 逐元素求值 + 一次 RNE cast；`silu` 的舍入顺序 = `x * sigmoid(x)`（与既有 `_math_scalar` 一致）；**无解析上界** |
| `concat` 全部 | **exact** | `exact_contract_bound`（proven，bound=0） | 纯数据搬运（逐元素复制），无算术、无转换；NaN/±0/±inf/次正规/bf16 位型原样保留 |
| `constant` 整数/布尔 | **exact** | `exact_contract_bound`（proven，bound=0） | 字面量在契约期做范围检查，materialize 只做精确整数/布尔 cast；无回绕/饱和/浮点截断 |
| `constant` 浮点 | `deterministic_bounded` | 同上 | 字面量先解码入 binary64，再到目标 dtype 一次 RNE cast；**无解析上界**（f32/bf16 有限溢出饱和为 signed inf；f16 有限溢出 fail-closed）。**超出 finite binary64 范围的整数字面量在契约期拒绝**（`constant_literal_not_representable`），不会变成执行期失败 |

`numeric.require_proven_deviation_bound = true` 时：

- 整数/布尔 exact 路径（`neg`/`sub`/`reduce_mean`/`constant` 的整数路径、`concat` 全部）
  正常解析；
- 所有浮点 bounded 路径（`neg/sub` 浮点、`reduce_mean` 浮点、三个激活、`constant` 浮点）
  **fail-closed**：`NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable`，
  portable provider 也生效。

## 3. 语义边界（契约文字 + 用例）

- **NaN**：`neg` 的 `-NaN` 为 NaN（符号位/payload 不属于 portable 值契约）；`sub` IEEE-754
  传播；`silu/sigmoid/softplus(NaN)=NaN`；`concat` 原样保留 NaN 位型；`constant` 的 NaN
  字面量冻结为规范拼写 `"nan"`，执行时物化为 canonical quiet **正** NaN（payload 与符号位
  不保留）；`reduce_mean` NaN 传染；+inf + -inf = NaN。`constant` 的**非 NaN** 位型（±0、
  次正规、±inf、普通有限值）按 §4.1 的规则保持（位型用例见 §7.1）。
- **±0**：`neg(-0.0)=+0.0`、`neg(+0.0)=-0.0`；`silu(+0.0)=+0.0`、`silu(-0.0)=-0.0`
  （冻结的 `x>=0` 分支对 `-0.0` 成立）；`softplus(±0.0)=ln 2`；`sigmoid(±0.0)=0.5`；
  `sub` 按 binary64 规则；`reduce_mean` 从整数 0 起加，全 `-0.0` 组得到 `+0.0`；
  `concat`/`constant` 原样保留符号位。
- **±inf**：`silu(+inf)=+inf`、`silu(-inf)=+0.0`（显式分支）；`softplus(+inf)=+inf`、
  `softplus(-inf)=+0.0`；`sigmoid(+inf)=1.0`、`sigmoid(-inf)=0.0`；`neg` 翻转符号；
  f32/bf16 溢出饱和为 signed inf，f16 有限溢出 fail-closed。`constant` 的 Python 浮点
  字面量（含 `1e400` 解析成的 `+inf`）按规范拼写处理；超出 finite binary64 的**整数**
  字面量在契约期拒绝。
- **次正规**：`concat`/`constant`/`neg` 原样保留（constant 非 NaN）；激活在 binary64 中求值
  后按目标格式 RNE（可下溢为次正规或 0）；`reduce_mean` 的 binary64 累加保留次正规、最后
  一次 cast 决定目标格式；用例含最小正 float32 次正规 `0x1p-149`。
- **bf16 位型**：`concat`/`neg` 与 `constant` 的非 NaN 值逐位保留；激活/mean 按 binary32
  位型做 RNE（NaN payload 不承诺保留）。用例覆盖 subnormal `0x0001`、`-inf 0xFF80`、
  NaN `0x7FC0` 与 RNE tie 边界。
- **整数回绕**：所有整数路径**不回绕、不饱和**。`sub` 越界、`neg` 最小负值取负、
  `constant` 整数输出越界（`constant_value_out_of_range`）、float 输出下超出 finite
  binary64 的整数字面量（`constant_literal_not_representable`，契约期）、`reduce_mean`
  越界商，全部 fail-closed（执行错误时输出缓冲保持原值；执行期结构化 reason
  `operand_value_rejected`）。
- **空张量 / 零长度轴**：`concat` 的零长度 segment 合法（输出轴长 = 各输入轴长之和，可为 0）；
  `constant` 的 `element_count=0` 需要零个字面量；`neg`/激活 0 元素输入 → 0 元素输出；
  `sub` 支持 `0` 维广播；`reduce_mean` 对 0 长度被约简轴在**契约层**拒绝
  （`empty_reduction_has_no_identity`，与 `reduce_max` 的“无单位元”口径对齐；
  `reduce_sum` 保留加性单位元 0）。
- **原地/别名**：全部 8 个契约 `must_not_alias`（`inputs_readonly_outputs_distinct`）；
  执行期输出与任一输入同 buffer → `output_input_alias`。

## 4. `constant` 的 value 域与 `reduce_mean` 语义（两个冻结决定）

### 4.1 `constant` value 域（唯一口径）

- 恰好出现 `value`/`values` 之一（缺失 → `constant_value_missing`，同现 → `constant_value_conflict`）。
- 字面量容器只接受**有序**嵌套 `list/tuple` 或单个标量；`Mapping`/`set`/`frozenset`/
  iterator/generator → `unordered_attribute_container`；字符串只在浮点 dtype 下接受规范拼写
  `nan`/`inf`/`-inf`（大小写不敏感，`+inf` 归一化为 `inf`），其余字符串 → `constant_value_domain_invalid`。
- **dtype 推导规则：不推导**。结果 dtype 必须由 `dtype`（通用入口）或 `output_dtype` 显式声明；
  两者同现必须一致（`dtype_mismatch`）；`bool` dtype 只接受 Python bool；整数 dtype 只接受
  Python int（bool 不算）且在契约期做范围检查（越界 → `constant_value_out_of_range`，
  **不做浮点→整数截断**）。
- **float dtype 的 binary64 可解码规则（round-2 收紧）**：整数字面量必须能解码进 finite
  binary64（`float(value)` 不溢出）；超出范围（如 `10**400`、`10**309`）在**契约期**结构化
  拒绝 → `constant_literal_not_representable`，不再到执行期才 fail-closed。Python 浮点
  字面量本身就是 binary64（`1e400` 在请求构造时已经是 `+inf`，按规范拼写 `inf` 处理）。
  边界用例：`10**308` 接受（int→binary64 按 IEEE-754 舍入；f32 目标随后饱和为 inf），
  `10**309` 拒绝。
- **shape 规则**：输出必须是 rank 0..4；rank 0 是 **scalar-kind** 结果（单字面量，
  `shape=[]`，与 U5 round-5 的 `kind="scalar"` 约定一致），rank 1..4 是 tensor 结果。
  没有 rank-0 tensor 拼写（kind 不歧义）；`output_shape` 必填；字面量个数必须**恰好等于**
  `element_count(output_shape)`（rank 0 时 == 1；无填充/广播），否则
  `constant_element_count_mismatch`。
- 非有限浮点字面量冻结为 `nan`/`inf`/`-inf`；`plan.to_json()` 往返、region replay、执行都使用
  同一规范形态（进 plan digest）。**NaN 规范化**：任意 NaN（含负号/带 payload）都冻结为
  `"nan"`，执行物化为 canonical quiet **正** NaN（0x7FC00000），payload/符号位不保留；非 NaN
  位型（±0、±inf、次正规、普通有限值）保持（位型用例见 §7.1）。
- 零输入表达与 `iota` 同机制：`request.inputs=[]`，`arity=0`；`plan_operation("constant", [],
  dtype="float32", output_shape=(n,), attributes={"value": ...})`。

### 4.2 `reduce_mean` 语义（唯一实现 `_reduce_mean_value`）

- 分组规则与 `reduce_sum/reduce_max` 同一实现（`_reduction_groups`）；每组候选按 flat 行主序输入序。
- **浮点路径**：`aggregate` 从整数 0 起，按 flat 升序做 binary64 加法；随后**一次** binary64
  除法除以组大小；最后**一次** RNE cast 到输出 dtype（与 `reduce_sum` 的“flat 升序 f64 累加 +
  一次 RNE cast”同族）。
- **整数路径**：和是精确 Python 整数；商 = `trunc(sum / count)`（**向零截断**，
  例：`mean([-3,2])=0`、`mean([-4,2])=-1`、`mean([1,2])=1`）；随后对输出 dtype 做带范围检查的
  cast，越界 fail-closed。该除法/取整规则是冻结的参考语义，用例逐条钉住。
- **空归约组**：任一被约简轴长度为 0 → 契约层拒绝（0/0 无单位元），
  与 `reduce_max` 的 `empty_reduction_has_no_identity` 对齐。

### 4.3 `concat` 是复制类，不是视图类

- `view_obligation = must_not_alias`（`inputs_readonly_outputs_distinct`），不进入 batch2 的
  `layout.view_mode` 三态机制；`layout_decision.resolved_view_mode="not_applicable"`，
  `layout_copy_bytes=0`。
- 真实 copy 字节数 = `element_count(output_shape) × dtype_size(output_dtype)`，是冻结
  `output_shape`/`output_dtype` 的确定性函数，二者都在 plan digest 内，因此**不额外**引入
  layout 字段也能复算；`view_mode=require` 对 `must_not_alias` 契约仍按既有口径拒绝
  （`contract_requires_distinct_storage`），不静默降级。
- 1 个输入的退化拼接到输出与输入逐位相同，但 portable trunk 仍物化独立 buffer
  （执行期 `_check_no_alias_any` 兜底），不构成零拷贝视图。

### 4.4 scalar-kind 结果 ABI（round-2）

`constant` 的 rank-0 结果是唯一启用该路径的契约（其他契约保持 tensor 结果）：

- **契约**：verifier 返回 `output_kind="scalar"`；`OpDefinition.verify_operands` 用
  `CoreType.scalar(dtype)` 做 lowering 交叉校验；`verify_generic_operation` 只在该契约的
  output spec `rank_min==0` 时接受 scalar declared kind，并要求 `shape=[]` 与 verifier
  推导 kind 一致（`result_kind_mismatch` / `result_kind_outside_contract_domain`）。
- **plan**：`request` 新增可选 `outputs=[{name:"output", role:"output", kind:"scalar",
  dtype, shape:[]}]`（仅在非 tensor 结果时写入，tensor plan 逐字节不变）；`ExecutionPlan`
  提供 `output_kind` / `output_entries` 属性；schema v3 校验器交叉检查 `outputs[0]` 与
  `output_shape`/`output_dtype`、scalar ⇒ `shape=[]`、单输出 arity。
- **provider / lowering / runtime**：portable provider 以 `CoreType.scalar(dtype)` 建程序、
  `LaunchRequest(outputs=[])`；`CpuScalarRuntime.launch` 的 scalar return 直接回传；输出
  buffer 参数对 scalar plan 结构化拒绝（`unsupported_output_handle`）。
- **v4 report**：report 的 `request` echo 复制 `outputs`，且报告内嵌的
  `plan_payload.document.request.outputs` 与 plan digest 锚定；`validate_report_schema` 通过。
- **region（round-5 已就绪，本切片不改 region.py）**：Core IR scalar-kind 结果在
  region `request.outputs[i]` 写 `kind="scalar"` + `shape=[]`，可作 scalar operand 与
  region exit；`plan_region` 对 constant scalar 走同一 `plan_operation` 路径，`kind` 两侧一致。
- **capability**：`ProviderCapability.rank_range` 下限放宽为 0（`0 <= min <= max`），
  `constant` 的收窄视图为 `(0, 4)`；旧 manifest/digest 不变。

## 5. 通用入口 / plan 变更（严格加法）

- `plan_operation` / `resolve_operation` / `OpDefinition.verify_generic_operation` 的 arity 校验：
  固定 arity 契约行为逐字节不变；`variable_arity=True` 的契约按 `[arity, max_arity]` 校验。
- 可变 arity 的角色名冻结为 `operand0..operandN-1`（region 请求构建器与 resolver 同源）；
  plan payload 的 `request.inputs[i].name/role` 为该拼写。
- zero-input 契约（`iota`/`constant`）的 `dtype`/`output_dtype` 双拼写现在做一致性交叉检查
  （此前 `iota` 在两者同现且不同时会静默采用 `output_dtype`；现为结构化 `dtype_mismatch`）。
- `provider_ids.PORTABLE_GENERIC_OPCODES` / `PORTABLE_IMPLEMENTATIONS` 增加 8 个 opcode。
- `execution/region.py` 的 region-plan 校验器对 `variable_arity` 契约按范围校验 operand 数
  （固定 arity 分支逐字节不变）；这是 region 层支持可变 arity 的最小改动，不升
  `REGION_SCHEMA_VERSION`（既有 v2 文档结构/语义不变）。
- `CpuScalarValueError`（`CpuScalarExecutionError` 子类）把执行期的 operand/output 值拒绝
  映射为结构化 `OP_CONTRACT_INVALID / operand_value_rejected`；既有 `pytest.raises(
  CpuScalarExecutionError)` 仍匹配（子类关系），非值类执行失败不被误标。

## 6. fail-closed 矩阵（新增触发器）

| 触发 | 错误（code / reason） | 发生层 |
|---|---|---|
| 未知 opcode（Tier 3：`split`/`div`/`gather`/`identity`/`embedding`） | `OP_CONTRACT_INVALID / opcode_not_registered` | registry/resolver |
| dtype 越域（`neg`/`sub` 的 bool；激活/`constant` 的不支持 dtype） | `dtype_outside_contract_domain` | verifier |
| 混合 dtype / scalar 与 tensor dtype 不一致（`sub` 无隐式提升） | `dtype_mismatch` | verifier |
| 两个 scalar 操作数的 `sub` | `scalar_result_not_supported` | verifier |
| rank 越界（输入 rank>4；`constant` rank >4；`constant` 秩 5 或输入符号维） | `rank_out_of_range` | verifier |
| 广播不合法（`sub`） | `broadcast_shape_mismatch` | verifier（复用 lowering） |
| 未知属性 | `unknown_attribute` | verifier（白名单） |
| 无序容器属性/字面量（set/generator/mapping） | `unordered_attribute_container`（例外：`reduce_mean` 的 `axes`/`axis` 走 batch1 既有轴校验 → `reduction_axis_invalid`） | verifier |
| `constant` float dtype 的整数字面量超出 finite binary64 | `constant_literal_not_representable` | verifier（契约期） |
| `concat` 操作数个数不在 [1, 64] | `arity_mismatch` | entry/resolver/verifier |
| `concat` 轴非法/越界/`axis`与`dim`冲突 | `concat_axis_invalid` / `conflicting_attribute_aliases` | verifier |
| `concat` 非轴维不相等 / rank 不一致 / dtype 不一致 | `concat_shape_mismatch` / `concat_rank_mismatch` / `dtype_mismatch` | verifier |
| `constant` 缺/重 `value`、容器非法、domains 非法 | `constant_value_missing` / `constant_value_conflict` / `unordered_attribute_container` / `constant_value_domain_invalid` | verifier |
| `constant` 整数越界、字面量个数不符 | `constant_value_out_of_range` / `constant_element_count_mismatch` | verifier |
| `constant` 缺 output dtype/shape | `output_required` / `request_dtype_required` | entry/verifier |
| `constant` `dtype` 与 `output_dtype` 冲突 / `dtype` 与字面量类别冲突 | `dtype_mismatch` / `constant_value_domain_invalid` | entry/verifier |
| `reduce_mean` 空被约简轴 | `empty_reduction_has_no_identity` | verifier |
| `reduce_mean` 轴重复/越界、`axes`/`axis` 冲突、`keepdims` 非 bool | `reduction_axis_invalid` / `conflicting_attribute_aliases` / `keepdims_not_bool` | verifier |
| `require_proven_deviation_bound=true` 遇浮点 bounded 路径 | `NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable` | resolver |
| 执行期整数越界（`neg` 最小值、`sub` 溢出、`constant` int 越界、f16 溢出） | `OP_CONTRACT_INVALID / operand_value_rejected` | portable provider |
| scalar plan 携带 output buffer | `unsupported_output_handle` | entry |
| `constant` declared kind 与 verifier 推导 kind 不一致 / 契约不支持 scalar 结果 | `result_kind_mismatch` / `result_kind_outside_contract_domain` | `verify_generic_operation` |
| scalar 结果 `shape` 非空 / provider 声明不一致 | `result_kind_mismatch` / `plan_field_inconsistent` | verifier / plan 校验器 |
| 重签名的 scalar kind/shape 篡改 | `ARTIFACT_MISMATCH / plan_declaration_mismatch`（重解析 canonical request 后比对） | entry `_verify_plan_binding` |
| 输出别名输入 | `output_input_alias` | provider |
| plan 被篡改 | `ARTIFACT_MISMATCH / plan_digest_mismatch` 等 | plan 校验器 |

## 7. 验证与证据

### 7.1 新增测试（`python/tests/ut/pypto_x/test_execution_ops_registry_batch3.py`，145 项）

覆盖：registry/inventory、8 个契约的 dtype×rank×shape×attribute 矩阵、差分矩阵
（NaN/±0/±inf/次正规/bf16 位型/整数极值/广播/空张量/零长度轴）、`constant` value 域
（含 round-2 的 binary64 可解码边界、NaN canonical 位型与 rank-0 scalar 正/负/往返用例）、
`reduce_mean` 浮点 f64 参考与整数向零截断、plan/report 自包含、冻结 plan JSON 重放、
篡改拒绝、8 opcode 的 region 规划（`plan_region` + `validate_region_plan(re_resolve=True)`）、
precision class 与 `require_proven_deviation_bound` 行为。

- focused（未持锁，默认 12 线程）：本切片 145 passed；与 batch1/batch2/region/discovery/
  matmul/view_mode/binding/report/cpu-scalar/op-bench 相关 16 个套件合并 →
  **832 passed / 0 failed**（约 22 s；`logs/focused_execution_suites_round2.log`）。
- 父提交非空转证据：新测试文件在父树 `9264deb93` 上 collect 即
  `ImportError: cannot import name '_constant_value' from 'pypto.backends.cpu.runtime'`，rc=2
  （`logs/parent_commit_new_tests_error.log`）。
- 差分口径：执行层 `execute_operation` 输出 vs 契约声明的 portable 参考 helper
  （`_reduce_mean_value` / `_concat_value` / `_constant_value` / `_neg_value` /
  `_math_value` / `_binary_value`）逐位比较（NaN 按 NaN、±0 按符号位）。
- round-2 专项用例：`10**400`/`-10**400`/`10**309` → `constant_literal_not_representable`；
  `10**308` 边界接受（f64 finite；f32 饱和 inf）；`1e400`（Python `+inf`）→ 规范拼写 `inf`；
  负 quiet NaN `0xFFC00000` → 规范 `"nan"`、执行输出 `0x7FC00000`；同请求的 `-0.0`/
  次正规/`+inf` 位型保持。
- rank-0 scalar 专项用例：float32/float64/int32/uint8/bool 正例（plan `outputs` kind、
  无 buffer 执行返回原始标量、report 自包含、JSON 重放）；NaN/±inf canonical；多字面量/
  缺 dtype/非法 dtype/`10**400`/buffer/重签名 kind+shape 篡改全部结构化拒绝；
  region 往返（scalar 结果作 operand 与 region exit，`validate_region_plan(re_resolve=True)`）。

### 7.2 零默认变化（contract / capability / plan / artifact / output digest）

`scripts/record_batch3_zero_drift.py --mode=frozen` 在父树与本树各跑一次，覆盖
**16 个既有契约**与 **17 个 frozen run**（matmul f32/bf16、qmatmul、batch1 8 个 op、
batch2 6 个 op，其中 mul 用 scalar 操作数）：

- 默认环境（capability digest `e0a9dac8…`）：
  `raw/frozen_record_parent_9264deb93.json` vs `raw/frozen_record_batch3.json`；
- local 锁 6 线程环境（capability digest `70d8cf05…`）：
  `raw/frozen_record_parent_9264deb93_lockenv.json` vs `raw/frozen_record_batch3_lockenv.json`；
- `scripts/compare_frozen_records.py` 两两比较均为 **ZERO-DRIFT**（对比按批切片口径
  排除 registry 清单一项；本切片的 frozen 记录本身不写 `registered_opcodes`，因此实际比较的
  是完整文档）：`raw/zero_drift_summary.txt`、`raw/zero_drift_summary_lockenv.txt`。
- 按 opcode 收窄的 capability 视图 digest：batch1（`exp`）`77714e28…`、batch2（`add`）
  `67cdb3f8…` 与父树一致（contract 集合分别 10/16 个）；batch3 的 24-contract 视图
  （`neg`/`concat`/`constant`）digest 记录在 `raw/batch3_record.json`，round-2 因
  `reduce_mean`/`constant` 契约修订（含 rank-0 scalar）而更新
  （`neg 786ce38e…`、`concat 79007d33…`、`constant 8ad6887c…`）。
- round-2 修订只在新增契约内部发生；16 个既有契约与 17 个 frozen run 在 round-2 树上
  重新录制后仍是 **ZERO-DRIFT**（本小节记录为 round-2 结果）。

### 7.3 batch3 记录（新契约执行证据）

`raw/batch3_record.json`：8 个新契约 digest + **12 个执行 run**（reduce_mean 浮点/整数、
concat、constant 浮点/整数 tensor、**constant rank-0 scalar float32/int64（round-2）**、
silu、neg、sub 广播、sigmoid、softplus）的 plan/artifact/output digest、`output_kind`、
precision class、output shape/dtype、layout 决策；其中浮点 reduce_mean 为
`deterministic_bounded`、整数 reduce_mean 为 `exact`、concat 为 `exact`、constant 整数
exact / 浮点 bounded。该记录在 round-2 commit 上重新生成，契约 digest 与 §1 修订后表一致。

### 7.4 规则 8 全量（持 local 锁）

见 §8.4。

## 8. 主验收指标（真实 Qwen3.5 三图，口径与 0021 §5 / 0022 §8 一致）

证据：`raw/qwen_region_missing_contracts.json`（脚本
`scripts/qwen_region_missing_contracts_batch3.py`，对三个 builder 的整 program 各跑一次
`plan_region`，read-only、不执行 provider；经 `run_local_heavy.sh` 的 local 锁）。

### 8.1 逐图分布（batch2 → batch3）

| builder | ops | batch2 缺失 contract（次数） | batch3 缺失 contract（次数） | batch3 unplannable |
|---|---|---|---|---|
| `build_attention_kv_cache` | 20 | concat 2, gather 2, div 1, sub 1 | **div 1, gather 2** | 0 |
| `build_qwen35_gdr_recurrent_state` | 123 | sub 4, identity 2, concat 1 | **identity 2** | 0 |
| `build_qwen35_text_decoder_graph` | 4550 | reduce_mean 79, concat 66, constant 66, silu 60, neg 30, sigmoid 24, split 24, sub 24, softplus 18, div 6, embedding 1 | **split 24, div 6, embedding 1** | **0**（round-2 后 constant 可规划） |
| **合计** | **4693** | **411** | **36** | **0** |

### 8.2 缺失 contract 合并表（13 → 5 类）

| opcode | batch2 缺失次数 | batch3 缺失次数 | 覆盖图 | 结论 |
|---|---|---|---|---|
| `reduce_mean` | 79 | **0** | decoder | 消失（新增契约；79/79 plan 成功） |
| `concat` | 69 | **0** | decoder + gdr + attention | 消失（69/69 plan 成功） |
| `constant` | 66 | **0** | decoder | 消失（66/66 plan 成功；rank-0 scalar-kind 结果，round-2 关闭 ERR-0016） |
| `silu` | 60 | **0** | decoder | 消失（60/60 plan 成功） |
| `neg` | 30 | **0** | decoder | 消失（30/30 plan 成功） |
| `sub` | 29 | **0** | decoder + gdr + attention | 消失（29/29 plan 成功） |
| `sigmoid` | 24 | **0** | decoder | 消失（24/24 plan 成功） |
| `softplus` | 18 | **0** | decoder | 消失（18/18 plan 成功） |
| `split` | 24 | **24** | decoder | 不变（多输出 ABI 未定义，Tier 3） |
| `div` | 7 | **7** | decoder + attention | 不变（Tier 3） |
| `gather` | 2 | **2** | attention | 不变（Tier 3） |
| `identity` | 2 | **2** | gdr | 不变（Tier 3） |
| `embedding` | 1 | **1** | decoder | 不变（Tier 3） |
| **合计** | **411 / 13 类** | **36 / 5 类** | — | **-8 类 / -375 次** |

### 8.3 已登记 op 的规划统计（三图合并）

`registered_planned_total`（region 入口对每个 op 都调用 `plan_operation`，不是首错即停）
从 batch2 口径的 **4282 → 4657**；新增的 8 个 opcode 供 375 次请求全部 plan 成功：

| opcode | 真实图出现 | plan 成功 | 缺失 contract | unplannable（原因） |
|---|---|---|---|---|
| `reduce_mean` | 79 | 79 | 0 | 0 |
| `concat` | 69 | 69 | 0 | 0 |
| `constant` | 66 | **66** | 0 | **0**（rank-0 scalar-kind 结果，round-2） |
| `silu` | 60 | 60 | 0 | 0 |
| `neg` | 30 | 30 | 0 | 0 |
| `sub` | 29 | 29 | 0 | 0 |
| `sigmoid` | 24 | 24 | 0 | 0 |
| `softplus` | 18 | 18 | 0 | 0 |
| **合计** | **375** | **375** | **0** | **0** |

三个图在 batch3 后仍是**结构化拒绝**（`region_contracts_missing`），因为 Tier 3 的
`split/div/gather/identity/embedding` 尚未注册——这是**预期**，不是回归。

### 8.4 规则 8 全量（round-2，持 local 锁）

- round-2 的 region 验收与规则 8 全量在**同一次 local 锁持有内**顺序执行（combined runner）：
  2026-09-20 01:23:55 取得锁 → region（rc 0）→ 01:46:02 完成（`logs/combined_timing.txt`、
  `logs/combined_lock_attempts.txt`；此前随其他 agent 排队，75 → sleep 300 重试，未绕过）；
- 命令：`env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut python3 -m pytest -q
  -rs -p no:cacheprovider python/tests/ut/pypto_x`，经 `run_local_heavy.sh` 的 local 锁、
  6 线程（systemd-run scope + 动态 cgroup）；
- collect（同一选择集）：**2586 tests**（`logs/rule8_collect.log`、`logs/rule8_collect.txt`）；
- 结果：**2574 passed / 12 skipped / 0 failed / 0 errors，rc 0，耗时 1297.79 s（21:37）**
  （`logs/rule8_full_single.log`、`logs/rule8_single_summary.txt`、`logs/rule8_rc.txt`、
  `raw/rule8_single_counts.json`）；
- 12 个 skip 分解：7 个 CUDA driver 不在本机（`test_cuda_qwen_c2.py`）+ 5 个 host-probe 守卫的
  frozen digest 钉（batch1 ×2、batch2 ×1、batch3 ×1、U5 region ×1）；均与既有口径一致；
- **全量尝试史（诚实附注）**：round-1（`a2373408d` 树，2575 collect）第一次**完成**的全量
  为 2561 passed / 2 failed / 12 skipped，两个失败都在 `test_op_bench_framework.py`
  （注入仍用刚注册的 `sub` 作为“未注册 opcode”示例）；按原注释语义改成仍属 Tier 3 的
  `div` 后 rc 0。round-2 在 `6ee250569` 上一次性 rc 0（本小节数字）。round-1 的
  attempt-1 日志保留为 `logs/rule8_full_single_attempt1_2failed.log`、
  `logs/rule8_single_summary_attempt1_2failed.txt`、`logs/rule8_rc_attempt1.txt`。



## 9. 未覆盖清单与 Tier 3 建议

### 9.1 本切片未覆盖（结构或入口边界，均结构化拒绝，不静默降级）

- **`split`（24 次）**：多输出 opcode；当前执行契约与 region schema 都以单输出为前提
  （region 预检 `multi_output_not_supported_by_execution_contract`）。需要先定义**多输出 ABI**
  （plan/report 的 outputs 列表、输出缓冲/别名义务、region 边界签名与 digest preimage 扩展），
  不能只注册一个单输出契约。
- **`div`（7 次）**：整数除法语义（截断/floor/除零）需要先冻结；float-only 口径与
  `reduce_mean` 的整数路径决定可复用。
- **`gather`/`embedding`（3 次）**：index dtype 域（int32/int64）、负 index、越界 fail-closed、
  输出 shape 规则与别名义务需要冻结。
- **`identity`（2 次）**：语义是 no-op copy；需要决定 `may_alias`（真视图）还是
  `must_not_alias`（物化），与 batch2 的 `view_mode` 三态一致。
- **`constant` scalar-kind 输出（round-2 已关闭）**：真实 decoder 图的 66 个 `constant` 输出是
  Core IR **scalar kind**；round-1 契约只允许 rank 1..4，因此 66 次全部 unplannable
  （`rank_out_of_range` 契约缺口，ERR-0016）。round-2 允许 rank-0 scalar-kind 结果
  （§4.1、§11.6），region round-5 已能表达 scalar 结果，66 次现全部可规划（§8）。
- **vector/vendor 路径**：8 个契约 `allowed_providers=("portable",)`；AVX2/AVX512/SVE
  与 vendor 候选结构化拒绝（`provider_not_allowed_by_contract`），原生实现不在本切片。
- **动态 shape / 多输出 / 其他非 tensor 结果**：动态 shape 与多输出结构化拒绝；
  `constant` 的 rank-0 scalar 结果已支持（round-2），其他 value kind 结构化拒绝；
  `constant` 不推断 dtype、不做隐式填充。

### 9.2 Tier 3 建议（按 0021 §5 剩余频次）

1. **`split`（24）**：先写多输出 ABI 提案（region schema 升版或加性字段 + `outputs[]` 绑定、
   输出别名义务、`getitem` 取用语义），再实现；这是唯一需要 schema 决定的 Tier 3 opcode。
   父方已在 `0024-2026-09-19-multi-output-abi-proposal` 给出 PROPOSAL，本切片与其口径一致
   （split 仍需独立切片实现，不在本切片范围）。
2. **`div`（7）**：float-only 先行（与 rsqrt/激活同族 `deterministic_bounded`），整数除法
   需冻结一条与 `reduce_mean` 向零截断一致的规则；除零必须结构化拒绝。
3. **`gather`/`embedding`（3）**：先冻结 index dtype（int32/int64；boolean/浮点拒绝）、
   负 index 语义（拒绝或规范化）、越界 fail-closed；`embedding` 是 axis-0 特化。
4. **`identity`（2）**：与 `view` 一起决定 `may_alias` 与物化口径；若沿用 batch2 视图三态，
   需明确 portable trunk 无 stride ABI 下的 `require` 行为。
5. **`constant` scalar 结果 / 零输入 scalar ABI**：已在本切片 round-2 落地（rank-0
   scalar-kind 结果 + additive `request.outputs`），不再属于 Tier 3。

## 10. 已知边界

- 标量操作数只在 `sub`（及 `where`/batch2 binary）显式允许处出现；`neg`/激活/`reduce_mean`/
  `concat`/`constant` 不接受 scalar 操作数（`constant` 是零输入，字面量在 attributes 内）。
- `reduce_mean` 的整数路径只在 portable scalar trunk 可用（`allow_integer_reduce_mean` 专属）；
  向量后端（AVX2/AVX512/SVE）保持 float-only 准入，且其既有浮点 mean 数值实现不在本执行契约
  的范围内（contract 只承诺 portable 参考的“flat 升序 f64 累加 + 一次除法 + 一次 RNE cast”）；
  本切片未做跨后端数值一致性声明。
- `constant` 的 rank-0 结果固定为 **scalar-kind**（`output_shape=[]`、单字面量、
  `request.outputs[0].kind="scalar"`）；不提供 rank-0 tensor 拼写。tensor 结果 rank 1..4，
  单元素 tensor 用 `(1,)` 表示。
- `concat` 的 `max_arity=64` 是本切片冻结的静态规划上限（真实图最大 8），不是运行时能力上限。
- 性能未测；region 规划耗时（decoder 整图约数十秒）只是本机 UNGATED 观测，不作为性能声明。
- 本切片改了 6 个既有测试文件中的 registry 清单/“未注册 opcode”示例断言（`sub`/`concat`/`neg`
  已注册，示例改用仍属 Tier 3 的 `div`/`gather`/`identity`）、op-bench selftest 的同名注入
  示例（`python/pypto/op_bench/runner.py`），以及 region 校验器的可变 arity 分支；
  round-2 还更新了 U5 round-5 `test_execution_region_planning.py` 中一条“constant 是契约缺口”
  的断言（缺口已关闭，改为 scalar 结果 plan/region 成功），并新增 manifest/plan/entry/
  providers/resolver 的 scalar-kind 加性路径。16 个既有契约的 digest、默认/按 opcode
  capability 视图与 17 个 frozen run 的 plan/artifact/output digest 在 round-2 树重新录制后
  仍逐字节不变（§7.2）。

## 11. round-2 契约修订记录（0051-A，commit `5c1d8dd4d`）

验收方 `3cb31311` 中期发现两处“契约文本与行为不一致”，U5 round-5 又用最小复现定位了
`constant` 的 rank-0 scalar 契约缺口；本切片以**单个 follow-up commit**（父提交
`6ee250569`）一并修复 A/B/C/D；按契约修订纪律逐项登记。

### 11.1 A：`reduce_mean` 整数示例更正（行为不变，digest 变更）

- 旧契约文字 `semantics.integer_path` 示例 `mean([-4,2]) = -2` 与同一句的“向零截断”规则
  及 `§4.2` 的 `-1` 矛盾；实现与测试一直是 **-1**（规则正确，示例笔误）。
- 修正：示例改为 `mean([-4, 2]) = -1`；未改任何验证/执行逻辑。
- digest 对照：`ec7b7a62d8e5b2c0f7e752980a8d0ab33fbcb9e48779029a230c0151498b1cca`
  → **`3ab5ee5aaec518b3af33fb175fc68b144a3b3faa730c073a0d9866ed82acf9c4`**。
- 影响面：`reduce_mean` 的按 opcode capability 视图属于 24-contract 集合，`neg`/`concat`/
  `constant` 三者的 scoped manifest digest 随之更新；加入 §11.6 的 rank-0 扩展后，最终值为
  `neg 786ce38e…` / `concat 79007d33…` / `constant 8ad6887c…`（行为不变）。

### 11.2 B：`constant` float 字面量的 binary64 可解码性（行为收紧，digest 变更）

- 旧行为：整数字面量 `10**400` 在契约期被接受，执行期 `_cast_value` 转换溢出 →
  `operand_value_rejected`；与契约文字“f32/bf16 有限溢出饱和 signed inf”不一致。
- 选择方案一（**契约期拒绝**，父方首选）：float dtype 的整数字面量必须可解码进 finite
  binary64（`float(value)` 不溢出），否则结构化拒绝
  **`constant_literal_not_representable`**（field `operation.attributes.value[i]`）；
  文本同步改为“字面量必须可解码为 binary64；超范围在契约期拒绝”，并覆盖 int/float 两侧：
  Python 浮点 `1e400` 在请求构造时已是 `+inf`，按规范拼写 `inf` 处理；`10**308` 在范围内
  接受（int→binary64 舍入；f32 目标随后饱和 signed inf）；`10**309`/`10**400` 契约期拒绝。
- 整数 dtype 路径不变（仍 `constant_value_out_of_range`）。
- digest 对照（含 §11.6 的 rank-0 扩展后的最终值）：`9717b6c33ece2f3ca795e577350c975b0f873509389869658fe62b5b4e8bab0d`
  （round-1）→ `6d1738176b2c7a420e30378e2910307d5df287bc667e6ce0f27984eccafa009d`
  （A/B/C 文字修订中间值）→ **`cf33e1a612ffd4f0b526247a976e56acd6052a4abe08fc2ff9b286c06e73a4d8`**
  （加入 rank-0 scalar-kind 输出后，见 §11.6）。

### 11.3 C1：`reduce_mean.axes` 无序容器错误码口径订正（仅文档）

- 实测 `axes={1}` / `axis={1}` → `reduction_axis_invalid`（`_verify_reduction` 的既有轴校验
  路径），不是 `unordered_attribute_container`；batch1 的该错误码是冻结行为，不在本切片改码。
- 订正 `§1` 与 `§6` 表：`constant` 的容器报 `unordered_attribute_container`，`reduce_mean`
  的轴容器报 `reduction_axis_invalid`（文档与行为一致）。

### 11.4 C2：`constant` NaN 位型口径统一（仅文档 + 位型用例）

- 旧 `§3` 写“`concat`/`constant` 原样保留 NaN 位型”，与 `§4.1` 的规范拼写实现冲突：
  `constant` 的 NaN 字面量冻结为 `"nan"`，执行物化为 canonical quiet **正** NaN
  （符号位/payload 不保留）。
- 订正 `§3`/`§4.1`/契约 `semantics.nan`：`concat` 是复制类、NaN 位型原样搬运；
  `constant` 的非 NaN 位型全保持，NaN 走规范拼写往返。新增位型用例：负 quiet NaN
  `0xFFC00000` → plan 属性 `"nan"` → 输出 `0x7FC00000`；同一请求的 `-0.0`、次正规、
  `+inf` 保持。
- 该文字订正与 B 的文本修改一起进入 `constant` 的新 digest（见 §11.2）。

### 11.5 修订后的验证

- 新增/更新测试后：batch3 单文件 **145 passed**；focused 16 套件 **832 passed / 0 failed**
  （`logs/focused_execution_suites_round2.log`）。
- 零默认变化：16 个既有契约与 17 个 frozen run 在 round-2 树上重跑仍 **ZERO-DRIFT**
  （默认 + OMP=6；`raw/zero_drift_summary*.txt` 为 round-2 结果）。
- 主验收（同一次 local 锁内）：缺失契约 **5 类/36 次**、`unplannable` **0**、
  `registered_planned_total` **4657**（§8）。
- 全量规则 8（combined 锁内，2026-09-20 01:23:55–01:46:02）：collect **2586**、
  **passed 2574 / skipped 12 / failed 0 / rc 0**（`logs/rule8_full_single.log`、
  `raw/rule8_single_counts.json`）。

### 11.6 D：`constant` rank-0 scalar-kind 输出（范围扩展，关闭 ERR-0016）

- **背景与归因订正**：U5 round-5 的最小复现证明 66 个 decoder `constant` 的 unplannable 是
  **契约缺口**（旧契约只允许 rank 1..4；`output_shape=[]` → `rank_out_of_range`），不是
  region 入口边界。round-1 的 `§0` 归因错误，已在 `§0` 勘误框和本记录订正（登记 ERR-0016）。
- **契约**：`constant` 输出 rank 0..4；rank 0（`output_shape=[]`）是 scalar-kind 单字面量
  结果，rank 1..4 是 tensor 结果；无 rank-0 tensor 拼写。verifier 返回 `output_kind`，
  `verify_generic_operation` 只在 output spec `rank_min==0` 时接受 scalar declared kind。
- **plan**：非 tensor 结果的 `request` 新增可选
  `outputs=[{name:"output", role:"output", kind:"scalar", dtype, shape:[]}]`（tensor plan
  逐字节不变）；`ExecutionPlan.output_kind` 读取该块；v3 校验器交叉检查
  `output_shape`/`output_dtype`/单输出 arity。
- **执行链**：provider 以 `CoreType.scalar` 建程序、`LaunchRequest(outputs=[])`，
  `CpuScalarRuntime.launch` 返回 scalar；`execute_operation` 返回原始标量，scalar plan
  携带 output buffer → `unsupported_output_handle`；v4 report 的 request echo 带 `outputs`，
  内嵌 plan payload 作为 ground truth。
- **capability / region**：`ProviderCapability.rank_range` 下限放宽为 0（`constant` 视图
  `(0,4)`）；region 侧沿用 round-5（未改 `region.py`），scalar 结果可作 operand/exit。
- **负例与锚定**：多字面量 → `constant_element_count_mismatch`；缺 dtype →
  `request_dtype_required`；非法 dtype → `request_dtype_invalid`；`10**400` →
  `constant_literal_not_representable`；不重签 kind/shape 篡改 → `plan_digest_mismatch`；
  重签名后重新解析 canonical request 比对 → `plan_declaration_mismatch`。
- **digest 对照**：`constant` `6d173817…` → **`cf33e1a612ffd4f0b526247a976e56acd6052a4abe08fc2ff9b286c06e73a4d8`**；
  scoped manifest `neg`/`concat`/`constant` → `786ce38e…` / `79007d33…` / `8ad6887c…`。
- **主验收（§8，round-2 树，combined 锁内运行）**：`unplannable` 66 → **0**；
  `registered_planned_total` 4591 → **4657**；缺失契约仍 5 类/36 次。


