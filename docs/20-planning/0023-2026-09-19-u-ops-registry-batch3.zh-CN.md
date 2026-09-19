# 0023 — U 线算子注册第三批（batch3）：8 个 Tier-2 opcode 契约 + 零默认变化 + 主验收指标 13→5

- 任务：`u-ops-registry-batch3`（batch 0051 切片 A）
- 实现基线：integration tip `9264deb93`（tree `ec93dfb19a9392cd930f787f1441f4a30800547c`，冻结）
- 分支：`work/u-ops-registry-batch3`（未 push）
- 实现 commit：`403633d97`（所有证据均在实现树上重跑；零默认变化对照含父树 `9264deb93`）
- 依赖：`0019`（batch1 的 schema v3 契约、通用入口、按 opcode 收窄的 capability 视图）、`0022`
  （batch2 的视图语义冻结与主验收指标口径）、`0021`（U5 只读 region 与 §5 真实图缺失契约频次表）
- 结论等级：**实现完成（静态 shape；portable provider 路径）**。未做性能声明；本机数字一律 **UNGATED**
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u-ops-registry-batch3/`

## 0. 摘要

本切片把执行层 opcode 契约从 16 个扩到 24 个，新增 **Tier 2 的 8 个 opcode**：

`reduce_mean`、`concat`、`constant`、`silu`、`neg`、`sub`、`sigmoid`、`softplus`

主验收指标（§8）：在 `9264deb93` 上用 U5 `plan_region` 对三个真实 Qwen3.5 图各跑一次，
缺失契约从 **13 类降到 5 类**（只剩 `split/div/gather/identity/embedding`），
8 个 opcode 的缺失次数全部归零；已登记契约的成功规划数从 **4282 升到 4591**，
另有 66 个 `constant` 因真实图输出是 **scalar kind** 而被 U5 region 预检列为 unplannable
（见 §8.3，属 U5 入口边界，不是契约缺口）。

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
6. 不修改 `pypto/__init__.py`、`python/pypto/portable/qwen35.py`，不新增环境变量。

## 1. 八个新契约的逐算子摘要

所有 8 个契约：rank 域（张量操作数）1..4（`constant` 输出 rank 1..4；`reduce_mean` 输出
rank 0..4）；`attributes` 白名单之外一律 `unknown_attribute`；序列型属性只接受有序
list/tuple（set/generator/mapping → `unordered_attribute_container`）。

| opcode | 元数/操作数 | shape 规则 | dtype 域 | 结果 dtype | 属性白名单 | portable executor 符号 |
|---|---|---|---|---|---|---|
| `reduce_mean` | 1（input） | rank(in) 1..4，rank(out) 0..4；`shape(out)=reduced_shape(in, axes, keepdims)`；**任一被约简轴长度为 0 → 契约拒绝**（mean 无单位元） | 全部 portable dtype 去 bool | = input dtype | `axes`/`axis`（同现必须归一化一致）、`keepdims`(bool) | `pypto.backends.cpu.runtime._reduce_mean_value` |
| `concat` | 1..64（variable arity） | 所有输入同 rank 1..4、同 dtype；非 `axis` 维必须逐维相等；`shape(out)[axis]=Σ shape(input_i)[axis]` | 全部 portable dtype（含 bool） | = 输入共同 dtype | `axis`/`dim`（默认 0，负值归一化；同现必须一致） | `pypto.backends.cpu.runtime._concat_value` |
| `constant` | 0（无输入） | 零输入；**显式声明** rank 1..4 输出 shape；`element_count(output_shape) == 字面量个数`（无隐式填充/广播） | 输出 ∈ 全部 portable dtype | 显式声明的 output dtype（**绝不从字面量推断**） | `value`/`values`（恰一；标量或有序嵌套 list/tuple） | `pypto.backends.cpu.runtime._constant_value` |
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
constant   sha256:9717b6c33ece2f3ca795e577350c975b0f873509389869658fe62b5b4e8bab0d
neg        sha256:016922779127b2b82cc9cc93dcc7058b90e937410663028f69efb53546b377f3
reduce_mean sha256:ec7b7a62d8e5b2c0f7e752980a8d0ab33fbcb9e48779029a230c0151498b1cca
sigmoid    sha256:abee314e247ed9aa7c59eaf9059a761c5ee83b4679699acfe99f5e87e05fff32
silu       sha256:50dede82914e16840b9e34f3107724fcc6342ff38651d46acd295d380e1f0162
softplus   sha256:ad362695e1498edd07b9647f77c2c9c2aa7f254c79a88d0a9fc6370b53b5f8da
sub        sha256:0a37e2a3da1c3c479437a9ae338a9211bc6ac17a10b504fe3cd6909410e63aa5
```

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
| `constant` 浮点 | `deterministic_bounded` | 同上 | 字面量到目标 dtype 一次 RNE cast；**无解析上界**（f32/bf16 有限溢出饱和为 signed inf；f16 有限溢出 fail-closed） |

`numeric.require_proven_deviation_bound = true` 时：

- 整数/布尔 exact 路径（`neg`/`sub`/`reduce_mean`/`constant` 的整数路径、`concat` 全部）
  正常解析；
- 所有浮点 bounded 路径（`neg/sub` 浮点、`reduce_mean` 浮点、三个激活、`constant` 浮点）
  **fail-closed**：`NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable`，
  portable provider 也生效。

## 3. 语义边界（契约文字 + 用例）

- **NaN**：`neg` 的 `-NaN` 为 NaN（符号位/payload 不属于 portable 值契约）；`sub` IEEE-754
  传播；`silu/sigmoid/softplus(NaN)=NaN`；`concat`/`constant` 原样保留 NaN 位型；
  `reduce_mean` NaN 传染；+inf + -inf = NaN。`constant` 的 NaN 字面量冻结为 `"nan"`。
- **±0**：`neg(-0.0)=+0.0`、`neg(+0.0)=-0.0`；`silu(+0.0)=+0.0`、`silu(-0.0)=-0.0`
  （冻结的 `x>=0` 分支对 `-0.0` 成立）；`softplus(±0.0)=ln 2`；`sigmoid(±0.0)=0.5`；
  `sub` 按 binary64 规则；`reduce_mean` 从整数 0 起加，全 `-0.0` 组得到 `+0.0`；
  `concat`/`constant` 原样保留符号位。
- **±inf**：`silu(+inf)=+inf`、`silu(-inf)=+0.0`（显式分支）；`softplus(+inf)=+inf`、
  `softplus(-inf)=+0.0`；`sigmoid(+inf)=1.0`、`sigmoid(-inf)=0.0`；`neg` 翻转符号；
  f32/bf16 溢出饱和为 signed inf，f16 有限溢出 fail-closed。
- **次正规**：`concat`/`constant`/`neg` 原样保留；激活在 binary64 中求值后按目标格式 RNE
  （可下溢为次正规或 0）；`reduce_mean` 的 binary64 累加保留次正规、最后一次 cast 决定
  目标格式；用例含最小正 float32 次正规 `0x1p-149`。
- **bf16 位型**：`concat`/`neg`/`constant` 的 bf16 值逐位保留；激活/mean 按 binary32 位型
  做 RNE（NaN payload 不承诺保留）。用例覆盖 subnormal `0x0001`、`-inf 0xFF80`、
  NaN `0x7FC0` 与 RNE tie 边界。
- **整数回绕**：所有整数路径**不回绕、不饱和**。`sub` 越界、`neg` 最小负值取负、
  `constant` 越界整数/浮点-整数截断、`reduce_mean` 越界商，全部 fail-closed
  （执行错误时输出缓冲保持原值；执行期结构化 reason `operand_value_rejected`）。
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
- **shape 规则**：输出必须是 rank 1..4 的 tensor（rank-0 结果不由执行 ABI 表示，与 batch2 的
  0-d 拒绝口径一致）；`output_shape` 必填；字面量个数必须**恰好等于** `element_count(output_shape)`
  （无填充/广播），否则 `constant_element_count_mismatch`。
- 非有限浮点字面量冻结为 `nan`/`inf`/`-inf`；`plan.to_json()` 往返、region replay、执行都使用
  同一规范形态（进 plan digest）。
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
| rank 越界（输入 rank>4；`constant` rank 0 或 >4） | `rank_out_of_range` | verifier |
| 广播不合法（`sub`） | `broadcast_shape_mismatch` | verifier（复用 lowering） |
| 未知属性 | `unknown_attribute` | verifier（白名单） |
| 无序容器属性/字面量（set/generator/mapping） | `unordered_attribute_container` | verifier |
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
| 输出别名输入 | `output_input_alias` | provider |
| plan 被篡改 | `ARTIFACT_MISMATCH / plan_digest_mismatch` 等 | plan 校验器 |

## 7. 验证与证据

### 7.1 新增测试（`python/tests/ut/pypto_x/test_execution_ops_registry_batch3.py`，138 项）

覆盖：registry/inventory、8 个契约的 dtype×rank×shape×attribute 矩阵、差分矩阵
（NaN/±0/±inf/次正规/bf16 位型/整数极值/广播/空张量/零长度轴）、`constant` value 域、
`reduce_mean` 浮点 f64 参考与整数向零截断、plan/report 自包含、冻结 plan JSON 重放、
篡改拒绝、8 opcode 的 region 规划（`plan_region` + `validate_region_plan(re_resolve=True)`）、
precision class 与 `require_proven_deviation_bound` 行为。

- focused（未持锁，默认 12 线程）：本切片 138 passed；与 batch1/batch2/region/discovery/
  matmul/view_mode/binding/report/cpu-scalar 相关 15 个套件合并 → **789 passed / 0 failed**
  （约 17 s；`logs/focused_execution_suites.log`）。
- 父提交非空转证据：新测试文件在父树 `9264deb93` 上 collect 即
  `ImportError: cannot import name '_constant_value' from 'pypto.backends.cpu.runtime'`，rc=2
  （`logs/parent_commit_new_tests_error.log`）。
- 差分口径：执行层 `execute_operation` 输出 vs 契约声明的 portable 参考 helper
  （`_reduce_mean_value` / `_concat_value` / `_constant_value` / `_neg_value` /
  `_math_value` / `_binary_value`）逐位比较（NaN 按 NaN、±0 按符号位）。

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
  `67cdb3f8…` 与父树一致（contract 集合分别 10/16 个）；batch3（`neg`/`concat`/`constant`）
  为新增的 24-contract 视图，digest 记录在 `raw/batch3_record.json`。

### 7.3 batch3 记录（新契约执行证据）

`raw/batch3_record.json`：8 个新契约 digest + 10 个执行 run（reduce_mean 浮点/整数、
concat、constant 浮点/整数、silu、neg、sub 广播、sigmoid、softplus）的
plan/artifact/output digest、precision class、output shape/dtype、layout 决策；
其中浮点 reduce_mean 为 `deterministic_bounded`、整数 reduce_mean 为 `exact`、
concat 为 `exact`、constant 整数 exact / 浮点 bounded。

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
| `build_qwen35_text_decoder_graph` | 4550 | reduce_mean 79, concat 66, constant 66, silu 60, neg 30, sigmoid 24, split 24, sub 24, softplus 18, div 6, embedding 1 | **split 24, div 6, embedding 1** | `constant` 66（输出是 scalar kind，region 预检拒绝） |
| **合计** | **4693** | **411** | **36** | **66** |

### 8.2 缺失 contract 合并表（13 → 5 类）

| opcode | batch2 缺失次数 | batch3 缺失次数 | 覆盖图 | 结论 |
|---|---|---|---|---|
| `reduce_mean` | 79 | **0** | decoder | 消失（新增契约；79/79 plan 成功） |
| `concat` | 69 | **0** | decoder + gdr + attention | 消失（69/69 plan 成功） |
| `constant` | 66 | **0** | decoder | 消失（注册成功；66 次因 scalar kind 进 unplannable，见 §8.3） |
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
从 batch2 口径的 **4282 → 4591**；新增的 8 个 opcode 供 375 次请求：

| opcode | 真实图出现 | plan 成功 | 缺失 contract | unplannable（原因） |
|---|---|---|---|---|
| `reduce_mean` | 79 | 79 | 0 | 0 |
| `concat` | 69 | 69 | 0 | 0 |
| `constant` | 66 | **0** | 0 | **66（`non_tensor_result_not_supported_by_execution_contract`）** |
| `silu` | 60 | 60 | 0 | 0 |
| `neg` | 30 | 30 | 0 | 0 |
| `sub` | 29 | 29 | 0 | 0 |
| `sigmoid` | 24 | 24 | 0 | 0 |
| `softplus` | 18 | 18 | 0 | 0 |
| **合计** | **375** | **309** | **0** | **66** |

三个图在 batch3 后仍是**结构化拒绝**（`region_contracts_missing`），因为 Tier 3 的
`split/div/gather/identity/embedding` 尚未注册——这是**预期**，不是回归。

### 8.4 规则 8 全量（持 local 锁）

- 命令：`env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut python3 -m pytest -q
  -rs -p no:cacheprovider python/tests/ut/pypto_x`，经 `run_local_heavy.sh` 的 local 锁、
  6 线程（systemd-run scope + 动态 cgroup；未绕过）；
- collect（同一选择集）：**2575 tests**（`logs/rule8_collect.log`、`logs/rule8_collect.txt`）；
- 结果：**2563 passed / 12 skipped / 0 failed / 0 errors，rc 0，耗时 1250.12 s（20:50）**
  （`logs/rule8_full_single.log`、`logs/rule8_single_summary.txt`、`logs/rule8_rc.txt`、
  `raw/rule8_single_counts.json`；2026-09-19 22:38:23–22:59:17）；
- 12 个 skip 分解：7 个 CUDA driver 不在本机（`test_cuda_qwen_c2.py`）+ 4 个 host-probe 守卫的
  frozen digest 钉（batch1 ×2、batch2 ×1、batch3 ×1）+ 1 个 U5 region 单算子基线
  host-probe 守卫；均与 batch1/batch2/U5 的既有口径一致；
- **诚实附注（全量尝试史）**：22:01:11 启动的一次全量在 22:10:41 被外部 supervisor SIGTERM
  中断（无指标数据，`rule8_lock_attempts.txt` 可见三次启动）；第一次**完成**的全量
  （22:15:17–22:36:11）为 **2561 passed / 2 failed / 12 skipped**，两个失败都在
  `test_op_bench_framework.py`：注入仍用 `sub` 作为“未注册 opcode”示例，而 batch3 已注册
  `sub`（`test_unregistered_opcode_fails_closed`、`test_selftest_all_error_injections_pass`）。
  按原注释语义把注入改为仍属 Tier 3 的 `div`（`op_bench/runner.py` + 同名测试）后，
  第二次完成的全量 rc 0；attempt-1 日志保留为
  `logs/rule8_full_single_attempt1_2failed.log`、`logs/rule8_single_summary_attempt1_2failed.txt`、
  `logs/rule8_rc_attempt1.txt`。归因：两个失败都是本切片注册表增长的直接后果
  （注入示例选用了刚注册的 `sub`），不是数值/语义回归；`test_op_bench_framework.py`
  单独复跑 32 passed，22:38 的第二次完成全量为最终规范结果。



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
- **`constant` 的 scalar-kind 输出**：真实 decoder 图的 66 个 `constant` 输出是 Core IR
  **scalar kind**；执行契约只产出 tensor（rank 0..4 的 rank-0 tensor 也不代表 scalar kind），
  U5 region 预检因此把它们列为 `unplannable_operations`（`non_tensor_result_not_supported_
  by_execution_contract`）。缺的是 “零输入 → scalar 结果” 的 region/执行 ABI 决定，不是
  `constant` 契约缺口。
- **vector/vendor 路径**：8 个契约 `allowed_providers=("portable",)`；AVX2/AVX512/SVE
  与 vendor 候选结构化拒绝（`provider_not_allowed_by_contract`），原生实现不在本切片。
- **动态 shape / 多输出 / 非 tensor 结果**：全部结构化拒绝；`constant` 不推断 dtype、
  不做隐式填充。

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
5. **`constant` scalar 结果 / 零输入 scalar ABI**：供后续 region 全图闭环与标量常量图。

## 10. 已知边界

- 标量操作数只在 `sub`（及 `where`/batch2 binary）显式允许处出现；`neg`/激活/`reduce_mean`/
  `concat`/`constant` 不接受 scalar 操作数（`constant` 是零输入，字面量在 attributes 内）。
- `reduce_mean` 的整数路径只在 portable scalar trunk 可用（`allow_integer_reduce_mean` 专属）；
  向量后端（AVX2/AVX512/SVE）保持 float-only 准入，且其既有浮点 mean 数值实现不在本执行契约
  的范围内（contract 只承诺 portable 参考的“flat 升序 f64 累加 + 一次除法 + 一次 RNE cast”）；
  本切片未做跨后端数值一致性声明。
- `constant` 的 rank-0 tensor 输出不在契约内（rank 1..4）；单字面量用 `(1,)` 表示，
  与 `iota` 的 rank 域一致。
- `concat` 的 `max_arity=64` 是本切片冻结的静态规划上限（真实图最大 8），不是运行时能力上限。
- 性能未测；region 规划耗时（decoder 整图约数十秒）只是本机 UNGATED 观测，不作为性能声明。
- 本切片改了 6 个既有测试文件中的 registry 清单/“未注册 opcode”示例断言（`sub`/`concat`/`neg`
  已注册，示例改用仍属 Tier 3 的 `div`/`gather`/`identity`）、op-bench selftest 的同名注入
  示例（`python/pypto/op_bench/runner.py`），以及 region 校验器的可变 arity 分支；
  16 个既有契约的 digest、默认/按 opcode capability 视图与 17 个 frozen run 的
  plan/artifact/output digest 逐字节不变（§7.2）。
