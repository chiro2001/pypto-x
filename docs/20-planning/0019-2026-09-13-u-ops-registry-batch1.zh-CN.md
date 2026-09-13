# 0019 — U 线算子注册第一批（batch1）：8 个 opcode 契约 + 通用 plan/execute

- 任务：`u-ops-registry-batch1`（batch 0048 切片 B）
- 基线：integration tip `923a72e26`
- 分支：`work/u-ops-registry-batch1`
- 结论等级：实现完成（未做性能声明；所有本机数字 UNGATED）
- 范围：只承诺 **portable provider** 执行路径；静态 shape；单算子级；不做 region/图级（U5 后续）

## 0. 摘要

本切片把执行层 opcode 契约从 2 个（`matmul`、`qmatmul_s8s8_s32`）扩到 10 个，新增：

`where`、`compare`、`iota`、`exp`、`reduce_sum`、`reduce_max`、`cast`、`reshape`

新增通用入口 `plan_operation` / `execute_operation`，与 `plan_matmul` / `execute_matmul` **复用同一套
resolver / policy / v4 report 机制**（不是第二套）。默认 policy 下 matmul/qmatmul 的 contract digest、
capability digest、plan digest、artifact digest、输出与基线树逐字节一致（见 §7 证据）。

### 0.1 冻结兼容策略（关键设计决定）

`OpDefinition` 文档中嵌有 `op_registry_schema_version`，该值参与 contract digest；而 contract digest
又进入 plan payload。因此：

- U1/U2 的 `matmul` / `qmatmul_s8s8_s32` 契约文档**冻结在 schema v2**（digest 不变）；
- 本批 8 个新契约文档使用 registry 级 schema **v3**（新增 `precision` / `semantics` /
  `attribute_schema` / `arity` / `input_roles` / `result_dtype_rule` / `shape_rule_id` 块）；
- `doctor` 报告 `op_registry_schema_version = 3`；
- 默认 capability probe 的 portable manifest **仍只声明冻结的两个契约**（否则 capability digest 变化会
  连带改变 matmul/qmatmul 的 plan digest）。batch1 请求走“按 opcode 收窄的 capability 视图”
  `operation_provider_capability(opcode)` / `with_operation_contracts(snapshot, opcode)`，该视图声明全部
  10 个契约并把 dtype/rank 前置条件收窄到本算子。该模式与 0047 exact-blocked 的 opt-in 视图同源。

## 1. 契约表（逐算子）

所有 8 个契约：`contract_version = "1.0"`（opcode 级）、`allowed_providers = ("portable",)`、
`effects = pure`、`aliasing = inputs_readonly_outputs_distinct`（`view_obligation = must_not_alias`）、
rank 域（张量操作数）1..4；标量操作数不在本批范围（lowering 支持，执行层契约暂不承诺）。
`left_shape/right_shape` 在 v3 plan 中只是 operand 0/1 的 v2 兼容投影，权威 operand 列表是
`request.inputs`。

| opcode | 元数/操作数 | shape 规则 | dtype 域 | 结果 dtype | 允许 attributes（白名单） | executor 符号（portable 参考） |
|---|---|---|---|---|---|---|
| `where` | 3（condition, input, other） | cond/input/other rank 1..4；三者在尾轴上广播；`shape(out)=broadcast(...)` | cond=bool；input=other=out∈全部 portable dtype | = input dtype | 无（未知属性拒绝） | `pypto.backends.cpu.runtime._where_value` |
| `compare` | 2（left, right） | rank 1..4；尾轴广播；`shape(out)=broadcast(...)` | left=right∈全部含 bool；out=bool | bool | `predicate`（必填，canonical `eq/ne/lt/le/gt/ge`；别名 `comparison/relation/op` 必须一致） | `pypto.lowering.cpu.compare_values` |
| `iota` | 0 | 无输入；rank(out) 1..4；`value(c)=start+c[axis]*step`（signed int64） | out∈{int32,int64} | 声明 out dtype | `axis`(默认0，归一化)、`start`(默认0)、`step`(默认1)；均在 signed int64 | `pypto.lowering.cpu.iota_values` |
| `exp` | 1 | 逐元素；`shape(out)=shape(in)` | in=out∈{float16,bf16,float32,float64} | = input dtype | 无 | `pypto.backends.cpu.runtime._math_value` |
| `reduce_sum` | 1 | rank(in) 1..4，rank(out) 0..4；`shape(out)=reduced_shape(in, axes, keepdims)` | in=out∈全部非 bool；bool 拒绝 | = input dtype | `axes`/`axis`（二者同现必须归一化后一致）、`keepdims`(bool) | `pypto.backends.cpu.runtime._reduce_value` |
| `reduce_max` | 1 | 同 reduce_sum；**任一被约简轴长度为 0 → 契约拒绝**（空 max 无单位元） | 同 reduce_sum | = input dtype | 同 reduce_sum | `pypto.backends.cpu.runtime._reduce_value` |
| `cast` | 1 | 逐元素；`shape(out)=shape(in)` | in/out∈全部 portable dtype（bool 含） | 声明 target dtype | `dtype/to/target_dtype/out_dtype`（必须一致且等于声明结果）、`mode/cast_mode`（只允许 CAST_NONE）、`satmode/saturation/saturate`（只允许 off）；未知属性拒绝 | `pypto.backends.cpu.runtime._cast_value_for_operation` |
| `reshape` | 1 | rank(in/out) 1..4；`element_count(in)==element_count(out)`；`shape/target_shape/out_shape` 与声明结果一致 | in=out∈全部 portable dtype | = input dtype | `shape/target_shape/out_shape`（一致）、`valid_shape/valid_shapes`（若非空必须等于目标 shape）、`inplace`（只允许 false）；未知属性拒绝 | `pypto.backends.cpu.runtime._reshape_value` |

广播的**精确规则**（与 lowering 的 `_broadcast_shape` 同一实现，执行层直接调用该函数后复核）：
从最后一维向前对齐；每个 offset 上出现的所有维度必须相等，或其中一个为 1；`0` 只与 `0`/`1` 广播
（结果为 0，即 `(0,)` 与 `(1,)` → `(0,)`）；其他组合一律 `broadcast_shape_mismatch` 结构化拒绝。
rank 差按“前缀缺省”处理（短 shape 视作左侧补 1）。

未知属性（Core IR container 本身是宽松的）在契约层一律 `unknown_attribute` 拒绝，绝不静默忽略。

## 2. 精度声明

每个契约文档携带 `precision` 块（`declaration_version=1`，规则按“请求 dtype path”匹配）：

### 2.1 可证明 exact（`deviation_bound_kind = exact_contract_bound`，`deviation_bound_available=true`）

| 路径 | 理由 |
|---|---|
| `iota`（int32/int64） | 序列在 signed int64 内计算，收窄前做范围检查；无浮点路径、无回绕 |
| `compare` 全部路径 | 逐元素 IEEE-754 比较，无算术；整数/布尔/浮点结果都是精确布尔 |
| `where` 的整数/布尔路径 | 选择已声明的元素并原样输出（输出 dtype = 被选 dtype，不做转换） |
| `reduce_sum` / `reduce_max` 整数路径 | sum 为精确 Python 整数后做**带范围检查**的输出 cast（越界 fail-closed）；max 为精确选择 |
| `cast` 的整数↔整数、整数↔布尔 | 精确值映射；越界 fail-closed（无回绕/饱和） |
| `reshape` 全部路径 | 只改变逻辑 shape，flat 值序列逐位保留 |

### 2.2 诚实声明为 `deterministic_bounded`（无已注册的 proven 包络）

| 路径 | 定义域 / 误差口径 |
|---|---|
| `exp` 浮点 | 输入任意 f16/bf16/f32/f64；结果 = 平台 libm 的 binary64 `exp`，随后一次 RNE cast 到输出 dtype。**无解析上界**：libm 误差与 cast 误差都没有注册过的解析包络；观测值绝不当上界 |
| `reduce_sum` 浮点 | aggregate 以整数 0 起、按输入 flat 行主序升序做 binary64 加法，再一次 RNE cast。**无解析上界**：每一步是 RNE f64 加，但组合误差没有注册包络 |
| `reduce_max` 浮点 | 比较选择、无算术；选择本身偏差为 0（非 NaN 输入）；仍按 float 路径声明为 `deterministic_bounded`，因为执行层没有注册 proven 包络 |
| `where` 浮点 | 选择本身逐位精确、偏差 0；仍声明 `deterministic_bounded`（同上，未注册包络） |
| `cast` 任一含浮点路径 | int→float 一次 RNE（int64→f64 精确、int64→f32 可能舍入）；float→int 截断向零 + 范围检查；float→float RNE。**无解析上界** |

`numeric.require_proven_deviation_bound = true` 时：

- exact 路径（其 precision 带 proven `exact_contract_bound`）→ 正常解析；
- float 路径（`deviation_bound_available=false`）→ **fail-closed**，抛
  `NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable`（field
  `numeric.require_proven_deviation_bound`），绝不把确定性执行顺序或实测快照当作 proven bound；
- 该检查对 **portable provider 也生效**（这是与 matmul/qmatmul 的关键差别：后者的 exact 性是冻结
  契约的一部分，仍走原路径，行为与 digest 不变）。

## 3. 语义边界（契约文字 + 用例）

- **NaN**：`where` 选择 NaN 原样输出，未选分支的 NaN 不传播；`compare` 遵循 IEEE-754（NaN 对所有
  `eq/lt/le/gt/ge` 为 false、`ne` 为 true）；`exp(NaN)=NaN`；`reduce_sum` NaN 传染、`+inf + -inf = NaN`；
  `reduce_max` 的 NaN 只在它是该 reduction group 的**第一个**贡献元素时胜出（后续 `value > NaN` 为
  false），否则被忽略——这是顺序相关语义，已显式写入契约；float→int 的 NaN fail-closed；
  `cast` 到 bool 时 NaN→True。
- **±0**：`where`/`reshape` 原样保留符号位；`compare` 中 `-0.0 == +0.0` 为 true、`-0.0 < +0.0` 为 false；
  `reduce_max` 中 ±0 相等，**第一个**元素胜出并保留其符号；`reduce_sum` 以整数 0 起加，`sum([-0.0])`
  结果为 `+0.0`；`cast` float→float 保留 ±0，float→int 的 -0.0 得 0。
- **±inf**：选择/比较按值处理；`exp(+inf)=+inf`、`exp(-inf)=+0.0`；`reduce_sum` +inf/−inf 相加为 NaN；
  float→int 的 inf fail-closed；f32/bf16 目标溢出饱和为 signed inf，**float16 溢出 fail-closed**（不返回 inf）。
- **次正规**：float→float 的 cast 与 `where`/`reshape` 原样保留次正规；`exp` 的极小结果按目标格式 RNE
  下溢为次正规或 0；`reduce_sum` 在 binary64 中保留次正规（最终 cast 按目标格式）。用例含最小正
  float32 次正规 `0x1p-149`。
- **bf16 位型**：bf16 目标按 binary32 位型做 **RNE**（显式 tie-to-even 位实现），NaN 的 payload 不承诺
  保留（HostTensor 用 Python float 存储）；`bf16` 的读入是精确解码，cast bf16→f32 逐位可逆。用例覆盖
  subnormal `0x0001`、`-inf 0xFF80`、NaN `0x7FC0` 与 RNE tie 边界 `0x3F818000 → 0x3F820000`。
- **整数回绕**：所有整数路径**不回绕、不饱和**。`cast` 越界、`reduce_sum` 整数和越界、`iota` 序列越界
  （int64/int32）、`float→int` 越界或非有限值，全部 fail-closed（执行错误，输出缓冲保持原值）。
- **空张量**：`where`/`compare` 的 0 维广播产生 0 元素输出；`exp`/`cast`/`reshape` 0 元素输入 → 0 元素输出
  （reshape 允许任意 0 元素目标 shape，因为 element_count 都是 0）；`iota` 的 0 长度轴产生空张量且不
  计算 start/step；`reduce_sum` 对 0 长度被约简轴返回加性单位元 0；`reduce_max` 对 0 长度轴在契约层拒绝
  （`empty_reduction_has_no_identity`）。
- **原地/别名**：全部 8 个契约 `must_not_alias`，`reshape.inplace=true` 在契约层拒绝；执行期输出与任一
  输入同 buffer → `output_input_alias`。

## 4. 通用入口

```python
from pypto.execution import ExecutionPolicy, plan_operation, execute_operation

# 一元：exp
res = plan_operation("exp", [(2, 3)], ["float32"])
out, report = execute_operation(res.plan, [tensor])      # tensor: HostTensor float32 (2,3)

# 三元广播：where（condition=bool，value dtype 由 dtype 参数给出）
res = plan_operation("where", [(2, 3), (2, 3), (1,)],
                     dtype="float32")                     # input_dtypes 缺省 = (bool, dtype, dtype)
out, report = execute_operation(res.plan, [cond, a, b])

# cast / reshape：结果声明可给在参数或 attributes，二者必须一致
res = plan_operation("cast", [(2, 3)], ["int32"], output_dtype="bf16")
res = plan_operation("reshape", [(2, 3)], ["float32"], attributes={"shape": [6]})

# iota：必须显式 dtype + output_shape（无输入）
res = plan_operation("iota", [], dtype="int32", output_shape=(8,),
                     attributes={"axis": 0, "start": 0, "step": 1})

# reduce：axes 缺省为全部轴，keepdims 缺省 False；输出形状由契约推导
res = plan_operation("reduce_sum", [(2, 3, 4)], ["float32"], attributes={"axes": [1]})
```

签名（实现）：

```text
plan_operation(opcode, input_shapes=(), input_dtypes=None, *,
               dtype=None, output_shape=None, output_dtype=None,
               attributes=None, policy=None, capability=None) -> ResolutionResult

execute_operation(plan, inputs, capability=None, output=None) -> (HostTensor, ExecutionReport)
```

- `plan_operation` 也接受 `matmul` / `qmatmul_s8s8_s32` 拼写，并委托到既有专用入口；
  委托路径与 `plan_matmul` / `plan_qmatmul` 的 plan digest 逐字节一致（测试钉住）。
- 冻结 plan 可用 `ExecutionPlan.from_dict(json.loads(plan.to_json()))` 重载；
  `execute_operation` 只校验、不重解析（declaration == execution）。
- provider 必须实现 `execute_operation`；否则结构化 `PROVIDER_UNAVAILABLE`
  （reason `provider_does_not_support_generic_execution`），并按声明 fallback 链处理，绝不静默换实现。
- 通用 plan 使用 `plan_schema_version = 3`（v2 matmul/qmatmul payload 不变）；v3 校验器强制
  `request.inputs`/`attributes`/`output_dtype` 与 `left/right_shape` 投影一致，并强制 precision 声明
  自洽（exact 必须有 proven bound；bounded 不得声明 `analytic_upper_bound`）。

## 5. fail-closed 清单

| 触发 | 错误（code / reason） | 发生层 |
|---|---|---|
| 未知/未注册 opcode（含 Core IR 有 alias 但无执行契约，如 `add`/`transpose`） | `OP_CONTRACT_INVALID / opcode_not_registered` | `plan_operation`/resolver |
| dtype 越域 | `OP_CONTRACT_INVALID / dtype_outside_contract_domain`（或 `dtype_mismatch`） | 契约 verifier |
| rank 越界 / 符号维 / 负维 | `rank_out_of_range` / `static_shape_required_by_portable_provider` / `negative_dimension` | resolver + verifier |
| 广播不合法 | `broadcast_shape_mismatch` | 契约 verifier（复用 lowering `_broadcast_shape`） |
| shape 规则违反（含 reduce_max 空轴、reshape element count、iota 越界） | `shape_rule_violation` / `empty_reduction_has_no_identity` / `reshape_element_count_mismatch` / `iota_attribute_invalid` | 契约 verifier |
| 未知属性 / 属性值越权 | `unknown_attribute` / `cast_mode_not_portable` / `reshape_inplace_not_portable` 等 | 契约 verifier |
| `require_proven_deviation_bound=true` 遇无 proven 包络的 float 路径 | `NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable` | resolver |
| vendor 不支持该 op（未列入 `allowed_providers`） | 候选项 `state=rejected / provider_not_allowed_by_contract`（结构化拒绝，不误选） | resolver |
| plan 被篡改（含重算 digest 的 v3 request 伪造、precision 声明改写） | `ARTIFACT_MISMATCH / plan_digest_mismatch` / `plan_field_inconsistent` | `ExecutionPlan.validate` |
| report 被改写 | `REPORT_SCHEMA_INVALID / report_*` | `validate_report_schema`（v4 自包含） |
| 执行时 operand/output 与冻结 plan 不符、输出别名输入 | `plan_request_mismatch` / `output_input_alias` | entry/provider |

## 6. 未覆盖清单与下一批建议

### 6.1 已桥接到执行层的其余 Core IR 算子（本批未注册，共 24 个 canonical）

`add`、`sub`、`mul`、`div`、`neg`、`broadcast`、`concat`、`constant`、`contiguous`、`embedding`、
`gather`、`identity`、`reduce_mean`、`rsqrt`、`sigmoid`、`silu`、`slice`、`softplus`、`split`、
`transpose`、`view`、`quantize_per_token_s8`、`dequantize_s8`、`dequantize_epilogue_bf16`

建议下一批（U 线 batch2）优先：`add/sub/mul/div/neg`（模型逐元素算术主干）→ `sigmoid/silu/softplus/rsqrt`
（Qwen 激活家族）→ `transpose/view/slice/concat/split/gather/embedding`（shape/数据搬运家族，需先决定
`view` 的别名/`must_alias` 语义）→ `reduce_mean`（浮点误差口径与 reduce_sum 同族）→ W8A8 的三个
量化 primitive（它们的 vendor 路径与 packed layout 已冻结在 0002/0010，可独立成片）。

### 6.2 前端家族桥接（未在本批）

- `pypto` classic Tensor 前端、`pypto_pro` Professional 前端的 op 包装层到 Core IR 的映射尚未接入
  `plan_operation`（本批只承诺 Core IR/CpuScalar 已桥接的 8 个 opcode）。
- Qwen 全图所需的高频算子仍缺 `rope`/`attention`/`gather`/`concat` 等（见 6.1 建议顺序）。
- composite 树化（另一 agent 的 qwen35 工作）不在本切片；本切片不改 `python/pypto/portable/qwen35.py`。
- region/图级执行（U5）、跨算子融合、vendor 对 8 个新 op 的原生实现均未开启；本批 vendor 只在策略层
  兼容（不支持即结构化拒绝）。

## 7. 验证与证据

- 逐算子契约违规矩阵（dtype×rank×broadcast×attribute）与逐算子差分：见
  `python/tests/ut/pypto_x/test_execution_ops_registry_batch1.py`（125 项）。
- 差分口径：执行层 `execute_operation` 输出 vs 契约声明的 portable 参考 helper 直接调用
  （`_where_value` / `compare_values` / `iota_values` / `_math_value` / `_reduce_value` /
  `_cast_value_for_operation` / `_reshape_value`），逐位比较（NaN 按 NaN、±0 按符号位）。
  机器可读汇总：`_meta .../raw/batch1_contracts.json`（31 例、0 不一致）。
- 零默认变化：`_meta .../scripts/record_execution_baseline.py` 在基线树与新树各跑一次，
  contract/capability/plan/artifact/output digest 全部逐字节一致：
  `_meta .../raw/baseline_execution_record.json` vs `raw/new_execution_record*.json`。
- CLI：`python -m pypto.execution.cli doctor --json` 的 `opcodes` 含 10 个契约、schema 3；
  `explain --op exp --input 2,3:float32` 给出 portable 候选 + vendor 拒绝原因。
- 规则 8 全量（`scripts/resource/run_local_heavy.sh` + local 锁）：
  collect **2312** / passed **2303** / skipped **9** / failed **0**，rc 0，耗时 22:15
  （`_meta .../logs/rule8_summary.txt`、`logs/rule8_rc.txt`）。9 个 skip 为 7 个 CUDA driver
  不在本机 + 2 个 host-guarded 基线 digest 钉（heavy 锁会把 `OMP_NUM_THREADS` 设为 6，AOCL
  vendor manifest 的线程事实因此变化、capability digest 随之变化；这不是本切片引入的漂移）。
- 零默认变化在两个环境各做一次“基线树 vs 本树”同环境对照：12 线程默认环境（digest
  `e0a9dac8…`）与 6 线程锁环境（digest `70d8cf05…`）下 matmul f32/bf16 与 qmatmul 的
  plan/artifact/output digest 均逐字节一致（`raw/zero_drift_diff.txt`、
  `raw/zero_drift_diff_lock_env.txt`）。

## 8. 已知边界

- 标量操作数（lowering 支持）不在本批执行契约内；需要时按同一 verifier 扩展。
- capability 的“按 opcode 收窄视图”是冻结 digest 兼容策略；待 U 线基线重新冻结后，应把 10 个契约
  折叠回默认 probe 并重录基线。
- float 路径没有 proven 解析包络，`require_proven_deviation_bound=true` 下不可用；需要该保证的模型
  路径应等待下一批给出可复算包络或 vendor 证明。
- 性能未测，不做任何性能声明；任何本机计时均为 UNGATED。
