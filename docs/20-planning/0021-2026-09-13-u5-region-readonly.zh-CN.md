# 0021 — U5 region 只读规划（batch 0049 切片 A）

- 任务：`u5-region-readonly`
- 实现基线：integration tip `a55a005eb`（batch-4 Tier-3 五个 opcode + 0053-A 多输出 ABI / `split` 已落地）
- round-6 commit：`7edda47edb35262b93e9c74449cc303c1f65638a`（分支 `work/u5-region-readonly`，单个新 commit；未 push）
- **schema**：`REGION_SCHEMA_VERSION = 3`（v1 → `region_schema_version_legacy`；v2 →
  `region_schema_version_v2_unsupported`；其余 → `region_schema_version_mismatch`）
- 依赖：`0019`（opcode 契约第一批，已落地）、`0018`（composite 组合化，已落地）、父方提案 `0020`、
  `0024`/`0026`（多输出 ABI；本切片实现其 step 3）
- 结论等级：**实现完成（只读）**。未做执行、未做 v5 报告、未做动态 shape、未做性能声明；本机数字一律 UNGATED
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u5-region-readonly/`

## 0. 摘要

本切片把 "图" 提升为一等公民的**第一步（只读）**：

- `plan_region(program, region_spec, policy, capability=None) -> RegionPlan`：把 `CoreProgram`
  中由静态 def-use 连通的算子集合冻结成 `REGION_SCHEMA_VERSION=3` 的 region plan 文档，
  逐 op 复用 `plan_operation`（不新造 resolver），产出 `region_digest` 与 region 级精度声明；
- **多输出算子**（round-6 / 0024 step 3）：`split` 的静态结果数由冻结 verifier 从
  opcode+attributes+input shape 决定，region 不再要求 `len(operation.outputs)==1`；
  `request.outputs` 是有序 `{name, role, kind, dtype, shape}` 列表，def-use 边按
  `(op, output_name)` 建立；边界 exit 记录 `(producer_op_index, output_index, output_name)`
  身份；`region_precision.outputs` 对每个 exit 输出独立走 0020 §2 合成（`is_upper_bound` 逐输出）；
- `explain_region(...)` / `explain_region_plan(...)`：逐 op 复用 U3 的 explain 文档
  （候选 / 拒绝原因 / tie-break / fallback / 边界表），region 级给出可复算的合成推导；
- `validate_region_plan(...)` / `load_region_plan(...)`：严格校验器。region digest 从文档自身
  复算；逐 op `single_op_plan_digest` 与 `plan_operation` 重放结果双向锚定；篡改一位即拒绝；
- CLI：`plan --region` / `explain --region` / `explain --region-plan`，退出码沿用既有约定
  （请求级错误 1，argparse 用法错误 2，doctor --strict 3 不变）；
- **不执行**：region 模块只调用 resolution 路径（`plan_operation`），不导入 runtime，不调用
  launch；单算子路径（`plan_matmul` / `plan_qmatmul` / `plan_operation`）行为与 digest 逐字节不变。

最有价值的副产品：对真实 Qwen3.5 图（attention / GDR / 全 decoder 4550 ops）跑整图
`plan_region`。round-5 曾用它得到**按 opcode 聚合的缺失 contract 清单**（batch2/3 注册优先级）；
round-6（batch-4 + 多输出 region）之后三图 **missing_contracts 0 类 / 0 次、unplannable 0 次、
3/3 整图可规划**（见 §5），并同时给出"多输出算子 1 次 op 引用 vs N 个输出"的口径对照。

## 1. region 口径与 `region_spec`

### 1.1 两种表述（schema 内冻结）

| kind | 语义 | 必填 |
|---|---|---|
| `program` | 整个 `CoreProgram`（可选 `function` 收窄到单个 CoreFunction） | — |
| `boundary` | 显式 `entry_inputs` / `exit_outputs` 之间、由 def-use 连通的静态子图 | `entry_inputs`、`exit_outputs` 非空；多函数 program 需 `function` |

Python 侧既可传 `RegionSpec`，也可传 mapping（`{"kind": ..., "entry_inputs": [...], ...}`）；
`None` 等价于 `{"kind": "program"}`。CLI 侧为 `--region program|boundary` +
`--program` + `--region-entry/--region-exit`（可重复）。

`boundary` 的语义：region = 从任一 entry 前向可达、且能到达任一 exit 的算子交集；
entry 只能是函数根作用域可见值（函数参数、根作用域定义值），不允许隐式捕获。
region 内算子按程序内冻结声明顺序（Core IR 要求 def-before-use，因此声明序即一个合法拓扑序）
重新编号为 0..N-1；`region_digest` 绑定这个顺序。

### 1.2 不做 op 索引区间

`region_spec` 刻意**不接受** `op_indices` 之类的原始索引区间，遇到未知字段结构化拒绝
（`region_spec_unknown_field`）。理由：原始索引不随图编辑保持稳定，同一份 region plan 的
digest 会在两次 revision 之间悄悄改变含义；而 `boundary` 由 SSA 值名 + 类型签名定义，
在加/删算子时要么保持语义、要么明确拒绝。

round-6 的唯一例外是 **exit 的 `(op_index, output_name)` 对**：`exit_outputs` 既可继续写
裸 SSA 名字（单输出表达式，v1/v2 完全兼容），也可写 `(op_index, output_name)` 对。`op_index`
接受该 producer 的函数体内 source index（`body.ops[op_index]`）或其 region-local 拓扑编号
（二者之一即可）。它不是索引区间：SSA 名字本身唯一确定值，pair 只是把 `(op, output)` 的
def-use 身份显式写出做冗余一致性校验（不匹配 → `region_spec_exit_output_index_mismatch`）；
`region_digest` 内仍使用 region-local 的 `producer_op_index` + `output_index`。

### 1.3 边界合法性（结构化拒绝清单）

`plan_region` 先跑 `CoreProgram.verify()`，再按下列顺序 fail-closed：

| 触发 | reason |
|---|---|
| program 不是 CoreProgram / verify 失败 | `region_program_invalid` / `region_program_invalid` |
| `region_spec` kind/字段非法 | `region_spec_kind_invalid` / `region_spec_unknown_field` / `region_spec_boundary_incomplete` 等 |
| 边界 tensor 在函数根作用域不可见（悬空） | `region_boundary_dangling_tensor` |
| `exit_outputs` pair 形状非法 / 索引与 producer 不符 | `region_spec_exit_output_invalid` / `region_spec_exit_output_index_mismatch` |
| 多输出契约的结果数/role/kind/shape 与 verifier 推导不一致 | `region_operations_unplannable`（reason 取自契约，如 `output_count_mismatch` / `output_shape_mismatch`） |
| 多函数 program 的 boundary spec 未指定 function | `region_function_required` / `region_function_not_found` |
| region 内算子读取未声明的外部值（跨 region 引用/隐式捕获） | `region_boundary_undeclared_input` |
| 声明的 entry 没有被 region 消费 | `region_unused_entry_input` |
| 声明的 entry 由 region 内算子产生 | `region_entry_input_defined_in_region` |
| entry 的 producer 在 region 起始之后 | `region_boundary_input_defined_after_use` |
| exit 不是 region 内算子的结果 | `region_exit_output_not_produced_by_region` |
| 前向引用/自引用（闭环、逆拓扑） | Core IR verify 或 `region_cycle_detected` |
| 边界选中 0 个算子 | `region_empty` |
| 边界 entry kind 不在 {tensor, scalar} | `region_boundary_operand_kind_invalid` |
| scalar entry 的 shape 非空 | `region_scalar_boundary_shape_invalid` |
| scalar exit 不是 entry pass-through / 操作结果为标量 | `region_boundary_non_tensor` |

所有错误都带 `code`（沿用 `OP_CONTRACT_INVALID`）、`field`、`requested`、`reason`、`details`，
可用 `to_dict()` 机器读取。

## 2. 冻结 region plan（`REGION_SCHEMA_VERSION=3`）

```jsonc
{
  "region_schema_version": 3,
  "kind": "region_plan",
  "region_id": "region.<region_digest hex 前 16 位>",
  "region_digest": "sha256:...",
  "region_spec": {"kind": "program", "function": null},   // 或 boundary + 名字/pair 列表
  "policy": { ... ExecutionPolicy.to_dict() ... },
  "policy_digest": "sha256:...",
  "capability_digest": "sha256:...",        // 传入/探测的 base capability snapshot
  "resolver_version": 2,
  "boundary_signature": {
    // entry: output_index 恒为 null；exit: (producer_op_index, output_index, name) 是身份
    "entry_inputs": [{"function","name","role","source","kind","dtype","shape","producer_op_index","output_index"}],
    "exit_outputs": [ ... ]
  },
  "operations": [ /* 拓扑序，见 §3 */ ],
  "op_count": N,
  "region_precision": { /* region 级精度声明与可复算推导，见 §4 */ }
}
```

`region_digest` 的口径（Merkle 式锚定，严格按 0020 §3）：

```text
region_digest = "sha256:" + sha256(canonical_json({
  "region_schema_version": 3,
  "boundary_signature": <boundary_signature>,
  "operations": [ <完整逐 op 条目（含单算子 plan digest、selected、
                    numeric_guarantee、precision/envelope、fallback、bound、
                    provider metadata）>, ... 按 region-local 拓扑序 ... ]
}))
```

要点：

- **版本纪律**：round-2 把 preimage 从"4 个 op 字段"扩成"完整 op 条目"（v1→v2）；
  round-6 改变了**边界表达**（exit 从算子级扩展为 `(producer_op_index, output_index, name)`，
  多输出 `request.outputs` 变为有序 5 字段列表），按 0024 §4 升 `REGION_SCHEMA_VERSION=3`。
  解码器：v1 → `region_schema_version_legacy`；v2 → 专用
  `region_schema_version_v2_unsupported`（requested=2/available=3）；其余未知版本 →
  `region_schema_version_mismatch`。绝不把 v2 文档按 v3 重新解释；
- **完整有序 op 条目**（不是只有 `single_op_plan_digest`）+ 边界签名 + schema 版本三者进
  preimage；`region_id` 由 digest 派生；round-2 起，逐 op 的
  `selected`/`numeric_guarantee`/`precision`/`envelope`/`fallback`/`bound`/provider metadata
  全部在 preimage 内，改任一字段即 `region_digest_mismatch`；
- `operations[].index` 是 **region-local** 拓扑位置 0..N-1（不是扁平化全局 op 序号）；
  文档内一切引用（exit 的 `producer_op_index`、def-use 检查）都用该编号；
- entry input 的 producer 若在 region 外，记 `source="outside_region_operation"` 且
  `producer_op_index=null`（region 外 op 不属于本冻结文档）；
- 逐 op plan digest 自身绑定 policy digest、capability digest、contract digest、precision/envelope；
  因此 **capability 变化如果影响任一 op 的解析，必然改变 region digest**；与 region 语义无关的
  变化也在 per-op plan payload 的 `target.capability_digest` 中体现。文档另存 base
  `capability_digest` 供 replay 锚定。
- 同 program + spec + policy + capability 两次解析的文档逐字节相同（测试钉住）。
- **边界 exit 顺序**（round-6）：canonical 顺序 = pass-through 在前、随后按
  `(producer_op_index, output_index, name)` 升序；多输出算子的多个 exit 按 producer 的
  result 顺序展开，`output_index` 显式记录结果位置。顺序进 preimage，校验器还做语义顺序检查
  （重排 + 重签 → `region_boundary_order_mismatch`）；
- **单输出表达式保持兼容**：单输出 op 的 `request.outputs` 仍是 `{name, shape, dtype}`（tensor）
  或附加 `kind="scalar"`；裸名字 exit 与 region-local producer 编号语义不变，只是 v3 额外写上
  `output_index`（单输出恒为 0）。schema 版本升级会改变所有 region digest（v3 与 v2 不可比）。

## 3. 逐 op 条目与双向锚定

每个 `operations[i]` 含（拓扑序位置 `index == i`）：

```jsonc
{
  "index": 0,
  "function": "qwen35_text_decoder",
  "path": "body.ops[3]",
  "op_name": "matmul",              // Core IR 原始拼写
  "opcode": "matmul",               // canonical opcode
  "request": {
    "op": "matmul",
    "input_shapes": [[...], [...]], "input_dtypes": ["bf16","bf16"],
    "output_shape": [...], "output_dtype": "bf16",
    "attributes": {...},
    "inputs":  [{"name","role","shape","dtype"}...],
    "outputs": [{"name","shape","dtype"}]   // 单输出 tensor；scalar 结果附加 "kind":"scalar"
    // 多输出契约（0053-A）：有序 5 字段列表，如
    // "outputs": [{"name","role":"segment","kind":"tensor","dtype","shape"}, ...]
  },
  "contract_digest": "sha256:...",
  "contract_version": "1.0",
  "single_op_plan_digest": "sha256:...",
  "plan_schema_version": 2,
  "provider_id": "portable",
  "provider_version": "...",
  "implementation": "...",
  "numeric_class": "deterministic_bounded",     // 该 op 的 contract 精度类
  "selected_numeric_class": "portable_bitwise", // provider manifest 的 numeric class
  "selected": { ...plan payload selected 原文... },          // round-2 新增，进 preimage
  "numeric_guarantee": { ...plan payload 原文... },           // round-2 新增，进 preimage
  "precision": {...} | null,                    // numeric_guarantee 的投影（可复算）
  "envelope": {...} | null,                     // numeric_guarantee 的投影（可复算）
  "fallback": { ...plan payload fallback 原文（含 events）... },  // round-2 原样落盘
  "fallback_used": true,                        // 由 events 派生，不改写 resolver 声明
  "bound": {"declared_class","proven","bound","bound_kind","is_upper_bound","reference","reasons"}
}
```

`single_op_plan_digest` 就是 `plan_operation(opcode, input_shapes, input_dtypes,
output_shape=..., output_dtype=..., attributes=..., outputs=<多输出时的 request.outputs>,
policy=<文档内 policy>, capability=<replay snapshot>)` 得到的 `ExecutionPlan.digest`；`request`
字段足以在**不依赖源 program** 的情况下重放。多输出 op 的 replay 会把文档
`request.outputs` 与重放 payload 的 `request.outputs`、`selected.outputs` 逐字段
（name/role/kind/dtype/shape/顺序）比较（`region_request_outputs_payload_mismatch` /
`region_selected_outputs_payload_mismatch`），`single_op_plan_digest` 同时约束全量列表。

### 3.1 锚定口径（round-2 修复的核心）

只有 `single_op_plan_digest` + 3 个 replay 字段（plan/contract/provider）不够：`region_digest`
的公开算法任何人都能重签，因此保证字段必须同时满足两条：

1. **进 preimage**：`operations` 的**完整条目**（`selected`/`numeric_guarantee`/`precision`/
   `envelope`/`fallback`/`fallback_used`/`bound`/provider metadata/`request`/digest ...）都进
   `region_digest`。只改保证字段而不重签 digest → `region_digest_mismatch`；
2. **replay 逐字段比对**：`re_resolve=True` 时，除 `single_op_plan_digest` / `contract_digest` /
   `contract_version` / `plan_schema_version` / provider metadata 外，还把文档里的
   `selected`、`numeric_guarantee`、`fallback` 与 `plan_operation` 重放出的 plan payload
   **逐字段相等**比对。即使攻击者重签了公开 digest，把 `exp` 的
   `precision` 从 `deterministic_bounded` 伪造成 `exact`/proven 仍会因
   `region_numeric_guarantee_payload_mismatch` 被拒；抹掉 `fallback.events` 会因
   `region_fallback_payload_mismatch` 被拒。

`region_spec`/`policy`/`capability_digest`/`region_precision` 不进 preimage，但分别由语义交叉
校验、policy digest 复算、replay capability 锚定与 `region_precision` 复算覆盖。

### 3.2 零输入 opcode 与 region-local 编号

- **零输入 opcode**：`request.inputs`/`input_shapes`/`input_dtypes` 允许为空；validator 不再
  假设"输入非空"，而是按冻结契约的 arity（`len(OpDefinition.inputs)`）校验
  `request.op`/operand 数/role/结果数。batch-1 的 `iota` 因此可以 `plan → validate → load →
  explain` 完整往返；
- **region-local 编号**：boundary 子图即使从全局第 1 个 op 开始，落盘后也重编号为 0..N-1；
  exit 的 `producer_op_index` 用 region-local 编号；region 外 producer 的 entry 记
  `source="outside_region_operation"` + `producer_op_index=null`；
- **自检不变量**：`plan_region` 在返回前对自己的 document 跑一次
  `validate_region_plan(..., re_resolve=False)`；任何"自产不可 load"的文档在 plan 阶段就
  结构化失败（boundary 编号、iota、空选择三类问题都会在此暴露）。

### 3.3 scalar 操作数的 region 表达与预检口径（round-4）

- **预检口径**：region 层不再对 scalar 操作数一刀切；`_plan_region_operation` 只拒绝
  `kind ∉ {tensor, scalar}`（`non_tensor_non_scalar_operand_not_supported`）以及
  scalar 带非空 shape（`scalar_operand_shape_invalid`）。tensor/scalar 的操作数请求
  **全部委托给 opcode 契约**（`plan_operation`，scalar 以 `shape=[]` + dtype 表达）；
  契约是否接受由冻结契约文字决定：`where` 的 fill、`add`/`mul` 两侧 scalar 合法，
  `exp`/`reduce_*` 等不接受 scalar 的 opcode 以契约 reason（例如 `rank_out_of_range`）
  结构化拒绝——不再是 region 层的 `scalar_operand_not_supported_by_execution_contract`。
- **如实表达**：`request.inputs[i]` 对非 tensor 操作数写 `"kind": "scalar"`、`shape: []`
  （缺省 `kind` 表示 tensor，与单算子 plan payload 的 batch-2 约定一致）；boundary signature
  的 `entry_inputs` 同样带 `kind`（tensor/scalar）。scalar 常量通常来自函数参数或 boundary
  spec 声明的 mid-graph 值，因此它是 region 的 **entry input**；scalar 不能是 region 内
  算子的结果（冻结契约只产生 tensor 结果，def-use 检查 `region_scalar_operand_from_operation`
  拒绝）。scalar exit 只允许 `entry_pass_through`（参数原样返回），任何算子结果 exit 必须是
  tensor。
- **锚定**：`kind` 字段位于逐 op 条目与边界签名内，二者都进 `region_digest` preimage；
  校验器还做语义交叉检查（operand `kind` 必须等于其 boundary entry 或 producer 的 kind）。
  删除/改写 scalar `kind`：不重签 digest → `region_digest_mismatch`；重签后 →
  `region_boundary_type_mismatch`（scalar entry 与默认 tensor operand 不一致）。
- **不升 schema 版本的理由**：这是一个**加性可选字段 + 验证放松**——不含 scalar 的 v2
  文档字段结构与语义完全不变（仍然合法、digest 不变），也不存在"旧字段被重新解释"的问题；
  因此 `REGION_SCHEMA_VERSION` 保持 **2**，v1 仍 `region_schema_version_legacy`。
- **实测**：三张真实 Qwen 图共 2127 次 batch-2 相关 opcode 引用（其中 batch-2 新登记 6 个
  opcode 恰为 2119 次）现在**全部 plan 成功**，region 级 scalar 预检造成的
  `unplannable_operations` 从 **212** 次降为 **0**（见 §5）。

> **精度修正（2026-09-19，由 `verify-0049-u5-region-r4` 独立复算）**：在 `8b5246557`（batch-2 已落地、scalar 委托前）实测
> `unplannable_operations` 为 **212**（`add:scalar` 176 + `mul:scalar` 29 + `where:scalar` 7），而非本文先前写的 205——
> 205 是作者把 `where` 的 7 次另列后的口径。round-4 后精确为 **212 → 0**。类数 19→13、六算子归零与其余结论不受影响。

### 3.4 scalar 结果的表达与边界口径（round-5）

- **逐 op 条目**：`request.outputs[i]` 对非 tensor 结果写 `"kind": "scalar"`、`shape: []`
  （缺省 `kind` 表示 tensor，与 operand 侧同一约定）；scalar 结果必须 `shape=[]`，否则
  `scalar_result_shape_invalid`；`kind ∉ {tensor, scalar}` 的结果在 region 层拒绝
  （`non_tensor_non_scalar_result_not_supported`）。`kind` 位于逐 op 条目内，进 `region_digest`
  preimage。
- **def-use**：生产者结果的 `kind` 与消费者 operand 的 `kind` 必须一致（`tensor`↔`tensor`、
  `scalar`↔`scalar`），不一致 → `region_operand_kind_mismatch`；scalar operand 现在既可以是
  boundary entry，也可以来自 region 内的 scalar 结果（round-4 的"scalar 不能由算子产生"限制已
  撤销，因为它与 batch-3 之后的契约能力不符）。
- **边界**：scalar **可以作为 region exit**，两种来源都允许：
  1. `source=operation_result` + `producer_op_index=<region-local>`（算子结果直接作为边界输出）；
  2. `source=entry_pass_through` + `producer_op_index=null`（既有语义不变）。
  exit 的 `kind="scalar"` 必须 `shape=[]`；validator 用 def-use 与 producer/entry 的 kind 做
  交叉检查（`region_boundary_type_mismatch`）。`kind ∉ {tensor, scalar}` →
  `region_boundary_operand_kind_invalid`；scalar boundary 带非空 shape →
  `region_scalar_boundary_shape_invalid`。**不做** scalar→tensor 的隐式改写/包装。
- **不升 schema（round-5 结论，已被 round-6 取代）**：round-5 的加性 `kind` 字段本身不需要升
  schema；但 round-6 改变了边界表达（exit 增加 `(producer_op_index, output_index, name)` 身份、
  多输出 `request.outputs` 有序列表），按版本纪律升 `REGION_SCHEMA_VERSION=3`。round-5 的 v2
  文档（含 scalar 结果）现在以 `region_schema_version_v2_unsupported` 拒绝，v1 仍
  `region_schema_version_legacy`。round-5 的 scalar 表达与锚定测试全部改为在 v3 文档上执行。
- **验证方式**：`reduce_sum` 声明 scalar 结果（Core IR `CoreType.scalar`）→ 作为 scalar 操作数
  被 `add` 消费、以及作为 region exit，均完成 `plan → validate → load → explain` 往返；
  改/删 `kind`（不重签 → `region_digest_mismatch`；重签 → `region_operand_kind_mismatch` /
  `region_boundary_type_mismatch`）全部拒绝。
- **契约缺口已由上游关闭**：round-5 发现的 66 次 `constant` rank-0 scalar 输出缺口
  （batch-3 契约只允许 rank 1..4）由 integration tip 的 `32e42abff`（`align batch-3 contracts
  and add scalar-kind constant results`）修复；round-6 三图证据里这 66 次全部 plan 成功（§5）。

严格校验器 `validate_region_plan(document, *, policy=None, capability=None, re_resolve=False)`
按固定顺序检查（任一失败抛 `ARTIFACT_MISMATCH`，不修复）：

1. 顶层 schema/kind、必填字段、未知字段、`resolver_version` 精确匹配；`region_schema_version`
   必须为 int 且 == 3（v1 → `region_schema_version_legacy`；v2 →
   `region_schema_version_v2_unsupported`；其他 → `region_schema_version_mismatch`）；类型为 bool
   的 `op_count` 等一律拒绝，未知 `policy.numeric.requirement` 抛结构化 `region_policy_invalid`
   （不再裸 `ValueError`）；`operations` 为空 → `region_operations_empty`；operand/result `kind`
   缺省视为 tensor，显式 scalar 必须 `shape=[]`；
2. `boundary_signature` 结构（角色/类型/静态 shape/entry `producer_op_index=null` 且
   `output_index=null`/exit `operation_result` 用 region-local `producer_op_index` + producer
   结果位置 `output_index`；exit 顺序必须 canonical，否则 `region_boundary_order_mismatch`）；
3. **从文档自身复算 `region_digest`（完整 op 条目），再复算 `region_id`**（改一位 →
   `region_digest_mismatch` / `region_id_mismatch`）；
4. 逐 op 结构 + `precision`/`envelope` 必须等于 `numeric_guarantee` 的投影 + `bound` 视图
   必须等于从 `numeric_guarantee` 重新归一化的结果（重签 digest 后仍抓 `bound` 篡改）；
   多输出 op 的 `request.outputs` 还会被冻结 verifier 从 opcode+attributes+input shape 重新
   推导并逐项比对（count/name/role/kind/dtype/shape/顺序；不匹配 →
   `region_operation_request_inconsistent`，`details.contract_reason` 给出契约 reason）；
5. 文档内部 def-use 闭包（重复定义、正向引用、隐式捕获、exit producer 一致性、
   entry/exit 与 request 的类型一致）；
6. `region_spec` 与边界/算子集合语义一致（region_spec 不进 digest，改用语义交叉校验）；
7. `policy` 复算 digest；`region_precision` 必须逐字节等于从 op 列表重新合成的结果；
8. `op_count == len(operations)`；
9. `re_resolve=True` 时：重放每个 op 的 `plan_operation`（多输出 op 带同一
   `outputs=request.outputs`），要求 `single_op_plan_digest` 一致，并对
   `selected`/`numeric_guarantee`/`fallback`/contract/plan schema/provider metadata 以及
   有序结果列表（request/selected outputs 的 name/role/kind/dtype/shape/顺序）**逐字段比对**；传入 capability 的 digest 必须等于文档 `capability_digest`
   （`region_capability_digest_mismatch`）。未显式传 capability 时探测本机，探测 digest 必须
   与文档一致，否则拒绝（合成 capability 的计划必须显式带 snapshot 才能 replay）。

`RegionPlan.validate(...)` 默认 `re_resolve=True`；`load_region_plan(path, ...)` 默认
`re_resolve=True`。因此"文档自证 + 单算子路径双向锚定"是同一次调用完成的。

## 4. region 级精度合成（0020 §2 落地）

### 4.1 逐 op 输入

对每个 op 取：

- `provider_class` = plan payload 的 `selected.numeric_class`（provider manifest 事实，
  用于 `numeric.requirement` 判定）；
- `declared_class` = contract 的 `numeric_guarantee.precision.class`（schema v3），否则
  `envelope.class`，否则 `selected.numeric_class`（绑定到冻结契约文字）；
- `proven` / `bound` / `bound_kind` / `is_upper_bound` / `reference`：来自 `precision` 或
  `exactness_envelope`。`proven` 要求 `deviation_bound_available=true` 且
  `deviation_bound_kind ∈ {exact_contract_bound, analytic_f32_chain_upper_bound}` 且
  `upper_bound.status == "proven"`；观测值（`measured_deviation_by_k`）永不当作界。
- `portable_bitwise` 且 provider 就是 portable trunk 时，绑定 `portable_trunk_bitwise_identity`
  （偏差 0，按构造可证）；非 portable provider 声称 bitwise 但无包络 → 记为 unproven。

### 4.2 合成优先级（唯一规则）

```text
best_effort                                        （任一 precision 或 provider 为 best_effort）
  > deterministic_bounded_unproven                 （任一 declared_class ∈ {bounded, bitwise} 且无 proven 包络）
  > deterministic_bounded                          （存在 proven bounded，且所有 op 都有界/恒等）
  > portable_bitwise                               （全部 op 是 exact / portable_bitwise）
  > exact                                          （全部 op exact）
```

- 任何情况下都给出 `provider_ids` / `provider_counts` / `mixed_providers`；
  `cross_provider_policy` 明写"各 provider 界独立相加，不做抵消/补偿/重排"；
- `all_portable_trunk` 单独记录"是否整个 region 都跑 portable trunk"（与 contract 精度声明解耦）；
  只有**所有 op 都真的跑 portable trunk** 时，`portable_bitwise` 的
  `deviation_bound.statement` 才写"逐位一致于 portable trunk"且 `is_upper_bound=true`；
  混 provider 的 portable_bitwise region 改写成条件式 statement、`is_upper_bound=false`、
  reasons `not_every_operation_ran_on_the_portable_trunk`（round-2 修复）；
- **回退如实标注**：逐 op 记录 `fallback.mode/primary/chain/used/degraded/events`（冻结 plan
  payload 只有 `events`/`degraded`；`used` 由 "events 非空（selected != preferred）" 派生，
  绝不把发生过的 host/scalar 回退写成未发生）。region 级另有 `fallback_ops`（含 mode/used/degraded）
  与 `fallback_modes`；
- `numeric.requirement` 用 `provider_class` 判定（与 resolver 的单算子语义一致），
  contract 精度类不参与 requirement，避免把 `exp(float)` 这类"provider 是 portable trunk、
  但 contract 声明无解析界"的算子误判为满足 "portable" 之外的东西；
- `require_proven_deviation_bound=true` 时：`unproven` / `best_effort` region 在 **region 级**
  fail-closed（`NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable`，
  field `numeric.require_proven_deviation_bound`，details 列出全部 unproven op）；
  best_effort region 另外带 `best_effort_region_not_usable_for_require_policy`。

### 4.3 `deviation_bound` 与"和是否为上界"

- `exact`：`{kind: exact_contract_bound, sum: 0, deviation_bound_by_k: {all_k_le_limit: 0}, is_upper_bound: true}`；
- `portable_bitwise`：全 portable region 为
  `{kind: portable_bitwise_trunk_identity, sum: 0, is_upper_bound: true}`，声明的是"与顺序执行的
  portable trunk 逐位一致"，不是"与数学参考的误差界"；混 provider 时为
  `{kind: portable_bitwise_mixed_provider_identities, is_upper_bound: false}`；
- `deterministic_bounded`：`{kind: sum_of_per_operation_proven_bounds, sum: Σ, is_upper_bound: ?}`。
  逐项列出 `terms`（index/opcode/provider/bound_kind/bound/is_upper_bound/reference）。
  `is_upper_bound=true` 当且仅当：
  1. 每一项的界都是 proven，且可数值相加（`int` 或同 key 的 `by_k` 逐项相加）；
  2. **所有项（含 0 界项）的 reference 一致**，且每一项都带 reference；
  否则给出 `cross_reference_bounds_not_comparable` / `bound_without_reference_present` 等理由，
  `is_upper_bound=false`，且在 `require_proven_deviation_bound=true` 下 region 级 fail-closed。
  （round-2 前的缺陷：reference 一致性只看非零界项，导致 "exact 0 界 ref A + bounded 5 界 ref B"
  错误地报 `is_upper_bound=true`；现在两个不同 reference 的项一律 false。）
- `deterministic_bounded_unproven` / `best_effort`：`{kind: none, sum: null, is_upper_bound: false}`，
  不做任何"求和成界"声明。

`region_precision["derivation"]` 记录 5 个布尔检查（best_effort / unproven / all-exact /
proven-bounded / portable-present）、`selected_class`、`selected_rule`、
`no_error_cancellation=true`、`order_sensitive=true`、`topological_order_bound=true`，
可被 `validate_region_plan` 逐字节复算。

### 4.4 逐输出合成（round-6 / 0024 §4）

`region_precision` 的 `declaration_version` 升 **2**（`rule_id = u5-region-precision-0020-v2`）：

- 顶层字段仍是**整个 region 全部 op** 的保守合成（语义与 round-2..round-5 相同，只是 op 集合
  现在可以含多输出算子——多输出算子只贡献 1 个 term）；
- 新增有序 `outputs` 块，每个 boundary exit 输出一块，按 §2 的 canonical exit 顺序。块内用
  `exit`（function/name/role/source/kind/dtype/shape/producer_op_index/output_index）标识输出，
  `op_indices` 是该输出的**传递 def-use producer 闭包**（冻结拓扑序），`class`/`selected_rule`/
  `provider_ids`/`provider_counts`/`mixed_providers`/`deviation_bound`/`unproven_ops`/
  `best_effort_ops`/`fallback_ops` 都是**只对该闭包**重新合成的结果：
  - `deviation_bound` 只列 `sum`/`kind`/`is_upper_bound`/`reasons`/`term_indices`（共享的逐 op
    `terms` 行只写一次，避免 decoder 规模下的重复放大）；`is_upper_bound` 逐输出独立判定；
  - 某 op 同时喂多个输出时，它的 term 出现在每个相关闭包里（各自独立求和，无跨输出抵消）；
  - `entry_pass_through` exit 的闭包为空，显式给 `{class: exact, kind:
    no_operation_entry_pass_through, sum: 0, is_upper_bound: true}`；
- 顶层 `fail_closed` = 顶层全 op 合成 fail-closed **或任一输出块 fail-closed**（保守取并集）；
  另有 `output_fail_closed` / `all_outputs_proven` 两个汇总布尔。`require_proven_deviation_bound`
  的实际 fail-closed 通常先在逐 op `plan_operation` 层触发（unproven op 直接
  `NUMERIC_GUARANTEE_UNMET`），region 合成是第二道防线；
- `compose_region_precision(terms, output_groups=...)` 是可复算纯函数：`output_groups` 由
  `_region_output_groups(operations, boundary_signature)` 仅从文档自身推导，所以
  `validate_region_plan` 能逐字节复算 `region_precision`；不带 `output_groups` 调用时保持
  v1 形状（合成 term 的对抗测试继续复用）。

## 5. 真实 Qwen 图上的分布（round-6）

证据：`raw/round6_qwen_region_evidence.json`（脚本
`scripts/round6_qwen_region_evidence.py`；三图整图 `plan_region`；read-only；统计口径也写在
JSON 的 `statistics_caliber` 内）。基线 integration tip `a55a005eb`（batch-4 Tier-3 五算子 +
0053-A 多输出 ABI / split 已落地）：

| builder | op 引用 | 输出数 | boundary exits | 缺失 contract | unplannable | 结果 |
|---|---|---|---|---|---|---|
| `build_attention_kv_cache` $(1,8,4,256)$ | 20 | 20 | 3 | 0 类 / 0 次 | 0 | 整图可规划 |
| `build_qwen35_gdr_recurrent_state` (1,4) | 123 | 123 | 2 | 0 类 / 0 次 | 0 | 整图可规划 |
| `build_qwen35_text_decoder_graph` (1,1,past=4096) | 4550 | 4592 | 51 | 0 类 / 0 次 | 0 | 整图可规划 |
| **合计** | **4693** | **4735** | **56** | **0 类 / 0 次** | **0** | **3/3 可规划** |

- **口径（重要）**：本表的"op 引用"= Core IR 算子个数（一个 3 输出 `split` 计 **1** 次）；
  "输出数"= `request.outputs` 条目数（同一 `split` 计 **3**）。decoder 的 24 个 `split`
  共产生 **66** 个输出（18 个三输出 + 6 个二输出），这正是 4693 → 4735 的差额来源。round-5
  的 `unplannable` 24 是"24 个 split op 引用"；round-6 两类计数都是 0。
- **round-5 → round-6 变化**：缺失 contract **5 类 / 36 次 → 0 类 / 0 次**；`unplannable`
  **24 → 0**（split 多输出 ABI + region 多输出）；66 次 `constant` rank-0 scalar 缺口由
  `32e42abff` 关闭。
- **region digest（schema v3）**：
  - attention：`sha256:77ff076eb5b33060fac86457af44686feba620d0787e98fadce48ae3d0478c59`（同文档按 v2 边界表达重算的 old-style digest：
    `sha256:92e21b47b42aae98d592ed74f43a40a619abe763ffa202396900d0ffdbfee2d9`，两者不同证明 schema/边界表达变化已进 digest）；
  - GDR：`sha256:16cc519fe814172588a695c6ad15da3ad0eebb0b8ccd85a8e67b94f214b1da3c`（v2-style `sha256:668bd4fa7930ae2bd21bd0fa8d53bb5ff5142a4ab7e2e5e0d845dddae6509bbb`）；
  - decoder：`sha256:b452e3812042647b8721d131ff244e0b5bc444a03e09a32c696d1156c6b22b63`；v2 无法表达（含 `split` 多输出请求），故不伪造 v2 digest
    （`v2_style_region_digest_reason = v2_cannot_represent_multi_output_operation`）。
- **逐输出精度**：attention 3 个 exit 各自合成（`new_k`/`new_v` exact 1 op，`output`
  deterministic_bounded_unproven 20 ops）；GDR 2 个 exit（`output` 122 ops / `new_state`
  93 ops，闭包不同）；decoder 51 个 exit 全部逐输出合成，closure 大小互不相同（例如
  `layer_03_new_k` 643 ops vs `layer_03_new_v` 616 ops），`position_ids` 是 exact 1 op，
  其余 50 个为 `deterministic_bounded_unproven`（详见 JSON 的 `region_precision.outputs`）。
  这直接证明"逐输出闭包"不是把整个 region 的合成复制一遍。
- **planning 耗时**（UNGATED，锁内 6 线程）：attention 0.13 s / GDR 0.72 s / decoder 38.07 s
  （含 digest 复算与 `plan_region` 自检）。不作为性能声明。

### 5.1 整图可规划的依赖状态

1. `constant` rank-0 scalar 输出：**已关闭**（`32e42abff`）；
2. `split` 多输出 ABI：**已关闭**（`f20678948` + `a55a005eb`）；
3. `div/gather/identity/embedding`：**已关闭**（batch-4 `3a9ec408f`）；
4. region 层多输出边界/逐输出精度：**round-6 本切片**。
5. 仍未做：执行（`execute_region_plan`）、v5 报告、动态 shape、多输出与 scalar-kind 结果组合、
   运行时可变输出个数、跨 provider 多输出语义差异（见 §8）。

## 6. explain 与 CLI

### 6.1 `explain_region` / `explain_region_plan`

复用 U3 的 explain 机制，不新造第二套解释层：

- 每个 op 的 `operations[i]` 复用 **公开** U3 入口 `discovery.explain(plan=...)` 的原样输出，
  含 `candidates` / `rejections` / `tie_break` / `fallback` / `rules` / `boundaries`，
  另加 `region_op_index`、`region_opcode`、`bound_view`、`region_fallback`；
- region 级 `composition` 给出 `derivation`（与文档内 `region_precision.derivation` 相同）、
  `bound_contributors`（哪些 op 贡献界）、`unproven_ops`、`provider_set`、`deviation_bound`、
  `outputs`（round-6：逐 exit 输出的独立合成块）、`fail_closed` / `fail_closed_reasons`；
- `explain_region_plan(document)` 会在重放前校验文档 digest，并把每个 op 的 digest 与
  `plan_operation` 重放结果比对；explain 文档自身的 `digest` 是 canonical sha256。

### 6.2 CLI

```bash
# 整图（缺失 contract 会以 exit 1 + 结构化 JSON 返回，details.missing_contracts 按 opcode 聚合）
python -m pypto.execution.cli plan --region program --program graph.json --json
python -m pypto.execution.cli explain --region program --program graph.json --json

# 边界子图
python -m pypto.execution.cli plan --region boundary --program graph.json \
    --region-entry X --region-exit Y [--region-entry ...] [--region-exit ...] [--json]

# 冻结 region plan 的离线 explain / 校验（plan 输出可直接 --out 保存）
python -m pypto.execution.cli explain --region-plan region.json [--snapshot snapshot.json] [--json]
```

退出码：0 成功；1 请求级结构化错误（`ExecutionError.to_dict()`，`--json` 时打印到 stdout）；
2 argparse 用法错误。`fail_closed`/`strict` 语义不新造：region 上的
`require_proven_deviation_bound` 直接复用 `NUMERIC_GUARANTEE_UNMET`，只有既有 doctor
的 `--strict` 才返回 3。

## 7. 验证与证据

### 7.1 对抗矩阵（全部通过）

| 用例 | region 声明 | fail-closed 断言 |
|---|---|---|
| 全 exact（reshape 链） | `exact`, sum=0, `deviation_bound_by_k={all_k_le_limit:0}` | strict 通过 |
| 全 portable_bitwise（matmul） | `portable_bitwise`, `portable_trunk_bitwise_identity`, sum=0 | strict 通过 |
| bounded 且全 proven（合成 vendor capability：2 个 matmul，各 proven bound=2） | `deterministic_bounded`, sum=4, `is_upper_bound=true`, provider set 列出 | strict 通过 |
| bounded 含 unproven（`exp`+`reshape`，portable） | `deterministic_bounded_unproven`, bound kind=none | 默认 policy 可规划（provider class=portable_bitwise，requirement 满足）；strict 在 **region 级** 失败，details 列 `exp` |
| 混 provider（真实 AOCL exact-blocked + portable；qmatmul+reshape，auto+strict） | `exact`, `mixed_providers=true`, provider set 2 个；无跨 provider 抵消 | 通过 |
| 混 provider 且全 unproven（真实 AOCL f32 + AOCL int8，vendor policy） | `deterministic_bounded_unproven`, provider set 2 个 | 不声明界；strict 结构化拒绝 |
| best_effort（合成 term；本树无 shipped best_effort provider） | `best_effort`, `best_effort_region_not_usable_for_require_policy` | 任何 require policy 不可用；fabricated best_effort provider 被 resolver 结构性拒绝 |
| host/vendor 回退（required AOCL 不可用 → fallback portable） | op.fallback `used/degraded/mode=declared/events`，region `fallback_ops` 有 mode | replay/verify 通过 |
| 混 provider 的 portable_bitwise（exact-blocked qmatmul + portable matmul） | `portable_bitwise_mixed_provider_identities`, `all_portable_trunk=false`, `is_upper_bound=false` | statement 与 provider 集合一致，无"逐位等于 portable trunk"的过度声明 |
| F3-B 0 界跨 reference（exact ref A + bounded ref B） | `deterministic_bounded`, sum 保留, `is_upper_bound=false` | reason `cross_reference_bounds_not_comparable`；strict fail-closed |
| boundary 往返（region 外 producer / 晚起子图） | region-local 编号 0..N-1；entry `source=outside_region_operation` + `producer_op_index=null` | validate/load/explain 全通 |
| 零输入 `iota` | `request.inputs=[]`，契约 arity=0 | validate/load/explain 全通 |
| 多输出 `split`（1 入 2/3 出） | `request.outputs` 有序 5 字段；exit `(producer, output_index, name)` | plan→validate→load→explain 全通；改 name/shape/顺序/增删条目全部拒绝（§7.2） |
| 多输出 + 下游消费者（split→exp） | def-use 边按 `(op, output_name)`；单输出表达式与 pair 表达式同 digest | 全通；pair 索引不匹配 → `region_spec_exit_output_index_mismatch` |
| 逐输出精度（一分支出 exact、另一分支含 unproven exp） | 每个 exit 独立 `class`/`is_upper_bound`；顶层保守并集 | 逐输出块可复算；require policy 在 unproven op 处 fail-closed |
| 嵌套/block region | — | `region_nested_region_not_supported` / `region_block_region_not_supported` |

### 7.2 确定性 / 自证 / 篡改

- 同 program+spec+policy+capability 两次 → 文档逐字节相同、digest 相同；
- `validate_region_plan` 从文档自身复算 `region_digest` / `region_id` / 逐 op `bound` 视图 /
  `region_precision` / `policy_digest` / def-use 闭包；`re_resolve=True` 再逐 op 与
  `plan_operation` 重放比对；
- 篡改矩阵（op digest 翻一位、边界 source 改写、删 op、rewrite bound、region class、region_id、
  policy/capability digest、未知字段、bool 型 schema/op_count、未知 requirement）全部结构化拒绝；
- **逐字段锚定矩阵（14 类，参数化测试）**：`precision`、`bound`、`numeric_class`、
  `fallback`、`selected_numeric_class`、`provider_version`、`implementation`、
  `contract_version`、`plan_schema_version`、`op_name`、`path`、`request.inputs[].role`、
  `selected` payload、`numeric_guarantee.precision` 逐项"改写 + 自洽重算"：
  不重签 digest 一律 `region_digest_mismatch`；重签后分别被
  `region_numeric_guarantee_payload_mismatch`（precision）/`region_operation_bound_mismatch`（bound）/
  `region_operation_numeric_class_mismatch`/`region_fallback_payload_mismatch`/
  `region_selected_payload_mismatch`/`region_contract_digest_mismatch`/
  `region_plan_schema_version_mismatch`/`region_operation_opcode_mismatch`（op_name 不是该
  opcode 的 Core IR 拼写）/`region_operation_path_invalid`（path 非扁平
  `body.ops[N]` 或函数内非递增）/`region_operation_request_inconsistent`（role）/
  `region_precision_projection_mismatch` 结构化拒绝；
- 序列化往返（`load_region_plan` / `RegionPlan.from_dict`）digest 不变；对 boundary / 整图 /
  零输入 iota 三种产出各跑 `plan → validate(re_resolve=False) → load → explain` 往返；
- **scalar 操作数锚定**：`where` fill / `add` / `mul` 的 scalar region 均做
  `plan → validate → load → explain` 往返；scalar operand 的 `kind` 进 preimage（删除 →
  `region_digest_mismatch`），重签后仍被边界交叉检查拒绝（`region_boundary_type_mismatch`）；
- **scalar 结果锚定（round-5）**：`reduce_sum` 声明 scalar 结果 → 作为 scalar 操作数被 `add`
  消费、以及作为 region exit（`source=operation_result`），均完成全往返；删除/改写 result
  `kind` 或伪造 tensor↔scalar 标签：不重签 → `region_digest_mismatch`，重签后 →
  `region_operand_kind_mismatch` / `region_boundary_type_mismatch`；
- **多输出锚定矩阵（round-6）**：改 `request.outputs` 的 name（不重签 → digest mismatch；
  重签且用于 exit → `region_boundary_producer_mismatch`；重签且是未使用结果 →
  `region_op_plan_digest_mismatch` 由 replay 抓住）、shape（重签 → 冻结 verifier
  `output_shape_mismatch` → `region_operation_request_inconsistent`）、顺序（互换 → 契约/def-use
  拒绝）、删除/新增条目（→ `output_count_mismatch` / duplicate name）；
  boundary exit 的 `output_index` 互换或 exit 列表重排（重签）→
  `region_boundary_output_index_mismatch` / `region_boundary_order_mismatch`；
- **版本拒绝**：round-1 文档（schema=1，digest `626cc908…`）→ `region_schema_version_legacy`；
  真实 v2 形状文档（exit 无 `output_index`、v1 形状 `region_precision`、schema-2 digest）→
  `region_schema_version_v2_unsupported`（`re_resolve=False/True` 都一样，先于任何语义解释）；
  当前 v3 文档正常接受；重签后的"空 operations"文档 → `region_operations_empty`；
- `plan_region` 返回前自检：自产 document 不能通过结构校验即 plan 失败；
- capability 变化（portable provider version 改一位）→ region digest 变化并可用新 snapshot replay；
- **显式 capability 不 probe**：monkeypatch `CapabilitySnapshot.probe_host` 抛异常后，
  `plan_region(..., capability=<snapshot>)` 与 `plan_operation(..., capability=<snapshot>)`
  仍正常工作（无 `/proc/cpuinfo`、无 pinned library dlopen）；默认路径（不传 capability）
  仍按 lazy 一次性探测。

### 7.3 零回归（单算子路径不变 + 新的 v3 单输出 region digest 钉）

证据：`raw/single_op_digests_923a72e26.json`、`raw/single_op_digests_tip.json`、
`raw/single_op_digest_zero_regression.json`（同一主机、同一 capability digest `e0a9dac8…`）：

| 项 | 923a72e26 | tip `a8722aeb8` | 结论 |
|---|---|---|---|
| `plan_matmul` f32 (2,3)×(3,4) | `e0d1287e…` | `e0d1287e…` | 逐字节一致 |
| `plan_matmul` bf16 | `7e95c65c…` | `7e95c65c…` | 逐字节一致 |
| `plan_qmatmul` (2,3)×(3,4) | `1717a8e5…` | `1717a8e5…` | 逐字节一致 |
| `plan_operation` matmul/qmatmul | 不存在（0048 新增） | 与专用入口一致 | 委托逐字节一致 |

单算子路径（`resolver.py`/`plan.py`/provider/契约）在 round-6 未改：`test_execution_matmul.py` /
`test_execution_ops_registry_batch1.py` / `test_execution_discovery.py` 全部保持通过。
region 侧因为 schema v3 + `output_index` 加字段，**所有 v2 region digest 都不再可比**；本切片
新增 host-guarded v3 单输出 digest 钉：`exp`+`reshape` 的 region digest
`7b81270154393ffa…`（capability digest `e0a9dac8…`；测试与单算子 baseline 共用 host guard）。
单输出 `request.outputs` 的字段拼写（tensor `{name,shape,dtype}` / scalar 加 `kind`）与 v1/v2
逐字节一致，多输出才会长出 `role`/`kind`。

新测试 `python/tests/ut/pypto_x/test_execution_region_planning.py` 在父提交 `4521558e6`
上 **collect 即 ModuleNotFoundError**（`No module named 'pypto.execution.region'`），非空转。
round-2 另外把验收方的最小复现脚本语义纳入测试：
`raw/round2_repro_results.json`（precision forgery 两段、fallback forgery、boundary 往返两种、
iota 往返、嵌套/block 拒绝、显式 capability 无 probe）。

### 7.4 focused / 规则 8

- focused（未持锁，round-6）：`test_execution_region_planning.py` →
  **79 passed**；与既有 `test_execution_matmul.py` +
  `test_execution_ops_registry_batch1.py` + `test_execution_discovery.py` 合并 →
  **322 passed**；batch2+batch3 契约套件单跑 **194 passed**
  （`logs/round6_focused_region.log`、`logs/round6_focused_combined.log`）。
- 规则 8 全量（hold local lock，6 线程，命令
  `env PYTHONUNBUFFERED=1 PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut
  python3 -m pytest -q -rs -p no:cacheprovider python/tests/ut/pypto_x`）：
  **collect 2755 / passed 2740 / skipped 15 / failed 0 / rc 0，耗时 21:22（1282.36 s）**
  （`logs/round6_rule8_full.log`、`logs/round6_rule8_rc.txt`、
  `logs/round6_rule8_elapsed_seconds.txt`、`logs/round6_rule8_summary.txt`）。15 个 skip =
  7 个本机无 CUDA driver（既有）+ 6 个 batch1..4 host-guarded 基线钉 + 2 个 region
  host-guarded 钉（既有单算子 baseline 1 个 + 本切片新增的 v3 单输出 region digest 1 个；
  heavy 锁的 capability digest 与默认环境不同，这是既有 skip-guard 机制）。
- 过程如实记录：第 1 次锁内全量运行（`logs/round6_rule8_full_preempted.log`）跑到约 31%
  被外部 supervisor 以 SIGTERM 抢占（rc=143，
  `logs/round6_lock_attempt_preempted_20260920T0639.log`）；队列改为"任何非零 rc →
  sleep 300 重试"，第 8 次取得锁后完整跑完（`logs/round6_lock_attempts.txt`）。

## 8. 登记边界（未做与已知限制）

- **不执行**：本切片没有 `execute_region_plan`，没有 runtime/launch 调用，没有 v5 报告；
- **单算子路径零语义变化**：`resolver.py` / `plan.py` / provider 实现未改；
  round-2 仅为验收要求把 `entry.capability_for_policy` 改成 **lazy probe**（显式 capability
  不再无条件 `probe_host()`），默认路径行为与 digest 逐字节不变（§7.3 重新验证）；
  `pypto/__init__.py` 未改（region API 经 `pypto.execution` 显式导入）；
- **scalar 操作数与结果（round-4/5）**：tensor/scalar 操作数一律委托 opcode 契约判定；
  scalar 以 `kind="scalar"` + `shape=[]` 进入逐 op 条目的 operand/result 与边界签名（进
  digest）；scalar 结果可被后续 scalar 操作数消费，也可作为 region exit（operation_result 或
  entry_pass_through）；def-use 对 producer/consumer kind 交叉检查；`kind ∉ {tensor, scalar}`
  的 operand/result 仍在 region 层结构化拒绝。**不做** scalar→tensor 隐式改写；
- **多输出（round-6）**：支持**静态可判定结果数**的多输出契约（当前 `split`；结果数由
  sizes/num_splits 决定），`request.outputs` 有序 5 字段、def-use 按 `(op, output_name)`、
  exit 按 `(producer_op_index, output_index, name)`、`region_precision.outputs` 逐输出独立合成。
  **不做**：运行时可变结果数、多输出 + scalar-kind 结果的组合（`split` 明确
  `multi_output_scalar_result_not_supported`）、跨 provider 的多输出语义差异/融合、输出级
  复用/别名规划（`may_alias` 只在单算子契约内声明）；
- **仍不支持的请求**：无输出（`operation_has_no_result`）、单输出契约却带多结果
  （`multi_output_not_supported_by_execution_contract`）、非 tensor/non-scalar 结果、
  动态/符号 shape；全部结构化拒绝，不静默降级；
- **嵌套/block region 明确拒绝**：`region_nested_region_not_supported` /
  `region_block_region_not_supported`；扁平 `CoreProgram`（Qwen 三个真实图均为扁平）是唯一
  已冻结路径。round-2 前它们会被"拍平"进计划，无法表达嵌套作用域，属实现与文档不一致，现已
  按 fail-closed 修正；
- **best_effort 在本树不可达**：现有 resolver 的 requirement 判定（portable/deterministic/bounded）
  都强于 best_effort，因此 shipped provider 无法产出 best_effort plan；合成 term 覆盖了组合器
  路径并验证"best_effort region 不可用于 require policy"；
- **性能**：region 规划耗时（round-6 decoder 整图约 38 s，含 digest 与自检）只是本机
  UNGATED 观测，不作为性能声明；
- **vendor 依赖**：AOCL 相关对抗用例依赖本机 pinned artifact 的可用性（测试中 skip-guard）；
- **显式 capability 的 probe 边界**：默认/显式 capability 路径都不再无条件 probe；仅当 policy
  显式 opt-in 0047 exact-blocked 时，其 manifest 由 pinned artifact 探测构造（policy 语义
  需要），此时会有一次该 artifact 的探测；默认与 portable 路径无 `/proc/cpuinfo`、无 dlopen；
- **capability replay**：用非默认/合成 capability 生成的 region plan，`load_region_plan`
  / `validate_region_plan(re_resolve=True)` 必须显式传入同一 capability snapshot；
  否则按 fail-closed 拒绝（`region_capability_digest_mismatch`）；
- **v1 region plan 不可 load**：round-1 的 schema=1 文档在 r2/r3 之后一律
  `region_schema_version_legacy` 拒绝（fail-closed 正确）；没有旧 artifact 需要兼容，因为该
  schema 从未随实现提交发布；
- **下一步（不属于本切片）**：`execute_region_plan`（第二步）与 region 级 report v5（第二步）、
  region workspace/缓存/并行调度（第三步）。
