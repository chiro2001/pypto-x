# 0021 — U5 region 只读规划（batch 0049 切片 A）

- 任务：`u5-region-readonly`
- 实现基线：integration tip `4521558e6`；实现 commit `46daf77d2`（分支 `work/u5-region-readonly`，未 push）
- 依赖：`0019`（opcode 契约第一批，已落地）、`0018`（composite 组合化，已落地）、父方提案 `0020`
- 结论等级：**实现完成（只读）**。未做执行、未做 v5 报告、未做动态 shape、未做性能声明；本机数字一律 UNGATED
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u5-region-readonly/`

## 0. 摘要

本切片把 "图" 提升为一等公民的**第一步（只读）**：

- `plan_region(program, region_spec, policy, capability=None) -> RegionPlan`：把 `CoreProgram`
  中由静态 def-use 连通的算子集合冻结成 `REGION_SCHEMA_VERSION=1` 的 region plan 文档，
  逐 op 复用 `plan_operation`（不新造 resolver），产出 `region_digest` 与 region 级精度声明；
- `explain_region(...)` / `explain_region_plan(...)`：逐 op 复用 U3 的 explain 文档
  （候选 / 拒绝原因 / tie-break / fallback / 边界表），region 级给出可复算的合成推导；
- `validate_region_plan(...)` / `load_region_plan(...)`：严格校验器。region digest 从文档自身
  复算；逐 op `single_op_plan_digest` 与 `plan_operation` 重放结果双向锚定；篡改一位即拒绝；
- CLI：`plan --region` / `explain --region` / `explain --region-plan`，退出码沿用既有约定
  （请求级错误 1，argparse 用法错误 2，doctor --strict 3 不变）；
- **不执行**：region 模块只调用 resolution 路径（`plan_operation`），不导入 runtime，不调用
  launch；单算子路径（`plan_matmul` / `plan_qmatmul` / `plan_operation`）行为与 digest 逐字节不变。

最有价值的副产品：对真实 Qwen3.5 图（attention / GDR / 全 decoder 4550 ops）跑整图
`plan_region`，得到**按 opcode 聚合的缺失 contract 清单**（见 §5），直接给出 batch2/3 的
opcode 注册优先级。

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

### 1.3 边界合法性（结构化拒绝清单）

`plan_region` 先跑 `CoreProgram.verify()`，再按下列顺序 fail-closed：

| 触发 | reason |
|---|---|
| program 不是 CoreProgram / verify 失败 | `region_program_invalid` / `region_program_invalid` |
| `region_spec` kind/字段非法 | `region_spec_kind_invalid` / `region_spec_unknown_field` / `region_spec_boundary_incomplete` 等 |
| 边界 tensor 在函数根作用域不可见（悬空） | `region_boundary_dangling_tensor` |
| 多函数 program 的 boundary spec 未指定 function | `region_function_required` / `region_function_not_found` |
| region 内算子读取未声明的外部值（跨 region 引用/隐式捕获） | `region_boundary_undeclared_input` |
| 声明的 entry 没有被 region 消费 | `region_unused_entry_input` |
| 声明的 entry 由 region 内算子产生 | `region_entry_input_defined_in_region` |
| entry 的 producer 在 region 起始之后 | `region_boundary_input_defined_after_use` |
| exit 不是 region 内算子的结果 | `region_exit_output_not_produced_by_region` |
| 前向引用/自引用（闭环、逆拓扑） | Core IR verify 或 `region_cycle_detected` |
| 边界选中 0 个算子 | `region_empty` |
| 边界值不是 tensor（标量 entry/exit） | `region_boundary_non_tensor` |

所有错误都带 `code`（沿用 `OP_CONTRACT_INVALID`）、`field`、`requested`、`reason`、`details`，
可用 `to_dict()` 机器读取。

## 2. 冻结 region plan（`REGION_SCHEMA_VERSION=1`）

```jsonc
{
  "region_schema_version": 1,
  "kind": "region_plan",
  "region_id": "region.<region_digest hex 前 16 位>",
  "region_digest": "sha256:...",
  "region_spec": {"kind": "program", "function": null},   // 或 boundary + 名字列表
  "policy": { ... ExecutionPolicy.to_dict() ... },
  "policy_digest": "sha256:...",
  "capability_digest": "sha256:...",        // 传入/探测的 base capability snapshot
  "resolver_version": 2,
  "boundary_signature": {
    "entry_inputs": [{"function","name","role","source","kind","dtype","shape","producer_op_index"}],
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
  "region_schema_version": 1,
  "boundary_signature": <boundary_signature>,
  "operations": [
    {"index": i, "function": fn, "opcode": opcode, "single_op_plan_digest": digest_i},
    ... 按拓扑序 ...
  ]
}))
```

要点：

- 有序 op digest 列表 + 边界签名 + schema 版本三者进 preimage；`region_id` 由 digest 派生；
- 逐 op plan digest 自身绑定 policy digest、capability digest、contract digest、precision/envelope；
  因此 **capability 变化如果影响任一 op 的解析，必然改变 region digest**；与 region 语义无关的
  变化也在 per-op plan payload 的 `target.capability_digest` 中体现。文档另存 base
  `capability_digest` 供 replay 锚定。
- 同 program + spec + policy + capability 两次解析的文档逐字节相同（测试钉住）。
- 边界签名中的 entry/exit 顺序按 `(function, name)` 规范排序，与用户声明顺序无关。

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
    "outputs": [{"name","shape","dtype"}]
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
  "precision": {...} | null,
  "envelope": {...} | null,
  "fallback": {"mode","primary","chain","used","degraded","events"},
  "bound": {"declared_class","proven","bound","bound_kind","is_upper_bound","reference","reasons"}
}
```

`single_op_plan_digest` 就是 `plan_operation(opcode, input_shapes, input_dtypes,
output_shape=..., output_dtype=..., attributes=..., policy=<文档内 policy>, capability=<replay
snapshot>)` 得到的 `ExecutionPlan.digest`；`request` 字段足以在**不依赖源 program** 的情况下重放。

严格校验器 `validate_region_plan(document, *, policy=None, capability=None, re_resolve=False)`
按固定顺序检查（任一失败抛 `ARTIFACT_MISMATCH`，不修复）：

1. 顶层 schema/kind、必填字段、未知字段、`resolver_version` 精确匹配；
2. `boundary_signature` 结构（角色/类型/静态 shape/entry 不得有 region 内 producer/exit source 语义）；
3. **从文档自身复算 `region_digest`，再复算 `region_id`**（改一位 → `region_digest_mismatch` /
   `region_id_mismatch`）；
4. 逐 op 结构 + `bound` 视图必须等于从 `precision`/`envelope` 重新归一化的结果
   （改 `bound` → `region_operation_bound_mismatch`）；
5. 文档内部 def-use 闭包（重复定义、正向引用、隐式捕获、exit producer 一致性、
   entry/exit 与 request 的类型一致）；
6. `region_spec` 与边界/算子集合语义一致（region_spec 不进 digest，改用语义交叉校验）；
7. `policy` 复算 digest；`region_precision` 必须逐字节等于从 op 列表重新合成的结果；
8. `op_count == len(operations)`；
9. `re_resolve=True` 时：重放每个 op 的 `plan_operation`，要求 `single_op_plan_digest`、
   `contract_digest`、`provider_id` 一致，并要求传入 capability 的 digest 等于文档
   `capability_digest`（`region_capability_digest_mismatch`）。未显式传 capability 时探测本机，
   探测 digest 必须与文档一致，否则拒绝（合成 capability 的计划必须显式带 snapshot 才能 replay）。

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
- `portable_bitwise`：`{kind: portable_bitwise_trunk_identity, sum: 0, is_upper_bound: true}`，
  声明的是"与顺序执行的 portable trunk 逐位一致"，不是"与数学参考的误差界"；
- `deterministic_bounded`：`{kind: sum_of_per_operation_proven_bounds, sum: Σ, is_upper_bound: ?}`。
  逐项列出 `terms`（index/opcode/provider/bound_kind/bound/is_upper_bound/reference）。
  `is_upper_bound=true` 当且仅当：
  1. 每一项的界都是 proven，且可数值相加（`int` 或同 key 的 `by_k` 逐项相加）；
  2. 非零界项的 reference 一致（portable 恒等项是 0，不改变参考系）；
  否则给出 `cross_reference_bounds_not_comparable` / `nonzero_bound_has_no_reference` 等理由，
  `is_upper_bound=false`，且在 `require_proven_deviation_bound=true` 下 region 级 fail-closed。
- `deterministic_bounded_unproven` / `best_effort`：`{kind: none, sum: null, is_upper_bound: false}`，
  不做任何"求和成界"声明。

`region_precision["derivation"]` 记录 5 个布尔检查（best_effort / unproven / all-exact /
proven-bounded / portable-present）、`selected_class`、`selected_rule`、
`no_error_cancellation=true`、`order_sensitive=true`、`topological_order_bound=true`，
可被 `validate_region_plan` 逐字节复算。

## 5. 真实 Qwen 图上的缺失 contract 清单

证据：`raw/qwen_region_missing_contracts.json`（脚本 `scripts/qwen_region_missing_contracts.py`；
对三个 builder 各跑一次 `plan_region(整个 program)`；read-only）。三个图全部**结构化拒绝**——
这是预期结果，因为 batch1 只登记了 10 个 opcode。

| builder | ops | 缺失 contract 的 op（出现次数） | 已登记但请求不可规划 |
|---|---|---|---|
| `build_attention_kv_cache` \((1,8,4,256)\) | 20 | broadcast 2, concat 2, gather 2, div 1, mul 1, sub 1, transpose 1 | `where` ×1（scalar operand） |
| `build_qwen35_gdr_recurrent_state` (1,4) | 123 | transpose 28, slice 20, mul 12, broadcast 8, add 4, sub 4, identity 2, concat 1 | — |
| `build_qwen35_text_decoder_graph` (1,1,past=4096) | 4550 | 见下表 | `where` ×6（scalar operand）、`compare` ×1（`broadcast` 属性不在 compare 契约白名单 → `unknown_attribute`） |

全 decoder / attention / GDR 合并后的缺失 contract 频次（按出现次数排序，机器可读字段
`missing_contract_priority`）：

| 优先级 | opcode | 合并出现次数 | 覆盖图 |
|---|---|---|---|
| 1 | `mul` | 544 | decoder + gdr + attention |
| 2 | `transpose` | 414 | decoder + gdr + attention |
| 3 | `broadcast` | 360 | decoder + gdr + attention |
| 4 | `slice` | 356 | decoder + gdr |
| 5 | `add` | 330 | decoder + gdr |
| 6 | `rsqrt` | 115 | decoder |
| 7 | `reduce_mean` | 79 | decoder |
| 8 | `concat` | 69 | decoder + gdr + attention |
| 9 | `constant` | 66 | decoder |
| 10 | `silu` | 60 | decoder |
| 11 | `neg` | 30 | decoder |
| 12 | `sub` | 29 | decoder + gdr + attention |
| 13 | `sigmoid` | 24 | decoder |
| 14 | `split` | 24 | decoder |
| 15 | `softplus` | 18 | decoder |
| 16 | `div` | 7 | decoder + attention |
| 17 | `gather` | 2 | attention |
| 18 | `identity` | 2 | gdr |
| 19 | `embedding` | 1 | decoder |

### 5.1 batch2/3 优先级建议

1. **batch2 第一优先级：`add/sub/mul/div/neg/broadcast`**。它们合计 1270+ 次，是 decoder
   逐元素算术主干；`mul` 单个就 544 次，没有它 attention/GDR/decoder 全部 region 不可规划。
2. **batch2 第二优先级：`transpose/slice/concat/split/constant/identity`**（shape/数据搬运 +
   常量）。`transpose` 414、`slice` 356、`concat` 69、`split` 24；`constant` 66 是唯一
   "无输入产生张量"的 opcode，注册前需要决定 value 属性的 canonical 域（建议先只支持
   标量/小 shape 的字面量）。`view`/别名语义仍需按 `0019 §6.1` 先冻结 `must_alias` 义务。
3. **batch2 第三优先级（激活家族）：`rsqrt/silu/sigmoid/softplus/reduce_mean`**（合计 296）。
   与 `exp` 同族，按 `deterministic_bounded`（无解析界）登记即可，不要为了"看起来精确"
   把 libm 路径写成 exact。
4. **batch3：`gather/embedding`（3 次）**。频次低但 attention/decoder 各自需要，属于
   数据搬运 + index dtype 语义，单独成片。
5. **两个当前已登记但真实图请求不可规划的点**（不是缺 contract，不应混入注册优先级）：
   - `where` 的 fill 是 scalar operand → `0019 §8` 明确"标量操作数不在本批执行契约内"。
     真实图要用 `where` 必须先扩展标量 operand 契约，或由前端把 fill 提升为张量；
   - `compare` 的 Core IR 属性含 `broadcast`，而 compare 契约白名单只有 `predicate` 族
     → `unknown_attribute`。需要决定是给 compare 契约增加 `broadcast`（语义等价于尾轴广播，
     需契约升版）还是让构图侧不写该属性。两件事都必须单独升契约版本，不能在本切片悄悄放宽。

以上频次是**注册优先级输入**，不是执行承诺；每个 opcode 仍需按 0019 的 verifier/precision
模式补齐契约、差分矩阵与零回归。

## 6. explain 与 CLI

### 6.1 `explain_region` / `explain_region_plan`

复用 U3 的 explain 机制，不新造第二套解释层：

- 每个 op 的 `operations[i]` 是 `discovery._explain_document(plan, "region_op", ...)` 的原样输出，
  含 `candidates` / `rejections` / `tie_break` / `fallback` / `rules` / `boundaries`，
  另加 `region_op_index`、`region_opcode`、`bound_view`、`region_fallback`；
- region 级 `composition` 给出 `derivation`（与文档内 `region_precision.derivation` 相同）、
  `bound_contributors`（哪些 op 贡献界）、`unproven_ops`、`provider_set`、`deviation_bound`、
  `fail_closed` / `fail_closed_reasons`；
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

### 7.2 确定性 / 自证 / 篡改

- 同 program+spec+policy+capability 两次 → 文档逐字节相同、digest 相同；
- `validate_region_plan` 从文档自身复算 `region_digest` / `region_id` / 逐 op `bound` 视图 /
  `region_precision` / `policy_digest` / def-use 闭包；`re_resolve=True` 再逐 op 与
  `plan_operation` 重放比对；
- 篡改矩阵（op digest 翻一位、边界 source 改写、删 op、rewrite bound、rewrite region class、
  rewrite region_id、rewrite policy/capability digest、未知字段）全部结构化拒绝；
- 序列化往返（`load_region_plan` / `RegionPlan.from_dict`）digest 不变；
- capability 变化（portable provider version 改一位）→ region digest 变化并可用新 snapshot replay。

### 7.3 零回归（923a72e26 → `46daf77d2`）

证据：`raw/single_op_digests_923a72e26.json`、`raw/single_op_digests_tip.json`、
`raw/single_op_digest_zero_regression.json`（同一主机、同一 capability digest `e0a9dac8…`）：

| 项 | 923a72e26 | tip `46daf77d2` | 结论 |
|---|---|---|---|
| `plan_matmul` f32 (2,3)×(3,4) | `e0d1287e…` | `e0d1287e…` | 逐字节一致 |
| `plan_matmul` bf16 | `7e95c65c…` | `7e95c65c…` | 逐字节一致 |
| `plan_qmatmul` (2,3)×(3,4) | `1717a8e5…` | `1717a8e5…` | 逐字节一致 |
| `plan_operation` matmul/qmatmul | 不存在（0048 新增） | 与专用入口一致 | 委托逐字节一致 |

新测试 `python/tests/ut/pypto_x/test_execution_region_planning.py` 在父提交 `4521558e6`
上 **collect 即 ImportError**（`cannot import name 'RegionPlan'`），非空转。

### 7.4 focused / 规则 8

- focused（未持锁）：`env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut
  python3 -m pytest -q -rs -p no:cacheprovider
  python/tests/ut/pypto_x/test_execution_region_planning.py` → **26 passed**（约 1.8 s）；
  与既有 `test_execution_ops_registry_batch1.py` 合并跑 → **151 passed**。
- 规则 8 全量（hold local lock，6 线程）：**（见 `logs/rule8_summary.txt` / `logs/rule8_rc.txt`）**。

## 8. 登记边界（未做与已知限制）

- **不执行**：本切片没有 `execute_region_plan`，没有 runtime/launch 调用，没有 v5 报告；
- **不动单算子路径**：`entry.py` / `resolver.py` / `plan.py` / provider 实现零改动；
  `pypto/__init__.py` 未改（region API 经 `pypto.execution` 显式导入）；
- **不支持的请求**：标量 operand（`where` fill 等）、多输出/无输出、非 tensor 结果、
  动态/符号 shape；这些全部结构化拒绝，不静默降级。带 block region / 嵌套 op region 的程序
  受 Core IR `verify()` 的可见性规则限制（嵌套 region 输出不泄漏回父作用域、block 输出不能
  作为函数返回），当前以 `region_program_invalid` 结构化拒绝；扁平的 `CoreProgram`（Qwen
  三个真实图均为扁平）是已验证路径；
- **best_effort 在本树不可达**：现有 resolver 的 requirement 判定（portable/deterministic/bounded）
  都强于 best_effort，因此 shipped provider 无法产出 best_effort plan；合成 term 覆盖了组合器
  路径并验证"best_effort region 不可用于 require policy"；
- **性能**：region 规划耗时（decoder 整图约 15 s）只是本机 UNGATED 观测，不作为性能声明；
- **vendor 依赖**：AOCL 相关对抗用例依赖本机 pinned artifact 的可用性（测试中 skip-guard）；
- **capability replay**：用非默认/合成 capability 生成的 region plan，`load_region_plan`
  / `validate_region_plan(re_resolve=True)` 必须显式传入同一 capability snapshot；
  否则按 fail-closed 拒绝（`region_capability_digest_mismatch`）；
- **下一步（不属于本切片）**：`execute_region_plan`（第二步）与 region 级 report v5（第二步）、
  region workspace/缓存/并行调度（第三步）。
