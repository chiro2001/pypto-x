# 0026 — 多输出算子 ABI 与 `split` 登记（batch 0053 切片 A）

- 任务：`u-multi-output-abi`（batch 0053 切片 A）
- 基线：集成树 tip **`ec7a9edc2`**，tree `6881c346bbc9462eca570b2cd0e9329b84fd9daf`
- 分支：`work/u-multi-output-abi`（worktree `/home/chiro/projects/pypto/worktrees/pypto-x/u-multi-output-abi`，未 push）
- 实现 commit：`2c5bab429`（execution 代码）+ `d4d06a971`（focused 测试）；HEAD tree `464e3a7149cb873a048c137799e80faa41384986`；
  基线 tree `6881c346bbc9462eca570b2cd0e9329b84fd9daf`；未 push（落地时由父方 cherry-pick/合入 integration）
- 结论等级：实现完成（未做性能声明；所有本机数字 UNGATED）
- 范围：0024 提案的 **step 1（ABI 层）+ step 2（登记 `split`）**；region 多输出（step 3）、`view`/`contiguous`/量化 primitive 不在本切片

## 0. 摘要

本切片交付 0024 提案的 step 1 + step 2：

1. **契约层**：`OpDefinition` 新增**可选**的有序 `OutputSpec` 结果表（`name`/`role`/dtype 规则/shape 规则）；
   现有 28 个单输出契约不声明该表面，contract/capability/plan/report digest 逐字节不变。
2. **`split`**：真实 Qwen 图最后一个缺失契约类（24 次）登记为第一个多输出契约；冻结 axis/dim 归一化、
   显式 `sizes`（不等分）与 `num_splits`（等分）两种互斥模式、静态可判定结果个数、逐输出 shape/dtype 规则、
   空段/空轴语义、`may_alias` + 允许物化（copy 字节数=全部输出之和，进 plan digest）、精度 `exact`。
3. **plan**：多输出契约的 `request.outputs` 与 `selected.outputs` 均为有序列表；`selected.outputs` 逐项携带
   `name/role/dtype/shape/dtype_rule_digest/shape_rule_digest`，顺序与全部字段都在 `plan_digest` 覆盖范围内；
   单输出 plan payload 逐字节不变（`plan_schema_version` 仍为 3）。
4. **report**：v4 新增**可选**顶层 `outputs` 块（有序，逐项 `name/role/dtype/shape/digest`），与内嵌 plan
   payload、`request.outputs` echo、`request.output_shape`/`output_dtype` 互证；单输出报告逐字节不变
   （`REPORT_SCHEMA_VERSION` 保持 4，理由见 §5）。
5. **执行**：多输出算子的 `execute_operation` 返回**有序 `HostTensor` 元组**；`output=None` 时按冻结结果逐项
   分配缓冲，或接受等长的 HostTensor 序列；缓冲数/形状/dtype、输入别名、输出间别名全部 fail-closed。
6. **主验收指标（实测，真实三图 `plan_region`）**：`missing_contracts` **1 类/24 次 → 0 类/0 次**；
   但 `unplannable` 从 0 → **1 类/24 次（`split`，reason `multi_output_not_supported_by_execution_contract`）**，
   因为 `region.py` 的结果个数前置检查（第 1688–1701 行）属于 0024 step 3，本切片按纪律未改；
   24 个真实 `split` 在**单算子 ABI 层** 24/24 规划并执行成功（`abi_level_split_planned=24`），
   证明剩余缺口只在 region 边界表达，不在 ABI。`registered_planned_total` = **4669**（非 split 全部 plan 成功）。

### 0.1 关键设计决定（摘要）

| 决定 | 选择 | 理由 |
|---|---|---|
| 契约表面 | `OutputSpec`（必填 dtype/shape rule，role 取闭集）；仅声明它的契约进入多输出 ABI | 单输出文档/摘要零变化 |
| 结果个数 | 由 opcode+attributes+rank **静态判定**；`split` 用 `sizes` 长度或 `num_splits` | 0024 §1 硬要求 |
| `sizes`/`num_splits` | **互斥**；`sizes` 不等分、`num_splits` 等分整除 | 单一事实来源，避免 lowering 的隐式优先级 |
| 空段 | 拒绝（size 必须 ≥1）；空 split 轴也拒绝 | 避免零宽结果歧义；与 lowering 的 0 容许相比更严格 |
| 视图义务 | `may_alias` + 允许物化；copy 字节数 = Σ 各输出字节，进 plan digest | 与 batch-2 视图族同源；portable 恒物化 |
| 精度 | 全部 dtype 路径 `exact`（纯搬运，逐位保留 NaN/±0/次正规/bf16 位型/bool） | 无算术、无转换 |
| plan 版本 | 保持 `plan_schema_version=3` | 新键只出现在新契约；28 个既有 plan payload 逐字节不变 |
| report 版本 | 保持 `REPORT_SCHEMA_VERSION=4`，`outputs` 为可选块 | 单输出报告逐字节不变；多输出报告严格解码、缺块即拒 |
| 结果名 | 调用方可给 Core IR 名字；缺省 `output0..N-1` | 多输出结果名是声明值，图名由 step 3 的 region 层传入 |

## 1. 契约层（`OpDefinition` / `OutputSpec`）

### 1.1 可选有序 `outputs`

- 新增 `pypto.execution.op_registry.OutputSpec`：`name`/`role`/`dtypes`/`rank`/`layout`/`dtype_rule`/`shape_rule`，
  可选 `repeat_rule`（“同一模板按 attributes 静态展开 N 项”）。
- `role` 限定闭集 `OUTPUT_ROLES = ("output","segment","values","indices","remainder")`；未知 role 构造即拒
  （`output_role_not_registered`）。
- `OpDefinition.outputs` 里只要有 `OutputSpec` 就是多输出契约；混用 `OperandSpec`/`OutputSpec`、
  空结果表、重复 name、非 generic verifier、空 dtype 域、非法 rank 范围都会在构造期结构化拒绝。
- 文档：仅多输出契约的 `outputs[i]` 多出 `dtype_rule`/`shape_rule` 键，并新增顶层 `output_count_rule`；
  `output_count_rule` 参与 contract digest。
- **现有 28 个契约**（matmul/qmatmul v2 文档 + batch1–4 的 v3 文档）不声明 `OutputSpec`，
  `OutputSpec`/`_normalize_generic_outputs`/多输出方法对它们全是惰性分支：**contract digest 逐字节不变**（§8 有 28 项钉住）。
- 注册表 schema 仍报告 `OP_REGISTRY_SCHEMA_VERSION = 3`：新契约沿用 v3 文档，只新增可选键；
  不升 registry schema 的理由与 batch3/4 相同——升号会改写全部 v3 文档摘要。

### 1.2 `split` 契约（冻结）

| 项 | 值 |
|---|---|
| canonical opcode / aliases | `split`（OP_ALIASES 全拼写，含 `tensor.split`/`core.split`/`pypto.split`/`pypto.op.joining.split`/`pil.split`） |
| contract_version / document schema | `1.0` / 3 |
| 元数 | 1 个 tensor 输入，rank 1..4 |
| 结果个数 | `len(sizes)` 或 `num_splits`，静态可判定 |
| role / 缺省名 | `segment` / `output0..N-1` |
| dtype | 与输入相同，portable 全域（bool 含）；无隐式转换 |
| shape | `shape_i = input_shape` 把归一化 axis 维替换为 `sizes[i]` |
| 精度 | `exact`（`exact_contract_bound`，deviation 0） |
| view_obligation | `may_alias`；`allows_materialize_copy=true`；copy 字节=Σ element_count(output_i)×dtype_size |
| 允许属性 | `axis`/`dim`、`sizes`/`split_sizes`/`sections`、`num_splits`；其余未知属性拒绝 |
| contract digest | `sha256:4835891032cf430f179b2ed66f16f8007ceb386aa685a8a5f5eb155ed6fd18b9` |
| scoped capability 视图 | 29 契约（batch4 28 + split），digest `sha256:f73c337c9fe9824a72d607a6b67fff7b5b847bd75667bf4fd8964da3ac847e43` |

语义文字（写入契约 `semantics`，并逐条有用例）：

- **axis**：`axis`/`dim` 缺省 0；两者同现必须归一化一致（否则 `conflicting_attribute_aliases`）；
  负轴按 rank 加一次；越界/非整数 → `split_axis_invalid`。
- **sizes 模式（不等分）**：`sizes`/`split_sizes`/`sections` 为有序 list/tuple（set/dict 拒绝
  `unordered_attribute_container`）；每项为正整数，和必须等于 axis 长度：
  - 非整数/bool/负数 → `split_sizes_invalid`；
  - 0 → `split_empty_segment_not_supported`；
  - 空表 → `split_sizes_invalid`；
  - 多个别名不一致 → `conflicting_attribute_aliases`；
  - 和不等 → `split_sizes_do_not_cover_axis`。
- **num_splits 模式（等分）**：正整数且整除 axis 长度：
  - 非整数/bool → `split_num_splits_invalid`；
  - ≤0 或 > axis 长度 → `split_num_splits_out_of_range`；
  - 不整除 → `split_num_splits_not_divisible`；
  - 归一到显式 `sizes`（plan 的 `request.attributes` 永远只冻结 `{"axis": int, "sizes": [...]}`，
    两种模式在 plan 里同一拼写，executor 只认这一种）。
- **两种模式互斥**：同现 → `split_attribute_conflict`；都不给 → `split_sizes_required`（“不可静态判定”）。
- **空段/空轴**：size 必须 ≥1；axis 长度必须 ≥1（`split_axis_empty_not_supported`）。
  非 axis 维可以为 0，产生 0 元素结果。
- **值与位型**：纯搬运；NaN 位型不承诺（HostTensor Python float 存储）、±0 符号、±inf、次正规、
  bf16 读入位型、bool 值原样保留（§7 差分与位型用例）。
- **view/别名**：契约 `may_alias`（有 stride 描述符的 provider 可以零拷贝），portable 恒物化 N 个独立行主序缓冲；
  `view_mode=require` 在无零拷贝 alias proof 时结构化拒绝（`ALIAS_PROOF_UNAVAILABLE`），不静默降级；
  runtime 仍要求输出与输入、输出之间存储互异。

## 2. plan 层

- 多输出契约（`definition.multi_output`）总是写：
  - `request.outputs = [{name, role, kind:"tensor", dtype, shape}, ...]`（有序）；
  - `selected.outputs = [{name, role, dtype, shape, dtype_rule_digest, shape_rule_digest}, ...]`（同序）。
- `request.output_shape`/`request.output_dtype` 保留为**首个结果**的 v2/v3 兼容投影；校验器强制
  `request.outputs[0]` 与之一致（所有 plan），多输出时长度 ≥2 且每项必须 `kind=tensor`、
  `name` 唯一、`role` 必填。
- `plan.validate()` 在不依赖 registry 的前提下，逐项核对 `selected.outputs` 与 `request.outputs`
  的 name/role/dtype/shape/顺序，并校验两个 rule digest 为 `sha256:`；不一致 → `plan_field_inconsistent`。
- `plan_digest` 是整份 payload 的 canonical JSON sha256：顺序、name、role、dtype、shape、两个 rule digest、
  `output_count_rule` 派生的 count、`layout_decision.layout_copy_bytes` 全部在内。改一位而不重签 →
  `plan_digest_mismatch`；重签后的语义篡改由 execution 的 canonical re-resolution 兜底（§2.1）。
- 默认名：`output0..N-1`（contract 模板名 + index）；真实图名由调用方按 Core IR result ref 传入，
  region step 3 可直接用 `_request_from_operation` 已有的 result 名。
- 单输出 plan payload 逐字节不变：`request.outputs` 仅在既有 scalar-kind 单输出路径存在（0051-A 冻结拼写），
  `selected.outputs` 不出现；规则 digest/`output_count_rule` 均不出现。`plan_schema_version` 保持 **3**。

### 2.1 重签篡改分层（真重签，无 stale digest）

| 篡改（重签后） | 拒绝层 / reason |
|---|---|
| request 与 selected 结果不一致（name/shape/role/顺序） | `ExecutionPlan.validate()` → `plan_field_inconsistent`（field `plan.selected.outputs.*`） |
| 多输出 plan 缺 `selected.outputs` | `plan_field_inconsistent`（field `plan.request.outputs`） |
| `selected.outputs` 缺 `request.outputs`/长度不等/重复 name | `plan_field_inconsistent` |
| role 改成契约外 | canonical re-resolution → `plan_policy_not_satisfiable:output_role_mismatch` |
| 逐输出 shape 改成与 sizes 不符 / count 改动 | `layout_decision_contract_mismatch`（copy 字节先变）或 `plan_policy_not_satisfiable:output_shape_mismatch` / `output_count_mismatch` |
| 只改 rule digest | `plan_selected_field_not_reconciled_with_provider_manifest`（payload 比对） |
| 顺序整体反转且投影自洽 | `plan_policy_not_satisfiable:output_shape_mismatch`（attributes 派生的 segment 形状不再匹配） |

边界（诚实登记）：结果 **name** 是声明值而非 attributes 派生量；重签且自洽的“换名”会被接受为另一份合法 plan
（digest 必然不同，非重签改名被 `plan_digest_mismatch` 拒绝）。这是“name 由 Core IR/图提供”的必然结果，
step 3 的 region `(op_index, output_name)` 边界才是名字的图级锚点。

## 3. 执行层

- `execute_operation(plan, inputs, capability=None, output=None)`：
  - 单输出 tensor：返回 `HostTensor`（不变）；
  - scalar-kind：返回原始标量（0051-A，不变）；
  - **多输出**：返回按冻结顺序排列的 `HostTensor` 元组。
- `output` 语义：多输出 plan 接受 `None`（逐结果 `HostTensor.zeros`）或等长 list/tuple；单张 HostTensor、
  标量、长度不符、逐项 shape/dtype 不符分别结构化拒绝（`unsupported_output_handle`/`plan_request_mismatch`）。
- portable provider 用冻结 `request.attributes` 重建**N 输出** Core IR program，`LaunchRequest(outputs=[...])`
  按序写回；`_check_operation_request`/`_check_no_alias_*` 校验数量/顺序/形状/dtype、输入别名
  （`output_input_alias`）与输出间别名（`output_output_alias`）。
- provider 返回结果再次与冻结声明对账（`execution_result_does_not_match_plan`），然后才构造报告。
- vendored provider 未实现多输出 → `provider_does_not_support_generic_execution`（结构化拒绝，不静默降级）；
  `allowed_providers=(portable,)` 使 vendor 在契约层就被拒。
- `execute_operation` 只校验不重解析；冻结 plan JSON `ExecutionPlan.from_dict(plan.to_dict())` 重放逐字节同 digest、
  同输出。

## 4. 结构化拒绝矩阵（新增，全部 fail-closed）

| 触发 | code / reason | 层 |
|---|---|---|
| 轴越界/非整数 | `OP_CONTRACT_INVALID / split_axis_invalid` | 契约 |
| axis/dim 冲突 | `conflicting_attribute_aliases` | 契约 |
| sizes 与 axis 长不符 | `split_sizes_do_not_cover_axis` | 契约 |
| sizes 空/负/非整数/bool | `split_sizes_invalid` | 契约 |
| 0 宽空段 | `split_empty_segment_not_supported` | 契约 |
| 空 split 轴 | `split_axis_empty_not_supported` | 契约 |
| num_splits ≤0 / >轴长 | `split_num_splits_out_of_range` | 契约 |
| num_splits 不整除 | `split_num_splits_not_divisible` | 契约 |
| num_splits 非整数/bool | `split_num_splits_invalid` | 契约 |
| sizes 与 num_splits 同现 | `split_attribute_conflict` | 契约 |
| 两者都不给（不可静态判定） | `split_sizes_required` | 契约 |
| 无序容器（set/dict） | `unordered_attribute_container` | 契约 |
| 未知属性 | `unknown_attribute` | 契约 |
| dtype 越域 | `dtype_outside_contract_domain` / `request_dtype_invalid` | 契约/resolver |
| 声明输出个数不符 | `output_count_mismatch` | 契约 |
| 声明 name 重复/非法 | `duplicate_output_name` / `output_name_invalid` | 契约 |
| 声明 role 非契约角色/非字符串 | `output_role_mismatch` / `output_role_not_registered` | 契约 |
| 声明 dtype/shape 与规则不符 | `output_dtype_mismatch` / `output_shape_mismatch` | 契约 |
| 声明 kind 非 tensor | `multi_output_scalar_result_not_supported` | 契约 |
| 空 outputs / 未知字段 | `output_list_empty` / `output_declaration_unknown_field` | resolver |
| 单输出契约收到 outputs | `multi_output_not_declared` | resolver |
| 首个输出投影矛盾 | `shape_rule_violation` / `dtype_mismatch` | 契约/resolver |
| 输出缓冲别名输入/彼此 | `output_input_alias` / `output_output_alias` | 执行 |
| 输出缓冲数量/形状/dtype 不符 | `plan_request_mismatch` / `unsupported_output_handle` | 执行 |
| provider 结果不匹配 | `execution_result_does_not_match_plan` | 执行 |
| report outputs 缺块/字段/互证不符 | `report_field_missing` / `report_field_wrong_type` / `report_field_inconsistent` | report |

## 5. report 层与版本决定

- `REPORT_SCHEMA_VERSION` **保持 4**。理由（给证据）：
  1. 新块是可选顶层键 `outputs`，只由多输出 plan 产生；单输出报告的字节内容与 v4 完全一致
     （测试 `test_single_output_report_stays_without_the_outputs_block` + 零漂移记录）；
  2. 多输出报告缺块时**不放行**：`validate_report_schema` 在 plan payload 声明有序结果而报告无 `outputs`
     时抛 `report_field_missing`；块存在时逐项严格类型/唯一名/sha256 digest，并三方互证
     （报告块 ↔ 内嵌 `plan_payload.document.selected.outputs` ↔ `request.outputs` echo），再核对
     `outputs[0]` 与 `request.output_shape`/`request.output_dtype`；
  3. 没有旧版本需要承担不兼容解码：v4 单输出文档语义未变，新增块是严格的附加解码路径。
- 逐项 `digest = sha256(canonical_json({name, role, dtype, shape}))`；rule digest 不进报告项，而由内嵌
  plan payload + `resolution.plan_digest` 锚定；报告项 digest 可由报告自身重算（报告内自证）。
- `report.request.outputs` echo 与内嵌 plan `request` 的逐字节一致由既有 `binding_verify` 保证；
  本切片再叠加一组专门的 `outputs` 互证检查。

## 6. 主验收指标（真实三图，实测）

运行环境：local 锁内（`OMP_NUM_THREADS=6`），只读规划，不执行 provider；三图 = attention / GDR / 整 decoder。
机器可读：`_meta/pypto-x/u-multi-output-abi/raw/qwen_region_missing_contracts_split.json`。

| 指标 | batch4 基线 | 本切片实测 |
|---|---|---|
| `missing_contracts` 类/次 | 1 类 / 24 次（split） | **0 类 / 0 次** ✅ |
| `unplannable` 类/次 | 0 | **1 类 / 24 次**（`split`，`multi_output_not_supported_by_execution_contract`）⚠️ |
| `registered_planned_total` | 4669 | **4669** |
| `abi_level_split_planned` | n/a | **24 / 24** ✅ |
| attention `region_digest` | `92e21b47…` | `92e21b47b42aae98d592ed74f43a40a619abe763ffa202396900d0ffdbfee2d9`（不变） |
| gdr `region_digest` | `aa413baa…` | `aa413baab388f682bd085f41e43e91941597cc32809188ef5b6dddc80c1ffef0`（不变） |
| decoder 状态 | rejected（missing 1 类/24） | rejected（missing 0 / unplannable 1 类/24） |

**诚实结论**：任务书写的“`missing_contracts` 1→0 且 `unplannable` 保持 0”两个条件在本切片的
“`region.py` 不动”约束下无法同时成立。`region._plan_region_operation` 第 1688–1701 行对
`len(operation.outputs) != 1` 的 Core IR 操作在**进入契约之前**直接返回
`multi_output_not_supported_by_execution_contract`；这是 0024 明确列为 step 3 的 region 多输出边界。
本切片达成的是：**缺失契约类清零**（硬指标）且 24 个真实 `split` 在单算子 ABI 上 24/24 成功；
整图 `plan_region` 的 `unplannable` 要清零必须在 step 3 打开这一前置检查，不能靠注册表绕过。
统计口径：多输出算子按 **1 次引用**计（与 region 现口径一致）；若 step 3 改成 N 个输出边界，
`missing_contracts`/`unplannable` 的 occurrence 口径需在 0024 step 3 文档中重写。

## 7. 验证与证据

### 7.1 新增测试（`python/tests/ut/pypto_x/test_execution_multi_output_abi.py`，75 项）

- 契约/注册表：`split` 摘要钉住、28 个既有契约摘要逐字节钉住、scoped capability 视图累计性（batch4 不吸收 split）、
  单输出契约无多输出表面；
- plan：有序 request/selected、规则 digest、copy 字节（含不等分总和）、默认名、`num_splits` 归一化、
  digest 随顺序/名字变化、JSON 重放、6 组真重签篡改分层；
- 差分：8 组（rank1–4、负轴、等分/不等分、bool/int64/uint16/fp16/bf16/fp32）对
  `_split_value` 参考 + 独立嵌套切片 oracle；特殊值位型；0 元素非轴维；
- 缓冲：自动分配、用户缓冲、单张/数量/形状/dtype/输入别名/输出间别名拒绝；
- 拒绝矩阵：20 组 attributes 参数化用例 + 空轴/arity/rank/dtype/声明输出矛盾等独立用例；
- report：有序块 + digest、7 组篡改、单输出无块；
- C 项：`request.outputs[i].shape`/`request.inputs[i].shape` 的缺失/类型错误 field 完整路径。
- 非空转：新文件 import `BATCH5_CANONICAL_OPCODES`，在父提交 `ec7a9edc2` collect 失败
  （`ImportError: cannot import name 'BATCH5_CANONICAL_OPCODES'`，rc=2，见日志 `parent_collect_new_tests.log`）。

### 7.2 focused（local 锁内，`OMP_NUM_THREADS=6`）

- `pytest -q python/tests/ut/pypto_x -k execution`：**1006 passed / 7 skipped / 0 failed**（47–51 s，两次锁内运行一致）；
  其中本切片文件 75 passed。
- 默认环境同一 focused 子集：1013 passed / 0 failed（46–49 s；默认环境无 AOCL 线程 env_default skip）。

### 7.3 规则 8 全量（local 锁内，单次运行）

- collect **2747**（基线 tip 2672 + 本切片新文件 75 = 2747；rc 0）；
- 运行：**2733 passed / 14 skipped / 0 failed / rc 0**，耗时 1253.6 s（20:53）；
  14 个 skip 与基线一致（7 个 CUDA driver 不在本机 + 若干 host-guarded 冻结摘要钉，含锁环境 OMP=6 导致的
  capability/plan 摘要 pin 跳过）。见 `logs/rule8_summary.txt`、`logs/rule8_rc.txt`、`logs/rule8_collect.txt`。

### 7.4 零默认变化（两次环境各一次父子对照）

- 28 个契约 digest 与父树逐字节一致（新测试内 28 项 pin + `raw/frozen_*` 记录）；
- `record_frozen_record.py`（沿用 batch4 记录的 24 契约 + 17 runs；`--mode=batch4` 再加 4 契约 + 5 runs）：
  - 默认环境：父 vs 子 **ZERO-DRIFT**（frozen/batch4 两份 JSON 相等）；
  - local 锁环境（`OMP_NUM_THREADS=6`，capability digest `70d8cf05…`）：父 vs 子 **ZERO-DRIFT**；
- `record_report_declarations.py`：对同样 22 个 run 记录「去掉 timing 块后的完整 v4 报告声明文档」及其
  canonical digest；默认与 OMP=6 两个环境下父 vs 子 **逐字节相等**（`raw/report_declarations_*.json`
  与 `raw/zero_drift_comparison_*.json`）。这是“单输出 report 逐字节不变”的机器可读证据
  （唯一被排除的是 wall-clock timing 字段）。
- capability 视图：batch1–4 的累计 manifest 摘要不变（batch4 仍 28 契约）；`split` 用新的 29 契约视图。

### 7.5 C 项修复（drive-by）

`plan._require_shape` 之前以 `path="shape"` 调用 `_walk`，缺失/类型错误时 field 报 `plan.shape`。
现在 `_require_shape` 接受完整 field 前缀，两个调用点分别传
`plan.request.inputs.{i}.shape` 与 `plan.request.outputs.{i}.shape`；缺失 → `plan_field_missing`，
类型/维度错误 → `plan_shape_field_invalid`，field 均为完整路径（测试三例钉住）。

## 8. 已知边界与未覆盖（诚实登记）

- **region 多输出（0024 step 3，父方另派）**：`region.py` 未改；`exit_outputs` 仍是算子级，
  `(op_index, output_name)` 对、逐输出精度合成、`REGION_SCHEMA_VERSION=3` 均未做。
  因此整 decoder 的 `plan_region` 仍以 `unplannable` 拒绝 24 个 `split`（§6）。
- `view`/`contiguous`/`quantize_per_token_s8`/`dequantize_*` primitive 未登记（本切片明确不注册）。
- 多输出与 scalar-kind 结果互斥（`multi_output_scalar_result_not_supported`）；运行时可变输出个数、
  动态 shape、跨 provider 多输出语义差异均结构化拒绝。
- 结果 name 的图级锚点、多输出算子在 region 内的 def-use/边界 digest 属 step 3。
- 未做任何性能声明；本机数字 UNGATED。

## 9. 下一步建议

1. **0024 step 3（region 多输出）**：最小改动是 `region.py` 的
   `_plan_region_operation` 结果前置检查按 verifier 的静态结果个数（`split` 的 sizes/num_splits）展开，
   `exit_outputs` 改为
   `(op_index, output_name)`，`_request_from_operation` 已天然产生有序 `outputs`；
   region 校验器的 `expected_output_count = len(definition.outputs)` 与
   “outputs shape == output_shape” 两处需改为逐输出口径，并升 `REGION_SCHEMA_VERSION=3`。
2. Tier-3 余项：`view`/`contiguous`（有 0022 视图语义可复用）与量化 primitive（vendor 路径已冻结在 0002/0010）。
3. 若未来出现固定多项输出（非 repeat 模板）契约，`OutputSpec` 列表已支持；只需在 verifier 里按 index 分配。
4. 统计口径：step 3 落地时应同时给出“多输出算子 1 次引用 vs N 个输出边界”的对照，避免 ERR-0015 类口径混乱。

## 10. 复现入口

- 契约/plan/report/执行：`python/tests/ut/pypto_x/test_execution_multi_output_abi.py`
- 零漂移记录：`_meta .../scripts/record_frozen_record.py`（`PYPTO_BASELINE_TREE=<tree>`）
- 报告声明逐字节：`_meta .../scripts/record_report_declarations.py` + `compare_zero_drift.py --env=<default|lockenv>`
- 主验收：`_meta .../scripts/qwen_region_missing_contracts_split.py`（只读规划 + ABI 级 split 24/24）
- 锁内 bundle：`_meta .../scripts/run_locked_bundle.sh` + `queue_locked_bundle.sh`
