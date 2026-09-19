# 多输出算子 ABI 提案（父方拟，2026-09-19）

状态：**PROPOSAL**（可逆；实现前只需改本文件与 lock）
动机：`split` 是真实 Qwen 图里最后一个"高频但被前置挡住"的算子——当前执行契约、plan、report 与 U5 region schema 都假设**单输出**，
验收已独立复现：未登记时 `region_contracts_missing`，已登记的多输出构造则 `multi_output_not_supported_by_execution_contract`。
本提案给出最小、向后兼容的多输出 ABI，供 Tier 3 使用。

## 1. 契约层（OpContract）

- 单输出算子**不变**（向后兼容）：现有 `output` 语义等价于 `outputs=[{name:"output", ...}]`（arity 1）。
- 新增可选 `outputs`：有序 `OutputSpec` 列表，每项含 `name`、`dtype` 规则、`shape` 规则（可依赖 attributes 与输入 shape）、`role`（如 `values`/`indices`/`remainder`）。
- 校验规则：
  1. `outputs` 的项数必须是**静态可判定**的（由 opcode+attributes+rank 决定；不允许"运行时才知道有几个输出"）；
  2. 每个输出的 dtype/shape 规则必须可复算并进入 contract digest；
  3. 未知 `role`、空列表、重复 name → 结构化拒绝；
  4. 既有单输出算子的 contract digest **逐字节不变**（新 schema 只对声明 `outputs` 的算子生效）。

## 2. 计划层（plan）

- `plan.selected.output` 对单输出保持现状；多输出算子在 plan 里新增有序 `outputs` 列表（每项 `{name, role, dtype, shape, dtype_rule_digest, shape_rule_digest}`）。
- `plan_digest` 覆盖 `outputs` 列表的全部字段（含顺序）——改 name/role/顺序/形状即拒绝。
- 执行绑定：多输出算子的 `execute_operation` 接收所有输入，返回**有序 HostTensor 元组**；单输出仍返回单元素元组或单张量（保持现有调用方兼容，具体由实现定，但必须有明确文档与测试）。

## 3. 报告层（ExecutionReport）

- v4 报告新增可选块 `outputs: [{name, role, dtype, shape, digest}]`（有序）；单输出报告的既有 `output` 字段保留（与 `outputs[0]` 一致，校验器要求二者互证）。
- `binding_verify` 对多输出做逐项校验（数量/顺序/dtype/shape/digest）。
- 版本：若新增块为**可选**且单输出报告逐字节不变 → 可不升 `REPORT_SCHEMA_VERSION`；否则升 v5 并在 0024 记录理由（由实现方判定并给证据）。

## 4. Region 层（U5）

- region 边界 `exit_outputs` 从"算子"扩展为 **(op_index, output_name)** 对；`region_digest` 覆盖 output 名字与顺序。
- region 精度声明**逐输出**合成（各输出独立走 0020 §2 的保守规则）；`is_upper_bound` 逐输出判定。
- schema：若边界表达改变 → `REGION_SCHEMA_VERSION` 升 **3**，v2 文档以命名错误拒绝（沿用 round-3 的版本纪律）。
- `unplannable`/`missing_contracts` 的统计口径要写明"多输出算子计 1 次引用还是 N 个输出"（避免再出现 ERR-0015 那类口径混乱）。

## 5. 落地顺序（建议）

1. **ABI 层**（契约 + plan + report，仅单输出算子走查，保证零默认变化）；
2. **登记 `split`**（含 dtype/轴/等分与不等分语义、空段语义、结构化拒绝矩阵）；
3. **region 层多输出**（schema v3 + 逐输出精度 + 往返）；
4. **Tier 3 余项**：`div/gather/identity/embedding`（单输出，可并行）；
5. 验收指标：真实 Qwen 图缺失契约 **5 → 0 类**（届时整图**可全图规划**，只剩执行/v5 未做）。

## 6. 明确不做

- 运行时可变输出个数；动态 shape；多输出算子与多输出的 region 之间做融合/代数化简；
- 跨 provider 的多输出语义差异（若 provider 不支持元组返回 → 结构化拒绝，不静默降级）。

## 7. 验收要求（实现时）

1. 单输出路径**零默认变化**（matmul/qmatmul/16 个已登记契约的 digest 逐字节不变）；
2. 多输出计划的 digest 覆盖顺序/name/role/形状，改一位即拒；
3. 多输出报告自证（`outputs` 与 `output` 互证、篡改拒绝）；
4. `split` 的差分与拒绝矩阵（含不等分/轴越界/空段）；
5. region 多输出边界往返（plan→validate→load→explain）+ 逐输出精度合成对抗用例；
6. 主验收指标从 5 类降到 0 类（真实三图），并把"每输出"统计口径写进文档。
