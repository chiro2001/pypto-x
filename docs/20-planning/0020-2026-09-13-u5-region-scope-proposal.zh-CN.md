# U5 region 作用域提案（父方拟，2026-09-13）

状态：**PROPOSAL**（父方按用户"继续自主执行"授权起草；实现前如需改动，只需改本文件与 lock，不影响任何已冻结结论）
关联：`docs/20-planning/0008`（U 线阶梯）、`0013`（U3 发现命令）、`0044`/`0046`（report v4）、`0018`（composite 组合化）、`0019`（opcode 契约第一批）

## 0. 为什么需要 U5

现状（父方核查）：

- U 线的 policy/plan/report 只覆盖**单算子**（`matmul`、`qmatmul_s8s8_s32`，0048 起扩展到 8 个高频 opcode）；
- attention / GDR / conv-state / 整个 decoder 是**图**，能构图、能编译、能执行（SVE256/A3、AVX2/AVX-512 有真实证据），但**没有**：
  1. 图级冻结计划（每个算子选了什么 provider/精度类别、它们如何合成）；
  2. 图级自包含报告（用户拿不到"这一整段计算"的可审计声明）；
  3. 图级 fail-closed 语义（`require_proven_deviation_bound` 今天只能对单算子生效）。
- 用户目标（原话）："用户应该能用我们的 pypto_x 写出适配不同模型的算子，并且甚至可以只用我们的 pypto_x 就搭建出整个模型所需要的算子"。

**U5 就是把"图"提升为一等公民**：region 级 plan / report / 精度合成 / fail-closed。

## 1. region 的定义（推荐口径：静态连通子图）

一个 **region** = 一个 `CoreProgram` 内**由 def-use 连通的算子集合**，带：

- `region_id`：稳定标识（由 region 内 op 的 canonical 顺序 + 输入/输出签名导出）；
- `entry_inputs` / `exit_outputs`：region 的边界 tensor（显式声明，不允许隐式捕获 region 外的中间值）；
- `operations`：region 内算子的**拓扑序**列表，每个元素是已冻结的单算子 plan（U 线既有对象）；
- `region_digest`：Merkle 摘要 = `sha256(canonical({op_digests...}) + boundary_signature + region_schema_version)`。

不选另外两种粒度的理由：

- **SSA 活性区间**：粒度太细、与"模型/子图"语义无关，用户无法表达"这一整段 attention"；
- **整个 CoreProgram**：粒度太粗，Qwen 整图 4500+ op 报错时无法定位，也无法与 vLLM/框架的分段边界对齐。
- 连通子图可以与模型的**结构边界**一一对应（attention 块、GDR 块、一层 decoder），且可**递归**（region of regions）。

## 2. 精度合成规则（推荐口径）

对 region 内每个算子取其实例化后的 `{numeric_class, exactness_envelope}`，按以下**保守可复算**规则合成：

| 情形 | region 级声明 |
|---|---|
| 全部算子 `exact` | `exact`；`deviation_bound_by_k = {all_k_le_limit: 0}` |
| 全部 `portable_bitwise`（或 exact 与它混合） | `portable_bitwise`（与 portable trunk 逐位） |
| 存在 `deterministic_bounded` 且**每个**都有 proven 包络 | `deterministic_bounded`；**包络 = 各算子界之和**（逐项写明，且必须声明该求和是否为上界：只有每个界都是上界时，和才是上界） |
| 存在无 proven 包络的算子 | `deterministic_bounded_unproven`；`require_proven_deviation_bound=true` 在 **region 级 fail-closed** |
| 存在 `best_effort` | region 级 `best_effort`（且 region 不可用于 `require` 类策略） |
| 混用不同 provider | region 声明必须列出 **provider 集合**；不做跨 provider 抵消/补偿，各算子界独立相加 |

其他硬规则：

1. **误差不可抵消假设**：禁止"两个算子的误差互相抵消"这类声明；上界一律相加。
2. **顺序敏感**：`deterministic_bounded` 的界必须随 op 顺序写明（例如 reduction 顺序、bf16 舍入顺序），region 的界绑定到冻结的拓扑序。
3. **不静默降级**：任何算子被 host/scalar 回退都要在 region plan 里如实标注（`mode`），并把该回退的影响写进 region 声明。

## 3. 计划与报告的锚定（推荐口径）

- **region plan**：一个冻结文档，包含 `region_schema_version`、`region_digest`、边界签名、逐 op 的单算子 plan digest 列表、逐 op 的 provider/精度摘要、以及 region 级精度声明；
- **region report**：在 v4 报告的基础上**升 v5**，新增两个必需块：
  - `region_plan`：上面的冻结文档（preimage 进入 digest）；
  - `region_guarantee`：region 级结论（class、界、provider 集合、`fail_closed` 原因若有）；
  校验仍不 probe、不重跑 resolver：`region_digest` 必须能从 `region_plan` 自身复算，逐 op 的 digest 必须与各单算子 plan 的 digest 一致（锚定关系双向可检）。
- **兼容**：单算子路径不变（`plan_matmul`/`plan_qmatmul` 与 v4 报告逐字节不变）；region 是**新增**入口。

## 4. 建议的用户入口（分三步落地）

1. **只读**（第一步，风险最低）：`plan_region(program, region_spec, policy)` → 冻结 region plan；`explain_region(...)` 复用 U3 的 explain 输出；CLI 加 `plan --region`/`explain --region`。
2. **执行**：`execute_region_plan(region_plan, bindings)` → 按冻结拓扑序用**既有单算子执行器**逐 op 执行 + 组合 region report（不引入新的调度器、不做跨算子融合）。
3. **优化**（后续，另立切片）：region 级 workspace/内存规划、缓存/记忆化、并行调度；任何优化都必须保持 region plan/report 的声明不变。

## 5. 明确不在本提案范围

- 动态 shape / 符号维度；
- 跨 provider 融合算子、跨算子代数化简；
- 性能声明（region 执行性能一律 UNGATED，直到独立验收）；
- 分布式/多设备 region（Ascend 放行后再议）；
- region 内的控制流（Core IR 目前无 control flow）。

## 6. 依赖与顺序

- **依赖 0019（opcode 契约第一批）**：region 内每个 op 必须有单算子契约才能形成单算子 plan；未登记 opcode 的 region 必须**结构化拒绝**（列出缺哪些 contract），不允许"部分声明"。
- **依赖 0018（composite 组合化）**：region 的自然来源是 composite 边界；组合化之后 region 边界才稳定可复算。
- 建议实现顺序：0019 → 0018 → **0020（本提案）第一步只读** → 第二步执行 → 第三步优化。

## 7. 验收要求（实现时）

1. region digest / region plan / region guarantee 可自证（改一位即失败）；
2. 逐 op plan digest 与 region plan 的锚定双向可检；
3. 精度合成规则用**对抗用例**验证（含"无 proven 包络"→ region 级 fail-closed、混用 provider、全 exact、全 portable_bitwise）；
4. 与单算子路径的**零回归**：既有 matmul/qmatmul 的 plan/report 逐字节不变；
5. region 执行输出与"逐 op 手工执行"的输出逐位一致（同一拓扑序）；
6. 未登记 opcode 的 region 结构化拒绝并列出缺失 contract。
