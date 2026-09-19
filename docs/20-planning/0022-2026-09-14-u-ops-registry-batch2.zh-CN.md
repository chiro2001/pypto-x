# 0022 — U 线算子注册第二批（batch2）：6 个 opcode 契约 + 视图语义冻结 + 两个 batch-1 补丁

- 任务：`u-ops-registry-batch2`（batch 0050 切片 A）
- 实现基线：integration tip `b643229bd`（U5 round-2 修复；本切片最初基于 `41ae82a01`，按父方要求 rebase 到 `b643229bd` 后复跑全部数字）
- 分支：`work/u-ops-registry-batch2`（未 push）
- 依赖：`0019`（batch1 的 schema v3 契约与“按 opcode 收窄的 capability 视图”）、`0021`（U5 只读 region 的真实图缺失契约频次表）
- 结论等级：**实现完成（静态 shape；portable provider 路径）**。未做性能声明；本机数字一律 **UNGATED**
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u-ops-registry-batch2/`
- 实现 commit：`79e528560`（rebase 后；分支 `work/u-ops-registry-batch2`，未 push；控制仓父方提交 0012→0022）

## 0. 摘要

本切片把执行层 opcode 契约从 10 个扩到 16 个，新增：

`mul`、`add`（逐元素二元，完整尾轴广播 + 标量操作数）、`rsqrt`（一元浮点）、
`transpose`、`slice`、`broadcast`（视图/数据搬运族，**先冻结视图语义再实现**）。

同时落地两个 batch-1 契约补丁：

- **补丁 A**：`where` 接受 **scalar fill operand**（真实图 7 处，attention ×1 + decoder ×6）；
- **补丁 B**：`compare` 属性白名单接受 **`broadcast`**（真实图 1 处，decoder `broadcast='trailing_singleton'`）。

两个补丁都是“只放宽准入”：batch1 的 `where`/`compare` **契约文档逐字节不变**（contract digest 冻结），
既有合法请求的 normalized request / plan / output 不变；新能力以 **verifier 级 audited admission extension**
实现（见 §4）。

主验收指标（§8）：在 `b643229bd` 上用 U5 `plan_region` 对三个真实 Qwen3.5 图各跑一次，
缺失契约从 **19 类降到 13 类**（聚合口径与 U5 round-2 一致，错误码/结构未变），
`mul/transpose/broadcast/slice/add/rsqrt` 全部从缺失契约表中消失（机器可读 JSON：
`raw/qwen_region_missing_contracts.json`）。

### 0.1 冻结兼容策略（关键设计决定）

1. **6 个新契约**沿用 registry 级 schema **v3**（`0019 §0.1`），`contract_version = "1.0"`，
   `allowed_providers = ("portable",)`、`effects = pure`、`verifier = generic`。
2. **冻结 10 个契约的 document 与 digest 一字节不改**。补丁 A/B 因此不写进被哈希的文档，而是写成
   `operation_contracts.py` 中显式的 admission extension 表（`WHERE_SCALAR_VALUE_OPERANDS` /
   `COMPARE_EXTRA_ATTRIBUTES`），并在 §4 写明其审计口径。
3. **操作域 capability manifest 按批次冻结累积集合**：某请求一旦在 batch N 解析，其 capability digest
   不能因为 batch N+1 注册新契约而漂移。因此 `operation_provider_capability(opcode)` 现在按 opcode 返回
   冻结集合：batch1 请求 → 10 个 contract（与 0048-B 完全一致）；batch2 请求 → 16 个 contract。
   默认探测（`default_providers`）仍只声明 matmul/qmatmul 两个契约，`matmul`/`qmatmul` 路径不受影响。
4. **视图族 lowering 只在 execution 层完成**：`add`/`mul` 的 tensor×tensor 广播由
   `build_binary_elementwise_program` 显式降为 `broadcast + elementwise` 两/三步程序（plan 仍只声明
   单个 contract opcode）；不修改 `pypto/lowering/*`、`pypto/backends/*`、compiler targets。
5. 不修改 `pypto/__init__.py`、`python/pypto/execution/region.py`、`python/pypto/portable/qwen35.py`。

## 1. 六个新契约的逐算子摘要

所有 6 个契约：rank 域（张量操作数）1..4；标量操作数只在契约显式允许处出现（见 §3.2）；
`attributes` 白名单之外一律 `unknown_attribute`；序列型属性只接受有序 list/tuple
（set/generator/mapping → `unordered_attribute_container`）。

| opcode | 元数 | shape 规则 | dtype 域 | 结果 dtype | 属性白名单 | portable executor 符号 |
|---|---|---|---|---|---|---|
| `add` | 2（left, right） | 尾轴对齐广播；scalar 为 rank-0 恒等；两个 scalar 拒绝（结果为标量，不在执行契约内） | 全部 portable dtype 去 bool | = 操作数共同 dtype | 无 | `pypto.backends.cpu.runtime._binary_value` |
| `mul` | 2（同 add） | 同 add | 同 add | 同 add | 无 | `pypto.backends.cpu.runtime._binary_value` |
| `rsqrt` | 1 | 逐元素，`shape(out)=shape(in)`，rank 1..4 | `float16/bf16/float32/float64` | = input dtype | 无 | `pypto.backends.cpu.runtime._math_value` |
| `transpose` | 1 | `shape(out)[i] = shape(in)[permutation[i]]`；rank 1..4 | 全部 portable dtype（含 bool） | = input dtype | `permutation/perm/dims/axes` 或 `dim0/axis0`+`dim1/axis1`；`inplace` 只 false；`valid_shape/valid_shapes` 必须等于输出 shape | `pypto.backends.cpu.runtime._transpose_value` |
| `slice` | 1 | 每个被切轴 `max(0, ceil((stop-start)/step))`；未列轴恒等；rank 1..4 | 全部 portable dtype（含 bool） | = input dtype | `starts/start`、`stops/stop`（必填，同现须一致）；`axes/axis`（可选，唯一）；`steps/step`（可选，**必须为正**）；`inplace`/`valid_shape/valid_shapes` 同上 | `pypto.backends.cpu.runtime._slice_value` |
| `broadcast` | 1 | 尾轴规则把 input 扩展到声明的 target shape；输出 rank 1..4；scalar input 可扩展到任意 target | 全部 portable dtype（含 bool） | = input dtype | `shape/target_shape/out_shape`（一致，且等于声明输出）；`inplace`/`valid_shape/valid_shapes` 同上 | `pypto.backends.cpu.runtime._broadcast_value` |

广播的精确规则与 batch1 完全同源：执行层直接调用 `pypto.lowering.cpu._broadcast_shape` 后复核
（尾轴对齐；每维相等或其一为 1；`0` 只与 `0`/`1` 广播，结果为 `0`）。

`contract_digest`（实现树，`raw/batch2_record.json`）：

```text
add       sha256:f7bc4d7b67339414e7c423fc860d0c4ca660390618c2524bac370e378c6e8c35
broadcast sha256:e7e0b09638ba61d6bf7142ba513857c2a544cbbcf6e3d673ad074bc70d7a6a4c
mul       sha256:dbf33bf2d369352749be3b0663d842deb2a155c791499280c85c4d43edef3281
rsqrt     sha256:c9ac801ee7d57c79737dc2318d1b4fbe9b1ee2a8f5a0d615322f42e2c13692bb
slice     sha256:a74ca05a5d884d29193036635a1e8892f0b5c58c7ea0c363d93c9184f8cb9ae7
transpose sha256:548aa6d25207438bc24419e91a104e75ea489c561f0fc015e0a701169b608bba
```

## 2. 视图语义冻结（transpose / slice / broadcast）

### 2.1 义务与 alias 口径

三个 opcode 在契约里声明 `view_obligation = may_alias`（`numeric.aliasing = "may_alias"`）：

- **逻辑上**它们是视图族：`transpose` 是轴置换、`slice` 是 offset/step 子视图、`broadcast` 是 stride-0 展开；
  provider 若具备 stride descriptor + alias proof，允许输出与输入共享存储；
- **portable trunk 没有 strided-view ABI**（`HostTensor` + 扁平 row-major buffer），因此 portable provider
  **始终物化一个独立的 row-major 输出缓冲**；`may_alias` 是“允许”而不是“必须”。

### 2.2 `view_mode` 三态（唯一冻结口径）

| `layout.view_mode` | 对三个视图 opcode 的含义 | portable 解析结果 |
|---|---|---|
| `forbid`（默认） | 禁止零拷贝视图，必须显式独立目标 | `resolved_view_mode = "materialized"`，`layout_copy_bytes = element_count(output) × dtype_size(output_dtype)`，`materialize_allowed = true` |
| `prefer` | 有可证明的零拷贝视图则用视图，否则用**显式、受预算约束**的物化拷贝 | portable 无法给出证明（`alias_proof = unsupported`），走显式物化；`materialize_allowed = true` |
| `require` | 必须给出零拷贝 alias 证明 | **结构化拒绝** `ALIAS_PROOF_UNAVAILABLE / portable_trunk_has_no_strided_view_abi`（field `layout.view_mode`）；**绝不静默降级为拷贝** |

- 对 `must_not_alias` 契约（matmul/qmatmul/batch1 8 个 + add/mul/rsqrt），`forbid`/`prefer` 仍解析为
  `resolved_view_mode = "not_applicable"`、`layout_copy_bytes = 0`（与 batch1 逐字节一致）；
  `require` 对它们仍是 `contract_requires_distinct_storage` 拒绝。
- `materialize` 预算：`layout.max_copy_bytes` 小于真实 copy 字节数时，`forbid`/`prefer` 也会
  `ALIAS_PROOF_UNAVAILABLE` 拒绝（见 `test_view_family_forbid_materialize_copy_respects_policy_budget`）。
- `copy_bytes` 由冻结 request 的 `output_shape`/`output_dtype` 复算；plan/report 的重新校验
  （`entry._verify_plan_layout_decision`、`report.validate_report_schema`）都传入同一 request 块，
  因此“物化字节数”是文档自证的一部分。

### 2.3 `inplace` / `valid_shape` / stride 规则

- `inplace=true` 对三个 opcode（以及 batch1 的 `reshape`）一律结构化拒绝
  （`transpose_inplace_not_portable` / `slice_inplace_not_portable` / `broadcast_inplace_not_portable`）：
  portable 输出必须是独立 buffer，执行期另有 `_check_no_alias_any` 兜底。
- `valid_shape/valid_shapes` 若非空必须逐维等于解析出的输出 shape（`*_valid_shape_not_representable`）；
  空/非法序列 → `*_valid_shape_invalid`。
- **stride 合法性**：不暴露 stride 属性；`transpose` 的合法输入是包含每个轴恰好一次的完整 permutation
  （负轴归一化；重复/缺失/越界 → `transpose_permutation_invalid`）；`slice` 的 step 必须为正
  （`slice_negative_step_not_supported`），start/stop 必须在 `[-dim, dim]` 内
  （`slice_spec_invalid`）；`broadcast` 的 target 必须尾轴兼容（`broadcast_shape_mismatch`）。
  `view`/`contiguous` 不在本批注册，stride 属性不在契约表面。

### 2.4 与既有 alias 校验（`layout_verify` / `binding_verify`）的关系

- 解析期：`layout_verify.canonical_layout_decision` 现在按 `definition.semantics["view"]` 构造
  `ViewContract`（非视图契约走冻结的 `matmul_view_contract`），capability 对视图族声明
  `alias_proof=unsupported`；决策（含 `alias_proof.digest`、`layout_copy_bytes`）进入 plan payload 与 digest。
- 校验期：`ExecutionPlan.validate` 做内容自证；`entry._verify_plan_layout_decision` 与
  `report.validate_report_schema` 从冻结 policy + contract + request 重新推导并逐字段比对；
  report 侧的 `alias_check` 由执行期的 `_check_no_alias_any` 升级为 `verified_distinct`。
- 执行期：portable provider 只允许 `materialized` 语义（输出独立 storage），任何输入别名输出 →
  `output_input_alias`。

## 3. add / mul：广播、标量与执行子集

### 3.1 完整尾轴广播

- 契约层实现完整尾轴规则（与 `where`/`compare` 同源）；两个 tensor 形状不同时，`add`/`mul`
  在 provider lowering 中被显式降为 `broadcast(operand) → elementwise` 的 Core IR 程序，
  **plan 仍只声明一个 contract opcode**，内部程序从冻结 request 重新推导，不引入用户可见属性。
  这样“合法广播”既不会静默改写结合律，也不会被无谓拒绝。
- 两个 scalar → `scalar_result_not_supported`（执行契约只产出 tensor）。
- 广播不合法（如 `(2,3)+(4,)`）→ `broadcast_shape_mismatch`；
  `0` 维与 `1` 维广播 → `0` 维输出，空张量语义见 §6。

### 3.2 标量操作数与 dtype 提升规则

- scalar 操作数可出现在任一侧；语义是 **rank-0 恒等广播**（不增加输出维）。
- **无隐式 dtype 提升**：scalar 必须已经与另一操作数同 dtype；混合 dtype → `dtype_mismatch`。
  前端若要 `int scalar + f32 tensor`，必须显式插入 `cast`。这是与 lowering 一致的最保守口径。
- `where` 的 scalar fill（补丁 A）用同一规则；`condition` 仍必须是 tensor-bool（rank 1..4）。
- 执行 ABI：plan 的 `request.inputs[i]` 只在 scalar 操作数上携带 `"kind": "scalar"`（既有请求不带该键，
  因此 batch1 digest 不变）；`execute_operation` 接受 `int`/`float` 句柄，provider 以
  `LaunchRequest.scalars` 传入标量并按冻结 dtype 做一次 range-checked cast。

## 4. 两个 batch-1 契约补丁（严格加法）

### 4.1 补丁 A：`where` 的 scalar fill operand

- **冻结文档不变**：`where` 的 schema-v3 document、`attribute_names`、`operand_dtypes`、contract digest
  （`sha256:497383c0c903fdd2c5a0e245653d58b9ce3cb9a3e6f09b1ea146fd6eb4681e64`）逐字节与 0048-B 相同。
- verifier 对 `input`/`other`（operand 1/2）允许 rank-0 scalar；标量必须与另一值操作数同 dtype，
  广播时只贡献“无维度”。
- 既有 tensor-only 请求：normalized request、plan digest、输出逐字节不变
  （`test_frozen_legal_plan_digests_are_unchanged` 的 `where_tensor_only` 点）。
- 真实图 7 处请求（attention ×1、decoder ×6）在 `plan_operation`/`execute_operation` 层全部可规划、可执行。
- **登记边界**：U5 `plan_region` 的 `region.py` 在任何 contract 解析之前就有“scalar operand →
  unplannable”预检（`scalar_operand_not_supported_by_execution_contract`），而本切片被明确要求不改
  `execution/region.py`（另一验收方在读）。因此真实图整图 region 仍把 `where`（以及 add/mul 的 scalar 请求）
  列进 `unplannable_operations`，但**不再进入 `missing_contracts`**。这是 U5 入口的边界，不是契约缺口。

### 4.2 补丁 B：`compare.broadcast` 属性

- `compare` 的 shape 规则本就是尾轴广播；真实 decoder 图写 `broadcast='trailing_singleton'`。
- **冻结文档不变**（attribute_names 仍只有 `predicate/comparison/relation/op`，contract digest
  `sha256:806ad35d4eff481a32e2fa3f4ea350d39827be1bb4c54167769893977fcb971d`）。
- verifier 额外接受 `broadcast`：`true` / `"true"` / `"trailing"` / `"trailing_singleton"` / `"auto"`
  → 规范化 `"trailing_singleton"`；**缺省 = 不写该属性**（行为与 batch1 完全一致）；
  `false`/`"false"`/`"none"`/`"off"` → `compare_broadcast_disabled_not_portable`
  （契约本来总是允许尾轴广播，声明“禁止”是无意义收窄）；其他值 → `compare_broadcast_invalid`。
- 既有不带该属性的请求：normalized attributes、plan digest、输出逐字节不变；
  真实图那 1 处请求现在可规划。

## 5. 精度声明

| opcode / 路径 | 类别 | bound kind | 理由 |
|---|---|---|---|
| `add`/`mul` 整数 | **exact** | `exact_contract_bound`（proven，bound=0） | 每元素 exact Python 整数运算 + 一次带范围检查的输出 cast；越界 fail-closed，无回绕/饱和 |
| `add`/`mul` 浮点 | `deterministic_bounded` | `deterministic_serial_order_no_registered_analytic_bound` | 操作数精确解码到 binary64，一次运算 + 一次 RNE cast；**无解析上界**，观测值绝不当界 |
| `rsqrt` 全部（float only） | `deterministic_bounded` | 同上 | `1.0/sqrt(x)` 由平台 binary64/libm 给出再一次 RNE cast；**无解析上界** |
| `transpose`/`slice`/`broadcast` 全部 | **exact** | `exact_contract_bound`（proven，bound=0） | 纯数据搬运/坐标重排，不做算术、不做转换；NaN/±0/±inf/次正规/bf16 位型原样保留 |
| 补丁 A `where` scalar fill | 沿用 batch-1 声明 | 整数/布尔 exact；浮点 bounded | 选择本身逐位精确，但执行层未注册浮点包络 |

`numeric.require_proven_deviation_bound = true` 时：

- 视图族（exact）与 add/mul 整数路径正常解析；
- add/mul 浮点与 rsqrt（`deviation_bound_available=false`）**fail-closed**
  （`NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable`，field
  `numeric.require_proven_deviation_bound`），portable provider 也生效。

## 6. 语义边界（契约文字 + 用例）

- **NaN**：`add/mul` IEEE-754 传播（NaN 参与 → NaN）；`rsqrt(NaN)=NaN`；视图族原样搬运 NaN 位型；
  slice/transpose/broadcast 不改变 NaN 的 payload（HostTensor 用 Python float 存储，payload 不承诺保留）。
- **±0**：`add` 按 binary64 IEEE 规则（`+0 + -0 = +0`，`-0 + -0 = -0`）；`mul` 符号按 IEEE；
  `rsqrt(+0)=rsqrt(-0)=+inf`（runtime 在 `value == 0.0` 处先判零）；视图族原样保留符号位。
- **±inf**：`rsqrt(+inf)=+0.0`；`rsqrt(-inf)=NaN`（负值路径）；`add/mul` 的 f32/bf16 溢出饱和为 signed inf，
  **float16 溢出 fail-closed**；`+inf + -inf = NaN`。
- **次正规**：add/mul 以 binary64 计算后按目标格式 RNE（可下溢为次正规或 0）；`rsqrt` 在 binary64 中计算
  （f16 最小次正规 `2**-24` → `4096.0`，无溢出）；视图族原样保留。
- **bf16 位型**：目标按 binary32 位型 RNE；`cast bf16→f32` 逐位可逆；视图族 bf16 值逐位保留。
- **整数回绕**：add/mul 整数路径不回绕、不饱和；结果超出输出 dtype → fail-closed
  （范围检查在 `_cast_value`，执行层包装为结构化错误）。
- **空张量 / 零长度**：`add/mul` 支持 `0` 维广播（`(0,) + (1,) → (0,)`，0 元素输出）；
  `rsqrt`/视图族 0 元素输入 → 0 元素输出；`slice` 的 `start >= stop` 产生 0 长度轴；
  `broadcast` 的 singleton→0 扩展产生 0 长度轴。
- **标量结果**：两个 scalar 的 add/mul、scalar 输出的请求全部结构化拒绝（执行 ABI 只返回 tensor）。

## 7. fail-closed 矩阵（新增触发器）

| 触发 | 错误（code / reason） | 发生层 |
|---|---|---|
| 未知 opcode（如 `sub`/`concat`） | `OP_CONTRACT_INVALID / opcode_not_registered` | registry/resolver |
| dtype 越域（bool 进 add/mul；非 float 进 rsqrt） | `dtype_outside_contract_domain` | verifier |
| 混合 dtype / scalar 与 tensor dtype 不一致（无隐式提升） | `dtype_mismatch` | verifier |
| rank 越界（>4 或 0-d tensor 放在不允许 scalar 的角色） | `rank_out_of_range` | verifier |
| 广播不合法（add/mul/broadcast） | `broadcast_shape_mismatch` | verifier（复用 lowering） |
| 两个 scalar 操作数 | `scalar_result_not_supported` | verifier |
| 未知属性 | `unknown_attribute` | verifier（白名单） |
| 无序容器属性（set/generator/映射） | `unordered_attribute_container` | verifier |
| `transpose` 轴重复/缺失/越界、permutation 与 dim0/dim1 冲突 | `transpose_permutation_invalid` | verifier |
| `slice` start/stop 越界、axes 重复、starts/stops 长度不一致 | `slice_spec_invalid` | verifier |
| `slice` step <= 0 | `slice_negative_step_not_supported` | verifier |
| `slice` 缺 starts/stops | `slice_spec_missing` | verifier |
| `broadcast` target 缺失 / 非法 / 与声明输出冲突 | `broadcast_target_shape_missing` / `broadcast_target_shape_invalid` / `shape_rule_violation` | verifier |
| `inplace=true`（视图族/reshape） | `*_inplace_not_portable` | verifier |
| `valid_shape` 与解析输出不一致 | `*_valid_shape_not_representable` | verifier |
| `compare.broadcast` 声明“关闭”/未知值 | `compare_broadcast_disabled_not_portable` / `compare_broadcast_invalid` | verifier（补丁 B） |
| `require_proven_deviation_bound=true` 遇浮点 bounded 路径 | `NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable` | resolver |
| `view_mode=require` 无法给出零拷贝证明 | `ALIAS_PROOF_UNAVAILABLE / portable_trunk_has_no_strided_view_abi` | layout_verify |
| `forbid/prefer` 物化超 `max_copy_bytes` | `ALIAS_PROOF_UNAVAILABLE` | layout_verify |
| 执行期标量越界/非法值 | `OP_CONTRACT_INVALID / operand_value_rejected` | portable provider |
| 输出别名输入 | `output_input_alias` | provider |
| plan 被篡改（含改写 `layout_copy_bytes`/`view_obligation`） | `ARTIFACT_MISMATCH / plan_digest_mismatch` 等 | plan/layout/report 校验器 |

## 8. 真实 Qwen3.5 图缺失契约频次表（batch1 → batch2）

证据：`raw/qwen_region_missing_contracts.json`（脚本 `scripts/qwen_region_missing_contracts_batch2.py`，
对三个 builder 的整 program 各跑一次 `plan_region`，read-only、不执行 provider）。
三个图仍结构化拒绝（预期，详见“unplannable”说明）。

| builder | ops | 缺失 contract（出现次数） | 已登记但 U5 预检不可规划 |
|---|---|---|---|
| `build_attention_kv_cache` | 20 | concat 2, gather 2, div 1, sub 1 | `mul`×1、`where`×1（scalar operand） |
| `build_qwen35_gdr_recurrent_state` | 123 | sub 4, identity 2, concat 1 | `mul`×4（scalar operand） |
| `build_qwen35_text_decoder_graph` | 4550 | reduce_mean 79, concat 66, constant 66, silu 60, neg 30, sigmoid 24, split 24, sub 24, softplus 18, div 6, embedding 1 | `add`×176、`mul`×24、`where`×6（scalar operand） |

合并后的**缺失契约频次（新表）**：

| 优先级 | opcode | 合并出现次数 | 覆盖图 |
|---|---|---|---|
| 1 | `reduce_mean` | 79 | decoder |
| 2 | `concat` | 69 | decoder + gdr + attention |
| 3 | `constant` | 66 | decoder |
| 4 | `silu` | 60 | decoder |
| 5 | `neg` | 30 | decoder |
| 6 | `sub` | 29 | decoder + gdr + attention |
| 7 | `sigmoid` | 24 | decoder |
| 8 | `split` | 24 | decoder |
| 9 | `softplus` | 18 | decoder |
| 10 | `div` | 7 | decoder + attention |
| 11 | `gather` | 2 | attention |
| 12 | `identity` | 2 | gdr |
| 13 | `embedding` | 1 | decoder |

**与 batch1 基准（0021 §5）对比**：

| opcode | batch1 缺失次数 | batch2 缺失次数 | 结论 |
|---|---|---|---|
| `mul` | 544 | 0 | 消失（新增契约） |
| `transpose` | 414 | 0 | 消失（新增契约） |
| `broadcast` | 360 | 0 | 消失（新增契约） |
| `slice` | 356 | 0 | 消失（新增契约） |
| `add` | 330 | 0 | 消失（新增契约） |
| `rsqrt` | 115 | 0 | 消失（新增契约） |
| 其他 13 类 | 411 | 411 | 不变（未注册） |

> **勘误（2026-09-19，由 `verify-0050-ops-batch2` 独立复算发现）**：上表"其他 13 类"原写 **254**，系笔误；
> 逐类相加应为 **411**（验收方独立复算与作者 JSON 一致）。类数 19→13、六算子归零的结论不受影响。

| **合计类数** | **19** | **13** | **-6 类** |

**逐 opcode 规划统计**（三图合并；region 入口对每个 op 都调用 `plan_operation`，不是首错即停）：

| opcode | 真实图出现 | `plan_operation` 成功 | U5 scalar 预检阻断 | 其他原因失败 |
|---|---|---|---|---|
| `mul` | 544 | 515 | 29（scalar operand） | 0 |
| `transpose` | 414 | 414 | 0 | 0 |
| `broadcast` | 360 | 360 | 0 | 0 |
| `slice` | 356 | 356 | 0 | 0 |
| `add` | 330 | 154 | 176（scalar operand） | 0 |
| `rsqrt` | 115 | 115 | 0 | 0 |
| **合计** | **2119** | **1914** | **205** | **0** |

`add`/`mul` 的 176+29 次请求带 scalar operand：契约层已可规划/可执行（§3.2、§4.1），但 U5
`region.py` 的 scalar 预检（本切片不得修改）把它们列进 `unplannable_operations` 而不是
`missing_contracts`。频次表（主验收指标）不受影响，且 `region_operations_unplannable` 中不再有
`compare`（补丁 B 已闭环）。

## 9. 验证与证据

### 9.1 新增测试（`python/tests/ut/pypto_x/test_execution_ops_registry_batch2.py`，38 项）

覆盖：registry/inventory、6 个新契约的 dtype×rank×shape×attribute 矩阵、add/mul 广播与 scalar
差分、rsqrt 边界语义、视图族三态与 copy bytes、inplace/valid_shape/无序容器、Core IR
`verify_operation` 补丁路径、`plan_region` 注册可见性、doctor/capability。

- focused（持 local 锁、6 线程，rebase 后）：execution 相关 10 个套件 →
  **556 passed / 4 skipped / 0 failed，rc 0**（约 23 s；skips 为 3 个既有 host/vendor
  守卫 + 本切片 1 个 host-probe 守卫的 frozen plan digest 钉）。
  未持锁默认环境：`test_execution_ops_registry_batch2.py` → **38 passed**；
  与 batch1/region/view_mode/discovery/matmul 合并 → **524 passed**（约 23 s）。
- 父提交非空转证据：新测试文件复制到 `41ae82a01` 与 `b643229bd` 父树各跑一次 →
  `ImportError: cannot import name 'BINARY_DTYPES'`，rc=2
  （`logs/parent_commit_new_tests_error.log`、`logs/parent_b643229bd_new_tests_error.log`）。
- 对抗/差分：广播差分 96 例、transpose 全排列差分、slice/broadcast 逐坐标差分全部一致
  （`scripts/` 下一次性检查脚本，见 `raw/batch2_differential_notes.md`）。

### 9.2 零默认变化（12 个 frozen run 的 contract/capability/plan/artifact/output digest）

`scripts/record_batch2_zero_drift.py --mode=frozen` 在父树与本树各跑一次（同机同环境）。
rebase 后父树取 `b643229bd`：默认环境与 local 锁 6 线程环境各做一次对照（锁环境由
`git archive b643229bd python` 解出只读父树）；

- `raw/frozen_record_parent_b643229bd.json` vs `raw/frozen_record_batch2_rebased.json`（默认环境）
  及 `raw/frozen_record_parent_b643229bd_lockenv.json` vs `raw/frozen_record_batch2_lockenv.json`
  （锁环境）：除 `registered_opcodes`（registry 清单按设计增长）外**逐字节一致**；
  包含 10 个 frozen contract digest、默认/按 opcode capability digest、12 个 run 的
  plan/artifact/output digest 与 request/numeric/layout 决策；
- `scripts/compare_frozen_records.py` → `ZERO-DRIFT`（`raw/zero_drift_summary.txt`）；
- 补丁 A/B 的既有合法请求：`where_tensor_only`、`compare_no_broadcast_attribute` 的 plan digest 钉在
  新测试常量里，与父树相同。

### 9.3 batch2 记录（新契约执行证据）

`raw/batch2_record.json`：6 个新契约 digest + 7 个执行 run（含 add tensor×tensor 广播、add
`(2,1)+(1,3)`、mul int32×scalar、rsqrt 边界、transpose/slice/broadcast）的 plan/artifact/output digest、
precision class、layout 决策。视图族 run 的 `layout_copy_bytes` 分别为 12/256/24 字节（真实物化字节数）。

### 9.4 规则 8 全量

- 命令：`env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut python3 -m pytest -q -rs
  -p no:cacheprovider python/tests/ut/pypto_x`，经 `scripts/resource/run_local_heavy.sh` 的 local 锁、
  6 线程（`setsid` 提交，systemd user service 监督；75/69 → sleep 300 重试，未绕过）；
- collect-only（同一选择集）：**2426 tests**（`logs/rule8_collect.log`、`logs/rule8_collect.txt`）；
- 结果：**2415 passed / 11 skipped / 0 failed / 0 errors，rc 0，耗时 1300.73 s（21:40）**
  （`logs/rule8_full_single.log`、`logs/rule8_single_summary.txt`、`raw/rule8_single_counts.json`）；
- 11 个 skip 分解：7 个 CUDA driver 不在本机（`test_cuda_qwen_c2.py`）+ 3 个 host-probe 守卫的
  frozen plan/output digest 钉（batch1 ×2、本切片 batch2 ×1）+ 1 个 U5 region 单算子基线
  host-probe 守卫；均与 batch1/U5 的既有口径一致；
- **chunk-order 对照实验（补充，不是本切片结论）**：为抗外部抢占曾把同一选择集切成 4 个
  `-q -rs` chunk 续跑，聚合为 2392 passed / 12 skipped / 22 failed；22 个失败全部是
  `test_execution_binding_forgery.py` 的 AOCL vendor report 用例（`provider_aocl.py` 的
  `thread_mode_runtime_mismatch`）。**父树 `b643229bd` 在相同 chunk_1 文件顺序下重放得到同一
  22 个失败**（父树另有 1 个 git-archive 路径用例失败；失败集合差 `mine \ parent = ∅`），
  且规范单次调用 rc 0 —— 该失败是 vendor 用例对进程内 BLIS 加载顺序的既有敏感性，
  不是 batch-2 引入的回归（`raw/rule8_chunk_order_failure_attribution.json`、
  `logs/rule8_chunk1_replay_*.log`）。

## 10. 登记边界与下一批建议

### 10.1 未覆盖 / 已知限制

- **U5 region 的 scalar 预检**：`region.py` 仍把任何 scalar operand 判为 unplannable；补丁 A/add/mul
  scalar 只在 `plan_operation`/`execute_operation` 层可用。解除需要 U5 入口的独立切片（本切片禁止改）。
- **`view_mode=require` 对视图族不可满足**：portable 无 strided-view ABI；需要零拷贝视图的模型路径必须
  等待带 stride descriptor 的 provider + alias proof，当前一律结构化拒绝。
- **add/mul vendor 路径**：`allowed_providers=("portable",)`；vendor 候选结构化拒绝
  （`provider_not_allowed_by_contract`），原生实现不在本批。
- **动态 shape / 多输出 / 非 tensor 结果 / 0-d tensor → scalar 的歧义**：全部结构化拒绝；
  `request.inputs[i].shape = []` 在允许 scalar 的角色上解释为 scalar。
- **`broadcast` 的 0 维广播**只支持 portable scalar 参考的语义（0 维结果），没有 stride-0 视图；
- 性能未测；region 规划耗时（decoder ≈ 28 s）只是本机 UNGATED 观测。

### 10.2 下一批（Tier 2）建议

按 §8 新表与 0021 建议：

1. `reduce_mean`（79，浮点误差口径与 reduce_sum 同族）、`concat`（69）、`constant`（66，
   需先冻结 `value` 属性的 canonical 域，建议只支持标量/小 shape 字面量）；
2. 激活家族 `silu`（60）/`sigmoid`（24）/`softplus`（18）：与 `exp`/`rsqrt` 同族，
   `deterministic_bounded`（无解析界），不要写成 exact；
3. 一元 `neg`（30）与逐元素 `sub`（29）：可复用本批 binary 广播 + scalar admission 路径；
4. `split`（24，多输出——需要执行契约扩展多输出 ABI）、`div`（7，整数除法语义需先冻结）、
   `gather`/`embedding`（3，index dtype + 数据搬运）、`identity`（2，语义是 no-op copy）。
