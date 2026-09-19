# 0025 — U 线算子注册第四批（batch4）：Tier-3 单输出四算子 + 除零/index 语义冻结 + 主验收指标 5→1

- 任务：`u-ops-registry-batch4`（batch 0052 切片 B）
- 实现基线：**rebase 后父树 integration tip `32e42abff`**（tree `d5974af0f33f63bcd7c1ff689c062a5d42b423b6`，
  = `6ee250569` + batch3 round-2 `5c1d8dd4`）；本切片最初基于 `a2373408d`，按父方要求 rebase 到 `32e42abff` 后重录全部数字。
- 零默认变化口径：**以 rebase 后新 tip `32e42abff` 为父树**（默认 + OMP=6 两环境）；
  旧 tip `a2373408d` 的对照结果保留为 `raw/frozen_record_parent_a2373408d_*.json` 与
  `raw/qwen_region_missing_contracts_parent_a2373408d.json`，作为新旧对照（见 §7.2 / §8）。
- 分支：`work/u-ops-registry-batch4`（未 push）
- 实现 commit（rebase 后）：`173745aec`（4 契约 + 执行层/测试）、`df2e7a11c`（rule-8 锁环境 host-probe 守卫 +
  针对 `32e42abff` 的 digest 重钉）
- 依赖：`0019`（schema v3 契约、通用入口、按 opcode 收窄的 capability 视图）、`0022`（视图语义三态与主验收指标口径）、
  `0023`（batch3 的 24 契约、零默认变化对照与 Tier-3 建议；round-2 已落地）、`0021 §5`（真实图缺失频次表）
- 结论等级：**实现完成（静态 shape；portable provider 路径）**。未做性能声明；本机数字一律 **UNGATED**
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u-ops-registry-batch4/`

## 0. 摘要

本切片把执行层 opcode 契约从 24 个扩到 **28 个**，新增 0021 §5 剩余 Tier-3 中的四个**单输出**算子：

`div`、`gather`、`identity`、`embedding`

`split`（24 次，多输出）**不在本切片**：它需要父方 `0024-2026-09-19-multi-output-abi-proposal` 的 ABI 落地，
本切片不注册、不改 region schema、不给出单输出退化捷径。

主验收指标（§8）：在 `32e42abff` 上用 U5 `plan_region` 对三个真实 Qwen3.5 图各跑一次，
缺失契约从 **5 类 / 36 次降到 1 类 / 24 次**（只剩 `split`），`div/gather/identity/embedding`
四个 opcode 的缺失次数全部归零；attention 与 gdr 两图变为**整图规划成功**，`unplannable` 0，
decoder 的已登记 op 成功规划数升到 **4669**（§8.3；较 `a2373408d` 上的 4603 多出的 66 次
是 batch3 round-2 的 `constant` scalar-kind 功劳，**不是本切片**，§8 分布表如实呈现）。

### 0.1 冻结兼容策略（关键设计决定）

1. **4 个新契约**沿用 registry 级 schema **v3**（`0019 §0.1`）：`contract_version = "1.0"`、
   `allowed_providers = ("portable",)`、`effects = pure`、`verifier = generic`；
   本切片**不改** 24 个既有契约的 document/digest，零默认变化以 rebase 后父树 `32e42abff`
   为准逐字节对照（§7.2）。round-2 对 `reduce_mean`/`constant` 的契约修订属于 batch3 切片，
   已包含在父树中，不是本切片引入。
2. **capability 视图按批次冻结累积集合**：`operation_provider_capability(opcode)` 现在按 opcode
   返回冻结集合——batch1 → 10 个 contract、batch2 → 16 个、batch3 → 24 个、batch4 → 28 个。
   默认探测（`default_providers`）仍只声明 matmul/qmatmul 两个契约，matmul/qmatmul 路径不受影响。
3. **整数 `div` 是显式增量准入**：`validate_program` 新增 `allow_integer_division=False` 关键字；
   只有 `CpuScalarProgram`、`lower_cpu_scalar`（portable trunk）和执行层的
   `verify_lowering_accepts` 对 `div` 传 `True`。向量 planner（AVX2/AVX512/SVE）保持 0051 前的
   float-only 准入，不会获得未实现的整数除法路径。
4. **除零是执行期结构化拒绝**：除数是运行时数据，plan 期不可见；执行器按值拒绝
   （`OP_CONTRACT_INVALID / division_by_zero`），浮点也不发布 `inf/NaN`。契约文档写明该口径，
   但不把“未知除数”伪造成 plan 期检查。
5. **index 值是运行时数据**：负 index / 越界 index 在执行期结构化拒绝，分别给
   `negative_index_not_supported` / `index_out_of_range`；契约层只冻结 dtype 域与 shape 规则，
   不伪造静态 index 值检查。
6. 不修改 `pypto/__init__.py`、`python/pypto/execution/region.py`、`python/pypto/portable/qwen35.py`，
   不新增环境变量。

## 1. 四个新契约的逐算子摘要

所有 4 个契约：`attributes` 白名单之外一律 `unknown_attribute`；序列型属性只接受有序 list/tuple。

| opcode | 元数/操作数 | shape 规则 | dtype 域 | 结果 dtype | 属性白名单 | portable executor 符号 |
|---|---|---|---|---|---|---|
| `div` | 2（left, right） | 尾轴对齐广播；scalar 为 rank-0 恒等（任一侧）；两个 scalar 拒绝 | 全部 portable dtype 去 bool | = 操作数共同 dtype | 无 | `pypto.backends.cpu.runtime._binary_value` |
| `gather` | 2（data, index） | `shape(out) = shape(data)[:axis] + shape(index) + shape(data)[axis+1:]`；rank(out) 1..4 | data=全部 portable dtype（含 bool）；index∈{int32,int64} | = data dtype | `axis`/`dim`（默认 0，负值归一化；同现必须一致） | `pypto.backends.cpu.runtime._gather_value` |
| `embedding` | 2（weight, index） | weight rank 恰为 2 `(vocab, dim)`；`shape(out) = shape(index) + (dim,)`；rank(out) 2..4 | weight=全部 portable dtype（含 bool）；index∈{int32,int64} | = weight dtype | `axis`/`dim`（只接受归一化 0；其他轴 `embedding_axis_not_supported`） | `pypto.backends.cpu.runtime._gather_value`（`embedding=True`） |
| `identity` | 1（input） | 逐元素恒等；`shape(out)=shape(in)`；rank 1..4 | 全部 portable dtype（含 bool） | = input dtype | `inplace`（只允许 false）、`valid_shape`/`valid_shapes`（必须等于输出 shape） | `pypto.backends.cpu.runtime._identity_value` |

广播的精确规则与 batch1/2/3 同源（执行层直接调用 `pypto.lowering.cpu._broadcast_shape` 后复核）：
尾轴对齐；每维相等或其一为 1；`0` 只与 `0`/`1` 广播（结果为 0）；其他组合一律 `broadcast_shape_mismatch`。
`div` 的广播与 scalar admission 复用 batch2/3 的 `_build_elementwise_binary_definition` 模板
（`build_binary_elementwise_program` 把 tensor×tensor 广播显式降为 `broadcast + elementwise`）。

`contract_digest`（rebase 后实现树，`raw/frozen_record_batch4_rebased.json`；4 个契约 digest 不因
round-2 rebase 变化）：

```text
div       sha256:8ff19a8110644dcd81898d7084213294b1d469085f7819b60eacc811b74529b7
embedding sha256:8a6ea6af4635b3f22a8d26ad8cb10c383a1ee85a7be109d171a6946bdddba986
gather    sha256:37933a86227c13c7006d7b978a66bdeff2487db13bdb155a1c49dcf95326425b
identity  sha256:e1a7e4f73600ede7097134af98d5f02d9a6584d5ab596b527dfce1346300b561
```

按 opcode 收窄的 28-contract capability 视图 digest（`raw/frozen_record_batch4_rebased.json`；
含 batch3 round-2 修订后的 `reduce_mean`/`constant` digest，故与 `a2373408d` 时代不同）：

```text
div       sha256:1e8954ee0e3f34856202f3ef75c1c4fd059f0bf05e9914c7f6c3570d2841c4b2
embedding sha256:1d8f150103bdce7ab288c0802a31c8e7eb893ad0aeecbbb55dd5fd410b8e0805
gather    sha256:74520cfc563d743ff25ac8b602662f8bc62d660833ec307b2a37573742385e71
identity  sha256:32a1975979a62356fdd7d7429d26de80f1448c3f24b51fcad6073908af84354f
```

## 2. 精度声明（诚实口径）

| opcode / 路径 | 类别 | bound kind | 理由 |
|---|---|---|---|
| `div` 整数 | **exact** | `exact_contract_bound`（proven，bound=0） | 商 = `trunc(left/right)`，用精确 Python 整数做**向零截断**（与冻结的整数 `reduce_mean` 同规则），随后一次带范围检查的输出 cast；除零与商越界都 fail-closed，无回绕/饱和 |
| `div` 浮点 | `deterministic_bounded` | `deterministic_serial_order_no_registered_analytic_bound` | 操作数精确解码到 binary64，一次除法 + 一次 RNE cast；**无解析上界**；除数为 ±0.0 时结构化拒绝，不发布 `inf/NaN` |
| `gather` 全部 | **exact** | `exact_contract_bound`（proven，bound=0） | 纯搬运：按显式 index 选择已存元素并逐位复制；无算术、无转换；NaN/±0/±inf/次正规/bf16 位型原样保留 |
| `embedding` 全部 | **exact** | 同上 | rank-2 权重表的 axis-0 行选择，逐位复制 |
| `identity` 全部 | **exact** | 同上 | 零偏移整 shape 恒等复制，逐位保留 |

`numeric.require_proven_deviation_bound = true` 时：

- `div` 整数路径、`gather`、`embedding`、`identity`（exact）正常解析；
- `div` 浮点（`deviation_bound_available=false`）**fail-closed**
  （`NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable`，field
  `numeric.require_proven_deviation_bound`），portable provider 也生效。

## 3. 语义边界（契约文字 + 用例）

- **NaN**：`div` 非零除数路径按 IEEE-754 传播 NaN；`gather`/`embedding`/`identity` 原样搬运 NaN 位型
  （HostTensor 用 Python float 存储，payload 不承诺保留）。
- **±0**：`div` 分子 ±0 的符号按 IEEE 保留（`-0.0/2.0 = -0.0`、`0.0/-2.0 = -0.0`）；
  任何零除数（`+0.0`/`-0.0`/整数 0）在除法前结构化拒绝；`gather`/`embedding`/`identity` 原样保留符号位。
- **±inf**：`div` 在非零除数下遵循 IEEE（`inf/finite = signed inf`、`finite/inf = signed zero`、
  `inf/inf = NaN`）；f32/bf16 溢出饱和为 signed inf，**float16 有限溢出 fail-closed**（`operand_value_rejected`）；
  搬运族原样保留。
- **次正规**：`div` 在 binary64 中计算后一次 RNE cast（可下溢为次正规或 0）；搬运族原样保留；
  用例含最小正 float32 次正规 `0x1p-149`。
- **bf16 位型**：`div` 按目标格式 RNE；搬运族 bf16 值逐位保留；
  用例覆盖 subnormal `0x0001`、`-inf 0xFF80`、NaN `0x7FC0`。
- **整数极值**：`div` 不回绕、不饱和；`int8(-128)/-1`、`int64(-2^63)/-1` 这类越界商 fail-closed
  （执行期 `operand_value_rejected`，输出缓冲保持原值）；大整数（`2^62+1`）除法不经过 binary64，结果精确。
- **空张量 / 零长度**：`div` 支持 `0` 维广播；`gather` 的零长度 index 维产生零长度输出维，
  零长度 data 轴只接受空 index（任何 index 值都越界）；`embedding` 的零长度 index 维产生零长度输出，
  `vocab=0` 时只有空 index 合法；`identity` 0 元素输入 → 0 元素输出。
- **原地/别名**：`div` 是 `must_not_alias`；`gather`/`embedding`/`identity` 声明 `may_alias`
  （允许有 stride/index 证明的 provider 共享存储），portable trunk 无 strided-view ABI，
  始终物化独立 row-major 输出；执行期 `_check_no_alias_any` 兜底。

## 4. 五个冻结决定

### 4.1 `div` 整数除法：向零截断，且除零结构化拒绝

- 整数路径：`quotient = trunc(left/right)`（`abs(a)//abs(b)` 加符号），**不是 floor**；
  例：`-7/2 = -3`、`7/-2 = -3`、`-7/-2 = 3`；随后对输出 dtype 做带范围检查的 cast。
- 浮点路径：非零除数下按 IEEE-754 binary64 一次除法 + 一次 RNE cast；
  **除数为 `+0.0`/`-0.0` 时结构化拒绝**（`division_by_zero`），不沿用 IEEE 的 `inf/NaN` 输出。
  这是有意的 fail-closed 口径：契约承诺“除零一律拒绝”，绝不静默发布无穷大。
- 两个 scalar 操作数 → `scalar_result_not_supported`（执行 ABI 只返回 tensor）。

### 4.2 index dtype 与负 index / 越界

- `gather`/`embedding` 的 index dtype 域**恰好是 `int32`/`int64`**；bool、浮点、其他整数宽度
  一律 `dtype_outside_contract_domain`（lowering 的宽松 validator 不构成契约许可）。
- **不支持负 index**：`selected < 0` → 执行期 `OP_CONTRACT_INVALID / negative_index_not_supported`；
  不实现 Python wrap，也不做 `+dim` 规范化。
- **越界 index**：`selected >= shape(data)[axis]`（embedding 为 `vocab`）→
  `OP_CONTRACT_INVALID / index_out_of_range`；绝不 clamp、不取模。
- index 值只来自执行期 tensor 数据，plan 期不推断；违规矩阵与差分用例覆盖负数、越界、空 index、
  零长度轴。

### 4.3 `gather` 的 shape 规则与视图义务

- `shape(out) = shape(data)[:axis] + shape(index) + shape(data)[axis+1:]`；`axis`/`dim` 默认 0，
  负值归一化，二者同现必须一致；输出 rank 必须在 1..4。
- `view_obligation = may_alias`（与 batch2 视图族/索引搬运口径一致）：逻辑上是指标选择的重排；
  有能力给出 index/stride 描述的 provider 可以别名源存储；portable trunk 始终物化
  `element_count(out) × dtype_size(out)` 字节的独立缓冲。
- `layout.view_mode` 三态：`forbid`/`prefer` → `resolved_view_mode="materialized"` 且有真实
  `layout_copy_bytes`；`require` → `ALIAS_PROOF_UNAVAILABLE / portable_trunk_has_no_strided_view_abi`，
  绝不静默降级为拷贝。

### 4.4 `embedding` 是 axis-0 特化

- weight 必须是 rank-2 `(vocab, dim)`；index rank 1..3（rank 4 会使输出 rank 5，拒绝）；
  `shape(out) = shape(index) + (dim,)`。
- 真实图写作 `embedding(weight, input_ids, axis=0)`；契约接受 `axis`/`dim` 但只接受归一化 0，
  其他轴/越界轴/非整数轴分别 `embedding_axis_not_supported` / `embedding_axis_invalid`。
- 与 `gather` 同享 index 值检查与 `may_alias` 物化口径；`exact`。

### 4.5 `identity` 的 view 三态

- `view_obligation = may_alias`：逻辑结果就是输入 buffer（stride 1），有 alias proof 的 provider
  可以零拷贝；**但 portable trunk 没有 strided-view ABI**，因此：
  - `forbid`（默认）/`prefer` → 显式物化独立 row-major 输出，`layout_copy_bytes = element_count × dtype_size`；
  - `require` → 结构化 `ALIAS_PROOF_UNAVAILABLE / portable_trunk_has_no_strided_view_abi`，**绝不静默降级**。
- `inplace=true` → `identity_inplace_not_portable`；`valid_shape`/`valid_shapes` 非空时必须等于输出 shape
  （否则 `identity_valid_shape_not_representable`；空/非法序列 → `identity_valid_shape_invalid`）。
- 值语义是逐位复制：NaN/±0/±inf/次正规/bf16 位型原样保留，`exact`。

## 5. fail-closed 矩阵（新增触发器）

| 触发 | 错误（code / reason） | 发生层 |
|---|---|---|
| 未知/未注册 opcode（`split`/`view`/合成名） | `OP_CONTRACT_INVALID / opcode_not_registered` | registry/resolver |
| `div` 的 bool / `gather`/`embedding` 非 int32/int64 index | `dtype_outside_contract_domain` | verifier |
| 混合 dtype / scalar 与 tensor dtype 不一致 | `dtype_mismatch` | verifier |
| 两个 scalar 操作数的 `div` | `scalar_result_not_supported` | verifier |
| `div` 广播不合法 | `broadcast_shape_mismatch` | verifier（复用 lowering `_broadcast_shape`） |
| rank 越界（`div`/`identity` 输入 rank>4 或 0；`gather`/`embedding` 输出 rank>4；weight 非 rank-2） | `rank_out_of_range` | verifier |
| `gather`/`embedding` 输出 shape 与规则不符、`identity` 输出 shape/dtype 不符 | `shape_rule_violation` / `dtype_mismatch` | verifier |
| 未知属性 | `unknown_attribute` | verifier（白名单） |
| 无序容器属性（`identity.valid_shape` 传 set/generator 等） | `identity_valid_shape_invalid` | verifier |
| `gather`/`embedding` 轴越界/非整数、axis 与 dim 冲突 | `gather_axis_invalid` / `embedding_axis_invalid` / `conflicting_attribute_aliases` | verifier |
| `embedding` 非 0 轴 | `embedding_axis_not_supported` | verifier |
| `identity` 的 `inplace=true` / 不可表示 `valid_shape` | `identity_inplace_not_portable` / `identity_valid_shape_not_representable` | verifier |
| `require_proven_deviation_bound=true` 遇浮点 `div` | `NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable` | resolver |
| `layout.view_mode=require` 对 `gather`/`embedding`/`identity` 无法证明零拷贝 | `ALIAS_PROOF_UNAVAILABLE / portable_trunk_has_no_strided_view_abi` | layout_verify |
| `div` 执行期除零（整数 0、float ±0.0，tensor/scalar 皆同） | `OP_CONTRACT_INVALID / division_by_zero` | portable provider |
| `gather`/`embedding` 负 index | `OP_CONTRACT_INVALID / negative_index_not_supported` | portable provider |
| `gather`/`embedding` 越界 index、零长度轴的任何 index | `OP_CONTRACT_INVALID / index_out_of_range` | portable provider |
| `div` 整数越界商 / f16 溢出 / 非有限值 cast 到整数 | `OP_CONTRACT_INVALID / operand_value_rejected` | portable provider |
| 输出别名输入 | `output_input_alias` | provider |
| plan 被篡改（含改写 `layout_copy_bytes`、embedding axis、index dtype） | `ARTIFACT_MISMATCH / plan_digest_mismatch` | plan 校验器 |
| region 文档被篡改 | `ARTIFACT_MISMATCH / region_digest_mismatch` | region 校验器 |

## 6. 通用入口与执行层变更（严格加法）

- `plan_operation` / `resolve_operation` 的固定 arity / 可变 arity 行为不变；4 个新契约都是固定 arity。
- `providers.build_operation_program` 的广播降级分支从 `add/mul/sub` 扩到 `+div`；
  `verify_lowering_accepts` 对 `div` 传 `allow_integer_division=True`。
- `pypto/lowering/cpu/__init__.py` 新增 `allow_integer_division=False` 关键字（`_validate_operation` /
  `validate_program`）；`CpuScalarProgram.__post_init__` 与 `lower_cpu_scalar` 对 portable trunk 传 `True`。
- `pypto/backends/cpu/runtime.py`：
  - `_binary_value("div", ...)` 整数路径改为精确向零截断；零除数抛
    `CpuScalarDivisionByZeroError`（`CpuScalarValueError` 子类）；
  - `_gather_value` 负/越界分别抛 `CpuScalarNegativeIndexError` / `CpuScalarIndexOutOfRangeError`；
  - 新增 `_identity_value`（`identity` 的 portable executor 符号，逐位复制）。
- `PortableProvider.execute_operation` 把上述子类映射为 `division_by_zero` /
  `negative_index_not_supported` / `index_out_of_range`；其他 `CpuScalarValueError` 仍是
  `operand_value_rejected`（既有 pytest 的 `CpuScalarExecutionError` 匹配关系不变）。
- `op_bench` 的“未登记 opcode”注入示例从真实算子名改为合成名
  `not_a_registered_opcode`（runner + 同名测试），并注释说明测试意图是
  “未登记 opcode”，避免下一次注册再次使证据失效；`test_op_bench_framework.py` 语义不变。

## 7. 验证与证据

### 7.1 新增测试（`python/tests/ut/pypto_x/test_execution_ops_registry_batch4.py`，75 项）

覆盖：registry/28 契约 digest、按批次冻结的 capability 视图、precision class 与
`require_proven_deviation_bound`；`div` 整数向零截断/极值越界/浮点 NaN/±0/±inf/次正规/f16 溢出/
广播/scalar/除零矩阵；`gather`/`embedding` 的 shape 与差分、index dtype 域、负/越界 index、
空张量/零长度轴、bf16 位型、`may_alias` + materialize 字节数 + `view_mode=require` 拒绝；
`identity` 的位型复制/全 dtype/零元素/三态视图/inplace/valid_shape/无序容器；
plan/report 自包含、冻结 plan JSON 重放、三类篡改拒绝、`plan_region` 四算子 region 规划与
`validate_region_plan(re_resolve=True)` 往返。

- focused（rebase 后默认环境，非持锁）：本切片 75 passed；与 batch1/batch2/batch3（round-2）/region/
  discovery/matmul/view_mode/binding/report/cpu-scalar/op-bench 相关 19 个套件合并 →
  **1000 passed / 0 failed**（约 49 s；`logs/focused_execution_suites.log`）。
- 锁同源 OMP=6 环境（同一 capability digest `70d8cf05…`，未单独抢锁）：本切片文件
  **73 passed / 2 skipped**（2 个 host-probe 守卫的 frozen digest 钉，与 batch1/2/3/U5 口径一致；
  `logs/focused_batch4_omp6_rebased.log`）。
- 父提交非空转证据：新测试文件在 rebase 后父树 `32e42abff` 上 collect 即
  `ImportError: cannot import name '_identity_value' from 'pypto.backends.cpu.runtime'`，rc=2
  （`logs/parent_32e42abff_new_tests_error.log`；最初父树 `a2373408d` 的同名证据为
  `logs/parent_commit_new_tests_error.log`）。
- 差分口径：执行层 `execute_operation` 输出 vs 契约声明的 portable 参考 helper
  （`_binary_value` / `_gather_value` / `_identity_value`）逐位比较（NaN 按 NaN、±0 按符号位）。

### 7.2 零默认变化（以 rebase 后新 tip `32e42abff` 为父树；旧 tip 对照并存）

`scripts/record_batch4_zero_drift.py --mode=frozen` 在父树（integration worktree，rebase 前为
`a2373408d`、rebase 后为 `32e42abff`）与本树各跑一次，覆盖 **24 个既有契约**与 **17 个 frozen run**
（matmul f32/bf16、qmatmul、batch1 8 个、batch2 6 个（含 mul scalar）、batch3 8 个）：

- **新 tip `32e42abff`（规范口径，默认环境，capability `e0a9dac8…`）**：
  `raw/frozen_record_parent_32e42abff_default.json` vs `raw/frozen_record_batch4_rebased_default.json`
  → **ZERO-DRIFT**（`raw/zero_drift_summary_rebased_default.txt`）；
- **新 tip `32e42abff`（规范口径，local 锁 OMP=6 环境，capability `70d8cf05…`）**：
  `raw/frozen_record_parent_32e42abff_lockenv.json` vs `raw/frozen_record_batch4_rebased_lockenv.json`
  → **ZERO-DRIFT**（`raw/zero_drift_summary_rebased_lockenv.txt`）；
- **旧 tip `a2373408d`（rebase 前历史对照）**：默认/锁环境同样 ZERO-DRIFT
  （`raw/frozen_record_parent_a2373408d_*.json`、`raw/frozen_record_batch4_default.json` /
  `raw/frozen_record_batch4_lockenv.json`、`raw/zero_drift_summary.txt` /
  `raw/zero_drift_summary_lockenv.txt`）；
- **新旧对照结论**：两个父树 tip 下，本切片对 24 个既有契约 + 17 个 frozen run 都是逐字节零漂移；
  唯一差异来自 batch3 round-2 自身的契约修订——`reduce_mean` digest `ec7b7a62…→3ab5ee5a…`、
  `constant` digest `9717b6c3…→cf33e1a6…`，batch3 的 24-contract 视图 digest 随之变化
  （`neg`/`concat`/`constant`：`e7fc4939…/af1df5bf…/68906ce0…` →
  `786ce38e…/79007d33…/8ad6887c…`），batch4 的 28-contract 视图与执行 plan digest 相应重钉；
  4 个 batch4 契约 digest 本身不变（§1）。
- 按 opcode 收窄的 capability 视图：batch1（`exp`）`77714e28…`、batch2（`add`）`67cdb3f8…`
  在两个 tip 下都不变（contract 集合 10/16 个）；batch4 的四个 28-contract 视图 digest 见 §1。

### 7.3 batch4 记录（新契约执行证据）

`raw/frozen_record_batch4_rebased.json`（`--mode=batch4`，rebase 后）：4 个新契约 digest + 5 个执行 run
（div 浮点、div 整数、gather、embedding、identity）的 plan/artifact/output digest、precision class、
layout 决策（`may_alias`/`materialized`/真实 `layout_copy_bytes`）；旧 tip 的同名证据为
`raw/frozen_record_batch4_parent_a2373408d.json`。

### 7.4 规则 8 全量（持 local 锁）

- 命令：`env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut python3 -m pytest -q -rs
  -p no:cacheprovider python/tests/ut/pypto_x`，经 `run_local_heavy.sh` 的 local 锁、6 线程
  （systemd-run scope + 动态 cgroup；75/69 → sleep 300 重试，未绕过）；
- **规范结果（rebase 后父树 `32e42abff`）**：
  - collect（同一选择集）：**2661 tests**（`logs/rule8_collect.log`、`logs/rule8_collect.txt`）；
  - **2647 passed / 14 skipped / 0 failed / 0 errors，rc 0，耗时 1310.57 s（21:50）**
    （`logs/rule8_full_single.log`、`logs/rule8_single_summary.txt`、`logs/rule8_rc.txt`、
    `raw/rule8_single_counts.json`；2026-09-20 02:45:31–03:07:25）；
  - 14 个 skip 分解：7 个 CUDA driver 不在本机（`test_cuda_qwen_c2.py`）+ 7 个 host-probe 守卫的
    frozen digest 钉（batch1 ×2、batch2 ×1、batch3 ×1、U5 region ×1、本切片 batch4 ×2）；
    均与 batch1/2/3/U5 的既有口径一致；
  - 尝试史（`logs/rule8_lock_attempts.txt`）：前 6 次（02:15:31–02:40:31）均遇同一把锁的
    `75 BUSY`（`verify-0051-ops-batch3-r2`、`lcvex` 在跑），按纪律 sleep 300 重试未绕过；
    第 7 次（02:45:31）取得锁并 rc 0。
- **历史结果（rebase 前父树 `a2373408d`）**：collect 2650；
  2636 passed / 14 skipped / 0 failed，rc 0，1320.37 s（`logs/rule8_full_single_parent_a2373408d.log`、
  `raw/rule8_single_counts_parent_a2373408d.json`）。其尝试史中另有一次作者侧事故与一次作者侧测试缺陷：
  1. 首次启动（23:42:23）在 23:48:48 被本轮作者会话的 shell 重置 SIGTERM 中断
     （`logs/run_local_heavy_rule8.log` 的 `terminate reason=supervisor_signal_15`），无指标数据；
     是作者侧操作事故，不是测试失败。
  2. 第一次**完成**的全量（23:49:19–00:11:11）为 **2636 passed / 12 skipped / 1 failed**：
     失败是本切片新测试的 `test_frozen_default_capability_and_plan_digests_are_unchanged`
     未按 batch1/2/3/U5 的既有口径加 host-probe 守卫，而锁环境的 capability digest 是
     OMP=6 的 `70d8cf05…`（默认环境是 `e0a9dac8…`）。已在 `aee7cdcb0` / `df2e7a11c`
     拆成两个带守卫的测试（锁环境 skip，默认环境断言）。归因：作者侧测试缺陷，
     **不是契约/执行/数值回归**；该次运行其余 2636 项与 12 个既有 skip 全部正常。
- 规范结果取 rebase 后第 7 次完成的 rc 0 全量；`raw/rule8_single_counts.json` 由
  `scripts/summarize_rule8.py` 从该次日志生成。

## 8. 主验收指标（真实 Qwen3.5 三图，口径与 0021 §5 / 0022 §8 / 0023 §8 一致）

证据：`raw/qwen_region_missing_contracts.json`（脚本
`scripts/qwen_region_missing_contracts_batch4.py`，对三个 builder 的整 program 各跑一次
`plan_region`，read-only、不执行 provider；经 `run_local_heavy.sh` 的 local 锁）。

### 8.1 逐图分布（batch3 round-1 → batch4，父树 `32e42abff`）

| builder | ops | batch3 缺失 contract（次数） | batch4 缺失 contract（次数） | batch4 状态 / unplannable |
|---|---|---|---|---|
| `build_attention_kv_cache` | 20 | div 1, gather 2 | **无** | `planned`，region digest `92e21b47…`，unplannable 0 |
| `build_qwen35_gdr_recurrent_state` | 123 | identity 2 | **无** | `planned`，region digest `aa413baa…`，unplannable 0 |
| `build_qwen35_text_decoder_graph` | 4550 | split 24, div 6, embedding 1 | **split 24** | `rejected`（只剩 split），unplannable 0 |
| **合计** | **4693** | **36 / 5 类** | **24 / 1 类** | **unplannable 0** |

attention / gdr 两图在 batch3 时因缺 contract 结构化拒绝；batch4 后**整图规划成功**
（`region_class = deterministic_bounded_unproven`，含 `div` 浮点这一未注册解析界的路径）。
decoder 仍是结构化拒绝，但拒绝原因只剩 `split`（多输出 ABI，见 §9）——这是**预期**，不是回归。
`constant` 的 66 次在父树 `32e42abff` 上已由 batch3 round-2 的 rank-0 scalar-kind 输出变为可规划
（`registered_planned_total` +66），**归 round-2 切片，不计入本切片**；旧 tip `a2373408d` 的
region JSON 保留为 `raw/qwen_region_missing_contracts_parent_a2373408d.json` 对照。

### 8.2 缺失 contract 合并表（5 → 1 类）

| opcode | batch3 缺失次数 | batch4 缺失次数 | 覆盖图 | 结论 |
|---|---|---|---|---|
| `split` | 24 | **24** | decoder | 不变（多输出 ABI 未定义，依赖 0024 提案） |
| `div` | 7 | **0** | decoder + attention | 消失（新增契约；7/7 plan 成功） |
| `gather` | 2 | **0** | attention | 消失（2/2 plan 成功） |
| `identity` | 2 | **0** | gdr | 消失（2/2 plan 成功） |
| `embedding` | 1 | **0** | decoder | 消失（1/1 plan 成功） |
| **合计** | **36 / 5 类** | **24 / 1 类** | — | **-4 类 / -12 次** |

### 8.3 已登记 op 的规划统计（三图合并，父树 `32e42abff`）

`registered_planned_total`（region 入口对每个 op 都调用 `plan_operation`，不是首错即停）
= **4669**：`a2373408d` 上 4603 + batch3 round-2 的 `constant` scalar-kind 66（他切片功劳）
+ 本切片 4 个 opcode 的真实图请求 12：

| opcode | 真实图出现 | plan 成功 | 缺失 contract | unplannable（原因） |
|---|---|---|---|---|
| `div` | 7 | 7 | 0 | 0 |
| `gather` | 2 | 2 | 0 | 0 |
| `identity` | 2 | 2 | 0 | 0 |
| `embedding` | 1 | 1 | 0 | 0 |
| **本批合计** | **12** | **12** | **0** | **0** |
| `constant`（batch3 round-2，对照） | 66 | 66 | 0 | 0 |
| `split`（未注册，对照） | 24 | 0 | 24 | 0 |

### 8.4 结论

- 主验收目标**精确达成**：缺失契约 **5 类 / 36 次 → 1 类 / 24 次**，只剩 `split`；
  `div/gather/identity/embedding` 四个 opcode 的缺失次数全部归零；`unplannable` 0。
- attention 与 gdr 两张图从“结构化拒绝”变成**整图规划成功**；decoder 的整体拒绝原因收敛为
  `split` 一项。
- 机器可读证据：`raw/qwen_region_missing_contracts.json`（`summary` / `missing_contract_priority` /
  `target_opcode_transition` / `planning_statistics` 字段）与
  `raw/missing_contracts_comparison.json`，脚本
  `scripts/qwen_region_missing_contracts_batch4.py`；本次运行在 local 锁、6 线程环境
  （capability digest `70d8cf05…`），与 §7.2 的 lockenv 对照同源。

## 9. 未覆盖清单与 `split` 依赖提示

- **`split`（24 次）不在本切片**：多输出 opcode；当前执行契约与 region schema 都以单输出为前提
  （region 预检 `multi_output_not_supported_by_execution_contract`）。需要先落父方
  `0024-2026-09-19-multi-output-abi-proposal` 的 ABI（plan/report 的 `outputs[]`、输出缓冲/别名义务、
  region 边界签名与 digest preimage 扩展），再独立成片实现；本切片不注册、不做“只取第一个输出”
  之类的退化捷径。
- **`view`/`contiguous`/量化 primitive** 仍未注册（0021 §6.1 的历史清单），全部结构化拒绝。
- **vector/vendor 路径**：4 个契约 `allowed_providers=("portable",)`；AVX2/AVX512/SVE 与 vendor
  候选结构化拒绝（`provider_not_allowed_by_contract`），原生实现不在本切片。
- **动态 shape / 多输出 / 非 tensor 结果**：全部结构化拒绝。
- **除零不发布 IEEE `inf/NaN`**：这是契约的有意收窄（fail-closed）；若未来模型路径需要 IEEE
  “除零得 inf/NaN”的语义，必须新开契约版本或属性，不能在现有 `div` 上静默放宽。
- **负 index 不支持**：需要 wrap 语义的调用方必须先在前端做显式规范化并插入 `where/cast` 等。

## 10. 已知边界

- 标量操作数只在 `div`（及 batch2/3 的 binary/where）显式允许处出现；`gather`/`embedding`/`identity`
  不接受 scalar 操作数。
- `gather`/`embedding` 的 index rank 文档域是 1..4（provider 请求包络），有效契约规则更窄：
  `gather` 要求合成输出 rank ≤4，`embedding` 要求 index rank ≤3（输出 `shape(index)+(dim,)` ≤4）；
  越界在 verifier 层 `rank_out_of_range`。
- 零除数 / 负 index / 越界 index 是**执行期值检查**：plan 期只做 dtype/shape/属性拒绝，
  plan digest 不承诺“除数非零”或“index 在界内”；执行期拒绝是 `OP_CONTRACT_INVALID`
  且输出缓冲保持原值。
- 性能未测；region 规划耗时只是本机 UNGATED 观测，不作为性能声明。
- 本切片改了既有测试中把 `div`/`gather`/`identity` 当“未登记 opcode”示例的断言
  （改为仍属未登记的 `split`/`view`/合成名）、`test_cpu_scalar.py` 的整数除法准入用例
  （默认仍拒绝，新增显式 opt-in 断言）与 `op_bench` 注入名；本切片不改 24 个既有契约的
  document/digest，零默认变化以 rebase 后父树 `32e42abff` 为对照逐字节成立（§7.2；
  batch3 round-2 自身对 `reduce_mean`/`constant` 的 digest 修订不属于本切片）。

## 11. rebase 记录（batch3 round-2 落地后）

父方在 batch3 round-2 落地后给出新 tip `32e42abff`（tree `d5974af0f33f63bcd7c1ff689c062a5d42b423b6`
= `6ee250569` + 作者 `5c1d8dd4`）。本分支已按约定 `git rebase --onto 32e42abff a2373408d`
完成（仅 `providers.py` 一处冲突：round-2 的 `results = CpuScalarRuntime().launch(request)`
scalar-kind 返回值与本切片的除零/index 子类异常处理合并，两者都保留；未回退 round-2 任何改动）。

- rebase 后 commit：`173745aec`（4 契约 + 执行层/测试）、`df2e7a11c`（rule-8 锁环境守卫 +
  针对 `32e42abff` 的 digest 重钉）；rebase 前的历史 commit 为 `6e1492cbb` + `aee7cdcb0`
  （备份分支 `backup/u-ops-registry-batch4-preRebase`）。
- **以 `32e42abff` 为父树的零默认变化**：24 个既有契约 + 17 个 frozen run，默认 + OMP=6 两环境
  均 ZERO-DRIFT（§7.2）；batch3 round-2 自身导致的 digest 变化（`reduce_mean`/`constant` 及
  batch3 24-contract 视图）已重钉在本切片测试中，不冒充“逐字节不变”。
- **以 `32e42abff` 为父树的 region**：缺失契约仍 **1 类 / 24 次（仅 `split`）**，
  `unplannable` **0**，`registered_planned_total` **4669**（= 4603 + round-2 的 `constant` 66
  + 本切片 12）；attention/gdr 整图 planned（§8）。
- focused（rebase 后）：19 套件 **1000 passed / 0 failed**；rule-8 全量按锁重跑（§7.4）。
- 本切片未触碰 integration / 其他 worktree，未 push；`pypto/__init__.py`、
  `execution/region.py`、`portable/qwen35.py` 未改。
