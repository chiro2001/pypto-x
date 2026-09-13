# int8 exact-blocked 供应商条目契约（AOCL s8s8s32os32 分块精确累加）

文档编号：`0015`

日期：2026-09-13（Asia/Shanghai）

状态：`IMPLEMENTED_PENDING_BATCH_VERIFICATION`

任务：`int8-exact-blocked`（lock：`waves.W8.batch_0047_int8_exact_blocked_2026_09_13`）

实现真源：`python/pypto/execution/provider_aocl_int8.py`（新 provider 条目与分块执行）、
`python/pypto/execution/capability.py`（opt-in 准入）、
`python/pypto/execution/resolver.py`（选择语义）、
`python/pypto/execution/entry.py`（capability 装配）、
`python/pypto/execution/provider_identity.py`（report schema v3/v4 身份锚）。
本文冻结契约与证明义务，不复制实现。

## 1. 结论

1. 新增**可选**（opt-in）provider 条目
   `vendor:aocl-lpgemm-int8-exact-blocked`。它调用同一 pinned 库、同一符号
   `aocl_gemm_s8s8s32os32`、同一参数语义（`alpha=1`、`beta=0`、`post_op=None`、
   row-major `A[M,K]`/`B[K,N]`、int32 `C[M,N]`），但把 K 维切成
   `Kc = min(2048, floor(2**24 / (max|a|·|b|)))` 列的小段，使内核在该段内每一次
   f32 往返都作用在 `|v| ≤ 2**24` 的整数上而成为**恒等**；段输出再用 Python 任意精度
   整数相加并做 int32 范围校验。相对契约的精确 int32 和，偏差上界**构造性为 0**。
2. envelope 复用 `deviation_bound_kind = "exact_contract_bound"`（不新增 kind），
   并额外写入可审计字段：`bound_derivation`（Kc 规则、`f32_exact_integer_limit=16777216`、
   每段界公式、顺序无关性）、`proof_obligation`、`residual_assumptions`（可检验项 +
   撤回路径）。`numeric_class = "exact"`，整个被接受的 K 范围（含 K=1024/1040/1041/
   2048/2049/16696/133144 与 all-127、mixed、random、all--128 拒绝面）都用差分矩阵钉住。
3. **零默认变化**：新条目不进 `default_providers`、不进
   `OpDefinition.allowed_providers`，默认 payload/policy 下的 capability 快照 digest、
   candidate 列表、plan digest、artifact digest、执行输出与 `393d28e0f` 逐字节一致
   （树间对照证据 `raw/zero_default_change_{baseline,worktree}.json`，diff 仅 tree label）。
4. `numeric.require_proven_deviation_bound=true` 下的新语义：显式 opt-in 后新条目进入
   candidate 并被选中、成功执行；`deviation_bound_kind="measured_snapshot"` 的
   legacy `aocl_int8` 仍以 `proven_deviation_bound_unavailable` fail-closed；fused
   `sym_quant` 路径仍无 proven bound、仍不可路由（见 §7）。
5. 性能 `UNGATED`：成本是 `ceil(K/Kc)` 次内核调用 + O(MK)+O(KN) 的 max|code| 扫描 +
   O(MN·ceil(K/Kc)) 的 Python 累加；本机是 KVM guest、无 cpufreq，本切片不作任何性能声明。

## 2. 坐标与身份

| 项 | legacy 条目 | 新条目 |
| --- | --- | --- |
| provider id | `vendor:aocl-lpgemm-int8` | `vendor:aocl-lpgemm-int8-exact-blocked` |
| provider version | `aocl-lpgemm.blis-5.3.2.zen4.dimt64.blasint32.int8.v1` | `aocl-lpgemm.blis-5.3.2.zen4.dimt64.blasint32.int8.exact_blocked.v1` |
| implementation | `aocl.lpgemm.s8s8s32os32.packed_b_direct` | `aocl.lpgemm.s8s8s32os32.kc_exact_blocked_int32_accum` |
| numeric_class | `deterministic_bounded` | `exact` |
| reduction order id | `aocl_lpgemm_s8s8s32os32_zen4_kc2048_f32block_chain_int32_v1` | `aocl_lpgemm_s8s8s32os32_kc_exact_blocks_then_python_int_accum_v1` |
| envelope | `measured_snapshot`、`deviation_bound_available=false`、无 `deviation_bound_by_k` | `exact_contract_bound`、`deviation_bound_available=true`、`all_k_le_limit: 0` |
| ABI/artifact format | `aocl-lpgemm.ctypes.v2.int8` / `aocl.lpgemm.int8.v1` | `aocl-lpgemm.ctypes.v2.int8.exact_blocked` / `aocl.lpgemm.int8.exact_blocked.v1` |

legacy 条目的字符串、digest、envelope 与行为**全部保持逐字节不变**；新条目的
`provider_artifact_digest` 是以新身份字符串 + **完整 envelope** + library fingerprint
为 preimage 的 sha256（envelope 改一位则 digest 必变，见测试
`test_envelope_is_covered_by_the_provider_artifact_digest`）。

## 3. Kc 规则与证明骨架

### 3.1 调用语义

现有 int32 路径（`prepare_int32`）对 pinned 库的调用为：

```text
aocl_gemm_s8s8s32os32(order='r', transa='n', transb='n',
                      m, n, k, alpha=1,
                      A[M,K], lda=K, mem_format_a='n',
                      B[K,N], ldb=N, mem_format_b='n',
                      beta=0, C[M,N], ldc=N, post_op=None)
```

即该路径只有 int32 与 f32 算术，无 bf16/f16，无 epilogue/缩放，无饱和声明。新条目保持
该调用逐参数不变，只把一次 `k` 拆成 `ceil(k/Kc)` 次 `width ≤ Kc` 的调用：A 用原
`lda=k` 与原缓冲区偏移（不复制），B 用原行主序缓冲区偏移（连续 K 段），输出每段一个
独立 int32 缓冲。

### 3.2 f32 精确整数面

binary32 有 24 位有效位：任意整数 `|v| ≤ 2**24 = 16777216` 可被精确表示。因此在整个
路径上，只要某个被转换到 f32 的中间值满足该界，`int32→f32→int32` 就是恒等。

### 3.3 每段部分和界

设 `max|a|`、`max|b|` 为**实际量化码**（O(MK)+O(KN) 扫描，dispatch 时对 A、prepare 时
对 B 各算一次）的最大绝对值。对任意 `width ≤ Kc` 段、任意输出元素，任意部分和（无论
内核以什么顺序形成）满足

```text
|partial| ≤ width · max|a| · max|b| ≤ Kc · max|a| · max|b|
```

### 3.4 Kc 取值

```text
Kc = min(2048, floor(2**24 / (max|a| · max|b|)))
```

- `max|a|·max|b| == 0` ⇒ `Kc = 2048`，精确和恒为 0；
- `Kc < 1`（即乘积 > 2**24）⇒ 结构化拒绝 `exact_block_kc_rule_unsatisfiable`（fail-closed）；
- 码域为 `[-127,127]`（契约保留 `-128`，适配层在调内核前拒绝），因此实际
  `Kc ∈ [1040, 2048]`（127·127=16129 ⇒ 1040；若 `-128` 被某路径放进公式，只会以 128
  进入、把 Kc 缩到 1024，不会放松界）。

由 §3.3、§3.4，**每个段内所有会被转成 f32 的整数中间值都 ≤ 2**24**，段内所有 f32 往返
恒等 ⇒ 该段输出就是该段列的精确整数和。

### 3.5 顺序无关性（与 ERR-0010 的关键区别）

上述论证不依赖内核内部归约顺序：无论内核是"每 KC=2048 块转一次 f32"、"每 64 列转一次"
还是"整段先在 f32 里累加、最后转回 int32"，每个中间值都是**至多 Kc 列**的整数部分和，
都被 `Kc·max|a|·max|b| ≤ 2**24` 界住，所以都恒等。本切片**不是** ERR-0010 中被证伪的
"块内精确 + 块间纯 f32 链"解析模型（Q8 parity 25/31）；恰恰相反，它以"把可能经过 f32 的
每个值都压进 f32 精确整数面"来**绕开**对该链条建模的需要。测试
`test_threads_are_bitwise_identical_order_invariance` 在 threads 1/2/6 下对同一输入要求
输出逐位相同，作为顺序无关性的行为证据。

### 3.6 精确累加与范围校验

各段 int32 输出（= 各段精确整数和）用 Python 任意精度整数逐元素相加；最后对每个输出
元素作 `[-2**31, 2**31-1]` 范围校验，越界则 `exact_block_accumulation_overflow_int32`
fail-closed。对契约码域，`K ≤ 133144 = (2**31-1)//(127·127)` 保证最大精确和
`2,147,479,576 ≤ INT32_MAX`（余量 4,071）；该范围校验是防御性第二道闸，而不是范围的
唯一依据。

## 4. envelope 全文（关键字段）

`deviation_bound_by_k` / `measured_deviation_by_k` 同时列出
`all_k_le_limit: 0` 与差分矩阵 K 集合的 0：

```json
{
  "class": "exact",
  "accumulation_width_bits": 32,
  "kc_block": 2048,
  "kc_cap": 2048,
  "k_exactness_boundary": 133144,
  "bitwise_max_k": 133144,
  "k_accumulation_limit": 133144,
  "deviation_bound_kind": "exact_contract_bound",
  "deviation_bound_available": true,
  "deviation_bound_by_k": {
    "all_k_le_limit": 0,
    "1024": 0, "1040": 0, "1041": 0, "2048": 0, "2049": 0, "16696": 0, "133144": 0
  },
  "measured_deviation_by_k": { "...同上..." },
  "bound_derivation": {
    "f32_exact_integer_limit": 16777216,
    "kc_rule": "kc = min(2048, floor(2**24 / (max_abs_a * max_abs_b)))",
    "chunk_partial_bound": "abs(sum over any kc columns) <= kc * max_abs_a * max_abs_b <= 2**24",
    "exact_round_trip": "24-bit significand => int32->f32->int32 identity for |v|<=2**24",
    "order_invariance": "holds for every intermediate regardless of the kernel's reduction order",
    "accumulation": "Python arbitrary-precision ints + int32 range check",
    "chunk_count": "ceil(k / kc)"
  },
  "proof_obligation": "show every kernel f32 intermediate is an integer partial sum of at most kc columns ...",
  "residual_assumptions": [ "…见 §5…" ],
  "upper_bound": { "status": "proven", "reason": "kc_rule_..._exact_python_int_accumulation", "blocking_evidence": [] },
  "not_covered": [ "fused sym_quant epilogue", "dequantization/epilogue error", "codes outside [-127,127]", "K > limit", "other builds/architectures" ],
  "code_range": [-127, 127],
  "reserved_code": -128,
  "wrap_behavior": "…fail closed with exact_block_accumulation_overflow_int32"
}
```

（全文由 `_exact_blocked_exactness_envelope()` 生成，并被 plan/report/binding_verify 三条
既有校验路径携带与 re-derive。）

## 5. 残余假设（登记边界）与撤回路径

本切片证明的是如下条件命题，**不是**对闭源内核的形式化证明：

> **前提**：pinned 库在 `aocl_gemm_s8s8s32os32` 的该调用路径上只做 int32 与 f32 算术
> （整数乘加 + 块边界 int32↔f32 转换；`alpha=1`、`beta=0`、`post_op=None`、无
> bf16/f16、无 epilogue、无饱和）。
> **结论**：该路径上任何被转换到 f32 的值都是 ≤ Kc 列的整数部分和，绝对值 ≤ 2**24，
> 故 f32 往返恒等；分块 + 精确整数累加得到契约的精确 int32 和。

可检验项（写入 `residual_assumptions[0].checkable_items`）：

1. 调用符号为 `aocl_gemm_s8s8s32os32`（probe 在 pinned 库上检查该符号存在）；
2. 调用参数为 `order='r'`、`transa/transb='n'`、`alpha=1`、`beta=0`、`post_op=None`，
   A/B 为 row-major、C 为 int32；
3. AOCL 文档/源码把 s8s8s32os32 描述为整数 MAC + int32 累加 + 块边界 int32↔f32 转换、
   无饱和/无 epilogue（引用见 `docs/20-planning/0006` §8/§13.9 与 ERR-0010 记录的
   `lpgemm_6x64rowmajor_s8_amd512vnni.c` / `lpgemm_s32_kern_macros.h` 路径说明）。

**撤回路径**：一旦有证据表明该路径存在其它有损运算（bf16/f16 舍入、epilogue 缩放、
饱和累加等），必须立即撤回本条目的 `exact_contract_bound` envelope 与 `exact`
numeric class，并让该 provider 在 `numeric.require_proven_deviation_bound=true` 下
fail-closed；已产出的 plan/report 因 digest 覆盖与 binding_verify re-derive 会拒绝
继续使用新 envelope。撤回条目登记在 `AOCL_INT8_EXACT_BLOCKED_WITHDRAWAL`。

## 6. 选择/拒绝语义

1. **默认不可选**：新条目不在 `default_providers(kind="cpu")` 中，默认 capability 快照
   digest 与基线完全一致；resolver 在非 opt-in 策略下**跳过**该条目（不产生 candidate
   记录），因此默认 plan payload/digest/选择/输出与基线逐字节一致。
2. **opt-in 触发**（`capability.policy_opts_into_exact_blocked`）：
   `numeric.require_proven_deviation_bound=true`，或
   `provider.mode=required` 且 `provider.required=<新 id>`，或
   `fallback.mode=declared` 且 `fallback.chain` 显式含新 id。
   此时 `capability_for_policy` 才把新条目加入快照（entry layer），
   `_execute_frozen_plan` 在 capability 缺省时从冻结 policy 文档重建同一 opt-in 快照。
3. **契约文档不变**：新条目**不**加入 `OpDefinition.allowed_providers`（加入会改变
   qmatmul 的 contract digest，进而改变所有 provider 的 contract digest 与默认
   plan digest）。resolver 只对"显式 opt-in 的新条目"开放该 contract-allowed 检查旁路，
   contract digest 相等性检查仍然执行。
4. **无 proven bound 路径仍拒绝**：legacy `measured_snapshot` 条目在
   `require_proven_deviation_bound=true` 下仍以 `proven_deviation_bound_unavailable`
   结构化拒绝（无 fallback 时 `NUMERIC_GUARANTEE_UNMET`）。fused `sym_quant` 不通过
   U1 SPI 路由（`routable_via_opcode=false`），新条目 `fused_variants={}` 且 envelope
   `not_covered` 明确不含它——本切片不改变 fused 路径的 fail-closed 语义。
5. **执行期 fail-closed**：K > 133144、rank≠2、m/k/n<1、码值非法（含 `-128`、bool、
   float）、Kc 公式无正解、累加越 int32 范围，均在写输出前结构化拒绝。
6. **plan/report/binding 一致**：plan `selected`/`numeric_guarantee`、report
   `guarantees`/`resolved`、report `provider_manifest.document` 三处 envelope 必须与
   manifest re-derive 一致；`report_binding_errors` 与 provider_identity 已覆盖新 id。
   篡改 envelope 后即使重算 plan/report digest 也会被拒绝。

## 7. 与 ERR-0010 的关系

- ERR-0010 的结论（legacy `aocl_int8` envelope 是实测快照、不是上界）**不撤销**：
  legacy 条目原样保留，仍不提供解析上界。
- 本切片没有复活 ERR-0010 证伪的"块间 f32 链"模型；本文的界是**顺序无关**的
  （§3.5），且不声称能解析建模真实内核的归约顺序。
- 本切片**不覆盖** fused `sym_quant` epilogue 的 bf16 误差，也不覆盖反量化后的输出误差；
  这些仍在 legacy 条目的 `not_covered`/`deterministic_bounded` 语义内。

## 8. 成本模型与性能边界

- 内核调用次数：`ceil(K/Kc)`（K=133144、all-127 时 129 次；Kc 随 max|code| 缩小）。
- 每次调用前的 max|code| 计算：O(MK)+O(KN) 一次。
- 累加：O(M·N·ceil(K/Kc)) Python 整数加法 + 一次 int32 范围扫描。
- 无额外拷贝/重排（A 用原 `lda=K` + 缓冲偏移，B 用原连续 K 段偏移）。
- **性能 UNGATED**：本机 KVM guest 无 cpufreq，且本切片未做计时矩阵；不得把该成本模型
  写成性能结论。

## 9. 测试与证据

- 测试：`python/tests/ut/pypto_x/test_execution_qmatmul_aocl_exact_blocked.py`
  （差分矩阵、threads 1/2/6 逐位一致、反例控制、fail-closed、解析语义、零默认变化、
  envelope digest 覆盖、plan/report/binding 携带）。
- 树间零默认变化对照：
  `_meta/pypto-x/int8-exact-blocked/raw/zero_default_change_{baseline,worktree}.json`
  （脚本 `scripts/zero_default_change_matrix.py`；两树 diff 仅 tree label）。
- 差分矩阵与反例控制原始 JSON/日志、规则 8 focused/full 结果：见
  `_meta/pypto-x/int8-exact-blocked/{raw,logs}/`。
- 契约文档不替代证据：任何"观测 0"都只是回归证据，界本身来自 §3 的构造。

## 10. 已登记边界（不做/未证）

1. 残余假设（§5）不是形式化内核证明；前提被推翻时按撤回路径处理。
2. fused `sym_quant`、反量化/epilogue 后的 bf16 误差不在本条目覆盖内。
3. `-128` 仍在契约层拒绝，不进入任何内核调用；Kc 公式对 128 的处理只是防御性论证。
4. K > 133144 仍整体 fail-closed（与 legacy/portable 一致），本切片不提高该上限。
5. 仅覆盖 pinned BLIS 5.3.2 zen4 dimt64/blasint32 构建与本机记录 shape/线程窗口；
   其它构建/架构需重新验证。
6. 性能未测（UNGATED）。
