# PyPTO-X 0048–0050 波次快照：composite 组合化 · U 线算子契约两批 · U5 region 只读

状态：`CLOSED_ALL_SLICES_VERIFIED`（三批同一冻结点收口；按自动推送政策已推送）
批次：`batch_0048_composition_and_ops_registry_2026_09_13`、`batch_0049_u5_region_readonly_2026_09_13`、`batch_0050_ops_registry_batch2_2026_09_14`
撰写：parent（自动批次；中间经用户 09-14 暂停与 09-19 恢复）

## 0. 一句话

四件事：**①** 把 Qwen3.5 的 20 个 portable composite 从"平的库"改成**可组合积木**并让整图**真正复用**它们（IR/执行逐位不变）；**②** U 线 opcode 契约从 **2 → 10**（第一批 8 个高频算子 + 通用 `plan/execute`）；**③** **U5 只读 region**：把"图"提升为一等公民（冻结 region plan、digest 锚定、保守精度合成、CLI），并在三轮验收攻击后补齐锚定/边界/零输入/版本纪律，再把 region 层 scalar 预检委托给契约；**④** **opcode 契约第二批**（Tier 1 六算子 + 两个 batch-1 补丁），使真实 Qwen 图**缺失契约 19→13 类**、`unplannable` **212→0**。

## 1. 冻结点

| 项 | 值 |
|---|---|
| integration tip（frozen） | **`9264deb93`**（tree `ec93dfb19a9392cd930f787f1441f4a30800547c`） |
| 本波提交（8 个） | `3770b2280` composite r1；`4521558e6` ops-batch1；`41ae82a01` U5 只读；`b643229bd` U5 r2；`177e9a05f` composite r2；`2064ca808` U5 r3；`8b5246557` ops-batch2；`9264deb93` U5 r4 |
| 补丁数 | **283**（0047 = 275；本波 +8）；干净树 `git am` **283/283**，复算 tree = 冻结 tree |
| 父方最终 gate | **rc=0，2426 passed / 11 skipped / 0 failed，1332.39 s（22:12）**；collect **2437**（11 skip = 7 无 CUDA driver + 4 host-guarded 基线钉） |
| 公开仓 | `origin/main` 已推送（含 283 补丁、0018–0022 契约文档、本快照与 ERRATA） |

## 2. 0048 切片 A：Qwen3.5 composite 组合化

- **机制**：新增私有原语 `_compose_program`（按 CoreType 严格绑定子程序参数、按内部 SSA 值可达切片、保序拼接、最长前缀 rename、冲突报错）。
- **整图已复用 composite**（验收方独立证明）：静态 **27 个调用点 / 8 个父函数**，整图一次构建触发 **105 次组合**（6 full-attention + 18 GDR 全覆盖；基线为 0）；`where/reduce_max/div/gather` 内联 4→1、`compare` 3→1、`iota` 9→3。
- **IR 等价**：**59/59** 快照 `canonical_json()` 逐字节相同（五维结构哈希全同），整图 bf16 v3 digest `66dd4077…`（4550 ops）不变；**执行等价**：cpu-scalar/AVX2/AVX-512 各 17/17、SVE256(QEMU) 14/14 输出字节 sha256 全同、`max_abs_diff=0.0`（不用容差）。
- **round-2 三收尾**：§7 补登记整图 49 次内联 stream RMSNorm（24+24+1，含原因与后续收敛路径）；`_compose_program` 对多余 binding / 非父 builder 绑定 / 参数与输出同名 / 空或非字符串 rename / 裸字符串 returns 全部 fail-closed；新增"父→子调用"守卫（spy 精确计数 6/4/2/1/2/105），并以**单点回退实验**证明"只有新守卫能抓"（旧断言全绿）。
- **判决**：`8e32b8ac` **PASS_WITH_BOUNDARIES**（round-1 + round-2；两处文档措辞已就地订正）。
- **登记边界**：整图 GQA repeat / MLP `silu*up` / `_gdr_recurrent_inline` / `_rms_gated_core` / lm_head·in_proj matmul 仍内联（数量与原因见 0018 §7）；Python 构图成本 0.123→0.147 s（UNGATED，非运行时成本）。

## 3. 0048 切片 B：U 线 opcode 契约第一批（8 算子）

- **新增契约**：`where`、`compare`、`iota`、`exp`、`reduce_sum`、`reduce_max`、`cast`、`reshape` + 通用 `plan_operation(...)` / `execute_operation(plan, inputs)`（复用同一 resolver/policy/v4 报告）。
- **零默认变化**：`matmul`/`qmatmul` 冻结在 schema v2，默认 probe 只声明这 2 个契约；新契约走 schema v3 + **按 opcode 收窄的 capability 视图**（兼容 shim）；12 线程默认与 6 线程锁环境双对照逐字段 IDENTICAL。
- **精度声明（诚实）**：整数/逻辑/视图类 `exact`（可证明）；浮点路径 `deterministic_bounded` 且**明确"无解析上界"**，`require_proven_deviation_bound=true` 下**全部 fail-closed**（含 portable）。
- **验收**：`f2b2f61d` **PASS_WITH_BOUNDARIES**（93 项独立 oracle/位型检查 0 hard failure；capability 收窄视图伪造被拒；全量 collect 2326 / 2317 passed）。
- **登记边界**：`execute_operation` 会重跑一次 resolver 做一致性校验（不 probe、不换 provider）；值越界抛 `CpuScalarExecutionError`（无 code/reason，后续错误包装项）；`provider.required` pin 下错误码不同；rank-5 reshape reason 映射。

## 4. 0049 切片 A：U5 只读 region（`plan_region`/`explain_region`/CLI）

- **产物**：`REGION_SCHEMA_VERSION=2` 的冻结 region plan（边界签名 + 拓扑序逐 op 条目 + Merkle region digest + region 级精度声明）+ `explain_region`（复用 U3，不新造解释层）+ CLI `plan --region` / `explain --region` / `explain --region-plan`；**不执行、不引入 v5**。
- **精度合成（保守）**：全 exact→exact；全 portable_bitwise（或与 exact 混合）→portable_bitwise；bounded 仅当每项都有 proven 包络才可"求和成界"且需 `is_upper_bound` 依据；**任一无证明 → `deterministic_bounded_unproven` 且 `require_proven_deviation_bound` region 级 fail-closed**；禁止误差抵消；混 provider 必列 provider 集合。
- **三轮修复（验收方抓到的硬伤，均已修并复验）**：
  1. **F1 锚定不完整**：原 preimage 只含 `op digest + 边界 + schema`，**10 类字段可无痕改写**（最小反例：`exp` 改成 exact+proven 0 而 digest 不变仍 ACCEPTED）→ 现 preimage 覆盖**完整 op 条目**并对 `selected`/`numeric_guarantee`/`fallback`/版本/provider metadata **逐字段 replay 比对**；14 类字段参数化回归全拒（含完全自洽重签伪造）。
  2. **F9 boundary 不可用**：mid-graph entry 与"子图不从全局 op 0 开始"都曾被拒 → region-local 重编号 0..N-1（region 外 entry 记 `producer_op_index=null`）+ 返回前自检 `validate(re_resolve=False)`；两类反例现 plan→validate→load→explain 全通。
  3. **F7 iota 零输入** 计划不可回读 → 按契约 arity 校验，零输入合法并往返。
  4. **round-3 版本纪律**：preimage 语义已变 ⇒ `REGION_SCHEMA_VERSION` 1→2（v1 以 `region_schema_version_legacy` fail-closed，绝不按 v2 重解释）；重签空 operations → `region_operations_empty`。
  5. **round-4 scalar 委托**：region 层不再一刀切拒绝 scalar，改由 opcode 契约判定；scalar 以 `kind:"scalar"`+`shape:[]` 进入 operands/boundary 并**进入 digest preimage**；三图 **unplannable 212→0**。
- **判决**：`e4bc892c` round-1 FAIL → r2/r3/r4 均 **PASS_WITH_BOUNDARIES**。
- **登记边界**：feature-forged capability（同 triple 只查 kind+triple）；多分量 region 口径；默认路径保留一次 host probe（显式路径 0 probe）；path 递增约束；**`kind` 未进 replay 输入**（把 operand 与 boundary 的 kind 一起改并重签仍 ACCEPTED —— 语义等价、已登记，修法=文档写明或把 kind 纳入重放输入）；`split` 多输出须先定义多输出 ABI（batch-3 前置）。

## 5. 0050 切片 A：U 线 opcode 契约第二批（Tier 1 + 两补丁）

- **新增 6 契约**：`add`、`mul`、`rsqrt`、`transpose`、`slice`、`broadcast`；**两个严格加法补丁**：`where` 接受 scalar fill、`compare` 接受冗余 `broadcast` 声明（既有合法请求 digest 逐字节不变）。
- **视图语义冻结**：视图族 `view_obligation=may_alias`、`allows_materialize_copy=true`（copy 字节数进 plan digest）；`view_mode=forbid`→物化、`prefer`→无法证明时物化、**`require`→`ALIAS_PROOF_UNAVAILABLE`（绝不静默降级）**；`inplace`/非法 `valid_shape`/无序容器/负 step/重复越界轴/不兼容广播全拒；运行期 `output_input_alias` 兜底。
- **主验收指标**：真实 Qwen 图缺失契约 **19→13 类**，六算子（544/414/360/356/330/115 次）**全部归零**；三图 4693 ops / **4282 已登记全部 plan** / 411 refs 仍缺契约（13 类）/ 0 unplannable；精确口径：scalar 预检造成的 unplannable 曾为 **212**（add 176 + mul 29 + where 7）。
- **零默认变化**：默认与 6 线程锁环境双对照 ZERO-DRIFT（仅 `registered_opcodes` 如预期变化）。
- **判决**：`f79f1dbf` **PASS_WITH_BOUNDARIES**（575 项独立探针含 270 例 fuzz 差分；无硬 FAIL）。
- **登记偏差**：D1 张量结果越界抛裸 `CpuScalarExecutionError`（无 code/reason，建议下批包装）；D2 0022 §8 的 254 实为 **411**（已勘误）；D3/D4/D5 三处 reason 归类/可观测性细节（仍 fail-closed）。
- **登记边界**：vendor 未开启；动态 shape/多输出/非 tensor 结果拒绝；`view_mode=require` 对 portable 视图族不可满足（无 strided-view ABI）。

## 6. 验收汇总

| 切片 | agent | 判决 |
|---|---|---|
| 0048 A composite 组合化 | `8e32b8ac` | **PASS_WITH_BOUNDARIES**（r1 + r2 定点复验） |
| 0048 B opcode 契约第一批 | `f2b2f61d` | **PASS_WITH_BOUNDARIES** |
| 0049 A U5 只读 region | `e4bc892c` | **FAIL → PASS_WITH_BOUNDARIES**（r2/r3/r4 三轮定点复验） |
| 0050 opcode 契约第二批 | `f79f1dbf` | **PASS_WITH_BOUNDARIES**（会话曾被用户暂停，恢复后完成） |

## 7. 证据路径

- 实现/证据：`_meta/pypto-x/{qwen35-composite-composition,u-ops-registry-batch1,u5-region-readonly,u-ops-registry-batch2}/`
- 验收：`_meta/pypto-x/{verify-0048-composite-composition,verify-0048-ops-registry,verify-0049-u5-region-r3,verify-0049-u5-region-r4,verify-0050-ops-batch2}/`
- 批级：`_meta/pypto-x/batch-0048-0050-closure/`（最终 gate、283 补丁重导与 `am` 复算）
- 补丁：`patches/pypto-x/`（**283**）+ `patches/README.md`（HEAD `9264deb93`）
- 契约文档：`0018`（composite）、`0019`（opcode batch1）、`0020`（U5 提案）、`0021`（U5 只读）、`0022`（opcode batch2）

## 8. 登记边界与口径

- **性能**：本波所有数字 **UNGATED**（12 vCPU KVM；含 Python 构图成本）。
- **文档计数两处勘误**（由独立验收发现并订正）：0022 §8 的"其他 13 类 254"→**411**；0021 的"205→0"→精确 **212→0**（见 ERR-0015）。
- **未覆盖面**：region 执行与 v5 报告（U5 step-2/3）、多输出 ABI（`split`）、动态 shape、vendor 路径、`view_mode=require` 正向满足。

## 9. 伴生：CUDA 计时腿完成（0046 遗留项，2026-09-19 补跑）

- 空闲窗口（时钟门 **VALID**（median_sm 2835 / max_sm 3090 MHz，ratio 0.9175），前后指纹无外来 app）：`status=PASS`、`ALL_PATHS_BIT_EXACT`。
- **结论：native sm_120 cubin 相对 PTX-JIT 无可测运行期优势**——f32 4096³ 三路径 median 0.059402/0.059175/**0.059156 s**（极差 0.42%）、bf16 0.039095/0.039037/**0.038970 s**（0.32%）；首次加载 0.11–0.46 ms、launch median 6 µs。数字仍 **UNGATED**。证据 `_meta/pypto-x/cuda-sm120/compare/{PERF_DONE.md,PERF_SUMMARY.json,sm120-compare.json}`。
