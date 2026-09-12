# PyPTO-X 0042 波次快照：report 单文档校验补强、U2b int8/W8A8 vendor 绑定、AVX2 broadcast 原生平面

状态：`CLOSED_LOCALLY_PENDING_THE_USER_PUSH_APPROVAL`
批次：`batch_0042_accumulating_2026_09_12`（见 `configs/development_lock.yaml`）
撰写：parent（自动批次）

## 0. 一句话

0042 由三个切片构成：**report 单文档校验补强**（项 1：4 类 cross-block + 2 类 contract-derived；项 1b：按 rule 名对账全部具名 `resolution.rules[*].details` 回显；项 1c：验收方穷举出的 R1–R10 运行时/文档派生回显 + 诚实边界声明）、**U2b**（int8/W8A8 的 AOCL vendor 绑定，新契约 `qmatmul_s8s8_s32`；包络经独立验收推翻后改为**观测快照 + fail-closed 门**，并修掉每 forward 入口开销）与 **AVX2 broadcast 原生平面**（Qwen3.5 AVX2 路径最后一处大 host_reference 面，decode 350/prefill 494 全部转 native）。

## 1. 冻结点

| 项 | 值 |
|---|---|
| 控制仓 | 收口提交见 §10（含 lock 记录、本快照、ERR-0009/0010、0006 §7.1/§8.2） |
| integration tip（frozen） | **`0376e5307`**（tree `8f705bb0d7cd4df145dc4bde92ddf3f92cb8037b`） |
| 任务分支 commit | 项 1 `748717529`（→ `15d0fe58d`）；项 1b `4ddf84e1c`（→ `53d3deab9`）；项 1b2 `7f5b26675`（→ `e860f69c6`）；项 1c `c2af5bb1d`+`512b2e5ff`（→ `134d1dd21`+`b89e0b521`）；U2b 初版 `7b77284e9`（→ `1d23a597f`）+ round-3 的 8 个提交（→ `0376e5307` 的末 8 个提交）；AVX2 `b37e2d58e`（→ `27a0cb477`） |
| 补丁数 | **250**（0041 = 235）；am 复算：250 个全部干净应用，复算 tree = `8f705bb0d…` = 冻结 tree |
| 规则 8 全量 | **rc=0，1838 passed / 7 skipped / 0 failed，943.10 s**；collect 1845（基线 `8fff50558` = 1674） |
| 备份同步 | 待 push 批准后执行 `scripts/remote/sync_private_backup.sh` |

## 2. 项 1：report cross-block（`748717529` → `15d0fe58d`）

**新增 must-reject**（`binding_verify.py::report_binding_errors(report, definition)`，由 `validate_report_schema` 传入注册 `OpDefinition`）：

| 检查 | 拒绝码 |
|---|---|
| `resolution.capability_digest == capability.snapshot_digest` | `report_resolution_capability_digest_mismatch` |
| `resolution.contract_digest == resolved.contract.digest` | `report_resolution_contract_digest_mismatch` |
| `dispatch.provider_declared == resolution.fallback_chain[0]`；`resolved.provider == dispatch.provider_invoked`；无 fallback 时三者相同 | `report_provider_declaration_mismatch` |
| `dispatch.fallback_used == resolution.fallback_used` | `report_dispatch_fallback_used_mismatch` |
| `guarantees.accumulation` 由注册契约表 + `request.dtype` 重算 | `report_accumulation_not_reconciled` |
| `guarantees.exactness_basis` 由 provider family 重算 | `report_exactness_basis_not_reconciled` |

**父方独立复跑（cherry-pick 后，`15d0fe58d`）**：聚焦 11 文件 **352 passed / 0 failed（169.59 s）**；验收方 V2–V5 套件 **58 passed / 0 failed（18.57 s）**；collect **1681**（0041 tip 1674，+7）。
证据：`_meta/pypto-x/view-mode-require/raw/focused_after_cherrypick_15d0fe58d.log`、`verifier_v2345_on_cherrypick_15d0fe58d.log`、`collect_only_on_cherrypick_15d0fe58d.log`、`_meta/pypto-x/view-mode-require/raw/RUN_NOTES_0042_report_cross_block.md`。
Cherry-pick 树一致：`15d0fe58d^{tree} == 748717529^{tree} == c54671ffd83778f8373a04c80d7291f2f72d6f3c`，patch-id `281eb0a9…`。

## 3. 项 1b：回显对账（独立验收驱动，TODO commit）

**缺口来源**：独立验收方（agent `cb32c1ba`）的穷举分类 + 父方两个探针：
- `_meta/pypto-x/view-mode-require/scripts/probe_request_block_hole.py`：`request.dtype`（配合一致 `guarantees.accumulation`）、`request.left_shape`、`request.op`、`resolution.rules[0].details.contract_digest` 单点篡改**曾被接受**；
- `probe_rule_echo_holes.py`：19 个 `resolution.rules[*].details` 回显篡改中 **18 个曾被接受**（仅 `provider.select.primary` 被拒）。
- 验收方穷举：`_meta/pypto-x/verify-0042-report-cross-block/raw/08_exhaustive_classes.log`、`13_ledger.log`（449 叶，335 曾被接受）。

**修复**：`binding_verify.py::report_rule_echo_errors`（按 **rule 名**而非下标逐条对账；缺失/重复/未知 rule fail-closed；位置型 `resolution.candidates[*].checks[...]` 明确声明为契约外说明层）。任务提交 `4ddf84e1c` → cherry-pick = **`53d3deab9`**（+24 条测试，collect 1705）。修复后父方探针：request 4/4 拒绝（`report_rule_echo_mismatch`/`report_request_opcode_mismatch`）、echo **0/19 接受**、真报告仍通过；父方合并探针（28 例）在 T1 = **27 拒 / 1 残留**（残留为位置型候选 rank 回显，交 1b2）。聚焦套件在 1b 树上 **376 passed**，验收方 V2–V5 **58 passed**。
证据：`_meta/pypto-x/view-mode-require/raw/{probe_rule_echo_holes_1b,probe_request_block_hole_1b,focused_after_1b,verifier_suites_after_1b}.log`、`probe_1b_acceptance_on_T1.json`、`collect_only_T1_53d3deab9.log`。

## 4. 项 2：U2b int8/W8A8 vendor 绑定（TODO commit）

范围：契约 `qmatmul_s8s8_s32`（int8×int8→int32，rank 2–4，K≤133144，码值 [-127,127]，packed `[K,out]` 直供 LPGEMM `mem_format_b='n'`，零 reorder/搬运）；provider `vendor:aocl-lpgemm-int8`（同一 Q8 固定库 sha256 `c7d74a31…`/zen4）。
- **位级包络**（1008 例：seeds 1–8 × {uniform[100,127] A/B 独立、all-127、uniform[-127,127]} × threads{1,2,6}，m=2,n=6）：K≤1024 全分布/全线程 **0**（bitwise_max_k=1024）；2048→1、3584→4、5099→8、16696→56、133144→457；声明值 = max(Q8 参考, 实测)，`q8_reference_bound_by_k`/`measured_bound_by_k`/`measurement_scope` 均进 manifest 并 hash 进 `provider_artifact_digest`。
- **fail-closed**：K≥133145 在任何 kernel 调用前拒绝（raw LPGEMM 会静默回绕，返回 −2147471616）；`-128` 适配层拒绝且输出逐元素不变；库缺失/sha 不符、线程漂移、重算 digest 的伪造 plan/report 共 9/9 PASS。
- **数值分类** `deterministic_bounded`；fused `sym_quant` 声明+实测但**未路由**（2-operand SPI 边界）；另发现 AOCL fused n=1 不写输出 → `min_n=2` fail-closed。
- **性能**（UNGATED）：同存储同 kernel 对比 Q8 `aocl_int8_unpacked` geomean **1.16×**（12 格）/ **1.71×**（剔除 Q8 自标注污染格）；fused 对 Q8 symquant 1.01×/1.59×。
- **跨任务交互 PASS**：portable 与 vendor 两条 qmatmul 真报告均通过 `validate_report_schema`（含 1b/1b2 的 named-rule echo 对账）；期间修复一处 1b misfire（`op_contract.verify.result_dtype` 多余回显键被拒 → 删除该键）。
- **ERRATA**：`ERR-0010`——Q8 §8.2 的 int8 包络不是上界（K/分布依赖；`133144→22` 被 `452/457` 取代，且与其自身 E3 all-127 观测 24 矛盾）；控制仓 `78ebe8c` 同时修订 0006 §8.2 为"实测包络、非上界证明"。
- 边界：fused 未路由（需新契约+SPI 扩展）；`auto_static` 无 cost model → vendor 需显式 opt-in；ErrorBudget 仅透传；包络非证明；vendor 仅 rank2（rank3/4/空轴回落 portable）。
证据：`_meta/pypto-x/u2b-vendor-int8/{raw/summary.md,raw/bit-envelope.json,raw/perf-matrix.json,raw/fail-closed.json,raw/fused-variant.json,logs/*}`。

## 5. 验收（独立）

| 项 | 验收 agent | 冻结 tip | 判决 | 关键数字 |
|---|---|---|---|---|
| 项 1 | `cb32c1ba` | `15d0fe58d` | **PASS_WITH_BOUNDARIES** | 6/6 声明条件 VERIFIED；无假阳性（portable + 真 AOCL vendor bf16/f32 + 真 fallback 均通过）；聚焦 352 passed/181.96 s；穷举 41 类 39 类有未对账成员，单叶 accepted portable 335/449、vendor-f32 347/536、fallback 404/518；rule-echo 62 叶中 61 曾被接受 |
| 项 1b/1b2 | `cb32c1ba` | `e860f69c6` | **PASS_WITH_BOUNDARIES** | rule-echo 60/62 叶被拒；单叶 accepted 降到 portable 110/449、vendor 126/536、fallback 174/518；4/4 真报告仍通过；聚焦 386 passed；新列出 R1–R9 与三处文档措辞问题 |
| 项 1c | 待 `cb32c1ba` 在 T6 复检 | `0376e5305`→`0376e5307` | 待定 | 父方 v2 探针：非基线 **37/37 全拒**（28 cross-block/echo + R1–R9），真 portable 与真 fallback 通过，2 登记边界通过 |
| 项 2（U2b） | `71599188` | `1d23a597f` | **FAIL（仅包络项）** | claim 1/3/4/5/6/7 VERIFIED；全量 UT 1744 collected = 1737 passed/7 skipped/0 failed/958.30 s；包络被其自有生成器扩 seed 推翻（seeds 1–1000 → 966 > 声明 457，中位数 458 已超；K=16696 → 75 > 56）；同窗口性能 0.932×（非加速）；新发现入口开销 20.7 ms–5.06 s vs op 0.29–2.61 ms |
| 项 2 修复轮 | 待 `71599188` 在 T6 复检 | `0376e5307` | 待定 | 包络改 `measured_snapshot` + `require_proven_deviation_bound` fail-closed 门 + 支配不变式；性能改同窗口 0.919×；入口开销 5244→152.8 ms（1×1024×8192） |
| AVX2 broadcast | `d119bbce` | `b89e0b521` | **PASS_WITH_BOUNDARIES** | 自建复现：census decode 350/350、prefill 494/494 native（真跑分派 844/844 native、0 reference）；86 例逐位矩阵 + **67,275 例 ctypes kernel fuzz** 全 0 mismatch（含 rank 0–16、23 个 Qwen 契约、f32/bf16 特殊位型）；共享谓词 identity True、9 种整型/bool 与 rank 17 走 host_reference 带显式 reason；payload 6/marker :7 与旧产物拒绝、缓存键改变且 cache miss；全量 UT 1824 passed/7 skipped/0 failed/964.64 s；perf 自测 decode 22.130→13.022 s、prefill 106.919→69.664 s，51 个整图输出 digest 一致。边界：view-kind 输入不可达、标量非有限值不可注入、不可表达维度在 codegen 前即拒（fallback reason 属防御性）、无真实权重 s3/L4 与 T=18 |

**执行边界（项 1 验收已证）**：伪造 plan 在 plan 层 `validate` PASS，但 `execute_matmul_plan` 报 `plan_declaration_mismatch`/`artifact_declaration_mismatch` 且**输出未动**；`validate_report_schema` 只在 `ExecutionReport.__init__`（执行之后）调用，执行路径不消费 report。⚠️ caveat：若下游从 report 重新规划，`request.op/shapes/dtype` 会变成执行相关。
验收证据：`_meta/pypto-x/verify-0042-report-cross-block/raw/18_FINAL_LEDGER.md`、`_meta/pypto-x/verify-0042-report-1b2/raw/18_FINAL_LEDGER_1b2.md`、`_meta/pypto-x/verify-0042-u2b/raw/verification-summary.json`、`12_focused_suite.*`、`11_execution_boundary.*`。

## 6. 规则 8 全量

- 冻结 tip：**`0376e5307`**；命令：`batch-0042-rule8/scripts/run_locked_retry.sh`（内部 `run_local_heavy.sh`，75/69 重试 300 s，`setsid`，从未绕过锁）。
- 结果：**rc=0，1838 passed / 7 skipped / 0 failed，943.10 s**（attempt 1）；collect **1845**；基线 `8fff50558` = **1674**；added 171 / removed 0（无收集丢失）；7 个 skip 全为 `test_cuda_qwen_c2.py`（本机无 CUDA 驱动）。
- 首跑曾 1 failed（`test_op_bench_framework.py::test_repeat_run_structure_and_median_within_dispersion`，`median_delta=0.004295 s = 11.2%` 略超该单测自带的 10% 噪声容忍，发生在 AVX2 验收方并发持锁重活时）；安静窗口重跑干净。失败证据保留在 `raw/failed-run-T6-attempt2.log`。

## 7. 已知边界（带进下一批）

1. **report 单文档不可自证类**（协调改写**全部副本**仍被接受）：不嵌 manifest 时的 `artifact.library.sha256`、`pack.strategy`（与 `timing.timing_basis.pack_semantics`）、`numeric_class`、`deterministic`、`reduction_order`；`artifact.pack.included_in_timing`；`resolution.plan_digest`、`artifact.content_digest`；以及单副本外部真源（`timing.*`、`resources.parallelism.*`、`gates.*`、`resolution.why`/`tie_break` 叙述等）。**说明层**（位置型 `candidates[*].checks[*]` 明细、`why`、counts/flags、`*_reconciled`、gates、timing 文本）已明确声明为非规范。
2. 可机检缓解：执行授权只以 `execute_matmul_plan` 的 canonical 重算为准（验收已证，含协调改写 envelope+basis 的伪造 plan 仍被拒）；report 校验在 `ExecutionReport.__init__` 内、执行**之后**运行，仓库内无执行路径消费 report；**下游不得据 report 重新规划**（否则 `request.op/shapes/dtype` 会变执行相关）。
3. 若要 report 单文档自证 → 需 **report schema v2**：内嵌 selected provider manifest + canonical decision digest（父方列为后续候选）。
4. **int8 包络**是**实测快照**（`deviation_bound_kind="measured_snapshot"`、`deviation_bound_available=false`、`upper_bound.status="not_provided"`），不是上界；要变成可证明上界需 kernel 形式化或自有精确 int32 累加恢复；`require_proven_deviation_bound=true` 时 vendor 路径 fail-closed（portable 的 `exact_contract_bound` 仍可用）。
5. **int8 性能**：同窗口对逐字节相同的 unpacked 基线 **0.919–0.932×**（持平略慢，不声称加速）；旧的 1.16×/1.71× 是跨窗口相除，已标 `cross_window_comparability=not_comparable`。"0 reorder" 是**适配器层**——LPGEMM 内部仍把传入 row-major B 打包（UNPACKED→PACK），同存储基线即那次 unpacked 调用本身。
6. **int8 覆盖**：vendor 仅 rank 2（rank 3/4 与空轴回落 portable）；fused `sym_quant` 已声明+实测但**未路由**（需新 fused 契约 + SPI 扩展；n=1 不写输出 → `min_n=2`，`K%4` 约束）；`auto_static` 无 cost model → vendor 需显式 opt-in；ErrorBudget 仅透传。
7. **AVX2 broadcast**：kernel 是 scalar odometer（非 SIMD 向量化扩张，不称吞吐）；rank > 16 与非 f32/bf16 仍走 host_reference（且 artifact 记录理由）；无真实权重 s3/L4 数值验收；perf 一律 UNGATED（12 vCPU KVM、无 cpufreq、共享窗口）。
8. **CUDA 线**：本机无 GPU/nvcc（`cuda_c1_acceptance_smoke: BLOCKED_TOOLCHAIN_nvcc_absent`），既定顺序 `avx512_and_sve256_first_cuda_later`。

## 8. 在途与排队

- 在途（截至收口）：`cb32c1ba` 在 T6 复检 1c；`71599188` 在 T6 复检 U2b claim-2 与两项副发现。**AVX2 broadcast 已完成**：`d119bbce` @ `b89e0b521` 判 **PASS_WITH_BOUNDARIES**（其代码在 T6 未变）。
- 排队（0043+ 候选）：U3（doctor/explain/plan）、U4（35 env 迁移）、U5（graph/region，等 N4）、report schema v2（manifest 内嵌）、fused `sym_quant` 契约决策、`auto_static` vendor 语义决策、memo 命中 ELF 加固、C8 L5/L6（等用户三句话）、int8 可证明包络（形式化或精确 int32 恢复）。

## 9. 待用户决策（截至收口）

L4 decode 口径（最靠前，决策包 `0011`，推荐 A+B）｜B4 阈值冻结｜L5/L6 语料与阈值（`0005 §10` 三句话）｜W8A8 之外的新量化方案｜W8J 注入门政策｜A3 chip7/NPU 放行。**0042 新增两项**：fused `sym_quant` 是否立项新契约；`auto_static` 是否引入"性能优先/允许 vendor"语义。

## 10. 证据路径

- 任务开发证据：`_meta/pypto-x/view-mode-require/`（项 1/1b/1b2/1c；含 `probe_report_acceptance.py` 与各 tip 的 json/log）、`_meta/pypto-x/u2b-vendor-int8/`（项 2 全部 raw/logs/scripts）、`_meta/pypto-x/avx2-broadcast-native/`（AVX2，含 `raw/NOTES.md` 与 perf）。
- 独立验收：`_meta/pypto-x/verify-0042-report-cross-block/`、`verify-0042-report-1b2/`、`verify-0042-u2b/`、`verify-0042-u2b-r2/`、`verify-0042-avx2-broadcast/`。
- 规则 8：`_meta/pypto-x/batch-0042-rule8/`（`raw/summary.txt`、`raw/full-pytest.out`、`raw/collect-tip.out`、`raw/collect-8fff50558.out`、`logs/resource-attempts.log`、`raw/failed-run-T6-attempt2.log`、`CLOSURE_CHECKLIST.md`）。
- 补丁：`patches/pypto-x/`（250）+ `patches/README.md`（基线 `34475e0d8`、HEAD `0376e5307`、417 文件、9.5 M）。
