# PyPTO-X 0043 波次快照：report schema v3（内嵌 provider manifest 与 resolution decision）

状态：`CLOSED_LOCALLY_PENDING_THE_USER_PUSH_APPROVAL`
批次：`batch_0043_report_schema_v3_2026_09_12`（见 `configs/development_lock.yaml`）
撰写：parent（自动批次）

## 0. 一句话

把 `ExecutionReport` 从"依赖文档内多副本互证"升级为**自包含**：内嵌本次实际执行的 provider manifest（`ProviderCapability.to_dict()`）与 plan 冻结的 `resolution_decision`，校验**只从内嵌 body 重算 digest** 并与注册身份锚对账，从而把 0042 遗留的一整类"全副本协调改写即通过"的 manifest-only 边界收成可对账字段；`REPORT_SCHEMA_VERSION` 2→3，旧版本 fail-closed 拒绝。

## 1. 冻结点

| 项 | 值 |
|---|---|
| integration tip（frozen） | **`0a63ed4dd`**（tree `b9803c1d63b66e7416560618b2392c00896b99eb`） |
| 任务分支 commit | `d54601eff`（round 1）→ cherry-pick `086441d49`；`4b6206f480`（round 2）→ cherry-pick `0a63ed4dd`（树与 patch-id 均一致） |
| 补丁数 | **254**（0042 = 252）；am 复算：254 个全部干净应用，复算 tree = 冻结 tree |
| 规则 8 全量 | **rc=0，1914 passed / 7 skipped / 0 failed，968.82 s**（attempt 1）；collect **1921**；基线 `94d1ab162` = 1861（+60，全部来自本批） |
| 控制仓 | 见 §6（lock 记录、本快照、HANDOFF、0006 §3.2/§7.1） |

## 2. 交付内容

- **版本语义**：`REPORT_SCHEMA_VERSION` 2→3，只接受 3；`<3` → `report_schema_version_legacy_rejected`（不升级、不改写），`>3` → `report_schema_version_mismatch`；缺字段/类型错各有具名码。
- **两个新顶层块**：`provider_manifest = {provider_id, digest, document, snapshot_digest}`（document = invoked provider 的 `ProviderCapability.to_dict()`）；`resolution_decision = {digest, document}`（plan 冻结的 declared/selected 决策；round-2 增必需字段 `selected_threads`/`selected_thread_mode`）。digest 均为 canonical JSON 的 sha256。
- **无 probe 校验**：只从内嵌 body 重算 digest + 注册身份常量；不 probe、不重跑 resolver（验收方用投毒 probe/库/fs 调用复现：毒函数调用数为 0，AOCL 报告无需本机 `.so`）。
- **新增 probe-free 身份模块** `python/pypto/execution/provider_identity.py`；`provider_aocl`/`provider_aocl_int8` 抽出共享 builder，输出与旧实现逐字节一致（验收方用 `git archive 94d1ab162` 对比确认）。
- **转为可对账的字段**（原 0042 登记为不可约边界）：`artifact`/`resolved.library` 指纹（含 AOCL pin：sha256、BLIS tag/commit、zen4、dim_t64/blas_int32、build options、symbol 列表；portable 必须 `{}`）、`artifact.pack.strategy`、`artifact.pack.included_in_timing`、`timing.timing_basis.pack_semantics`、`numeric_class`、`deterministic`、`reduction_order`/`_id`、`resolved.implementation`、provider/artifact 版本常量、vendor `artifact.provider_artifact_digest`、单副本 `resolution.plan_digest`。
- **round-2 收口**：R1 `exactness_basis_detail`（`report_exactness_basis_detail_not_reconciled`）、R2 `provider_accumulation`（`report_provider_accumulation_not_reconciled`）、**R3 报告两副本 vs 内嵌 manifest 的 `exactness_envelope`（`report_exactness_envelope_not_reconciled`，修掉"报告与自身内嵌 manifest 自相矛盾"）**、R4 局部（四份线程事实一致 → `report_provider_manifest_thread_facts_mismatch`）、R5 局部（decision 的 `selected_threads/mode` 与 `resolved_policy` + 候选 echo 对账 → `report_resolution_decision_not_reconciled`）。
- **语义澄清**：`guarantees`/`resolved.{exactness_envelope, exactness_basis_detail, provider_accumulation, reduction_order, deterministic, numeric_class}` 一律取 **invoked**（executed-path）值；int8→portable runtime fallback 报告因此携带 portable 的 `exact_contract_bound`。

## 3. 验收（独立）

| 项 | agent | 冻结 tip | 判决 | 关键数字 |
|---|---|---|---|---|
| schema v3（round 1） | `f0258e32` | `086441d49` | **PASS_WITH_BOUNDARIES** | 枚举 claim 全 verified（含 28 份真实冻结 v2 报告 28/28 legacy-rejected、两块与 digest 独立重推、无 probe 强于声明、59 例协调矩阵 47 拒/12 已登记边界、锚中和 31→22 flip、builder 逐字节一致、7/7 真报告通过）；**但边界清单不完备**（R1–R5 存活）→ 触发 round-2 |
| schema v3（round 2） | 实现方自审 + 父方复核 | `0a63ed4dd` | 待独立复检（可选） | 父方：无误伤（portable + 真 AOCL bf16/f32 + 真 runtime fallback）；collect **1921**；v3 文件 **60 passed**；qmatmul **59 passed**；合并探针 **38/38 全拒、0 接受**（含 `plan_digest`/`pack.included_in_timing` 两个原边界） |

- 回归：round-1 全量 **1905 passed / 7 skipped / 0 failed / 1030.63 s**；round-2（冻结 tip）全量 **1914 passed / 7 skipped / 0 failed / 968.82 s**；7 skip 全为无 CUDA 驱动。
- 证据：`_meta/pypto-x/report-schema-v2/`（实现方 brief/raw/logs）、`_meta/pypto-x/verify-0043-report-v3/`（独立验收 ledger + 脚本）、`_meta/pypto-x/batch-0043-report-v3/`（批级全量）。

## 4. 仍为边界（已逐条登记最小反例，0006 §7.1 / brief §5）

1. **运行态/实测事实**：`timing.*`、`artifact.content_digest`、`dispatch.threads_used` 具体值、coverage 计数、`gates.*` 文本。
2. **未 pin 的 probe 事实**：host features、target snapshot digest、AOCL fingerprint 的 `path`/`name`/`size_bytes`/`default_threads`/BLIS 版本后缀，以及**线程模型的具体数值**（四事实一致性已强制，但全量协调改写仍可通过）。
3. **plan payload 未内嵌**：协调改写 `resolution.plan_digest`（或 decision body）并重算 digest 仍被接受（单副本/只改 digest 被拒）。
4. **整份报告替换为另一注册 provider 的完整自洽故事**（无签名/attestation）。
5. **无外部真源字段的全副本协调改写**：`request.dtype/shape` 等。
6. **说明层**：位置型 `checks`/`rejections` 明细、`why`、coverage counts/flags、非 selected candidate 的版本/numeric/deterministic echo、`fallback_events[*]` payload、线程环境账目。

`validate_report_schema` **仍不是执行授权**：它在 `ExecutionReport.__init__`（执行之后）运行，仓内无执行路径消费 report，执行授权只认 `execute_matmul_plan` 的 canonical 重算；下游不得据 report 重新规划。

## 5. 排队（0044+ 候选）

U3（doctor/explain/plan）、U4（35 env 迁移）、U5（graph/region，等 N4）、memo 命中 ELF stat/re-hash 加固、C8 L5/L6（等用户三句话）、fused `sym_quant` 契约、`auto_static` vendor 语义、int8 可证明包络（形式化或精确 int32 恢复）、plan payload/签名（若要收 §4 的 3/4 类）。

## 6. 证据路径

- 实现方：`_meta/pypto-x/report-schema-v2/{brief.zh-CN.md,raw,logs,scripts}`。
- 独立验收：`_meta/pypto-x/verify-0043-report-v3/{raw/18_FINAL_LEDGER.md,raw/13_ledger.json,scripts,logs}`。
- 批级全量：`_meta/pypto-x/batch-0043-report-v3/{raw,logs}`（`full-pytest.out`/`summary.txt`/`collect-tip.out`/`collect-94d1ab162.out`）。
- 补丁：`patches/pypto-x/`（254）+ `patches/README.md`（HEAD `0a63ed4dd`）。
