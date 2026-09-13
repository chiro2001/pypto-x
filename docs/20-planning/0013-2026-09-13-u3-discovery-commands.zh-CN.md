# U3 发现命令契约：doctor / explain / plan + capability snapshot

文档编号：`0013`

日期：2026-09-13（Asia/Shanghai）

状态：`IMPLEMENTED_PENDING_BATCH_VERIFICATION`

用途：冻结 U3（`docs/20-planning/0008-...` §7 任务阶梯）三条只读发现命令的命令契约、输出
schema、fail-closed 边界与未做项。设计主体见 `docs/10-architecture/0006-...`（§3.3
capability、§5 解析、§8 报告与可发现性、§13.5 单一真源）。

## 1. 三条命令

| 命令 | 回答 | 是否执行 provider | 输入 |
|---|---|---|---|
| `doctor` | 这台机器/这份快照有什么事实、哪些 fail-closed 条件成立 | 否（仅探测已 pin 的产物） | 无；可选导入的 capability snapshot；`--strict` 见 §3.5 |
| `explain` | 为什么选了某个 provider、每个候选为何被拒 | 否 | 请求（op/shape/dtype/policy/snapshot）或 plan/report 文档 |
| `plan` | 给定请求的冻结 plan（含 digest）与解析所用 snapshot | 否 | 请求 + 可选 policy/snapshot |

入口：`python -m pypto.execution.cli <command> [--json]`（`doctor` 另有 `--strict`），或
`python3 scripts/perf/execution_cli.py <command>`（wrapper 会设置
`PYPTO_X_PORTABLE_ONLY=1` 并把 `python/` 放进 `sys.path`）。Python API 从
`pypto.execution` 导出：`doctor` / `explain` / `plan_request` /
`doctor_document` / `explain_document` / `plan_document`（纯 mapping 便捷入口）/
`export_capability_snapshot` / `import_capability_snapshot` 及 `DoctorReport` /
`Explanation` / `PlanDocument`。

所有命令只读：不执行 op、不写执行状态；`plan --out` 只写用户显式指定的 plan JSON。

## 2. 输出 schema

三个文档都以 `DISCOVERY_SCHEMA_VERSION = 1` 为 schema，`kind` 分别为
`doctor` / `explain` / `plan`，并带 `sha256:` canonical digest（digest 输入不含
自身字段）。

### 2.1 `doctor`

```text
{schema_version, kind: "doctor", source: local_probe|imported_snapshot,
 host: {triple, machine, platform, features, python, byteorder},
 opcodes: [{opcode, contract_version, contract_digest, registry_schema_version}],
 capability: {schema_version, kind, triple, features, snapshot_digest, provider_ids},
 providers: [ProviderCapability.to_dict() + {manifest_digest
             [, manifest_canonical_digest]}],
 pinned_libraries: [{provider_id, library_name, path, exists, size_bytes,
                     sha256_expected, sha256_actual, sha256_matches,
                     blis_version_pin, blis_version_actual, arch_pin, arch_actual,
                     available, reason}],
 environment: {schema_version, source: "pypto.execution.env_registry",
               variables: [{name, category, status, value_kind, interpreted_value, ...}],
               unregistered: [{name, risk, status, policy}],
               allowlist: [...], raw_values_included: false,
               deprecation_warnings_enabled, deprecation_warning_switch},
 env_registry: {schema_version, migration_var_count, extra_var_count,
                runtime_var_count, control_var_count, unregistered_count, new_env_policy},
 fail_closed_checks: [{check, status: ok|failed, reason, details}],
 fail_closed: bool, read_only: true, probe_scope: "pinned artifacts only",
 path_policy: "basename"|"full", digest}
```

- `environment` 由 `pypto.execution.env_registry` 驱动（U4，40 行 = 35 迁移 + 2 extras +
  2 线程 + 1 控制开关）；**不输出原始值**，只输出 `set/unset` 与类型化解释
  （flag / integer / non_integer）；未登记 `PYPTO_X_*` 只列名称/风险（敏感名脱敏）。
- `pinned_libraries` 来自 provider manifest 内嵌的已 pin 事实；本地 probe 只探测已 pin
  的 AOCL 路径，不探测未 pin 的库。
- **路径隐私**：`provider.library.path` 与 `pinned_libraries[].path` 默认只给 basename
  （sha256/arch 仍在各自字段），`--show-paths` / `show_paths=True` 才输出完整 host 绝对
  路径；`path_policy` 记录本次输出采用哪种口径，并进入 `digest`。
- **内层 digest 可自证（round-3）**：`providers[].manifest_digest` 定义在**实际输出**的
  provider 行上（`manifest_digest == sha256(canon(row 去掉 manifest_digest))`），因此默认
  脱敏输出也可从行内复算；同一行额外携带 `manifest_canonical_digest`（未脱敏完整
  manifest 的 digest，默认脱敏时才有），`--show-paths` 时两者相等。
- `fail_closed_checks` 有 **8 个不同 check name**（capability digest 自检、契约 digest
  自检、已登记 opcode 覆盖、provider 与 registered identity 一致、provider artifact
  digest 可复算、pinned library 存在/sha256/arch/符号/ABI、payload/ABI 版本登记、schema
  版本）；provider identity / payload 行按 provider 展开，所以行数 ≥ 8（本机 11 行）。
- **注册 manifest 不可清空（round-3）**：provider identity 检查除字段子集外，还把
  registered provider 条目对照本地 canonical manifest 的集合形状校验——规范上非空的
  `contracts/dtypes/numeric_modes/required_features/preconditions/threads/pack/library`
  必须保持非空、静态必需键必须存在、规范为空的集合不得新增条目（探针相关可选键不强制，
  数值不 pin 本机）。doctor 收到这类对象时报 `provider_identity` failed、
  `fail_closed=true`（默认 rc 0、`--strict` rc 3）；CLI `doctor --snapshot` 的文件导入
  路径按 §2.4 直接结构化拒绝（rc 1）。
- 自洽快照里出现**未注册 opcode**（例如 portable 里多出 `add`）或未知 provider identity
  时，doctor 把它作为 `fail_closed=true` 的事实列出（默认 rc 0、`--strict` rc 3），不得抛
  异常；导入本身仍是结构校验（未注册 opcode 结构合法即可导入）。
- `fail_closed=true` 表示存在失败条件（例如库缺失）。doctor 默认是事实报告：退出码 0，
  条件在 `fail_closed_checks` 与人类摘要中显式列出（避免 CI 在缺少非必需 pin 库的机器上
  误报）；加 `--strict` 时 `fail_closed=true` 退出码为 3。导入快照非法等**请求级**错误
  始终结构化报错（退出码 1）。详见 §3.5。

### 2.2 `explain`

```text
{schema_version, kind: "explain", input_kind: request|plan|report,
 request, policy_applicability, requested_policy, resolved_policy,
 selected, numeric_guarantee, guarantees（报告形视图）, layout_decision,
 candidates: [Candidate.to_dict()（含 state/reason/scope/checks）],
 rejections: [resolver rejection records],
 tie_break, numeric_class_order, fallback, rules, why,
 boundaries: [{field, layer, why}], resolver_version,
 plan_digest, capability_digest, contract_digest,
 execution_facts?: {...仅 report 输入...}, digest}
```

- `candidates` / `rejections` / `rules` / `tie_break` / `fallback` 直接来自 resolver
  trace（`RESOLVER_VERSION=2`），不重跑选择逻辑；`why` 复用执行层的既有解释格式器。
- `explain --plan` 接受 `plan` 命令输出的 wrapper 文档（含 capability snapshot），也
  接受裸 plan 文档；`explain --report` 先跑 v4 report schema/对账校验，再从
  `plan_payload.document` 取回冻结 plan。
- `boundaries` 列出只有执行层能定的字段：`timing.*`、`artifact.content_digest/
  abi_version/format`、`dispatch.*` 对账、`resources.parallelism.thread_state`、
  `coverage.*`、`gates.*`、provider 加载期事实，以及 report 输入的 `why`
  （未锚定说明层副本，explain 原样呈现但不背书，见 §3.8）。此命令不执行，不产生这些值。

### 2.3 `plan`

```text
{schema_version, kind: "plan", requested_op,
 plan: {plan payload + plan_digest}, plan_digest,
 capability_snapshot: {schema + digest [, canonical_digest]}, capability_digest,
 request, resolved_policy, path_policy, digest}
```

- 只调 resolver（纯函数），不执行 provider。
- 同 `(policy, capability snapshot, request)` ⇒ 同 `plan_digest`；导入/导出的 snapshot
  round-trip 后 plan digest 不变；冻结 plan + 同一 snapshot 可被
  `execute_matmul_plan` / `execute_qmatmul_plan` 接受。
- wrapper 顶层 `digest` 是**必需**字段（覆盖 wrapper 去掉自身后的全部字段，含
  `capability_snapshot` 与 `path_policy`）；`explain --plan <wrapper>` 缺失或复算不符即
  `ARTIFACT_MISMATCH/plan_wrapper_digest_mismatch`，嵌套 `plan_digest` 仍按 plan schema
  校验。
- **内层 snapshot 可自证（round-3）**：默认脱敏时 `capability_snapshot.digest` 重定义为
  实际输出（basename）视图的 canonical digest，行内保留 `canonical_digest` 指向未脱敏
  完整快照（等于 wrapper 顶层 `capability_digest`）；`--show-paths` 时不带
  `canonical_digest`，内嵌文档即规范导出，`digest == capability_digest`。两种输出均可从
  文档自身复算 `digest`。
- `capability_snapshot.providers[].library.path` 默认只给 basename；`--show-paths` 显示
  完整路径（两种输出各自自洽，`plan_digest` 不变）。

### 2.4 capability snapshot 导出/导入

- `export_capability_snapshot` 返回 `CapabilitySnapshot.to_dict()`（含 digest）。
- **digest 可自证**：`digest == sha256(canonical_json(export_document 去掉 digest))`
  （preimage 使用与导出文档相同的 `schema_version` 键，不再改名成
  `capability_schema_version`）。
- `import_capability_snapshot` 为严格导入：要求登记的 `schema_version`（int，恰为 2）、
  必需的顶层字段（`schema_version/kind/triple/features/providers/digest`）、
  `sha256:` digest，并按登记 schema **递归类型严格**校验所有嵌套字段：拒绝
  `bool`↔`int`、`int`↔`float` 混淆、未知/缺失嵌套键、重复 `provider_id`、以及已知 opcode
  与本地 `OpDefinition.contract_digest` 不符的 `contracts`；之后再做一次类型严格的规范
  结构相等检查。未知/缺字段/非规范结构/digest 不符均 fail-closed
  （`ARTIFACT_MISMATCH` / `CAPABILITY_UNAVAILABLE`），不做默认填充。
- CLI 读取 JSON 时用 `object_pairs_hook` 拒绝任意层级的重复键，并拒绝
  `NaN/Infinity/-Infinity`；解析失败同样转成结构化错误文档（rc 1，无 traceback）。
  `import_capability_snapshot` 作为库 API 只接收已解析的 mapping，重复键需在调用方解析层
  拦截。
- `CapabilitySnapshot.from_dict` 的宽松行为（缺 digest 默认接受）保留为 U1 登记边界，
  仅严格导入路径受上述要求约束。
- **注册 manifest 集合不可清空（round-3）**：对 `provider_id` 在本地注册表中的条目，
  严格导入还要求文档保留 canonical manifest 的集合形状：规范非空的
  `contracts/dtypes/numeric_modes/required_features/preconditions/threads/pack/library`
  必须非空、对应静态必需键必须存在、规范为空的集合不得新增条目；探针相关可选键与
  本机数值不 pin。清空任一集合的 10 个复现（portable/aocl 的上述字段）全部
  `ARTIFACT_MISMATCH/capability_structure_mismatch`；CLI 文件路径因此 rc 1 结构化拒绝。
  若这类 snapshot 以对象形式（宽松 `from_dict` 边界）进入 doctor，doctor 的
  `provider_identity` 将其报为 failed、`fail_closed=true`（默认 rc 0、`--strict` rc 3）。
- 自洽但引用**未注册 opcode**的 snapshot 结构上仍可导入；doctor 将其报告为
  `fail_closed=true` 的事实（见 §2.1），resolver 只会按请求的已登记 opcode 工作。

## 3. 边界与未做项

1. 不新增 provider、不改变任何执行路径行为与数值；U1/U2 的 plan/report/echo/payload
   对账与 fail-closed 校验原样保留（发现层只读）。
2. 不输出私有端点/凭据/A3 访问信息；`doctor` 环境变量白名单与泄漏测试见证据。
3. 性能/时间数字一律 `UNGATED`；`explain` 只复制已验证 report 的原始值。
4. 本切片不做：`--emit` 之外的 CLI 配置发现（`pypto doctor` 顶层入口）、cost model /
   tuning profile、graph/region 作用域、35 个环境变量的迁移（U4）、remote capability
   probe、快照签名/attestation。
- `doctor` 的 `source=imported_snapshot` 只报告快照内已捕获事实，不重新探测；本地
  probe 路径才读取已 pin 库的 sha256/arch。

### 3.5 doctor 退出码与 `--strict`

- 默认：`doctor` 是事实报告，无论 `fail_closed` 与否都退出码 0；`fail_closed_checks`、
  `fail_closed` 字段与人类摘要完整列出条件，便于 CI/脚本自行判定。
- `--strict`：当 `fail_closed=true` 时退出码 **3**（报告仍会正常打印；`--json` 下文档仍可
  解析），便于需要"缺库即失败"的流水线。
- argparse 用法错误保持 conventional 退出码 **2**（与 strict-fail-closed 的 3 不同，
  无需读输出即可区分）；请求级错误（非法 snapshot、读文件失败、结构化解析失败等）在
  default/`--strict` 两种模式下都退出码 **1**，`--json` 时 stdout 必为可解析的
  `ExecutionError` 文档、stderr 无 traceback。
- `doctor --help` 的 epilog 明确写出四个码：`0 report` / `1 request-level error` /
  `2 argparse usage error` / `3 fail_closed=true with --strict`；`--strict` 帮助行也写明
  "request-level errors exit status 1; usage errors exit 2"。round-3 修复补齐了此前
  漏写的请求级 1。

### 3.6 跨主机静态规划（接受为边界）

- `plan` / `explain` 使用的 snapshot 必须与本机 probe 对齐（`kind=cpu` 且 triple 等于本机
  探测 triple），否则 `CAPABILITY_UNAVAILABLE`；`doctor --snapshot` 同样只接受本机对齐快照。
- 对齐口径就是 `kind + triple`：**同 triple、更弱 features / 更少 provider 的 snapshot
  会被接受**（显式测试钉住）。这是"快照是本机事实的子集"的直接后果，不额外声明能力；
  解析器只会从快照里列出的 provider 中选，缺失 provider 自然成为候选拒绝项。
- 为什么不做跨主机：已登记 provider 都在本机执行，其 version/pin/library/thread 事实来自
  本机 probe 与已 pin 产物；用一份外部快照在本机解析会让 plan 声明一个本机无法验证或执行的
  target，属于 0006 §2.1 "声明=执行" 的反面，因此本切片宁可拒绝也不假装支持。

### 3.7 `CapabilitySnapshot.from_dict` 宽松边界登记

- U1 语义保留：`CapabilitySnapshot.from_dict` 允许缺 `digest`/`triple` 等字段并按默认值
  重建（这是 U1 已登记的兼容边界）。
- 发现层的严格导入路径不受影响：`import_capability_snapshot` 在调用 `from_dict` **之前**
  先做必需字段/未知字段/递归类型/schema/digest 检查，之后再做类型严格的规范结构逐字段相等
  检查。
- 当前宽松路径的调用方：生产代码里只有 `import_capability_snapshot` 一处（且先经过严格
  检查）；直接以宽松方式调用的是 U1 回归测试
  `python/tests/ut/pypto_x/test_execution_matmul.py::test_capability_digest_mismatch_and_contract_mismatch_fail_closed`。
  resolver / plan / report / entry 均不经过该宽松默认路径。

### 3.8 report `resolution.why` 未锚定（接受为边界）

- v4 report 可携带 `resolution.why`；它来自执行层解释格式器，是**说明层副本**，v4 schema
  不对其内容做锚定/attestation（已在 `binding_verify` 的残留清单登记）。
- `explain(report=...)` 按契约原样呈现该字符串，并在 `boundaries` 中列出
  `why (report input)` 条目，明确"explain 不为其背书"；不把它当作执行事实，也不因它改变
  provider 选择或 plan/report 对账。
- 测试 `test_explain_echoes_report_why_without_endorsing_it` 钉住该行为。
