# U4 环境变量登记与迁移契约（env registry / 兼容层 / 未登记保护）

文档编号：`0014`

日期：2026-09-13（Asia/Shanghai）

状态：`FOLLOWUP_FIXED_PENDING_REVERIFICATION`（batch 0046 U4 独立验收 `verify-0046-u4` 判
`PASS_WITH_BOUNDARIES`；4 项边界已回修：仓库范围补登记 6 名、全表 `affects=true` 6 行筛查、
§9 计数订正、已知边界登记，见 §3.1/§7/§9）

用途：冻结 U4（`docs/10-architecture/0005-...` §3.2 的 35 个实测环境变量、
`docs/10-architecture/0006-...` §9 的迁移契约）的登记表、兼容语义、弃用警告、未登记变量
保护与 doctor 集成方式。实现真源为 `python/pypto/execution/env_registry.py`
（`ENV_REGISTRY_SCHEMA_VERSION=1`），本文只记录契约与已接受边界，不复制实现。

## 1. 结论

1. **P0 登记**：35 个迁移变量全部登记，另有 U4 执行层扫描发现的 2 个运行期读取变量
   （`PYPTO_X_OP_BENCH_AOCL_LIB`、`PYPTO_X_SVE_W8A8_DECODE_MEMO`）、followup 仓库范围
   扫描（`python/` + `tools/` + `scripts/`，`verify-0046-u4` 声明 3）补登记的 6 个工具/证据
   读取变量（见 §3.1）、2 个被 capability/policy digest 捕捉的线程变量
   （`OMP_NUM_THREADS`、`BLIS_NUM_THREADS`）和 1 个 registry 控制开关，共 **46 行**
   （35 + 8 + 2 + 1）。`ENV_REGISTRY_SCHEMA_VERSION` 仍为 1：6 名全部复用现有
   `category`/枚举值，未引入新类别。每行含
   `name/category/meaning/default/legal_domain/scope/consumer_kind/replacement/
   deprecation_status/affects_artifact_semantics/semantic_scope/digest_status/
   effect_evidence/effect_unproven`。
2. **效果证据**：静态调用点审计（46 行覆盖、无零命中）+ 可运行实验（plan digest 环境矩阵、
   fail-closed replay）；仓库范围扫描 33 个 `PYPTO_X_*` 读取名全部已登记（未登记 0，见 §9）；
   全部 46 行的 effect 均有文件/行号锚点，`effect_unproven=false`。
3. **P1 兼容层**：旧变量继续生效，语义零变化；deprecated 变量按进程一次性输出
   `EnvDeprecationWarning`，可用 `PYPTO_X_ENV_DEPRECATION_WARNINGS=0` 静默；未登记的
   `PYPTO_X_*` 读取/设置输出一次性 `UnregisteredEnvWarning` 并进入 doctor 报告
   （名称级，值不回显）。
4. **语义筛查**：35 个迁移变量不改变 portable/AOCL 已登记 provider 的 execution plan/
   artifact 字节（10 个代表性变量 plan digest 与基线逐一相等）；全表 6 个
   `affects_artifact_semantics=true` 的行（5 个迁移变量 + `PYPTO_X_OP_BENCH_AOCL_LIB`）已按
   同一口径逐一筛查，全部为 `outside_registered_execution_providers`（后者另注"仅 bench
   工具"），详见 §7；`OMP_NUM_THREADS`/`BLIS_NUM_THREADS` 会改变 capability 快照 digest，
   跨环境 replay fail-closed（`ARTIFACT_MISMATCH:capability_digest_mismatch`）。
5. 本切片**不改任何数值行为**、不破坏 `pypto.framework`/driver 的既有读取方式；
   fail-closed 相关变化均有 focused 测试；性能数字 `UNGATED`。

## 2. 单一真源 registry

`python/pypto/execution/env_registry.py` 是环境变量的唯一登记处：

- 常量：`MIGRATION_ENV_VARS`（冻结 35）、`EXTRA_ENV_VARS`（U4 新增登记 8 = 执行层扫描 2
  + followup 仓库范围扫描 6）、`RUNTIME_ENV_VARS`（线程 2）、`CONTROL_ENV_VARS`（开关 1）、
  `ALL_ENV_VARS`（46）。
- 只读 API：`all_env_vars/migration_env_vars/extra_env_vars/runtime_env_vars/
  control_env_vars/registered_names/lookup_env_var/is_registered/environment_rows/
  unregistered_env_rows/unregistered_env_names/registry_document/policy_hints_from_env/
  warnings_enabled/read_env/warn_deprecations/reset_warnings`。
- 警告类型：`EnvDeprecationWarning`、`UnregisteredEnvWarning`。
- 字段语义：
  - `category`：`build_artifact_dir/toolchain_path/performance/runtime_device/model_asset/
    vllm_integration/dev_resource/provider_thread_env/registry_control`；
  - `deprecation_status`：`deprecated`（用户可见，迁移警告）| `internal`（工具/内部，
    不主动向用户告警）| `active`（线程变量，非弃用）；
  - `affects_artifact_semantics`：是否可能改变制品字节；
  - `digest_status`：`not_an_artifact_input` | `outside_registered_execution_providers` |
    `captured_in_capability_or_policy_digest` | `effect_unproven`（保留值，当前为 0）；
  - `effect_evidence`：静态调用点锚点（`文件:行`），`effect_unproven` 为未证实标记。
  - 静态扫描范围（followup 冻结）：仓库范围扫描固定为 `python/` + `tools/` + `scripts/`
    （递归 `.py`，排除 `.git`/`build`/`__pycache__`/`.pytest_cache`/`node_modules`/
    `.mypy_cache`/`.tox`），与 `verify-0046-u4` 口径一致（regex + AST，覆盖
    `environ.get`/`environ[...]`/`getenv`/`setdefault`/`pop`/赋值读取）；守卫测试
    `test_execution_env_registry.py::test_repository_scope_scan_has_no_unregistered_pyp_to_x_reads`，
    原 `python/pypto` 子范围守卫保留为执行层子集测试。followup 扫描结果：原始冻结 tip
    `393d28e0f`（1387 个 `.py`）与 rebase 后 base `3c2756c0e`（1390 个 `.py`）均为
    33 个读取名、6 个未登记；followup 树上 33/33 已登记、未登记 0。
- 兼容入口：`read_env(name, default)` 返回值与旧 `os.environ.get` 完全一致；仅额外
  触发一次性警告，绝不改变数值/路径解析结果。

## 3. 迁移变量登记表（35）

下表为 `0005` §3.2 的冻结清单；完整 `meaning/default/legal_domain/semantic_scope/
effect_evidence` 见 registry 与 `_meta/pypto-x/u4-env-migration/raw/env_registry_table.json`。

| 变量 | 类别 | 弃用状态 | 影响制品语义 | digest 判定 | 替代入口 |
|---|---|---|---|---|---|
| `PYPTO_X_AVX2_BUILD_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `build.targets.avx2.dir` |
| `PYPTO_X_AVX2_ARTIFACT_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `artifact_store.targets.avx2.dir` |
| `PYPTO_X_AVX2_PROBE_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `capability.probe_cache.avx2.dir` |
| `PYPTO_X_AVX512_BUILD_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `build.targets.avx512.dir` |
| `PYPTO_X_AVX512_ARTIFACT_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `artifact_store.targets.avx512.dir` |
| `PYPTO_X_AVX512_PROBE_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `capability.probe_cache.avx512.dir` |
| `PYPTO_X_SVE256_BUILD_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `build.targets.sve256.dir` |
| `PYPTO_X_SVE256_ARTIFACT_DIR` | build_artifact_dir | deprecated | 否 | `not_an_artifact_input` | `artifact_store.targets.sve256.dir` |
| `PYPTO_X_ASCEND_BUILD_DIR` | build_artifact_dir | internal | 否 | `outside_registered_execution_providers` | `build.targets.ascend.dir` |
| `PYPTO_X_ASCEND_ARTIFACT_FORMAT` | build_artifact_dir | internal | 是 | `outside_registered_execution_providers` | `artifact.format (Ascend provider profile)` |
| `PYPTO_X_AARCH64_SYSROOT` | toolchain_path | deprecated | 是 | `outside_registered_execution_providers` | `toolchain.aarch64.sysroot` |
| `PYPTO_X_CANN_INSTALL_PATH` | toolchain_path | internal | 否 | `outside_registered_execution_providers` | `toolchain.ascend.install_path` |
| `PYPTO_X_CANN_META_ROOT` | toolchain_path | internal | 否 | `not_an_artifact_input` | `evidence.ascend.cann_toolkit_root` |
| `PYPTO_X_PTO_ISA_ROOT` | toolchain_path | internal | 是 | `outside_registered_execution_providers` | `toolchain.pto_isa.root` |
| `PYPTO_X_PYPTO_ROOT` | toolchain_path | internal | 否 | `not_an_artifact_input` | `developer.pypto_root` |
| `PYPTO_X_HWHIAIUSER_HOME` | toolchain_path | internal | 否 | `not_an_artifact_input` | `toolchain.ascend.hwhiaiuser_home` |
| `PYPTO_X_TORCH_CACHE_DIR` | toolchain_path | deprecated | 否 | `not_an_artifact_input` | `cache.torch_dir` |
| `PYPTO_X_AVX512_GEMM_THREADS` | performance | deprecated | 否 | `outside_registered_execution_providers` | `execution.resources.max_threads` |
| `PYPTO_X_AVX512_WEIGHT_CACHE` | performance | deprecated | 否 | `not_an_artifact_input` | `cache.weight_pack.enabled` |
| `PYPTO_X_VLLM_GEMM_THREADS` | performance | deprecated | 否 | `outside_registered_execution_providers` | `execution.resources.max_threads` |
| `PYPTO_X_ASCEND_DEVICE_ID` | runtime_device | deprecated | 否 | `outside_registered_execution_providers` | `target.device_id / placement.device` |
| `PYPTO_X_ASCEND_HOOKS` | runtime_device | internal | 是 | `outside_registered_execution_providers` | `Ascend provider/plugin registry` |
| `PYPTO_X_SVE_QEMU_CPU` | runtime_device | internal | 否 | `not_an_artifact_input` | `target.qemu.cpu` |
| `PYPTO_X_SVE_TEST_MODE` | runtime_device | internal | 是 | `outside_registered_execution_providers` | `validation.mode` |
| `PYPTO_X_PORTABLE_ONLY` | runtime_device | deprecated | 否 | `not_an_artifact_input` | `ExecutionPolicy(profile='portable', provider.mode='portable')` |
| `PYPTO_X_QWEN35_ASSET_DIR` | model_asset | deprecated | 否 | `not_an_artifact_input` | `assets.models.qwen35.asset_dir` |
| `PYPTO_X_QWEN35_ASSETS_DIR` | model_asset | deprecated | 否 | `not_an_artifact_input` | `assets.models.qwen35.assets_dir` |
| `PYPTO_X_QWEN35_WEIGHTED_WORKTREE` | model_asset | deprecated | 否 | `not_an_artifact_input` | `assets.models.qwen35.weighted_worktree` |
| `PYPTO_X_VLLM_ENABLE` | vllm_integration | deprecated | 否 | `outside_registered_execution_providers` | `integration.vllm.enable` |
| `PYPTO_X_VLLM_GATE_JSON` | vllm_integration | deprecated | 否 | `outside_registered_execution_providers` | `integration.vllm.gate_json` |
| `PYPTO_X_VLLM_LOG` | vllm_integration | deprecated | 否 | `not_an_artifact_input` | `integration.vllm.log` |
| `PYPTO_X_TORCH_FALLBACK_WARNINGS` | vllm_integration | deprecated | 否 | `not_an_artifact_input` | `observability.fallback_warnings` |
| `PYPTO_X_LOCAL_HEAVY_RUNNER` | dev_resource | internal | 否 | `not_an_artifact_input` | `developer.resource.runner` |
| `PYPTO_X_RESOURCE_LOCK_ROOT` | dev_resource | internal | 否 | `not_an_artifact_input` | `developer.resource.lock_root` |
| `PYPTO_X_RESOURCE_USAGE_DIR` | dev_resource | internal | 否 | `not_an_artifact_input` | `developer.resource.usage_dir` |

### 3.1 仓库范围补登记（followup，6 行）

`verify-0046-u4` 声明 3 的仓库范围扫描（`python/` + `tools/` + `scripts/`；原始冻结 tip
`393d28e0f` 1387 个 `.py`，rebase 后 base `3c2756c0e` 1390 个 `.py`）发现 6 个被读取但
未登记的 `PYPTO_X_*` 名；两版 base 的读取名集合相同（33 个），它们全部落在工具/证据/UT
辅助路径，不在执行层 provider/plan/report。followup 将其登记进 `EXTRA_ENV_VARS`
（`ENV_REGISTRY_SCHEMA_VERSION` 仍为 1，复用既有 category/枚举），使"仓库范围扫描 0
未登记"成为可测事实：

| 变量 | 类别 | 作用域 | legal_domain | 影响制品语义 | digest 判定 | 替代入口 |
|---|---|---|---|---|---|---|
| `PYPTO_X_AOCL_LIB` | toolchain_path | tooling | file path | 否 | `outside_registered_execution_providers` | `scripts/perf --lib` / vendor survey config |
| `PYPTO_X_AVX2_LAYOUT_SHA256_MANIFEST` | dev_resource | test | file path | 否 | `not_an_artifact_input` | pytest test-local fixture/parameter |
| `PYPTO_X_DRIVER_PATH` | toolchain_path | tooling | file path | 否 | `not_an_artifact_input` | 探针默认候选路径（暂无 CLI） |
| `PYPTO_X_LIBXSMM_LIB` | toolchain_path | tooling | file path | 否 | `outside_registered_execution_providers` | vendor_gemm_matrix 工具配置 |
| `PYPTO_X_VENDOR_EVIDENCE` | dev_resource | developer | directory path | 否 | `not_an_artifact_input` | vendor survey `--evidence` |
| `PYPTO_X_WORKTREE` | dev_resource | developer | directory path | 否 | `not_an_artifact_input` | vendor survey `--worktree` |

这 6 名均为 `deprecation_status=internal`（工具内部变量，不主动告警）；登记后设置这些名字
不再触发 `UnregisteredEnvWarning`，doctor 行集合由 40 变 46，其余执行层行为不变。读取点
行号见 `_meta/pypto-x/u4-followup-registry/raw/repo_scan_before.json` /
`repo_scan_after.json`，字段与白名单一致性见 `raw/registry_parity.json`。

## 4. 兼容层与一次性弃用警告

- **优先级**沿用 `0006` §9：代码 API / 显式 CLI > 项目配置 > 兼容环境变量 > 平台默认值。
- **旧变量继续生效**：`read_env` 与全部既有 `os.environ.get` 调用点的返回值不变；
  U4 只做登记、警告与可发现性，不做任何默认值切换或路径重写。
- **一次性**：同一进程内每个变量最多告警一次（以 `deprecated:<name>` /
  `unregistered:<name>` 为键），重复读取不再刷屏；`reset_warnings()` 仅供测试与长生命周期工具。
- **静默开关**：`PYPTO_X_ENV_DEPRECATION_WARNINGS` ∈ `{0,off,false,no}`（大小写不敏感）
  时完全静默；未设置或其它值保持开启。doctor 输出
  `deprecation_warnings_enabled` 与 `deprecation_warning_switch`。
- **触发时机**：`warn_deprecations()` 在 CLI 命令分发前集中调用；执行层新代码应使用
  `read_env()` 以在使用点获得同样的一次性告警（`policy_hints_from_env` 提供声明式等价
  片段，但**不隐式应用**，环境不能覆盖显式 `ExecutionPolicy`）。
- **已接受边界**：本切片不逐个改造 35 个变量的历史调用点（`pypto.framework`、
  `pypto_x_vllm`、driver、tools 各有一批读取点，逐个注入会引入 import cycle 与上游
  改动风险）。因此警告是"进程级 + 使用点 API"两层，而非"每次 env 读取必定告警"；
  `read_env` 是后续替换的目标接口。该边界在 `raw/warning_behavior.json` 有行为证据。

## 5. 未登记 `PYPTO_X_*` 保护

- **检测**：对当前进程环境中所有 `PYPTO_X_*` 名称做注册表比对；未登记名称进入
  `unregistered_env_rows()`。静态侧由固定范围的仓库扫描守卫（`python/` + `tools/` +
  `scripts/`，口径见 §2；followup 后 33 个读取名全部已登记、未登记 0）。
- **告警**：每个未登记名称按进程一次性输出 `UnregisteredEnvWarning`，提示"执行层不
  承认该变量；请登记或移除"；`read_env("PYPTO_X_*")` 亦触发同类告警。
- **报告**：doctor `environment.unregistered[]` 输出
  `{name, risk, status:"unregistered", policy}`；`name` 对包含
  `secret/token/password/credential/api_key` 等敏感片段的名称整体替换为
  `<redacted-unregistered-name>`，任何情况下不回显变量值，`raw_values_included=false`。
- **风险分级**：`behavior_or_numeric_risk`（`*_HOOKS/_MODE/_FORMAT/_THREADS/_DEVICE_ID/_ENABLE`）、
  `path_or_cache_risk`（`*_DIR/_PATH/_ROOT/_HOME/_LOG/_CACHE`）、`unknown_risk`。
- **为何不 fail-closed**：未登记变量在 U1–U3 执行层中本来就不参与 provider 选择或
  artifact 生成；若仅因宿主设置了无关 `PYPTO_X_*` 就拒绝执行，会把"未知风险"升级为
  "可用性故障"。当前策略是显著一次性告警 + doctor 报告 + 敏感名脱敏；待 P2 引入
  配置文件/严格模式时，可在 doctor `--strict` 或显式 `ExecutionPolicy` 下升级为 fail-closed。

## 6. doctor 集成

`python -m pypto.execution.cli doctor` 的 `environment` 块完全由 registry 驱动
（`source: "pypto.execution.env_registry"`），包含：

- `schema_version`、`variables[46]`（`status/value_kind/interpreted_value` 等，值不外显；
  flag/integer 只给解释值）、`unregistered[]`、`allowlist`、`raw_values_included:false`、
  `deprecation_warnings_enabled`、`deprecation_warning_switch`；followup 新增 6 行后
  doctor 输出必然变化 6 行，这是预期：受影响的测试/冻结值只有 registry 行数与 extra 计数
  断言（`extra_var_count` 2→8、`variables`/doctor 行 40→46、`extra_env_vars()` 集合新增
  6 名），已随实现更新并在 §9 记录；执行层 plan/artifact/report digest 不因此变化
  （§9 同一请求 digest 证据）。
- `env_registry` 计数块：`migration_var_count/extra_var_count/runtime_var_count/
  control_var_count/unregistered_count/new_env_policy`；
- 新增 `PYPTO_X_*` 功能代码若绕过 registry 直接读 env，会被
  `test_execution_env_registry.py` 的静态扫描测试发现（`env_read_scan.json` 中
  `unregistered_reads: []`）。

doctor 的退出码契约（与 U3 round-2/round-3 修复后一致）：默认 rc=0，`--strict` 且
`fail_closed=true` 时 rc=3，argparse 用法错误 rc=2，请求级错误 rc=1；`doctor --help`
epilog 写明全部四个码。doctor/plan 的 `library.path` 默认只给 basename
（`--show-paths` 才显示完整 host 路径），`path_policy` 进入文档 digest；默认脱敏时
`providers[].manifest_digest` / 内嵌 `capability_snapshot.digest` 定义在实际输出视图上
（分别带 `manifest_canonical_digest` / `canonical_digest` 锚点），因此脱敏输出仍可自证。
这些都不影响 env registry 的 46 行内容。

## 7. 语义影响筛查结论（先量后改）

方法：对 portable + 已 pin AOCL provider 的 matmul/qmatmul plan 做环境矩阵重放
（`raw/plan_digest_env_matrix.json`）：

- 10 个代表变量（覆盖各 category）单独设置时，plan digest 与基线
  `sha256:e0d1287e...814df` 逐一相等；35 个迁移变量均不变更已登记 provider 的 plan/artifact
  字节。
- `affects_artifact_semantics=true` 的全表 6 行按同一口径逐一筛查
  （followup `raw/affects_screening.json`）：5 个迁移变量
  （`PYPTO_X_AARCH64_SYSROOT`、`PYPTO_X_ASCEND_ARTIFACT_FORMAT`、`PYPTO_X_ASCEND_HOOKS`、
  `PYPTO_X_PTO_ISA_ROOT`、`PYPTO_X_SVE_TEST_MODE`）+ `PYPTO_X_OP_BENCH_AOCL_LIB`。
  逐个设典型合法值后 plan/capability/wrapper digest、portable artifact sha256、AOCL
  artifact sha256 与 AOCL 输出 sha256 全部与基线相等。
- 静态调用点：`python/pypto/execution/` 的 provider 核心文件（providers/provider_aocl/
  provider_aocl_int8/resolver/plan/entry/report/manifest/capability/provider_identity/
  discovery/policy）对 6 名 0 命中（只出现在 `env_registry.py`）。5 个迁移变量仍判定
  `outside_registered_execution_providers`，登记为未来接入 Ascend/AArch64 native provider
  时必须进入 policy/capability digest 的风险点（"先量后改"）。
- `PYPTO_X_OP_BENCH_AOCL_LIB` 的唯一非 registry 读取点是
  `python/pypto/op_bench/provider_aocl.py`（常量定义）与 `python/pypto/op_bench/runner.py`
  （写 env），属 op-bench 工具，不进入已登记 provider 的 plan/artifact；结论明确写为
  **outside_registered_execution_providers + 仅 bench 工具**。
- `OMP_NUM_THREADS` / `BLIS_NUM_THREADS` 会进入 capability 快照 digest；将
  `BLIS_NUM_THREADS=2` 冻结的 plan 在默认环境重放（不传 capability）→ 结构化
  `ARTIFACT_MISMATCH:capability_digest_mismatch`，符合 fail-closed 预期。
- 默认路径 digest 不变：对同一 matmul 请求（default/portable/AOCL）比较三棵树——原始冻结
  tip `393d28e0f`、rebase 后 base `3c2756c0e`（int8 exact-blocked + AVX2 position-native）
  与 followup 树 `8cf9a1b4e`：plan/capability/wrapper digest、artifact/输出 sha256 以及
  去掉 timing 的 report 摘要逐项相等（`raw/plan_digest_invariance.json`）。
- `raw/` 保留 `env_registry_table.json`、`env_callsite_audit.json`、
  `plan_digest_env_matrix.json`、`warning_behavior.json`、`env_read_scan.json`、
  `doctor_environment.json`、`acceptance_summary.json`（全部块 `pass`）；followup 证据
  （仓库范围扫描、affects 筛查、digest 不变、白名单一致性）见 §9 与
  `_meta/pypto-x/u4-followup-registry/raw/`。

## 8. 未做项与后续

1. **P2 默认切换**：文档/示例只用 API 与配置文件，内部 driver 改为显式传 policy；
   env 不再创建新功能。届时把 P1 警告升级为 `CONFIG_ENV_DEPRECATED` 结构化事件。
2. **P3 移除**：跨一个主版本并具备迁移检查器后才移除纯内部变量；用户可见别名保留到
   major release 并在 changelog/ERRATA 登记。
3. 旧调用点的 `read_env` 替换（非执行层）仍待排期；在此之前依赖 CLI 进程级警告与 doctor。
4. Ascend/AArch64/vLLM 接入执行层时，5 个语义变量必须先进入 provider capability /
   policy digest，再允许参与 artifact 选择。
5. 线程变量如需成为一等 policy，应经 `execution.resources.max_threads`，不再新增 env。
6. followup 只登记 6 个工具/证据读取名，未把其调用点改为 `read_env`（避免工具行为/输出变化）；
   这些名字仍是直接 `os.environ` 读取，登记只消除未登记告警并纳入 doctor 白名单。

## 9. 验收与证据

- 实现 commit（`work/u4-env-migration`，不 push）：U3 round-2 修复 `0e688aacb` 在前，
  U4 `3d43592cb` 在其上，U3 round-3 修复 `625484741b357e82900c912c0b29e7ecc9b8fed3` 为当前 tip；base `7378aa943`。
- focused（U4 时点）：`python/pypto.execution` 相关 420 passed（含 discovery 68 + registry 16）；
  collect 2049 vs base 1994（+55 = U3 round-2 +26、U4 +16、round-3 +13）。
- U3 round-2/3 独立复现：`raw/u3fix_repro/` + `logs/u3fix_*.log`、`logs/r3_*.log`；
  `raw/v3_snapshot_tamper.json` 80 cases / 0 violations；`raw/r2_targeted.json`
  49 cases / 0 violations；`repro_empty_mapping_holes` 10/10 拒绝。
- **§9 计数订正（followup）**：上面的 2049 是 U4 实现时点的历史值；原始冻结 tip `393d28e0f`
  实测 collect **2059**（`verify-0046-u4` 独立复现，`logs/full_ut_collect.log`），+10 来自先于
  冻结 tip 的祖先 `27fbae5e4`（op-bench 测试改动），与 registry 行数/类别/digest 无关。
- **rebase 后的 base 与 followup 计数**：base `3c2756c0e`（int8 exact-blocked `c91d771b9`
  + AVX2 position-native）collect **2169**；followup 后 collect **2170** = 2169 + 1 条仓库
  范围扫描测试（`logs/collect_after_summary.txt`）。
- **followup focused**：`test_execution_env_registry.py` 17 passed（16 条既有 + 仓库范围扫描）
  + `test_execution_discovery.py` 68 passed = **85 passed**，rc=0，11.89s
  （`logs/focused_registry.log`、`logs/focused_registry_discovery.log`）。
- **followup 实现 commit**（分支 `work/u4-followup-registry`，不 push）：
  `8cf9a1b4e`（rebase 到 `3c2756c0e`；rebase 前 `fc2ce8abf`）。`git log --oneline -3`：
  `8cf9a1b4e fix(execution): register repo-scope tool env reads from verify-0046-u4` ←
  `3c2756c0e test(avx2): cover scalar operands, scalar where conditions and empty outputs` ←
  `2f0b31d0f feat(avx2): native iota/compare/where position-control plane (payload 7)`。
- **followup 规则 8 全量**：`2163 passed, 7 skipped in 1243.30s (0:20:43)`，collect 2170，
  rc=0；走共享 `local` 锁，75=BUSY 时 sleep 300 重试（成功 attempt 2 于 18:17:32 取得锁、
  18:38:17 完成）；`logs/full_ut_summary.txt`、`logs/full_ut_outer.log`、
  `logs/full_ut_attempts.txt`、`logs/full_ut_resource.log`。
- **followup 证据目录**：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u4-followup-registry/`
  （`brief.zh-CN.md` + `raw/repo_scan_before.json`、`raw/repo_scan_after.json`、
  `raw/affects_screening.json`、`raw/plan_digest_invariance.json`、`raw/registry_parity.json`
  + `logs/` + `scripts/`）；证据不含原始 env 值、私有端点或凭据。
- 性能：`UNGATED`。

### 9.1 已知边界（followup 登记，不改行为）

1. **一次性告警状态与模块重载**：`importlib.reload(env_registry)` 会清空模块级 `_WARNED`，
   重载后同一变量会再次告警；正常重复 `import` 不会。这是 Python 模块重载的既有语义，
   `reset_warnings()` 文档已说明，followup 只登记不改变。
2. **AOCL 只在 zen4 本机真实执行**：artifact/输出字节对比在本机固定 pin 的 AOCL-BLIS 5.3.2
   zen4 产物上真实执行；Ascend、AArch64 native（SVE）与 GPU 不在本机覆盖范围，未在这些
   后端验证 5 个语义变量的正向语义，registry 将其登记为未来 provider 接入风险点。
3. **Ascend/AArch64 native/GPU 未覆盖**：本批的全部正向结论只覆盖 portable + 已注册 AOCL
   provider；Ascend hook、SVE native、CUDA/其他 GPU 路径既未执行也未声明。
4. **锁内 monitor 注入线程环境**：`run_local_heavy.sh` 的 monitor 固定向子进程注入
   `OMP_NUM_THREADS=<selected cpus>`（本机 max_cpus=6 时为 `OMP_NUM_THREADS=6`，同时注入
   OPENBLAS/MKL/NUMEXPR/RAYON 等）。因此规则 8 锁内环境与干净 shell 的 capability 快照
   digest 可能不同；本 followup 的 digest 筛查/不变性脚本在子进程内显式清除
   `OMP_NUM_THREADS`/`BLIS_NUM_THREADS` 后测量，这是 harness 事实而非 registry 行为。
5. **登记 6 个工具/证据名的行为边界**：设置这些名字不再触发 `UnregisteredEnvWarning`，
   doctor 白名单由 40 变 46；它们不进入已登记 provider 的 plan/artifact/report，锁内工具
   行为不变（调用点未改为 `read_env`）。

