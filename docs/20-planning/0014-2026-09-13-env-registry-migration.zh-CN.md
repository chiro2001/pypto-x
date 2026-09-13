# U4 环境变量登记与迁移契约（env registry / 兼容层 / 未登记保护）

文档编号：`0014`

日期：2026-09-13（Asia/Shanghai）

状态：`IMPLEMENTED_PENDING_BATCH_VERIFICATION`

用途：冻结 U4（`docs/10-architecture/0005-...` §3.2 的 35 个实测环境变量、
`docs/10-architecture/0006-...` §9 的迁移契约）的登记表、兼容语义、弃用警告、未登记变量
保护与 doctor 集成方式。实现真源为 `python/pypto/execution/env_registry.py`
（`ENV_REGISTRY_SCHEMA_VERSION=1`），本文只记录契约与已接受边界，不复制实现。

## 1. 结论

1. **P0 登记**：35 个迁移变量全部登记，另有 U4 扫描发现的 2 个运行期读取变量
   （`PYPTO_X_OP_BENCH_AOCL_LIB`、`PYPTO_X_SVE_W8A8_DECODE_MEMO`）、2 个被
   capability/policy digest 捕捉的线程变量（`OMP_NUM_THREADS`、`BLIS_NUM_THREADS`）和
   1 个 registry 控制开关，共 40 行。每行含
   `name/category/meaning/default/legal_domain/scope/consumer_kind/replacement/
   deprecation_status/affects_artifact_semantics/semantic_scope/digest_status/
   effect_evidence/effect_unproven`。
2. **效果证据**：静态调用点审计（40 行覆盖、无零命中）+ 可运行实验（plan digest 环境矩阵、
   fail-closed replay）；35 个迁移变量的 effect 均有文件/行号锚点，`effect_unproven=false`。
3. **P1 兼容层**：旧变量继续生效，语义零变化；deprecated 变量按进程一次性输出
   `EnvDeprecationWarning`，可用 `PYPTO_X_ENV_DEPRECATION_WARNINGS=0` 静默；未登记的
   `PYPTO_X_*` 读取/设置输出一次性 `UnregisteredEnvWarning` 并进入 doctor 报告
   （名称级，值不回显）。
4. **语义筛查**：35 个迁移变量不改变 portable/AOCL 已登记 provider 的 execution plan/
   artifact 字节（10 个代表性变量 plan digest 与基线逐一相等）；5 个变量被判定为
   `outside_registered_execution_providers` 并标记为未来 provider 接入时的风险点；
   `OMP_NUM_THREADS`/`BLIS_NUM_THREADS` 会改变 capability 快照 digest，跨环境 replay
   fail-closed（`ARTIFACT_MISMATCH:capability_digest_mismatch`）。
5. 本切片**不改任何数值行为**、不破坏 `pypto.framework`/driver 的既有读取方式；
   fail-closed 相关变化均有 focused 测试；性能数字 `UNGATED`。

## 2. 单一真源 registry

`python/pypto/execution/env_registry.py` 是环境变量的唯一登记处：

- 常量：`MIGRATION_ENV_VARS`（冻结 35）、`EXTRA_ENV_VARS`（U4 新增登记 2）、
  `RUNTIME_ENV_VARS`（线程 2）、`CONTROL_ENV_VARS`（开关 1）、`ALL_ENV_VARS`（40）。
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
  `unregistered_env_rows()`。
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

- `schema_version`、`variables[40]`（`status/value_kind/interpreted_value` 等，值不外显；
  flag/integer 只给解释值）、`unregistered[]`、`allowlist`、`raw_values_included:false`、
  `deprecation_warnings_enabled`、`deprecation_warning_switch`；
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
这些都不影响 env registry 的 40 行内容。

## 7. 语义影响筛查结论（先量后改）

方法：对 portable + 已 pin AOCL provider 的 matmul/qmatmul plan 做环境矩阵重放
（`raw/plan_digest_env_matrix.json`）：

- 10 个代表变量（覆盖各 category）单独设置时，plan digest 与基线
  `sha256:e0d1287e...814df` 逐一相等；35 个迁移变量均不变更已登记 provider 的 plan/artifact
  字节。
- `OMP_NUM_THREADS` / `BLIS_NUM_THREADS` 会进入 capability 快照 digest；将
  `BLIS_NUM_THREADS=2` 冻结的 plan 在默认环境重放（不传 capability）→ 结构化
  `ARTIFACT_MISMATCH:capability_digest_mismatch`，符合 fail-closed 预期。
- 5 个 `affects_artifact_semantics=true` 的变量
  （`PYPTO_X_AARCH64_SYSROOT`、`PYPTO_X_ASCEND_ARTIFACT_FORMAT`、`PYPTO_X_ASCEND_HOOKS`、
  `PYPTO_X_PTO_ISA_ROOT`、`PYPTO_X_SVE_TEST_MODE`）均落在
  `outside_registered_execution_providers`：当前 35 变量筛查**未发现执行层真实 bug**，
  因此没有最小修复；它们登记为未来接入 Ascend/AArch64 native provider 时必须进入
  policy/capability digest 的风险点（"先量后改"）。
- `raw/` 保留 `env_registry_table.json`、`env_callsite_audit.json`、
  `plan_digest_env_matrix.json`、`warning_behavior.json`、`env_read_scan.json`、
  `doctor_environment.json`、`acceptance_summary.json`（全部块 `pass`）。

## 8. 未做项与后续

1. **P2 默认切换**：文档/示例只用 API 与配置文件，内部 driver 改为显式传 policy；
   env 不再创建新功能。届时把 P1 警告升级为 `CONFIG_ENV_DEPRECATED` 结构化事件。
2. **P3 移除**：跨一个主版本并具备迁移检查器后才移除纯内部变量；用户可见别名保留到
   major release 并在 changelog/ERRATA 登记。
3. 旧调用点的 `read_env` 替换（非执行层）仍待排期；在此之前依赖 CLI 进程级警告与 doctor。
4. Ascend/AArch64/vLLM 接入执行层时，5 个语义变量必须先进入 provider capability /
   policy digest，再允许参与 artifact 选择。
5. 线程变量如需成为一等 policy，应经 `execution.resources.max_threads`，不再新增 env。

## 9. 验收与证据

- 实现 commit（`work/u4-env-migration`，不 push）：U3 round-2 修复 `0e688aacb` 在前，
  U4 `3d43592cb` 在其上，U3 round-3 修复 `625484741b357e82900c912c0b29e7ecc9b8fed3` 为当前 tip；base `7378aa943`。
- focused：`python/pypto.execution` 相关 420 passed（含 discovery 68 + registry 16）；
  collect 2049 vs base 1994（+55 = U3 round-2 +26、U4 +16、round-3 +13）。
- U3 round-2/3 独立复现：`raw/u3fix_repro/` + `logs/u3fix_*.log`、`logs/r3_*.log`；
  `raw/v3_snapshot_tamper.json` 80 cases / 0 violations；`raw/r2_targeted.json`
  49 cases / 0 violations；`repro_empty_mapping_holes` 10/10 拒绝。
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u4-env-migration/`
  （`brief.zh-CN.md` + `raw/` + `logs/`）；证据不含原始 env 值、私有端点或凭据。
- 性能：`UNGATED`。

