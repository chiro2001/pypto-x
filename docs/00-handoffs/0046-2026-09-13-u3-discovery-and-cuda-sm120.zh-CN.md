# PyPTO-X 0046 波次快照：U3 发现命令（doctor/explain/plan）与 CUDA sm_120 原生目标

状态：`CLOSED_LOCALLY_ALL_VERIFICATIONS_REPORTED`（三条任务级判决均已终态；按自动推送政策已自动推送）
批次：`batch_0046_u3_discovery_and_cuda_sm120_2026_09_13`（见 `configs/development_lock.yaml`）
撰写：parent（自动批次）

## 0. 一句话

两条互不相干的切片：**U3** 把 U 线的事实/原因/可重放暴露成三个只读命令（外加严格 capability snapshot I/O 与 CLI），**CUDA sm_120** 把 sm_120/sm_121 做成显式可选的一等 arch（默认路径逐字节不变），并在真 5080 上验证三路径逐位一致与整图 digest 相同。

## 1. 冻结点

| 项 | 值 |
|---|---|
| integration tip（frozen） | **`393d28e0f`**（tree `eebb0d584c6f1d8088637a4f7c59a484e49e5bd7`） |
| 任务分支 commit | U3：`c60658974`+`f742ad840`+`095573974` → cherry-pick `d16152f92`+`2f7c49134`+`7378aa943`；CUDA：`333edbfb2`+`6f0b25fe7`+`e77508aa0` → cherry-pick `1fdecc8db`+`63261656b`+`ac86bce53`（后者基于 T14，用 patch-id 判等价，全部一致） |
| 补丁数 | **269**（上一 tip 264 + 收口期 5 个提交：U3 round-2 `835f453ff`、U4 registry `533e9f180`、CUDA r2 `a52687ade`、op-bench 加固 `27fbae5e4`、U3 round-3 `393d28e0f`）；am 复算：269 个全部干净应用（`269/269`），复算 tree `eebb0d584` = 冻结 tree |
| 规则 8 全量（T20，最终 tip） | **rc=0，2052 passed / 7 skipped / 0 failed，1013.32 s**（attempt 1，抖动未复发）；collect **2059**；基线 `2d6e87c7d` = **1952** |

## 2. 切片 A：U3 发现命令（`DISCOVERY_SCHEMA_VERSION=1`）

- **`doctor`**（事实，只读）：host triple/features、已注册 opcode + 契约 digest、provider capability/manifest digest、pin 库探测（存在/sha256/arch/ABI）、**白名单**环境摘要（只报 set/unset + 类型化解释，**不回显原始值**）、9 类 fail-closed 检查；**绝不输出私有端点/凭据**。默认 `rc=0`（条件显式列出），**`--strict` 在 `fail_closed=true` 时 `rc=2`**；请求级错误两种模式都 `rc=1`。
- **`explain`**（原因）：request/plan/report 三输入；**直接复用 resolver trace**（`candidates[].checks`/`rejections`/`rules`/`tie_break`/`fallback`/`resolved_policy`），输出选中项、guarantees（precision_class 镜像）、逐候选拒绝原因、fallback 链与事件、`why`、执行层边界表；report 输入先过 v4 校验再从 `plan_payload.document` 取冻结 plan；**未改 resolver**。
- **`plan`**（可重放）：冻结 plan + 解析所用 capability snapshot（含 digest），不执行；同 policy+同 snapshot 两次 digest 相同、导出导入后相同、`execute_matmul_plan` 同 snapshot 通过且 report `plan_digest` 一致。
- **snapshot I/O**：`export_capability_snapshot` + 严格 `import_capability_snapshot`（schema/必需与未知字段/类型/sha256 digest/规范结构逐字段相等）；11/11 篡改用例结构化拒绝。
- **CLI**：`python -m pypto.execution.cli doctor|explain|plan [--json] [--strict]`（另有 `scripts/perf/execution_cli.py`）。
- **登记边界**：`plan`/`explain` 只接受与本机 probe 对齐的 snapshot（跨主机静态规划不在本切片，理由：外部 snapshot 会声明一个本机无法验证/执行的 target，违反"声明=执行"）；`CapabilitySnapshot.from_dict` 的 U1 宽松语义保留，但发现层导入先做严格检查，且宽松路径生产代码唯一调用方就是严格导入自身。
- 文档：`docs/20-planning/0013-2026-09-13-u3-discovery-commands.zh-CN.md`（控制仓 `f80b099`→`df8d0c5`→`22c5d7ad`）。

## 2b. 切片 A2：U4 环境变量登记与迁移（`ENV_REGISTRY_SCHEMA_VERSION=1`）

- **单一真源**：`python/pypto/execution/env_registry.py`，**40 行 = 冻结迁移 35（22 deprecated + 13 internal）+ 扫描新增 2（`PYPTO_X_OP_BENCH_AOCL_LIB`、`PYPTO_X_SVE_W8A8_DECODE_MEMO`）+ 线程 2（`OMP_NUM_THREADS`、`BLIS_NUM_THREADS`）+ 控制开关 1（`PYPTO_X_ENV_DEPRECATION_WARNINGS`）**；doctor 的 `environment` 行严格由 registry 生成（白名单 40、`raw_values_included=false`）。
- **效果证据**：35/35 迁移变量均有静态调用点锚点（首个读点行号）；`unregistered_reads: []`；10 个代表性变量逐一设置时基线 plan digest 不变（`sha256:e0d1287e…`）；线程变量会改变 capability 快照 digest，跨环境 replay 结构化 fail-closed（`ARTIFACT_MISMATCH/capability_digest_mismatch`）。
- **语义筛查（先量后改）**：5 个 `affects_artifact_semantics=true` 全部判定 `outside_registered_execution_providers`（当前只注册 portable/AOCL）→ **未发现真实执行层 bug，故无最小修复**；作为未来 Ascend/AArch64 native/vLLM 接入时的 provider digest 风险点登记（`0014` §7/§8）。
- **兼容层**：旧变量行为零变化（`read_env` 与 `os.environ.get` 逐值一致）；deprecated 变量按进程一次性告警、`PYPTO_X_ENV_DEPRECATION_WARNINGS=0` 静默；未登记 `PYPTO_X_*` 一次性告警 + doctor 列名（敏感名脱敏、值不外显）；`policy_hints_from_env` 只给声明式提示，不隐式覆盖显式 policy。
- 测试：新增 `test_execution_env_registry.py` 16 条 focused；collect 1994→2020（U3 r2）→**2036（U4）**→2049（U3 r3）→**2059（op-bench 加固）**。文档 `docs/20-planning/0014-2026-09-13-env-registry-migration.zh-CN.md`。
- **验收状态（已回）**：独立验收 `f2d5db72` @`393d28e0f` 判 **PASS_WITH_BOUNDARIES**（12 条声明 11 条成立；加载树已按 ERR-0013 核实）。
  唯一被推翻的是**范围性**声明「仓库范围零未登记读取」：`tools/`、`scripts/perf` 与 1 个 UT 辅助文件里读了 6 个未登记名（`PYPTO_X_AOCL_LIB`/`PYPTO_X_AVX2_LAYOUT_SHA256_MANIFEST`/`PYPTO_X_DRIVER_PATH`/`PYPTO_X_LIBXSMM_LIB`/`PYPTO_X_VENDOR_EVIDENCE`/`PYPTO_X_WORKTREE`，全在执行层之外）；实现方扫描只覆盖 `python/pypto`，两者在该范围内一致。
  另两处文档漂移：`affects_artifact_semantics=true` 实为 6 行（5 个迁移变量 + extra `PYPTO_X_OP_BENCH_AOCL_LIB`），及 0014 §9 的 collect 2049 对冻结 tip 实测 2059（+10 来自 `27fbae5e4`）。已派收尾切片 `u4-followup-registry`（agent `925c7693`）登记 6 名 + 补筛 + 订正计数。
  验收方独立跑成规则 8 全量：rc=0 / 2052 passed / 7 skipped / 1009.75 s（collect 2059），focused 84 passed。

## 3. 切片 B：CUDA sm_120 / sm_121 显式原生目标

- **默认零改动**：默认仍 `sm_80`/PTX 8.0/payload v2；同一程序与上一 tip 的 `ptx_sha256`/`payload_sha256`/`content_digest` **逐字节一致**。
- **显式选择**：`arch=` / `ptx_target='sm_120'` / `TargetSpec(attributes={'arch':'sm_120'})` 三条路径；payload **升 v3**，`arch_selection{policy,source,requested_arch,ptx_target,ptx_version}` 进 payload+metadata 并被 digest 覆盖；旧 reader 读 v3 fail-closed；新 decoder 严格区分 v2（禁 selection）/v3（必须有且交叉校验）；冲突/非字符串/版本过低均拒绝；`CUDA_LOWERING_VERSION` 保持 2。
- **真机**（GamePC RTX 5080 + nvcc 13.3.73）：`sm80_ptx` / `sm120_ptx` / `sm120_cubin` 对 add f32/bf16 与 matmul f32/bf16 **逐位一致**；bf16 整图（4538 ops、371 in/51 out、packed 43200）三路径 + 两次 launch **输出 digest 相同** `c36ea307aa08e447d970268e526e835fdcbd7f58aec01ed0624f94946e3f6e3e`（前缀基线无 output digest，本轮首次给出）。
- **未做**：CUDA 上的 W8A8 **整图**（C7 图路径仅 CPU runtime，CUDA 只有 op 级 kernel——不以 op 级冒充端到端）；**PTX-JIT vs native cubin 计时腿**因游戏负载门 `perf deferred`（2026-09-13 04:27–04:58 UTC 有界重试 8 次、**8/8 脏**：util 24–28%、SM 2.3–2.4 GHz、power 104–105 W、mem 10.6 GB、无 compute apps；**零数字被记录**，重跑命令与 watcher 存 `cuda-sm120/compare/PERF_DEFERRED.md`，等干净窗口即可补测）。
- **边界**：WSL2 GPU-PV 时钟不可 pin（性能须 taskset+时钟断言、UNGATED）；cubin arch-locked（`sm_80` cubin 在 sm_120 报 209，PTX JIT 才前向兼容；sm_120 需 PTX ≥8.7，8.0/8.6 报 218）；`sm_121` 仅 emitter；只把已有 PTX 编到 sm_120，不做 Blackwell 专用指令。

## 4. 验收

| 切片 | agent | 判决 |
|---|---|---|
| U3 | `72415831` | **FAIL**（初检 @`7378aa943`，7 条发现）→ **FAIL**（r2 @`a52687ade`，1 个同族存活洞：清空注册 provider 集合仍被严格导入接受）→ **round-3 已修，由父方以自建 harness 收口为 10/10 REFUSED**（`ARTIFACT_MISMATCH`/`capability_structure_mismatch`，内部校验器对同样输入 raise）；plan wrapper digest 改为必填、help 补齐 4 个退出码、redacted 行自带 canonical digest 锚点。 |
| CUDA sm_120 | `2bda30dc` | **PASS_WITH_BOUNDARIES**（初检，6 条 fail-safe 边界）→ **PASS**（r2：D1–D6 全部闭合，含构造期版本拒绝、arch 与显式 target 互斥、版本精确 pin、类型化 arch_selection 错误、v2 仅默认路径、target attribute 双侧交叉校验；`sm_999` 拒绝、`render_ptx` 默认路径三树逐字节一致，边界测试在旧 tip 上 13/17 失败）。 |
| memo 完整性（0044 残余） | `2bda30dc` | **PASS_WITH_BOUNDARIES**（TOCTOU/FUSE/无界读 → round-2 已收/登记） |

父方复核（T16）：collect **1994**、discovery 文件 **29 passed**、`python -m pypto.execution.cli doctor` rc=0 且输出 digest；报告侧无误伤（portable + 真 AOCL f32/bf16 + 真 runtime fallback）；合并探针 **38/38 全拒**。

## 5. 证据路径

- U3：`_meta/pypto-x/u3-discovery/{brief.zh-CN.md,raw,logs}`；验收 `_meta/pypto-x/verify-0046-u3/`（初检）与 `verify-0046-u3-r2/`（r2）。
- U4：`_meta/pypto-x/u4-env-migration/{brief.zh-CN.md,raw,logs,scripts}`；补派验收 `_meta/pypto-x/verify-0046-u4/`。
- CUDA sm_120：`_meta/pypto-x/cuda-sm120/{validation.json,brief.zh-CN.md,sm120/,compare/,e2e/,tests/,raw}`；验收 `_meta/pypto-x/verify-0046-cuda-sm120/`。
- 批级全量：`_meta/pypto-x/batch-0046-u3-cuda-sm120/{raw,logs}`（T20 全量 + 基线 collect + 抖动负控）。
- 补丁：`patches/pypto-x/`（269）+ `patches/README.md`（HEAD `393d28e0f`）。
- 审计记录：本批的 U3 round-3 收口、harness 硬编码路径陷阱、op-bench 抖动加固分别见 `ERRATA.zh-CN.md` 的 ERR-0013 与 `configs/development_lock.yaml` 的 `op_bench_dispersion_flake`。

## 6. T20 收口（2026-09-13）

- 最终 tip `393d28e0f`：U3 3 提交 + CUDA 3 提交 + op-bench 加固 1 提交（净新增 5）。
- 批级全量：**2052 passed / 7 skipped / 0 failed / 0 error，1013.32 s，rc=0**（attempt 1，未重试）；collect 2059（基线 1952）。
- 补丁重导 269 个并在干净树上 `git am` 复算 269/269，复算 tree 与冻结 tree 相等。
- **验收方 harness 陷阱**：`u3_common.py` 内 `sys.path.insert(0, "…/verify-*/scripts")` 使"复跑"实际加载**旧冻结树**，其 round-3 复跑因此报"仍接受"；父方以自建 harness 直接验证冻结 tip，10/10 拒绝。已登记为 **ERR-0013**。
- op-bench 抖动（本批第 2 次 + 0044 1 次）已由 `27fbae5e4` 做**负载感知**加固：最多 3 对独立测量、任一对过严格判据即通过；仅当全失败且 `load1 > max(4, 0.25·nproc)` 或 `cpu PSI avg10 > 2` 才接受 ≤50% 相对漂移并打印理由；≥50% 或结构不符一律失败；锁内 `repeat=2,R=21` 严格规则不受影响。
- 性能数字状态：本批全部 **UNGATED**（含 CUDA 计时腿的 defer）。
