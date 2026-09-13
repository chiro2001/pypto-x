# PyPTO-X 0046 波次快照：U3 发现命令（doctor/explain/plan）与 CUDA sm_120 原生目标

状态：`CLOSED_LOCALLY_VERIFICATION_IN_FLIGHT`（两条独立验收在跑；按自动推送政策，判决回齐后推送）
批次：`batch_0046_u3_discovery_and_cuda_sm120_2026_09_13`（见 `configs/development_lock.yaml`）
撰写：parent（自动批次）

## 0. 一句话

两条互不相干的切片：**U3** 把 U 线的事实/原因/可重放暴露成三个只读命令（外加严格 capability snapshot I/O 与 CLI），**CUDA sm_120** 把 sm_120/sm_121 做成显式可选的一等 arch（默认路径逐字节不变），并在真 5080 上验证三路径逐位一致与整图 digest 相同。

## 1. 冻结点

| 项 | 值 |
|---|---|
| integration tip（frozen） | **`7378aa943`**（tree `4f0e146038e5e63bdee1506d9cc302379e459193`） |
| 任务分支 commit | U3：`c60658974`+`f742ad840`+`095573974` → cherry-pick `d16152f92`+`2f7c49134`+`7378aa943`；CUDA：`333edbfb2`+`6f0b25fe7`+`e77508aa0` → cherry-pick `1fdecc8db`+`63261656b`+`ac86bce53`（后者基于 T14，用 patch-id 判等价，全部一致） |
| 补丁数 | **264**（0044 = 258）；am 复算：264 个全部干净应用，复算 tree = 冻结 tree |
| 规则 8 全量 | **rc=0，1987 passed / 7 skipped / 0 failed，961.91 s**（attempt 1）；collect **1994**；基线 `2d6e87c7d` = **1952**（runner 基线路径未对上预置目录，父方手工补跑基线 collect 到 `raw/collect-2d6e87c7d.out`） |

## 2. 切片 A：U3 发现命令（`DISCOVERY_SCHEMA_VERSION=1`）

- **`doctor`**（事实，只读）：host triple/features、已注册 opcode + 契约 digest、provider capability/manifest digest、pin 库探测（存在/sha256/arch/ABI）、**白名单**环境摘要（只报 set/unset + 类型化解释，**不回显原始值**）、9 类 fail-closed 检查；**绝不输出私有端点/凭据**。默认 `rc=0`（条件显式列出），**`--strict` 在 `fail_closed=true` 时 `rc=2`**；请求级错误两种模式都 `rc=1`。
- **`explain`**（原因）：request/plan/report 三输入；**直接复用 resolver trace**（`candidates[].checks`/`rejections`/`rules`/`tie_break`/`fallback`/`resolved_policy`），输出选中项、guarantees（precision_class 镜像）、逐候选拒绝原因、fallback 链与事件、`why`、执行层边界表；report 输入先过 v4 校验再从 `plan_payload.document` 取冻结 plan；**未改 resolver**。
- **`plan`**（可重放）：冻结 plan + 解析所用 capability snapshot（含 digest），不执行；同 policy+同 snapshot 两次 digest 相同、导出导入后相同、`execute_matmul_plan` 同 snapshot 通过且 report `plan_digest` 一致。
- **snapshot I/O**：`export_capability_snapshot` + 严格 `import_capability_snapshot`（schema/必需与未知字段/类型/sha256 digest/规范结构逐字段相等）；11/11 篡改用例结构化拒绝。
- **CLI**：`python -m pypto.execution.cli doctor|explain|plan [--json] [--strict]`（另有 `scripts/perf/execution_cli.py`）。
- **登记边界**：`plan`/`explain` 只接受与本机 probe 对齐的 snapshot（跨主机静态规划不在本切片，理由：外部 snapshot 会声明一个本机无法验证/执行的 target，违反"声明=执行"）；`CapabilitySnapshot.from_dict` 的 U1 宽松语义保留，但发现层导入先做严格检查，且宽松路径生产代码唯一调用方就是严格导入自身。
- 文档：`docs/20-planning/0013-2026-09-13-u3-discovery-commands.zh-CN.md`（控制仓 `f80b099`→`df8d0c5`→`22c5d7ad`）。

## 3. 切片 B：CUDA sm_120 / sm_121 显式原生目标

- **默认零改动**：默认仍 `sm_80`/PTX 8.0/payload v2；同一程序与上一 tip 的 `ptx_sha256`/`payload_sha256`/`content_digest` **逐字节一致**。
- **显式选择**：`arch=` / `ptx_target='sm_120'` / `TargetSpec(attributes={'arch':'sm_120'})` 三条路径；payload **升 v3**，`arch_selection{policy,source,requested_arch,ptx_target,ptx_version}` 进 payload+metadata 并被 digest 覆盖；旧 reader 读 v3 fail-closed；新 decoder 严格区分 v2（禁 selection）/v3（必须有且交叉校验）；冲突/非字符串/版本过低均拒绝；`CUDA_LOWERING_VERSION` 保持 2。
- **真机**（GamePC RTX 5080 + nvcc 13.3.73）：`sm80_ptx` / `sm120_ptx` / `sm120_cubin` 对 add f32/bf16 与 matmul f32/bf16 **逐位一致**；bf16 整图（4538 ops、371 in/51 out、packed 43200）三路径 + 两次 launch **输出 digest 相同** `c36ea307aa08e447d970268e526e835fdcbd7f58aec01ed0624f94946e3f6e3e`（前缀基线无 output digest，本轮首次给出）。
- **未做**：CUDA 上的 W8A8 **整图**（C7 图路径仅 CPU runtime，CUDA 只有 op 级 kernel——不以 op 级冒充端到端）；**PTX-JIT vs native cubin 计时腿**因游戏负载门 `perf deferred`（未记录任何数字，正在有界窗口重试）。
- **边界**：WSL2 GPU-PV 时钟不可 pin（性能须 taskset+时钟断言、UNGATED）；cubin arch-locked（`sm_80` cubin 在 sm_120 报 209，PTX JIT 才前向兼容；sm_120 需 PTX ≥8.7，8.0/8.6 报 218）；`sm_121` 仅 emitter；只把已有 PTX 编到 sm_120，不做 Blackwell 专用指令。

## 4. 验收

| 切片 | agent | 判决 |
|---|---|---|
| U3 | `72415831` | 待回（重点：输出 digest 自算一致、无私有端点/原始 env 泄漏、`--strict` 退出码、explain 与 trace 逐字段一致、plan 可重放、snapshot 篡改矩阵、CLI 退出码、全量 UT 独立复跑） |
| CUDA sm_120 | `2bda30dc` | 待回（重点：默认路径逐字节不变、三条显式路径与失败模式、`arch_selection` 真被 digest 覆盖、真机三路径逐位、整图 digest、游戏门下的 defer 处理、全量 UT） |

父方复核（T16）：collect **1994**、discovery 文件 **29 passed**、`python -m pypto.execution.cli doctor` rc=0 且输出 digest；报告侧无误伤（portable + 真 AOCL f32/bf16 + 真 runtime fallback）；合并探针 **38/38 全拒**。

## 5. 证据路径

- U3：`_meta/pypto-x/u3-discovery/{brief.zh-CN.md,raw,logs}`；验收 `_meta/pypto-x/verify-0046-u3/`。
- CUDA sm_120：`_meta/pypto-x/cuda-sm120/{validation.json,brief.zh-CN.md,sm120/,compare/,e2e/,tests/,raw}`；验收 `_meta/pypto-x/verify-0046-cuda-sm120/`。
- 批级全量：`_meta/pypto-x/batch-0046-u3-cuda-sm120/{raw,logs}`。
- 补丁：`patches/pypto-x/`（264）+ `patches/README.md`（HEAD `7378aa943`）。
