# PyPTO-X：Git worktree 与 subagent 并行开发计划

更新日期：2026-09-06（Asia/Shanghai）

## 总体原则

PyPTO-X 的实现主仓是 `upstream/pypto`，公共 wheel 继续同时提供 `pypto` 与 `pypto_pro`；跨架构主入口是 Tensor frontend，Pro 保留为 Ascend expert dialect。该仓当前的 edge 快照为：

```text
34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
```

`upstream/pypto` 保持在干净的 `master` edge 快照。目标无关的 W1 可以在已审计 edge lock 上推进；stable lock 仍需真实 CANN toolkit/NPU 验证，并作为 Ascend adapter 回归门禁。所有实现分支都从一个集成分支创建。

## Worktree 拓扑

```text
upstream/pypto (master, pinned snapshot; read-only)
        │
        ▼
port/pypto-x-integration  → ../worktrees/pypto-x/integration
        │
        ├── work/target-abi          → ../worktrees/pypto-x/target-abi
        ├── work/core-ir             → ../worktrees/pypto-x/core-ir
        ├── work/tensor-core-bridge  → ../worktrees/pypto-x/tensor-core-bridge
        ├── work/ascend-adapter      → ../worktrees/pypto-x/ascend-adapter
        ├── work/cpu-scalar          → ../worktrees/pypto-x/cpu-scalar
        ├── work/cpu-vector-common   → ../worktrees/pypto-x/cpu-vector-common
        ├── work/cpu-avx2            → ../worktrees/pypto-x/cpu-avx2
        ├── work/cpu-avx512          → ../worktrees/pypto-x/cpu-avx512
        ├── work/cpu-sve256          → ../worktrees/pypto-x/cpu-sve256
        ├── work/gpu-common          → ../worktrees/pypto-x/gpu-common
        └── work/verification         → ../worktrees/pypto-x/verification
```

创建集成 worktree：

```bash
scripts/worktree/create.sh \
  --task integration \
  --branch port/pypto-x-integration \
  --base master
```

创建 task worktree：

```bash
scripts/worktree/create.sh \
  --task target-abi \
  --branch work/target-abi \
  --base port/pypto-x-integration
```

脚本会拒绝覆盖已有路径；不会执行删除、reset 或 checkout 操作。

## 任务拆分和依赖

| 波次 | task | 主要目录/文件 | 依赖 | 推荐资源 | 合并条件 |
|---|---|---|---|---|---|
| W0 | `architecture` | `docs/`、`configs/` | 无 | 本地 | RFC 和任务清单冻结 |
| W1 | `target-abi` | `pypto/target`、`pypto/compiler`、`pypto/abi` | W0 | 本地 | 目标、制品和 launch 契约可独立导入与测试 |
| W1 | `core-ir` | `pypto/core_ir`、Tensor frontend export adapter、IR dump | W0；接口以 RFC 为准 | 本地 | 同一 Tensor Kernel 可导出稳定 Core IR |
| W1 | `verification` | `pypto_x` 差分/快照 harness、离线工具 | W0 | 本地/QEMU | 无设备、无模型测试门禁可运行 |
| W2 | `tensor-core-bridge` | Tensor PIL/native 结果 → Core IR adapter | `core-ir`、`verification` | 本地 | 实际 Tensor frontend 解析一次并导出，未支持节点明确失败 |
| W2 | `ascend-adapter` | CCE backend/codegen/runtime adapter | `target-abi`、`core-ir` | CANN/NPU（若有） | 现有 Ascend 回归不退化 |
| W2 | `cpu-scalar` | CPU lowering/compiler/runtime | `target-abi`、`core-ir` | 本地 x86_64 | add/reduce/matmul 正确 |
| W3 | `cpu-vector-common` | 目标无关 CPU vector lowering | `cpu-scalar` | 本地 | vector/tail 语义与 scalar 差分通过 |
| W3 | `cpu-avx2` | AVX2 lowering | `cpu-vector-common` | 本地 | YMM 汇编、tail 和 dispatch 正确 |
| W3 | `cpu-avx512` | AVX-512 lowering | `cpu-avx2` | 本地 | ZMM/mask 汇编与能力分层正确 |
| W4 | `cpu-sve256` | AArch64 SVE lowering，不做 NEON | `cpu-avx512` | QEMU → 鲲鹏 | QEMU 功能通过；真机结果单独标记 |
| W4 | `gpu-common` | Grid/Block/Subgroup、GPU ABI | `target-abi`、`core-ir`、`cpu-sve256` | 本地/RTX 5080 | 无厂商概念的 GPU IR 稳定 |
| W4 | `cuda` | NVVM/CUDA/NVRTC backend | `gpu-common` | RTX 5080 | elementwise/softmax/matmul 通过 |
| W5 | `hip` | ROCDL/HIP/HIPRTC backend | `gpu-common`、`cuda` | AMD 6750GRE（接入后） | gfx target 和 wave32/64 正确 |
| W6 | `qwen35-08b-model` | 0.8B text harness、custom op/model wrapper | AVX-512 基线；前端过渡决策 | 按后端顺序 | 先 BF16 整网，再 W8A8-linear；9B/27B 做无权重 shape harness |

W1 的 `target-abi` 和 `core-ir` 可以并行，但必须先评审同一份接口草案；W2 之前由协调者合并并解决接口冲突。CPU 路线按 AVX2 → AVX-512 → SVE256 验证共享 vector lowering；CUDA/HIP 不应在 GPU common ABI 冻结前各自发明 runtime 接口。

## 文件所有权，避免冲突

每个 task 只修改自己的目录；跨目录变更先在 issue/RFC 中登记。推荐所有权：

```text
target-abi       : python/pypto/{target,compiler,abi}、对应单测
core-ir          : python/pypto/core_ir、frontend/core_export.py、对应单测
tensor-core-bridge: python/pypto/frontend/pil_core_bridge.py、core_export bridge、对应单测
ascend-adapter   : backend/ascend、codegen/ascend、Ascend runtime
cpu-scalar       : pypto/lowering/cpu、compiler/targets/cpu、backends/cpu
cpu-vector-common: cpu/vector（不改 Core IR）
cpu-avx2/avx512  : cpu/x86（不改 Core IR）
cpu-sve256       : cpu/aarch64/sve（不改 Core IR）
gpu-common       : lowering/gpu、backend/gpu、runtime GPU ABI
cuda/hip         : 各自 vendor codegen/runtime，禁止修改 gpu-common 契约
verification     : pypto_x 差分/快照 harness、离线验证工具（不改实现 API）
```

## Subagent 启动协议

每个 subagent 启动时必须由协调者传入以下任务字段：

```text
task_name=<唯一任务名>
worktree=<绝对路径>
branch=<分支名>
started_at=<ISO-8601 UTC 时间>
smoke_once=true
wait_timeout_seconds=3600
poll=false
```

特别约定：

- 启动调用中**不设置 `model` 或 `model_name` 字段**；采用平台默认模型。
- 不下载、不加载、不要求任何 LLM 权重；冒烟测试只能使用小型算子、IR 或硬件能力探针。
- subagent 启动后只执行一次统一冒烟测试，并把 `started_at` 原样写入日志。
- 冒烟测试失败要记录失败原因，不循环重试；由协调者决定是否开新任务修复。
- 派发完成后，父 agent 使用一次长等待（`3600s`），不做周期轮询。只有收到完成/失败事件或用户新指令时才继续处理。

如果使用本产品的 agent 工具，调用形态应遵守以下原则（示意，不包含 `model` 参数）：

```text
spawn_agent(
  task_name="cpu-scalar",
  message="started_at=2026-09-06T...Z; worktree=/home/chiro/projects/pypto/worktrees/pypto-x/cpu-scalar; ...; 先执行一次 smoke；不下载模型；完成后等待。"
)

wait_agent(timeout_ms=3600000)  # 只调用一次，不轮询
```

## 一次冒烟测试的顺序

subagent 的第一组动作固定为：

```bash
scripts/smoke/pypto_pro_smoke.sh \
  --agent-id <task-name> \
  --started-at <ISO-8601 UTC> \
  --worktree <worktree> \
  --target <host|qemu-aarch64|nvidia-5080|amd-6750gre|kunpeng-sve256> \
  --log-dir <独立日志目录>
```

测试脚本不接受 `--model`、`--model-path` 或类似参数；它只检查源码、编译器、设备能力和最小 IR，不触碰模型。

## 提交和合并协议

1. subagent 在自己的 worktree 提交一个或多个逻辑完整的 commit。
2. 提交信息包含 task 名，例如 `feat(pypto-x): add target registry`。
3. 返回：commit SHA、修改目录、smoke 日志路径、测试结果、未解决风险。
4. 协调者在 integration worktree 执行 cherry-pick/merge，并运行一次集成 smoke。
5. 发生冲突时由协调者解决；禁止 subagent 直接改别人的 worktree。
6. 合并后再决定是否删除 task worktree；删除必须是明确的人工动作。

## 跨仓库变更

PyPTO_PRO 主体在 `upstream/pypto`。涉及 `pypto-gym` 的模型集成使用独立 worktree：

```text
../worktrees/pypto-gym/<task-name>
```

不要把 `pypto` 和 `pypto-gym` 两个仓库混在同一个 Git worktree。跨仓库任务必须记录两边 commit 的配对关系和兼容的安装版本。
