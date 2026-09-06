# PyPTO-X（PyPTO_PRO）无模型一次冒烟测试规范

更新日期：2026-09-06（Asia/Shanghai）

## 目的

冒烟测试用于确认 subagent 的 worktree、源码、基础工具链和目标资源可用。它不是模型测试，也不是性能基准。

每个 subagent 在启动后只执行一次。测试结果必须记录启动时间，便于区分排队时间、编译时间和执行时间。

## 必填参数

```text
--agent-id       唯一任务名
--started-at     ISO-8601 UTC 时间，例如 2026-09-06T08:00:00Z
--worktree       该 subagent 的绝对 worktree 路径
--target         host | qemu-aarch64 | nvidia-5080 | amd-6750gre | kunpeng-sve256
--log-dir        该 subagent 独占的日志目录
```

脚本明确拒绝以下参数：

```text
--model
--model-name
--model-path
--weights
```

因此启动协议不会指定模型，也不会隐式下载模型权重。

## 测试内容

### `host`

- 检查 worktree 是 Git 仓库；
- 记录 HEAD、branch 和工作树状态；
- 执行 `git diff --check`；
- PyPTO worktree 对 `python/pypto_pro` 做 Python 语法编译；PyPTO-Gym worktree 对 `src/pypto_gym` 做语法编译；
- 检查 `clang`、`cmake`、`python3` 等基础命令。

### `qemu-aarch64`

- 检查 AArch64 交叉编译器、sysroot 和 QEMU；
- 编译 `aarch64_feature_probe.c`；
- 使用 `QEMU_CPU=max,sve256=on` 运行；
- 记录 SVE/SVE2 和 VL 结果。

### `nvidia-5080`

- 在 WSL shell 中运行 `nvidia-smi`；
- 记录 GPU 名称、显存和驱动；
- 检查 `nvcc`/PyTorch 是否存在；
- 缺少 CUDA 工具链时返回 `BLOCKED_TOOLCHAIN`，不下载模型。

### `amd-6750gre`

- 运行 `rocminfo`/`rocm-smi`/`hipcc --version`；
- 记录 gfx target 和 wavefront 信息；
- 卡未接入时返回 `BLOCKED_DEVICE`。

### `kunpeng-sve256`

- 仅允许在 `uname -m=aarch64` 的真实机器上运行；
- 记录 HWCAP、SVE VL、编译器和 NUMA 信息；
- 不把此测试结果与 QEMU 性能混合。

## 结果格式

日志至少包含：

```text
agent_id=
started_at=
smoke_started_at=
smoke_finished_at=
worktree=
target=
head=
branch=
status=PASS|BLOCKED_TOOLCHAIN|BLOCKED_DEVICE|FAIL
commands=
notes=
```

日志目录必须由 task 独占，例如：

```text
logs/agents/cpu-scalar/20260906T080000Z/
```

## 失败处理

- `FAIL`：源码、脚本或测试本身失败；subagent 记录原因后继续其诊断，不重跑 smoke。
- `BLOCKED_TOOLCHAIN`：资源存在但缺少编译工具；记录阻塞，不伪造 PASS。
- `BLOCKED_DEVICE`：目标设备尚未接入；允许继续做静态开发。
- 协调者可以在修复后创建一个新的 task/worktree；不要在同一 subagent 内循环冒烟。

## 父 agent 等待策略

subagent 派发后，父 agent 只使用一次一小时等待：

```text
wait_timeout_seconds=3600
poll=false
```

不要每隔几秒/几分钟查询 agent 状态。完成事件、失败事件或用户新消息会打断长等待。
