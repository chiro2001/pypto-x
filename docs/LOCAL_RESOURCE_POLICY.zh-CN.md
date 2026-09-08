# PyPTO-X 本机重任务资源锁与监控策略

更新日期：2026-09-09（Asia/Shanghai）

## 权威协议

跨项目锁由 `/home/chiro/projects/.resource-locks/README.md` 定义：

- `local` 独占本地主机 heavy 计算；
- `gamepc` 独占远程 GamePC 的 heavy CPU/host-memory 计算；RTX 5080 GPU 由 PyPTO-X 独占，不由该锁表示；
- `status` 只显示状态，只有 `resource-lock run` 成功才算取得资源；
- 返回 75 表示 BUSY，返回 69 表示准入资源不足；两者都必须等待，不能裸跑；
- 禁止抢占 owner、终止其他项目持有者、删除 `.pid` 或 `.guard`；
- heavy 目标必须保持前台受监督，不能 daemonize 或派生脱离监督的后台重任务。

## 哪些命令必须申请 `local`

以下任一条件成立即按 heavy 处理：

- 全量 `python/tests/ut/pypto_x` pytest；
- 多用例 QEMU/native 编译与执行；
- Qwen 真实大 shape lowering、artifact compile 或生成大 iteration plan；
- 并行 C/C++/wheel 构建；
- 预计使用本机至少一半 CPU（当前为 6/12）或 4 GiB 内存；
- 无法可靠估计资源，但可能生成与 tensor `numel` 成正比的 Python 对象。

Git/`rg`、文档/YAML/AST 检查、单个轻量能力探针和明确的小型 focused test 通常不需要锁。拿不准时按 heavy 处理。

## 统一入口

```bash
scripts/resource/run_local_heavy.sh \
  --task qwen35-m1e-full \
  --agent root-or-subagent-id \
  -- env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python \
  pytest -q --confcutdir=python/tests/ut/pypto_x python/tests/ut/pypto_x
```

包装器先读取全局锁和当前资源，再调用：

```text
/home/chiro/projects/.resource-locks/resource-lock run local pypto-x <task> <agent> ...
```

默认策略：

| 项目 | 默认值 |
|---|---:|
| 启动最低 `MemAvailable` | 8192 MiB |
| 系统安全余量 | 4096 MiB |
| 最大 CPU | 6（当前 12 个 online CPU 的一半） |
| `MemoryMax` | 启动时 `MemAvailable - 4096 MiB`，且不超过 20480 MiB |
| `MemoryHigh` | `MemoryMax` 的 85% |
| task swap | `MemorySwapMax=0` |
| 采样周期 | 2 秒 |
| 低内存停止 | 连续 3 次低于 4096 MiB |

包装器同时设置 CPU affinity、CPU quota、`OMP/BLAS/MAX_JOBS/CMake/pytest-xdist` 并行度，并把命令放入 user systemd scope。未能建立 cgroup 时 fail closed，不退化为无保护执行。

## 运行时监督

`monitor_local_heavy.py` 持续记录：

- 全机 `MemAvailable`；
- 目标进程树 RSS、CPU capacity 使用率和进程数；
- load average、runnable 数；
- CPU/memory PSI `avg10`。

低内存或持续极端 CPU pressure 达到门槛时，supervisor 先向目标进程组发送 TERM，10 秒后仍未退出才发送 KILL，并返回 70。它不终止其他项目或系统进程。cgroup `MemoryMax` 用于阻挡两次采样之间的突发分配。

日志默认写到：

```text
../worktrees/_meta/pypto-x/resource-usage/<task>/<UTC timestamp>.log
```

## Subagent 规则

每个实现任务的启动消息必须显式携带：

```text
resource_lock_root=/home/chiro/projects/.resource-locks
local_heavy_policy=locked
local_heavy_runner=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh
local_min_available_mib=8192
local_safety_floor_mib=4096
local_max_cpus=6
```

subagent 可运行一次轻量统一 smoke；其后的 full tests、大 shape 探测和并行编译必须经 heavy runner。不要用 `python -u -` 或 heredoc 直接承载 heavy 大 shape 工作；先保存可审计 driver，再在锁内运行。锁忙或准入失败时报告并等待协调者。

## GamePC CPU 锁与独占 GPU

RTX 5080 GPU 当前由 PyPTO-X 独占。GPU-only 能力探测、driver 调用和低 host 开销的 GPU kernel 执行不申请 `gamepc`，即使另一项目正持有该锁做远端 CPU 工作也可并行；仍需记录显存、compute process 和 GPU 错误。

CUDA host 编译、并行构建、大量 CPU 数据准备或其他明显占用 GamePC CPU/host memory 的阶段必须申请 `gamepc`，且只在该 heavy 阶段持续持有。若本机同时做 heavy 工作才申请 `local,gamepc`。受锁保护的 SSH 必须同步等待，不能后台化后提前释放。
