# PTO-ISA CPU_SIM 基线验证报告（pto-isa-cpu-sim-baseline）

本报告基于固定快照 `upstream/pto-isa@668248ec886447a83787200786fe6f461169b701`，在本地 x86_64 KVM guest 上构建并运行
上游 CPU_SIM（PTO Tile ISA 的纯 CPU 功能仿真后端）。所有结论都有 `raw/` 下的原始日志支撑；未跑通/未覆盖的部分明确标注为
boundary 或 BLOCKED，不做通过性粉饰。

---

## 0. 结论速览

**能做什么**

- 能用上游自带入口 `python3 tests/run_cpu.py` 在纯 CPU 上**完整构建** 125 个 CPU ST 目标（Clang 22.1.8 / C++20 / Release）。
- 能**全量跑通** CPU ST：125 个用例、**1293 个 gtest 断言用例、0 失败、0 错误、4 skipped**（skipped 全部是 `ttrace`
  在非 trace 构建下的设计性跳过，见 §4.2）。
- 能跑 **gemm demo**：`max_abs_diff=1.19e-07`，耗时 2.37s（含构建）。
- 能给出**指令级 trace**：trace 模式下可记录 opcode / block_idx / sequence_id / Tile 操作数（地址、shape、layout、dtype），
  并通过 `LaunchKernelMultiCore` 落盘为 `cpu_sim_traces/<kernel_name>/launch_<id>/trace.jsonl`（本报告用新增驱动实测，见 §5）。
- 具备**多核执行上下文**：`LaunchKernelMultiCore` 每模拟核一个 CPU 工作线程，默认 4 核（`PTO_CPU_SIM_NUM_CORES`），
  `get_block_idx()`/`get_subblockid()` 按线程返回 launch 上下文；实测 2 核 trace 的 `block_idx` 分别为 0/1。
- 具备 **A2A3 / A5 两套模拟内存模型**（每线程 UB/L1/L0A/L0B/L0C，A2A3 默认），并支持容量环境变量覆盖。
- bf16 **有条件可用**：`g++ 16.1.1` + `--enable-bf16`（C++23）下 5 个代表用例 67 tests / 0 failures（spot check，见 §4.4）。

**不能做什么（本次实测 + 上游文档双重确认）**

- **不是周期精确模型**：CPU_SIM 是功能仿真，逐指令同步执行、不做时序/流水/带宽建模。上游另有独立的 costmodel/perf_sim
  （`include/pto/costmodel/perf_sim/`、`tests/run_costmodel.py`），本次未纳入。
- **不是 CANN CA / 功能模型**：CPU_SIM 只实现 PTO Tile ISA；`aclInit/aclrtSetDevice/aclrtMalloc/...` 是
  `include/pto/common/cpu_stub.hpp` 里的主机侧 stub，不含 CANN runtime、GE、AscendCL 语义。
- **不能执行 PyPTO 图或 Core IR**：CPU_SIM 的输入是 C++ PTO kernel（`TLOAD/TADD/TSTORE/...` 模板 API），
  没有 IR 解析、图调度、算子注册表或张量运行时；与 PyPTO-X 的 Tensor/Core IR 之间不存在现成通路。
- **`SYNCALL` 是空操作**，不能当 CPU 线程屏障（文档 `docs/coding/cpu_sim_zh.md:82`；代码 `cpu_stub.hpp:616-639`）。
- **`CollEngine::CCU` 未实现**，编译期 deferred-fail（文档 `cpu_sim_zh.md:81`；代码 `include/pto/cpu/comm/TGather.hpp:164-176`）。
- **GlobalData `TALLOC`/`TPUSH`/`TPOP`/`TFREE` 流程不支持**（文档 `cpu_sim_zh.md:78`；CPU 头文件中没有 TAlloc 实现，
  只有 `include/pto/npu/{a2a3,a5}/TAlloc.hpp`）。TileData 的 `TPUSH/TPOP/TFREE` 通过主机侧 FIFO 模型支持。
- **Tile 跨线程共享不安全**：`Tile` 内存访问与惰性分配都没有线程间同步（文档 `cpu_sim_zh.md:9`）。
- **trace 插桩覆盖不全**：基础设施可用（我们的探针实测产出 2970 B / 12 行 `trace.jsonl`，含 `TASSIGN/TLOAD/TADD`），
  但 `TSTORE`/`TDIV`/`TQUANT` 等入口没有插桩，上游自带 `ttrace` 用例因此 **4/4 FAIL**（§5.2）。trace 缺失不能当"指令没执行"的证据。
- **bf16 只做了 spot check**：`clang++ 22.1.8` 没有 `std::bfloat16_t`，`--enable-bf16` 必须落到 `g++ 16.1.1`；
  125 目标的 bf16 全量构建/运行未做（§4.4）。
- **默认环境有两处硬缺陷**（非本项目代码问题），必须先做一次性规避才能构建/跑 bf16，见 §3.1 / §4.3。

---

## 1. 启动协议字段（原样）

```text
task_name=pto-isa-cpu-sim-baseline
worktree=/home/chiro/projects/pypto/worktrees/pto-isa/cpu-sim-baseline
branch=work/pto-isa-cpu-sim-baseline
repository=upstream/pto-isa（快照 commit 668248ec886447a83787200786fe6f461169b701）
base=master
started_at=2026-09-10T09:20:00Z
smoke_once=true
wait_timeout_seconds=3600
poll=false
resource_lock_root=/home/chiro/projects/.resource-locks
local_heavy_policy=locked
local_heavy_runner=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh
local_min_available_mib=8192
local_safety_floor_mib=4096
local_max_cpus=6
```

一次性 smoke（`scripts/smoke/pypto_pro_smoke.sh`，只跑一次，未重跑）日志
`logs/20260910T092000Z/smoke.log`：

```text
head=668248ec886447a83787200786fe6f461169b701
branch=work/pto-isa-cpu-sim-baseline
status=FAIL
notes=worktree 中没有可识别的 pypto、pypto_pro 或 pypto_gym Python 源码
```

说明：该 smoke 脚本面向 PyPTO/PyPTO-Gym worktree，对 pto-isa worktree 的 Python 语法编译步骤无对象可编译，
因此按脚本自身逻辑返回 FAIL；worktree/Git/工具链检查本身通过。这是**预期行为**，不是 pto-isa 环境故障，
按规范不做第二次冒烟。

---

## 2. 环境与工具链（`raw/env.txt`）

| 项目 | 实测值 |
|---|---|
| 主机 | `Linux server-mini 7.1.5-arch1-2 #1 SMP PREEMPT_DYNAMIC x86_64`，`systemd-detect-virt=kvm` |
| CPU | `AMD Eng Sample: 100-000000956-50_Y`，`nproc=12`，`getconf _NPROCESSORS_ONLN=12` |
| 内存 | MemTotal 30,753,688 kB；验证开始时 MemAvailable ≈ 24.9 GiB |
| C++ 编译器 | `g++ (GCC) 16.1.1`（支持 `std::bfloat16_t`）、`clang++ 22.1.8`（不支持） |
| CMake | 4.4.2，默认 generator = Unix Makefiles |
| Python | `/home/chiro/miniforge3/bin/python3` 3.12.10（numpy 2.5.1） |
| gtest | 系统包 `gtest 1.17.0-2`，**只有共享库** `/usr/lib/libgtest{,_main}.so`，无 `.a` |
| 磁盘占用 | 主构建目录 146 MB（125 个二进制），trace 构建 11 MB |

---

## 3. T1 构建

### 3.1 attempt 1：按上游默认入口构建 → **链接失败（已定位根因）**

命令（证据 `raw/t1-attempt1-abi0-LINKFAIL.log`，关键片段 `raw/t1-attempt1-first-failure.txt`）：

```bash
python3 tests/run_cpu.py --clean --verbose --no-install --build-dir <EV>/build
```

- cmake configure **成功**；`cmake --build` 在**链接**阶段失败，首个失败目标为 **`setgetval`**：

```text
undefined reference to `testing::internal::GetBoolAssertionFailureMessage(testing::AssertionResult const&, char const*, char const*, char const*)'
make[2]: *** [testcase/setgetval/CMakeFiles/setgetval.dir/build.make:119：bin/setgetval] 错误 1
```

- 根因：`tests/cpu/st/CMakeLists.txt:109-140` 的 ABI 自动探测执行
  `nm -C <IMPORTED_LOCATION> | grep 'GetBoolAssertionFailureMessage\[abi:cxx11\]'`。系统 gtest 是**共享库**，
  裸 `nm` 对 `.so` 不打印符号表（需要 `nm -D`），grep 必然失败 → 项目强制 `PTO_GLIBCXX_USE_CXX11_ABI=0`；
  而 Arch 的 libgtest.so 是 **ABI=1**（符号带 `[abi:cxx11]`）→ 所有测试对象以 ABI=0 编译后链接失败。
  实测 CMakeCache：`PTO_GLIBCXX_USE_CXX11_ABI:STRING=0`。
- 最小复现（`scripts/diag_gtest_abi.sh`）：

```text
ABI=0 clang++ -> LINK FAIL      ABI=1 clang++ -> LINK OK
ABI=0 g++     -> LINK FAIL      ABI=1 g++     -> LINK OK
```

### 3.2 attempt 2（唯一一次规避）：显式固定 ABI=1 → 构建成功

规避方式：在 `run_cpu.py` 之前，用**与上游完全相同的参数**预配置同一 build dir，只额外固定
`-DPTO_GLIBCXX_USE_CXX11_ABI=1`（进入 CMakeCache 后，`run_cpu.py` 的 configure 不会覆盖它）：

```bash
cmake -UTEST_CASE -S tests/cpu/st -B <EV>/build \
  -DCMAKE_BUILD_TYPE=Release -DPTO_CPU_SIM_ENABLE_BF16=OFF -DPTO_CPU_SIM_TRACE_MODE=OFF \
  -DPTO_GLIBCXX_USE_CXX11_ABI=1 -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++
python3 tests/run_cpu.py --verbose --no-install --build-dir <EV>/build --xml-dir <EV>/raw/gtest-xml
```

结果：**125/125 个 gtest 目标构建成功**，`[PASS] build`，无警告级错误（仅有上游 `-Wswitch` / 未向量化告警）。
脚本：`scripts/01_build_and_run_full.sh`；日志：`raw/t2-attempt2-build-and-run-full.log`。

### 3.3 并行度控制（实测：`CMAKE_BUILD_PARALLEL_LEVEL` 无效）

`run_cpu.py` 调用 `cmake --build <dir> --parallel`（不带数字），CMake 会把**裸 `-j`** 传给 GNU make：

```text
MAKE_WRAPPER argv: -f Makefile -j clean
MAKE_WRAPPER argv: -f Makefile -j          <- 裸 -j = 无限并行
```

实测 `CMAKE_BUILD_PARALLEL_LEVEL=3` 与 `MAKEFLAGS=-j3` **都不生效**（8 个 2 秒任务仍在 ~2.07s 内全部并发完成；
串行参照 16.33s）。因此本报告用一个 make 包装器 `scripts/bin/make`：顶层调用把裸 `-j` 改写为 `-j6`，
递归子 make 继承 jobserver 令牌（全局并发仍 ≤6）。再加上 runner 的 `CPUQuota=600%` 双重保险。

### 3.4 构建资源数字（全部来自 runner 监控日志 `raw/resources.json`）

| 阶段 | 墙钟时长 | 峰值任务树 RSS | 最低 MemAvailable | 峰值 CPU（6 核 quota 占比） | 峰值进程数 |
|---|---:|---:|---:|---:|---:|
| attempt 1（失败） | 8s | 1229 MiB | 23,594 MiB | 51.8% | 18 |
| attempt 2（构建成功，含被中止的首轮全量跑） | 134s | **1758 MiB** | 23,381 MiB | 83.0% | 20 |
| 全量跑 + trace + bf16 + demo（round1） | 44s | 870 MiB | 23,552 MiB | 33.1% | 13 |

上限约束：`--max-cpus 6 --memory-max-mib 8192`，实际峰值 RSS 仅为上限的 ~21%，全程 `MemAvailable` 未低于 23.3 GiB，
memory PSI `avg10` ≤ 0.09，**未触发任何限流或安全停止**。

---

## 4. T2 全量 CPU ST

### 4.1 全量结果

命令（`scripts/01b_run_full_with_deps.sh`，经 runner 持锁运行）：

```bash
PYTHONPATH=<EV>/pydeps python3 tests/run_cpu.py --verbose --no-install \
  --build-dir <EV>/build --xml-dir <EV>/raw/gtest-xml
```

| 指标 | 数值 |
|---|---:|
| 用例（gtest 二进制） | **125 / 125 全部 PASS** |
| gtest 断言用例 | **1293** |
| 失败 | **0** |
| 错误 | **0** |
| 跳过 | **4**（全部为 `ttrace`，非 trace 构建下的设计性跳过） |
| 测试执行总计（run_cpu.py SUMMARY 的 TOTAL 行） | **1.75 s** |
| 单用例耗时量级 | 绝大多数 2–30 ms；最慢 `twait` 818 ms，其次 `tpushpop`/`tmatmul` 量级 10–100 ms |
| 单用例 XML 统计 | 见 `raw/summary.json`（`full_run_bf16_off.per_case`） |

**总计分组结论：125/125 全部跑完，无未跑批次、无失败清单。**

### 4.2 代表用例单跑（`raw/singles/`，脚本 `scripts/12_singles.sh`）

| 用例 | 覆盖点 | 命令 | 结果 |
|---|---|---|---|
| `tadd` | 基础向量 | `-t tadd` | 5 tests PASSED |
| `tadd` | `-g` 过滤演示 | `-t tadd -g 'TADDTest.case_float_*'` | 1 test PASSED |
| `tmatmul` | 矩阵/仿真 Cube 路径 | `-t tmatmul` | 10 tests PASSED |
| `tci` | 常量生成 | `-t tci` | 4 tests PASSED |
| `tcmps` | 比较语义 | `-t tcmps` | 20 tests PASSED |
| `targreduceop` | arg-reduce | `-t targreduceop` | 24 tests PASSED |
| `ttrace` | trace（非 trace 构建对照） | `-t ttrace` | 4 tests **SKIPPED**，原因原文 `PTO_CPU_SIM_TRACE_MODE is disabled for this build.` |

### 4.3 环境缺陷 2：`gen_data.py` 缺 `en_dtypes`/`ml_dtypes`

首轮全量跑在字母序 `tcvt` 处整体中止（`run_cpu.py` 的批量模式遇到 `gen_data` 异常会直接抛出、终止整轮）：

```text
File ".../build/gen_data.py", line 15, in <module>
    import en_dtypes as end
ModuleNotFoundError: No module named 'en_dtypes'
```

受影响用例：`tcvt`、`tmatmul_mx`、`tmatmul_mx_nddn`、`tstore`、`ttrans`（import `en_dtypes`）；
`tmatmul_mx`、`tmatmul_mx_nddn`、`tpushpop_fixpipe`、`tstore_acc2gm`（import `ml_dtypes`）。
规避：把依赖装进**证据目录**（不动系统环境），并加进 `PYTHONPATH`：

```bash
python3 -m pip install --target <EV>/pydeps --no-cache-dir en_dtypes ml_dtypes   # en_dtypes 0.0.4 / ml_dtypes 0.6.0
```

---

### 4.4 bf16 覆盖（spot check，非全量）

- 上游 `AGENTS.md:217`：CPU simulator 的 bfloat16 需要 **GCC≥14**。本机实测：`clang++ 22.1.8 -std=c++23` 报
  `no type named 'bfloat16_t' in namespace 'std'`，`g++ 16.1.1` 通过；`run_cpu.py --enable-bf16` 的自动探测也选中 `/usr/bin/g++`。
- bf16 用例是用 `#ifdef CPU_SIM_BFLOAT_ENABLED` 编译期开关控制的（例如 `tests/cpu/st/testcase/tadd/main.cpp:113-115`），
  非 bf16 构建里这些用例**根本不存在**（不是 skip）。静态统计（精确正则匹配 TEST/TEST_F 名）：**47 个用例目录里共 88 个 bf16 测试**（125 个 main.cpp 共 1187 个测试声明）。
- 做法：用 g++ 建一个 `PTO_CPU_SIM_ENABLE_BF16=ON`（C++23）的独立 build dir，只构建 5 个代表目标，
  再以 `--enable-bf16` 运行（脚本 `scripts/04_bf16.sh`，日志 `raw/t4-bf16.log`）：

| 用例 | bf16 OFF | bf16 ON | 失败 |
|---|---:|---:|---:|
| `tadd` | 5 | **6** | 0 |
| `tmatmul` | 10 | **12**（`case_bf16_1`、`case_bf16_bias_1`） | 0 |
| `tci` | 4 | 4 | 0 |
| `tcmps` | 20 | **21**（`case_bf16_32x32_32x32_32x32_GE`） | 0 |
| `targreduceop` | 24 | 24 | 0 |
| **合计** | — | **67 tests / 0 failures** | 0 |

- **boundary**：没有做 125 目标的 bf16 全量构建与运行，因此本报告**不声明**"bf16 全量通过"。

---

## 5. T3 指令 Trace

### 5.1 机制（文档 + 代码）

- 构建期开关：`PTO_CPU_SIM_TRACE_MODE`（`run_cpu.py --trace-mode`）；运行期：`PTO_CPU_SIM_TRACE_ENABLE`、
  `PTO_CPU_SIM_TRACE_DIR`（默认 `cpu_sim_traces`，相对进程 cwd）。
- 记录内容：`block_idx`、`sequence_id`、`opcode`、输入/输出 Tile（地址、shape、layout、dtype）、标量输入。
- 落盘路径：`LaunchKernelMultiCore` 把各核 trace chunk 合并写入
  `<trace_root>/<kernel_name>/launch_<id>/trace.jsonl`（`cpu_stub.hpp:366-426`、`CreateKernelTraceDir`）。

### 5.2 上游自带 `ttrace` 用例：trace 模式下 **4/4 FAIL**（本快照缺陷，已用上游入口复现）

用上游入口复现（`scripts/03b_trace_probe.sh`，日志 `raw/t3b-trace-probe.log`）：

```bash
python3 tests/run_cpu.py --trace-mode --no-install -t ttrace --build-dir <EV>/build-trace-upstream
```

失败原文（`raw/t3-trace.log` / `raw/t3b-trace-probe.log` 一致）：

```text
main.cpp:74  trace.size() = 1               期望 2        （TADD + TDIV，只记录到 TADD）
main.cpp:122 trace[0].output_tiles.size()=1 期望 2        （TROWARGMAX 双输出只记录 1 个）
main.cpp:161 trace.size() = 0               期望 2        （TQUANT 无记录）
main.cpp:209 未找到 "opcode":"TSTORE"
```

根因（代码级）：trace 记录由 `include/pto/common/pto_instr.hpp` 的 `MAP_INSTR_IMPL*` 宏在公开指令入口处插桩，
但本快照**插桩覆盖不全**：

```text
TADD(172 行) / TLOAD(255 行) / TROWARGMAX(1750 行)  -> 走 MAP_INSTR_IMPL，有 trace
TSTORE(355 行) -> 直接调 TSTORE_IMPL，无 trace scope
TDIV(491 行)   -> 直接调 TDIV_IMPL，无 trace scope
TQUANT(2587+)  -> 无 trace scope
TROWARGMAX     -> MAP_INSTR_IMPL(..., dst, src, tmp) 只声明 1 个输出 Tile（测试期望 2 个）
该文件内 MAP_INSTR_IMPL* 调用点 124 处，而 PTO_INST 公开指令定义 259 处
```

结论：**trace 基础设施可用，但上游 trace 测试在本快照为红**；这是快照自身的插桩覆盖问题，
在 trace 模式下与我们的环境无关（同一次构建里 `tadd` 是 PASS 的）。§5.3 的独立探针也复现了同一现象
（记录到 `TASSIGN/TLOAD/TADD`，但缺失 `TSTORE`）。

### 5.3 PyPTO-X 探针：实测产出 `trace.jsonl`（新增驱动）

因为本快照 CPU ST 树里**没有任何用例调用 `LaunchKernelMultiCore`**（全局 grep 只有定义、无调用者），
新增一个最小驱动 `tools/pypto_x_validation/cpu_sim_trace_probe/`（本文件为新增，未修改任何上游源码）：

```bash
cmake -S tools/pypto_x_validation/cpu_sim_trace_probe -B <EV>/build-trace-probe \
  -DPTO_CPU_SIM_TRACE_MODE=ON -DCMAKE_CXX_COMPILER=/usr/bin/clang++
cmake --build <EV>/build-trace-probe
PTO_CPU_SIM_TRACE_DIR=<EV>/raw/traces-probe <EV>/build-trace-probe/pto_cpu_sim_trace_probe 2
```

【实测结论】

```text
cores=2 trace_records_in_main_thread=0 result=PASS
2970	<EV>/raw/traces-probe/pypto_x_trace_probe/launch_0/trace.jsonl
--- bytes=2970 lines=12
--- 唯一 opcode： TASSIGN ×6、TLOAD ×4、TADD ×2
--- block_idx：   0 ×6、1 ×6
```

`trace.jsonl` 前 3 行**原文**：

```json
{"block_idx":0,"sequence_id":0,"opcode":"TASSIGN","input_tiles":[],"scalar_inputs":[{"dtype":"uint64","value":"0"}],"output_tiles":[{"address":"0x7f42e9315010","shape":[4,32],"layout":"ND","dtype":"float32"}]}
{"block_idx":0,"sequence_id":1,"opcode":"TASSIGN","input_tiles":[],"scalar_inputs":[{"dtype":"uint64","value":"4096"}],"output_tiles":[{"address":"0x7f42e9316010","shape":[4,32],"layout":"ND","dtype":"float32"}]}
{"block_idx":0,"sequence_id":2,"opcode":"TASSIGN","input_tiles":[],"scalar_inputs":[{"dtype":"uint64","value":"8192"}],"output_tiles":[{"address":"0x7f42e9317010","shape":[4,32],"layout":"ND","dtype":"float32"}]}
```

要点：

- 路径、文件名、JSON Lines 结构、`block_idx`/`sequence_id`/`opcode`/输入输出 Tile 元数据**全部与文档一致**；
  本机 2 核运行时两个核的 trace chunk 被合并进同一个 `launch_0/trace.jsonl`（block_idx 0 与 1 各 6 条）。
- 探针只声明 `output_tiles` 的单输出计数，`TASSIGN` 也被记录（说明记录粒度是**指令级**，包含地址绑定类指令）。
- **旁证**：探针 kernel 里显式调用了 `TSTORE`，但 12 条记录中没有 `TSTORE` —— 与 §5.2 的插桩缺口结论完全一致
  （同一个原因、两个独立观测）。因此 trace 记录**不能**当作"某指令是否执行"的完备证据。

---

## 6. T4 Demo（gemm）

命令 `python3 tests/run_cpu.py --demo gemm --verbose --no-install`（20 分钟硬预算，实际 2.37s）。原文：

```text
gemm_demo: M=32 K=16 N=32
max_abs_diff=1.19209e-07
perf: avg_ms=0.22625 matmul_flops=32768 gflops=0.144831
[PASS] demo: gemm (2.37s)
```

`flash_attention_demo` / `mla_attention_demo` 未跑（时间预算让给 trace/bf16 证据）。
demo 构建目录 `demos/cpu/gemm_demo/build` 在脚本结束时删除，worktree 不留构建产物。

---

## 7. 能力边界总表（能 / 不能）

| 维度 | 结论 | 证据 |
|---|---|---|
| PTO Tile ISA 指令功能仿真 | **能**：1293 用例 0 失败 | §4.1 |
| A2A3 / A5 模拟内存模型（UB/L1/L0A/L0B/L0C） | **能**：每线程独立区域 + 容量环境变量覆盖 | `NPUMemoryModel.hpp:41-88`、`cpu_sim_zh.md:29-53` |
| 多核执行上下文 | **能**：`LaunchKernelMultiCore`，默认 4 核 | `cpu_sim_zh.md:85-89`；探针实测 2 核 block_idx=0/1 |
| 指令级 trace（记录） | **能**（基础设施），但插桩覆盖不全：`ttrace` 4/4 FAIL | §5.1 / §5.2 |
| 文件级 trace（`trace.jsonl`） | **能**：探针实测 2 核 → `.../launch_0/trace.jsonl`（2970 B / 12 行）；上游 CPU ST 无调用者 | §5.3 |
| bf16 | **有条件能**：必须 GCC≥14；clang++ 不可用。5 个代表用例 67 tests / 0 fail | §4.4 / §8 / 上游 `AGENTS.md:217` |
| 周期精确 / 时序 / 带宽 | **不能**（另有 costmodel/perf_sim，本次未涉及） | `include/pto/costmodel/perf_sim/` |
| CANN CA / AscendCL 功能模型 | **不能**：`acl*` 为主机 stub | `cpu_stub.hpp` |
| PyPTO 图 / Core IR 执行 | **不能**：无 IR 前端 | 无对应入口 |
| `SYNCALL`（含 Soft/workspace 形式） | **不能**：空操作，非屏障 | `cpu_sim_zh.md:82`；`cpu_stub.hpp:616-639` |
| `CollEngine::CCU`（TGATHER/TSCATTER/TBROADCAST/TREDUCE CCU 路径） | **不能**：编译期 deferred-fail | `cpu_sim_zh.md:81`；`cpu/comm/TGather.hpp:164-176` |
| GlobalData `TALLOC`/`TPUSH`/`TPOP`/`TFREE` | **不能** | `cpu_sim_zh.md:78`；CPU 头无 TAlloc |
| TileData `TPUSH/TPOP/TFREE`（主机 FIFO 模型） | **能** | `cpu_sim_zh.md:78` |
| Tile 跨线程共享 | **不能**：无同步保证 | `cpu_sim_zh.md:9` |
| `__PTO_AUTO__` 惰性分配 | **能**（主机内存，非片上地址） | `cpu_sim_zh.md:55-70` |
| 多 NPU / URMA 通信套件（`tests/cpu/comm/st`） | **未覆盖**：需 2/4/8 NPU 与 URMA opapi，不属 run_cpu.py 入口 | `tests/run_comm_test.sh` |

---

## 8. 要在 PyPTO-X 里用 CPU_SIM，需要哪些前置

1. **IR→PTO 桥（最关键）**：CPU_SIM 只吃 C++ PTO kernel（`TLOAD/TADD/TSTORE` 模板 API + Tile/GlobalTensor 类型）。
   PyPTO-X 的 Tensor/Core IR 需要先 lower 成 PTO C++ 源或等价的 PTO 指令序列，且要能把 PyPTO 的 tensor/外部 buffer
   绑定翻译成 `GlobalTensor`/`TASSIGN` 地址模型（M1J/M1K 的 external byte-view contract 正好是这一层的输入）。
2. **数据/内存绑定约定**：要么用 `TASSIGN` 显式绑定模拟 UB/L1/L0x 地址，要么定义 `__PTO_AUTO__` 走主机惰性分配；
   两者语义不同（前者模拟片上地址，后者不模拟）。
3. **多核语义要对齐**：`LaunchKernelMultiCore` 提供 block/subblock 上下文，但 `SYNCALL` 是空操作，
   跨核数据交换必须用 CPU_SIM 已实现的同步/通信原语，否则需要自己在外层做 join。
4. **bf16 路径必须用 GCC≥14**：本机固定 `--cxx g++`（或在 CI 里显式选 g++-14+），否则 bf16 用例根本不会被编译进去。
5. **两个环境缺陷要在脚本里固化规避**：gtest ABI=1 预配置（§3.1/§3.2）；`tests` 包遮蔽的 shim（§4.3/§8 注）。
6. **trace 只能当"部分覆盖的调试手段"**：插桩覆盖不全（§5.2），不能把 trace 缺失当成"该指令没执行"的证据。
7. **不要把它当性能模型**：需要周期/带宽结论时应改用 costmodel/perf_sim 或真机。

---

## 9. 证据清单与复跑

**提交**：worktree 分支 `work/pto-isa-cpu-sim-baseline` 新增一个 commit
`69249f6061af1697edd9105e644951882c05bfc2`（`test(pypto-x): add CPU_SIM trace probe driver for validation`），
只包含新增文件 `tools/pypto_x_validation/cpu_sim_trace_probe/{main.cpp,CMakeLists.txt}`；
提交后 worktree `git status --short` 为空，`upstream/pto-isa` 主仓 `git status --short` 为空且 HEAD 仍为
`668248ec886447a83787200786fe6f461169b701`（未被写入任何构建产物）。
`<EV>/build*` 之外的构建目录全部位于证据目录内，未进入主仓。

```text
<EV>/validation.json                     结构化结果
<EV>/brief.zh-CN.md                      本文件
<EV>/raw/env.txt                         环境与工具链
<EV>/raw/t1-attempt1-abi0-LINKFAIL.log   attempt1 完整构建日志（ABI=0 链接失败）
<EV>/raw/t1-attempt1-first-failure.txt   首个失败目标与首条错误摘录
<EV>/raw/t2-attempt2-build-and-run-full.log  attempt2 构建成功 + 首轮全量跑（在 tcvt 因缺依赖中止）
<EV>/raw/t2-run-full-with-deps.log       最终全量 CPU ST 运行（125/125 PASS）
<EV>/raw/gtest-xml/*.xml                 125 份 gtest XML（计数来源）
<EV>/raw/singles/*.log                   代表用例单跑原文
<EV>/raw/t3-trace.log                    第一次 trace 阶段（ttrace 4/4 FAIL）
<EV>/raw/t3b-trace-probe.log             上游入口 trace 复现 + 探针首次构建（含编译错误）
<EV>/raw/t3c-trace-probe.log             探针最终构建与运行
<EV>/raw/t4-bf16.log                     bf16 代表用例
<EV>/raw/t5-demo-gemm.log                gemm demo
<EV>/raw/t1-cmake-parallel-probe.txt      CMake 裸 -j / 环境变量无效 / 包装器生效的实测记录
<EV>/raw/resource-usage/*.log             runner 监控日志副本（原始位置 ../_meta/pypto-x/resource-usage/pto-isa-cpu-sim-baseline/）
<EV>/raw/summary.json                    用例统计（脚本生成）
<EV>/raw/resources.json                  runner 资源数字（脚本生成）
<EV>/raw/chain-phases*.log               各阶段起止与返回码
<EV>/raw/lock-retry-*.log                75/69 等待重试记录（未绕过包装器）
<EV>/scripts/                            全部可复跑脚本 + make 并行度包装器 + pyshim
<EV>/pydeps/                             证据目录内的 en_dtypes / ml_dtypes
```

复跑顺序（全部 heavy 步骤经 `run_local_heavy.sh`）：

```bash
bash <EV>/scripts/00_env.sh
bash <EV>/scripts/diag_gtest_abi.sh
bash <EV>/scripts/run_with_lock_retry.sh 2400 <EV>/raw/lock-retry.jsonl -- \
  <runner> --task pto-isa-cpu-sim-baseline --agent pto-isa-cpu-sim-baseline \
  --max-cpus 6 --memory-max-mib 8192 -- bash <EV>/scripts/01_build_and_run_full.sh
bash <EV>/scripts/12_singles.sh
bash <EV>/scripts/06_analyze.py && bash <EV>/scripts/07_resources.py
```

---

## 10. 未解决风险与 boundary

1. **trace 插桩覆盖不全是上游快照问题**：`ttrace` 4/4 FAIL 会让"trace 模式全量回归"永远是红的；
   在 PyPTO-X 侧要么只用我们自己的探针，要么等上游补齐插桩。
2. **bf16 只做了 5 个代表用例的 spot check**：没有做 125 目标的 bf16 全量构建/运行，因此
   "bf16 全量通过"不成立（本次不声明）。
3. **`--clean` 与 ABI 规避天然冲突**：一旦有人用 `run_cpu.py --clean` 重建，ABI=0 误判会复现，
   必须走 `scripts/01_build_and_run_full.sh` 里的预配置流程。
4. **未覆盖**：costmodel/perf_sim、`tests/cpu/comm/st`（URMA/多 NPU）、NPU 后端、Ascend 硬件相关路径。
5. **性能数字不可用**：本机是 KVM guest、无 cpufreq 锁频，demo 的 `gflops=0.145` 只是功能烟测的副产品，
   按 `docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` 不得作为性能门槛。
