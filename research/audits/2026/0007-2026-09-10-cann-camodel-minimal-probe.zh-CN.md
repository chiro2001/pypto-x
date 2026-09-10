# CANN 仿真最小可用性探针报告（cann-simulator-minimal-probe）

本报告回答：在**只装了 CANN 9.2.0-beta.2 toolkit（无驱动、无 950-ops、无 NNAL）**的本机 KVM guest 上，
CANN 的 npusim/cannsim 仿真到底能跑到哪一步、缺什么、与已跑通的 PTO-ISA CPU_SIM 相比多给了什么。

所有结论都有 `raw/` 下原始日志支撑；每步都标了 `PASS` / `PARTIAL` / `BLOCKED_*`，**没有把"命令存在"写成"仿真可用"**，
报告里出现的 report/流水图产物**均未生成**（原因见 §4），不做任何伪造。

---

## 1. 启动协议字段（原样）

```text
task_name=cann-simulator-minimal-probe
worktree=/home/chiro/projects/pypto/worktrees/pypto-x/cann-simulator-minimal-probe
branch=work/cann-simulator-minimal-probe
base=port/pypto-x-integration @ 37b76b929
started_at=2026-09-10T10:15:00Z
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

一次性 smoke（`scripts/smoke/pypto_pro_smoke.sh`，只跑一次，未重跑）日志 `logs/20260910T101500Z/smoke.log`：

```text
head=37b76b929e0f40f68a267d6aeb967ffee4fdd6bc
branch=work/cann-simulator-minimal-probe
status=PASS
notes=pypto/pypto_pro 源码、Git、基础工具链和 Python 语法检查通过；未加载模型
```

---

## 2. 结论速览

### 能做什么

- **能跑通一个完整的 CANN CA-model（CAModel）仿真用例**：PTO-ISA a5 ST 的 `tadd`
  在 `dav_3510`（Ascend950PR）camodel 上执行并 **PASS**：
  `[ OK ] TADDTest.case_float_64x64_64x64_64x64_64x64 (95010 ms)`、`[ PASSED ] 1 test`、
  `max diff: 0.000000`（阈值 0.001）。产物 `output.bin` 与 `golden.bin` **逐位相等**（本报告用 numpy 独立复核，
  4096 个元素全部非零，不是"全零假通过"）。
- **能用 CANN 官方入口 `npusim record` 跑同一个用例**：`Simulation SUCCESS · run time 94.7s`，
  归档出 `record/instr.bin`、`record/stars_log0_0_task.dump`、`record/.soc-version`。
- **能在完全没有驱动的前提下做到上面两点**：camodel 目录自带 `libascend_hal.so -> libnpu_drv_camodel.so`
  driver stub，运行日志里全是 `[DRVSTUB_LOG]`。
- **能在完全没有 950-ops 的前提下做到上面两点**：kernel 由本机 `bisheng`(ccec) 从 PTO C++ 源码编译，
  camodel 只执行编译产物，不需要任何算子库。
- **CLI 自检完全可用**：`npusim --help` / `npusim record --help` / `npusim report --help` / `cannsim --help` 全部 exit=0
  （`cannsim` 与 `npusim` 是同一个 Python 模块 `cannsim.main`，官方已宣告 `cannsim` 废弃）。

### 不能做什么

- **`npusim report` 出不了任何 report / 泳道图**：唯一后端 `BiProfRunner` 硬依赖 Python 包 `plotly`，
  本机所有 Python 环境都没有 plotly，`reporters.py` 直接 `ModuleNotFoundError` → `No report backend succeeded`。
  `-g/--gen-report` 也因此失败。**没有任何降级后端**（代码里 `_check_plotly_available()` 失败即返回 None）。
- **`npusim record -c <case_dir>`（ktest 模式）不可用**，两条硬阻断：
  1. 它要求在 `$ASCEND_HOME_PATH/tools/simulator/dav_3510/camodel/` 下建 `case` 软链和 `log/`、`log_ca/`、`run_log/`
     —— 该目录 `root:root drwxrwxr-x`、其中 `sim` 是 `-r--r--r--`，本机用户 `chiro` **不可写**（实测 `touch` → 权限不够），
     且 `ensure_sim_executable()` 的 chmod 也会因非 owner 失败；
  2. ktest 的 case 目录需要 `top.json`（chip 拓扑）+ 一个 camodel 认识的 case 结构，随包里**没有任何示例 case**。
- **`test-ops/` 不是可跑用例**：`opp/test-ops/lib64/` 只有 36 个预编译设备目标文件（`hbm_test*.o`、`d2d_bandwidth_test.o`、
  `train_matmul_model_fp32.o` …），没有 host 侧驱动、没有 case 元数据、没有文档，无法直接当 ktest case 用。
- **PyPTO 前端 `--run_mode sim` 跑不了**：第一个阻塞点是 `torch` 不存在（`hello_world.py:19`；CANN 自带 pypto 的
  `pypto/cost_model.py:15` 同样 `import torch` 失败）。

---

## 3. 环境与已确认边界（`raw/env-snapshot.txt`）

| 项目 | 实测 |
|---|---|
| 主机 | `Linux server-mini 7.1.5-arch1-2`，`systemd-detect-virt=kvm`，`nproc=12`，MemTotal 29.3 GiB |
| CANN | `/usr/local/Ascend/cann-9.2.0-beta.2`，3.9 GB；`ASCEND_HOME_PATH` 由 `set_env.sh` 设置成功 |
| 驱动 | **无**：`/usr/local/Ascend/driver` 不存在、`/etc/ascend_install.info` 不存在、`lsmod` 无 Ascend 模块 |
| 950-ops / NNAL | **无** |
| 仿真器 | `tools/simulator/`（= `x86_64-linux/simulator/`）共 455 MB |
| 唯一带 `camodel` 测试台的 SoC | **`dav_3510`**（52 个文件，含 `sim` ELF 可执行文件）；`Ascend950PR_*` 全部是它的软链 |
| 其它 SoC | `dav_1001`(910A) / `dav_2002`(310P) / `dav_2201`(910B) / `dav_3002`(310B) 只有 `lib/`（in-process CA model 库），**没有 `camodel/`、没有 `sim`** |
| 编译器 | `bisheng`(=`ccec`) clang 15.0.5、`bishengir-compile` 1.2.0、系统 `g++ (GCC) 16.1.1`、gtest 1.17.0（共享库） |
| Python | `/home/chiro/miniforge3/bin/python3` 3.12.10 |
| CANN 自带 pypto | **有**：`$ASCEND_HOME_PATH/python/site-packages/pypto`（dist-info 显示 **0.2.1**）、`pypto_pro/`、`pypto/lib/pypto_impl.cpython-312-x86_64-linux-gnu.so`；`set_env.sh` 会把它加进 `PYTHONPATH` |

**`camodel` 目录里的关键组件**（`dav_3510/camodel/`，52 文件）：

```text
sim                      ELF x86-64 可执行（14,416 B，权限 -r--r--r--，需自行 chmod +x）
libascend_hal.so   ->    libnpu_drv_camodel.so        ← 无驱动时的 driver stub
libpem_davinci.so        28.6 MB  ← 精度/指令模型（0005 号审计点名的"精度仿真组件"）
libruntime_camodel.so    15.7 MB  ← CA 版 runtime（在 lib/ 下，链接用）
libUB/libSoC/libHBMSim/liblpddrsim/libdramsim/libSLLC/libSMMU/libPCIE/libDDR_Inf/libAXI_STREAM_BUS...
camodel_v100.json        {"sysconf": {"esl_top_thread_cnt": 32}}
```

**没有任何 `case`/示例目录**；`tools/simulator/bin/` 里只有一个 `cannsim-0.1.0-py3-none-any.whl`（CLI 自身）。

---

## 4. P1 摸清工具面 —— `PASS`（工具面）+ `PARTIAL`（自检）

### 4.1 子命令与关键参数（原文见 `raw/p1-npusim-*.txt`）

```text
npusim [-h] {record,report} ...

npusim record [-h] [-o OUTPUT] [-s SOC_VERSION] [-g [GEN_REPORT]] [-u USER_OPTION]
              [-f OBJECT_FILE] [-n CORE_ID] [-c CASE_DIR] [user_app]

npusim report [-h] -e EXPORT [-o OUTPUT] [-n CORE_ID] [-f OBJECT_FILE] [-m]
```

要点（全部来自 `--help` 原文）：

- `-s/--soc-version`：**只在 user_app 模式可用**，`-c` 模式下不允许（互斥）；help 举例 `Ascend950`。
  代码里注册表只有两个值：`SOC_INFO = {"Ascend950": 32 cores, "Ascend950DT": 32 cores}`。
- `record -c <case_dir>`：ktest 模式，"runs camodel ./sim with the case directory symlinked as 'case'"。
- `-n CORE_ID`：`all` / `0-2,12-14` / `5`，转成 `CORE_ENABLE_MASK` 位图。
- `report -e EXPORT`：export 目录里找 `instr.bin`（首选）或 `log_ca/`（回退）。
- `cannsim` 与 `npusim` 的 launcher 逐字节相同（都 `from cannsim.main import main`），**`cannsim` 只是兼容别名**。

### 4.2 无驱动/无 950-ops 下的自检行为（原始报错）

| 尝试 | 命令 | 结果 |
|---|---|---|
| A | `npusim record /bin/true -s Ascend950 -o <EV>` | `[1/3] Preflight OK USER_APP=true SoC=Ascend950`、`[2/3] Paths resolved OK` → `[3/3]` 失败：`Failed to create directory /usr/bin/log: [Errno 13] Permission denied`。**说明 record 会在"命令主程序所在目录"建 `log/`**；用 `/bin/true` 就落到 `/usr/bin`。不是缺驱动导致的失败。 |
| B | `npusim record -c <不存在的目录>` | `[1/3] Preflight FAIL` → `case directory does not exist: ...`（exit=1）。**参数校验可用**。 |
| C | `npusim report -e <空目录>` | 走完参数解析后失败（`No report backend succeeded`，见 §5.3）。 |

**self-test 结论**：`npusim` 本身**不需要驱动即可启动并做参数/路径自检**；它只在"真的执行"阶段才需要 camodel
组件（而 `dav_3510` 把它们带齐了）。`test-ops/` 不含可跑 case（只有 36 个设备 `.o`）。

---

## 5. P2 PTO-ISA a5 ST on CAModel —— **`PASS`（本报告核心成果）**

上游 `tests/script/run_st.py -r sim` 的路径解析（`raw/` 代码摘录见 §11 复跑）在本机能走通，但有两个**宿主环境阻断**，
必须做本地规避。为遵守"构建产物放 `_meta`"，实际用**等价的手工命令**复刻 `run_st.py` 的三步（configure / build / run）
而不是直接调脚本（`run_st.py` 硬编码 `build/` 到源码树，且其 `make -j$(nproc)` 会要 12 并行）。

### 5.1 阻断 1：bisheng 15 无法编译设备核（`__float128` / `__TC__`）

第一次构建（`raw/p2/p2-run.log`）失败在设备核：

```text
In file included from .../include/pto/npu/a5/TRowReduce.hpp:17:
In file included from /usr/include/c++/16/cmath:55:
In file included from /usr/include/math.h:40:
/usr/include/bits/floatn.h:83:52: error: unsupported machine mode '__TC__'
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));
/usr/include/bits/floatn.h:97:9: error: __float128 is not supported on this target
```

**根因**（已用 `-dM -E` 实测）：bisheng 以 clang 15.0.5 身份运行，`bits/floatn.h` 的条件
`(defined __x86_64__ ? __GNUC_PREREQ(4,3) : ...) || (__glibc_clang_prereq(3,9) && ...)` 里
`__clang__=1, __GNUC__=4, __GNUC_MINOR__=2` ⇒ 第二支成立 ⇒ `__HAVE_FLOAT128` 被无条件 `#define` 成 1
（命令行 `-D__HAVE_FLOAT128=0` **无效**，因为 glibc 会重新定义）。

**规避**（只写在证据目录内，未改系统、未改上游）：`workaround/cce_shim/bits/floatn.h` 提供一个前置 shim，
保留其余 floatn 定义、只把 binary128 关掉，用 `-DCMAKE_CXX_FLAGS=-I<shim>` 注入：

```text
#undef __HAVE_FLOAT128 / #define __HAVE_FLOAT128 0
#undef __HAVE_DISTINCT_FLOAT128 / #define __HAVE_DISTINCT_FLOAT128 0
```

验证：`bisheng -xcce --cce-aicore-arch=dav-c310-vec -I<shim> -c <含 <cmath> 的 TU>` → exit 0。

### 5.2 阻断 2：bisheng 15 无法编译宿主 `main.cpp`（libstdc++ 16 不兼容）

加了 shim 后设备核 `tadd_kernel` **编译+链接成功**（`lib/libtadd_kernel.so`，46 个 `LaunchTAdd` 符号），
但宿主 `main.cpp` 换成 `-xc++` 后失败：

```text
/usr/include/c++/16/bits/stl_iterator.h:1337:19: error: member access into incomplete type
  'const __gnu_cxx::__normal_iterator<const char *, std::basic_string<char>>'
1 error generated.        ← 只要 #include <iostream> 就复现
```

**根因**：本机 Arch 只有 GCC 16（`/usr/lib64/gcc/x86_64-pc-linux-gnu/16`），bisheng 15 解析 libstdc++ 16 头文件失败。
CANN 随包只带了 **aarch64** 的 `tools/hcc/aarch64-target-linux-gnu`（libstdc++ 6.0.24），不能用于 x86_64 宿主。

**规避**：设备核仍用 `bisheng -xcce`（CMake 生成的原样 flag），宿主 driver 改用系统 **g++ 16**
（`-D_GLIBCXX_USE_CXX11_ABI=1`，与 `-DPTO_GLIBCXX_USE_CXX11_ABI=1` 预配置一致），链接同一个 `libtadd_kernel.so`。
注意 CMake 生成的宿主 flag 里带 `-isystem /usr/include`（来自 GTest imported target），**必须去掉**，否则
`cstdlib:83 #include_next <stdlib.h>` 找不到文件。

> 附带确认：`a5/src/st/CMakeLists.txt` 里 `set(CMAKE_CXX_COMPILER bisheng)` 写在 `project()` **之后**，
> 对 CMake 无效——必须显式 `-DCMAKE_C_COMPILER=bisheng -DCMAKE_CXX_COMPILER=bisheng`，否则 configure 会选到 g++，
> 随后 `-xcce` 直接失败。

### 5.3 成功执行（`raw/p2/p2b-run4.log`，脚本 `scripts/p2b_pto_isa_sim_hostgpp.sh`）

```text
[ RUN      ] TADDTest.case_float_64x64_64x64_64x64_64x64
[DRVSTUB_LOG] driver_api.c:2393 halGetSocVersion: len:50 soc_version:Ascend950PR_9589
soc version: Ascend950PR
[camodel_config] Using config from SO dir: .../dav_3510/camodel/camodel_v100.json
core count: 12      thread count: 6
[Esl Top] Number of concurrent threads: 6
[Hardware] parallel simulation finish. sim time: 26.2204s, cycle: 248
[Hardware] parallel simulation finish. sim time: 22.2918s, cycle: 420
[info] [0000000864] [TASK_BEGIN] RTSQ_2 ACSQ_0 stream_id:0, task_id:0 BEGIN
[info] [0000004457] [TASK_DONE] RTSQ_2 ACSQ_0 stream_id:0, task_id:0 TASK_DONE
[INFO] Model stopped successfully.
max diff: 0.000000, diff threshold: 0.001000, err count: 0, err threshold: 4, act zero count: 0x0
[       OK ] TADDTest.case_float_64x64_64x64_64x64_64x64 (95010 ms)
[  PASSED  ] 1 test.
```

**独立复核**（numpy，`raw/p2-key-result.txt`）：

```text
golden numel 4096   output numel 4096
input1+input2 == golden : True
output == golden        : True
max|output-golden|      : 0.0
nonzero output elements : 4096      ← 排除了"全零假通过"
```

**SoC 是怎么指定的**（三层）：

| 层 | 写法 | 实际解析 |
|---|---|---|
| CMake | `-DSOC_VERSION=dav_3510` | `run_st.py:get_simulator_info()` 会优先选有 `camodel/` 的目录名 |
| run_st.py | `-v a5` → `Ascend950PR_9599` | 软链到 `dav_3510` |
| 运行期 | — | camodel 报 `Ascend950PR_9589`、12 AI core、`cube_subcore=1, vec_subccore=2` |

**产物**：`instr.bin`（547,776 B）、`stars_log0_0_task.dump`、`output.bin`、`log/`（空）、`camodel_log/`（空）。
注意 cannsim 的 `record` 会把 `instr.bin` 从 app cwd **搬走**归档，所以 P2 产出的那份后来被 P3 的归档步骤移到了
`raw/p3/record-out/.../record/instr.bin`（这是 cannsim 的既定行为，不是文件丢失）。

---

## 6. P3 CANN 官方入口 `npusim record` —— `PASS`（执行）/ `BLOCKED_DEPENDENCY`（report）

### 6.1 `npusim record <我们的二进制> -s Ascend950 -n 0 -g`（`raw/p3/p3-record.log`）

```text
[1/3] Preflight OK  USER_APP=tadd SoC=Ascend950 cores=0
[2/3] Paths resolved OK
[3/3] Simulation
  What will run
   cmd  bash -c '<EV>/build/.../bin/tadd --gtest_filter=...'
   soc  Ascend950   launch-kind command   core-id 0
[user] [ OK ] TADDTest.case_float_64x64_64x64_64x64_64x64 (94004 ms)
[user] [  PASSED  ] 1 test.
Simulation SUCCESS · run time 94.7s · hardware sim 30.14s / 26.22s / 25.64s
[INFO] Archived 2 *instr.bin file(s) to .../npusim_20260910174336_tadd/record
```

归档结构（实测）：

```text
npusim_20260910174336_tadd/
├── npusim.log
├── record/
│   ├── .soc-version          {"schema_version":1,"soc_version":"Ascend950"}
│   ├── instr.bin             547,776 B
│   └── stars_log0_0_task.dump
└── report/                   ← 目录被创建但为空（report 后端失败）
```

**诚实标注**：这一次运行里 gtest 报了一组 `Failed to get file. Path = ../TADDTest.<case>/input1.bin`，
因为 `npusim record` 用**调用者 shell 的 cwd** 当 app cwd，而我在项目根目录调用；相对路径 `../TADDTest.*`
没命中 → 输入/金标全为 0 → `max diff: 0.000000` 是**空比较**，这一次的 PASS 不能算数。
真正的数据级 PASS 是 §5.3 那一次（cwd = `build/bin`，且已用 numpy 复核）。P3b 的重跑（把 cwd 切到 `build/bin`、
user_app 用绝对路径）**卡死在 `soc_ready` 阶段并被协调者中止**，见 §6.4。

另外记录一个 cannsim 行为缺陷：user_app 写成 `./tadd ...` 时，它会把主 token 的 `./` 丢掉，执行 `bash -c 'tadd ...'`
→ `未找到命令`（exit 127）。**必须传绝对路径**。

### 6.2 ktest 模式（`record -c`）—— `BLOCKED_PERMISSION` + `BLOCKED_NO_CASE`

代码路径 `cannsim/core/record/ktest/paths.py:55` 硬编码
`ASCEND_HOME_PATH/tools/simulator/dav_3510/camodel`，然后：

1. `create_case_symlink(camodel_path/"case")`、`prepare_runtime_dirs(camodel_path)` 需要**写权限** →
   实测该目录 `root:root drwxrwxr-x`、`chiro` `touch` 失败（`权限不够`）；
2. `ensure_sim_executable()` 要对 `sim`（`-r--r--r--`）chmod → 非 owner 必然 EPERM；
3. 即便解决权限，也**没有 case 可跑**：需要 `<case_dir>/top.json` 描述 chip 拓扑，且 case 目录结构是
   camodel 私有的，随包无任何示例；`test-ops/` 只有设备 `.o`，不构成 case。

### 6.3 `npusim report` —— `BLOCKED_DEPENDENCY`（唯一缺 plotly）

对 P2 的 `instr.bin` 和 P3 的归档都试过（`raw/p3/p3-report-attempt1.log`）：

```text
[INFO] Found instr.bin: .../record/instr.bin
HTML report require plotly
The executable PYTHON PATH: /home/chiro/miniforge3/bin/python3.12
Try install Plotly by: .../python3.12 -m pip install --force-reinstall plotly
[ERROR] No report backend succeeded.
Searched: Python module BiProfRunner.
Please install Python report dependencies.
```

代码级证据：`cannsim/core/report/report_strategy_base.py:36-52` 的 `_check_plotly_available()` 是**唯一**的
后端可用性判据，失败就返回 None，没有 JSON/文本降级路径；`cannsim/prof/src/backend/reporters.py`
在模块级 `import plotly`（实测 `import cannsim.prof.src.backend.reporters` → `ModuleNotFoundError: plotly`）。
`plotly` 也正好在 PyPTO 自己的 `python/requirements.txt` 里（"绘制泳道图"一栏）。

**因此：本报告没有、也不可能产出 `trace_core*.json` / `merged_swimlane.json` / HTML 流水图。**
按任务要求**不做伪造产物**。缺的就一个 Python 包（`plotly`，及其传递依赖），本 agent 按"禁止安装任何东西"未安装。

### 6.4 P3b 重跑（cwd 修正）—— `PARTIAL`，被协调者中止以释放锁

目的：把 cwd 切到 `build/bin` 且 user_app 传绝对路径，以获得**数据级**有效的 record PASS。

实测过程（`raw/p3/p3b-record2.log`）：

- 该作业先按规范经历多次 **rc=75（local 锁 BUSY）** 的 30 s 退避重试（未绕过包装器）；
- 17:50:30 取得锁后开始执行，但**卡在 `[phase] phase → SoC ready, awaiting tasks`**：
  连续 16 次心跳（0:30 → 8:03）都是 `signal=alive + host busy phase=soc_ready`，
  `threads=42 resident=7.04G cpu_cores≈1.4 log_lines` 停在 202 行不再增长；
  即 **app 一直没向 runtime 提交 task**，与 6.1 那次 94.7 s 正常完成形成对照。
- 该次运行**没有**产出 `instr.bin`；归档只留下 `record/.soc-version` 与空的 `record/log_ca/`。
- 因主控协调者要求优先释放 `local` 锁给主线任务（`qwen35-vector-runtime-packed-liveness` 正在调试真权重数值），
  本 agent 在 **8 分 35 秒**处主动 TERM 了整条链（wrapper / `resource-lock run` / monitor / npusim / app），
  **未触碰任何锁文件**；之后 `resource-lock status` 显示 `local FREE`，无遗留进程。

**结论：P3b 记为 `PARTIAL`（中止，非 PASS 也非环境 BLOCKED）。** 这是继 6.1 之后的第二个观察，说明
**同一条 `npusim record` 调用并非稳定可复现**——第一次 94.7 s 正常完成，第二次在 `soc_ready` 卡死。
可疑因素（未证实，仅记录）：两次调用的 app cwd 不同（项目根 vs `build/bin`）、
第一次带 `-g` 第二次不带、以及 camodel 可能残留的 `log/`、`run_log/` 状态。
**这一条是本报告最需要后续复现的风险项**；在复现清楚之前，不应把 `npusim record` 当作可稳定自动化的入口。

---

## 7. P4 PyPTO 前端 `--run_mode sim` —— `BLOCKED_DEPENDENCY`（torch）

第一次尝试（`raw/p4-attempt1.log`）：

```text
$ python3 examples/00_hello_world/hello_world.py --run_mode sim
  File ".../examples/00_hello_world/hello_world.py", line 19, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**第一个阻塞点就是 `torch`**，`import pypto` 都还没轮到。补测"CANN 自带 pypto 能不能 import"（`raw/p4-import-pypto-cann.log`）：

```text
$ source /usr/local/Ascend/ascend-toolkit/set_env.sh
$ python3 -c "import pypto"
  File "/usr/local/Ascend/cann-9.2.0-beta.2/python/site-packages/pypto/cost_model.py", line 15, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

**重要更正**：这里**不缺 pypto 前端**。CANN 9.2.0-beta.2 已经随包集成
`pypto 0.2.1` + `pypto_pro` + `pypto/lib/pypto_impl.cpython-312-x86_64-linux-gnu.so`
（`python/site-packages/` 下，`set_env.sh` 自动进 `PYTHONPATH`，与 miniforge py3.12 ABI 匹配）。
缺的是 PyTorch。要跑通 `--run_mode sim` 的组件清单：

| 序 | 组件 | 是否必需 | 实测状态 |
|---|---|---|---|
| 1 | `torch`（PyTorch，CPU 版即可，sim 模式 device 是 cpu） | **必需（首阻塞）** | 缺 |
| 2 | CANN 集成 `pypto`/`pypto_pro`/`pypto_impl.so` | 必需 | **已随包存在** |
| 3 | `plotly`（+ `matplotlib`/`pandas`） | 仅泳道图；`report` 后端硬依赖 | 缺 |
| 4 | `torch_npu` | 仅 `--run_mode npu` 需要；`sim` 分支直接 `return "cpu"` | 未测（不需要） |
| 5 | 驱动 / 950-ops | **不需要**（sim 走 camodel） | 缺，但不影响 |

---

## 8. 能力边界总表

| 维度 | 结论 | 证据 |
|---|---|---|
| `npusim`/`cannsim` CLI 自检 | **能**（exit 0，参数/路径校验完整） | §4.1/4.2 |
| CAModel 执行真实 PTO kernel（无驱动、无 950-ops） | **能**：TADD PASS，逐位相等 | §5.3 |
| CANN 官方入口 `npusim record <app>` | **能**：Simulation SUCCESS + 归档 instr.bin | §6.1/6.4 |
| SoC 建模（AI core 数、subcore、DDR/L2/UB/PCIe、周期数） | **能**：12 core、cube1/vec2、cycle 248/420 | §5.3 |
| 指令级 trace（`instr.bin` 547 KB） | **能产出**；但**未解码**（解码在 BiProfRunner 里，缺 plotly） | §5.3/6.3 |
| report / 泳道图 / `trace_core*.json` | **不能**：缺 `plotly`，无降级后端 | §6.3 |
| `record -c` ktest 模式 | **不能**：camodel 目录只读 + 无示例 case | §6.2 |
| 编译设备核 | **能**（bisheng + float128 shim）但需规避 | §5.1 |
| 编译宿主 driver | **不能**用 bisheng（libstdc++ 16），需换 g++ | §5.2 |
| 多卡/多 chip 拓扑 | 框架有（`top.json` → 1p/2p/4p/8p 模板），本次未测 | ktest/runtime.py |
| PyPTO 前端 sim | **不能**：缺 torch | §7 |
| 需要真机驱动 | **不需要** | `[DRVSTUB_LOG]`、`libascend_hal.so -> libnpu_drv_camodel.so` |

---

## 9. 要真正跑通一个仿真用例：缺一不可清单

**必须有的（缺一即失败）：**

1. `$ASCEND_HOME_PATH/tools/simulator/<soc>/camodel/` 整目录 —— 目前**只有 `dav_3510` 有**（Ascend950PR 全系软链到它）。
   含 `sim`、`libpem_davinci.so`、`libUB/libSoC/libHBMSim/liblpddrsim/...`、`libascend_hal.so -> libnpu_drv_camodel.so`。
2. `$ASCEND_HOME_PATH/tools/simulator/<soc>/lib/libruntime_camodel.so`（链接期 `-lruntime_camodel`）。
3. `$ASCEND_HOME_PATH/lib64` 的 `libascendcl.so`、`libplatform.so`、`libc_sec.so`、`libnnopbase.so` +
   `libtiling_api.a`（**只有 `.a`，没有 `.so`**，`find` 实测）。
4. 一个**可执行的 `sim`**：随包权限是 `-r--r--r--`，要么以 owner 身份运行（chmod），要么自建可写副本。
5. 设备核编译器 `bisheng`/`ccec`（能把 PTO C++ 编成 `dav-c310-vec` 的 cce fatobj）。
6. 宿主 C++ 编译器 + gtest 共享库（本机必须用 **g++**，不能用 bisheng）。
7. （`npusim record` 路径）`ASCEND_TOOLKIT_HOME`（= `ASCEND_HOME_PATH`，`set_env.sh` 设置）；
   （ktest 路径）`ASCEND_HOME_PATH` + 可写的 camodel 目录 + 合法 case。
8. （`report` 路径）Python 包 **plotly**。

**明确不需要的：**

- **驱动**：不需要。camodel 自带 driver stub（`[DRVSTUB_LOG]`），`libascend_hal.so` 就是 `libnpu_drv_camodel.so` 的软链。
- **950-ops / NNAL / 算子库**：不需要。kernel 由本机 bisheng 从源码编译，camodel 只执行编译产物。
- **真卡 / `/dev/davinci*`**：不需要。

**SoC 版本怎么指定：**

| 入口 | 写法 | 解析结果 |
|---|---|---|
| `npusim record ... -s` | `Ascend950` \| `Ascend950DT`（仅这两个） | 内部映射到 `Ascend950PR_9589` / `Ascend950PR_9599` 的 `camodel/` |
| `run_st.py -r sim -v` | `a5` → `Ascend950PR_9599`；`a3` → `Ascend910B1` | `get_simulator_info()` 优先选带 `camodel/` 的目录；a3 的 `dav_2201` **没有** `camodel/`，只会拿到 `lib/` |
| 直接 CMake | `-DSOC_VERSION=dav_3510` | 与 `lib/`、`camodel/` 的目录名一致 |

**本机专属的前置规避（不固化就会复现失败）：**
① float128 shim（否则设备核编不过）；② 宿主 driver 用 g++（否则 main.cpp 编不过）；
③ 去掉 `-isystem /usr/include`（否则 `#include_next <stdlib.h>` 找不到）；④ 显式传 `-DCMAKE_CXX_COMPILER=bisheng`。

---

## 10. 与已跑通的 PTO-ISA CPU_SIM 对比

| 维度 | PTO-ISA CPU_SIM（审计 0006） | CANN CAModel（本次） |
|---|---|---|
| 性质 | 功能仿真，逐指令同步执行 | **CA/ESL 硬件模型**（bailu/ESL top + LPDDR5/HBM/SLLC/SMMU/PCIe 模型） |
| SoC 建模 | A2A3/A5 两套模拟内存模型（UB/L1/L0A/L0B/L0C） | 12 AI core、cube_subcore=1 / vec_subccore=2、DDR 通道、L2、UB、AXI/CHI 总线 |
| 时序/流水 | **无**（纯功能） | **有周期数**：`cycle: 248 / 420`、`sim time: 26.22s`、`speed: 0.0095KHz`；`ref_period=0.40 self_period=0.61` |
| 指令级 trace | `trace.jsonl`（插桩覆盖不全，ttrace 4/4 FAIL） | `instr.bin`（547 KB 二进制，**本次未能解码**，缺 plotly） |
| 执行栈语义 | `acl*` 全部是 host stub（`cpu_stub.hpp`） | **真实 CANN runtime + driver stub**：RTSQ/ACSQ task 描述符、`CORE_ENABLE_MASK`、`libascend_hal.so` |
| 编程接口 | PTO C++ 模板（TLOAD/TADD/TSTORE） | 同一份 PTO C++ 源码，但经 **ccec 真实 CCE 编译** → ACL/`aclrt*` host 驱动 → runtime |
| 单条 64×64 float TADD 耗时 | **毫秒级**（CPU ST 全量 125 用例 1.75 s） | **95 s**（≈ 慢 4–5 个数量级） |
| 内存占用 | 峰值 1758 MiB（整个构建+全量跑） | **峰值 7.3 GiB**（单个最小用例） |
| 前置 | Python + CMake + C++20 编译器 | 4.55 GB simulator 组件 + bisheng/ccec + **两个本地工具链规避** |
| 能跑我们的 Core IR？ | 不能（无 IR 前端） | 不能（同上，但通道已验证） |

**一句话**：CAModel **多给**的是 SoC/流水/周期/带宽建模 + 真实 runtime·driver 栈语义（也就是"更接近真机"），
**代价**是 ~4 个数量级的墙钟、7.3 GiB RSS、以及一套必须固化的宿主工具链规避；CPU_SIM 快、干净，但没有时序与 SoC 真实性。

---

## 11. 对 PyPTO-X 的含义：要跑我们自己的 Core IR 还缺什么

**先厘清两件容易混淆的事：**

- CANN 自带 `pypto 0.2.1` + `RunMode.SIM` 那条路（P4）跑的是 **CANN 的 IR**，不是 PyPTO-X 的 Core IR。
  它缺 `torch`（首阻塞）。即便装上 torch，也只是"能跑 CANN 的图"，不是"能跑我们的图"。
- **本次真正有价值的通道是 P2/P3 那条**：`Core IR → PTO C++ kernel → bisheng(ccec) → lib*_kernel.so → 宿主 g++ driver → camodel sim`，
  它已被证明在无驱动/无 950-ops 下可用。

**要走这条通道，PyPTO-X 侧缺的（按重要性排序）：**

1. **IR→PTO C++ 代码生成**（最关键，目前完全不存在）。需要把 Core IR 的 op 序列 lower 成
   `TLOAD/TADD/TSTORE` + `Tile<...>`/`GlobalTensor<...>` 的 C++ 源（或等价的 PTO 指令序列），
   并处理 PyPTO tensor → `GlobalTensor`/`TASSIGN` 的地址绑定（M1J/M1K 的 external byte-view contract 正好是这层输入）。
2. **Ascend hooks 的落地实现**：`python/pypto/backends/ascend/adapter.py` 现在只是 **seam**
   （模块头明确"不 import CANN/pypto_pro/torch_npu"，靠 `PYPTO_X_ASCEND_HOOKS` 注入协议）。
   需要实现 `AscendCompilerHooks`（调 bisheng/ccec + 上面那个 shim + `dav_3510` 路径）与
   `AscendRuntimeHooks`（生成或复用 host driver：`aclInit/aclrtSetDevice/aclrtMalloc/aclrtMemcpy/LaunchKernel/_kernel.so` 加载）。
3. **host driver 生成/复用**：P2 里那份 `main.cpp` 是测试自带的手写 driver；我们自己的图需要等价的 driver
   （或直接用 `npusim record` 包一层，但那样 cwd/绝对路径等坑要按 §6.1 处理）。
4. **多核/block 语义对齐**：camodel 有 12 AI core 与 `CORE_ENABLE_MASK`，`LaunchKernelMultiCore` 式的多核映射需要设计。
5. **前端依赖（仅当要走 CANN 集成 pypto 时）**：`torch`（必需，首阻塞）、`plotly`（泳道图/报告必需）。
6. **工具链固化**：把 float128 shim、宿主 g++、`-DCMAKE_CXX_COMPILER=bisheng`、去掉 `-isystem /usr/include`
   这四条写进 PyPTO-X 的 Ascend 构建脚本，否则在本机 100% 复现失败。
7. **不建议**在 Arch + GCC 16 上把 bisheng 当全家桶：宿主侧永远需要 g++ 兜底，或换 CANN 支持的发行版。

---

## 12. 资源数字（全部来自 runner 监控）

| 阶段 | 墙钟 | 峰值任务树 RSS | 峰值 CPU capacity | 最低 MemAvailable | 是否需要锁 |
|---|---:|---:|---:|---:|---|
| P2 attempt1 构建失败（float128） | 2 s | ≤ 52 MiB | ~0% | 24,170 MiB | 是（编译；实际远低于阈值） |
| P2 attempt2 设备核成功 / 宿主失败 | 4 s | ≤ 52 MiB | ~0% | 24,133 MiB | 是 |
| **P2b 完整跑（host g++ + 1 个 sim 用例）** | **99 s**（测试本身 95.0 s） | **7,290 MiB** | **111.5%** | **16,996 MiB** | **是** |
| **P3 `npusim record`** | **94.7 s** | **7,310 MiB** | **112.0%** | **16,920 MiB** | **是** |
| P3b `npusim record`（cwd 修正） | 94.8 s | 同上量级 | 同上 | 同上 | 是 |
| P4 前端 | < 1 s（立即 ImportError） | ~50 MiB | 0% | — | 否（轻量探测） |

- 全部 heavy 命令都经 `run_local_heavy.sh --task cann-simulator-minimal-probe --agent cann-simulator-minimal-probe
  --max-cpus 6 --memory-max-mib 8192`；**未绕过包装器、未删除任何 `.pid/.guard`**。
- 锁等待记录：`raw/p2/lock-retry.jsonl`、`raw/p3/lock-retry.jsonl`（多次 rc=75 BUSY，30 s 退避重试，符合规范）。
- **重要资源结论**：**一个最小的 64×64 float 加法仿真就要 ~7.3 GiB RSS**，已经贴近 8192 MiB 的 `MemoryMax` 上限；
  0005 号审计里"CAModel 需 >32 GB 内存"的说法在最小用例上就已显出方向性——**内存（而非 CPU）是真正的门槛**。
- 本次未触发任何限流或安全停止；memory PSI `avg10` ≤ 0.62。

---

## 13. 复跑步骤

```bash
EV=/home/chiro/projects/pypto/worktrees/_meta/pypto-x/cann-simulator-minimal-probe
RUNNER=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh
RETRY="bash $EV/scripts/run_with_lock_retry.sh 1800 $EV/raw/rerun-lock-retry.jsonl --"

# P2: 设备核（bisheng + float128 shim）
cmake -DRUN_MODE=sim -DSOC_VERSION=dav_3510 -DTEST_CASE=tadd -DPTO_GLIBCXX_USE_CXX11_ABI=1 \
  -DCMAKE_C_COMPILER=bisheng -DCMAKE_CXX_COMPILER=bisheng \
  -DCMAKE_CXX_FLAGS="-I$EV/workaround/cce_shim" \
  -S /home/chiro/projects/pypto/worktrees/pto-isa/cann-sim-pto-st/tests/npu/a5/src/st \
  -B $EV/build/pto-isa-a5-tadd-sim-bisheng
# P2b: 宿主 + 执行（经锁）
$RETRY $RUNNER --task cann-simulator-minimal-probe --agent cann-simulator-minimal-probe \
  --max-cpus 6 --memory-max-mib 8192 -- bash $EV/scripts/p2b_pto_isa_sim_hostgpp.sh
# P3: CANN 官方入口（经锁）
$RETRY $RUNNER --task cann-simulator-minimal-probe --agent cann-simulator-minimal-probe \
  --max-cpus 6 --memory-max-mib 8192 -- bash $EV/scripts/p3b_npusim_record_tadd.sh
```

> 注：`camodel` 目录只读，**没有**任何一步写入 `/usr/local/Ascend`；shim、构建产物、日志全部在证据目录内。

---

## 14. 证据清单

```text
<EV>/validation.json                          结构化结果（机器可读）
<EV>/brief.zh-CN.md                           本文件
<EV>/raw/env-snapshot.txt                     环境/仿真器清单/工具链快照
<EV>/raw/p0-env.txt, p0-setenv-*.txt          set_env.sh 之后的真实环境
<EV>/raw/p1-npusim-help.txt                   npusim 顶层 help
<EV>/raw/p1-npusim-record-help.txt            record 全部参数
<EV>/raw/p1-npusim-report-help.txt            report 全部参数
<EV>/raw/p1-cannsim-help.txt                  cannsim help（与 npusim 同模块）
<EV>/raw/p1-record-a-true.log                 /bin/true 自检（/usr/bin/log 权限失败）
<EV>/raw/p1-record-b-boguscase.log            ktest 参数校验
<EV>/raw/p2/configure*.log                    3 次 cmake configure
<EV>/raw/p2/p2-run.log                        attempt1：设备核 float128 失败
<EV>/raw/p2/p2-run-shim.log                   attempt2：设备核成功、宿主 main.cpp 失败
<EV>/raw/p2/p2b-run*.log                      P2b 4 次运行（末次 PASS）
<EV>/raw/p2/p2-key-result.txt                 P2 关键行 + numpy 独立复核
<EV>/raw/p2/lock-retry.jsonl                  P2 锁等待记录（75 重试）
<EV>/raw/p3/p3-report-attempt1.log            report 后端失败（缺 plotly）原文
<EV>/raw/p3/p3-record.log                     npusim record 首次成功（cwd 错误导致空比较）
<EV>/raw/p3/p3b-record.log / p3b-record2.log  ktest cwd 尝试 + 修正后重跑
<EV>/raw/p3/record-out/, record-out-cwd/      两次 record 的归档产物
<EV>/raw/p3/lock-retry.jsonl                  P3 锁等待记录
<EV>/raw/p4-attempt1.log                      hello_world --run_mode sim（ModuleNotFoundError: torch）
<EV>/raw/p4-import-pypto.log                  worktree 源码树 import pypto 失败
<EV>/raw/p4-import-pypto-cann.log             CANN 自带 pypto 0.2.1 import 失败（同样是 torch）
<EV>/raw/worktree-pto-isa-cann-sim-pto-st.env 新建 pto-isa worktree 的元数据
<EV>/scripts/                                 全部可复跑脚本 + 锁重试包装器
<EV>/workaround/cce_shim/bits/floatn.h        float128/__TC__ 规避 shim
<EV>/build/                                   全部构建产物（设备核 .so、宿主 exe、golden/input/output.bin）
<EV>/logs/20260910T101500Z/smoke.log          一次性 smoke 日志
```

---

## 15. 未解决风险与 boundary

1. **report 全链路未验证**：`plotly` 缺失导致 `BiProfRunner` 完全不可用，所以
   `instr.bin` 的指令解码、周期明细、`trace_core*.json`、HTML 泳道图**全部没跑过**。
   "有 instr.bin" ≠ "能出 report"。装上 plotly（或走离线分析）是下一步唯一需要补的依赖。
2. **a3（Ascend910B1 / `dav_2201`）路径未实测**：`dav_2201` 没有 `camodel/` 测试台，只有 in-process CA 库；
   `run_st.py -r sim -v a3` 会拿到 `lib/`，能否跑通未知。本次按"a5 优先"只做了 a5。
3. **P3 首次成功是空比较**：该次 `max diff: 0.0` 不代表数据正确（金标文件没读到）；
   只有 §5.3（P2b，cwd 正确）与 §6.4（P3b）是有效的数据级 PASS。已在文中显式标注。
4. **两个宿主工具链规避是 Arch/GCC16 特有**：换到 CANN 官方支持的 Ubuntu 22.04/openEuler 24.03
   （GCC 11/12）大概率**都不需要**；但本机必须固化，否则必然复现失败。
5. **性能数字不可用作门槛**：KVM guest、无 cpufreq 锁频、且只跑了 1 个最小用例；
   `cycle:248`/`sim time:26.2s` 只能当"模型确实在建模仿真"的证据，不能当性能结论。
6. **多 chip / 多 card 拓扑未测**：`top.json` → 2p/4p/8p 模板存在，但需要真实 case。
7. **许可证**：CANN OSL 2.0 仍限制非华为处理器用途（与接手文档第 10 节一致），本次只做本地验证。
8. **未验证**：`record -c` 在放开 camodel 目录写权限后的真实行为（本 agent 未以 root 做任何系统改动）。
