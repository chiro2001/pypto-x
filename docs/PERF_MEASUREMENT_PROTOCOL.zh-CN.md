# PyPTO-X 性能测量协议与门槛策略（冻结提案）

- 文档类型：规划交付物（**纯规划，未执行任何 benchmark / heavy 测量 / 软件安装**）
- 任务：`qwen35-perf-threshold-protocol`
- 生成时间：2026-09-10T08:03:16Z 起
- 依据基线：`port/pypto-x-integration @ bcf9516e6419d232338988238ffa8f79e59079c1`
- 配套文件：同目录 `perf_protocol.proposed.yaml`（机器可读协议草案）、`validation.json`（引用来源索引）

> 本协议的目标是**冻结"怎么测"**，不是**冻结"测到多少"**。
> 本阶段明确**不设定任何绝对数值门槛**；`configs/model_targets.yaml:84` 的
> `performance_threshold_policy: defer-until-first-runnable-backends` 保持不动。

---

## 0. 启动协议字段（原样记录）

```text
task_name=qwen35-perf-threshold-protocol
worktree=/home/chiro/projects/pypto/worktrees/pypto-x/qwen35-perf-threshold-protocol
branch=work/qwen35-perf-threshold-protocol
base=port/pypto-x-integration @ bcf9516e6419d232338988238ffa8f79e59079c1
started_at=2026-09-10T08:03:16Z
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

Smoke（**仅一次，未重跑**）：

```text
agent_id=qwen35-perf-threshold-protocol
started_at=2026-09-10T08:03:16Z
smoke_started_at=2026-09-10T08:06:58Z
smoke_finished_at=2026-09-10T08:06:58Z
target=host
head=bcf9516e6419d232338988238ffa8f79e59079c1
branch=work/qwen35-perf-threshold-protocol
status=PASS
notes=pypto/pypto_pro 源码、Git、基础工具链和 Python 语法检查通过；未加载模型
log_file=/home/chiro/projects/pypto/worktrees/_meta/pypto-x/qwen35-perf-threshold-protocol/logs/20260910T080316Z/smoke.log
```

---

## 1. 现状判定（每条都有证据）

这一节先回答一个前置问题：**现在到底有没有"后端性能"可测？** 答案是：**几乎没有**。
如果不先把这一点写进协议，后续一定会把 Python 解释器开销误报成后端性能。

### 1.1 执行模型：全部是逐 op 派发，没有任何融合

| 后端 | 执行模型 | 证据 |
|---|---|---|
| CPU scalar | 逐 op Python 解释 | `python/pypto/backends/cpu/runtime.py:133-134` `for index, operation in enumerate(function.body.operations): _execute_operation(...)` |
| CPU vector（AVX2/AVX-512/SVE256 的 **plan** runtime） | 逐 op Python 解释 | `python/pypto/backends/cpu/vector/runtime.py:462` `_execute_operation`、`:966` `_execute_matmul`；该文件全文无 `ctypes`/`CDLL` |
| CPU AVX2/AVX-512/SVE256 **native** runtime | 每 op 一次 ctypes 调用 + Python plan 派发 | `python/pypto/backends/cpu/x86/avx2.py:11` `import ctypes`；原生侧是**固定通用 kernel 库**，不是按 shape 生成的特化 kernel（`python/pypto/compiler/targets/cpu_avx2.py:32` `_NATIVE_SOURCE`，符号仅 16 个：`ptx_avx2_copy_f32`、`ptx_avx2_binary_f32`、`ptx_avx2_matmul_f32` … `:303/:325`） |
| NVIDIA CUDA | **每个 op 一个 PTX kernel**，每 op 一次 `cuLaunchKernel`，整图结束才同步一次 | `python/pypto/compiler/targets/cuda.py:923-926` `for index, (operation, plan) in enumerate(...): kernels.append(_Kernel(index, ...).render())`；`python/pypto/backends/cuda/runtime.py:398` `module.get_function("px_op{index}")`、`:410` `self.driver.launch_kernel(...)`、`:435` 单次 `context_synchronize()` |
| AMD gfx1036 | 无 runtime（`BLOCKED_DEVICE`） | `docs/RESOURCE_MATRIX.zh-CN.md:167-171`、`:179` |

**标量 matmul 是纯 Python 三重循环**，内层每次乘加都过一次 `_cast_value`：

```python
# python/pypto/backends/cpu/runtime.py:915-924
for row in range(m):
    for column in range(n):
        aggregate = 0
        for inner in range(k):
            value = aggregate + (left[...] * right[...])
            aggregate = _cast_value(value, "float32") if fp32_accumulation else value
        output.append(_cast_value(aggregate, dtype))
```

Qwen3.5-0.8B 的 `(1,1,past=4096)` 图里有 **253 个 matmul**（见 1.3），其中 lm_head 投影是
`1×1024×248320 ≈ 2.54×10^8` 次 MAC → 在 scalar 后端大约是 **10^9 量级的 Python 解释器迭代**。
**结论：当前 CPU scalar 路径的任何 token/s 数字度量的是 CPython，不是 CPU。**

### 1.2 无任何 device-side 计时设施

全仓搜索 `cuEvent*` / `EventRecord` / `ElapsedTime`：**`python/pypto/` 下 0 命中**。
CUDA 侧只能靠 `context_synchronize()` 前后 wall clock（`python/pypto/backends/cuda/runtime.py:435`），
因此当前最小可分辨的"kernel 时间"实际上包含：HtoD/DtoH + 253+ 次 launch + 一次全设备同步。
**协议必须要求先补 event 计时，否则 CUDA 数字不可用于 kernel 级比较。**

### 1.3 唯一已有的 lowering 计时实测值

`../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/qwen35-full-lower.json`：

```text
profile            = {batch: 1, steps: 1, past_length: 4096}
operation_total    = 4532   # graph v2；v3（decay 修复后）为 4550
plan_total         = 4532
llvm_ir_bytes      = 10510232
resource.lower_seconds = 1.929875544999959
resource.max_rss_mib   = 95.3828125
```

op 直方图（同文件 `canonical_operation_counts`）：

| op | 数量 | op | 数量 | op | 数量 |
|---|---:|---|---:|---|---:|
| cast | 1124 | reduce_mean | 79 | split | 24 |
| reshape | 632 | concat | 66 | sub | 24 |
| mul | 531 | constant | 66 | softplus | 18 |
| transpose | 385 | silu | 60 | div | 6 |
| broadcast | 350 | reduce_sum | 42 | reduce_max | 6 |
| slice | 336 | neg | 30 | where | 6 |
| add | 326 | exp | 24 | iota | 3 |
| matmul | 253 | sigmoid | 24 | compare | 1 |
| rsqrt | 115 | | | embedding | 1 |

op 数**与 `past_length` 基本无关**（metadata compact）：
`python/tests/ut/pypto_x/test_qwen35_full_decoder_connectivity.py:158-161`
对 `past ∈ {0, 4096}` 断言 `len(operations) < 6000` 且 `len(canonical_json()) < 3_000_000`。

### 1.4 模型级（L1）目前**完全没有**计时证据

M1K-CPU 的 24 层 synthetic 执行报告
（`../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cpu-buffer-ingestion-final-r2/qwen-{scalar,avx2,avx512,sve256}.json`）
逐字段核对：只有 `status/backend/graph_digest/input_count/output_count/inputs_unchanged/
instrumentation/logits_count/...`，**没有任何 `*_seconds` / `*_rss_*` / `throughput` 字段**。
M1K-CUDA 证据（`.../integration-w6-qwen35-bf16-cuda-buffer-ingestion-final/validation.json`）同样只有
`inputs=371 / outputs=51 / state_inputs=48 / packed_nbytes=43200 / launches=2`，无计时。

**这是当前性能阶段的证据空白：L1 从未被测量过。**

### 1.5 编译缓存不存在持久化

`python/pypto/compiler/cache.py:16-33` 的 `ArtifactCache` 是纯内存 dict，
docstring 明写 `persistent cache policies remain backend-specific`。
所以 `configs/model_targets.yaml:94` 的 `compile-and-cache-time` 里，
`cache hit` 目前只能报 `not_implemented`。

### 1.6 现有 Ascend ST "perf" 测试不是性能门槛

`python/tests/st/test_perf.py:15` 自述 `by comparing profiling data against expected baselines`，
但断言实际比对的是 kernel detail 字段一致性（`:334-349`），不是计时。
且它依赖 `import torch` / `import torch_npu`（`:30-31`），本机与 5080 上都不可运行。
**不得把它当性能验收依据。**

### 1.7 环境实测（本次只读探测，未跑任何 benchmark）

**本机（local dev machine）**

```text
CPU model        : AMD Eng Sample: 100-000000956-50_Y
online CPUs      : 12          NUMA nodes: 1        L3: 192 MiB (12 instances)
ISA flags        : avx2, avx512f, avx512bw, avx512vl, avx512_bf16, fma, f16c
                   （无 amx_bf16 / amx_tile）
virtualization   : systemd-detect-virt = kvm；/proc/cpuinfo 含 hypervisor
MemTotal         : 30753688 kB (~29.3 GiB)     MemAvailable: ~24 GiB
cpufreq          : /sys/devices/system/cpu/cpu0/cpufreq/ 不存在
                   → scaling_governor / scaling_driver / no_turbo / boost 全部不可读
toolchain        : clang 22.1.8, gcc 16.1.1, cmake 4.4.2, python 3.12.10 (conda-forge)
system BLAS      : libblas.so.3.12.0 / libcblas.so.3.12.0 → nm -D | grep -c openblas = 0
                   → netlib reference，只可作 correctness 对照
numpy            : 2.5.1，链接 scipy-openblas 0.3.33.112.0
                   /home/chiro/miniforge3/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-017048f4.so
                   导出 scipy_openblas_set_num_threads64_ / scipy_openblas_get_num_threads64_
torch / scipy    : 缺失
affinity         : 0-11
resource-lock    : local FREE, gamepc FREE
```

> **两个必须写入协议的硬事实**
> 1. 本机是 **KVM guest**（不是裸机），且 `cpufreq` 完全不可用 →
>    **频率既读不到也锁不住**。任何绝对数值门槛在本机天然脆弱。
> 2. 本机的优化 BLAS 只存在于 **numpy 的 scipy-openblas 轮子**里；
>    系统 `libblas` 是 netlib reference。把系统 BLAS 当性能基线是错的。

**RTX 5080（`192.168.101.5` WSL2）**

```text
GPU              : NVIDIA GeForce RTX 5080, 16303 MiB
KMD / CUDA UMD   : 616.92 / 13.4        ← 与文档记录的 610.62 / 13.3 不一致（漂移）
compute cap      : 12.0
clocks (idle)    : clocks.sm = 427 MHz / clocks.max.sm = 3090 MHz   ← 7.2x 差距, P8
                   clocks.mem = 405 MHz / clocks.max.mem = 15001 MHz
power            : limit 365.00 W == default 365.00 W
persistence_mode : Enabled        compute_mode: Default
temperature      : 43 C            utilization.gpu: 5%
memory           : used 3913 MiB, free 12065 MiB
compute apps     : 空（无 compute process）
clocks event     : Reliability: Active（其余 Not Active）
CUDA 库          : ldconfig -p | grep -i cublas  → 空
                   /usr/lib/wsl/lib/ 只有 libcuda.so[.1][.1.1], libnvidia-ml.so.1,
                   nvidia-smi, libnvcuvid, libnvoptix, libnvgpucomp 等
                   → 无 libcublas、无 libcudart
tools            : nvcc 缺失（command -v nvcc 无输出）
```

> **三个必须写入协议的硬事实**
> 1. **cuBLAS 在 5080 WSL 上不存在** → CUDA 侧的 vendor 基线当前只能标 `MISSING`。
> 2. **空闲时钟 427 MHz vs 峰值 3090 MHz** → 不做 clock warmup 直接测，会得到低 7 倍的数字。
> 3. 驱动已从 610.62/13.3 升到 **616.92/13.4**；`docs/RESOURCE_MATRIX.zh-CN.md:40` 与
>    `HANDOFF.zh-CN.md:233` 需要更新（本报告不修改控制仓，交父 agent 归档）。

**鲲鹏 920B / QEMU / AMD gfx1036**（沿用既有冻结结论，不重新探测）

- 920B：`docs/RESOURCE_MATRIX.zh-CN.md:12`、`HANDOFF.zh-CN.md:251` —— 2 vCPU KVM guest，
  `configs/development_lock.yaml:310` 已明写 `ecs_native_sve256_is_kvm_functional_and_disassembly_evidence_not_performance_threshold`。
- QEMU：`docs/RESOURCE_MATRIX.zh-CN.md:94-105` 明确列出"不能用于"清单。
- AMD：`docs/RESOURCE_MATRIX.zh-CN.md:179` 静态 C3 PASS，runtime `BLOCKED_DEVICE`。

---

## 2. 测量对象分层

三层，**层与层之间不得混报**。每个数字必须携带 `level` 字段。

### L0 · 算子级 microbench

**驱动方式（强制）**：直接构造 `Artifact` + `LaunchRequest` 调用 backend runtime，
**不得**通过 `build_qwen35_text_decoder_graph` 走图遍历。
理由：只有绕开图，才能把 `dispatch_seconds` 与 `kernel_seconds` 分离。

**shape 必须取自 Qwen3.5-0.8B 真实 profile**（`python/pypto/portable/qwen35.py:47-73`）：

```text
hidden=1024  intermediate=3584  num_layers=24
query_heads=8  kv_heads=2  head_dim=256  rotary_dim=64
linear heads 16x128(key) / 16x128(value)  conv_kernel=4
full_attention_indices=(3,7,11,15,19,23)   → 18 GDR + 6 full attention
vocab=248320 (= python/pypto/portable/qwen35.py:434)
rms_norm_eps=1e-6  max_position_embeddings=262144
```

**算子清单与对应实现位置**

| 组 | 算子 | 实现符号（integration HEAD） | shape 集 |
|---|---|---|---|
| matmul | rank-2/3/4，exact batch | `backends/cpu/runtime.py:881` `_matmul_value`（`:891-902` 支持 rank 2–4，batch 前缀必须 exact 相等）；`backends/cpu/vector/runtime.py:966` `_execute_matmul`；`compiler/targets/cpu_avx2.py:303/325` `ptx_avx2_matmul_{f32,bf16}` | M,N,K ∈ {1, 128, 512, 1024, 3584, 4096}；batch ∈ {1, 8, 16}；dtype ∈ {bf16×bf16→bf16, fp32×fp32→fp32} |
| elementwise | add/sub/mul/div/neg | `backends/cpu/runtime.py:717` `_binary_value`、`:743` `_unary_value`；`compiler/targets/cpu_avx2.py:108/155` `ptx_avx2_binary_{f32,bf16}` | 元素数 ∈ {4096, 65536, 1M, 4M}；标量广播与非广播两种 |
| reduction | reduce_sum / reduce_max / reduce_mean | `backends/cpu/runtime.py:824` `_reduce_value`；`cpu_avx2.py:178/189/211/226/285/294` | 末轴 / 中间轴 / 全 reduce；输入 ∈ {1×1024×1024, 1×4096×256} |
| softmax | stable softmax | `python/pypto/portable/qwen35.py:889` `build_stable_softmax` | `(1,8,T,4096)`，T ∈ {1,128} |
| conv1d | causal conv1d + state | `qwen35.py:2793` `build_causal_conv1d`、`:2721` `build_causal_conv1d_state`、`:2804` `build_qwen35_gated_delta_conv_state` | `(1,T,2*16*128+16*128)` with `conv_channels = 2*16*128 + 16*128 = 6144`，T ∈ {1,128,512} |
| GDR step | 单步递推 | `qwen35.py:3265` `_gdr_step`（**每步恰 21 个 `builder.op`**，无内部分支）、`:3446` `_build_gdr_recurrent_state`、`:3558` `build_qwen35_gdr_recurrent_state` | `[1,16,128,128]` 非零 FP32 state（对齐 `docs/RESOURCE_MATRIX.zh-CN.md:121` 已验收的 shape） |
| layout/indexing | reshape/transpose/slice/split/concat/broadcast/gather/where/cast | `backends/cpu/runtime.py:509-661`（`_reshape_value`…`_gather_value`）、`:750` `_broadcast_value`、`:775` `_where_value` | 取自 1.3 直方图里 cast=1124 / reshape=632 / transpose=385 / broadcast=350 / slice=336 的实际 shape 众数 |
| math | exp/rsqrt/sigmoid/silu/softplus/neg | `backends/cpu/runtime.py:710` `_math_value`、`:671` `_math_scalar`；`cpu_avx2.py:271/277` `ptx_avx2_math_{f32,bf16}` | 元素数 ∈ {4096, 65536, 1M} |

**microbench 的强制双数字**：每个 case 必须同时报

```text
kernel_seconds   = 单次 launch 内 R 个同类 op 的总时间 / R
dispatch_seconds = (R 次独立 launch 的总时间 - 单次 launch 内 R 个 op 的总时间) / R
```

这是识别"框架开销伪装成后端性能"的唯一可靠手段，也是反模式 #3 的检测依据。

### L1 · 模型级

**prefill 候选 T**：`T ∈ {16, 128, 512}`。
- `T=16` 来自 `configs/model_targets.yaml:14` `prompt_lengths: [16, 128, 1024]` 的最小值（最保守、最便宜）。
- 建议首轮只做 **T=16 与 T=128**，`T=512` 需先看 2.3 的 op 数增长再定。
- `T=1024/4096/32768` 属 `later_prompt_lengths`（`configs/model_targets.yaml:15`），本阶段不列入。

**decode**：`T=1`，`past ∈ {0, 128, 512, 4096}`（4096 对齐已冻结的 `(B=1,T=1,past=4096)` profile）。

**必须声明执行模型**。每个 L1 数字都要带：

```yaml
execution_model: op_by_op_dispatch   # 当前唯一可取值；fused 尚未存在
dispatch_count: <图 op 数；graph v3 (T=1/past=4096) 为 4550，v2 为 4532>
dispatch_seconds: <float>
kernel_seconds: <float|null>
```

**必须先跑 dispatch floor**：用同一张图、同样 op 数，但把每个 op 的 shape 压到最小
（如 `(1,1)`），得到 `dispatch_floor_seconds`。若
`wall_seconds < 2 * dispatch_floor_seconds`，则该 L1 数字标 `dispatch_dominated=true`，
**不得**用于任何跨后端比较。

**L1 的当前可交付性判定**

- CPU scalar：`UNGATED`（纯 Python 三重循环，见 1.1）。
- CPU AVX2/AVX-512：仍逐 op ctypes 派发，`dispatch_dominated` 极可能为真 → 预期 `UNGATED`。
- CUDA：253 个 matmul × 每 op 一次 launch + 全同步 → 预期 `dispatch_dominated` → `UNGATED`。
- 结论：**L1 目前只能产出"端到端可用性"和 dispatch floor 两个数，不能产出后端性能结论。**

### L2 · 编译 / lowering 级

三个独立可测点，**必须分开报**：

| 测点 | 调用 | 现有实测 | 备注 |
|---|---|---|---|
| lowering | `compiler.lower(program, target)` | AMDGPU gfx1036 全图 `1.9299 s` / `95.38 MiB`（`qwen35-full-lower.json`） | 与后端无关的图规模指标 |
| compile | `compiler.compile(target_ir, options)` | **无实测** | AVX2/AVX-512 会真正调 clang（`compiler/targets/cpu_avx2.py:612-666`）；CUDA 只渲染 PTX，**无 ptxas**（`compiler/targets/cuda.py:919-930`） |
| 产物大小 | `len(artifact.payload)` / `llvm_ir_bytes` / PTX bytes / `.so` bytes | AMDGPU `llvm_ir_bytes = 10,510,232` | CUDA / CPU 侧无实测 |

**重要口径提醒**：AVX2 编译的是**固定源码**（`_NATIVE_SOURCE`，`cpu_avx2.py:32`），
编译命令固定为 `clang -shared -fPIC -O2 -std=c99 -mavx2 [-mfma] <src> -lm -o <out>`
（`cpu_avx2.py:695-698`）。因此 **AVX2 的 `compile_seconds` 与图规模基本无关**，
不能拿它当"编译速度达标"。真正随图增长的是 `lower_seconds`。

**cache 字段**：`cache_state ∈ {cold, warm, not_implemented}`；当前
`compiler/cache.py:16-33` 只有进程内缓存 → 首轮一律 `not_implemented`。

---

## 3. 指标与统计口径

### 3.1 指标定义

| 指标 | 单位 | 定义 | 强制 |
|---|---|---|---|
| `wall_seconds` | s | 从进入 measurement 函数到全部输出落地的单调时钟差 | 是 |
| `kernel_seconds` | s | 排除派发与 I/O 的内层时间；CUDA 需 event 计时，当前 `null` | 是（可 null） |
| `dispatch_seconds` | s | 每 op 派发开销（见 2.1 差值法） | L0/L1 强制 |
| `tokens_per_second` | tok/s | decode：`T_generated / decode_wall_seconds`；prefill：`T_prompt / prefill_wall_seconds` | L1 强制 |
| `ops_per_second` | op/s | `dispatch_count / wall_seconds` | L1 强制 |
| `gflops` | GFLOP/s | `total_flops / kernel_seconds / 1e9`；无 kernel 计时时用 `wall_seconds` 并标 `gflops_scope: end_to_end` | L0 强制 |
| `peak_rss_mib` | MiB | `resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024.0`（同 `qwen35_runtime_binding_metadata_driver.py:20-21`、`amd_gfx1036_c3_static_lower_driver.py:38`） | 是 |
| `peak_device_memory_mib` | MiB | `cuMemGetInfo` 峰值 / `--query-compute-apps` 最大值 | GPU 强制 |
| `lower_seconds` / `compile_seconds` / `artifact_bytes` | s / s / B | 见 L2 | L2 强制 |

### 3.2 FLOP 计数口径（**必须逐算子写死，禁止口算**）

仓库当前**没有任何 FLOP 计数代码**（全仓 `grep -i flops` 在 `python/pypto/` 与
`python/tests/ut/pypto_x/` 下 **0 命中**）。因此以下公式是协议首创，必须作为
`flop_accounting` 字段逐条写入证据：

| 算子 | FLOP 公式 | 说明 |
|---|---|---|
| matmul (rank 2–4) | `2 * B * M * N * K` | **bf16 与 fp32 同为 2 FLOP/MAC**；禁止给 bf16 记 1 |
| conv1d（因果） | `2 * T * C_out * C_in * K` | K = `linear_conv_kernel_dim` = 4 |
| elementwise binary (add/sub/mul/div) | `1 * N_out` | |
| elementwise unary (neg/cast) | `0` | cast 不计算术 FLOP，单独报 `cast_bytes_per_second` |
| reduction sum / mean | `(N_in / N_out - 1) * N_out` | |
| reduction max | `0`（不计 FLOP） | 单独报 `compare_ops_per_second` |
| exp / sigmoid / silu / softplus / rsqrt | `0`（不计入 GFLOPS） | 单独报 `transcendental_ops_per_second`；禁止把 libm 调用算成 FLOP |
| softmax | `3 * N`（max + exp + sum + div 只计可计部分） | 必须同时报 `transcendental_ops_per_second` |
| embedding / gather / iota / where | `0` | 报 `bytes_per_second` |

另外强制两个派生字段：

```yaml
gflops_scope: kernel_only | end_to_end
arithmetic_intensity_flop_per_byte: <float>   # total_flops / total_bytes_moved
memory_bound: <bool>                          # AI < 10 FLOP/byte 时为 true
```

`memory_bound: true` 时**禁止**与峰值算力对比、禁止写"达到峰值 X%"。

### 3.3 统计流程

```text
1. clock_warmup   : 连续跑 payload >= 2.0 s（仅 GPU 需要断言时钟，见 4.3）
2. warmup         : W = 5 次，取最后一次
3. warmup 验收    : |last_warmup - min(first 5 measured)| / min(...) < 0.05
                    不满足则再 warmup 5 次，最多 3 轮
                    3 轮后仍不满足 → status = INVALID_WARMUP
4. measured       : R = 21 次（L0 microbench R = 51）
5. 统计量         : min, median, p95(nearest-rank, 不插值), max,
                    stdev (sample, n-1), cv = stdev/median, p95_over_median
6. 异常值         : 不剔除。标记 |x - median| > max(3 * 1.4826 * MAD, floor)
                    其中 floor = max(1 µs, 0.02 * median)
                    若 outlier_ratio > 0.10 → status = UNSTABLE
7. 方差门禁       : 见 3.4
```

**R 的取值理由**：中位数在 R≥11 已稳定；p95 在 R=21 时对应第 20 位，
分位误差可接受。R < 11 一律 `gate_eligible=false`。

**修订记录（2026-09-11，ERR-0004）**：第 6 步原为 `|x - median| > 3 * 1.4826 * MAD`
（无下限）。在 B3b（CUDA GEMM vs cuBLAS）实测中发现该判据在 WSL2 上**不可复现**：
被作废序列的 MAD 仅占中位数 0.06–0.83%，使门限低至 0.28–3.7%，于是偏离中位
0.3–6% 的 3–6 个样本即触发 `UNSTABLE`；典型反例是一组 `cv = 0.0023`、
`p95/median = 1.0014` 的统计上极稳序列仍被判无效（独立复验 2 campaign / 8 轮 →
5 轮作废、0 个有效 campaign；实现方数据亦停在 2/21 边缘）。根因是 MAD 在该平台上
度量的是**计时器微秒级抖动**而非分布污染。修订引入下限 `max(1 µs, 2% × median)`，
使判据只在偏差具备物理意义时才标记异常；真正的污染（如 cuEvent rate 偏移 +8~10%）
仍会被标记。证据：`_meta/pypto-x/verify-cuda-gemm-baseline-r2/{brief.zh-CN.md,raw/}`。

### 3.4 方差上限（**超过即判无效**）

| 环境 | `cv = stdev/median` 上限 | `p95/median` 上限 |
|---|---|---|
| 本机 local（KVM guest，无 cpufreq） | **0.05** | **1.25** |
| RTX 5080（时钟可断言 ≥0.9×max） | **0.03** | **1.15** |
| RTX 5080（`clock_unlocked=true`） | 0.05 | 1.25 |
| 鲲鹏 920B | 不适用（`gate_eligible=false`） | 不适用 |

任一条不满足 → `statistics.verdict = INVALID_VARIANCE`，该轮全部数字**不得**进入任何比值门槛，
`gate.relative = UNGATED`。

---

## 4. 环境控制

### 4.1 本机（local）

**强制入口**（`docs/LOCAL_RESOURCE_POLICY.zh-CN.md:31-37`、`docs/RESOURCE_MATRIX.zh-CN.md:196`）：

```bash
scripts/resource/run_local_heavy.sh \
  --task <task> --agent <agent-id> --max-cpus 6 -- \
  env PYTHONPATH=python python3 <driver.py> --output <evidence.json>
```

调用链（已核对源码）：
`run_local_heavy.sh` → `resource-lock run local pypto-x <task> <agent>`
→ `systemd-run --user --scope -p MemoryHigh/MemoryMax/MemorySwapMax=0/CPUQuota=(max_cpus*100)%/TasksMax=512`
→ `monitor_local_heavy.py`（`scripts/resource/run_local_heavy.sh:119-133`）。

**逐条控制项**

| 控制项 | 现状 | 协议要求 |
|---|---|---|
| 全局锁 | `resource-lock status` → `local FREE` | 测量轮**必须**持 `local` 锁；返回 75/69 必须等待（`docs/LOCAL_RESOURCE_POLICY.zh-CN.md:11-14`） |
| CPU 亲和性 | `monitor_local_heavy.py:191` 用 `allowed[:max_cpus]` | 同一 task 的所有轮次 `--max-cpus` 必须**完全相同**；必须原样记录 resource log `event=start` 行的 `cpu_affinity=0,1,2,3,4,5` |
| CPU quota | `CPUQuota=(max_cpus*100)%` | 6 → 600%；不得中途改 |
| 线程数 | `monitor_local_heavy.py:204-208` 把 `OMP/OPENBLAS/MKL/NUMEXPR/RAYON/MAX_JOBS/CMAKE_BUILD_PARALLEL_LEVEL/PYTEST_XDIST_AUTO_NUM_WORKERS` 全部设为 `max_cpus` | 记 `threads=6`；**基线 BLAS 也要显式 setter 复核**（`OPENBLAS_NUM_THREADS` 是只读一次的），见 5.1 |
| 频率 | 本机 **无 cpufreq** | 记 `cpufreq_available=false`、`governor=null`、`frequency_control=unavailable_kvm_guest`；**本机绝对门槛永久 `UNGATED`** |
| 内存 | `MemorySwapMax=0` | 若 supervisor 以 70 终止（`LOCAL_RESOURCE_POLICY.zh-CN.md:69`），该轮作废，不得降级裸跑 |
| 噪声门禁 | sample 里已有 `load1` / `cpu_psi_avg10` / `available_mib` | 一轮测量必须满足 `max(load1) < online_cpus` 且 `max(cpu_psi_avg10) < 25` 且 `min(available_mib) > 4096`，否则 `gate.verdict=ENV_NOISY` |
| 工作集 | — | TMPDIR/build/artifact/日志一律放 `../worktrees/_meta/pypto-x/<task>/`（`docs/RESOURCE_MATRIX.zh-CN.md:202`），并在证据里记录绝对路径 |

### 4.2 RTX 5080

- **不申请 `gamepc` 锁**（`docs/LOCAL_RESOURCE_POLICY.zh-CN.md:94`、`configs/development_lock.yaml:19-21`）。
- 但**必须**在测量前后各执行一次并记录：

```bash
ssh -o BatchMode=yes 192.168.101.5 'wsl.exe -e bash -lc "
  nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,clocks.max.memory,\
power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,memory.total,\
persistence_mode,compute_mode --format=csv;
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv;
  nvidia-smi -q -d PERFORMANCE | head -25"'
```

- **`clocks.sm` 断言**：负载中采样的 `clocks.sm` 必须 `>= 0.9 * clocks.max.sm`。
  依据：本次实测空闲时 `clocks.sm=427 MHz`、`clocks.max.sm=3090 MHz`（7.2× 差距，P8）。
  WSL 下 `nvidia-smi -lgc` 锁定不可用（需管理员权限且 WSL 支持有限），
  因此走 **2 s clock warmup + 断言 + 记录**，不写"已锁定"。
  断言失败 → `status=INVALID_CLOCK`。
- **compute process 洁净性**：测量前后 `--query-compute-apps` 都必须为空
  （沿用 M1K-CUDA 验收的 `compute_processes_before_empty/after_empty` 写法，
  见 `.../integration-w6-qwen35-bf16-cuda-buffer-ingestion-final/validation.json`）。
- **驱动漂移**：实测 KMD `616.92` / CUDA UMD `13.4`；文档记录 `610.62` / `13.3`。
  每轮证据必须记实际值，并允许跨驱动版本的数字**不可比**（`comparable=false`）。

### 4.3 鲲鹏 920B（只作原生功能参照）

- `docs/RESOURCE_MATRIX.zh-CN.md:12`：openEuler 22.03、HiSilicon、**2 vCPU**、KVM。
- `configs/development_lock.yaml:310`：`ecs_native_sve256_is_kvm_functional_and_disassembly_evidence_not_performance_threshold`。
- 协议：920B 数字写入 `reference_native` 区块，**永久** `gate_eligible=false`，
  不得进入 `baseline.ratio`，不得作为任何门槛分母。

### 4.4 QEMU（**数字禁止**）

- `docs/RESOURCE_MATRIX.zh-CN.md:94-105` 已列出"不能用于"清单。
- 协议：`evidence_class` 枚举为 `functional_only | performance`；QEMU 只能取 `functional_only`，
  证据里 `numbers_rejected=true`。检测方式：`QEMU_CPU` 环境变量存在或 host 为 `qemu-*` 时自动拒绝。

### 4.5 AMD gfx1036

- `docs/RESOURCE_MATRIX.zh-CN.md:167-171`：WSL 无 `/dev/kfd`、无 ROCm/HIP/HSA。
- 允许出 L2 指标（`lower_seconds`、`llvm_ir_bytes`）；
  `kernel_seconds` / `tokens_per_second` 必须为 `null`，并带
  `reason: BLOCKED_DEVICE`，`native_execution_claim: false`。

---

## 5. 基线选择策略

### 5.1 CPU

**首选（`T0_external_blas`）**：`numpy 2.5.1` 内置的 `scipy-openblas 0.3.33.112.0`。

理由：本次实测确认它是本机唯一带真实 SIMD 的优化 BLAS；系统 `/usr/lib/libblas.so.3.12.0`
是 **netlib reference**（`nm -D | grep -c openblas == 0`），只可作 correctness 对照。

强制要求：

```python
# 基线线程数必须与 PyPTO-X 侧一致，并在测量前后各读回一次
import ctypes, numpy as np
lib = ctypes.CDLL(".../numpy.libs/libscipy_openblas64_-017048f4.so")
lib.scipy_openblas_set_num_threads64_(6)
assert lib.scipy_openblas_get_num_threads64_() == 6
```

- `blas_threads` 必须 == `run_local_heavy.sh --max-cpus`（默认 6），并记 `threads_verified: true`。
- 基线 shape 必须与 PyPTO-X 被测 case **逐维一致**（含 batch、layout、dtype）。
  不允许"shape 不同就比 GFLOPS"。
- 基线必须用 `np.matmul` / `np.dot`（走 BLAS gemm），不许用 `np.einsum` 的 Python 回退路径。

**明确排除**：系统 `libblas.so.3`（netlib reference）、任何纯 Python 参考实现。

**降级链**

| tier | 名称 | 可用性（本次实测） | 可比性 |
|---|---|---|---|
| `T0_external_blas` | scipy-openblas（numpy） | **可用** | 时间比可跨实现比较 |
| `T1_vendor_gpu_lib` | cuBLAS / hipBLAS | **不可用**（5080 WSL 无 libcublas；gfx1036 `BLOCKED_DEVICE`） | — |
| `T2_self_version_ratio` | PyPTO-X vs PyPTO-X（上一冻结 SHA，同机同 affinity 同日） | 可用 | **只有比值可比，绝对值不可比** |
| `T3_absolute_only` | 无基线 | 可用 | 只报绝对数，标 `UNGATED` |

`baseline.tier ∈ {T2, T3}` 的数字：

- **禁止**跨后端排名；
- **禁止**写成"达到峰值 X%"；
- **禁止**进入 G3 比值门槛。

**可选：独立手写参考**。可以做，但必须满足
(a) 与 PyPTO-X 实现路径不同源；
(b) 用 `objdump -d` 证明自己确实生成了目标 ISA 指令（YMM/ZMM 计数 > 0，参照
`python/tests/ut/pypto_x/test_cpu_avx2.py:1021` 已有做法）；
(c) 明确标 `tier=T2_self_version_ratio`，因为它是我们自己写的，不是第三方基线。

### 5.2 CUDA

- `baseline.name = cublas`，`baseline.availability = MISSING`
  （实测 `ldconfig -p | grep -i cublas` 为空；`/usr/lib/wsl/lib/` 无 `libcublas*`）。
- `baseline.name = cudart`，同样 `MISSING`。
- 因此 CUDA 侧当前**只能**走 `T2`（PyPTO-X 自比）与 `T3`（只报绝对值）。
- 若用户后续授权安装 Toolkit（本地安装属"安装软件"，需用户批准），
  `T1_vendor_gpu_lib` 自动启用，G3 比值门槛才可在 CUDA 上成立。
- 在补上 event 计时（`cuEventCreate/Record/ElapsedTime`）之前，
  CUDA 的 `kernel_seconds` 一律 `null`，`gflops_scope = end_to_end`。

### 5.3 无可用基线时的统一降级

```text
1. 若 baseline.availability ∈ {MISSING, BLOCKED_DEVICE, BLOCKED_TOOLCHAIN}
   → baseline.tier = T3_absolute_only，gate.relative = UNGATED
2. 若存在本 task 的上一冻结版本证据（同 host fingerprint）
   → 升级为 T2_self_version_ratio，可出 ratio，但仍 UNGATED 于门槛
3. 任何情况下都不得用另一个 host 的历史数字当基线
```

`host_fingerprint` 定义为：

```text
sha256(cpu_model | nproc_online | cpu_affinity_list | virtualization |
       cpufreq_available | gpu_name | gpu_driver_version | mem_total_mib)
```

`host_fingerprint` 不同的两条证据之间，**禁止**计算 `ratio`。

---

## 6. 门槛阶梯与冻结流程

### G1 · correctness（已冻结，本协议不新增要求）

| 项 | 内容 |
|---|---|
| 前置 | 无 |
| 通过条件 | 现有 `validation.json`：`status=PASS`、`source_clean=true`、`actual_head == expected_head`、`full_pypto_x` 汇总 PASS |
| 证据字段 | 沿用现有 schema（`schema_version/task/branch/worktree/expected_head/actual_head/source_clean/scope`） |
| 谁批准 | 委派验收 agent（现有协议，`AGENTS.md:29`） |
| 失败回退 | 实现 agent 修复 → 从新 integration HEAD 重开验收 worktree |

### G2 · stability（**本协议新增**）

| 项 | 内容 |
|---|---|
| 前置 | G1 PASS |
| 通过条件 | 同一构建、同一 host、**同一 `cpu_affinity`**，连续 **3 轮**独立进程测量，每轮 R=21；每轮 `cv <= 3.4 表值` 且 `p95/median <= 3.4 表值`；三轮中位数离散度 `(max_median - min_median)/median <= 0.10` |
| 证据字段 | `stability.rounds=3`、`stability.per_round[].{min,median,p95,cv,p95_over_median,verdict}`、`stability.median_spread_ratio`、`stability.verdict` |
| 谁批准 | 实现 agent 自证 + 委派验收 agent 抽查重跑一轮（必须复现） |
| 失败回退 | `INVALID_VARIANCE` / `INVALID_WARMUP` / `ENV_NOISY` / `INVALID_CLOCK` → **整轮数字作废并重跑**；禁止"多跑几次取最好" |

### G3 · relative-to-baseline（**本协议新增，当前唯一可用的比较级**）

| 项 | 内容 |
|---|---|
| 前置 | G1 PASS **且** G2 PASS **且** `baseline.tier ∈ {T0, T1}` **且** `dispatch_dominated == false` |
| 比值口径 | `ratio = pypto_x_median / baseline_median`（时间比，越小越好）。用 **median**，禁止用 min |
| 候选阈值来源（**只记公式**） | `candidate_ratio = median(measured_ratio over 3 rounds)`；`frozen_ratio = max(candidate_ratio, 1.0) * 1.10`。阈值 key 必须是五元组 `(op, shape_tuple, dtype, isa, cpu_affinity)`，**禁止全局单阈值** |
| 证据字段 | `baseline.{tier,name,availability,median,ratio}`、`gate.relative`、`ratios[].{key,candidate_ratio,frozen_ratio,approved_by,approved_at}` |
| 谁批准 | 验收 agent 出 `candidate_ratio`；**用户批准**后才写入 lock 文件 |
| 失败回退 | `ratio > frozen_ratio` → 标 `REGRESSION_SUSPECT`，允许**一次**复测（同 host、同 affinity）；复测仍超 → 开优化 task；**禁止直接上调阈值** |

### G4 · absolute threshold（**本阶段明确不设置**）

| 项 | 内容 |
|---|---|
| 前置 | 至少一个后端达到 `execution_model = fused`（即不再是 op-by-op 派发）**且** `baseline.tier ∈ {T0, T1}` **且** G3 已连续 2 个冻结版本无回归 |
| 通过条件 | 本阶段**不定义**。`configs/model_targets.yaml:84` 的 `defer-until-first-runnable-backends` 保持不变 |
| 谁批准 | **用户** |
| 失败回退 | 不适用（尚未启用） |

### 回退矩阵

| 症状 | gate | 动作 | 谁决定 |
|---|---|---|---|
| `cv > 上限` 或 `p95/median > 上限` | G2 | 整轮作废、重跑；连续 3 轮失败 → 上报环境问题 | 实现 agent |
| `clocks.sm < 0.9 × clocks.max.sm` | G2 | 延长 warmup 重跑；仍失败 → 记录 `clock_unlocked=true` 并降级为 `T3` | 实现 agent |
| `max(load1) >= online_cpus` 或 PSI ≥ 25 | G2 | 换时段重跑；不得删掉慢的那几轮 | 实现 agent |
| resource-lock 返回 75/69 | — | **等待**，不得绕过、不得裸跑（`docs/LOCAL_RESOURCE_POLICY.zh-CN.md:11-14`） | 协调者 |
| supervisor 返回 70（OOM/高压） | — | 该轮作废；缩规模或换时段 | 实现 agent |
| `dispatch_dominated == true` | G3 | 该数字不进比例门；先做 dispatch 优化或补 event 计时 | 实现 agent + 用户 |
| `ratio > frozen_ratio` | G3 | `REGRESSION_SUSPECT` → 一次复测 → 开优化 task | 实现 agent |
| `host_fingerprint` 变化 | G3 | `ratio` 强制作废，重测基线 | 实现 agent |

---

## 7. 证据格式

完整机器可读草案见同目录 **`perf_protocol.proposed.yaml`**。
以下为内嵌摘要（字段名与 YAML 一致）：

```yaml
schema_version: 1
artifact_kind: pypto-x.perf-evidence
task: qwen35-perf-threshold-protocol
agent: <agent-id>
timestamp_utc: <ISO-8601>

git: {repository: upstream/pypto, head: <sha>, branch: <str>, dirty: false}

host:
  class: local | nvidia-5080-wsl2 | amd-igpu-gfx1036 | kunpeng-920b-ecs | qemu-aarch64
  fingerprint: <sha256>
  uname: <str>
  nproc_online: 12
  cpu_affinity: [0,1,2,3,4,5]
  cpu_model: "AMD Eng Sample: 100-000000956-50_Y"
  virtualization: kvm
  cpufreq_available: false
  governor: null
  frequency_control: unavailable_kvm_guest
  mem_total_mib: 30032
  mem_available_min_mib: <int>
  load1_max: <float>
  cpu_psi_avg10_max: <float>
  gpu:
    name: "NVIDIA GeForce RTX 5080"
    driver_kmd: "616.92"
    cuda_umd: "13.4"
    clocks_sm_mhz: <int>
    clocks_max_sm_mhz: 3090
    clock_ratio: <float>
    compute_processes_before: []
    compute_processes_after: []

resource_lock:
  lock: local
  runner: scripts/resource/run_local_heavy.sh
  task: <str>
  monitor_log: <path>
  cpu_affinity: [0,1,2,3,4,5]
  max_cpus: 6

environment_freeze:
  toolchain: {clang: "22.1.8", gcc: "16.1.1", python: "3.12.10", cmake: "4.4.2"}
  threads: 6
  blas: {name: scipy-openblas, version: 0.3.33.112.0, threads: 6, threads_verified: true}
  qwen_weights_loaded: false

measurement:
  level: L0_op | L1_model | L2_compile
  execution_model: op_by_op_dispatch | fused | not_applicable
  dispatch_count: 4532   # graph v2；v3 为 4550
  dispatch_dominated: <bool>
  warmup_iterations: 5
  measured_iterations: 21
  clock_warmup_seconds: 2.0

statistics:
  unit: seconds | tokens_per_second | gflops
  min: <float>
  median: <float>
  p95: <float>
  max: <float>
  stdev: <float>
  cv: <float>
  p95_over_median: <float>
  outliers_marked: <int>
  outlier_ratio: <float>
  verdict: VALID | INVALID_VARIANCE | INVALID_WARMUP | ENV_NOISY | INVALID_CLOCK | UNSTABLE

metrics:
  wall_seconds_median: <float>
  kernel_seconds_median: <float|null>
  dispatch_seconds_median: <float|null>
  tokens_per_second: <float|null>
  ops_per_second: <float|null>
  gflops: <float|null>
  gflops_scope: kernel_only | end_to_end
  flop_accounting: "<formula string>"
  arithmetic_intensity_flop_per_byte: <float|null>
  memory_bound: <bool>
  transcendental_ops_per_second: <float|null>
  peak_rss_mib: <float>
  peak_device_memory_mib: <float|null>

compile:
  lower_seconds: <float|null>
  compile_seconds: <float|null>
  artifact_bytes: <int|null>
  cache_state: cold | warm | not_implemented

baseline:
  tier: T0_external_blas | T1_vendor_gpu_lib | T2_self_version_ratio | T3_absolute_only
  name: scipy-openblas | cublas | pypto-x@<sha> | none
  availability: available | MISSING | BLOCKED_DEVICE | BLOCKED_TOOLCHAIN
  median: <float|null>
  ratio: <float|null>
  comparable: <bool>
  incomparable_reason: <str|null>

gate:
  correctness: PASS | FAIL | NOT_RUN
  stability: PASS | FAIL | NOT_RUN
  relative: PASS | FAIL | UNGATED
  absolute: UNGATED
  eligible_for_threshold: false

notes: []
```

**证据落盘约定**（沿用现有 `_meta` 布局）：

```text
../worktrees/_meta/pypto-x/<task>/
  perf-evidence/<case-key>.json     # 单个 case 的上面这份 JSON
  resource-usage/...                # run_local_heavy.sh 的 monitor 日志
  validation.json                   # 汇总 + 引用
```

---

## 8. 反模式清单（含**检测方法**）

| # | 反模式 | 检测方式 | 处置 |
|---|---|---|---|
| 1 | QEMU/KVM 数字当门槛 | `host.virtualization == kvm` 且 `host.class ∈ {qemu-aarch64, kunpeng-920b-ecs}` | `gate_eligible=false`；QEMU 强制 `numbers_rejected=true` |
| 2 | 跨机器直接比 | 两条证据 `host.fingerprint` 不同却有 `baseline.ratio` | 拒绝该 ratio |
| 3 | 把 Python 解释开销算进 kernel 性能 | `execution_model == op_by_op_dispatch` 且 `dispatch_seconds` 缺失，或 `dispatch_dominated == true` 仍报 `gflops_scope: kernel_only` | 强制补 dispatch floor；此前 `UNGATED` |
| 4 | 无 warmup 报数 | `warmup_iterations < 5` 或 `statistics.verdict == INVALID_WARMUP` | 数字作废 |
| 5 | 单次测量 | `measured_iterations < 11` | `gate_eligible=false` |
| 6 | 混淆 bf16/fp32 算力口径 | `flop_accounting` 字段缺失，或 bf16 matmul 用了 `1 * B*M*N*K` | 数字作废，按 `2 * B*M*N*K` 重算 |
| 7 | 把静态 lowering 说成运行性能 | `scope.native_execution_claim == false` 或 AMD 无 runtime 却报 `kernel_seconds != null` | 只允许 L2 指标 |
| 8 | 6 CPU 与 12 CPU 的两次结果相比 | 两条证据 `resource_lock.cpu_affinity` 列表不一致 | 拒绝该 ratio |
| 9 | 拿系统 netlib BLAS 当性能基线 | `baseline.name == system-libblas` 且 `tier == T0` | 降级为 correctness-only |
| 10 | 忽略 GPU 时钟状态 | `clocks_sm_mhz / clocks_max_sm_mhz < 0.9` | `INVALID_CLOCK` |
| 11 | 把 `lower_seconds` 当"编译速度达标" | AVX2 路径下 `compile_seconds` 与图规模无关（`cpu_avx2.py:32` 固定源码） | 分开报 lower/compile，各自单独门槛 |
| 12 | 用 Ascend ST `test_perf.py` 当性能门槛 | 它断言的是 kernel detail 字段一致性（`python/tests/st/test_perf.py:334-349`），且依赖 torch/torch_npu | 不作为依据 |
| 13 | 只看 logits 不看 state | 属 G1 范围 | 引用 `configs/model_targets.yaml:86-90` 的 required_reports |
| 14 | 忽略 `MemorySwapMax=0` 导致的 OOM | monitor 返回 70 | 该轮作废，不得"降级裸跑" |
| 15 | 拿旧驱动的 5080 数字与新驱动比 | `gpu.driver_kmd` 不同 | `comparable=false` |
| 16 | 用 `T>1` 的 GDR 图规模当"融合性能" | `qwen35.py:1854-1855` 明写 `t_gt_1_gdr: static_sequential_ssa_unroll`、`operation count grows linearly with steps` | 标 `unroll=true`，不得称融合 |

---

## 9. 仍需用户决策的问题

1. **本机绝对门槛的终局**：本机是 KVM guest 且无 `cpufreq`（`cpufreq_available=false`）。
   是否接受"本机只做纵向（跨版本）可比、绝对数值永久 `UNGATED`"？若希望有绝对门槛，
   需要一台裸机/可锁频的机器。
2. **5080 CUDA Toolkit 安装授权**：没有 `nvcc`/`libcublas`/`libcudart`，CUDA 只能停在 `T3`。
   是否授权安装 CUDA Toolkit 或单独放置 `libcublas`？（安装软件当前需用户明确批准）
3. **CPU 基线是否需要独立手写 SIMD 实现**：与 scipy-openblas 比 gemm 时，
   OpenBLAS 的打包/分块优势会掩盖我们自己的问题。是否要额外投入一个自研基线（`T2`）？
4. **阈值承载位置**：新建 `configs/perf_lock.yaml`，还是扩展 `configs/development_lock.yaml`？
5. **prefill T 的范围**：`T>1` 的 GDR 是静态 SSA 展开，op 数线性增长。按
   `_gdr_step` 每步 21 个 op × 18 个 GDR 层测算：
   `ops(T) ≈ 4550 + 378 × (T-1)`（graph v3；v2 基数为 4532）→ `T=16 ≈ 10,220`、`T=128 ≈ 52,556`、`T=512 ≈ 197,708`。
   （**这是估算，不是实测**；`_gdr_step` 在 `qwen35.py:3265-3445` 内恰有 21 个
   `builder.op(` 调用且无内部分支，已逐行核对。落地前应用一个只数 op、不做 lower 的
   轻量 driver 复核。）
   是否接受 T=128 的 lowering 成本？还是首轮只做 `T=1 decode + T=16 prefill`？
6. **`gamepc` 锁的边界**：GPU-only 测量不持锁，但单轮重复测量若长时间占满 GPU（> 5 min），
   是否需要通知/持锁？
7. **AMD gfx1036 的 L2 指标是否纳入 G3**：在拿到 `/dev/kfd` 之前，
   是否把静态 `lower_seconds` / `llvm_ir_bytes` 当作可回归指标？
8. **文档漂移修正归属**：5080 驱动从 `610.62 / 13.3` 变为 `616.92 / 13.4`，
   涉及 `docs/RESOURCE_MATRIX.zh-CN.md:40` 与 `HANDOFF.zh-CN.md:233`。
   本任务只读控制仓，需要父 agent 归档更新。

---

## 10. 引用索引

所有引用均来自**只读**检查，未修改 `upstream/*`、integration worktree 或其他 agent 的 worktree。

### 10.0 可复现性、行号漂移与配套文件

本目录同时产出三个可复跑的验证件：

| 文件 | 作用 |
|---|---|
| `citation_check.py` | 用**锚点字符串**在真实文件里重新定位报告中每条引用，输出解析出的行号 + 文件 sha256；`--output validation.json` |
| `environment_probe.py` | 只读环境探测（`lscpu` / `ldconfig` / `nm` / `numpy.show_config` / `nvidia-smi` 查询），产出 `environment-probe.json` |
| `validation.json` | 汇总：启动协议字段、smoke、引用校验（165/165 OK）、独立搜索计数、GDR op 计数、op 直方图 |

**执行方式**（均为轻量只读命令，不需要 `local` 锁）：

```bash
python3 citation_check.py --output validation.json      # 退出码 0 = 全部引用命中
python3 environment_probe.py --output environment-probe.json
```

**行号漂移告警（本任务实际观测到）**

控制仓由其他 agent 并发写入，**行号不稳定**。本任务执行期间实测：

| 文件 | 变化 |
|---|---|
| `configs/development_lock.yaml` | `known_limits:` 从第 **1132** 行移到第 **1141** 行；文件从 1143 行增长到 1152 行 |
| `AGENTS.md` | 从 62 行缩短到 59 行，`## Subagent 协议` 从第 37 行移到第 **29** 行 |
| `scripts/resource/run_local_heavy.sh` | `max_cpus` 默认值从第 87-89 行移到第 **76-80** 行 |
| `docs/20-planning/0002-…-w8a8-linear-contract.zh-CN.md` | 目标行从 378 移到 **399** |

integration worktree 固定在 `bcf9516e6`，其行号稳定；控制仓行号**只是快照**。

> **协议建议（已写入 `perf_protocol.proposed.yaml` 的 `evidence_schema`）**：
> 性能证据引用源码时，**必须同时记录文件 sha256**，行号只作辅助。
> 本报告第 10.1–10.3 节的行号均以 `validation.json` 中
> `snapshot.control_repo.head = 4b8bac7e2a41cd2daad12bfa6d386a0863c1c15f` 为准。

### 10.1 integration worktree 源码（`/home/chiro/projects/pypto/worktrees/pypto-x/integration` @ `bcf9516e6`）

| 路径 | 行 | 用途 |
|---|---|---|
| `python/pypto/portable/qwen35.py` | 47-73 | Qwen3.5-0.8B shape metadata（hidden/intermediate/layers/heads/conv） |
| | 421-470 | `_decoder_shape_profile`：`model` 与 `synthetic_test` 两种 profile |
| | 434 | `vocab = 248320` |
| | 1496-1514 | `build_qwen35_text_decoder_graph(batch, steps, past_length, ...)` 签名 |
| | 1854-1855 | `t_gt_1_gdr: static_sequential_ssa_unroll` / `operation count grows linearly with steps` |
| | 2571 | `for time_index in range(steps)`（GDR 静态展开） |
| | 3238-3252 | `_gdr_time_slice` docstring（intentionally uses static SSA unrolling） |
| | 3265-3445 | `_gdr_step`：恰 21 个 `builder.op(`，无分支（脚本复核） |
| | 3446 / 3558 | `_build_gdr_recurrent_state` / `build_qwen35_gdr_recurrent_state` |
| | 2721 / 2793 / 2804 | `build_causal_conv1d_state` / `build_causal_conv1d` / `build_qwen35_gated_delta_conv_state` |
| | 889 | `build_stable_softmax` |
| | 1432 / 1436 | `parameter_count_estimate`（~873M）/ `parameter_byte_count` |
| `python/pypto/backends/cpu/runtime.py` | 81-152 | `CpuScalarRuntime.launch` |
| | 133-134 | 逐 op Python 解释主循环 |
| | 509-661 | `_reshape_value` … `_gather_value`（layout/indexing） |
| | 671 / 710 / 717 / 743 | `_math_scalar` / `_math_value` / `_binary_value` / `_unary_value` |
| | 750 / 775 / 824 | `_broadcast_value` / `_where_value` / `_reduce_value` |
| | 881-925 | `_matmul_value`，`:891-902` rank 2–4 + exact batch，`:915-924` 纯 Python 三重循环 |
| `python/pypto/backends/cpu/vector/runtime.py` | 462 / 966 | `_execute_operation` / `_execute_matmul`（同为 Python 派发；全文无 ctypes） |
| `python/pypto/backends/cpu/x86/avx2.py` | 11 | `import ctypes` |
| | 113-125 | dtype→ctype 映射与 `_POINTER` |
| `python/pypto/compiler/targets/cpu_avx2.py` | 32 | `_NATIVE_SOURCE`（固定通用 kernel 库） |
| | 89-325 | 16 个原生符号（`ptx_avx2_copy_f32` … `ptx_avx2_matmul_bf16`） |
| | 612-666 | `_compile_native` 调 clang |
| | 695-698 | 规范化编译命令 `clang -shared -fPIC -O2 -std=c99 -mavx2 [-mfma] <src> -lm -o <out>` |
| `python/pypto/compiler/targets/cuda.py` | 919-931 | `render_ptx`：每 op 一个 `_Kernel`，拼成单 module |
| `python/pypto/backends/cuda/runtime.py` | 398 | `module.get_function("px_op{index}")` |
| | 410-417 | 每 op 一次 `launch_kernel` |
| | 425 / 435 | 标量回退同步 / 整图 `context_synchronize()` |
| `python/pypto/compiler/api.py` | 80-117 | `compile_program` = lower + compile |
| `python/pypto/compiler/cache.py` | 16-33 | `ArtifactCache` 仅内存（无持久 cache） |
| `python/tests/ut/pypto_x/qwen35_runtime_binding_metadata_driver.py` | 20-21 / 28-32 / 57-59 | `ru_maxrss` 读法、RSS delta 字段写法 |
| `python/tests/ut/pypto_x/amd_gfx1036_c3_static_lower_driver.py` | 32-38 | `time.monotonic()` 包 `compiler.lower` + `ru_maxrss/1024.0` |
| | 64-67 | `resource.{lower_seconds,max_rss_mib}` 字段命名 |
| `python/tests/ut/pypto_x/test_qwen35_full_decoder_connectivity.py` | 158-161 | `past ∈ {0,4096}` 下 op 数 < 6000、JSON < 3 MB |
| `python/tests/ut/pypto_x/test_cpu_avx2.py` | 1021 | objdump 反汇编检查既有做法 |
| `python/tests/st/test_perf.py` | 15 / 30-31 / 334-349 | Ascend ST "perf" 实为字段一致性断言；依赖 torch/torch_npu |

### 10.2 控制仓（`/home/chiro/projects/pypto/pypto_x`）

| 路径 | 行 | 用途 |
|---|---|---|
| `HANDOFF.zh-CN.md` | 201-262 | 第 8 节资源实况（本机/5080/AMD/鲲鹏/QEMU） |
| | 233 | 5080 驱动记录 `610.62`（**已漂移**） |
| | 243 | Driver API + PTX JIT，无 nvcc |
| | 251 | 920B = 2 vCPU KVM guest |
| | 427-446 | 第 11 节待确认事项 |
| `AGENTS.md` | 29 | subagent 协议（一 task/分支/worktree，长等待） |
| | 49 | 性能门槛延后到首批后端完成 |
| | 57 / 58 / 59 | 验收 worktree 隔离 / QEMU 不得作性能结论 / 920B 不形成门槛 |
| `docs/RESOURCE_MATRIX.zh-CN.md` | 9 | 本机：heavy 需 `local` 锁，不形成性能门槛 |
| | 10 | 5080：GPU-only 无需 `gamepc`；当前为 correctness kernel |
| | 11 | AMD：静态 C3 PASS，runtime `BLOCKED_DEVICE` |
| | 12 | 920B：2 vCPU KVM，不代表裸机性能 |
| | 13 / 94-105 | QEMU「不能用于」清单 |
| | 40 / 46 | 5080 驱动 `610.62`（**已漂移**）/ 缺 nvcc 等 |
| | 121 | GDR 固定 `[1,16,128,128]` FP32 state 验收记录 |
| | 195-202 | 资源使用原则 |
| `docs/LOCAL_RESOURCE_POLICY.zh-CN.md` | 11-14 | 75/69 必须等待，禁止绕过 |
| | 16-27 | 必须申请 `local` 的条件 |
| | 46-58 | 默认策略表（8192/4096/6 CPU 等） |
| | 69 | supervisor 终止返回 70 |
| | 79-88 | subagent 启动字段 |
| | 94-96 | GPU-only 不申请 `gamepc` |
| `configs/model_targets.yaml` | 12-16 | `generation`：prompt/decode 长度候选 |
| | 65-70 | `backend_order` |
| | 82-97 | `validation`：`performance_threshold_policy` 及 required_reports |
| `configs/development_lock.yaml` | 5 | `integration_commit` = `bcf9516e6…` |
| | 8-21 | `resource_policy` |
| | 310 | 920B 非性能门槛的 freeze 记录 |
| | 1004 | `performance_threshold: false` |
| | 1118-1123 | 权重授权、revision、sha256、字节数 |
| | 1141-1152 | 当前 `known_limits` |
| `docs/20-planning/0001-…-mvp.zh-CN.md` | 244-257 | 正确性与报告要求；性能门槛延后 |
| `docs/20-planning/0002-…-w8a8-linear-contract.zh-CN.md` | 399 | 不做性能承诺与门槛 |
| `scripts/resource/run_local_heavy.sh` | 76-80 | `max_cpus` 默认 `(online+1)/2` |
| | 119-133 | `resource-lock run` → `systemd-run` → monitor 链路 |
| `scripts/resource/monitor_local_heavy.py` | 146-148 | 子进程 `setsid` + `sched_setaffinity` |
| | 191 | `allowed[:max_cpus]` 选核 |
| | 204-208 | 强制线程环境变量 |

### 10.3 证据文件

| 路径 | 用途 |
|---|---|
| `../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/qwen35-full-lower.json` | 唯一已有 `lower_seconds=1.9299` / `max_rss_mib=95.38` / `llvm_ir_bytes=10510232` / op 直方图 |
| `../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/validation.json` | 现有 `validation.json` 字段 schema 参考 |
| `../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/resource-qwen-lower.log` | monitor 日志格式（`event=start|sample|finish`） |
| `../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cpu-buffer-ingestion-final-r2/qwen-{scalar,avx2,avx512,sve256}.json` | L1 无计时字段的证据空白 |
| `../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cuda-buffer-ingestion-final/validation.json` | CUDA 371 input / 51 output / 48 state / compute process 前后为空 |
| `../worktrees/_meta/pypto-x/qwen35-perf-threshold-protocol/logs/20260910T080316Z/smoke.log` | 本任务一次性 smoke |

### 10.4 本次只读探测命令（无 benchmark、无 heavy、无安装）

```text
lscpu / getconf _NPROCESSORS_ONLN / /proc/cpuinfo / /proc/meminfo
systemd-detect-virt
ls /sys/devices/system/cpu/cpu0/           # 确认无 cpufreq
ldconfig -p | grep -Ei 'openblas|mkl|blas|lapack'
nm -D /usr/lib/libblas.so.3.12.0 | grep -ci openblas      # → 0（netlib）
ldd .../numpy/_core/_multiarray_umath*.so | grep -i blas  # → libscipy_openblas64_
nm -D .../numpy.libs/libscipy_openblas64_-*.so | grep -E 'openblas_set_num_threads'
python3 -c "import numpy; numpy.show_config(mode='dicts')"
resource-lock status
ssh 192.168.101.5 'wsl.exe -e bash -lc "ldconfig -p | grep -i -E cublas; \
  ls /usr/lib/wsl/lib/; nvidia-smi --query-gpu=... ; \
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; \
  nvidia-smi -q -d PERFORMANCE"'
```

**未执行**：任何 pytest / benchmark / 计时测量 / 图形构建 / 大 shape lowering / 软件安装 / 权重加载。

---

## 11. 一句话结论

当前 PyPTO-X 的所有可运行后端都是**逐 op 派发**（CPU 是 Python 解释，AVX2/AVX-512 是每 op 一次
ctypes 调用，CUDA 是每 op 一个 PTX kernel 加一次全同步），且**没有任何 device-side 计时**、
**没有 cuBLAS 基线**、**本机无法锁频**、**L1 从未被计时**。
因此本阶段唯一正确的动作是：**冻结"怎么测"（分层 + 统计 + 环境 + 基线降级 + 证据 schema + 反模式检测），
把绝对门槛明确留空**；先把 `dispatch_seconds`、kernel event 计时和 `compile_seconds` 三个缺口补上，
再谈倍率。
