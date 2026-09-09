# PyPTO-X 可用资源与测试矩阵

更新日期：2026-09-09（Asia/Shanghai）

## 资源总览

| 资源 | 当前状态 | 主要用途 | 当前实测/已知信息 | 限制 |
|---|---|---|---|---|
| 本地开发机 | 可用；heavy 需全局 `local` 锁；AVX2/AVX-512 Qwen parity 与 M1I decoder PASS | Core IR、ABI、CPU scalar/x86、PTO simulator、QEMU | x86_64，12 vCPU，Clang 22.1.8，GNU objdump 2.47；AVX2 `474 passed`；AVX-512 `483 passed, 7 skipped`；M1I 全量 `491 passed, 7 skipped`，真实 profile 4,532 ops 的 scalar/vector/AVX lowering+compile 与 synthetic 执行通过 | full pytest、大 shape lowering/compile 与并行构建必须走 `scripts/resource/run_local_heavy.sh`；默认保留 4 GiB、最多 6 CPU；当前仅为正确性/汇编证据，不形成性能门槛 |
| RTX 5080（`192.168.101.5`） | 在线；GPU 由 PyPTO-X 独占；CUDA C2 与 M1I synthetic PASS | NVIDIA CUDA/PTX/NVVM 验证 | WSL2、RTX 5080 16,303 MiB、CC 12.0、KMD 610.62/CUDA UMD 13.3；Driver/PTX 已通过 math/layout/indexing/composites，并执行24层 synthetic decoder | GPU-only 无需 `gamepc` 锁；host-heavy 才持锁。无 `nvcc`/NVRTC/CUDART/SDK headers/PyTorch/Triton；当前是 correctness kernel，不是 fusion/性能结论 |
| AMD 6750GRE 12G | 暂未接入 | AMD HIP/ROCDL、wave 和显存测试 | 当前机器 `lspci` 未发现该卡，`rocminfo/rocm-smi` 不可用 | 接入前不能声明 ROCm 支持或性能；具体 gfx target 以 `rocminfo` 为准 |
| 鲲鹏 920B ECS（SVE256） | 按量实例运行中 | 原生 AArch64/SVE256 功能、汇编，后续受控性能探测 | openEuler 22.03、HiSilicon、2 vCPU、GCC 10.3.1、KVM；HWCAP SVE=1、SVE2=0、VL=32；M1C1–M1I 功能/contract 与24层 synthetic decoder 已通过 | ECS 是 KVM guest，不代表裸机/整机性能；`iota/compare` 是 host-reference；按量计费，状态见 `~/tools/ecs-920B/state.env` |
| QEMU AArch64 | 可用 | AArch64/SVE/SVE2 功能和编译验证 | `qemu-aarch64` 11.0.3；已验证 `max,sve256=on` 可报告 SVE/SVE2，VL=32 bytes | 不能代表鲲鹏吞吐、缓存、内存带宽或指令时序 |

## 5080 WSL 连接方式

SSH 默认 shell 是 Windows `cmd`，直接执行 `uname`、`true` 等 Unix 命令会失败。统一使用：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 192.168.101.5 \
  'wsl.exe -e bash -lc "uname -a; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"'
```

2026-09-09 恢复后探测结果摘要：

```text
Linux GamePC 6.6.87.2-microsoft-standard-WSL2
NVIDIA GeForce RTX 5080, 16303 MiB, KMD 610.62, CUDA UMD 13.3
Compute capability 12.0, later probe 11389 MiB free
GPU memory used 4460 MiB, GPU util 2%, compute processes none
WSL: 24 CPU, MemAvailable about 30 GiB
Clang 18.1.3, CMake 3.28.3, Python 3.12.3
libcuda.so.1 present
nvcc/NVRTC/CUDART/CUDA SDK headers/PyTorch/Triton absent
```

建议安装阶段按顺序验证：

```bash
command -v nvidia-smi
command -v nvcc
command -v clang
command -v cmake
command -v python3
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

`nvcc`/NVRTC/PyTorch 缺失时，完整 Toolkit 路线标记 `BLOCKED_TOOLCHAIN`，但可以先用现有 `libcuda.so.1`、手写 PTX 和 Python `ctypes` 验证 CUDA Driver API 后端。不得把 Linux UAPI 的 `/usr/include/linux/cuda.h` 误认成 CUDA SDK header。

CUDA C1 已完成并由独立验收代理在 5080 上确认：FP32/BF16 elementwise、BF16 RNE、非末轴 reduction、全负 reduce-max、rank-2 matmul、K=0、zero-work/canary、多操作链、empty reduce-sum、重复 launch 与 context/module/allocation cleanup 均通过；验收前后无 compute process 残留。证据位于 `../worktrees/_meta/pypto-x/integration-w4-cuda-c1-final/validation.json`。由于 `nvcc` 缺失，统一 smoke 仍按规范标记 `BLOCKED_TOOLCHAIN`；这不否定不依赖 Toolkit 的 Driver/PTX 实测，也不构成 NVVM/Tensor Core/性能结论。

CUDA C2 在此基础上补齐 math、compare/iota/position、broadcast/where、layout/indexing、多输出 split、rank-3/4 matmul、scalar SSA 和 Qwen portable composites。独立验收为 `476 passed, 7 skipped`，GPU common vendor/ABI 定向测试 `45 passed`，CPU 联合回归 `260 passed`；5080 前后无 compute process 残留。证据位于 `../worktrees/_meta/pypto-x/integration-w4-cuda-c2-final/validation.json`。数学仍使用 PTX approximate 指令，布局/索引/batched matmul 仍是通用 correctness kernel。

M1I 在 exact integration HEAD `ec60f95979a56a9646d84f163912e677e9eb08ac` 上完成24层无权重 decoder connectivity。真实 `(B=1,T=1,past=4096)` profile 的 scalar、vector-common、AVX2、AVX-512、SVE256、GPU common 与 CUDA lowering/compile 均通过且 vector iteration domain 全部 compact；缩小但拓扑等价的24层 synthetic graph 已在 scalar、AVX2、AVX-512、QEMU SVE256、920B native 和 RTX 5080 Driver/PTX 实际执行通过。证据位于 `../worktrees/_meta/pypto-x/integration-w6-qwen35-decoder-connectivity-final-r3/validation.json`；这仍不代表带权整网推理或性能结论。

## QEMU SVE/SVE2 验证

本地已有：

```text
/usr/bin/qemu-aarch64
/usr/bin/qemu-aarch64-static
/usr/bin/qemu-system-aarch64
/usr/bin/aarch64-linux-gnu-gcc
/usr/bin/aarch64-linux-gnu-g++
/usr/aarch64-linux-gnu
```

功能测试命令约定：

```bash
QEMU_CPU=max,sve256=on \
  qemu-aarch64 -L /usr/aarch64-linux-gnu <aarch64-test-binary>
```

这里 `VL=32` 表示 32 字节，即 256 位 SVE 向量长度。`QEMU_CPU=max` 的实现特征不等于鲲鹏 920B；QEMU 结果只能用于：

- AArch64 二进制能否启动；
- SVE/SVE2 指令和 predicate 逻辑是否正确；
- 动态 VL、尾块和 ABI 测试；
- 编译器生成代码的基本合法性。

不能用于：

- SVE 性能排名；
- 鲲鹏缓存/NUMA/带宽结论；
- 生产吞吐或功耗估计。

## 鲲鹏 920B ECS 接入结果

2026-09-08 已按接入清单完成首轮验证：

- `aarch64` little-endian、HiSilicon、2 vCPU、openEuler 22.03、KVM；
- `getauxval + prctl`：SVE=1、SVE2=0、VL=32 bytes；
- 原生 GCC triple：`aarch64-linux-gnu`；
- FP32/BF16 19 元素 masked tail、guarded canary、全负 reduce-max、9×5×10 matmul：`PASS`；
- 原生热点反汇编：`whilelo=2`、`ld1w=3`、`st1w=4`、z=36、p=19、NEON v/q=0。
- M1C2a compact layout：FP32 transpose/slice SVE indexed gather、BF16 identity SVE u16 copy、BF16 reorder scalar fallback、未对齐 descriptor、tamper 与 canary 均 `PASS`。
- M1C2b compact indexing：split/concat、int32/int64 gather/embedding、native bad-index/hash/descriptor 与 wire version pairing 均 `PASS`；BF16 gather/embedding 为明确 scalar fallback。
- M1D portable composites：BF16 RMSNorm、stable Softmax、partial RoPE、GQA、decoder/attention 多操作链与 guarded canary 均 `PASS`。
- M1E functional Conv1D/state：integration HEAD `41523cfc3` 的静态 SVE ELF 在原生机器执行 concat/slice/cast/broadcast/mul/add/silu、guarded canary、Conv 输出和 `new_state` carry 均 `PASS`；未加载模型权重。
- M1F attention/KV：integration HEAD `2d7da37b9` 的 SVE v5/vector-plan v2 ELF 执行 guarded self-test、rank-4 按 head 调用 rank-2 runner、BF16 KV concat/state carry 与输入不变性均 `PASS`；这是正确性 fallback，不是 fused batched kernel。
- M1G GDR recurrent state：integration HEAD `962f422e1` 在原生机对固定 `[1,16,128,128]` 非零 FP32 state 执行 48 次 rank-2 runner 调用，guarded self-test、输入不变性和 prediction/state/output golden 均 `PASS`；chunk 仍是 sequential reference，T=128 尚未验收。

基础证据见 `../worktrees/_meta/pypto-x/integration-w4-gpu-common-final/logs/20260908T080242Z-ecs-920b-native-validation.json`；M1C2a/M1C2b 证据分别见 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2a-final/validation.json` 与 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2b-final/validation.json`；M1D/M1E/M1F/M1G 证据见对应的 `integration-w6-qwen35-*-final/` 目录。

接入时发现并修复了 native runtime 错误注入 cross sysroot、以及通过 compiler basename 误拒原生 `cc` 的问题。当前结果可作为原生功能与汇编证据；性能门槛仍需单独设计、固定 affinity/频率/工作集并重复测量。

## 鲲鹏机器接入清单

拿到机器后先运行：

```bash
uname -a
uname -m
lscpu
getconf LONG_BIT
cc --version
clang --version
```

然后检查：

```text
HWCAP.SVE
HWCAP2.SVE2
实际 SVE vector length（应确认是否为 256 bit）
NUMA 节点、CPU affinity、内存带宽工具
```

测试分层：

1. `cpu-scalar` 正确性；
2. SVE256 predicate/尾块正确性；
3. SVE/SVE2 多 VL 兼容性（若机器支持）；
4. 最后才做性能和 PMU 测量。

## AMD 6750GRE 接入清单

卡接入系统后先记录：

```bash
lspci -nn | rg -i 'amd|vga|3d|display'
rocminfo
rocm-smi
hipcc --version
```

随后确认：

- ROCm 版本是否支持该消费级 RDNA2 GPU；
- `gfx` target 字符串；
- wavefront 是 32 还是 64；
- FP16/BF16/FP32、原子和矩阵指令能力；
- 12 GiB 显存是否足以承载测试工作集。

在这些信息确认前，AMD worktree 只做代码生成和静态检查，不做性能承诺。

## 资源分配

| 工作内容 | 首选资源 | 备用资源 | 验收结果 |
|---|---|---|---|
| Target ABI/Core IR | 本地 | 无 | IR snapshot、ABI 单测 |
| CCE 回归 | 有 Ascend/CANN 的机器 | PTO-ISA simulator | 既有 Ascend 测试不退化 |
| CPU scalar/x86 | 本地 | 无 | scalar golden、AVX 汇编检查 |
| SVE256/SVE2 | 鲲鹏 920B ECS（native） | QEMU（功能） | VL/predicate/尾块；性能结论另行冻结 |
| NVIDIA | 5080 WSL | 无 | CUDA artifact、运行结果 |
| AMD | 6750GRE 接入后 | 无 | HIP artifact、gfx/wave 检查 |
| 9B/GDR | 后期各目标 | 5080 或鲲鹏 | 先单算子，再报告模型覆盖率 |

## 资源使用原则

- 跨项目锁的权威协议是 `/home/chiro/projects/.resource-locks/README.md`。`local` 保护本机 heavy，`gamepc` 只保护远端 heavy CPU/host-memory；对应 heavy 阶段只有 `resource-lock run` 成功才算取得。RTX 5080 GPU 由 PyPTO-X 独占，GPU-only 工作不申请 `gamepc`。
- 本机 full suite、大 shape lowering/compile、并行构建或预计使用至少一半 CPU/4 GiB 内存的任务必须经 `scripts/resource/run_local_heavy.sh`。返回 75/69 时等待，不能降级为裸跑。
- heavy runner 默认以 user cgroup 限制 MemoryHigh/MemoryMax、禁用该任务 swap、限制 CPU quota/affinity，并由 supervisor 每 2 秒检查 `MemAvailable`、任务树 RSS/CPU、load 与 PSI；资源日志写入 `../worktrees/_meta/pypto-x/resource-usage/`。
- Smoke 测试不加载模型、不需要模型路径、不产生大权重文件。
- 5080 先用于 elementwise、softmax、matmul 和 GPU ABI，不直接从 9B 端到端开始。
- SVE 已完成 QEMU 与鲲鹏 ECS native 功能验证；ECS KVM 数字暂不直接作为生产性能门槛。
- AMD 卡接入前不能把 HIP backend 标成“可运行”；最多标成“编译路径开发中”。
- native compiler task 使用 `../worktrees/_meta/pypto-x/<task>/` 下独占的 `TMPDIR`、build、artifact 和日志目录，避免共享 `/tmp` 的容量/配额抖动，也避免不同 agent 共享未完成生成物。
