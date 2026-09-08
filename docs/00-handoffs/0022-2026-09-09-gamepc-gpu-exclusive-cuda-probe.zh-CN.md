# PyPTO-X GamePC GPU 独占与 CUDA 恢复探测

归档序号：`0022`

归档日期：2026-09-09（Asia/Shanghai）

状态：`RTX5080_GPU_EXCLUSIVE_DRIVER_PTX_BOOTSTRAP_READY`

## 资源规则勘误

用户明确：RTX 5080 GPU 可由 PyPTO-X 持续独占；`gamepc` 锁只协调 GamePC 上的大量 CPU/host-memory 占用。

因此：

- GPU-only 的 `nvidia-smi`、Driver API probe、GPU kernel 执行和少量同步 SSH 不申请 `gamepc`；
- CUDA host-heavy 编译、并行构建、大量 CPU 数据准备或明显占用远端内存的阶段，必须在该阶段持续持有 `gamepc`；
- 混合流程尽量拆成锁内 host-heavy 阶段与不持锁 GPU-only 阶段；
- 另一项目持有 `gamepc` CPU 锁时，PyPTO-X 仍可进行低 host 开销的独占 GPU 工作。

该规则已同步到 `/home/chiro/projects/.resource-locks/README.md` 与项目 AGENTS/资源/验收协议。

## 5080 恢复探测

2026-09-09 06:31 CST 的 GPU-only 只读探测：

```text
GPU              NVIDIA GeForce RTX 5080
VRAM             16,303 MiB total / about 4,460 MiB used
compute cap      12.0; later probe 11,389 MiB free
compute process  none
GPU util         about 2%
KMD              610.62
CUDA UMD         13.3
WSL              Linux 6.6.87.2, 24 CPU, about 30 GiB MemAvailable
tools            Clang 18.1.3, CMake 3.28.3, Python 3.12.3
driver library   /usr/lib/wsl/lib/libcuda.so.1
```

缺失：`nvcc`、NVRTC、CUDART、CUDA SDK headers、PyTorch、Triton。`/usr/include/linux/cuda.h` 是 Linux UAPI header，不是 NVIDIA CUDA SDK。

`nvidia-smi` 没有报告 compute process；约 4.46 GiB 占用视为 WSL/WDDM 显示或保留内存，当前不归因于计算任务。可用显存约 11.8 GiB，实际准入仍应在每次执行前复核。

证据：

```text
../worktrees/_meta/pypto-x/gamepc-cuda-probe-20260909/validation.json
```

## CUDA 启动路线

首个 CUDA backend 不等待 Toolkit 安装：

1. Python `ctypes` 加载 `libcuda.so.1`，完成 device/context/module/memory/launch ABI；
2. 从已冻结 GPU common IR 生成最小 PTX，并由 CUDA Driver API JIT；
3. 先验证 FP32/BF16 elementwise、reduce/Softmax 和 matmul 的小型正确性；
4. host-heavy 编译阶段按新规则申请 `gamepc`；纯 GPU 执行不申请；
5. 若后续确需 NVRTC/nvcc/PyTorch，先向用户报告并取得安装授权。

CUDA 可与 Qwen M1H position/control 并行，不能修改已冻结的 NVIDIA/AMD 共用 GPU ABI。
