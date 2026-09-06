# PyPTO-X 可用资源与测试矩阵

更新日期：2026-09-06（Asia/Shanghai）

## 资源总览

| 资源 | 当前状态 | 主要用途 | 当前实测/已知信息 | 限制 |
|---|---|---|---|---|
| 本地开发机 | 可用 | Core IR、ABI、CPU scalar/x86、PTO simulator、QEMU | x86_64，12 vCPU，Clang 22.1.8，GNU objdump 2.47，CMake，QEMU 11.0.3；暴露 AVX2/FMA、AVX-512F/BW/DQ/VL/VNNI/BF16 与 xsave/xgetbv；有 `aarch64-linux-gnu-gcc/g++` | `hypervisor`/KVM 环境，只用于功能、汇编和相对调试；当前数字不直接冻结为性能门槛；没有真实 NPU，本地 GPU 只有 Virtio 显示设备 |
| RTX 5080（`192.168.101.5`） | SSH 可达 | NVIDIA CUDA/NVVM/HIP 公共层验证 | Windows + WSL2；WSL Ubuntu 24.04.4；RTX 5080 16,303 MiB；驱动 610.62 | 当前 WSL 没有 `nvcc`，Python 没有 torch，需要先装工具链；不下载模型作为 smoke 前置条件 |
| AMD 6750GRE 12G | 暂未接入 | AMD HIP/ROCDL、wave 和显存测试 | 当前机器 `lspci` 未发现该卡，`rocminfo/rocm-smi` 不可用 | 接入前不能声明 ROCm 支持或性能；具体 gfx target 以 `rocminfo` 为准 |
| 鲲鹏 920B（SVE256） | 待借用 | 原生 AArch64、SVE256、性能和 PMU | 用户预计可提供带 SVE256 的机器 | 型号/OS/编译器/NUMA 尚未确认；以机器能力探测为准 |
| QEMU AArch64 | 可用 | AArch64/SVE/SVE2 功能和编译验证 | `qemu-aarch64` 11.0.3；已验证 `max,sve256=on` 可报告 SVE/SVE2，VL=32 bytes | 不能代表鲲鹏吞吐、缓存、内存带宽或指令时序 |

## 5080 WSL 连接方式

SSH 默认 shell 是 Windows `cmd`，直接执行 `uname`、`true` 等 Unix 命令会失败。统一使用：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 192.168.101.5 \
  'wsl.exe -e bash -lc "uname -a; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"'
```

本次探测结果摘要：

```text
Linux GamePC 6.6.87.2-microsoft-standard-WSL2
Ubuntu 24.04.4 LTS
NVIDIA GeForce RTX 5080, 16303 MiB, driver 610.62
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

`nvcc` 或 PyTorch 缺失时，只标记 CUDA smoke 为 `BLOCKED_TOOLCHAIN`，不要让它阻塞 Core IR/CPU 任务。

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
| SVE256/SVE2 | QEMU（功能） | 鲲鹏（真实硬件） | VL/predicate/尾块 |
| NVIDIA | 5080 WSL | 无 | CUDA artifact、运行结果 |
| AMD | 6750GRE 接入后 | 无 | HIP artifact、gfx/wave 检查 |
| 9B/GDR | 后期各目标 | 5080 或鲲鹏 | 先单算子，再报告模型覆盖率 |

## 资源使用原则

- Smoke 测试不加载模型、不需要模型路径、不产生大权重文件。
- 5080 先用于 elementwise、softmax、matmul 和 GPU ABI，不直接从 9B 端到端开始。
- QEMU 先验证 SVE 代码路径，鲲鹏拿到后再把性能结论迁移到真实硬件。
- AMD 卡接入前不能把 HIP backend 标成“可运行”；最多标成“编译路径开发中”。
