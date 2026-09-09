# PyPTO-X Qwen3.5 M1H 与 CUDA C1 完成快照

归档序号：`0023`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_M1H_CUDA_C1_COMPLETE_AVX_PARITY_CUDA_C2_READY`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    2c99c4b319164006ba96696a42fe80757119cab7
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

从 M1G 冻结点到当前 HEAD 的 integration commits：

```text
564e643ab  feat(pypto-x): close qwen35 position control
620d6f794  fix(pypto-x): harden qwen35 compare compatibility
7eff3a056  feat(cuda): add driver api and ptx bootstrap
2c99c4b31  fix(cuda): harden ptx artifact and edge validation
```

## Qwen3.5-0.8B M1H

M1H 冻结 compare/iota/position/causal mask 语义，并提供固定 24 层 text decoder structural manifest：18 个 GDR 层、6 个 full-attention 层（3/7/11/15/19/23）。Core 中只保留明确的 `iota`，不保留含糊的 `arange` alias。

独立验收结果：

```text
full suite             455 passed
Python 3.7 AST         97 files PASS
compileall             199 files PASS
namespace packages     128 PASS
QEMU smoke             PASS; SVE/SVE2; VL=32
920B contract/canary   PASS; SVE=1, SVE2=0, VL=32
```

position 的 T=128/past=0、T=1/past=128 执行通过，T=1024/past=0、T=1/past=4096 编译通过；plan 均为 compact domain，无逐元素 `index_map`。旧 vector/SVE/AVX artifact 与 fallback metadata 篡改均 fail closed。

边界保持明确：SVE `iota/compare` 是 Python `host_reference` fallback，`native_position_ops=false`；24 层 graph 只执行 identity shell 并携带 structural manifest，不是带权整网执行。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1h-final/validation.json
```

## CUDA C1

CUDA C1 在不安装 Toolkit 的条件下实现 Python `ctypes + libcuda.so.1` Driver API、确定性 PTX、device/context/module/memory/launch/sync/cleanup，并执行 FP32/BF16 elementwise、reduce_sum/reduce_max 与 rank-2 matmul。

独立验收结果：

```text
full suite             470 passed
CUDA mock/static       15 passed
Python 3.7 AST         104 files PASS
compileall             203 files PASS
namespace packages     128 PASS
RTX 5080 Driver/PTX    PASS; CC 12.0; driver 610.62
```

真机覆盖 BF16 RNE tie/odd tail、非末轴 reduction、全负 reduce-max、K=0 matmul、zero-work、guarded canary、多操作链、empty reduce-sum、5 次重复 launch，以及 context/module/allocation cleanup；验收前后没有残留 compute process。

统一 `nvidia-5080` smoke 在 WSL 中只执行一次。GPU 可见，但因无 `nvcc` 按规范为 `BLOCKED_TOOLCHAIN`；实际 Driver/PTX 路径不依赖 `nvcc` 并已通过。这是正确性 bootstrap，不是 NVVM、Tensor Core 或性能证据；当前 PTX 使用 `sm_80` target，由 CC 12.0 Driver 前向 JIT。

证据：

```text
../worktrees/_meta/pypto-x/integration-w4-cuda-c1-final/validation.json
```

## 资源规则

RTX 5080 GPU 由 PyPTO-X 持续独占。GPU-only probe、Driver/PTX JIT 与小型 kernel 不申请 `gamepc`；只有大量 GamePC CPU 或 host-memory 阶段才持续持锁。本机 full suite 和其他 heavy 命令仍必须通过 `scripts/resource/run_local_heavy.sh` 申请 `local` 锁并接受 cgroup/资源监控。

本阶段没有下载或加载模型权重，没有安装 CUDA Toolkit、PyTorch 或 Triton。

## 下一步

1. AVX2 回填 Qwen M1A–M1H 冻结语义，并独立验收；随后处理 AVX-512 parity。
2. CUDA C2 补齐 exp/rsqrt/sigmoid/silu/softplus、broadcast/where、layout/indexing，再组合 Softmax/RMSNorm/RoPE。
3. 两条路径达到 parity 后连接无权重完整 decoder connectivity。
4. 权重、参考框架和存储获得明确授权后，运行固定 revision 的 Qwen3.5-0.8B BF16；之后进入 W8A8-linear。
5. HIP 等待 AMD 6750GRE 接入；stable CANN/NPU 回归仍等待实际环境。
