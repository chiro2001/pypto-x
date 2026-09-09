# PyPTO-X AVX2 Qwen parity 与 CUDA C2 完成快照

归档序号：`0024`

归档日期：2026-09-09（Asia/Shanghai）

状态：`AVX2_QWEN_PARITY_CUDA_C2_COMPLETE_AVX512_IN_PROGRESS`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    5b0474f0db26d611526070dc4298170672e264af
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

本阶段 integration commits：

```text
c0b0abead  feat(pypto-x): close qwen35 avx2 parity
5b0474f0d  feat(cuda): add qwen c2 math layout and indexing
```

## AVX2 Qwen M1A–M1H parity

AVX2 已能接收 M1A–M1H 冻结语义并执行 Qwen RMSNorm/L2Norm/SwiGLU/Softmax/RoPE/GQA/causal-mask、attention、Conv state、GDR recurrent state 与 position/control correctness graph。AVX2 lowering 升至 v2，并要求 compact vector plan；旧 v1、non-compact、tampered artifact 和错误 cache key 均 fail closed。

独立验收：

```text
full suite             474 passed
AVX2 focused           63 passed
Python 3.7 AST         105 files PASS
compileall             203 files PASS
namespace packages     128 PASS
CPUID/OSXSAVE/XGETBV   PASS
assembly               YMM/FMA/non-FMA PASS; no ZMM/EVEX/0x62
```

执行边界：基础 FP32/BF16 elementwise/reduction/rank-2 matmul 使用 native AVX2；数学超越函数使用逐 lane libm；cast、broadcast/where、layout/indexing、compare/iota 使用显式 host-reference；batched matmul 由 host loop 调用 rank-2 native runner。这是 correctness parity，不是全部 native SIMD 或性能结论。

实现代理唯一 smoke 为 `PASS`，但写入了旧 `/home/chiro/projects/worktrees/...` 根；独立验收保留该偏差且没有为修正路径而重跑。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-avx2-parity-final/validation.json
```

## CUDA C2

CUDA C2 在 C1 Driver API + PTX JIT 上补齐：

```text
exp/rsqrt/sigmoid/silu/softplus/reduce_mean
broadcast/where/compare/iota/position
reshape/view/transpose/contiguous/slice
split/concat/gather/embedding
rank-2/3/4 matmul
scalar SSA host synchronization
Qwen RMSNorm/Softmax/RoPE/GQA/causal-mask/decoder composites
```

GPU common IR/lowering 与 CUDA payload/lowering 升至 v2；公共 GPU ABI 保持 vendor-neutral，旧版本、PTX/binding/digest/tamper/cache 均 fail closed。

独立验收：

```text
full suite                    476 passed, 7 skipped
GPU common vendor/ABI         45 passed
CUDA host/mock                17 passed, 7 skipped
CPU AVX/SVE joint regression  260 passed
RTX 5080 extended             PASS
```

5080 实测覆盖 math 特殊值、BF16 RNE tie/odd tail、六谓词 compare 与广播、int32/int64 iota、position prefill/decode/大 shape、layout/indexing/OOB、split 全输出 binding、rank-2/3/4 matmul/K=0、有符号/无符号/浮点/BF16 scalar ABI、scalar SSA、多 kernel 与 context/module/allocation cleanup。执行前后无 compute process 残留。

统一 smoke 只执行一次，因无 `nvcc` 为 `BLOCKED_TOOLCHAIN`；Driver/PTX 路径无需 Toolkit 并已真机通过。当前 PTX 使用 approximate math 与 `sm_80` forward JIT；布局、索引和 batched matmul 是通用 correctness kernel，不宣称 fused、NVVM、Tensor Core 或性能。

证据：

```text
../worktrees/_meta/pypto-x/integration-w4-cuda-c2-final/validation.json
```

## 下一步

1. `qwen35-avx512-parity` 已从 `5b0474f0d` 启动，补齐同一 M1A–M1H 语义并保留 ZMM/opmask 与 feature layering 证据。
2. AVX-512 验收后实现无权重、非 identity-shell 的24层 decoder connectivity。
3. 建立 typed-buffer/mmap 权重路径；获得授权后运行固定 revision 的 Qwen3.5-0.8B BF16。
4. BF16 闭环后进入 W8A8-linear；HIP 等待 AMD 6750GRE 接入。

RTX 5080 GPU 继续由 PyPTO-X 独占；GPU-only 工作不申请 `gamepc`，GamePC host-heavy 才持锁。本机 heavy 始终使用 `local` runner。本阶段未下载或加载模型权重，未安装 CUDA Toolkit/PyTorch/Triton。
