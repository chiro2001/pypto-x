# PyPTO-X AVX-512 Qwen parity 完成快照

归档序号：`0025`

归档日期：2026-09-09（Asia/Shanghai）

状态：`AVX512_QWEN_PARITY_COMPLETE_DECODER_CONNECTIVITY_READY`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    63c9a4fdd6c1282aa8226b157d81dccbea6b6f21
parent HEAD         5b0474f0db26d611526070dc4298170672e264af
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

实现与集成：

```text
task commit         76325e3b863a4695c882c48932927d3c216c5e91
integration commit  63c9a4fdd6c1282aa8226b157d81dccbea6b6f21
```

## 完成能力

AVX-512 lowering 升至 v2，接收 Qwen M1A–M1H 全部冻结 primitive，并运行 RMSNorm、Softmax、SwiGLU、attention、Conv state、GDR recurrent state 与 position/control correctness graph。

执行模式保持可审计：

- FP32/BF16 `identity/add/sub/mul/div/neg/reduce_sum/reduce_max/reduce_mean/matmul` 使用 ZMM/opmask native；
- `exp/rsqrt/sigmoid/silu/softplus` 使用显式 scalar/libm lane；
- `cast/broadcast/where/layout/indexing/compare/iota/position` 使用显式 host-reference；
- rank-3/4 batched matmul 由 host loop 调用 rank-2 native runner；
- AVX2 artifact 不能直接交给 AVX-512 runtime，fallback 必须显式提供；
- DQ/BW/VL capability 虽可探测，但在没有 emitting lowering 时继续明确拒绝。

## 独立验收

```text
full suite             483 passed, 7 skipped
AVX-512 focused        76 passed
Python 3.7 AST         108 files PASS
compileall             203 files PASS
namespace packages     128 PASS
CPUID/OSXSAVE/XGETBV   PASS; XCR0=743
```

主机探测确认 AVX-512F/DQ/BW/VL/VNNI/BF16/FMA。重新生成的 native artifact 反汇编确认：

```text
base       ZMM/opmask; no accidental VNNI/BF16 opcode
FMA        vfmadd
BF16       vdpbf16ps
VNNI       vpdpbusd
```

Qwen 专项覆盖 T=128/1024/4096 compact position plan、rank-3/4 matmul scalar differential，以及小型非零 GDR state differential。未下载、读取或加载模型权重，没有冻结性能门槛。

实现代理和验收代理最初把证据写到了旧 `/home/chiro/projects/worktrees/...` 根；没有重跑任何测试。最终证据被复制到标准项目目录，19个核心文件 SHA256/大小完全一致，唯一预期差异是标准 `validation.json` 中的绝对路径修正，原目录保留。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-avx512-parity-final/validation.json
../worktrees/_meta/pypto-x/integration-w6-qwen35-avx512-parity-final/relocation-audit.txt
```

## 下一步

启动 `qwen35-no-weight-decoder-connectivity`：把当前24层 identity shell 替换为非 identity Core graph。权重只作为显式参数契约，不读取 checkpoint；逐层连接18个 GDR、6个 full-attention、MLP/residual/final RMSNorm/LM head，并函数式传递 KV/Conv/GDR state。固定真实 shape 做 compile/compact/size 门禁，缩小但拓扑等价的 synthetic graph 用于 scalar、AVX2、AVX-512、SVE256 与 CUDA 正确性执行。
