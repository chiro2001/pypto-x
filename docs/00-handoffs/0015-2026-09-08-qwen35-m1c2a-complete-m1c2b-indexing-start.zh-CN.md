# PyPTO-X Qwen3.5 M1C2a 完成与 M1C2b SVE256 起点

归档序号：`0015`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M1C2A_COMPLETE_M1C2B_SVE256_INDEXING_STARTED`

## M1C2a 冻结结果

```text
task commits      = 0ea1b3b23, 015506dca, 3ca34cfd6
integration HEAD  = 202b045fba9b02b4982453eac42c648ecb49dbc9
```

完成 `reshape/view/transpose/contiguous/slice` 的 vector-common/SVE256 接入。最终设计使用 O(rank) compact affine descriptor，记录 shape、row-major source strides、permutation 与 slice start/step；不在 plan、artifact 或 wire 中保存逐元素 index map。

验证：

- PyPTO-X 全量 `380 passed in 84.60s`；
- Python 3.7 AST、compileall、127-package discovery、diff check PASS；
- integration QEMU smoke PASS；
- 32K×2048 静态 shape 证明 plan/descriptor/artifact metadata 大小与 numel 无关；
- 920B native 的 FP32/BF16 transpose、slice、BF16 identity copy、奇数 payload 未对齐 descriptor、tamper、v1 layout 拒绝和 guarded canary PASS；
- FP32 热点含 SVE u32 indexed gather，BF16 identity copy 含 `ld1h/st1h`，两个热点 NEON v/q 均为 0。

必须保留的真实边界：

- BF16 transpose/slice 在 SVE1 上是 scalar lane reorder fallback；
- FP32 indexed gather 的 source index 超过 uint32 时明确拒绝；
- AVX2、AVX-512 和 GPU 对这些 layout op 仍明确拒绝；
- 920B ECS 是 2-vCPU KVM，仅作为功能、协议与汇编证据，不设置性能门槛。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2a-final/
```

## M1C2b 任务

```text
repository  = upstream/pypto
base        = 202b045fba9b02b4982453eac42c648ecb49dbc9
branch      = work/qwen35-sve256-indexing
worktree    = ../worktrees/pypto-x/qwen35-sve256-indexing
started_at  = 2026-09-08T14:17:35Z
resource    = QEMU then Kunpeng 920B ECS
```

范围：

```text
split / concat
gather / embedding
```

继续使用 compact descriptor，不允许按输出元素把索引复制进 plan/artifact。`split` 必须保持 M1B 的可序列化多输出 SSA；portable `gather` 仍是 index-select/embedding 语义，不得误映射成上游同名 gather-elements。920B 只作功能与汇编验证，不下载或加载模型权重。
