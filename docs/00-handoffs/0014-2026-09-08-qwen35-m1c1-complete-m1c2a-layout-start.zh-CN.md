# PyPTO-X Qwen3.5 M1C1 完成与 M1C2a SVE256 起点

归档序号：`0014`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M1C1_COMPLETE_M1C2A_SVE256_LAYOUT_STARTING`

## M1C1 冻结结果

```text
task commit       = 8b4fa2100cb262cb8d57f5e35aaad22c19161232
integration HEAD  = 556a1bb4c8b48ae52b0a7e52c56048b621890dc7
```

完成 M1A 9 组原语的 vector-common/SVE256 接入。AVX2、AVX-512 和 GPU 未实现这些新原语，仍会明确拒绝。

验证：

- PyPTO-X 全量 `360 passed in 66.92s`；
- Python 3.7 AST、compileall、127-package discovery、diff check PASS；
- integration QEMU smoke PASS；
- 修正后 920B native wire 与 guarded canary PASS；
- `ptx_rsqrt_f32` 热点含 `whilelo/ld1w/st1w/fsqrt/fdivr/z/p`，NEON v/q 为 0。

必须保留的真实边界：

- `exp/sigmoid/silu/softplus` 是逐 lane libm fallback；
- BF16 math/where 与 cast 的数值转换逐 lane；
- broadcast/where 的广播输入在 host 预展开，native runner 执行 copy/select；
- 因此 M1C1 是正确性闭包，不是完整 SVE 向量数学性能实现。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c1-review/
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c1-final/
```

## M1C2a 任务

```text
repository  = upstream/pypto
base        = 556a1bb4c8b48ae52b0a7e52c56048b621890dc7
branch      = work/qwen35-sve256-layout
worktree    = ../worktrees/pypto-x/qwen35-sve256-layout
started_at  = 2026-09-08T12:39:09Z
resource    = QEMU then Kunpeng 920B ECS
```

范围：

```text
reshape / portable view
transpose
contiguous
slice
```

要求把数据移动/索引信息显式写入 plan/artifact/wire ABI；除 reshape/view/contiguous 的合法 identity/copy 语义外，不允许在 Python host 先算出完整输出再让 native runner 复制。M1C2b 再处理 split/concat/gather/embedding。

920B 继续只作功能和汇编验证，不设置性能门槛；不下载或加载模型权重。
