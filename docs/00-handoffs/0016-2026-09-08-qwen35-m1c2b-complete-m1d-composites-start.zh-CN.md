# PyPTO-X Qwen3.5 M1C2b 完成与 M1D SVE256 起点

归档序号：`0016`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M1C2B_COMPLETE_M1D_SVE256_COMPOSITES_STARTED`

## M1C2b 冻结结果

```text
task commits      = d8df8637f, eea99d332
integration HEAD  = 31a5c46b010a621e54b8dedaa81c97a98dfc4e3e
```

完成 `split/concat/gather/embedding` 的 vector-common/SVE256 接入。静态 plan/artifact 使用 O(rank + segment count) descriptor，dynamic indices 只作为运行时 tensor 数据，split 保持多输出 SSA。

验证：

- PyPTO-X 全量 `393 passed in 116.36s`；
- focused `13 passed`，另有 12 个多轴、零长度、scalar index 的独立差分场景；
- Python 3.7 AST、compileall、127-package discovery、diff check PASS；
- task 与 integration 各自唯一 QEMU smoke PASS；
- 920B native 的 FP32/BF16 split/concat、int32/int64 gather/embedding、bad-index/hash/descriptor、wire v1/v3 兼容与 v4 pairing、guarded canary PASS；
- `ptx_index_f32` 含 `whilelo1/ld1w2/st1w1`，NEON v/q 为 0。

必须保留的真实边界：

- BF16 gather/embedding 是 scalar lane reorder fallback；
- FP32 gather 的 source offset 限于 uint32；
- split 当前每个输出单独调用 runner，并为每个输出重复发送 source；
- Python runtime 仍使用 host list 和 wire byte buffer 暂存；
- AVX2、AVX-512 和 GPU indexing 仍明确拒绝；
- 920B ECS 只作功能、协议与汇编证据，不设置性能门槛。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2b-final/
```

## M1D 任务

```text
repository  = upstream/pypto
base        = 31a5c46b010a621e54b8dedaa81c97a98dfc4e3e
branch      = work/qwen35-sve256-composites
worktree    = ../worktrees/pypto-x/qwen35-sve256-composites
started_at  = 2026-09-08T15:36:53Z
resource    = QEMU then Kunpeng 920B ECS
```

首批组合：

```text
RMSNorm / L2Norm / SwiGLU / stable Softmax
RoPE / GQA repeat-KV / causal mask
```

优先组合已冻结的 Core primitive，不把组合名称直接变成 target-specific fused opcode。测试继续使用小型确定性张量与 Qwen shape metadata，不下载或加载模型权重。
