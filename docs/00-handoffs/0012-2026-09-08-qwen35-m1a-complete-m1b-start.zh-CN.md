# PyPTO-X Qwen3.5 M1A 完成与 M1B 起点

归档序号：`0012`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M1A_COMPLETE_M1B_STARTING`

## M1A 冻结结果

```text
task branch       = work/qwen35-portable-primitives
task commit       = 5f37d771a63a506a1cdac898f215649771bf97b2
integration HEAD  = 8a95d5c50c58f96a5cd458a65569a11a0be30a3e
```

已完成 `cast/exp/rsqrt/sigmoid/silu/softplus/reduce_mean/broadcast/where` 的 Core/bridge alias、CPU scalar validation/runtime 与标准库 golden。CPU vector common 和 GPU common 对尚未实现的原语明确拒绝，不伪装为已支持。

主线审查额外修正：

- `rsqrt` 与上游文档对齐：负数返回 NaN，零返回 Inf；
- 大正数 `exp` 溢出返回 Inf；
- M1A PIL alias 推断为 `PURE`；
- 上游 `expand_clone` 映射为 portable `broadcast`。

验证：

- task 唯一 host smoke `PASS`；
- focused bridge/vector/GPU 回归 `100 passed`；
- PyPTO-X 全量 `329 passed in 54.33s`；
- Python 3.7 AST（7 files）、compileall、package discovery（127 packages）、diff check `PASS`；
- integration smoke `PASS`；
- 未下载、读取或加载模型权重。

证据：

```text
../worktrees/_meta/pypto-x/qwen35-portable-primitives/smoke/20260908T101228Z/smoke.log
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1a-final/
```

## M1B 任务

```text
repository  = upstream/pypto
base        = 8a95d5c50c58f96a5cd458a65569a11a0be30a3e
branch      = work/qwen35-shape-layout
worktree    = ../worktrees/pypto-x/qwen35-shape-layout
started_at  = 2026-09-08T10:50:21Z
```

范围：

```text
reshape / view
transpose / contiguous
slice / split / concat
gather / embedding
```

M1B 只冻结 Core/bridge 和 CPU scalar 语义，不写 AVX/SVE/GPU kernel。M1C 再把 M1A/M1B 语义接入 vector-common、AVX 与 SVE256，并在当前 ACTIVE 的 920B ECS 上完成原生正确性/汇编验收。

## 保持不变的边界

- 920B 是 2 vCPU/4 GiB KVM guest，不据此冻结性能门槛；
- RTX 5080 GamePC 关机，CUDA 等待资源；
- 不下载或加载权重；视觉编码器与 MTP 不在当前范围；
- 真实权重执行前仍需 typed buffer/mmap。
