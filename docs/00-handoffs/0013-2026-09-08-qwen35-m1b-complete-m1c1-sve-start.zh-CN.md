# PyPTO-X Qwen3.5 M1B 完成与 M1C1 SVE256 起点

归档序号：`0013`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M1B_COMPLETE_M1C1_SVE256_MATH_STARTING`

## M1B 冻结结果

```text
task commit       = 51873c3ebef84a30561f879446b1af2693d5517c
integration HEAD  = 8e840f11344a8ac9ec5368b69a5ce0e0a66281b9
```

完成 portable `reshape/view/transpose/contiguous/slice/split/concat/gather/embedding` 的 Core/PIL bridge、CPU scalar validation/runtime 和标准库 golden。非等长 split 使用 Core 多输出 SSA，不把结果隐藏为 Python tuple。

兼容性审查冻结：

- portable `gather` 是 index-select 语义，供 embedding 使用；
- 上游 `pypto.gather` 是 gather-elements，暂不映射；
- 上游 `pypto.view` 是带 offset 的局部视图，暂不映射到 reshape-like portable view；
- reshape 的 partial `valid_shape` 和 in-place aliasing 显式 unsupported；
- 不以“同名可解析”代替语义兼容。

验证：focused `120 passed`；全量 `349 passed in 57.24s`；Python 3.7 AST、compileall、127-package discovery、diff check 与 integration smoke 均 PASS。证据位于：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1b-final/
```

## M1C1 任务

```text
repository  = upstream/pypto
base        = 8e840f11344a8ac9ec5368b69a5ce0e0a66281b9
branch      = work/qwen35-sve256-math
worktree    = ../worktrees/pypto-x/qwen35-sve256-math
started_at  = 2026-09-08T11:28:24Z
resource    = QEMU then Kunpeng 920B ECS
```

M1C1 只处理 M1A 数值原语：

```text
cast
exp / rsqrt
sigmoid / silu / softplus
reduce_mean
broadcast / where
```

先建立 ISA-neutral vector plan，再接入 SVE256 compiler/runtime。若 SVE 无原生 transcendental 指令，允许把 `expf/log1pf` 的逐 lane libm fallback 作为明确 metadata/报告字段，但不得把 fallback 宣称为完整 SVE 向量数学实现。AVX/GPU 不因 SVE 支持而自动开放。

## 920B 验收

- 当前实例 ACTIVE、SVE=1、SVE2=0、VL=32 bytes；
- 先 QEMU 功能门禁，再原生检查 FP32/BF16、尾块、canary、全负/特殊值与热点汇编；
- 不运行或加载模型权重；
- KVM guest 结果只作功能/汇编证据，不设性能门槛。
