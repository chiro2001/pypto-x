# 0003 · 2026-09-07 · PyPTO 算子可移植占比审计

状态：`COMPLETE`

基线：PyPTO `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad`；PyPTO-Gym `945a360e12592239a3549cb62d0db37af32bbc03`。

## 分类口径

- `P0`：数学/张量语义可直接建模为目标无关 Core IR；
- `P1`：语义可移植，但需 target lowering、算法重写、库或 capability 约束；
- `X`：语义本身绑定 CANN/CCE/AICPU/AICore、物理 MemorySpace/Pipe、设备通信或调试副作用。

主指标是 `potential=(P0+P1)/N`。`strict=P0/N` 受“将 reduction/exp/transpose 设为 Core primitive 还是 composite”的方言设计影响，只用于观察复杂度。

## 结果

| 口径 | N | P0 | P1 | X | strict | potential | 当前完整实现直接复用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| classic `pypto/op` 唯一 `@op_wrapper` | 136 | 49 | 72 | 15 | 36.0% | 89.0% | 0% |
| classic，排除 11 个 distributed 算子 | 125 | 49 | 72 | 4 | 39.2% | 96.8% | 0% |
| `pypto_pro` CCE 唯一注册 op | 301 | 83 | 28 | 190 | 27.6% | 36.9% | 0% |
| PyPTO-Gym 去重逻辑融合算子 | 83 | 10 | 73 | 0 | 12.0% | 100% | 0%；宽松计 fallback 为 1.2% |

不计算一个全局加权百分比，因为三个分母互相重叠：Gym 复合算子由 classic primitive 组成，Pro 的 301 项又同时包含语义 op、VF/SIMT 和物理资源 op。

## Classic PyPTO

136 个唯一公共 wrapper 中，比较、基本算术、逻辑、cast、where、创建、concat/transpose 等主要归 P0；reduction、matmul/conv、gather/scatter、sort/topk、超越函数、量化和随机等主要归 P1。

15 个 X 包含 11 个 distributed/SHMEM/PE 操作、`index_add_*_ub` 和 pass verify 调试副作用。排除 distributed 后，数学/张量语义的潜在可移植率达 96.8%。

但现有 wrapper 之后的 C++ lowering、Tile graph、NPU arch/MemorySpace 检查和运行时仍绑定 Ascend，因此完整实现直接复用率为 0%。可复用资产是 API 签名、参数检查、shape/effect 语义和 golden。

## PyPTO Pro

301 个注册项的命名空间分布：

```text
block 128, vf 84, simt 48, system 21,
debug 5, bare hardware/control 10, ptr 3, struct 2
```

190 个 X 中，173 个来自非 `block.*` 命名空间；另有 17 个 `block.*` 是当前绑定 GM/UB/L0/Acc、MTE/M/FIX Pipe 的 load/store/move/matmul/gemv 等操作。

111 个 P0+P1 名称可保留语义、shape/type 规则或 parser 结构，但当前 301 项全部同时设置 CCE Pipe 和 CCE codegen callback，所以 callback/lowering 的直接复用率是 0%。第一版 Pro portable profile 可在 70–90 个语义 API 范围内再归一化。

## PyPTO-Gym

83 个去重逻辑融合算子的数学语义均可在 CPU/GPU 上重新实现；其中 attention、MLA、GDR/KDA、grouped GEMM/MoE、FP8/MXFP8/antiquant、paged KV cache 等 73 个需要实质性 target lowering/算法重写。

当前 109 个非 common-utils Tensor 实现文件中，101 个包含 classic `pypto`，100 个使用 `@pypto.frontend.jit`，并有多处 `torch_npu`/NPU dispatch。因此生产 kernel 直接复用率为 0%；一个纯 Torch MRoPE fallback 可宽松计为 1.2%，但它不是 PyPTO 融合 kernel。

GDR 子集的数学语义归 P1，当前实现复用率为 0%。可复用 Torch golden、shape contract、精度阈值和算法文档。

## 工程含义

1. 公共 Core IR 不应追求线性搬运 301 个 Pro 注册项；先归一化数学语义。
2. classic 是获取现有生态语义覆盖的更好入口，但仍需重写非 Ascend lowering。
3. Pro 中约 63.1% 的当前 op 应留在 Ascend dialect/target，而不是公共 Core IR。
4. Gym 的 100% 语义潜在可移植率不代表低开发量；73/83 是复杂 P1，工作量远大于简单 elementwise。
5. MVP 应按 workload closure 验收，不应按算子名称百分比验收。

建议 MVP 首先覆盖 add/sub/mul/div、cast/select、exp/rsqrt/relu、view/reshape/transpose、load/store 抽象、row sum/max、softmax 分解、FP32 累加 matmul、BF16/FP16 输入、RMSNorm/RoPE、for/if、动态 shape 和尾块；GDR 在这些 primitive 稳定后进入第二阶段。

## 验证记录

- classic smoke：PASS，`../worktrees/_logs/portability-classic/20260906T161629Z/smoke.log`
- Pro smoke：PASS，`../worktrees/_logs/portability-pro/20260906T161629Z/smoke.log`
- Gym smoke：PASS，`../worktrees/_logs/portability-gym/20260906T161629Z/smoke.log`
- 三个任务均为只读静态审计，没有修改上游源码。

## 证据入口

- [classic operator modules](../../../upstream/pypto/python/pypto/op/)
- [Pro CCE backend](../../../upstream/pypto/framework/src/interface/pypto_pro/backend/)
- [Pro language API](../../../upstream/pypto/python/pypto_pro/language/)
- [PyPTO-Gym operator library](../../../upstream/pypto-gym/src/pypto_gym/ops/)
