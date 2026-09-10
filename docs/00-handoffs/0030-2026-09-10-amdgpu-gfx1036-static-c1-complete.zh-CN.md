# PyPTO-X AMDGPU gfx1036 静态 C1 完成快照

归档序号：`0030`

归档日期：2026-09-10（Asia/Shanghai）

状态：`AMDGPU_GFX1036_STATIC_C1_COMPLETE_C2_READY_RUNTIME_BLOCKED_DEVICE`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    e6e8360702d39da9f11e6352b94714c6ed23901a
parent milestone    fe6f3270b973b08be98cacfec04a6a4e9482e2b0
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

实现：

```text
task branch          work/hip-gfx1036-static-backend
task commits         51d2bd49b, fc0ba3694
integration commits  4f5f31d0c, e6e836070
verification branch  verify/hip-gfx1036-static-backend
```

## 目标与边界

固定目标：

```text
profile    amd-igpu-gfx1036
triple     amdgcn-amd-amdhsa
processor  gfx1036
artifact   pypto-x.amdgpu.llvm
runtime    BLOCKED_DEVICE
```

目标身份来自 GamePC Windows OpenCL，不是 `rocminfo`。wavefront、XNACK、native BF16、INT8、MFMA、WMMA 保持 `unknown`；`gfx1031/6750GRE` alias 被明确拒绝。

本阶段生成普通 LLVM AMDGCN IR，不是 MLIR ROCDL dialect，也不声称 HIP runtime execution。

## 静态 C1 能力

支持的 exact contract：

- `identity`：input/output exact shape；
- `add`、`mul`：两输入与输出 exact shape，不支持 broadcast；
- `reduce_sum`：仅全轴归约为一个元素；
- `matmul`：仅 rank-2、shape严格匹配；
- storage dtype：FP32、BF16；BF16以i16存储、FP32运算、显式RNE写回。

每个 kernel 显式使用：

- `ptr addrspace(1)` global pointer；
- `llvm.amdgcn.workitem.id.*` / `workgroup.id.*`；
- 64-bit flattened index 与 bounds/tail；
- 真实 load、ALU、store。

`llvm-as` 与 `llc -march=amdgcn -mcpu=gfx1036` 生成 assembly 和 `elf64-amdgpu` object；artifact envelope 对 program/plan/target/LLVM/assembly/object/toolchain digest 做 fail-closed 校验。

## 独立验收

```text
focused static tests     16 passed
full pypto_x             538 passed, 7 skipped
LLVM/object artifacts    5 PASS
artifact tamper          12 cases rejected
Python 3.7/compileall    PASS
python -S/package/diff   PASS
```

独立反汇编确认 `gfx1036` object 包含 global load、ALU、global store、workitem/workgroup ID 和 bounds。broadcast、shape mismatch、zero shape、bad axes/rank/dtype 均明确拒绝。

Qwen3.5真实 decode profile：

```text
Core operations          4,532
C1 statically supported    844
explicit capability gap  3,688
whole-model kernels          0
```

844不是按op名称估算，而是逐operation用与compiler相同的 `validate_static_operation()` 判定。首轮实现曾把broadcast add/mul计入1,152；主审查发现后收紧为exact-shape，并将数字修正为844。

runtime 的 availability、load、workspace_size 与 launch 都返回 `BLOCKED_DEVICE`；没有设备工作或模型权重。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-final/validation.json
../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-final/qwen_gap.json
```

唯一 smoke 记录为 `BOOTSTRAP_BLOCKED`：普通 source checkout import 触发上游 online-build loader，但未安装 distribution metadata；按协议未重跑。portable-only、`python -S` 与正式测试均通过。

## 下一步

进入 C2 layout/indexing/control 静态 codegen，优先闭包 Qwen gap 中数量最高且不依赖device library的 cast、reshape、transpose、broadcast、slice、concat/split、sub/div/neg、constant/iota/compare/where。每个新增op必须有真实LLVM/object门禁和独立host oracle审计；在ROCm/HIP runtime可用前仍不得声称真机执行。
