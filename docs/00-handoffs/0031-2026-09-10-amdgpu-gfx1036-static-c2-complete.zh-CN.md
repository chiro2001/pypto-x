# PyPTO-X AMDGPU gfx1036 静态 C2 完成快照

归档序号：`0031`

归档日期：2026-09-10（Asia/Shanghai）

状态：`AMDGPU_GFX1036_STATIC_C2_COMPLETE_C3_READY_RUNTIME_BLOCKED_DEVICE`

## 冻结点

```text
integration HEAD    45e8459e79ea979beb58661eb77b30a50f133f75
parent C1           e6e8360702d39da9f11e6352b94714c6ed23901a
task commits        5fe946173, 8369a7ff6
integration commits 237a27614, 45e8459e7
verification        verify/hip-gfx1036-static-c2-r2
```

## C2 能力

C2 在C1基础上增加：

```text
sub / div / neg
cast: BF16/FP32/INT32/INT64/BOOL contract
constant
reshape / view / contiguous
transpose rank 1-4
broadcast / positive-step slice
concat / split
gather / embedding
iota / compare / where
```

所有索引描述保持O(rank+segments)，不保存逐元素map。LLVM/artifact版本升至v2，旧C1 artifact要求重编译。BF16以i16 storage表示并保留NaN/Inf/signed-zero；gather负数/OOB发射trap。

## 首轮失败与修复

首轮独立验收为 `FAIL`，发现：

1. transpose非自逆permutation使用了错误坐标映射；
2. split first-match CFG不能独立写多个输出；
3. zero-shape与rank-0 transpose未拒绝；
4. validator、GPU-common lowering、emitter对bool/int contract不一致；
5. 最小signaling NaN被BF16 RNE变成Inf；
6. generic ELF、伪造assembly/toolchain在重算外层digest后仍可通过。

修复后：transpose使用inverse assignment；split逐输出独立写；zero/rank门禁fail closed；GPU-common copy dtype与emitter统一；NaN强制非零quiet mantissa；artifact decode重跑exact `llvm-as/llc`并逐文本/逐字节比较derived outputs。

## r2 验收

```text
focused C1+C2              35 passed
full pypto_x               557 passed, 7 skipped
18 LLVM/ELF artifacts      PASS
artifact tamper            23/23 rejected
GPU-common+CUDA+CPU        308 passed, 7 skipped
runtime                    BLOCKED_DEVICE
```

独立oracle覆盖非对称transpose、非等长split、zero/rank、dtype parity、BF16 sNaN/qNaN/Inf/zero。所有artifact通过 `llvm-as`、`llc -mcpu=gfx1036`、`llvm-objdump`和严格decode。

Qwen3.5真实profile：

```text
total Core ops     4,532
C2 supported       4,098
remaining gap        434
```

剩余gap：`exp 24`、`rsqrt 115`、`sigmoid 24`、`silu 60`、`softplus 18`、`reduce_sum 42`、`reduce_max 6`、`reduce_mean 79`、非rank-2 `matmul 66`。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c2-final-r2/validation.json
../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c2-final-r2/brief.zh-CN.md
```

## 边界与下一步

本阶段没有HIP真机执行，没有整网native kernel，也没有模型权重；static artifact decode现在依赖同版LLVM工具链重推导，这是刻意的fail-closed边界。

下一步C3实现math、任意轴reduction与rank-3/4 batched matmul，目标达到4,532/4,532静态lowerable。即使达到100%，在GamePC WSL提供 `/dev/kfd`、ROCm/HIP runtime并完成设备执行前，状态仍必须是 `BLOCKED_DEVICE`。

---

> **勘误（2026-09-10）**：本文中的 `4,532`（Core operations / 静态覆盖）属于 graph contract v2。
> 后续发现 GDR decay 门缺 `exp(A_log)` 并已修复（契约升到 v3），新计数为 `4,550`（(1,1,4096)）/ `6,728`（(1,5,0)）。
> 详见 `ERRATA.zh-CN.md` 的 ERR-0001。
