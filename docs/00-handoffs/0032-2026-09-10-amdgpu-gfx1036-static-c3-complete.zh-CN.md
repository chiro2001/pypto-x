# PyPTO-X AMDGPU gfx1036 静态 C3 完成快照

归档序号：`0032`

归档日期：2026-09-10（Asia/Shanghai）

状态：`AMDGPU_GFX1036_STATIC_C3_COMPLETE_RUNTIME_BLOCKED_DEVICE`

## 冻结点

```text
integration HEAD    bcf9516e6419d232338988238ffa8f79e59079c1
parent C2           45e8459e79ea979beb58661eb77b30a50f133f75
task commits        633dcdce0, 0ef4a6516, 5e621c353
integration commits 133dcaca6, 64703045a, bcf9516e6
verification        verify/hip-gfx1036-static-c3
```

## C3 能力

C3 在 C2 基础上补齐：

```text
math       exp / rsqrt / sigmoid / silu / softplus
reduction  reduce_sum / reduce_max / reduce_mean，任意合法 axes/keepdims
matmul     rank-2/3/4，exact batch prefix，支持 K=0
```

数学实现是显式 LLVM 算术，不依赖 device library 未解析符号；BF16 使用 i16 storage、FP32 arithmetic 和显式 RNE。reduction 与 matmul 的索引描述保持 compact，不物化逐元素 metadata。artifact/LLVM lowering 版本升至 v3，旧 v1/v2 artifact fail closed。

## 主审查修复

首个候选未直接合入。主审查发现并修复：

1. magic-seed `rsqrt` 对正 FP32 subnormal 会在 Newton 过程中溢出为 `+Inf`；现先以 `2^24` 归一化并以 `2^12` 恢复结果。
2. `silu(-Inf)` 原先形成 `-Inf * 0 = NaN`；现显式返回 `+0`。
3. `rsqrt(-0)` 原先返回 `-Inf`；现与 Core scalar/CUDA 契约一致返回 `+Inf`。
4. 旧 LLVM FP32 hex literal 将 f32 bits 左移32位，LLVM 22会解析成错误数值；现先精确舍入到f32，再以等值f64 bit spelling编码。
5. `exp` 向零舍入阈值使用严格 `<`，在half-min-subnormal边界返回了最小subnormal；现使用ordered `<=`并验证阈值相邻bit pattern。
6. 原数学“oracle”只记录文字声明；现实际host编译执行production LLVM，BF16穷举也走production load/math/store raw i16路径。

## 独立验收

```text
full pypto_x                         577 passed, 7 skipped
FP32 math                            5 × 718 cases，0 mismatch
BF16 production storage math        5 × 65,536 raw inputs，0 mismatch
reduction/matmul host LLVM oracle    10 exact PASS
representative AMDGPU artifacts      3 PASS，0 unresolved symbols
artifact tamper/old versions         11 categories rejected
runtime                              BLOCKED_DEVICE
```

独立 reduction/matmul oracle 将生产生成的单kernel LLVM只作host ABI适配，并逐logical output index执行，覆盖非连续/负 axes、keepdims、全轴/空axes、全负max、NaN传播、BF16 RNE、rank-2/3/4多batch、`K=0`与canary。

Qwen3.5-0.8B真实profile `(B=1,T=1,past=4096)`：

```text
Core operations     4,532
static plans        4,532
LLVM IR bytes       10,510,232
lowering time       about 1.93 s
peak RSS            about 95.4 MiB
```

这是一次完整 `AmdgpuStaticCompiler.lower`；没有生成/链接4,532-kernel整图object，没有加载模型权重，也没有HIP真机执行。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/validation.json
../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/brief.zh-CN.md
```

## 边界与下一步

GamePC WSL当前没有 `/dev/kfd`、`rocminfo`、`hipcc`、`libamdhip64` 或HSA runtime；`/dev/dri`存在不能替代HIP。因此W5静态operator closure已完成，但AMD运行态仍为 `BLOCKED_DEVICE`，没有真机正确性或性能结论。

项目下一条模型主线是获用户明确授权后执行固定revision的Qwen3.5-0.8B BF16纯文本模型，再规划W8A8-linear；AMD主线则等待可用ROCm/HIP设备环境。两者都不会因静态4,532/4,532而自动解锁。

---

> **勘误（2026-09-10）**：本文中的 `4,532`（Core operations / 静态覆盖）属于 graph contract v2。
> 后续发现 GDR decay 门缺 `exp(A_log)` 并已修复（契约升到 v3），新计数为 `4,550`（(1,1,4096)）/ `6,728`（(1,5,0)）。
> 详见 `ERRATA.zh-CN.md` 的 ERR-0001。
