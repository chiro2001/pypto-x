# W3：x86 CPU vector、AVX2 与 AVX-512 契约

状态：`DRAFT_PENDING_W2B`

日期：2026-09-07（Asia/Shanghai）

## 1. 顺序与范围

```text
CPU scalar golden
      ↓ differential
CPU vector common plan
      ↓
AVX2 + FMA
      ↓
AVX-512 capability variants
```

W3 不开发 NEON，不引入 SVE/GPU 概念，不改变 Core IR。性能门槛仍按用户决策延后；本波次先固定实际向量指令、运行时派发和奇数尾块的正确性。

## 2. CPU vector common

- 从 CPU scalar 已支持的 CoreProgram 生成 target-owned loop/vector plan；逻辑 Tile 不等于寄存器宽度。
- plan 记录 iteration domain、contiguous axis、dtype、reduction 和 tail policy，不记录 AVX/SVE intrinsic 名。
- 共享 shape/op validation 与 scalar golden；未知 op、symbolic shape、非连续 view 首版显式拒绝。
- artifact metadata 记录 required CPU features、vector-plan version、ABI、dtype/layout 和 specialization；cache key 继续使用 W1 稳定字段。
- runtime 在 load/launch 前核对 host CPU 与 OS extended-state 支持，不能只看 `/proc/cpuinfo` flag。

## 3. 能力探测

- x86 使用 CPUID + OSXSAVE/XGETBV 验证 OS 能保存 YMM/ZMM/opmask 状态。
- AVX2 至少要求 AVX、AVX2、OS YMM state；FMA 作为独立 capability。
- AVX-512 按 F/BW/DQ/VL、VNNI、BF16 分层，不用一个 `avx512=true` 概括。
- 探针结果可注入 fake capability 做单测；真实主机探针必须记录 compiler、triple 和原始 feature。
- 不支持请求特征时结构化 unavailable，禁止启动可能产生 SIGILL 的 artifact。

## 4. AVX2

- 第一闭包与 scalar 对齐：连续 elementwise、reduction 和二维 matmul。
- FP32 使用 YMM/FMA；BF16 输入按 RNE 语义转换到 FP32 计算，输出再 RNE 到 BF16。AVX2 不假定 BF16/VNNI 指令。
- 不能整除向量宽度的尾部使用明确 scalar cleanup 或安全 mask/load 方案，不越界读写。
- 生成物必须用反汇编证明热路径含 YMM 指令，且不含 ZMM/AVX-512 指令。

## 5. AVX-512

- 复用同一 vector plan，末端选择 ZMM/opmask；tail 优先使用 mask。
- BF16/VNNI 只能在对应 CPUID 与 OS state 通过时选择；否则回退到 AVX-512F 或 AVX2 已验证变体。
- artifact required-features 必须精确到实际发射的指令集，cache 不得让高能力 artifact 在低能力 CPU 命中。
- 反汇编必须证明预期 ZMM/opmask/BF16/VNNI 指令，并验证 fallback artifact 不含这些指令。

## 6. 正确性门禁

- 每个 vector variant 与 CPU scalar 做 differential；BF16 按 FP32 accumulation + BF16 output 的参考比较。
- 覆盖长度 `0/1/VL-1/VL/VL+1/2VL+3`、非对齐地址、零维/空维约定、矩形 matmul 和 reduce 多轴。
- 覆盖 capability 缺失、artifact feature 篡改、错误 target、只读输出和缓存隔离。
- QEMU 不用于 x86；本地主机结果记录其 KVM 身份，不把当前数字冻结为性能阈值。

## 7. 完成定义

- `cpu-vector-common`：target-owned plan 与 scalar 差分测试通过，不声称实际 SIMD。
- `cpu-avx2`：实际编译、反汇编、运行与 tail 差分通过。
- `cpu-avx512`：能力分层、实际编译/反汇编/运行和低能力 fallback 通过。
