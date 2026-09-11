# PyPTO-X B6 提案：layout 类算子原生化 + dispatch 压缩

文档编号：`0004`

日期：2026-09-11（Asia/Shanghai）

状态：`PROPOSED_PENDING_USER_APPROVAL`

用途：把 W8B B3c（本机 L0/L1 性能分解，`work/local-perf-l0-l1` @ `1006ba571`）暴露出的**最大单一浪费**做成一个可执行、可验收的任务。本文自包含，数字全部来自 B3c 证据目录，未做外推。

关系：B3c = 度量（已完成，验收在途）；**B6 = 依据 B3c 结论的优化**。B4（性能阈值冻结）仍需用户批准，与本文无依赖。

---

> **勘误（ERR-0006）**：本文 §1.1 的 "dispatch" 是残差 `launch − Σ(op)` 的旧称。N4.0 实测（2026-09-12）表明该残差
> **不是逐 op 派发**，而是每 launch 的 artifact 校验 / 重复 lowering / ELF decode（loop 内胶水仅 0.003 ms/op）。
> 表格中的数字仍有效（都是实测），但**读法**应以 ERR-0006 为准。

## 1. 背景（不看会话也能读）

B3c 在 AVX-512 后端、s3 场景（T=5 prefill + 1 decode step，真实 Qwen3.5-0.8B 权重）上跑了 3 轮，给出 launch / op-sum / dispatch 三分账：

| 段 | launch 中位 | op sum | 残差（当时记作 dispatch） | 残差占比 |
|---|---:|---:|---:|---:|
| T=5 prefill（6728 ops） | 47.63 s | 28.0–29.0 s | 19.3–19.7 s | **40.3–41.3%** |
| T=1 decode（past=5，4550 ops） | 22.47 s | 11.4–11.5 s | 10.8–11.2 s | **48.7–49.3%** |

也就是说：**op 自身只占 59%/51%，另外约一半是每 op 的运行时 Python 派发（已定位为 runtime 侧 per-op dispatch，不是 harness）**。

### 1.1 op 侧内部构成（per-op 中位，3 轮）

prefill（op 合计 28.19 s）：

| 算子 | 调用数 | 合计 s | ms/call | 模式 |
|---|---:|---:|---:|---|
| reshape | 776 | **8.34** | 10.75 | `host_reference` |
| slice | 696 | **7.37** | 10.59 | `host_reference` |
| transpose | 889 | **7.27** | 8.18 | `host_reference` + `native_avx512_packed_transpose` |
| concat | 84 | 1.89 | 22.48 | `host_reference` |
| matmul | 469 | 1.34 | 2.85 | 原生（已 blocked-gemm） |
| cast | 1484 | 1.03 | 0.69 | 原生 |
| split | 24 | 0.80 | 33.36 | `host_reference` |
| 其余（broadcast/mul/add/reduce/…） | — | 0.13 | — | 原生 |

decode（op 合计 11.46 s）：transpose 3.17 / slice 2.88 / reshape 2.34 / matmul 1.09 / concat 1.03 / cast 0.74 / split 0.16。

**结论：layout 类（reshape+slice+transpose+concat+split）在 prefill 占 25.66 s / 28.19 s = 91.0%，在 decode 占 9.57 s / 11.46 s = 83.5%。** 相对地，已经原生化的 matmul/cast/broadcast/reduce 全部加起来不到 3.2 s。

---

## 2. 根因（已核实到代码，不是猜测）

1. **AVX-512 后端把 layout 类整体判给 `host_reference`**：`python/pypto/backends/cpu/x86/avx512.py` 的 `plan_execution` 只对 `identity/add/mul/reduce_*`（FP32/BF16）与 `exp/rsqrt/…` 给原生模式；`reshape/view/transpose/contiguous/slice` 落到 `mode="host_reference"`。
2. **fallback 路径是「buffer → Python list → 逐元素索引 → list → buffer」**：`_reference_operation()` 把每个输入 `_buffer_values()` 成 list，交给 `backends/cpu/vector/runtime.py`；其中 layout 分支是
   `result = _vector_map(plan, [source[_layout_source_index(layout, index, ctx)] for index in range(layout.output_elements)], ...)`，
   即**每输出元素一次 Python 函数调用 + 一次 `_unravel` 坐标分解**。10.75 ms/call 对应的是这个解释器开销，不是内存带宽。
3. **已有可用的机器**：`layout_plan` 描述符（`lowering/cpu/vector/plan.py::LayoutPlan`）**本来就是为原生 runner 设计的**——`kind ∈ {logical_layout_copy, transpose, slice}`，字段只有 `input_shape/output_shape/source_strides/permutation/slice_starts/slice_steps`，且已带 `descriptor_digest`，注释明确写着「native runner derives each source index from an output coordinate and these rank-sized arrays」。转置也已经有一个原生模式存在（`native_avx512_packed_transpose`），说明 buffer 级实现路径是通的。
4. **零拷贝视图能力已存在但没被 layout 用到**：`avx512.py::_buffer_view(buffer, offset, count)` 已经是「同 owner、零拷贝、带 offset 的 buffer 视图」。

---

## 3. 方案（分层，逐层可独立验收）

### N1（主项）descriptor 驱动的原生 layout runner

为 `LayoutPlan` 增加 target 侧原生执行：按 `descriptor_digest` 缓存一段生成的 C 源码（编译产物复用现有 AVX-512 编译缓存与 `_NativeBuffer`/`_ctype` 栈），对每个输出坐标做 rank ≤ `_LAYOUT_MAX_RANK` 的循环嵌套写入：

- `kind=logical_layout_copy`：源与目标是同一 C 序（reshape/view/contiguous）→ **memmove 级连续拷贝**；
- `kind=slice`：内层轴 `slice_steps==1` 且段连续 → `memcpy` 段循环；非单位步长 → 步长循环；
- `kind=transpose`：blocked/tiled 拷贝（rank-2 先做，rank>2 退化为坐标循环但仍在 C 里）；
- dtype 用 `_ctype(dtype)` 泛化，标量与指针窄化（int8/bf16/fp32）保持一致。

验收要求：与 `host_reference` 结果**逐位一致**（`sha256` 全 buffer 比对，含空轴/退化 shape）。

### N2（视图快路径）纯视图不拷贝

当 `output_elements == source_elements` 且 descriptor 等价于「同一段内存的重解释」时（`logical_layout_copy` 无 stride 变化），直接返回 `_buffer_view(input, 0, count)` 的别名而不是新分配+拷贝。**必须先由 liveness/`plan_release` 确认该 SSA 值在使用期内不会被写覆盖**——若不满足，N2 只作为「读侧零拷贝」而不能省掉写侧；此判定要写成显式前置检查，检查不通过就退回 N1，不允许「猜」。

### N3（次热点）concat / split 原生化

concat 1.89 s/84 calls（22.48 ms/call）、split 0.80 s/24 calls（33.36 ms/call），量级与 layout 同源（都是段结构 + host list）。rank-2 沿 axis 段拷贝即可覆盖绝大多数调用；split 在 `sizes` 连续时可退化为多个 `_buffer_view`。

### N4（另一半账）dispatch 压缩

dispatch 19.3–19.7 s（prefill）/ 10.8–11.2 s（decode）是本批**与 op 同量级**的另一半。候选：合并同类连续 op 的派发（plan 层面批处理）、减少每次派发中的对象构造（`plan_execution` 返回 dict + tuple 输入 + 大量字符串拼接）、把「无依赖连续段」在 launch 前压成批。**N4 单独度量、单独验收**，不与 N1–N3 混在一个结论里。

---

## 4. 验收口径（fail-closed）

1. **正确性**：N1/N2/N3 全量 pytest 收集数不得减少；与现有 `host_reference` 路径做**逐位一致**差分（覆盖 T=1/T=5/T=18、空轴、非单位步长 slice、rank-2/3/4 transpose、bf16/fp32/int8）；
2. **不回退**：既有原生模式（blocked-gemm、cast、reduce、broadcast、W8A8）模式判定不得改变；
3. **性能**：给出改动前后 **per-op ms/call 中位**（同协议、同轮次口径，≥3 轮）与 **launch 墙钟**；只有 per-op 数字变化才允许声称「该 op 变快」，墙钟变化要单列；
4. **数字纪律**：本机 12 vCPU KVM guest 无 cpufreq → 本机性能**永久 `UNGATED`**，只能报中位与离散度，不得报「加速比 = 权威结论」；
5. **边界显式**：N2 若因 liveness 不成立而未启用，必须在证据里写成「未启用 + 原因」，不得算进收益。

---

## 5. 风险

| 风险 | 处置 |
|---|---|
| N2 别名破坏 SSA 语义（写覆盖） | 前置检查 + 不成立即退回 N1；单独负例测试 |
| 生成 C 的 rank 上界与 `_LAYOUT_MAX_RANK` 不符 | 复用同一常量；超界走 `host_reference` 并计数 |
| `slice_steps>1` 的语义（stride）写错 | 以 descriptor 的 `slice_starts/steps` 为唯一真源，写 spec 化单测 |
| 改动触及共享的 vector/runtime | N1 只走 target 侧新路径，`host_reference` 保持为等价基线（不删） |
| 本机 perf 噪声 | 沿用 `run_local_heavy.sh`；返回 75/69 即等待重试 |

---

## 6. 资源与成本

- 位置：本机（AVX-512 12 vCPU）；持本机重任务锁；**不需要 GamePC / A2 / A3**（A2 已暂停、920B 已释放、A3 的 NPU 归用户）。
- 预算：实现 + 聚焦测试 + 3 轮 L1 复测，预计 1 个重时段；N4 另算一个时段。
- 并发：与在途的 C5/B3c/C6 三个验收不冲突（验收不写 integration）。

---

## 7. 不做什么

- 不改图/不删 layout op（那是另一个方向的优化，需先有 N1–N3 的「原生后还剩多少」数据）；
- 不动 CUDA/AMD/Ascend 后端；
- 不改 `LayoutPlan` 已冻结的 descriptor 语义（digest 一旦变化即触发 artifact 版本升级与全量回归重跑）；
- 不以「加速比」替代位级一致证据。
