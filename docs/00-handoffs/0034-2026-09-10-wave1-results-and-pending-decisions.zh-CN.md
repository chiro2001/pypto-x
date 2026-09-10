# PyPTO-X W6 波次 1 结果与待决策快照

归档序号：`0034`

归档日期：2026-09-10（Asia/Shanghai）

状态：`QWEN35_BF16_WAVE1_PARTIAL_COMPLETE_WEIGHTED_EXECUTION_PENDING_PLUS_USER_DECISIONS`

## 1. 波次 1 任务结果

| task | 状态 | 交付 |
|---|---|---|
| `qwen35-bf16-reference` | **完成 PASS** | commit `f3375f30e`，已 cherry-pick 进 integration（`52b7e3d7c`） |
| `w8a8-linear-contract` | **完成** | 契约 `docs/20-planning/0002-…`；D1–D12 已获用户批准为报告默认值（控制仓 `4b8bac7`） |
| `amd-igpu-runtime-feasibility` | **完成** | 审计 `research/audits/2026/0004-2026-09-10-amd-gfx1036-runtime-feasibility.zh-CN.md` |
| `qwen35-perf-threshold-protocol` | **完成** | 协议 `docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` + `configs/perf_protocol.proposed.yaml` |
| `qwen35-bf16-weight-ingestion` | 进行中 | 波次 2 依赖项 |
| `qwen35-gdr-t128-validation` | 进行中 | 关闭 `gdr_t128_not_validated` |

## 2. 官方参考（gold）已冻结

固定 revision `2fc06364715b967f1860aea9cf38778875588b17`，`local_files_only=True`，torch 2.14.0+cpu / transformers 5.17.0 / numpy 2.5.3：

```text
prompts      en_continuation / zh_continuation / chat_zh_user（官方 chat template）
每 prompt    prefill 全位置 + 末位 logits、4 步 greedy 逐步 logits 与 argmax、
             逐层 hidden states [25,seq,1024]（含被官方 tie 覆盖的第 23 层原始输出）
float32 主参考 + bf16 对照：prefill max_abs 0.235/0.278/0.611，mean_abs 0.026/0.024/0.045，
             两者 4 步 greedy token 完全一致
自校验       手工增量 greedy 与 model.generate token 全一致；逐步 logits max_abs ≤ 9.5e-6
资源         峰值 RSS 5598.6 MiB；14.1 s；纯 CPU；全部经 heavy runner
```

父 agent 抽查：`"The capital of France is"` → `" Paris"`；中文 chat → `你好！我是 Q`，与逐行 argmax 一致。
该参考是波次 2 logits/逐层对齐的唯一真值来源；bf16 对照给出了 O(0.1) 的 dtype 误差带。

## 3. AMD `gfx1036` 运行态结论（审计 0004）

```text
官方支持面   gfx1036 不在 ROCm 10.0.0 矩阵、Radeon/Ryzen Linux 与 WSL 矩阵、HIP SDK for Windows 表任一表中
             （HIP SDK 原文：未列出即不受官方支持）；RDNA2 dGPU 在 Windows HIP SDK 中已标 ❌
/dev/kfd     WSL2 GPU-PV 架构下不存在且无法修复（官方 AMD SMI 文档原文）
第三方证据   ROCm 7.1.1 + Granite Ridge 裸机可编译并执行 gfx1036 kernel；
             同页 rocminfo：wave32、2 CU、L1 16KB/L2 256KB、Fast F16 = TRUE（仅第三方参考，非本项目实测）
本机实测     WSL 经 Mesa 25.2.8 d3d12 + /dev/dxg 可真实 dispatch GL compute（两次结果一致）
             但该路径是 GLSL→DXIL，不能执行我们的 amdgcn-amd-amdhsa ELF，且无 bf16/int8/subgroup 扩展
6750GRE      当前不在 PCI 总线上（仅 Present=False 幽灵记录），"安装失败"首先是硬件枚举问题
```

**判定**：HIP 运行态在本机现有形态下不可达，`BLOCKED_DEVICE` 保持；AMD 唯一正式证据继续是静态 C1–C3 + host oracle。
**唯一现实的真机路径**：GamePC 装裸机 Linux 双系统（影响 5080 独占验证），或采购官方支持的 AMD dGPU（RX 7600 gfx1102 / RX 9060 gfx1200 级；不要买 RDNA2 走 Windows）。

## 4. 性能协议结论（`docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md`）

最重要的现状判定：**当前所有可运行后端都是逐 op 派发、零融合**——CPU scalar/vector 是纯 Python 解释，AVX2/AVX-512 native 每 op 一次 ctypes 调用，CUDA 每 op 一个 PTX kernel + 一次 launch；标量 matmul 是纯 Python 三重循环。因此现阶段的 token/s 度量的是 CPython 而不是 CPU/GPU。
协议冻结：3 层测量对象（L0 算子 / L1 模型 / L2 编译）、强制 `dispatch_seconds` 差值法、warmup≥5、R=21/51、CV≤0.05 与 p95/median≤1.25 的硬方差门槛、逐算子 FLOP 口径（bf16 也记 2 FLOP/MAC）、G1 正确性 → G2 稳定性 → G3 相对比值 → G4 绝对值（本阶段明确不设）。
新事实：5080 WSL 无 cuBLAS/nvcc（CUDA 基线永久 `MISSING`，停在 G3 之前）；驱动已漂移到 616.92/13.4；空闲时钟 427 MHz vs 峰值 3090 MHz；本机是 KVM guest 且无 cpufreq → 本机绝对门槛永久 `UNGATED`。

## 5. 待用户决策（阻塞项）

1. **是否授权在 5080 WSL 安装 CUDA Toolkit（nvcc/cuBLAS）？** 不装则 CUDA 永远无法进入 G3 相对门槛。
2. **prefill T 范围**：T>1 的 GDR 是静态 SSA 展开，估算 `ops(T) ≈ 4532 + 378×(T-1)`（T=128 ≈ 52,538 ops，估算非实测）。首个带权整网是否只做 T=1 decode + T=16 prefill？
3. **AMD 路线**：接受"长期只有静态证据"，还是考虑裸机 Linux 双系统 / 采购官方支持 dGPU？（同时需确认 6750GRE 是否物理在位）
4. 非阻塞：本机绝对门槛永久 `UNGATED` 是否接受；W8A8 阈值 D5 在 BF16 基线后冻结。

## 6. 边界

- 本快照仍**没有**任何 PyPTO-X 后端的带权整网执行结果；gold 参考只证明官方实现可跑，不证明 PyPTO-X 可跑。
- 性能协议是提案，`configs/perf_protocol.proposed.yaml` 尚未生效。
- AMD 的 wave32/2CU/Fast F16 是第三方证据，不得写成项目实测能力。

## 7. 用户决策更新（2026-09-10 追加）

```text
prefill 范围   采用自动模式：首个带权整网只做 gold 参考对应的 T=5/8/18 + 4 步 decode；
               T=64/128 prefill 待首轮对齐通过后另立任务（避免一次进入约 5 万 ops 的静态展开）
CUDA 工具链    授权现在安装 CUDA Toolkit（nvcc + cuBLAS），任务 cuda-toolkit-wsl 已派发；
               范围仅限 toolkit 包，禁止任何驱动类包；性能数字在协议生效前仍是原始事实
AMD 路线       接受 AMD 长期只有静态证据（静态 C1–C3 + host oracle 为唯一正式证据）；
               不再在 WSL2 内寻找 /dev/kfd，不装 ROCm/HIP 赌 gfx1036，不启动 HIP runtime 实现工作
```

仍未决：6750GRE 是否物理在位（仅影响未来是否可能恢复 gfx1031 目标，不阻塞当前计划）；
W8A8 的 L4–L6 具体阈值按 D5 在 BF16 整网基线之后冻结。
