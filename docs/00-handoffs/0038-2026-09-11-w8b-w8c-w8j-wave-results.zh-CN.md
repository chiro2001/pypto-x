# PyPTO-X W8B/W8C/W8J 波次快照：blocked GEMM、W8A8 图路径与 vLLM 接入面

归档序号：`0038`

归档日期：2026-09-11（Asia/Shanghai；本轮工作跨 09-11 夜间）

状态：`W8B_B1_AVX2_PACKED_VERIFIED_W8B_AVX512_BLOCKED_GEMM_VERIFIED_W8B_B3B_R3_PASS_W8C_C3_C4_VERIFIED_W8C_W8A8_GRAPH_PATH_VERIFIED_W8J_J1_J2_VERIFIED_ERR_0003_ERR_0004`

本快照覆盖：W8B B1（AVX2 packed 参数级内核）、`avx512-blocked-gemm`（AVX-512 packed/blocked matmul，
prefill matmul 12.4× 且位级零漂移）、B3b（CUDA GEMM vs cuBLAS 相对基线 + 协议修订 ERR-0004）、
W8C C3/C4（AVX2 / AVX-512 VNNI 的 W8A8 widening）、**W8A8-linear 图路径与真权重量化接入**（含 ERR-0003 落实）、
W8J J0–J2（torch 桥 + J1.1 packed 平面 + vLLM CPU 插件）、以及一次跨 4 次 artifact 版本升级的
BF16 整网累积回归。

---

## 1. 冻结点

```text
upstream base        34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
integration base     8700f741629616bdc16848e49ffe7fecde6807d3（0037 波次收口点）
integration head     227506c862e1a5de5be17f59f6f1c45ca75b26e9
本轮 integration +16 提交（8700f7416..227506c86）
patches              134（base 34475e0d8；上一波次 118）
控制仓 main          见 configs/development_lock.yaml（control_repo_head）
```

本轮 16 个提交：

```text
26a444b86  perf(pypto-x): add native AVX2 packed cast/transpose/embedding kernels
18c5abec7  feat(pypto-x): add native AVX2 W8A8 widening path
ac5a44d54  feat(pypto-x): add torch custom-op bridge for the AVX-512 linear kernel (W8J)
f1a606023  perf(pypto-x): add AVX-512 packed/blocked matmul (W8B avx512-blocked-gemm)
d5082bb1d  perf(pypto-x): stage transposed AVX-512 packed weight tiles
d5cdb8518  fix(pypto-x): evict packed-weight cache entries without dict.popitem(last=)
a3a50b937  feat(pypto-x): add native AVX-512 VNNI W8A8 widening path
fa8ab359d  fix(pypto-x): make the torch linear bridge fail open on real build failures (W8J J1 fix 1)
9ab96f52b  perf(cuda): add GEMM vs cuBLAS relative-baseline harness (B3b)
273461930  feat(pypto-x): execute the torch linear bridge on the packed plane (W8J J1.1)
4d6e80125  fix(pypto-x): count the per-dtype pack module apart from shape kernels (W8J J1.1)
336cead94  feat(pypto-x): add the vLLM CPU general plugin for packed dense Linear (W8J J2)
691c593b5  feat(pypto-x): add W8A8-linear graph path and quantized weight ingestion
378b738a8  fix(perf): enforce protocol outlier gate and per-side warmup in G2 verdict
901040cec  fix(perf): require one batch size across aggregate rounds
227506c86  fix(perf): apply ERR-0004 outlier floor and close R2-D2 round validity
```

---

## 2. W8B：硬化与性能

### 2.1 B1 `avx2-packed-kernels`（PASS_WITH_BOUNDARIES）

- AVX2 补齐 packed 参数级 `cast`/`transpose`/`embedding`（此前落共享 host reference）。
- 验收：AVX2 vs AVX-512 **逐位一致 55 组 / 96,784 元素 / 0 mismatch**（含 bf16↔f32 全位型穷举、
  ±0/Inf/NaN/subnormal/RNE tie、非方阵、embedding 边界、越界拒绝）；独立 FAIL-CLOSED **15/15**；
  边界审计 16/16；objdump `ymm=298 / zmm=0 / EVEX=0`。
- 契约：payload 1→2、ABI marker `pypto-x.cpu.avx2:2→:3`（旧产物硬拒）。
- 回归：base 753/746/7 → head **815/808/7/0**。
- 边界：`transpose`/`embedding` 是 target 自有分块 C kernel（非 SIMD），只有 `cast` 是 256-bit 向量；
  覆盖范围限 rank-2 (1,0) transpose / axis-0 int64 embedding / f32-bf16 cast+copy，其余仍 host reference。

### 2.2 `avx512-blocked-gemm`（PASS_WITH_BOUNDARIES；本波次最大性能收益）

- K64×N64 packed/blocked rank-2 matmul：f32 k-major；bf16 按列 k-pair 交错（一条 64B load = 16 列 × 1 k 对）；
  4 路 ZMM 列累加；**保持 k 顺序 ⇒ 与旧内核逐位一致**。另有 transposed (N,K) 根的 64×64 tile 暂存、
  batched rank-4 指针循环（去 Python list 物化）、有界线程 1..6、opt-in 跨 launch 权重缓存。
- **性能（en T=5 整网）**：prefill matmul **17.262s → 1.392s（12.4×）**，其中 batched **16.457s → 0.197s（83.5×）**；
  prefill launch 59.2→47.3s；wall 173.4→161.8s。vs 旧内核真实形状 3.0–32×。
- **数值零漂移**：24 小形状 + 83 基准形状逐位一致；**en T=5 整网 5 run × 76 输出 = 380/380 sha256 与 base 相同**。
- 契约：payload 3→4、marker 4→5；dispatch 谓词（参数链 packed 判定）篡改 8/8 硬拒。
- 回归：base 881/874/7 → head 892/885/7/0。
- 诚实边界：vs oneDNN 6T 的 f32 大 K（M=64、K=8192）仍约 **5.6× 慢**（作者原摘要遗漏，验收方 D2 修正）；
  单个超限 entry 可突破缓存字节上限（软上限）。

### 2.3 B3b `cuda-gemm-baseline`（r3 PASS；B4 输入就绪）

- 交付 `scripts/smoke/cuda_gemm_baseline_smoke.py`（measure / g1 / aggregate 三模式；cuBLAS 经 ctypes
  `cublasSgemm_v2` / `cublasGemmEx` 同口径计时）。
- **G1 PASS / G2 PASS / G3_ELIGIBLE**，4 个 eligible key 的 `candidate_ratio`（ours/cuBLAS，corrected median）：

```text
matmul (4096,4096,4096) f32 sm_80 0-5   16.625991    （独立复验 16.586493，−0.238%）
matmul (4096,4096,4096) bf16            31.376162    （独立复验 31.552105，+0.561%）
matmul (2048,4096,4096) f32             16.819266    （独立复验 16.952400，+0.792%）
matmul (2048,4096,4096) bf16            31.504972    （独立复验 31.504705，−0.001%）
```

- Qwen 形态（`128×3584×1024`、`1×248320×1024`）在当前逐 op 派发下 dispatch-dominated（share 0.67–0.75），
  按协议只记录、不进比例门。
- **复验波折（值得记住）**：首次复验发现 harness 漏实现协议 §3.3 第 6 步（D1）→ 修复 + K=10 重测 →
  复验又发现**协议本身在 WSL2 不可复现**（R2-D1）→ 触发 **ERR-0004 协议修订**（MAD 门加 floor）→
  重聚合后 K=10 三轮仍 VALID、数值不变（≤5e-6），而 K=5 在新判据下**仍 UNGATED**（证明修订不是放水）。
- 平台：`wsl2_cuevent_rate_offset`（+8.6~10.0%），raw + `kernel_seconds_corrected` + `platform_flags` 同报。
- **B4（阈值冻结）仍待用户批准**；`frozen_ratio=null`，未写任何 lock/阈值文件。

### 2.4 未做

`B3c`（本机 L0/L1 性能）需静默窗口；`B2`（SVE fallback 分类清零）随 920B 封存。

---

## 3. W8C：W8A8-linear 实现

### 3.1 C3 `qwen35-w8a8-cpu-avx2`（PASS_WITH_BOUNDARIES）

- `qmatmul_s8s8_s32` 走 Core IR 全链 native（`vpmovsxbw + vpmaddwd + vpaddd`）；另 3 个 opcode 为
  native target entry point（Core IR 多输出接线留 C7/C8）。
- 能力声明 `CapabilitySet(dtypes=int8/int32/f32/bf16, matrix=s8s8_s32_widening)`，probe-gated，无 VNNI over-claim。
- 独立 harness **238/238**；objdump 无 EVEX/zmm。
- 契约：AVX2 payload 2→3、marker :3→:4。回归 815/808/7 → **877/870/7/0**。

### 3.2 C4 `qwen35-w8a8-cpu-avx512`（PASS_WITH_BOUNDARIES）

- `vpdpbusd`（u8×s8）实现 s8×s8：`Σa·b = Σa(b+128) − 128·rowsum(a)`；
  最坏 K=133144、a=b=127：偏置中间值 4,311,868,440 → 精确 2,147,479,576。
- 独立 harness **116/116**；反汇编 `vpdpbusd=1`、`vpdpwssd=0`；真实 CPUID VNNI + xcr0=0x2e7 probe-gate。
- 契约：AVX-512 payload 4→5、marker :5→:6。回归 892/885/7 → **983/976/7/0**。

### 3.3 W8A8-linear 图路径与真权重量化（PASS_WITH_BOUNDARIES）——**本波次能力里程碑**

- `build_qwen35_text_decoder_graph(precision="w8a8-linear")` = **graph contract v4**：
  int8 权重 + fp32 per-output-channel scale 为函数参数；链
  `quantize_per_token_s8`（双输出）→ `qmatmul_s8s8_s32` → `dequantize_epilogue_bf16`。
  **BF16 v3 逐字节不变**（digest `66dd4077…`、4,550 ops、368 region）。
- region 真实计数：**518 默认 / 554 全 186 / 520 默认+lm_head / 556 全开**（ERR-0003 修正成立）。
- 量化正确性：权重量化器与 C1 golden **逐位一致**（复验方自写 Fraction-exact oracle 15/15 组）；
  真权重默认 150 层：int8 **497,025,024 B** + scale 1,597,440 B、饱和 **0**、manifest digest `4fe99771…`。
- **端到端 en T=5（真权重，W8A8 vs BF16）**：prefill cos **0.9956335779** / max_abs 2.734375 / mean 0.24627，
  argmax 相同、top5 5/5；decode 4 步 cos 0.99602–0.99812、argmax 全同；
  **token 链 11751→13→198→760→6511 与 BF16 完全一致**；逐层 min cos 0.98679425（首次分叉 layer_00_output）；
  state 全有限（max 37.25）。
- **覆盖率（真实 vs 旧 static）**：`w8a8_linear_coverage = 0.9988146977479258`；
  `whole_net_int8_compute_ratio` T=5 **0.6483367350209979** / T=1 0.6484198826534429；
  旧 static 估计 0.9768160741885626 **偏高**，差异已**完全对账**（旧式漏 lm_head matmul 1,271,398,400 MAC、
  GDR 递推公式差 17,694,720、attention past 差 122,880）。
- **诚实边界**：实测**超出契约 §5.2 暂定 L4 阈值**（max_abs ≤0.5 / mean ≤0.05 / cos ≥0.999）——实现方明确声明、
  未虚假达标；R12 正式门禁留 C8，且需先由用户裁定阈值口径。
- 契约：AVX2 payload 3→4/:5、AVX-512 payload 5→6/:7、W8A8/packed kernel version→2。
- 回归：collect 983 → 997；全量 **991 passed / 7 skipped / 0 failed**。
- 已知缺口：`quantize_lm_head=True` 的**真实权重 ingestion 仍 fail-closed**（同一 checkpoint 张量不可被两参数认领）；
  556 已在 schema/layout/synthetic 层验证。跨后端整网非逐位一致（int8 op 级均逐位对齐，差异源自 f32 孤岛累加顺序翻转 RNE 边界）。

### 3.4 未做

C5（SVE，随 920B 封存）、C6（CUDA/AMD static）、C7（1→6→24 阶梯）、C8（R12–R14 正式门禁）。

---

## 4. W8J：把内核放到 torch / vLLM 里

### 4.1 J0 spike（Gate 三项 = 是）

- vLLM CPU（`vllm-cpu==0.29.0`）**原生支持 Qwen3.5**（自带 `qwen3_5.py` + CPU GDN 路径）；
  本地权重 greedy 16 token 连贯、两次运行 token 一致；baseline **1.75 tok/s**、peak RSS 10.7 GiB。
- 注入可行：`vllm.general_plugins` + 自定义 `LinearMethodBase` 实测拦截 **912 次 apply / 114/114 前缀**。
- 环境：独立 conda env `pypto-x-w8j`（3.5 GiB；清华源无 torch CPU wheel，需 `--extra-index-url https://download.pytorch.org/whl/cpu`）。

### 4.2 J1 torch 桥（r1 FAIL → 修复 → r2 PASS_WITH_BOUNDARIES）

- API：`torch.ops.pypto_x.{linear, linear_strict, pack_weight, linear_prepacked, register_torch_ops}`。
- r1 阻断：真实构建失败（只读缓存/编译器失败/产物损坏）**不 fail-open** → 修复：真实异常族统一包装为
  `PyPTOXBuildError`（reason_code 分档），open 路径计数+warn-once+精确回退，strict 结构化；
  另修产物自愈（隔离脏 `.so` 后重建）、`pack_weight` 降到 **1 次** payload 拷贝、文档改为 inference-only。
- 缓存：key 含 `(key_version, op, dtype, m, k, n, target, lowering/payload/ABI/format 版本)`；
  cold 6 builds/6 compiles → warm **0 builds / 0 compiles / 6 disk hits**。
- 回归：主 suite 983/976/7/0；torch suite 76 passed/1 skipped。

### 4.3 J1.1 桥切 packed 平面

- `declared_plane == executed_plane == "packed"` 成为不变量（加载期与 artifact metadata 交叉核对 + 单测 pin）；
  **32/32 行 packed 与 legacy 逐位一致**；cache key v2，旧 sidecar 自然 miss。

### 4.4 J2 vLLM 插件（PASS_WITH_BOUNDARIES，0 阻断）

- 可分发包 `python/pypto_x_vllm/` + `vllm.general_plugins` entry point；**不写 site-packages**（PYTHONPATH 即装即卸）；
  模块顶层类（spawn 安全）；`get_quant_method` 仅接管 LinearBase。
- E2E：5 次运行 token 链与 baseline 完全一致；**logprobs top-5 80 对 max_abs_delta = 0.0**；卸载即回 baseline。
- **性能门禁 0/12**（12 个真实形状，三配置）：

```text
our_1T_default      4.71–12.69×（慢于 vLLM 原生 oneDNN 6 线程）
our_6T_default      5.57–14.42×
our_6T+weight_cache 5.19–14.47×
→ 0/12 通过 ≤1.0 门禁 ⇒ 注入默认关闭（3648 次 fallback、0 注入、reason=shape_disabled）
```

- **结论口径**：W8J 的接入面完整、可插拔、可卸载、fail-open、可审计，且端到端数值与 baseline 完全一致；
  但**在当前 x86 CPU 上我们的内核尚不能与 oneDNN 竞争**，因此注入默认关闭——这是**性能边界，不是机制缺陷**。
  要让内核真正参与推理，只有：① 继续投入内核性能（对标 oneDNN 是硬仗）；② 用户明确接受"更慢但自主栈"；
  ③ 换到 oneDNN 不占优的场景（W8A8 整型路径、昇腾线）。

### 4.5 J3/J4

J3（整网且我们的内核真正参与）被性能门禁挡住；J4 收口随本快照完成。

---

## 5. 横切

### 5.1 累积 BF16 回归（**零漂移**）

在 chain 末尾（`fa8ab359d`，即 A1/C3/C4/J1/J1fix/GEMM 全部合入后）独立复跑 en T=5：

```text
380/380 输出 sha256 与冻结基线逐一相同（5 run × 76 输出）；5 个 dump npz 文件字节级相同
argmax 5/5；cos 0.999972352851843；max_abs 0.159756183624；band 0.235211610794/0.999956322868403
final_hidden 0.999944851495287/0.334758758545；逐层 min cos 0.999930983236776 @ layer_12_output
decode 链 11751→13→198→760→6511；state 有限（|max| 0.550185–13.273482）
图契约 (1,1,4096)=66dd4077…/4550 ops、(1,5,0)=6728 ops
旧版本产物（payload4/marker:5、payload2/marker:3）与回滚 payload 全部被 HEAD 拒绝
全量 983/976/7/0
```

即：**4 次 artifact 版本升级（AVX2 1→2→3→4、AVX-512 3→4→5→6）后 BF16 语义零漂移**。

### 5.2 ERR-0003（契约 555 → 556）

W8A8 契约 §3.1 全开 region 计数笔误，真实矩阵证实 **556**；契约文档已修正，`development_lock.yaml` 与
本快照登记一致。

### 5.3 ERR-0004（性能协议 §3.3 第 6 步 MAD 门加 floor）

- 问题：原判据 `|x−median| > 3×1.4826×MAD` 无下限，在 WSL2 上 MAD 仅占中位 0.06–0.83%，
  门限低到 0.28–3.7%，把**微秒级计时抖动**当污染 → G2 有效性不可复现（复验方 2 campaign / 8 轮 →
  5 轮作废、0 个有效 campaign；`cv=0.0023 / p95/median=1.0014` 的极稳序列也被判 UNSTABLE）。
- 修订（用户 2026-09-11 批准方案 a）：`|x−median| > max(3×1.4826×MAD, floor)`，`floor = max(1 µs, 2% × median)`；
  `outlier_ratio > 0.10 → UNSTABLE` 不变。真污染（如 cuEvent rate +8~10%）仍被标记。
- 影响面：B3b 的 candidate_ratio 数值稳定（两方独立复现偏差 ≤0.79%），但**B4 冻结必须在修订后重聚合**。

---

## 6. 新增 known limits（本波次）

```text
avx2_packed_nan_payload_differs_from_scalar_matches_avx512      # B1：NaN payload 与 scalar 语义不同（与 AVX-512 一致）
w8a8_avx2_q3_native_saturation_count_gt0_not_externally_observable  # C3：Q3 饱和计数公开路径结构性不可达
avx512_packed_gemm_f32_large_k_still_about_5_6x_slower_than_onednn_6t  # GEMM：f32 大 K 仍慢 ~5.6×
avx512_weight_cache_byte_cap_is_soft_for_single_entry           # GEMM：缓存字节上限可被单条 entry 突破
avx512_w8a8_target_has_no_fma_float_matmul_ulp_shift_within_artifact  # C4：同 artifact 内 float matmul ULP 级变化
w8j_torch_bridge_inference_only_no_autograd                     # J1：无 backward
w8j_torch_bridge_bf16_above_k128_not_bitwise                    # J1/J1.1：bf16 K>128 非逐位
w8j_vllm_cpu_injection_disabled_by_performance_gate_our_kernel_4_7_to_14_5x_slower_than_onednn  # J2
w8a8_selected_150_layer_scheme_exceeds_tentative_l4_thresholds  # W8A8 图路径
w8a8_lm_head_dual_packing_real_weight_ingestion_fail_closed     # W8A8 图路径
```

**已解除**：`w8j_torch_bridge_executes_legacy_zmm_not_packed_plane`（J1.1 已切 packed 平面，J2 验收确认
`declared==executed==packed`）；登记保留以保留历史。

---

## 7. 未解决与下一步（供接手 Agent）

1. **发布动作待批准**：控制仓 main 有多个本地提交未 push；`patches/` 已刷新为 134；推送属发布动作，需用户明确指示。
2. **B4 性能阈值冻结**：B3b 的 4 个 candidate_ratio 已就绪（cpu_affinity 口径），但仍缺 `B3c`（本机 L0/L1）；
   冻结需用户批准并写入 `configs/perf_lock.yaml`。
3. **W8A8 阈值口径**：L4 暂定阈值被现 scheme 超出；需用户裁定"接受该误差预算"或"先做校准/平滑"，
   之后才能启动 C8（R12–R14）。
4. **`quantize_lm_head` 真权重 ingestion**：闭合 556 全开配置的最后一个已知缺口。
5. **C7 阶梯**（1→6→24 层，R10/R11）与 **C6**（CUDA/AMD static）。
6. **内核性能专线**：只有把 GEMM 追到 ≤1.0× oneDNN，J2 的注入才会真正启用（或用户明确接受更慢的自主栈）。
7. **封存项**：A2(910B3) 暂停待通知（E4/E5、W8H/W8I 后续）；920B 已释放，需 B2 时按 `~/tools/ecs-920B/` 重建。

---

## 8. 证据路径

```text
W8B B1                 _meta/pypto-x/verify-avx2-packed-kernels/
W8B blocked-gemm       _meta/pypto-x/verify-avx512-blocked-gemm/
W8B B3b                _meta/pypto-x/cuda-gemm-baseline/、verify-cuda-gemm-baseline{,-r2,-r3}/
W8C C3                 _meta/pypto-x/verify-qwen35-w8a8-cpu-avx2/
W8C C4                 _meta/pypto-x/verify-qwen35-w8a8-cpu-avx512/
W8C graph path         _meta/pypto-x/qwen35-w8a8-graph-path/、verify-qwen35-w8a8-graph-path/
W8J J0                 _meta/pypto-x/w8j-vllm-cpu-spike/
W8J J1/J1.1/J2         _meta/pypto-x/w8j-linear-custom-op/{,fix-round-1}、w8j-vllm-injection/、
                       verify-w8j-linear-custom-op{,-r2}/、verify-w8j-vllm-injection/
累积回归               _meta/pypto-x/verify-bf16-cumulative-en-t5/
```
