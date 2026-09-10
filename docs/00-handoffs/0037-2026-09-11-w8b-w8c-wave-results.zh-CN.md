# PyPTO-X W8B/W8C 波次快照：reduce/broadcast 原生化、CUDA 事件计时、W8A8 契约底座

归档序号：`0037`

归档日期：2026-09-11（Asia/Shanghai；本轮工作跨 09-10→09-11）

状态：`W8A_VERIFIED_W8C_C1_C2_VERIFIED_W8B_B5_REDUCE_BROADCAST_VERIFIED_W8B_B3A_CUDA_EVENT_TIMING_VERIFIED_ASCEND_A2_ACCEPTANCE_PASS`

本快照覆盖：W8C C1+C2（W8A8 scheme/binding 契约底座）、W8B B5（AVX-512 reduce 超线性修复 + broadcast 原生化）、
W8B B3a（CUDA cuEvent 计时，含平台时钟偏移发现与门禁重设计）、ERR-0003 与会话期间的独立性验收结论。

---

## 1. 冻结点

```text
upstream base        34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
integration base     9aae4649e58cd993f756fe85d897d866dd990e15
integration head     8700f741629616bdc16848e49ffe7fecde6807d3
控制仓 main（本地） 见 configs/development_lock.yaml（control_repo_head）
patches              118（base 34475e0d8）
```

本轮 integration 新增 7 个提交（自 `dca302ef4` 起）：

```text
030e68fe1  feat(cuda): add cuEvent kernel timing to the PTX runtime
61745243c  feat(pypto-x): add W8A8 scheme descriptor and scalar reference goldens
7e9067fce  feat(pypto-x): add W8A8 binding schema v2 and packed layout v2
657b38884  fix(cuda): make the cuEvent self-proof prove the timer rate, not raw equality
988f9ad12  fix(cuda): flag wsl2_cuevent_rate_offset whenever raw wall agreement fails
94ead3e2c  perf(pypto-x): make AVX-512 reduce grouping and broadcast native/T-linear
8700f7416  fix(cuda): calibrate the event timer with a kernel-free sleep span
```

## 2. W8B B5：reduce 超线性修复 + broadcast 原生化（PASS）

**根因**：`Avx512Runtime._execute_reduction` 对每个输出元素全量扫描输入并逐 `(out,in)` 对重算坐标，
`[1,T,16,128]` 末轴 reduce 需 `32768·T²` 次 Python 配对（T=18 约 10.6M 对、单算子 ~7s），
T=5→18 实测 13.06× 与 `(18/5)²=12.96` 吻合；`broadcast` 被固定在 `host_preexpanded_operations`，
runtime 落 `host_reference`，每个算子 3 遍全量 Python 物化。

**修复**：新增 `ptx_avx512_broadcast_{f32,bf16}` 与 `ptx_avx512_gather_offsets_{f32,bf16}`；
reduce 走"trailing 轴指针偏移 / 非 trailing 轴 native gather + 同一 reduce kernel"；
broadcast 移出 host fallback；ABI marker `cpu.avx512:3→4`、payload `2→3`（旧产物在
`_validate_primary_semantics`、native module 校验、SO ABI marker 比对三处被硬拒）。

**结果**（T=18 chat prefill，`--stage s2 --prompt chat_zh_user`）：

```text
prefill 墙钟      750.718s → 208.174s（作者）/ 233.33s（验收方复跑，+12% 负载差）
逐算子合计        695.627s → 152.613s（作者）/ 171.32s（验收方）
per-op ms/call    reduce_sum 5963.25→2.454、reduce_mean 1990.53→0.584、
                  reduce_max 249.73→0.695、broadcast 146.12→0.146（962/962 native，host_reference=0）
T18/T5 标度       12.75×/12.81×/38.81×/1.89× → 1.50×/1.71×/3.83×/1.22×（超线性消失）
位级一致          作者 58 例/315 pattern；验收方自建 base 树 43 例/265 pattern：mismatch=0
整网              76/76 输出 raw_sha256 与 base 相同；comparisons/graph_digest 一致
全量 UT           753 collected / 746 passed / 7 skipped / 0 failed（271.49s）
```

**未改路径仍是下一步热点**（T=18，秒）：matmul 65.9、reshape 27.8、slice 27.3、transpose 20.7。

**独立验收**：`verify-qwen35-native-reduce-broadcast`（agent `88e97414`）PASS，无阻断项。
非阻断 discrepancy：作者 T5 行混用 after wall 与 base op_seconds（正确口径：wall 268.64→66.121，op_s 112.895→44.700）、
集成计数应为 746/7/753、验收方 T=18 +12% 属主机负载、混合 dtype 回退公共 API 不可达、f32/bf16 极值累加语义为既存。

**新增 known limits**：`avx512_non_trailing_reduce_offsets_not_scale_tested`、
`avx512_broadcast_rank_gt16_or_mixed_dtype_host_fallback`、`cpu_reshape_slice_concat_still_host_reference`。

## 3. W8B B3a：CUDA cuEvent 计时（三轮验收后 PASS）

**实现**：`driver.py` 绑定 `cuEventCreate/Record/Synchronize/ElapsedTime/Destroy` 并提供 `CudaEvent`（幂等 close、
异常路径必 destroy）；`runtime.launch_timed` 产出 `kernel_seconds`（raw event）、`wall_seconds`、
`dispatch_seconds`（派生 = wall−kernel）；驱动缺 event API 时 `kernel_seconds=null` + `timing_error`，launch 不失败。

**平台发现（本轮最重要的测量学结论）**：GamePC WSL2 上 **cuEvent 速率系统性偏慢**，与 kernel 大小/网格无关：

```text
已知 5.000s host sleep 夹在单对 event 内：event 实测 4.6972 / 4.7685 s
19s batch 三方时钟：本机控制时钟 18.7046s、WSL realtime 18.8319s、cuEvent 17.5996s（慢 6.28%）
逐 launch event vs cuCtxSynchronize host 等待比值：1.056–1.068（1024/2048/4096/8192）
```

→ 原始"raw cuEvent vs wall ≤5%"门禁在该平台**不可能稳定通过**（首轮独立验收 3/3 FAIL 即由此而来）。

**门禁重设计**（不放松、只重新表述被证明的量）：

```text
PASS iff rate_spread_pct ≤ 3  AND  linearity_pct ≤ 3  AND  corrected_wall_agreement_pct ≤ 5
rate      = host_wall / cuEvent_span（run 内 kernel-free record→sleep→record，实测、不硬编码）
corrected = batch_event × median(rate)
raw 口径  单独上报；raw 差>5% → raw_wall_agreement=false + platform_flags=["wsl2_cuevent_rate_offset"]
```

修复轮 3 次全新运行全 PASS（raw p50 38.58–38.98ms、corrected 40.37–42.71ms、rate 1.0437–1.0972、
clock 0.9078–0.9100、peak_rss 141.4–145.6 MiB，余量 2.6–4.8×）。`kernel_seconds` 仍为 raw，
`kernel_seconds_corrected` 单列，不替代 raw。

**独立验收**：`verify-cuda-kernel-timing`（agent `3eb15053`）三轮：
r1 FAIL（门禁不可复现 + 缺 `peak_rss_mib` + 措辞/口径）→ 修复 → r2 PASS（3 条不阻断）→
delta 修复（kernel-free 标定、MiB 解析）→ r3 **PASS，0 discrepancy**。

## 4. W8C C1+C2：W8A8 契约底座（PASS）

**C1**（`758827cf3`）：`QuantizedTensorDesc`（§3.3 字段，`per_group` reject，to_dict/from_dict 往返）+
4 个 opcode scalar golden（`quantize_per_token_s8` / `dequantize_s8` / `qmatmul_s8s8_s32` /
`dequantize_epilogue_bf16`），Q1–Q6 全覆盖，R1–R5 通过。
`CoreProgram.SCHEMA_VERSION` **保持 1**（新类型是加法、不入 CoreProgram 序列化）。

**C2**（`b3de4d99e`）：binding schema v2（`BINDING_SCHEMA_VERSION 1→2`，悬空 `scale_entry`/未知键拒绝）、
packed layout v2（`PACKED_LAYOUT_VERSION 1→2`，默认 518 region = 320 param + 150 quant_scale + 48 state，
v1 `reject_and_recompile`）、覆盖率 manifest：

```text
w8a8_linear_coverage          = 0.9988146977479258
whole_net_int8_compute_ratio  = 0.9770048309178744（150/186，static shape model @ batch1/prefill1/past1）
```

**独立验收**：`verify-qwen35-w8a8-scheme-binding`（agent `d640b1c6`）PASS；
独立 oracle scheme 84/84、binding 58/58；integration HEAD 全量 749 collected / 742 passed / 7 skipped / 0 failed。
非阻断 discrepancy：VD-1 base 漂移（真实漂移 0）、VD-2 计数口径（742/7/749）、VD-3 契约算术（见 ERR-0003）、
VD-4 线性 MAC 未乘 batch（batch=1 不受影响）、VD-5 饱和计数结构、VD-6 自报布尔已审计。

**未实现且 fail-closed**：`precision='w8a8-linear'` 图路径、量化 tied `lm_head` 双 packing（§3.4）。

## 5. 勘误 ERR-0003

契约 §3.1 全开 region 计数 555 与 §3.4 lm_head 双 packing（+2）算术不自洽（186 线性 = 554，+2 = 556）。
裁定：以 §3.4 为准，全开应为 556；该路径未实现且 fail-closed，不影响已冻结计数。
详见 `docs/00-handoffs/ERRATA.zh-CN.md#err-0003`。

## 6. 新增 known limits（development_lock.waves.W8.known_limits_changed_2026_09_11）

```text
wsl2_cuevent_rate_offset                              # 平台：raw cuEvent 偏低 4.4–9.1%，run 间漂移
avx512_non_trailing_reduce_offsets_not_scale_tested   # B5：该路径 Qwen 0 次使用，未做大集压测
avx512_broadcast_rank_gt16_or_mixed_dtype_host_fallback
cpu_reshape_slice_concat_still_host_reference
w8a8_linear_graph_precision_not_implemented
w8a8_lm_head_dual_packing_pending_contract_erratum
w8a8_coverage_is_static_shape_model_estimate
```

## 7. 未解决与下一步（供接手 Agent）

1. **性能续作**：T=18 剩余大头 matmul 65.9s / reshape 27.8s / slice 27.3s / transpose 20.7s；
   W8B 表 B1（AVX2 packed）、B2（SVE fallback）、B3b（cuBLAS GEMM 基线）、B3c（本机 L0/L1）、B4（perf freeze）均未做。
2. **W8C 续作**：C3（AVX2 widening）→ C4（VNNI/`vpdpwssd`）→ C5（SVE）→ C6（CUDA/AMD 静态）→ C7/C8（layer ladder 与整网验证）；
   R6/R10–R14 尚未验收；D5 阈值需在 BF16 基线后冻结。
3. **框架接入（待用户裁决）**：是否立 W8J（torch custom-op / vLLM plugin / HF 集成），优先级与形态待定；
   不做 tokenizer/sampling/batching/PagedAttention/分布式/autograd。
4. **W8D/E**：D2（T=64/128 代价评估）、E4（Core IR→PTO/CCE codegen，A2 限定式关闭后的真实缺口）是否启动待定。
5. **W8H/W8I 后续范围**：ACL graph 口径精度复跑、并发 sweep、`logprobs=-1` 全词表往返是否补做待定。
6. **本机磁盘**：约 78 GB 可用，§8 可删清单（安装包/venv/build 目录）未获批准。
7. **发布**：控制仓 main 本地已提交、私有备份待同步；公开 push 需用户明确批准。

## 8. 证据路径

```text
_meta/pypto-x/qwen35-native-reduce-broadcast/          # B5 实现与 heavy 证据
_meta/pypto-x/verify-qwen35-native-reduce-broadcast/   # B5 独立验收（自建 base 树/自写 emitter）
_meta/pypto-x/cuda-kernel-timing/                      # B3a 实现（report/platform-timer-evidence/run-history/probes）
_meta/pypto-x/verify-cuda-kernel-timing/               # B3a 首轮验收（FAIL 证据）
_meta/pypto-x/verify-cuda-kernel-timing-r2/            # B3a 复审 PASS
_meta/pypto-x/verify-cuda-kernel-timing-r3/            # B3a delta 复审 PASS（0 discrepancy）
_meta/pypto-x/qwen35-w8a8-scheme-binding/              # C1+C2 实现
_meta/pypto-x/verify-qwen35-w8a8-scheme-binding/       # C1+C2 独立验收（84/84 + 58/58）
```
