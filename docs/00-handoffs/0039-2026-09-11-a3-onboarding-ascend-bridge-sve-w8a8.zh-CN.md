# PyPTO-X 0039 波次快照：A3 共享机接入、Ascend IR→PTO 首桥、SVE 四类原生化与 W8A8 556 全开

归档序号：`0039`

归档日期：2026-09-11（Asia/Shanghai；0038 收口后至本轮交接）

状态：`A3_ONBOARDED_ASCEND_N1_N4_PASS_W8B_B2_SVE_VERIFIED_W8C_LMHEAD_556_VERIFIED_C5_B3C_VERIFICATION_IN_FLIGHT_C6_IN_FLIGHT_N5_PAUSED_BY_USER_NPU`

本快照覆盖：用户借入的 A3 共享机（鲲鹏 920B CPU + 昇腾 910C）接入与纪律；Ascend 线 N1–N4（自检 / PTO-ISA bring-up / vllm-ascend 基线 / **IR→PTO 最小桥**）；
W8B B2（SVE256 四类原生化）与 B3c（本机 L0/L1 + dispatch 分解）；W8C lm_head 双 packing（**556 全开**）与 C5（SVE W8A8）；
以及 U/P 两条新线的提案（**未立项**，待用户决定）。

> 上下文压缩交接：本文件 + `HANDOFF.zh-CN.md` §0 + `configs/development_lock.yaml` 的 `in_flight_2026_09_11_batch2` / `pending_user_decisions_2026_09_11` 足以零记忆恢复。

---

## 1. 冻结点

```text
upstream base        34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
integration base     8700f741629616bdc16848e49ffe7fecde6807d3（0037 收口）
integration head     bef73643b… → 195ace3eb8a521950c72ffc62f0349aed6c7ac04（0039 交接后 C6 合入）
patches              134（base 34475e0d8；本批新增提交尚未重新导出）
控制仓 main          8 个本地提交未推送（含 A3 登记 / N2–N4 验收 / E4 step2 暂停 / 本快照）
```

本批 23 提交（`8700f7416..bef73643b`）：

```text
26a444b86  perf(pypto-x): add native AVX2 packed cast/transpose/embedding kernels          (B1)
18c5abec7  feat(pypto-x): add native AVX2 W8A8 widening path                              (C3)
ac5a44d54  feat(pypto-x): add torch custom-op bridge for the AVX-512 linear kernel        (J1)
f1a606023  perf(pypto-x): add AVX-512 packed/blocked matmul                               (GEMM)
d5082bb1d  perf(pypto-x): stage transposed AVX-512 packed weight tiles
d5cdb8518  fix(pypto-x): evict packed-weight cache entries without dict.popitem(last=)
a3a50b937  feat(pypto-x): add native AVX-512 VNNI W8A8 widening path                      (C4)
fa8ab359d  fix(pypto-x): make the torch linear bridge fail open on real build failures    (J1 fix)
9ab96f52b  perf(cuda): add GEMM vs cuBLAS relative-baseline harness                       (B3b)
273461930  feat(pypto-x): execute the torch linear bridge on the packed plane             (J1.1)
4d6e80125  fix(pypto-x): count the per-dtype pack module apart from shape kernels
336cead94  feat(pypto-x): add the vLLM CPU general plugin for packed dense Linear          (J2)
691c593b5  feat(pypto-x): add W8A8-linear graph path and quantized weight ingestion       (W8A8 v4)
378b738a8  fix(perf): enforce protocol outlier gate and per-side warmup in G2 verdict
901040cec  fix(perf): require one batch size across aggregate rounds
227506c86  fix(perf): apply ERR-0004 outlier floor and close R2-D2 round validity         (ERR-0004)
17c5c5542  feat(pypto-x): ingest the opt-in tied lm_head dual packing (W8C 556 open)      (lm_head)
26b32f588  docs(portable): describe the W8A8 dual-packing layout contract
1f0882c07  docs(portable): account for the optional 321st dual-packing parameter
ac3410083  spike(e4): Core IR -> PTO C++ minimal bridge                                   (N4/E4 step1)
4da3ac151  feat(sve256): execute iota/compare and broadcast/where natively                (B2)
e8041fd97  feat(pypto-x): add native SVE256 W8A8 widening path                            (C5)
bef73643b  perf(cpu): add local L0 microbench and L1 op-dispatch decomposition            (B3c)
```

---

## 2. A3 共享机接入（用户借用）

```text
配置    鲲鹏 920B CPU（aarch64，SVE VL=32 原生，含 svebf16/svei8mm/svef32mm/svef64mm）
        + 昇腾 910C NPU（CANN 9.1.0 / driver 26.1.1）；640 核 / 2 TB / /home 13 TB
工作环境 用户授权自建容器（从 vllm-ascend deepseek-v4.1 镜像创建）；容器内 torch 2.10 /
        torch_npu 2.10.0.post4 / vllm 0.27.1 / vllm-ascend 0.1.dev5097 / python 3.12
纪律    CPU 固定最后一个 NUMA node；NPU 只用 chip7（其余留给用户量化任务）；
        只写 home 与自己容器；不改宿主配置、不碰他人容器/进程；共享机 → 安静使用；
        大文件不从 SSH 传（在 A3 侧下载/生成）；不设卡锁（用户明确）
已知问题 chip7 存在间歇性 aivec UB-OOB 故障窗（2026-09-11 20:55–21:21 CST 实测，
        ACL 507035，上游 canonical 用例同样中招）→ 上卡前先跑上游 TADD 冒烟
```

---

## 3. Ascend 线（N1–N5）

| 任务 | 状态 | 关键结论 |
|---|---|---|
| **N1** `a3-ascend-selfcheck` | ✅ PASS | torch/acl/aclnn/bisheng/mspti 真机验证；vllm-ascend 插件自动注册；`Ascend910_9382`（SOC `ascend910_9391`） |
| **N2** `a3-pto-isa-bringup` | ✅ PASS（**单用例限定**） | clone pin `668248ec`（与本机逐字节一致）；gtest 装 home（无 root）；`tassign/TASSIGNTest.case1` chip7 **exit 0 / 5/5 PASSED / 1.65 s**；tassign 仅 16 B 分配故 `npu-smi` 无 PID（用 512 MiB 正对照证明采样链路） |
| **N3** `a3-vllm-ascend-baseline` | ✅ PASS（profiling NOT_COVERED） | **Qwen3.5-0.8B 三 prompt 前 4 token 与冻结 gold 全一致**；两进程逐位确定；logprobs decode argmax 12/12；HBM 峰值 23.5–23.9 GB；TPOT default 3.6–4.0 ms |
| **N4** `ir-to-pto-minimal-bridge` | ✅ **验收 PASS_WITH_BOUNDARIES**（0 阻断） | **我们自己的 Core IR 首次自动生成 PTO C++ 并在真机逐位一致**：3 个 64×64 f32 程序（copy/fill/add）→ bisheng → `pto_vec_st` → chip7 6/6 次 3/3 PASS；与 numpy **逐位**（0/4096）；验收方自造 31 例 fail-closed 27 拒 0 hole |
| **N5** `ir-to-pto-minimal-bridge-2` | ⏸ **PAUSED_BY_USER_NPU_NEEDED** | 目标：external buffer→GlobalTensor/TASSIGN ABI + op 扩展（`trowsum`/`tmatmul`）；**0 次上卡**；设计草案在 brief §5；A3 残留 `~/pypto-x-a3/e4s2/`（容器内 `/root`，恢复时改用 `/home` 路径并修 `CMAKE_PREFIX_PATH`） |

**N4 的"最小桥 vs 真桥"差距清单**（E4 后续排期依据）：external buffer ABI 未做；op 集仅 3；仅 f32/rank-2/静态；tiling/事件链写死、单核；产物未接 pypto Runtime/LaunchRequest；无 golden 协议/多 kernel 图/错误注入；pin/SOC/编译器路径硬编码；未进 CI/契约。

---

## 4. W8B（CPU 硬化与性能）

### 4.1 B2 `sve256-fallback-closure`（✅ 验收 PASS，0 阻断）

- 四类**全部 native**：`iota`（ELF op27 `PXIO`）、`compare`（op28 `PXCP`）、`broadcast`（op16 `PXBD`）、`where`（op17 `PXWH`）；`host_preexpanded_operations=[]`
- 契约：lowering 6→7、runner v4→v5；旧工件 fail-closed
- **QEMU 与 A3 原生逐输出 SHA-256 50/50 一致**；验收方自造 40 例/50 输出（vs scalar 50/50、独立 Python oracle 47/47 双跑一致）；guard monkeypatch 0 trip
- **集成口径回归**：collect 1008→1027、**1020 passed / 7 skipped / 0 failed**（实现方报的 998→1017 是其分支口径）
- 仍降级（未承诺）：exp/sigmoid/silu/softplus、cast/bf16_math/bf16_where、bf16_reorder（ELF 标量 lane）；batched matmul host loop；AVX2/AVX-512 position_control host_reference

### 4.2 B3c `local-perf-l0-l1`（实现已合入；验收 in flight）

- **dispatch_seconds 占 wall ≈41%（T=5 prefill 19.6/48.6 s）与 ≈49%（T=1 decode 11.2/22.7 s）**；3 轮离散 1.7%/3.3%
- **归因实验**（`--no-op-recorder` 对照）证明：差值**不是** harness/进度日志，而是 **runtime 逐 op Python 派发循环本身**
- L0：AVX-512 固定 per-launch **1.22–1.32 ms**；pass2 15 case **0/15 过 cv 门**（KVM ~2 ms launch 调度尾部）→ 只作 T3、UNGATED
- **B1 GEMM 落地后热点换人**：`reshape 8.34 s / slice 7.37 s / transpose 7.27 s`（全 host_reference），matmul 仅 1.34 s
- 窗口非静默（有其他 agent 轻活，已标注）；与 B5/B3b 数字不可比（已逐条注明）

---

## 5. W8C（W8A8-linear）

### 5.1 lm_head 双 packing（✅ 验收 PASS_WITH_BOUNDARIES，0 阻断）

- **556 全开真权重路径可用**；region `368 / 518 / 520 / 554 / 556`（556 = schema 321 + scales 187 + states 48）
- 守恒式 `1,504,791,232 = 1,261,450,880 + 751,899,712 − 508,559,360`；覆盖率全开 **linear 1.0 / whole-net 0.9807973840843761**
- 真权重 en T=5：全开 vs 默认 cos **0.9972200432103332** / max_abs 1.671875 / argmax 5/5；token 链 `[11751,13,198,760,6511]`
- 验收方自造 16 例 fail-closed 全拒；量化器 4 种 chunk 全 bit-exact
- **诚实边界**：两配置均**超 L4 暂定阈值**（max_abs 2.69/2.76 > 0.5、cos 0.99604/0.99563 < 0.999），如实登记、留 C8

### 5.2 C5 `qwen35-w8a8-sve256`（实现已合入；验收 in flight）

- 4 opcode 全 native：`qmatmul_s8s8_s32`（**两级 `sunpklo` s8→s16→s32 + 整数 `mla`**）、`quantize_per_token_s8`、`dequantize_s8`、`dequantize_epilogue_bf16`
- **刻意未用 SDOT/I8MM**（D10 widening fallback 为契约默认；`sve2=false` 故 SDOT 本不可用），附 HWCAP + 反汇编双证据
- QEMU 与 A3 原生 **report_digest 完全相同**（21/21 逐位等价）；lowering 7→8、runner v5→v6
- **连带修复**既有缺陷：`_write_storage` 整数 raw-address 输出 `TypeError` + 整数 raw 输入 `_CTYPE` 缺键 → 新增 `_storage_scalar_value` + 3 个单测（建议关闭对应 known limit）
- 回归：base 1027 → head 1066 collect；1059 passed / 7 skipped

---

## 6. 在途与排队

```text
C5 验收      verify-qwen35-w8a8-sve256（agent 14ddcd05）
B3c 验收     verify-local-perf-l0-l1（agent f178cb9b）
C6 验收      verify-qwen35-w8a8-gpu-kernels（agent 85b8a8ec；实现已合入 195ace3eb）
N5 封存      ir-to-pto-minimal-bridge-2（等用户 NPU 用完）
（注：C6 实现已完成并合入——真机 sm_120 53/53 PASS，真机抓到 2 个 fake-driver 不可见的 PTX 地址 bug 并已修；
  其 AMD static 半**未做**；本批 integration HEAD 因此前移到 195ace3eb）
```

---

## 7. U/P 两条新线的提案（**未立项**，待用户决定）

用户提问"是否有上层用户接口线与用户侧性能控制线"，核查结论：

```text
用户接口      无统一 API/CLI/配置对象；仅有内部 driver 脚本 + Tensor 前端（开发者向）
              W8J 是"框架适配"方向的半条线（且因性能门禁默认关闭）
性能控制      无；现状是 35 个散落的 PYPTO_X_* 环境变量（路径/测试/性能/行为混杂），
              无分类、无默认值策略、无用户文档、无"性能/精度权衡"入口
```

建议切法（详见 lock 的 `up_lines_proposal_2026_09_11`）：
- **U 线**：U1 统一 API（load/generate）｜U2 配置对象 + 文档（收敛 35 个 env）｜U3 CLI（可选）｜U4 结构化错误 + `doctor()`
- **P 线**：P1 旋钮矩阵（线程/缓存/后端/dispatch，含默认值与生效验证）｜P2 用户可读性能报告（dispatch vs op）｜P3 性能/精度权衡入口（精度 ↔ 误差带，fail-closed）｜P4 硬优化（reshape/slice/transpose 原生化 + dispatch 削减）
- 注意：P4 = W8B 续作；W8J = U 线在框架方向的分支

---

## 8. 待用户决策（4 项）

1. **B4 性能阈值冻结**（B3b 的 4 个 `candidate_ratio` 已就绪；B3c 完成后更完整）——需用户批准才写 `configs/perf_lock.yaml`
2. **C8 的 L4 阈值口径**：接受当前 W8A8 误差预算，还是先做校准/平滑——**决定 C7/C8 能否启动**
3. **内核性能投入**：W8J 注入当前 4.71–14.47× 慢于 oneDNN；投性能专线 / 接受更慢自主栈 / 换场景
4. **U/P 两条线是否立项**（含范围与优先级）

---

## 9. 证据路径

```text
A3 自检            _meta/pypto-x/a3-ascend-selfcheck/
PTO-ISA bring-up   _meta/pypto-x/a3-pto-isa-bringup/
vllm-ascend 基线   _meta/pypto-x/a3-vllm-ascend-baseline/
IR→PTO 最小桥      _meta/pypto-x/ir-to-pto-minimal-bridge/（+ verify-ir-to-pto-minimal-bridge/）
IR→PTO step2(暂停) _meta/pypto-x/ir-to-pto-minimal-bridge-2/
B2 SVE 四类        _meta/pypto-x/sve256-fallback-closure/（+ verify-sve256-fallback-closure/）
B3c 本机性能       _meta/pypto-x/local-perf-l0-l1/（+ verify-local-perf-l0-l1/）
W8A8 lm_head 556   _meta/pypto-x/w8a8-lm-head-ingestion/（+ verify-w8a8-lm-head-ingestion/）
C5 SVE W8A8        _meta/pypto-x/qwen35-w8a8-sve256/（+ verify-qwen35-w8a8-sve256/）
```
