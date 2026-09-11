# PyPTO-X 0040 波次快照：C8 参照金标与 L4、vendor GEMM 选型、launch 残差归因、U 线首切片与集成体检

归档序号：`0040`

归档日期：2026-09-12（Asia/Shanghai；0039 交接后至本批收口）

状态：`DRAFT_PENDING_FREEZE`（收口时把"冻结点"一节补成最终值；正文事实均已在 `configs/development_lock.yaml` 与各任务证据目录中登记）

本快照覆盖：C8 的参照 gold/band 与 L4 判定（Q1–Q4；Q5 被 A3 断线阻塞）｜vendor GEMM 两轮选型（Q8 x86/AOCL、Q8b aarch64/NVIDIA）｜B3c 判决与 N4.0 的残差再归因（ERR-0006）+ launch-artifact-cache｜C6 CUDA W8A8 计时证据与 quantize 核重写｜B6 AVX-512 layout 原生化｜KF-1 导入事故的修复与守卫｜U 线立项（0008）与 U1 首切片验收（FAIL，3 条阻断回修中）｜Q10 SVE256 int8 layout 与 SVE256 活性回收｜integration 全量体检（HEALTHY）与 D-1/D-2 修复｜A3 断线（仍未恢复）。

> 上下文压缩交接：本文件 + `HANDOFF.zh-CN.md` §0 + `configs/development_lock.yaml` 的 `waves.W8.in_flight_2026_09_11_batch2` / `pending_user_decisions_2026_09_11` 足以零记忆恢复。

---

## 1. 冻结点（**收口时填写**）

```text
upstream base        34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
integration base     195ace3eb8a521950c72ffc62f0349aed6c7ac04（0039 交接点）
integration head     TBD_AT_FREEZE（本批已知节点：健康检查受检 7e2aea514 → D-1/D-2 修复 77981213e → U1 回修 8360a9aee）
patches              TBD_AT_FREEZE（0039 收口后为 144）
控制仓 main          TBD_AT_FREEZE（本批本地提交，未推送）
```

本批新增/合入的提交以 `development_lock.yaml` 的 `integration_commits` 为准；父 agent 已用
`git merge-base --is-ancestor` 对全部 22 个登记 SHA（外加一个 8 提交区间）做过**祖先审计**（`integration_history_audit_2026_09_12`），结论 0 缺失。

---

## 2. C8：参照 gold/band 与 L4（Q1–Q4 完成，Q5 阻塞）

- **参照 gold/band**（`c8-w8a8-reference-gold`）：三种 prompt 的 W8A8 参照金标与 gate band 冻结；逐位校验 150/150 张量、399,360 元素 ULP=0。
- **轴 A 主门 L4**（`c8-l4-avx512`，AVX-512）：
  - prefill 口径 **6/6 PASS**（chat 20/22 + 2 个 near-tie flip 单列）；
  - decode 口径 **6/6 FAIL（仅 cosine 门）**；C7 三成分分解：真实小缺口（范数比 1.09–1.29）＋ cosine 门与 max_abs 门**差 50–170×** ＋参照自身 1–2% 不确定；
  - 同一批输出用 prefill band 判 decode → 5/6 PASS；用 declared decode key 判 → 6/6 FAIL。**口径待用户裁决（L4）**。
- **Q5（SVE256 整网 L4）**：被 SVE256 int8 layout 缺口阻塞 → 由 Q10 解封；**A3 断线中**，恢复后与 Q9 的 A3 leg、liveness 的 A3 确认**合并成一趟**做（HANDOFF §4.9）。

## 3. vendor GEMM 选型（Q8/Q8b 完成，"采纳 vendor"已由用户定）

- **x86（AOCL/LPGEMM，源码构建显式 zen4）**：`libblis-mt.so.5.3.2` sha256 `c7d74a31…`；位级包络 K≤1024 精确、K=3584 差 4、K=133145 回绕、`-128` 静默；`int32_f32block_bounded` 作为 opt-in 提速档；十二形状 geomean 相对 oneDNN：AOCL int8 0.57/0.61、bf16 0.76，我方 int8 qmatmul 24.5、bf16 6.72。
- **aarch64**：primary=KleidiAI（int32 精确、需私有 packed-LHS；`params.lhs_zero_point=1` 是 flag 陷阱）；oneDNN 3.12 aarch64 不支持 per-token（status=3）；vLLM 走 oneDNN+ACL，要求 `[K,N]` 且 `stride(0)==1`。
- **NVIDIA sm_120**：融合 int8 不可用，但 `cublasGemmEx int8→s32` 可用且与 C6 int32 逐位一致（12/12）。
- **落地顺序（父 agent 2026-09-12 决定）**：vendor provider（x86 AOCL=U2、CUDA 两趟=cublasGemmEx）**必须在 W8A8 v2（M2）契约变更之后**做——v2 会移动四个后端（含 CUDA）的权重 region layout，在 v1 上实现等于返工。

## 4. N4.0：残差再归因 + launch-artifact-cache（ERR-0006）

- B3c 的"逐 op 41–49% 派发开销"是**残差误读**：实为每 launch 的 artifact 校验/重复 lowering/ELF decode；loop 胶水只有 0.003 ms/op。
- 去重后 prefill 24.460 → 18.092 s；**launch-artifact-cache**（进程级 artifact→target_ir 缓存 + `_LaunchValidationMemo`，键含 content_digest/format/entrypoint/abi/sha256(target)/冻结 metadata，且要求 `entry.metadata is artifact.metadata`）再降到 **11.745 s（−38.2%）**、decode **8.078 s（−32.1%）**；同 artifact 重复 launch 0.63 s。
- 遗留（单列）：`CpuVectorProgram.plan_digest` 每次访问全量重序列化（T=5 0.381 s / T=1 0.245 s，每次 launch 访问 4 次），建议在 lowering 构造期冻结该字段——**未做**。
- 位级：6 轮（base×3 / after×3）51 个 prefill 输出与每 decode step 51 个输出 buffer 的 raw sha256 + artifact content digest 全同；fail-closed 8 类元数据篡改全拒。

## 5. C6：CUDA W8A8 计时证据与 quantize 重写

- **证据（全部 UNGATED，GPU 被机主 VR 占用）**：`quantize_per_token_s8` 占完整 W8A8 op 的 **84.7–92.2%**（每元素 `div.rn.f64`）；vendor 两趟只值 ~6%（两条路都要做 quantize）。
- **重写（`cuda-quantize-rewrite`，integration `7bd750d0c`）**：一行 128 线程 block-per-row + `max.f32`/`shfl` 归约 + 单次精确 FMA 残差校验复现 `cvt.rni.s32.f64`（无 f64 除法）→ 289–1034 µs 降到 **11.2–22.2 µs（中位 24×）**，位级 58/58、独立复核 121/121。
- **后果**：quantize 不再是瓶颈，`qmatmul`（25–87 µs）成为新瓶颈 → vendor 两趟 GEMM 重新有意义（见 §3 的顺序约束）。
- **D-1（本批体检发现，已修）**：`scripts/perf/c6_timing_harness.py` 被 cherry-pick 顺序回退，丢失 GPU preflight 与 `dispatch_overhead` 的 TimingProxy 修复 → 已在 tip `77981213e` fold 回并加 6 条静态守卫（修复前文件 4/6 红）。

## 6. B6：AVX-512 layout 原生化（已验收 PASS_WITH_BOUNDARIES）

- 原生 layout 描述符 PXLD-v2（payload 6→7、ABI :7→:8），f32/bf16/int8 七算子；85 新例、81 buffer sha256 逐位。
- UNGATED 收益：`host_reference_s` 22.75 → 0.004 s；per-op median reshape 10.51→0.022 / slice 10.22→0.027 / concat 21.98→0.048 / split 32.83→0.042 / transpose 8.08→2.42 ms/call。
- 已知边界：非 f32/bf16/int8 dtype 仍 host reference（显式 metadata 原因）；view/alias hard-off（liveness 未接入 executor）；rank2 (1,0) transpose 走既有 packed 平面、sNaN 行为与 host reference 不同（base/head native bytes 相同，非 B6 引入）。
- **AVX2 仍 host reference**（`layout_descriptor_wire: host-reference:v1`）→ 已派 `avx2-layout-native`（本批外，见 §9）。

## 7. U 线：立项、0006 设计收敛与 U1 首切片

- **0005/0006/0008**：用户侧算子接口方案落档（`docs/10-architecture/0005`、`0006`，含 §13 算子族/契约字段 v2 与外部审理回执 §13.9/§14.3）；U 线立项，U1 = `matmul` 垂直切片（policy → resolver → plan → report，portable provider）。
- **U1 实现**（integration `c19bce342`，43 测试）：位级 6/6 与 portable trunk 一致；plan digest 可复现（进程内/多变 hash seed/往返）；报告 98/98 必需路径删除被拒；collect +43/−0。
- **U1 独立验收判 FAIL**（3 条阻断，全部落在 U1 自己声明的验收标准内）：
  1. **B1 target fail-open**：非宿主 triple（aarch64/cuda/riscv）被伪造成 CPU capability 后照跑，报告写"声明 A、执行 B"；
  2. **B2 `policy_from_dict` 吞 `schema_version`**：`999 / 1.5 / "1" / 缺失` 全被当 v1 接受；
  3. **B3 `output` 别名输入**：`output=left` 静默覆盖输入数据且 plan 无 alias 记录。
  - 复验入口：`worktrees/_meta/pypto-x/verify-u1-matmul-policy-slice/work/repro_blockers.py`（修复后应 rc=0）。
  - 回修中；复验通过后再合入 integration 并跑规则 8 全量。
- **U2（第一个 vendor provider）**：依赖 U1 复验通过 + 顺序约束（§3）。

## 8. KF-1、Q10、liveness 与 P 线

- **KF-1（导入事故）**：C8/Q2 带入的相对导入回退使一条 portability override 测试失败（当时无人跑到）→ 修复 `e198817c` + tools 目录 4 处同类潜伏（`b2e466557`）+ 守卫测试（9 例）。
- **Q10（SVE256 int8 layout 原生化）**：integration `e5cb96719`；版本链 lowering 9 / payload 2 / runner v7 / layout wire v4+v5 / `LAYOUT_NATIVE_KERNEL_VERSION=1` / int8 dtype code 2；本地 99 新例、7 QEMU 套件 289 passed；A3 原生 43/43（断线前）、52 wire 0 mismatch、6144×1024→1024×6144 int8 transpose sha256 与 numpy 一致（5.849 s）；独立验收在跑。
- **SVE256 活性回收**：integration `d223ad3a0..7e2aea514`；本地三套件 127/91/67 全绿、on/off 两次 launch 逐位一致；A3 断线前观测稳态 RSS 2,335 MiB（峰值 25,170 MiB 定位在 LM-head transpose 的 Python list 边界 op 6723），**A3 侧确认待恢复**。
- **P 线与 W8A8 v2**：M1 设计完成（`docs/20-planning/0010`），契约 v2 = region `[in,out]` + `PACKED_LAYOUT_VERSION` 2→3 + `BINDING_SCHEMA_VERSION` 2→3，四后端 kernel 零改动；重基线 4–7 h；**M4 必须等本批收口**。

## 9. 集成体检与 D-1/D-2

- **HEALTHY**（受检 `7e2aea514`，经 `run_local_heavy.sh` local 锁）：**1394 collected / 1387 passed / 7 skipped / 0 failed / 0 error**，rc=0，758.99 s；收集数三方 diff **+190/−3** 逐项溯源（新增全部有文件级来源；消失 3 个是 Q10 版本改名）；gold 抽检 16/16 sha256 + 聚合/index digest MATCH；证据 `worktrees/_meta/pypto-x/integration-health-check`。
- **D-1**（C6 harness cherry-pick 顺序回退）与 **D-2**（`tools/c8_sve256_a3/pack_ours_logits.py:66` 未定义名）：已修复、已合入 `77981213e`（+ 6 条静态 AST 守卫），独立验收在跑；**规则 8 全量需在新 tip 重跑**。
- 新增流程（HANDOFF §4.9 第五条）：同一文件被"功能 + 修复"提交各自整份改写时必须按分支顺序 pick，且合入后对每个受影响文件做 `git diff <最新源提交> <tip> -- <file>` 差分复核——**全绿不等于内容对**。

## 10. 在途与排队（收口时按 lock 更新）

`verify-c6-harness-preflight-fold`（D-1/D-2 验收 + 新 tip 全量）｜`verify-sve256-int8-layout`（Q10）｜`op-bench-framework`（Q9）｜`c7-decode-gap-localization`（oracle swap）｜`u1-matmul-policy-slice` 回修 + 复验｜`avx2-layout-native`（本批外，AVX2 原生 layout）｜A3 相关 leg 全部 `PENDING_A3_OUTAGE`。

## 11. 待用户决策（7 项，见 `pending_user_decisions_2026_09_11`）

L4 是否含 decode 口径（**最靠前**）｜B4 性能阈值冻结（已推迟到 op-bench + 真实工业框架对比后）｜L5/L6 语料与阈值（决策包在 `0005 §10`，三句话）｜`view_mode=require` 提级｜W8A8 v1 之外的新量化方案｜W8J 注入门政策（硬门 vs 显式 opt-in）｜A3 chip7/NPU 放行（E 线 N5）。

## 12. 证据路径

```text
C8 参照 gold/band        ../worktrees/_meta/pypto-x/c8-w8a8-reference-gold
C8 L4 AVX-512            ../worktrees/_meta/pypto-x/c8-l4-avx512
C7 缺口定位              ../worktrees/_meta/pypto-x/c7-decode-gap-localization
vendor 选型 x86          ../worktrees/_meta/pypto-x/vendor-gemm-survey
vendor 选型 aarch64/NV   ../worktrees/_meta/pypto-x/vendor-gemm-survey-2
B6 layout 原生化         ../worktrees/_meta/pypto-x/b6-layout-native（验收 verify-b6-layout-native）
C6 计时                  ../worktrees/_meta/pypto-x/c6-timing-evidence
CUDA quantize 重写       ../worktrees/_meta/pypto-x/cuda-quantize-rewrite
launch-artifact-cache    ../worktrees/_meta/pypto-x/launch-artifact-cache
U1 首切片                ../worktrees/_meta/pypto-x/u1-matmul-policy-slice（验收 verify-u1-matmul-policy-slice）
Q10 int8 layout          ../worktrees/_meta/pypto-x/sve256-int8-layout（验收 verify-sve256-int8-layout）
SVE256 活性              ../worktrees/_meta/pypto-x/sve256-liveness-release
集成体检                 ../worktrees/_meta/pypto-x/integration-health-check
D-1/D-2 修复验收         ../worktrees/_meta/pypto-x/verify-c6-harness-preflight-fold
```
