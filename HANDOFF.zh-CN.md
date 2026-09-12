# PyPTO-X 接手文档

状态：`W8A_BF16_ALIGNED_EN_ZH_CHAT_T18_NEAR_TIE_PASS_W8H_W8I_VERIFIED_W8B_B1_B2_B3A_B3B_B3C_B5_B6_VERIFIED_W8C_C1_C2_C3_C4_C5_C6_VERIFIED_W8C_W8A8_GRAPH_PATH_VERIFIED_W8J_J1_J2_VERIFIED_ERR_0003_0004_0005_0006_0007_0008_KF_0001_CLOSED_A3_ONBOARDED_N1_N5_PAUSED_VENDOR_GEMM_ADOPTED_Q1_Q2_Q3_Q4_Q5_Q6_Q7_Q8_Q8B_Q9_Q10_N4_C6_C7_DONE_W8A8_V2_M2_DONE_VIEW_MODE_REQUIRE_STABLE_U2A_VENDOR_PROVIDER_DONE_AVX2_LAYOUT_DONE_SVE256_W8A8_THROUGHPUT_FIXED_0040_AND_0041_PUSHED_0042_IN_FLIGHT_REPORT_CROSS_BLOCK_15D0FE58D_1B_ECHO_RECONCILE_U2B_INT8_AVX2_BROADCAST_RUNNING_CUDA_TOOLCHAIN_BLOCKED`

最后更新：2026-09-12 16:45 CST（Asia/Shanghai；**0042 进行中**：report-only cross-block 已并入 integration（cherry-pick `748717529`→**`15d0fe58d`**，树一致；聚焦 352 passed、验收方 V2–V5 58 passed、collect 1681），但独立验收抓到**同类剩余缺口**——`request.*` 及其在 `resolution.rules[*].details`/`coverage.*` 的回显，与 `capability`/`resolved.target`、`provider.select`/`filter`、`policy.applicability`、`layout.alias_proof` 的普通字段回显，均可单点篡改而被接受（echo 篡改 18/19 ACCEPTED，仅 `provider.select.primary` 被拒）→ **1b 对账修复在跑**；U2b（int8 vendor）与 AVX2 broadcast 原生平面并行在跑；本机无 GPU/nvcc，CUDA 线按既定顺序仍被 toolchain 阻塞）｜**0041 已交付并推送**：integration `8fff50558`、补丁 235、规则 8 全量 1667 passed/7 skipped/0 failed、五条任务级验收全 PASS_WITH_BOUNDARIES；`origin/main = 0d0ec76`；A3 已恢复；`view_mode=require` 提级生效）

> **零记忆恢复（上下文压缩后）**：按顺序读 本文件 §0 → `docs/00-handoffs/ERRATA.zh-CN.md` → `docs/00-handoffs/0039-2026-09-11-a3-onboarding-ascend-bridge-sve-w8a8.zh-CN.md` → `configs/development_lock.yaml` 的 `waves.W8.in_flight_2026_09_11_batch2` 与 `pending_user_decisions_2026_09_11`。

项目根目录：`/home/chiro/projects/pypto/pypto_x`

> 首次接手请按 `AGENTS.md` 的顺序读：本文件 → `docs/00-handoffs/ERRATA.zh-CN.md` → `docs/20-planning/0003-…roadmap.zh-CN.md` → 各规范。

## 0. 恢复第一小时（零记忆 TL;DR）

本仓的设计目标是**任意全新 agent 只靠仓内文档即可接手**；若你（或新上下文）什么都不知道，按此执行：

```text
1. 读：AGENTS.md → 本文件（§1–§2）→ ERRATA.zh-CN.md（含 ERR-0002）→ 路线图 §6/§10。
2. 跑 §12 自检。0036 冻结点为 integration @ dca302ef4、111 个补丁；**0037 波次后**为 integration @ 8700f7416、118 个补丁；
   两把锁 FREE、CANN toolkit 在 /usr/local/Ascend。
   自检不重跑 smoke、不下载依赖、不加载权重。
3. 不要做的事：不把 chat(T=18) 说成"真实数值累积分歧/失败"（ERR-0002 已定位为 driver 缺陷并修复；
   严格 argmax 17/18 + row 14 near-tie，按 2026-09-10 修订判据 PASS，但 17/18 必须同时可见）；
   不把 T=1/decode 写成"端到端无变化"（算子层 no-op，端到端 logits 因 state 继承而变）；
   不把静态 lowering 说成真机 PASS；不把 A2 的限定式关闭说成"PyPTO Ascend 后端已通"；
   不推公开仓/不改历史/不发权重证据，除非用户当次明确指示。
4. 用户说"继续"时的默认动作（按优先级）：
   a) **先读 lock 的 `waves.W8.batch_0041_accumulating_2026_09_12`（+ `in_flight_2026_09_11_batch2` 里仍标 in-flight 的项）**——它才是权威清单。
      当前（2026-09-12 13:20 CST / 05:20Z）：**0041 已交付并推送**（`origin/main = 0d0ec76`，integration tip `8fff50558`，
      235 补丁，规则 8 全量 1674 collected / 1667 passed / 7 skipped / 0 failed，五条任务级验收全 PASS_WITH_BOUNDARIES）；
      **0042 在跑**（integration tip `15d0fe58d`）：1b 回显对账修复（agent `ecf51a8a`）／U2b int8 vendor（agent `264646fb`）／
      AVX2 broadcast 原生平面（agent `c8a613dc`）／0042 report 独立验收（agent `cb32c1ba`）；**A3 空闲**（2026-09-11T23:51Z 起可用）；
      **本机无 GPU / nvcc**（lock: `cuda_c1_acceptance_smoke: BLOCKED_TOOLCHAIN_nvcc_absent`），CUDA 线按 `avx512_and_sve256_first_cuda_later` 仍被工具链阻塞。
   b) **0041 已交付的主要结论**（引用以 lock/ERRATA 为准）：
      · **AVX2 layout 原生化**（`1b1c40656`+`7d8762abe`）：decode launch 28.5→20.4 s、prefill 127.9→109.3 s；per-op layout 中位 0.02–0.04 ms；
        剩余 host_reference 仅 `broadcast`（~39 s/494 calls）与 `where/compare/iota`（<1 s）。
      · **W8A8 契约 v2（packed `[K,out]`）**：M2a `fca4a6d9b`（每 forward int8 权重 transpose 150/186/151/187→0、region 全表对拍、
        五执行面逐位一致、四后端 kernel 0 行改动；`BINDING_SCHEMA_VERSION`/`PACKED_LAYOUT_VERSION` 2→3、`WeightLayout.layout_version` 1→2、
        payload 不升）；M2b 重基线（AVX-512 六段 615 s / AVX2 六段 2810 s、prefill 4/4 PASS、AVX-512 v1↔v2 765 arrays 0 mismatch）。
      · **Q5 SVE256 整网 L4（A3 原生）**：6/6 配置 digest/op 数与静态 census 全对拍、每 forward int8 transpose=0、qmatmul 全 native、
        `emulated=false/vl_bytes=32`；**prefill 主判定 6/6 PASS**；两批总墙钟 2356 s（39.3 min）；decode 扩展与 x86 同模式
        （default 两口径 FAIL；full decode-band FAIL / prefill-band PASS）。诚实边界：default/en 与 full/zh 的 decode 链在 prefill-next-token
        行因 near-tie 转向（`common_prefix_match=False`，主判定仍 PASS）。
      · **SVE256 W8A8 吞吐修复**（`8dabdf9e6`/`3c4fae931`/`35100030f`）：根因是每节点 O(整图) 的两层开销（per-node re-decode +
        每 dispatch metadata 深 thaw 535,957 次），修为 identity-keyed 有界 memo + O(1) 声明门；A3 真机 W8A8 算子 13.59/6.75/13.51 s →
        **7.12/16.08/32.05 ms**（T=5），6 配置从外推 11.8 h → 39.3 min；边界：memo 命中不复查盘上 ELF 摘要（可选加固 stat/rehash ~1 ms）。
      · **U2a（第一个 vendor provider，AOCL bf16/f32）**（`3dfac2bbe`）：数值仅 (5,1024,32) 差 bf16 5.96e-8 / f32 8.20e-8（保持
        `deterministic_bounded`）；Q8 同协议 12/12 ≤2×（geomean 0.87–0.97）；plan digest 冻结 `thread_mode`；D1/F5 两条 U1 follow-up 已修。
      · **`view_mode=require` 提级**（`8a52bfa98..36fa3fd3f` + 修复 `b82cb2022`/`8fff50558`）：五条可机检 proof 条件 + 三态语义 +
        AVX-512 proof_gated 零拷贝（与 copy 逐位一致）+ SVE256/AVX2 显式 `alias_proof=unsupported`；**提级自 2026-09-12 生效**
        （0006 provisional 已移除）。独立验收曾抓 3 条阻断反例（plan/report 可自洽伪造），合并回修后全拒。
      · **forge 修复轮**：新增 `layout_verify.py`/`binding_verify.py` + 执行层 canonical 重算；**公开校验面**与**执行面**的边界已写进 0006
        （`ExecutionPlan.validate` 是 content-local 门、不是执行授权）。
   c) **待用户裁决（6 项，见 `pending_user_decisions_2026_09_11`）**：L4 decode 口径（**最靠前**，决策包 `docs/20-planning/0011`，
      推荐 A+B）｜B4 阈值冻结｜L5/L6 语料与阈值（`0005 §10` 三句话）｜W8A8 之外的新量化方案｜W8J 注入门政策｜A3 chip7/NPU 放行。
      （`view_mode=require` 已于 2026-09-12 裁决并生效。）
   d) **0042（进行中，2026-09-12T08:35Z；integration tip `15d0fe58d`）**：
      · **已并入 — report-only cross-block**：任务 commit `748717529` → cherry-pick **`15d0fe58d`**（树 `c54671ffd` 与任务分支完全一致）；
        聚焦 11 文件 **352 passed**（169.59 s）、验收方 V2–V5 **58 passed**（18.57 s）、collect **1681**（0041 tip 1674，+7）。
        新增 4 条 cross-block must-reject（capability/contract digest、provider_declared、fallback_used）+ 2 条 contract-derived
        （`guarantees.accumulation` 由注册 `OpDefinition`+`request.dtype` 重算、`exactness_basis` 由 provider family 重算）；
        仍属单文档边界的字段（`artifact.pack.included_in_timing`、`resolution.plan_digest`、不嵌 manifest 时的库指纹等）见 0006 §7.1。
      · **1b（在跑）— 回显对账**：独立验收 + 父方探针抓到**同类剩余缺口**：`request.{op,dtype,left/right/output_shape}` 与其在
        `resolution.rules[op_contract.lookup|verify].details`、`coverage.*.opcode` 的回显，以及 `capability`/`resolved.target`/`resolved_policy.target`/
        `provider.select|filter`/`policy.applicability`/`layout.alias_proof` 的普通字段回显，均可**单点篡改而被接受**（echo 篡改 18/19 ACCEPTED，
        仅 `provider.select.primary` 被拒）。父方探针：`_meta/pypto-x/view-mode-require/scripts/probe_request_block_hole.py` 与
        `probe_rule_echo_holes.py`（证据 `raw/probe_*.json`）；验收方穷举分类 `_meta/pypto-x/verify-0042-report-cross-block/raw/08_exhaustive_classes.log`。
        修复要求：按**规则名**（不是下标）逐条对账、缺失/重复回显 fail-closed、不可对账者登记为边界。agent `ecf51a8a`（分支 `work/view-mode-require`，已 rebase 到 `15d0fe58d`）。
      · **在跑 — U2b**：int8/W8A8 的 AOCL vendor 绑定（agent `264646fb`，分支 `work/u2b-vendor-int8`，已 rebase 到 `15d0fe58d`）；
        新契约 `qmatmul_s8s8_s32`（packed `[K,out]` 直供 LPGEMM；K≤1024 与契约逐位一致、大 K 仅 `deterministic_bounded`、
        K≥133145 回绝、`-128` 拒绝）。**交叉影响**：其 `accumulation` 表只有 `integer` 键，必须验证真报告能通过 1 的 cross-block 校验。
      · **在跑 — AVX2 broadcast 原生平面**（AVX 线）：agent `c8a613dc`，分支 `work/avx2-broadcast-native`（基于 `15d0fe58d`）；
        镜像 AVX-512 `ptx_avx512_broadcast_f32/bf16` + `broadcast_native_contract()` 先例，把 `broadcast` 移出 `host_preexpanded_operations`，
        payload 5→6 / marker :6→:7；目标消掉 prefill 剩余 ~39 s 的大头；perf 复用 `_meta/pypto-x/avx2-layout-native/scripts/l1_avx2_instrument.py` 同款 harness（UNGATED）。
      · **未动**：U3（doctor/explain/plan）、U4（35 env 迁移）、U5（graph/region，等 N4）、memo 命中 ELF 加固、C8 L5/L6（等用户三句话）。
      · **A3 空闲**（2026-09-11T23:51Z 起可用），新 A3 leg 直接派即可，不必再挂 PENDING。
5. 并发与资源：平台无 subagent 并发上限（旧文档的 ≤3 属自设假设，ERR-0005 已更正）；重任务一律经 scripts/resource/run_local_heavy.sh（返回 75/69 就等待重试，
   禁止绕过）；**等锁时挂后台或长 timeout，不要在前台循环空转**（会耗尽 agent 回合，S1 曾因此中断一次）；
   A2/920B/A3 都是共享资源，约定安静使用。
6. 重任务需要用户批准才启动（AGENTS.md「Subagent 协议」第一条）。
7. 提交纪律：公开主仓 = 本目录（origin）；改完实现 → cherry-pick 到 integration → 独立 verify → 更新
   development_lock/快照/ERRATA → scripts/remote/export_patches.sh → git push origin main → scripts/remote/sync_private_backup.sh。
   （0038 收口已 push 到 7a064ab；**0039 批次及之后已 push 到 7857ae6，补丁 144**；**0040 批次已推送**（`7857ae6 → 41982db`、补丁 217、私有备份已同步）；**0041 批次已本地收口**：integration `8fff50558`、补丁 235（am 复算 tree 一致）、五条任务级验收全 PASS_WITH_BOUNDARIES、规则 8 全量 1667 passed/7 skipped/0 failed，控制仓本地提交——**push 需用户当次明确批准**。）
8. **cherry-pick 后必须立刻跑一次全量 collect + pass/fail 对比**（KF-1 教训，2026-09-12）：任何改动合入 integration
   之后，即使看起来只碰工具目录或文档，也要跑一次全量 pytest 并对比收集数与失败集合；"改动小"不构成跳过理由。
   触发过的事故：C8/Q2 的 cherry-pick 引入 `tools/qwen35_reference/run_reference.py` 的导入回退，导致一条
   portability override 测试失败，而在 U1 之前**没有人跑到过它**（B6 head 当时 1200 passed / 0 failed）。
```

找不到答案时，先查 `_meta` 证据目录（`../worktrees/_meta/pypto-x/<task>/`）与 `docs/00-handoffs/` 的历史快照，
再问用户；**不要凭记忆编造数字**——所有阶段性数字都应以 `configs/development_lock.yaml` 与 0039 快照为准。

## 1. 接手摘要

**PyPTO-X（PyPTO Cross-Architecture）** 以官方 PyPTO 同仓的 Tensor/Professional 双前端为基线，把可移植语义与 Ascend 专属 dialect 分层，逐步支持：

- 鲲鹏/AArch64 CPU：scalar reference、SVE256（不开发 NEON 优化后端）；
- x86_64 CPU：scalar、AVX2、AVX-512（后续 AMX）；
- NVIDIA GPU：GPU 公共层 → CUDA/NVVM；
- AMD GPU：GPU 公共层 → HIP/ROCDL；
- Ascend CCE/CANN 保持为一个 target plugin，避免功能回退。

**当前进度一句话**：W1–W4（Core IR/ABI、CPU scalar/AVX2/AVX-512/SVE256、GPU common、CUDA C1–C2）与 AMD `gfx1036` 静态 C1–C3 已冻结；Qwen3.5-0.8B M0–M1K 已冻结；**W8A 真权重 BF16 整网对齐已收口**——en(T=5)/zh(T=8) strict PASS，chat(T=18) 严格 argmax 17/18、row 14 near-tie，按 2026-09-10 修订判据 PASS；期间修复 ERR-0001（decay 门缺 `exp(A_log)`）与 ERR-0002（driver cos/sin 布局缺陷，此前 chat"真实累积分歧"结论作废）；控制仓已公开化；**昇腾 A2(910B3) 真机验收 PASS**（限定式关闭 blocked），W8H/W8I vllm-ascend 基线**独立验收 PASS**；**0037 波次（2026-09-11）**：W8C C1+C2（W8A8 scheme/binding，R1–R5/R7–R9）独立验收 PASS，W8B B5（AVX-512 reduce 超线性修复 + broadcast 全 native）使 T=18 chat prefill 从 750.7s 降到 208.2s（整网位级一致），W8B B3a（CUDA cuEvent 计时）三轮验收后 PASS，并发现平台级 `wsl2_cuevent_rate_offset`（raw cuEvent 偏低 4.4–9.1%，run 内标定+corrected 字段已落地）。

## 2. 当前的量化状态（最重要）

```text
真权重 BF16、AVX-512、公开图 v3（graph_digest 66dd4077…，4,550 ops @ (1,1,4096)）
判据（2026-09-10 用户裁定 + 同日 near-tie 修订）：相对 gold 自身 fp32↔bf16 噪声底 band
  逐行 argmax 与 gold 一致（硬门槛）；near-tie 例外：gold 该行 top1/top2 margin ≤ 0.5×band
  且 ours argmax == gold top-2 时，记 near-tie flip（不判失败但必须单列计数）
  max|Δlogits| ≤ 2×band；cosine ≥ band_cosine − 1e-4
  band：en 0.235212 / zh 0.277682 / chat 0.611310

en(T=5)     PASS  argmax 5/5  cos 0.99997235  max_abs 0.159756（0.68×band）  逐层≥0.99993098
zh(T=8)     PASS  argmax 8/8  cos 0.99993803  max_abs 0.235296（0.85×band）  逐层≥0.99993980
chat(T=18)  PASS（修订判据）严格 argmax 17/18（唯一 mismatch row 14）cos 0.99989976
            max_abs 0.416867（0.68×band）逐层≥0.99988217
            → row 14 near-tie：gold fp32 top1 271=19.956333160 / top2 198=19.841325760，
              margin 0.115007=0.18813×band（阈值 0.5×band=0.305655），ours argmax=198=gold runner-up
              = 17/17 有效行 + 1 near-tie；**严格 17/18 必须保留可见**
三 prompt decode token 链与 gold 全一致；state 全有限，recurrent |state|max en 0.55–13.27 / zh 0.82–13.29 / chat 0.69–14.14
实测 ops(T) = 6,728 + 540×(T−5)（图契约未变，公式仍有效）
修复后复跑窗口：window1 958.8 s / recovery 768.7 s / window2 1499.1 s，合计锁内 ≈53.8 min；
旧 per-prompt wall（132/377/932 s）与峰值 RSS 产生于旧 driver，仅作量级参考，不作为修复后冻结数字
```

**结论口径**：
- ERR-0002（driver cos/sin 布局缺陷）修复后，chat(T=18) 不存在"真实数值累积分歧"；此前定位（row 11 起、
  首个不达标层 gold index 8 / layer_07）作废。严格口径 17/18 与修订后的 near-tie 例外必须同时可见。
- **T=1/decode**：算子层布局是 no-op（单 position cos/sin 位序列逐位相同），但端到端 decode logits 因继承
  prefill state 而变化；旧 decode cos/max_abs 数字作废（token 链仍全对）。**不得写"decode 端到端无变化"**。
- 引用 A2 真机结论必须带限定：仅 PTO-ISA tassign 单用例 + 注入式最小固定 64×64 f32 add hook 真实上卡；
  不代表 Core IR→PTO、classic/Pro JIT/OPC 或模型级。

## 3. 已确定的技术决策（新增/变更部分）

1. `pypto` Tensor 与 `pypto_pro` Professional 是同一 wheel 内的并存前端；Tensor frontend 作为跨架构 Core IR 主语义入口，Pro 保留为 Ascend expert dialect。
2. 目标无关 Core IR 只表达 Tensor/Scalar/Shape/View/控制流/逻辑 Tile 与副作用；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe、CCE 语义留在 Ascend target。
3. CPU 共享 lowering，优先级 AVX2 → AVX-512 → SVE256（无 NEON）；NVIDIA/AMD 共享 GPU IR/Runtime ABI，再分别下降 CUDA/NVVM 与 HIP/ROCDL。
4. 首个实际模型固定 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期只做纯文本 BF16 与 W8A8-linear；不做 W8A16/INT4/FP8/KV-GDR state 量化；视觉编码器不入 MVP。
5. **对齐判据用相对 dtype 带宽**（§2），不再用绝对 max_abs 阈值；`decode_logits[0] ≡ prefill_next_logits`，第 k 步对齐 `decode_logits[k+1]`。
6. **near-tie 例外**（2026-09-10 用户裁定）：逐行 argmax 仍为硬门槛；仅当 gold 该行 top1/top2 margin ≤ 0.5×band
   且 ours argmax == gold top-2 token 时，记 near-tie flip、不判失败但必须单列计数。严格口径结果必须同时可见。
   当前 chat 仅 row 14 适用（en/zh 无适用的 mismatch 行）。
7. **ERR-0002**：`qwen35_weighted_execution_driver.py` 的 cos/sin 布局缺陷已修复（head-major，integration `5f479d1a1`）；
   图契约零改动；旧 driver 产出的 prefill/decode logits 数字作废。`327b17158` 调试变体未合入、未采纳。
8. **仓库布局**：公开主仓 `chiro2001/pypto-x`（Apache-2.0 + NOTICE + `patches/`）是工作副本；私有归档 `chiro2001/pypto-x-private` + 本地备份 `../pypto_x_private_bkp`；**实现补丁以 `patches/` 形式发布**（111 个，base `34475e0d8`，HEAD `dca302ef4`）。
9. **发布纪律**：推公开仓属发布动作需用户明确指示；不得上传上游源码、模型权重、运行证据；访问脚本/端点只留本地私有侧。
10. AMD `gfx1036` 运行态判定**架构性不可达**，用户决定长期只保留静态证据。
11. Ascend 线：本机 CANN 9.2.0-beta.2 仅 toolkit；CA-model 可跑但极慢（95 s/小算子、7.3 GiB），只能做指令级细看；
    真机走 A2(910B3)：W8A-C 真机验收 PASS（限定式关闭 `ascend_cann_bisheng_npu_regression_blocked`，
    仅覆盖 tassign 单用例 + 注入式最小固定 64×64 f32 add hook）；**A2 NPU 任务必须走 `/root/a2-npu-lock/`**
    （只保护 NPU 执行，下载权重/编译/环境准备可并行；request→using→done + 历史保留；180 s 轮询；
    TTL 6 h + PID 僵死判定；A2 时钟约快 8 h）。

## 4. 目录与 Git 状态

```text
控制仓/公开主仓   /home/chiro/projects/pypto/pypto_x   origin = chiro2001/pypto-x（公开）
私有备份          ../pypto_x_private_bkp（archive 分支含第三方离线副本；main 跟随公开）
现有 HEAD         origin/main = **7857ae6**（0040 已 push：41 个提交）；本地 main 与 origin 同步
实现主仓          upstream/pypto @ 34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad（只读）
集成分支          port/pypto-x-integration @ **c2ec98f3c**（在 bef73643b 之上：C6 CUDA W8A8 → B6 AVX-512 layout 原生化 05a0b92a7 → C8/Q2 W8A8 参照 gold）
补丁集            patches/pypto-x/ **144 个**（base 34475e0d8；已随 7857ae6 推送）
任务 worktree     ../worktrees/pypto-x/<task>；证据 ../worktrees/_meta/pypto-x/<task>/
```

`upstream/*` 只读；`upstream/PTOAS/.codex/CLAUDE.md` 的 dirty 来自上游 CRLF/`.gitattributes` 不一致，不是人工改动。

## 5. 已完成的规划与规范交付

`README.md`、`AGENTS.md`、`docs/00-handoffs/`（含 `ERRATA.zh-CN.md`）、`docs/10-architecture/0001–0004`、
`docs/20-planning/0001（MVP）/0002（W8A8 契约，D1–D12 已批准）/0003（W8 并行路线图 v2，含外部评审修订）`、
`docs/WORKTREE_AGENT_PLAN`、`docs/PROJECT_LAYOUT`、`docs/RESOURCE_MATRIX`、`docs/LOCAL_RESOURCE_POLICY`、
`docs/SMOKE_TEST_SPEC`、`docs/PERF_MEASUREMENT_PROTOCOL`（提案）、`configs/*.yaml`、`patches/README.md`、
审计 `research/audits/2026/0001–0007`。

## 6. Subagent 与协作协议（要点）

- 每个 subagent：一 task、一 branch、一 worktree；启动字段必须含 `started_at`、绝对 worktree、branch、
  `smoke_once=true`、`wait_timeout_seconds=3600`、`poll=false`、`resource_lock_root=/home/chiro/projects/.resource-locks`、
  `local_heavy_policy=locked`、`local_heavy_runner=…/scripts/resource/run_local_heavy.sh`、
  `local_min_available_mib=8192`、`local_safety_floor_mib=4096`、`local_max_cpus=6`。
- **并发**：平台无硬性上限（ERR-0005）；按资源锁与用途铺并行度，同一把锁的等待者不超过 2 个。
- 阶段验收必须独立：从待验收 integration HEAD 建 `verify/<phase>` worktree，源码只读，只写独占 `_meta`。
- 父 agent 不自己跑重任务；同一任务禁止同时提交两个重命令；不在锁内跑无关命令。
- 重任务经 `run_local_heavy.sh`（75/69 等待重试，禁止绕过）；大内存任务显式登记 `memory_max_mib`。
- 冻结流水线（每个阶段收口）：实现 commit → 合并 integration → 独立验收 → 更新 `development_lock` 与快照/勘误 →
  重跑 `scripts/remote/export_patches.sh` → push 公开仓 + `sync_private_backup.sh`。

## 7. 建议的下一步波次（W8 及以后）

```text
① 0039 批次（A3 接入 + Ascend N1–N4 + B2 SVE + lm_head 556）已收口；C5/B3c 验收与 C6 在途（详见 0039 快照）
② **在途恢复**：C5 验收 / B3c 验收 / C6 实现（若上下文压缩期间中断，按 lock 的 `in_flight_2026_09_11_batch2` 重新派发或验收）；
   **N5 等用户 NPU 用完再恢复**（先跑上游 TADD 冒烟；A3 残留目录与路径坑见 0039 快照 §3）
③ 发布动作（**需用户批准**）：控制仓 main 8 个 0039 提交未 push；收口需先重跑 `export_patches.sh`（134→?）；
   push 后跑 `scripts/remote/sync_private_backup.sh`
④ **性能续作（B3c 已给出靶点）**：本机热点已是 **reshape 8.34 s / slice 7.37 s / transpose 7.27 s（全 host_reference）**，
   外加 **"残差"占 wall 41–49%**（**注意：ERR-0006 已更正——该残差不是逐 op 派发，而是每 launch 的 artifact 校验/重复 lowering/ELF decode**，逐 op 胶水仅 0.003 ms/op）；这两项直接决定 T=5/T=18 prefill 与 decode 的墙钟
⑤ W8C 续作：C6（CUDA/AMD 静态 W8A8）、C7 layer ladder（1→6→24）、C8 正式门禁（R12–R14）——
   **C8 之前需用户裁定 L4 阈值口径**（现 W8A8 scheme 超出暂定阈值：max_abs 2.69/2.76 vs ≤0.5）
⑥ B4 阈值冻结：B3b 的 4 个 candidate_ratio 已就绪、B3c 已给出 dispatch 分解——**需用户批准**才写 `configs/perf_lock.yaml`
⑦ W8J 续作：J3 需先解决内核性能（4.71–14.47× 慢于 oneDNN 6T，注入默认关闭）
⑧ **U/P 两条新线待立项**（用户接口 / 用户侧性能控制；提案见 0039 快照 §7 与 lock 的 `up_lines_proposal_2026_09_11`）
⑨ 平台注记：GamePC WSL2 的 cuEvent rate offset（wsl2_cuevent_rate_offset）与 ERR-0004 的
   outlier floor 规则必须遵守；CUDA 计时必须 raw + corrected + platform_flags 同报
⑩ 可选：若用户要求 chat 严格 18/18，评估任务分支 327b17158（--debug-f32-residual，未合入、未采纳）
```

## 8. 可用资源实况

```text
本机               12 vCPU（KVM guest，无 cpufreq）/ 29 GiB 内存 / 磁盘余 ~78 GB；CANN 9.2.0-beta.2 toolkit 已装
                   重任务必须经 run_local_heavy.sh 取得全局 local 锁（默认 MemoryMax=min(启动 MemAvailable−4096, 20480) MiB）
GamePC 192.168.101.5  WSL2 24 线程 / 30 GiB（宿主 61.4 GiB）/ RTX 5080 16 GB；CUDA Toolkit（nvcc 13.3.73+cuBLAS 13.6）
                    GPU-only 短探测不申请 gamepc；host-heavy 才持锁；Linux 命令经 wsl.exe -e bash -lc
鲲鹏 920B ECS       2 vCPU / 2.5 GiB / SVE=1 VL=32、SVE2=0；约定串行；QEMU 只作功能验证
                    **2026-09-11 已按用户指示释放**（实例+系统盘+EIP 删除，计费停止）；释放前 /root 工作目录
                    已归档 ../worktrees/_meta/pypto-x/ecs920b-release-archive-20260911/（758 文件 / 9.8 MB，条目核对一致）；
                    需用时以 ~/tools/ecs-920B/create.sh + setup-access.sh 重建
A3（用户借用共享机） 鲲鹏 920B CPU（aarch64，**SVE VL=32 原生**，含 svebf16/svei8mm）+ 昇腾 910C NPU（CANN 9.1.0；
                    容器内 torch 2.10 / torch_npu 2.10 / vllm 0.27.1）；640 核 / 2 TB / /home 13 TB
                    **2026-09-11 借入，用于替代已释放的 920B ECS（SVE 原生腿）并承接 Ascend 线**
                    纪律：CPU 固定最后一个 NUMA node；**NPU 只用 chip7**（其余留给用户量化任务）；
                    只写 home 与自己创建的容器；不改宿主配置、不碰他人容器/进程；共享机 → 安静使用（少量核、短任务）
                    访问方式与端点在本地私有侧，不进仓；**这是共享机，不设卡锁（用户明确指示）**
昇腾 A2(910B3)      容器 256 vCPU / 2 TB / 1×910B3（64 GB HBM）/ CANN 9.0.0 / driver 25.2.0
                    **2026-09-11 起暂停（用户指示）**：机时预算 100 h、已用 ~6 h，暂停不消耗机时且文件保留；
                    待用户通知恢复 → **恢复前 A2 任务一律不发射**（E4/E5、W8H/W8I 后续范围、stable CANN 9.2.0-beta.2
                    上卡、classic/Pro JIT/OPC/gym 验证；另 A2 上 ~8.03 GiB profiler trace 待清理）；
                    暂停时无在途任务、卡锁已释放（.using→.done）、无残留进程
                    W8A-C 真机验收 PASS；W8H/W8I vllm-ascend 基线**独立验收 PASS**
                    **卡占用协议**：容器固定目录 /root/a2-npu-lock/（README.zh-CN.md + a2_card_lock.sh）；
                    流程 request→using→done（历史保留），180 s 轮询，TTL 6 h + PID 僵死判定；
                    **只保护 NPU 执行**（下载权重/编译/环境准备可并行）；A2 时钟约快 8 h；
                    卡锁协议已由独立验收实际使用验证（.using→.done、无残留进程、HBM 回落 3441/65536 MiB）
                    访问脚本与端点只在本地私有侧（~/tools/a2-910b/ + ~/.ssh/a2-910b.env），不进仓；约定串行
                    出网仅 HTTPS(gitcode/pypi)；无 22 出网、无 TUN/NET_ADMIN（VPN 不可行）
资源锁             local / gamepc 当前均 FREE；锁协议见 /home/chiro/projects/.resource-locks/README.md；
                   A2 卡锁仅在 A2 容器内、与 local/gamepc 锁相互独立
```

## 9. 已验证结果（择要，完整见 `configs/development_lock.yaml`）

- W1–W4：Core IR、Target ABI、CPU scalar/AVX2/AVX-512/SVE256、GPU common、CUDA C1–C2 全部独立验收通过；
- AMD `gfx1036` 静态 C1–C3：math/reduction/rank-2..4 matmul，Qwen 静态 4,532/4,532（graph v2；v3 为 4,550，**AMD 尚未对 v3 重跑**）；
- Qwen M0–M1K：24 层无权重 decoder、3/320/48 binding、CPU/CUDA external ingestion；
- **W8A 真权重对齐**：en/zh strict PASS；chat 严格 argmax 17/18 + row 14 near-tie，按修订判据 PASS（§2）；
  decode token 链全对；state 有限；
- ERR-0001 decay 修复 + 独立验收 PASS_WITH_BOUNDARIES（liveness A/B 逐位一致；packed 内核逐位一致）；
- **ERR-0002 driver cos/sin 布局缺陷修复 + 独立验收 PASS**（`5f479d1a1`；图契约零改动；694 collected/687 passed/7 skipped）；
- **W8A-C Ascend A2(910B3) 真机验收 PASS + 独立验收 PASS**：PTO-ISA tassign NPU ST、CANN mspti aclnn Add 样例、
  注入式 hook 真实 bisheng 编译 22,728 B / sha256 4d7f704c… / live 9/9 / 真机 max_abs_err=0.0；
  限定式关闭 `ascend_cann_bisheng_npu_regression_blocked`，新增 3 条限定项；
- **W8G qwen35-portability-cleanup 完成 + 独立验收 PASS**（integration c464927fa）；
- **W8H vllm-ascend E2E 基线 + W8I profiling 基线：独立验收 PASS**（复跑/重算一致；非阻断差异已登记）；
  性能协议仍为提案，profiler 开销不得当模型性能引用；
- CPU vector runtime liveness + AVX-512 packed cast/transpose/embedding：真权重整网峰值从 309.6 GiB 投影降到 3.9 GiB 实测；
- GDR T=128：五后端 + 920B 原生 PASS，关闭 `gdr_t128_not_validated`；
- 基础设施：CUDA Toolkit、本机 CANN toolkit、PTO-ISA CPU_SIM 125/125、CANN CA-model 最小用例（审计 0006/0007）；
- **W8C C1+C2 独立验收 PASS**（R1–R5/R7–R9；QuantizedTensorDesc + 4 opcode + binding v2/layout v2/coverage；
  `w8a8_linear_coverage=0.9988…`、`whole_net_int8_compute_ratio=0.9770…`，static shape model）；
- **W8B B5 独立验收 PASS**：T=18 chat prefill 750.718s→208.174s（验收复跑 233.33s），reduce_sum 5963→2.45 ms/call、
  broadcast 962/962 native；位级一致（58/315 与 43/265 两套独立用例）；全量 753/746/7/0；
- **W8B B3a 独立验收 PASS（r3，0 discrepancy）**：raw cuEvent `kernel_seconds` + 平台速率标定 +
  `kernel_seconds_corrected` + `platform_flags`；`wsl2_cuevent_rate_offset` 已进 known limits。
- **W8B B1 独立验收 PASS_WITH_BOUNDARIES**：AVX2 packed cast/transpose/embedding 与 AVX-512 逐位一致
  （55 组/96,784 元素/0 mismatch）、fail-closed 15/15；全量 815/808/7/0。
- **W8B avx512-blocked-gemm 独立验收 PASS_WITH_BOUNDARIES**：prefill matmul 17.262→1.392s（12.4×）、
  batched 83.5×；24+83 形状与整网 380/380 输出**位级一致**；全量 892/885/7/0。
- **W8B B3b 独立验收 r3 PASS**：CUDA GEMM vs cuBLAS 的 4 个 candidate_ratio 就绪
  （16.626/31.376/16.819/31.505，两方独立复现偏差 ≤0.79%）；frozen_ratio 仍为 null（B4 待批准）。
- **W8C C3/C4 独立验收 PASS_WITH_BOUNDARIES**：AVX2 `vpmovsxbw+vpmaddwd+vpaddd`、AVX-512 `vpdpbusd`+符号补偿；
  全量 877/870/7/0 与 983/976/7/0。
- **W8C W8A8 图路径独立验收 PASS_WITH_BOUNDARIES**：真权重 W8A8 端到端（en T=5 prefill cos 0.99563、
  token 链与 BF16 全同）；region 518/554/520/**556**；覆盖率真实计数 0.6483（旧 static 0.9768 差异完全对账）；
  全量 991/7/0；**L4 暂定阈值被超出，已如实登记、R12 留 C8**。
- **W8J J1/J1.1/J2 验收**：torch 桥（r2 PASS，fail-open/自愈/单次拷贝）+ 桥切 packed 平面（32/32 逐位）
  + vLLM 插件（114/114 fail-open、token 与 logprobs 与 baseline 完全一致）；
  **性能门禁 0/12 → 注入默认关闭**（4.71–14.47× 慢于 oneDNN 6T，性能边界非机制缺陷）。
- **累积 BF16 回归零漂移**：4 次 artifact 版本升级后 en T=5 380/380 sha256 与冻结基线逐一相同。
- **ERR-0003**（契约 §3.1 555→556）与 **ERR-0004**（性能协议 MAD 门加 floor，用户批准）均已落地。
- **0039 批次（A3 接入 + Ascend）**：
  - **A3**：鲲鹏 920B（SVE VL=32 原生）+ 昇腾 910C（CANN 9.1.0）接入；自建容器、chip7 独占、node7 CPU；原生 SVE256 首光通过
  - **N1** Ascend 自检 PASS（torch/acl/aclnn/bisheng/mspti + vllm-ascend 插件）
  - **N2** PTO-ISA bring-up PASS（**单用例限定**）：`tassign` chip7 exit 0 / 5/5 PASSED
  - **N3** vllm-ascend 基线 PASS：**Qwen3.5-0.8B 三 prompt 前 4 token 与冻结 gold 全一致**（新机新版本基线，与 A2 不可比）
  - **N4** IR→PTO 最小桥 PASS_WITH_BOUNDARIES（0 阻断）：**Core IR → 自动生成 PTO C++ → chip7 真机逐位一致**（3 个 64×64 f32 程序）
  - **B2** SVE256 四类原生化验收 PASS：`iota/compare/broadcast/where` 全 native；**QEMU 与 A3 原生 50/50 逐输出 sha256 一致**
  - **W8A8 lm_head 556 全开**验收 PASS_WITH_BOUNDARIES：真权重可用；覆盖率 linear 1.0 / whole-net 0.9808；仍超 L4 暂定阈值（如实登记）
  - **B3c**：**残差占 wall 41–49%**（ERR-0006：实为每-launch setup，非逐 op 派发）；热点转为 reshape/slice/transpose（全 host_reference）
  - **C5** SVE W8A8：4 opcode 全 native（`sunpklo`×2 + 整数 `mla`，附 HWCAP/反汇编）；连带修复既有整数存储缺陷（验收在途）
  - **N5**（E4 step2）：已按用户要求暂停（NPU 被用户量化占用）

## 10. 许可证状态

- **本仓自有内容**（docs/configs/scripts/patches）：**Apache-2.0**（`LICENSE` + `NOTICE`）；`patches/` 是对上游代码的补丁文本，应用后衍生作品仍受上游许可约束；
- 上游 CANN Open Software License 2.0 仍限制在华为 AI 处理器/软件场景；**对外发布非华为处理器衍生后端前**需取得新许可证、双许可证或明确书面例外（维护方态度积极，但尚无正式文本）；
- 发布纪律见 §3.7。本文件不构成法律意见。

## 11. 新 Agent 首次回复前必须确认

1. 已读 `AGENTS.md`、本文件、`ERRATA.zh-CN.md`（含 ERR-0002）、W8 路线图；
2. 理解 §2 的判据口径：chat(T=18) 严格 argmax 17/18 + row 14 near-tie（修订判据 PASS），
   不得表述为"真实数值累积分歧"或"完全无保留的对齐"，17/18 必须同时可见；
3. 理解 ERR-0002：T=1 算子层 no-op，但端到端 decode 因 state 继承而变，不得写"decode 端到端无变化"；
4. 理解仓库布局（公开主仓为工作副本、私有备份、补丁 118 个、发布纪律）；
5. 理解并发上限与资源锁纪律，以及 A2 卡锁协议（/root/a2-npu-lock/，只保护 NPU 执行）；
6. 知道 A2(910B3) W8A-C 真机验收已 PASS（限定式），W8H/W8I 独立验收已 PASS；访问脚本在本地私有侧；
7. 未经用户明确指示不推送公开仓、不发布权重/证据、不改写公开历史。

## 12. 快速自检命令

```bash
cd /home/chiro/projects/pypto/pypto_x

git status --short && git log --oneline -1 && git remote -v | head -2
git -C upstream/pypto status --short && git -C upstream/pypto rev-parse HEAD
git -C ../worktrees/pypto-x/integration log --oneline -1
scripts/worktree/status.sh
bash -n scripts/worktree/*.sh scripts/smoke/*.sh scripts/resource/*.sh scripts/remote/*.sh
python3 -c "import yaml;[yaml.safe_load(open(p)) for p in ['configs/development_lock.yaml','configs/agent_tasks.yaml','configs/upstream_lock.yaml']];print('yaml ok')"
/home/chiro/projects/.resource-locks/resource-lock status     # 只观察；取得锁必须用 run
ls patches/pypto-x/*.patch | wc -l                            # 0038 冻结点为 134；0039 收口后重新导出再更新此数
```

除非用户明确要求，接手自检**不重跑** smoke、不下载依赖、不加载权重。
