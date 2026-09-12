# PyPTO-X 勘误索引

本文件登记**已冻结结论被后续发现推翻或修正**的条目。历史快照文件只追加勘误、不覆盖原结论；本文件是统一索引，各受影响证据目录内另有就地 `ERRATUM.zh-CN.md`。

## ERR-0001（2026-09-10）：Qwen3.5 GDR decay 门缺 `exp(A_log)`

**状态**：修复已合入 integration `9aae4649e`，并由独立验收 agent 复核为 **PASS_WITH_BOUNDARIES**（worktree `verify/w8a-decay-fix`，证据 `_meta/pypto-x/verify-w8a-decay-fix/`）。三个 prompt 的全覆盖（T=5/T=8/T=18）仍在进行中（任务 `qwen35-weighted-multiprompt`）。

### 验收对表述的修正（2026-09-10，独立验收发现）

本勘误初稿的若干表述不准确，已按独立验收结论更正：

```text
1. "exp 节点数 == 18" 不成立：全图 op name=='exp' 实为 114 个
   （本轮新增 18 个 decay 门 + 90 个 recurrent step decay + 6 个 softmax）；18 只指新增节点。
2. c3e3a8cb… 的归属：它是旧 WeightLayout.layout_digest，不是 PackedLayout.schema_digest；
   PackedLayout.schema_digest 恒等于 graph_digest（c1a5663b… → 66dd4077…）。
   新版 WeightLayout.layout_digest 尚未复算（由 qwen35-weighted-multiprompt 补齐）。
3. "371 参数"表述错误：371 是 binding schema 的 input 总数（320 参数 + 48 state + 3 runtime）；参数数是 320。
   字节账（1,504,786,048 B / 368 region / 1,574,877,952 B / align 64）本身正确。
4. "generate_logits 与 decode_logits 同索引、仅 EOS 被 mask"过度一般化：
   对 bfloat16 全 4 行、float32 第 1..3 行逐位成立；float32 第 0 行有 204,370 个非 EOS 位置不同（max|Δ| 8.1e-6）。
5. "native artifact digest 全变"不成立：v2 与 v3 的 .so 逐字节相同（95f80ffc…，29,184 B）；
   fail-closed 来自 load 期的 program_digest/plan_digest/target_digest/execution_modes 校验，而不是产物内容变化。
6. 容差口径：max_abs ≤ 0.30 只对 gold float32 成立；对 gold bfloat16 为 0.34375。
7. 本勘误初稿曾在本文件里提前写"已验收"，时序偏早；现已改为引用实际验收结论。
```

### 验收边界（仍未覆盖）

```text
- zh_continuation(T=8) / chat_zh_user(T=18) 的 prefill+decode 尚未跑（任务 qwen35-weighted-multiprompt）
- decode 第 4 步无 gold 参考行
- packed 内核只抽样复核（2 个 cast + 1 个非方阵 transpose），embedding 与复合形态未独立复现
- 真实权重 RSS 收益来自 liveness + packed 两项叠加，未单独归因
- AMD 静态 compiler 尚未对 v3 图（4,550 ops）重跑
- 无性能结论（本机 KVM guest 无 cpufreq，绝对门槛永久 UNGATED）
```

> 更新（2026-09-10，见 [ERR-0002](#err-0002)）：`zh_continuation`/`chat_zh_user` 已由
> `qwen35-weighted-multiprompt` 补齐，但其 prefill 数字受 driver 布局缺陷影响作废（替换值见 ERR-0002）；
> decode 第 4 步仍无 gold 参考行；AMD 静态 compiler 仍未对 v3 重跑。


**错在哪**：图 v1/v2 在 `python/pypto/portable/qwen35.py::_append_gdr_graph` 把 decay 门写成
`-A_log ⊙ softplus(a + dt_bias)`，官方实现（`transformers/models/qwen3_5/modeling_qwen3_5.py:619`，
配合 `:537` 的 `A_log = nn.Parameter(torch.log(A))`）是 `-exp(A_log) ⊙ softplus(...)`。

**后果**：decay = exp(gate) 对多数 head > 1，FP32 recurrent state 指数爆炸
（|state|max 0.60 → 2.8e4 → 2.36e16），随后被 gated RMSNorm 归一化掩盖，
表现为整网 logits 与官方 gold 完全对不上（cosine 0.777）。

**受影响（作废或需按新图重算）**：

```text
- 所有在"完整 decoder 图"上得到的数值结论（含首次真权重整网执行，图 v2 结果）
- graph contract version 2；graph_digest c1a5663b...；PackedLayout schema_digest c3e3a8cb...
- Core operation 计数 4,532（(B=1,T=1,past=4096)）与 6,710（(B=1,T=5,past=0)）
- 静态 AMD/Qwen 覆盖结论 "4,532/4,532 ops" 是针对旧图；新图为 4,550，静态 compiler 尚未对新图重跑
- M1J 冻结证据（3/320/48 与 layout digest）
```

**不受影响**：

```text
- M1G（functional / T=64 large shape / 920B native）：gate 为入参，decay=exp(gate) 语义本来就正确
- M1D 复合 builder、build_qwen35_gdr_recurrent_state、SVE256 GDR state 测试：同上
- 各后端 lowering（只实现 exp/softplus/neg/mul 原语，不含 decay 计算）
- 权重映射与字节账（371 参数 / 1,504,786,048 B / 368 region / 1,574,877,952 B 不变）
```

**修复后的真实基线**：

> **⚠ 勘误（2026-09-10，见 [ERR-0002](#err-0002)）**：本段数字产生于存在 cos/sin 布局缺陷的旧
> `qwen35_weighted_execution_driver.py`，**已整体作废**；替换数字（en/zh/chat 三 prompt）与新的
> near-tie 判据见下方 ERR-0002。图契约 v3 的 digest/ops 与 decode token 链仍有效。

```text
graph contract version   3
graph_digest(past=4096)  66dd4077...
Core operations          4,550（(1,1,4096)）/ 6,728（(1,5,0)）
prefill 对齐（真权重）    argmax 5/5、cosine 0.9999136、max_abs 0.2338（gold dtype 带 0.2352）  ← 作废（旧 driver）
decode 对齐              11751 → 13 → 198 → 760 逐步与 gold 一致
全量回归                 674 passed, 7 skipped
```

**已就地的勘误文件**：

```text
_meta/pypto-x/verify-qwen35-bf16-runtime-binding/ERRATUM.zh-CN.md
_meta/pypto-x/integration-w6-qwen35-bf16-runtime-binding-final-r2/ERRATUM.zh-CN.md
_meta/pypto-x/qwen35-bf16-weight-ingestion/ERRATUM.zh-CN.md
_meta/pypto-x/parent-real-profile-probe/ERRATUM.zh-CN.md
```

**历史快照的追加标注**：`docs/00-handoffs/0026/0030/0031/0032/0033/0034` 文末已各追加一行勘误指引。

**滚动文档与配置的就地更正**（2026-09-10，含独立验收指出的 8 处遗漏）：

```text
HANDOFF.zh-CN.md                  §9 前加统一勘误横幅（4,532/6,710 属 v2）
README.md / AGENTS.md             已标注 v2/v3 计数
docs/WORKTREE_AGENT_PLAN.zh-CN.md 已标注
docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md  示例与估算基数改为 v3（4550）
configs/perf_protocol.proposed.yaml      同上
docs/RESOURCE_MATRIX.zh-CN.md            AMD 小节加勘误横幅 + 表内计数标注
configs/development_lock.yaml            wave1 历史 digest 后追加 erratum 字段
docs/20-planning/0003-...roadmap         排期数字按 v3 口径
python/tests/ut/pypto_x/amd_gfx1036_c3_static_lower_driver.py   过时 docstring（由 A1 任务修）
python/tests/ut/pypto_x/qwen35_weighted_execution_evidence.py   过时 plans 数字（由 A1 任务修）
```

## ERR-0002（2026-09-10）：Qwen3.5 W8A prefill driver 的 cos/sin 运行时布局缺陷

**状态**：修复已合入 integration `5f479d1a1`（cherry-pick 自 `work/qwen35-t18-divergence-localization @ d05e25734`，base `9aae4649e`），
独立验收 **PASS**（worktree `verify/qwen35-t18-divergence-localization`，证据 `_meta/pypto-x/verify-qwen35-t18-divergence-localization/`）。
验收期间 integration 前进到 `dca302ef4`，`5f479d1a1` 仍是其 ancestor 且 driver blob（`d6a28f602…`）未变。

**错在哪**：`python/tests/ut/pypto_x/qwen35_weighted_execution_driver.py::rope_bits_for_positions`
把 cos/sin 运行时输入按 `[position][head][column]`（position-major）展平；但图契约
（`python/pypto/portable/qwen35.py:1560-1561`）的输入 shape 是 `(batch, heads, steps, rotary_dim)`，
图内 k 侧沿 `axis=1` 切 head（`:2190-2204`）、q 侧与该形状逐元素相乘，因此扁平 buffer 必须是
`[head][position][column]`（head-major）。驱动侧少了 head/position 维换序，且字节长度校验只看 nbytes，
缺陷长期未被发现。

**后果**：T≠heads（T=5/8/18，heads=8）时位置表整体错位——图内 `(head h, token t)` 实际读到第
`floor((T*h+t)/8)` 个位置行；T=18 时表现为位置表分块错位。**此前"chat(T=18) 真实数值累积分歧"
（3.78×band、row 11 起放大、首个不达标层 gold index 8）的结论作废**；根因是**测试驱动缺陷**，
不是模型/GDR/累加精度问题。

**修复**：`d05e25734` → integration `5f479d1a1`；driver-only + 新增聚焦回归测试
`python/tests/ut/pypto_x/test_qwen35_weighted_driver_rope_layout.py`（旧实现下 2 failed/1 passed，修复后 3 passed）；
`DRIVER_VERSION = 2`。**图契约零改动**：digest `(1,1,4096)=66dd4077…/4,550 ops`、
`(1,18,0)=001f32b7…/13,748 ops` 不变；无契约版本升级、无后端/权重改动。

### 判据修订：near-tie 例外（2026-09-10 用户裁定，条件已验证）

```text
逐行 argmax 与 gold 一致为硬门槛；唯一例外：
  若 gold 该行 top-1/top-2 margin ≤ 0.5×band 且 ours argmax == gold top-2 token，
  则该行记为 near-tie flip，不判失败，但必须单列计数（严格口径结果必须同时可见）。
实测 chat row 14（0-based）满足：
  gold fp32 top1 271 = 19.956333160 / top2 198 = 19.841325760，margin 0.115007 = 0.18813×band；
  阈值 0.5×band = 0.305655；ours argmax = 198 = gold runner-up（ours top1/top2 为 19.875 并列）
  → 采纳后 chat = 17/17 有效行 + 1 near-tie = PASS。
其他 margin ≤ 0.5×band 的行：chat row 3（margin 0.221712 = 0.3627×band，ours 与 gold 一致，本来 matched）；
  en/zh 各 0 行。严格口径原始结果 chat 17/18（FAIL）必须保留可见，不得只报 17/17+1。
```

### 替换数字（pre-fix 全部作废）

| prompt | T | argmax（严格） | cosine | max_abs | band | 逐层 min cos | 判定 |
|---|---:|---|---:|---:|---:|---|---|
| en_continuation | 5 | 5/5 | 0.99997235 | 0.159756（0.68×band） | 0.235212 | 0.99993098（layer_12_output） | PASS |
| zh_continuation | 8 | 8/8 | 0.99993803 | 0.235296（0.85×band） | 0.277682 | 0.99993980（layer_21_output） | PASS |
| chat_zh_user | 18 | **17/18**（唯一 row 14 near-tie） | 0.99989976 | 0.416867（0.68×band） | 0.611310 | 0.99988217（layer_23_output） | 修订后 PASS |

```text
final_hidden vs gold：en 0.99994485 / 0.334759；zh 0.99992611 / 0.290360；chat 0.99984263 / 0.430467
decode token 链（不变）：en 11751→13→198→760→6511；zh 271→248068→271→248069→271；
                          chat 109266→6115→103724→1167→16451
state：有限；recurrent |state|max en 0.55–13.27 / zh 0.82–13.29 / chat 0.69–14.14
全量 ut：694 collected / 687 passed / 7 skipped（7 个 skip 全为 test_cuda_qwen_c2.py 无 CUDA 驱动设备项）；
         base 9aae4649e 为 674 passed/7 skipped（新增 13 = portability 10 + 修复测试 3）
```

**重要口径更正**：T=1/decode 在**算子/布局层是 no-op**（单 position cos/sin 位序列修复前后逐位相同），
但**端到端 decode logits 会变**——decode 消费 prefill 产出的 KV/recurrent state；三个 prompt 各自的
51 个 prefill 输出中 **43 个 sha256 改变**，最早改变的是 `layer_03_new_k`。因此
**不得写"decode 端到端无变化"**，也不得把旧证据里的 decode cosine/max_abs 当作修复后数字
（token 链不变，逐步 logits 数字以验收 `validation.json` 为准）。

**未采纳的替代路线**：`327b17158`（`--debug-f32-residual` 调试变体，改 `python/pypto/portable/qwen35.py`）
**未合入 integration、未采纳**，保留在任务分支 `work/qwen35-t18-divergence-localization`；
仅作为"未来若要求严格 18/18"的可选后续路线，当前不进入任何冻结结论。

### 受影响任务与作废数字清单

```text
qwen35-weighted-multiprompt           en/zh/chat 全量 prefill/逐层/误差带数字作废
                                      （旧 en cos 0.99991364/max 0.233829；zh 0.99973494/0.463474；
                                        chat 0.99799387/2.310620、row 11 起放大、layer_07/gold index 8 定位）
qwen35-vector-runtime-packed-liveness en T=5 prefill logits 数字作废（旧 0.99991364/0.233829）
                                      与 final_hidden 数字作废（旧 0.99984563/0.38160705）
verify-w8a-decay-fix                  en T=5 prefill logits 数字作废（C3/C4 的 logits 部分，旧 0.9999136/0.2338）；
                                      decode token 链仍有效
qwen35-t18-divergence-localization    该任务自身 before/after 表：before 数字作废，after 以独立验收为准
qwen35-bf16-weighted-execution        BLOCKED_CAPABILITY，未产出 prefill 数字（driver 拷贝更旧），无可勘误
A2 vllm-ascend / portability 等       虽含旧 driver 拷贝，但未产出 Qwen prefill 数字，不受影响
```

**滚动文档/配置中的旧数字位置**（本批已就地更正）：`HANDOFF.zh-CN.md:56-58`、`README.md:21`、
`AGENTS.md:48-49`、本文件 ERR-0001 段落、`0035` 快照 §2、路线图 `0003` §2/§6/§10、
`configs/development_lock.yaml` 的 `wave8a_results` 与 `current_task`。

**不受影响**：

```text
- portable 图 v3 / graph_digest 66dd4077… / 4,550（(1,1,4096)）/ 13,748 ops（(1,18,0)）/ 权重映射 / layout digests
- ERR-0001 的 GDR decay exp(A_log) 修复本身（独立的真实缺陷）
- 后端 lowering / packed 内核 / 非 Qwen-driver 证据（liveness A/B、AMD 静态、CUDA、SVE 等）
- decode 单步的 RoPE 输入本身（但 decode 状态继承 prefill，需按修复后口径引用）
```

**历史快照的追加标注**：`docs/00-handoffs/0035-*.md` 文末追加勘误指引（不覆盖原文）。

**就地勘误文件**（2026-09-10 追加，只新增、不改原文件）：

```text
_meta/pypto-x/qwen35-weighted-multiprompt/ERRATUM.zh-CN.md
_meta/pypto-x/qwen35-vector-runtime-packed-liveness/ERRATUM.zh-CN.md
_meta/pypto-x/verify-w8a-decay-fix/ERRATUM.zh-CN.md
```

**证据**：`_meta/pypto-x/verify-qwen35-t18-divergence-localization/{validation.json,brief.zh-CN.md,raw/}`
（near-tie 明细 `raw/recompute_metrics.json`、T=1 口径 `raw/decode_invariance.json`、
driver 盘点 `raw/evidence_inventory.json`）。

## ERR-0003（2026-09-11）：W8A8 契约 §3.1 全开 region 计数 555 与 §3.4 lm_head 双 packing 算术不自洽

**状态**：契约勘误（**不影响已实现行为**）。`qwen35-w8a8-binding`（C2）**未实现**该路径；
遇到量化 tied `lm_head` 显式拒绝（fail-closed），因此不存在静默错误——本条只修正文档算术。

**发现问题**：独立验收 `verify-qwen35-w8a8-scheme-binding`（agent `d640b1c6`），discrepancy **VD-3**。

**算术核对**：

```text
契约 §3.1：全开权重 region 数写作 555
契约 §3.4：lm_head 为 tied weight，量化后需要"双 packing"（+2 region）
已实现口径：186 个线性层 → 554 region；再 +2 = 556
即：555 只对应"lm_head +1"，与 §3.4 的"+2"不一致
```

**裁定**：以 §3.4 语义为准，全开应为 **556**，555 为文档笔误；本阶段未实现该路径，
不改变已冻结的计数（BF16 368 / W8A8 默认 518 = 320 param + 150 quant_scale + 48 state）。

**待办**：后续实现 §3.4/§3.5 路径（C3+ 或契约修订）时同步把 §3.1 的 555 修正为 556，
并验证 lm_head 双 packing 的 region 与覆盖率口径。

**证据**：`_meta/pypto-x/verify-qwen35-w8a8-scheme-binding/{validation-verifier.json}`（VD-3 明细）；
契约 `docs/20-planning/0002-2026-09-10-qwen35-w8a8-linear-contract.zh-CN.md` §3.1/§3.4。

## ERR-0004（2026-09-11）：性能协议 §3.3 第 6 步 MAD 离群门在 WSL2 上不可复现

**状态**：协议已修订——`docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` §3.3 第 6 步与
`configs/perf_protocol.proposed.yaml` 的 `outlier_rule` 均已更新（用户 2026-09-11 批准方案 a）。

**发现问题**：W8B B3b 的独立复验（`verify-cuda-gemm-baseline-r2`，agent `7de9f7c4`）按协议字面执行
G2 有效性判定时，**2 个 campaign / 8 轮尝试 → 5 轮作废、0 个 3 连有效轮**；
实现方自己的 K=10 campaign 也有 3 个 eligible 序列恰好停在 `2/21 = 0.0952`（离作废线只差一个样本）。

**根因**：原判据 `|x − median| > 3 × 1.4826 × MAD` **无下限**。实测被作废序列的 MAD 仅占中位数
0.06–0.83%，门限低至 0.28–3.7%，于是**偏离中位 0.3–6% 的 3–6 个样本**即触发
`outlier_ratio > 0.10 → UNSTABLE`。典型反例：某序列 `cv = 0.0023`、`p95/median = 1.0014`
（统计上极稳）仍被判无效。即 MAD 在该平台度量的是**计时器微秒级抖动**，不是分布污染。

**影响**：B3b 的 candidate_ratio **数值本身稳定**（实现方 vs 复验方偏差 ≤0.68%），
但"按门判为有效"不可复现 → **B4 阈值冻结暂缓**（修订后需在静默窗口重跑/重聚合）。

**修订**：第 6 步改为 `|x − median| > max(3 × 1.4826 × MAD, floor)`，其中
`floor = max(1 µs, 0.02 × median)`；`outlier_ratio > 0.10 → UNSTABLE` 不变。
真污染（如 cuEvent rate 偏移 +8~10%）仍会被标记。

**不受影响**：cv / p95 方差门（3.4）、clock/noise/双侧 warmup 判据、`frozen_ratio` 需用户批准的规则、
G1/G3 定义、既有 CPU 侧（local_kvm_no_cpufreq）与 A2/920B 相关口径。

**证据**：`_meta/pypto-x/verify-cuda-gemm-baseline-r2/{brief.zh-CN.md,validation.json,verify_verdict.json,raw/}`
（8 个轮次原件全部保留，含 5 个作废轮与两个真实 harness UNGATED 聚合）；
实现方记录 `_meta/pypto-x/cuda-gemm-baseline/d1-rerun/{brief.zh-CN.md,raw/{k5,round1..3}.json}`。

---

## ERR-0005：并发上限"≤3 subagent"是项目自设假设，不是平台限制

**现状更正**：`AGENTS.md`、`HANDOFF.zh-CN.md` §0/§5、`docs/20-planning/0003-…` §4.1 与 §10 中写有
"平台 4 个 agent 槽位含父 agent → 活动 subagent ≤3"。**该表述没有平台依据**：平台对 subagent 的并发
**没有硬性上限**，这条是项目在 W8 路线图里自设的资源纪律，并被后续文档当作平台事实引用。

**影响**：不影响任何已完成的验收与数字；只影响编排——此前多次因为"3 槽已满"而把可并行的任务排队等待。

**更正后的纪律**（替代原表述）：

1. 平台无硬性并发上限；**真正的约束是资源锁**（`local` / `gamepc` / A3 CPU 时段 / A2·A3 的 NPU）与机器负载；
2. 重任务仍必须经 `scripts/resource/run_local_heavy.sh` 取锁，抢不到退避重试，禁止绕过；
3. 约定：**同一把锁的等待者不超过 2 个**（避免排队噪声），同一台机器不并行跑多个 heavy（锁已保证）；
4. 按用途铺并行度：local-heavy / 远端(GamePC) / A3 / 轻任务 / 文档与评审，互不挤占；
5. 父 agent 负责限制"每台机器上的重度并行度"，而不是限制 subagent 总数。

**依据**：用户 2026-09-12 明确指示（"subagent 其实没有并发限制"）。

---

## ERR-0006：B3c 的"dispatch 开销 41–49% 主要来自逐 op 派发循环"结论被 N4.0 实测推翻

**原结论**（0039 批次，`waves.W8.w8b_b3c_local_perf_verification` 与快照 0039 均引用）：
`dispatch ≈ launch − Σ(op)` 占 prefill 40–49%、decode 49%，"归因 = runtime 的逐 op 派发循环
（`runtime_per_op_dispatch_loop`）"。

**N4.0 的三档分解实测**（integration `c2ec98f3c`，3 轮中位，全部 `UNGATED`）：

| 段 | T=5 prefill | T=1 decode |
|---|---|---|
| launch | 24.460 s | 14.931 s |
| **setup** | **19.148 s（81.1%）** | **10.757 s（72.3%）** |
| loop | 4.458 s（18.8%） | 4.122 s（27.6%） |
| teardown | 0.030 s（0.1%） | 0.021 s |
| loop 内的**纯胶水**（`release_dead_values` + 循环开销） | **0.027 s** | **0.020 s** |
| 每 op 胶水 | **0.003 ms/op** | **0.003 ms/op** |

`setup` 的构成（prefill）：`lower_avx512` ×3 = 5.56 s、`_validate_primary_semantics` ×2 = 5.56 s（exclusive，含 2 次
lower）、`_program_digest_from_metadata` ×2 = 1.26 s、`_decode_artifact` 本体 3.43 s、
`_packed_dispatch_for` 首次 1.51 s；参数绑定与 `plan_release` 各 ~0.004 s。

**为什么 B3c 会误判**：`dispatch` 从来不是被直接测量的量，而是残差 `launch − Σ(op)`；而 setup 里
的**多次 lower / 全量 plan 序列化深比较 / 元数据重建**代价与**程序规模（op 数）相关**，摊到每 op
就是 ~2.8 ms —— 于是被读成了"每 op 派发开销"。**N4.0 的实测把这条抹平到 0.003 ms/op。**

**正确结论**：残差**不是**逐 op 派发，而是**每 launch 的 artifact 校验 / 重复 lowering / ELF decode**。

**影响**：
1. B3c 的 `attribution: runtime_per_op_dispatch_not_harness` 字段应读作"残差、未分解"，**不是**归因结论；
2. "把图执行外包给框架"因此**不成立**：整段 loop 胶水即使全交出去也只能省 0.017/0.012 s（≈ launch 的 0.07%），
   而 setup 是**我们自己的 fail-closed 校验与 lowering 路径**，任何外部框架都消不掉；
3. 正确方向是**在自家做 launch/进程级 artifact→target_ir 缓存**（N4.0 建议）。

**已落地的部分修复**：per-launch `_LaunchValidationMemo`（`1dc9c1261`）在同一 launch 内复用 primary 校验，
实测 prefill 24.460 → 18.092 s（−26.0%）、decode 14.931 → 11.819 s（−20.8%）；51 个输出 buffer + artifact digest
的 sha256 全一致，fail-closed 用例全拒。
**仍剩余**：去重后 prefill 仍有 `_validate_primary_semantics`（2.92 s + lower 1.80 s）与 `_decode_artifact`
本体（3.10 s + lower 2.00 s）对同一 6728-op 程序各做一次 lower + 全量 plan 深比较（≈9.8 s；decode ≈3.2 s）。

**证据**：`_meta/pypto-x/n4-dispatch-residual/{brief.zh-CN.md, raw/decomposition-table.md, raw/analysis.json, raw/{base,after}_r{1,2,3}.*}`。

---

## ERR-0007：aarch64 vendor int8 的布局证据被父 agent 转述反了（stride 语义）

**错误**：父 agent 在把 `axis-b-a3-runtime` 的证据转给 W8A8 v2（路线 c）的 M1 设计任务时，写成
"oneDNN+ACL 要求 `[K,N]` ⇒ **c1 落地后 aarch64 的 vendor 路径将不再需要任何转置**"。

**事实**（M1 核实并提出，父 agent 复核两处原始证据后确认）：

| | 需要的字节序 | 证据 |
|---|---|---|
| 四后端 portable kernel | `[K,N]` **行主序、N 连续**（读 `right[i*n + j]`） | `cpu_avx2.py:511/520/526`；AVX-512/SVE256/CUDA 同构 |
| aarch64 oneDNN+ACL | `[K,N]` 且 **`stride(0)==1`** ⇒ **K 连续**，即 `[N,K]` 行主序的**转置视图** | `_meta/pypto-x/axis-b-a3-runtime/brief.zh-CN.md:188-191` |
| c1 的打包布局 | `[K,out]` 行主序 ⇒ **N 连续** | 立项书 `0009` §3 |

**结论**：**c1 与 oneDNN+ACL 要的是相反的字节序**；c1 的收益只覆盖 portable 四后端
（kernel 0 改动、SVE256 解封、每 forward 转置归零），**不**消除 vendor 侧的布局需求。

**范围决策（父 agent，2026-09-12）**：**不要求 portable 与 vendor 共用同一 int8 storage**——因为实际上存在
**三种互不兼容**的需求（portable `[K,N]` N 连续 ／ oneDNN+ACL 的 K 连续转置视图 ／ KleidiAI 的私有 packed RHS），
单一布局不可能通吃；强行共用只能走 c2（4 个 NT kernel + R5 重验）或双 region（+474 MiB 存储/搬运）。
vendor 侧各腿在 **ingestion 期做一次性 repack**，并把"需要哪种布局"登记为 provider capability 的前置条件。
c2 不被否决，而是"当且仅当将来要求共用 storage 时启用"。

**教训**：跨任务转述**布局/stride 语义**时，必须引用**原始证据的原文与行号**，并显式写出
"哪个维度连续"——`stride(0)==1` 这种写法极容易被读反。本项目的"布局类"结论今后一律按此格式登记。

---

## ERR-0008：cherry-pick sequencer 未清理 + 后续 `--abort` 静默回退分支，丢掉了 Q3 的 4 个提交

**现象**：`integration` 上"应该已合入"的 Q3（`c8-l4-avx512`）4 个提交实际不存在；`tools/c8_l4_compare.py` 是
Q5 版（`TOOL_VERSION=1`、CLI `--ours-dir`），而 Q3 规范版（`TOOL_VERSION=4`、CLI `--ours-manifest`）只存在于
`work/c7-decode-gap-localization`。**由集成健康检查任务发现**（`integration-health-check`，早期发现）。

**根因（integration reflog 直接可见）**：

```text
@{19}/{18}  cherry-pick: acbf4d340 / ad28cc8a8   ← Q3 的提交确实曾落在 integration 上
@{17}       reset: moving to acece0a92           ← 一次 cherry-pick --abort 按 sequencer 记录的起点重置了分支
```

父 agent 处理 Q3 冲突时用 `git commit -m` 手动收尾（`cherry-pick --continue` 因 `EDITOR unset` 失败），
**但 sequencer 状态没有被清掉**；其后处理 Q8b 时又出现"拣选操作已在进行"，父 agent 执行
`git cherry-pick --abort`，于是分支被重置到 sequencer 记录的起点 `acece0a92`，
**静默丢弃了已提交的 Q3 4 个提交**（它们因 `work/c7-decode-gap-localization` 恰好从 `ad28cc8a8` 创建而幸存）。

**修复**：重新 cherry-pick `f2e17777f^..ad28cc8a8` 到 integration → 新 tip `c8ab4c6f6`；
校验：规范版 `TOOL_VERSION=4`、归档版文件在、两者 `py_compile` 通过。Q3 的 L4 判定证据本身**不受影响**
（它在 `ad28cc8a8` 上产出，内容与重拣后一致）。

**教训（流程级）**：

1. **不要用 `git commit` 收尾 cherry-pick**：要么 `cherry-pick --continue`（先设 `GIT_EDITOR=true`），要么
   明确 `cherry-pick --quit` 清掉 sequencer；**sequencer 挂着时任何后续 `--abort` 都会回退分支**；
2. **"已合入"必须有独立核验**：登记 `integration_commits` 前应 `git merge-base --is-ancestor <sha> <tip>` 逐个验证
   （本次若做一步就能当场发现）；该核验已加入本仓的惯例；
3. 这也解释了为什么"cherry-pick 后必须立刻跑全量"这条规矩（KF-1 教训）是必要的——**分支状态也会骗人**。

## ERR-0009：0042 项 1 的"report 单文档边界"表述被独立验收证伪（校验面按已知缺口逐点补，而非按副本穷举）

**现象**：0042 项 1（`748717529`，cherry-pick = `15d0fe58d`）新增 4 类 cross-block + 2 类 contract-derived must-reject 后，
`binding_verify` 的模块 docstring 与 `0006 §7.1` 声称"report 中每个在另一块有真源的字段都被重算并比较；剩余边界仅
manifest-only 字段与 `resolution.plan_digest`"。**独立验收（agent `cb32c1ba`，冻结 `15d0fe58d`）穷举证伪**：

- 按"代码 provenance 等价类"枚举 41 类，**39 类存在 ≥1 未对账成员**；仅 `request.policy_digest↔resolution.policy_digest`
  与 vendor `artifact.library↔resolved.library` 是完整单点受检。
- 单叶扫描：portable **335/449**、vendor-f32 **347/536**、fallback **404/518** 个叶子的单点篡改被 `validate_report_schema` 接受。
- `resolution.rules[*].details` 回显面 62 叶中 **61 叶**可单点篡改被接受（仅 `provider.select.primary` 被拒）。
- 两个**真实绕过**（不是"未登记边界"）：`resolution.fallback_chain=[]` 使 `report_provider_declaration_mismatch` 的关系检查
  被整体跳过；`request.dtype="complex64"` 配合 integer 表值的 `guarantees.accumulation` 被接受（dtype 无成员校验）。
- 父方独立探针在修复前同样复现：`probe_request_block_hole.py` 4/4 接受、`probe_rule_echo_holes.py` 18/19 接受。
- 执行面不受影响（验收步骤 11）：`validate_report_schema` 只在 `ExecutionReport.__init__`（执行之后）调用；伪造 plan 仍被
  `execute_matmul_plan` 的 canonical 重算以 `plan_declaration_mismatch`/`artifact_declaration_mismatch` 拒绝且输出未动。
  ⚠️ 但若下游消费者从 report 重新规划，则 `request.op/shapes/dtype` 会变成执行相关——此 caveat 必须写进契约。

**根因**：校验面是**按"已知的 4 对 cross-block"逐点补**出来的，而不是按"文档内每个字段的全部副本"做 provenance 穷举驱动；
且文档用了过宽措辞，把"权威块"与"说明层"（`resolution.rules[*]`、`resolution.candidates[*].checks[*]`、`resolution.why/tie_break`、
`coverage.*`、叙述性 timing/resources 字段）混为一谈。

**修复**（0042 项 1b/1b2，同一分支）：`binding_verify.py::report_rule_echo_errors` 按 **rule 名**（非下标）对账命名回显，
缺失/重复/未知 rule **fail-closed**；补齐 `request.{op,dtype,shapes}`、`coverage.*`、`artifact.*` 版本、capability/resolved_policy/
target 副本、threads、guarantees 派生字段；拒绝空/重复 fallback chain；对被选中候选补 rank/dtype 校验；并把 docstring 与
`0006 §7.1` 改成精确表述（normative = 被对账的字段；explanation layer = 明确列举且**非证据**；执行授权只以 canonical 重算为准）。
修复后父方合并探针：**28 篡改中 27 拒**（唯一残留为位置型 candidata rank 回显，见 1b2 处理）。

**教训**：
1. **校验器的边界声明必须由穷举驱动**（按字段副本的 provenance 等价类），不能由"已知缺口清单"驱动；
   缺口清单只能证明"修了什么"，不能证明"还剩什么"。
2. **validator 的 docstring/文档是契约的一部分**：过宽措辞（"every field …"）会被独立验收当作可证伪声明；
   写"我们检查 X"而不是"不存在未检查的 Y"。
3. **未登记 ≠ 边界**：单文档中真源在外的字段（timing/resources/叙述串）必须**显式**排除在证据契约外并给理由，
   否则"边界清单"会把绕过（如空 chain 跳过检查）与固有不可验证混在一起。

## ERR-0010（2026-09-12，三次修订）：AOCL int8 偏差包络先被 Q8 低估、修订后的实测快照也不是上界（最终：不提供解析上界，改由 policy fail-closed）

**现象（第一次）**：`docs/20-planning/0006` §8.2 记录 AOCL `s8s8s32os32` 的实测偏差包络
`5099 → ≤6`｜`16696 → ≤33`｜`133144 → 22`。U2b 在同一固定库（BLIS 5.3.2 zen4、
sha256 `c7d74a31…`）上用 `uniform[100,127]`、seeds 1–8 复测得 `5099 → 8`、`16696 → 56`、
`133144 → 457`，且 `133144` 的 all-127 构造用例是 24（连 Q8 自己 E3 的 24 都未被其记录的 22 覆盖）。

**现象（第二次，独立验收 agent `71599188`）**：把同一生成器扩到 seeds 1–200 后，
`16696 → 75`（seed 147）、`133144 → 841`（seed 115）；验收方另一 seed 族到 `133144 → 924`
（seed 48），648 用例扫描到 5099 → 7、16696 → 62、133144 → 678；继续扩到 **seeds 1–1000**
后同一生成器到 `133144 → 966`（seed 428），且 1000 个种子里有 503 个超过第一轮声明的 `457`
（`16696` 侧为 48/1000 超过 `56`）。即 U2b 第一轮修订出的"声明包络 = max(Q8 参考, 实测)"
仍**不是上界**，且随扫描规模继续上抬。原始证据：
`_meta/pypto-x/u2b-vendor-int8/raw/{bit-envelope.json,envelope-refutation.json}`、
`_meta/pypto-x/verify-0042-u2b/raw/{envelope-author-scope.json,envelope-author-scope-1000.json,envelope-exceedance-stats.json,envelope-top-seeds.json,bit-envelope.json}`。

**现象（第三次，独立验收 agent `71599188` 的 r2 复检）**：同一生成器继续扩到
**seeds 1001–6000** 后 `133144 → 1067`（seed 1591，threads 1/2/6 复核一致，provider 1715435392
vs exact 1715436459）；**seeds 6001–12000** 另有 `133144 → 1043`（seed 6323），`16696` 在
12000 seeds 内保持在 75（seed 8410）。即 r2 冻结时记录的 `966` 又被刷新到 `1067`；这再次
说明该表的性质是**可继续上抬的实测快照**，而不是界。原始证据：
`_meta/pypto-x/verify-0042-u2b-r2/raw/{extension-scan-1001-6000.json,extension-scan-6001-12000.json,extension-witness-crosscheck.json}`、
本任务 round-4 证据 `_meta/pypto-x/u2b-vendor-int8/raw/envelope-extension.json`（自有新窗口
12001–15000 + witness 复核 + domination）。

**根因**：
1. 偏差同时依赖 **K 与码值分布**：KC=2048 的 int32 block partial 在 f32 上链式累加，
   大码值下 f32 ULP 达 4–256，"3 seeds × 单一分布"只是观测快照；E10 本身只测到 K=5099，
   更大的 K 来自少量构造用例，不是 max-over-distributions。
2. 更关键：**真实内核不是可解析建模的"块内精确 int32 + 块间纯 f32 链"**。pinned zen4
   int32 输出内核（`kernels/zen4/lpgemm/s8s8s32/lpgemm_6x64rowmajor_s8_amd512vnni.c` +
   `u8s8s32/lpgemm_s32_kern_macros.h`）把 s8 加 128 转 u8 走 `vpdpbusd`，再用 **int32**
   减去 B 列和补偿；每个 KC block 结束时把 int32 累加器 `_mm512_cvtepi32_ps` 转 f32、
   必要时做 post-op/beta，再 `_mm512_cvtps_epi32` 写回 int32；m/n fringe 还走不同 kernel。
   Q8 自己的"块间 f32 链"模型只 25/31 用例逐位吻合（mixed 差 2–5），不是内核语义。
3. 因此按文档语义推导的递归界（K=16696 为 60，见 `_meta/pypto-x/u2b-vendor-int8/raw/analytic-bound-attempt.json`）被真实 provider 观测的 75 直接证伪；
   而"任意 f32 求和顺序"的通用界（K=133144 约 8.4e3）又超过 int32 头寸（最大精确和
   2,147,479,576 距 2³¹ 仅 4,072），连"不会静默回绕"都无法证明。

**修复（最终语义，未放宽契约）**：
1. `vendor:aocl-lpgemm-int8` manifest 明确声明 `deviation_bound_kind = "measured_snapshot"`、
   `deviation_bound_available = false`，**删除** `deviation_bound_by_k`；观测值改名
   `measured_deviation_by_k`（聚合 internal 扫描 + 外部 seeds 1–200/1–1000/1001–6000/
   6001–12000 扫描：16696→75、133144→1067），
   并在 `measurement_scope` 记录分布/seed/线程/外部证据路径与
   `observations_are_not_a_bound=true`；`upper_bound.status="not_provided"` 附阻塞证据。
   每次刷新都进 `provider_artifact_digest`，快照值可随新证据继续上抬。
2. 数值分类仍 `deterministic_bounded`：K≤1024 全分布/全线程逐位；K≥133145 在任何 kernel
   调用前 fail-closed；`-128` 与 `True/False`（round 4 恢复）在适配层拒绝。
3. 新增 policy 旋钮 `numeric.require_proven_deviation_bound=true`：resolver 只接受
   `deviation_bound_kind ∈ {exact_contract_bound, analytic_f32_chain_upper_bound}` 的 provider；
   拿到 `measured_snapshot`（或 deterministic-bounded 但无 envelope）时结构化拒绝
   `proven_deviation_bound_unavailable`（无 fallback 时 `NUMERIC_GUARANTEE_UNMET`），
   portable 参考主干因 `exact_contract_bound` 仍可用。plan/report schema 同步规定：
   measured-snapshot envelope **不得**携带 `deviation_bound_by_k`，**必须**携带
   `measured_deviation_by_k`（plan/report 两侧对称）；proven kind 的 bound 必须逐 K 支配
   自己的观测，否则 fail-closed。

**覆盖与不覆盖（本任务结论的边界）**：被观测/建模的量是
`|AOCL_int32 - 精确 int32 参照|`（LC 契约的 int8×int8→int32 段），输入限码值
`[-127,127]`、`K ≤ 133144`、m/n/线程在记录 scope 内。它**不覆盖**：
反量化/epilogue 之后的输出误差（例如 fused `(acc×s_w)×s_a` 的 bf16 误差）、`-128` 与
K>133144（两者在适配层 fail-closed，不进入 kernel）、其它 AOCL 构建/架构、以及
scope 之外的 shape/线程。解析上界若要成立，必须覆盖内核的真实算术路径；本轮结论是
**不提供**。

**教训**：
1. **字段名就是契约**：`deviation_bound_by_k` 一旦存在就会被消费者当成上界；实测快照必须
   用 `measured_*` 命名并带 `*_kind`，且让"要求上界"的消费者 fail-closed。
2. **模型界 ≠ 内核界**：从源码/文档推导的界必须先用真实 kernel 路径做反例扫描；本轮
   "60 被 75 证伪"说明只做位级 parity 的少量用例不足以支撑一个界。
3. 误差预算类字段必须有 **bound kind + coverage + not-covered** 三段式语义，不能只给数字。


## ERR-0011：任务分支被重建后，按"尾部单提交"cherry-pick 会产生不一致的中间态

**现象**：0042 项 2（U2b）在 round-3 期间把分支 `work/u2b-vendor-int8` 重建（rebase/重放），round-1 的 squash 提交 `7b77284e9`
不再是其祖先。父 agent 只按"最后两个提交"cherry-pick `11879ec27` + `de919e6b5`：冲突解完后，被冲突文件与源提交**仍不一致**，
第二个提交又在 `plan.py`/`report.py`/测试三处冲突——因为这两个提交依赖同分支上未被合入的另外 6 个提交
（`ab58c11d9..87aa3fbae`：包络改快照、支配不变式、文档、测试等）。

**根因**：把"任务分支=当前 integration tip + 我已知的那几个提交"当成不变量，而没有在 pick 前核对分支相对当前 tip 的**完整范围**；
分支被重建后，"尾部单提交"的父状态不在 integration 里，于是 pick 只能得到半成品（且第二个提交必然冲突）。

**修复**：`git cherry-pick --abort`（integration 未受污染，回到 `b89e0b521`）；改为让**作者自己 rebase 到当前 tip**
（`git rebase b89e0b521`），再整段重放 `b89e0b521..<新 HEAD>`（本次 8 个提交，无冲突），并核对 cherry-pick 后 **tree 完全一致**
（`8f705bb0d…`）与 patch-id 一致。

**教训**：
1. **pick 前必做**：`git log --oneline <integration_tip>..<branch>` 看完整范围；范围里出现的每个提交都要有归属（已合入/待合入/已知依赖）。
   若分支被重建过（旧 SHA 不在历史），一律按"整段范围"处理，或让作者 rebase 后重放。
2. **两个提交以上、且与 integration 有同文件交叉时，优先让作者 rebase**：作者懂两侧语义，且能顺带在新 tip 上复跑自己的测试；
   父 agent 手解冲突容易在"看起来能过"的状态下把半成品合进去。
3. **等价性判据要选对**：基线相同的 cherry-pick 用 **tree 相等**；基线不同的 cherry-pick 用 **patch-id 相等**（本次 AVX2 切片即基于旧 tip，
   树必然不同但 patch-id `83811f53…` 相同）。两者都不可省。

## ERR-0012：CUDA 线"被工具链阻塞"的结论来自**引用过期 lock 字段**（实际 5080 + nvcc 13.3 早已可用）

**现象**：2026-09-12/13 的父 agent 报告与若干快照/HANDOFF 反复写"本机无 GPU/nvcc，CUDA 线被 toolchain 阻塞"，
并引用 lock 的 `gamepc_toolkit: nvcc_nvrtc_cudart_sdk_headers_absent` 与 `cuda_c1_acceptance_smoke: BLOCKED_TOOLCHAIN_nvcc_absent`。
用户在 2026-09-13 指出"先前应该在 GamePC 上用 5080 完成过执行"。实测（只读 SSH 探测，未申请 `gamepc` 锁——按约定 GPU-only 探测不需要）：

```text
GamePC 192.168.101.5（Windows + WSL2）：
  nvcc:      Cuda compilation tools, release 13.3, V13.3.73  (/usr/local/cuda-13.3)
  nvidia-smi: NVIDIA GeForce RTX 5080, 16303 MiB, driver 616.92
  gamepc 锁: FREE；GPU owner: pypto-x-exclusive
```

**根因**：两处**过期快照**没有被后来的完成记录覆盖所推翻：
1. `gamepc_toolkit: ..._absent` 出自 2026-09-09 的 `gamepc-cuda-probe`；此后 `cuda-toolkit-wsl` 任务
   （commit `0f2e82158`，`toolkit: cuda_13_3_nvcc_13_3_73_cublas_13_6_0_2`、`install_bytes: 10710024192`）已把它安装完毕并 `status: complete`；
2. `cuda_c1_acceptance_smoke: BLOCKED_TOOLCHAIN_nvcc_absent` 同理；而**同一条记录里**早已有
   `cuda_c1_real_5080_driver_ptx: PASS`（真实 5080 上跑过 fp32/bf16 elementwise/reduce/matmul/multi-op chain，证据 `_meta/pypto-x/integration-w4-cuda-c1-final`）。
另外驱动已从 610.62 **漂移到 616.92**（`configs/perf_protocol.proposed.yaml` 里那条"驱动漂移待归档"的问题也因此有了实测值）。
父 agent 只在本机（12 vCPU KVM guest）跑 `nvidia-smi`/`which nvcc` 失败后，就复述了上述过期字段，没有去 GamePC 复核。

**修复**：
1. lock 里两处过期字段已就地追加 `*_superseded_2026_09_13` 说明（保留历史、标注当前真值）；
2. 本 ERRATA 与 HANDOFF 的"CUDA 被阻塞"表述全部订正为"CUDA 线**可用**（GamePC 5080 16 GB + nvcc 13.3.73 + cuBLAS，`gamepc` 锁空闲、GPU 独占）；
3. CUDA 线重新排队（C1 acceptance smoke 重跑、cuBLAS 基线、W8A8 CUDA kernel 真机复验）。

**教训**：
1. **禁止把 lock 里的状态字段当作"当前事实"引用**——它们是某次探测的时间快照；引用前必须看同一记录里是否有更晚的完成/覆盖记录，或直接复测。
2. **"本机"≠"本项目可用资源"**：本项目有 GamePC（RTX 5080）、A3（SVE256）、920B 等外部资源，判断某条线是否可跑必须逐资源确认（HANDOFF §0 应列出各资源的可用性与其锁）。
3. 同一条记录内自相矛盾（`BLOCKED_TOOLCHAIN` 与 `real_5080_driver_ptx: PASS` 并存）时，必须先解决矛盾再对外报告。
