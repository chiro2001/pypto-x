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
