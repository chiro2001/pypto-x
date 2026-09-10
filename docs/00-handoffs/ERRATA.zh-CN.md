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

```text
graph contract version   3
graph_digest(past=4096)  66dd4077...
Core operations          4,550（(1,1,4096)）/ 6,728（(1,5,0)）
prefill 对齐（真权重）    argmax 5/5、cosine 0.9999136、max_abs 0.2338（gold dtype 带 0.2352）
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
