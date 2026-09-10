# PyPTO-X 勘误索引

本文件登记**已冻结结论被后续发现推翻或修正**的条目。历史快照文件只追加勘误、不覆盖原结论；本文件是统一索引，各受影响证据目录内另有就地 `ERRATUM.zh-CN.md`。

## ERR-0001（2026-09-10）：Qwen3.5 GDR decay 门缺 `exp(A_log)`

**状态**：已修复并验收到 integration `9aae4649e`。

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

**历史快照的追加标注**：`docs/00-handoffs/0026/0030/0031/0032/0033/0034` 文末已各追加一行勘误指引；
滚动文档 `HANDOFF.zh-CN.md`、`README.md`、`AGENTS.md`、`docs/WORKTREE_AGENT_PLAN.zh-CN.md`、
`docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` 已就地更正并标注。
