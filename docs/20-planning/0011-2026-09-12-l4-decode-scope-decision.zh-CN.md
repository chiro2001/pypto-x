# L4 口径决策包：门是否包含 decode 扩展段（含 C7 最终数据）

编号：`0011`｜日期：2026-09-12（Asia/Shanghai）｜状态：**待用户裁决**（`pending_user_decisions_2026_09_11` 第 1 项）

本文件把 L4 口径裁决所需的全部事实压成一页：契约 D5 要求 L4–L6 阈值由用户确认；C8 主判定（Q3/Q5 口径）
与 C7 缺口定位（Q1–Q4，2026-09-12 已闭环）已给出可直接裁决的数据。**本文件不改任何判据**，只列选项与代价。

---

## 1. 现在的判定是什么

| 口径 | 结果 | 证据 |
|---|---|---|
| **prefill**（row0=prefill_next + prefill 块） | **6/6 PASS**（chat 2 个 near-tie flip 单列） | `c8-l4-avx512`、`c8-w8a8-reference-gold` |
| **decode**（declared decode key，rows0–3 的 cosine 门） | **6/6 FAIL（仅 cosine 门）** | 同上 + C7 `raw/tables_decode_gap.md` |
| 同一批输出改用 prefill band 判 decode | 5/6 PASS（仅 zh/default 差 −0.00326） | C7 §3 |

## 2. 支撑"门不自洽"的四条硬事实（都可复算）

1. **门限之间差 50–170×**：cosine 门隐含允许的误差范数比只有 **1.0056–1.0197×**；max_abs 门允许 **2.000×**；
   而我们 decode 实测范数比 1.090–1.292（最差 1.574，zh/default/full row0）。
2. **参照实现自己过不了这道门**（正交零假设检验）：把"与 gold 正交、范数等于**参照自身**误差"的误差代进同一
   cosine 门，**12 行里有 5 行仍然失败**（default/full 各 5/12）——即门比参照自身的噪声还严。
3. **参照自身不确定性 1–2%**：gold(BF16/参照路径) vs fp32 的 cosine 本身只有 0.983–0.998（中位 0.9953），
   声明的 decode floor = 参照自身 cos − 1e-4，落在该不确定度之内。
4. **但缺口是真实的、不是口径造成的全部**：范数比中位 **1.12**、最差 1.51；C7 的 state-swap 分解显示
   缺口 = **state 回灌（解释 62–119%，中位 ~96%）＋同 state 单步计算差异（47–115%，中位 ~82%）**，两者不共线、部分抵消；
   逐层消融**无单一热点层**（top-8 层合计只占 26–51%，9/24 层单独换 gold 反而变差）。

## 3. 三个可选项与代价

| 选项 | 含义 | 代价 | 备注 |
|---|---|---|---|
| **A（收口）** | L4 判 **prefill 口径 PASS**；decode 单独登记为"**口径待重定义**的 gap"，不算 L4 失败 | 0（只需注册） | 与已有 C8 Q3 判定一致；decode 事实仍如实登记 |
| **B（重定义后重判，推荐与 A 并用）** | decode 门改用与 max_abs 同量级的口径，例如 **误差范数比 ≤ max(参照自身比, 1.25)** 或 **相对参照自身 cosine 的退化 ≤ δ**；随后重跑一次 C8 decode 判定 | 一次 C8 decode 重判（~分钟级，需 local 锁；A3 不可用时用 AVX-512 腿） | 新门应写成契约里**可复算**的公式，不用"感觉合理"的常数 |
| **C（维持现门）** | decode 维持 6/6 FAIL；需要先把 56% 的 compute 缺口下钻到算子，再决定是否改实现 | 增量诊断 +（可能的）实现修复 | 只有在你需要"现门必须过"时才走这条路 |

**C7 下钻实验结果（2026-09-12 05:38Z，已跑完）**：把 zh/default decode step2 的输入 state 固定为 gold、逐段对比中间张量后，缺口是**分布式**的——layer 输出相对 L2 从 layer_00 的 0.054 逐层增长到 layer_23 的 0.213（无跳变）；四个 block 段（input→input_norm、input_norm→token_mixer、post_attention_norm→mlp、mlp→output_residual）的中位相对 L2 都在 0.144–0.150；**top-5 段的平方份额只有 8.9%**，没有支配性算子/段。首次可测偏差出现在 `layer_00_branch_output`（rel_l2 0.0147、cosine 0.99990）。

**父 agent 建议：A + B。** 理由：A 让 C8 的 L4 判定立刻有结论、不阻塞后续（Q5 重跑/W8A8 v2），
B 把 decode 事实保留为可复算的门（不放过真实缺口），而 C 在拿到"哪几个算子在 decode 上贡献 56%"之前
投入产出比不明——C7 已把下一步最小实验写清（`brief.zh-CN.md` §5，最小复现 = `zh_continuation` + default scope + 3 个 decode step）。

## 4. 裁决后我会立刻做的事

1. 按裁决更新 `configs/development_lock.yaml` 的 L4 口径与该批已冻结结论（不改历史数据，只改判据登记）；
2. 选项 B：把新门写成公式 → 跑一次 decode 重判（AVX-512 腿，A3 恢复后补 SVE256 腿）→ 登记门/阈值版本；
3. 选项 C：下钻实验**已完成**（结论：分布式、无支配性算子，证据 `c7-decode-gap-localization/raw/compute_drilldown_step2.json`）；若你要走 C，下一步是把"分布式小差异"逐个算子归因（工作量与收益需另评估）；
4. 无论哪个选项，`docs/20-planning/0010`（W8A8 v2）与 Q5 重跑不受影响，可并行推进。

## 5. 证据索引

```text
C8 参照 gold/band    ../worktrees/_meta/pypto-x/c8-w8a8-reference-gold
C8 L4 AVX-512        ../worktrees/_meta/pypto-x/c8-l4-avx512
C7 缺口定位（本轮）  ../worktrees/_meta/pypto-x/c7-decode-gap-localization
  · raw/tables_decode_gap.md        逐行曲线 / 门限预算 / 正交零假设
  · raw/state_swap_experiment.json  18/18 分解 + 逐层消融
  · raw/state_swap_summary.json     汇总
  · brief.zh-CN.md §2.4/§2.5/§4/§5  结论与最小复现命令
```
