# W8A8 契约 v2：消除源头 int8 转置（路线 c）立项书

文档编号：`0009`

日期：2026-09-12（Asia/Shanghai）

状态：`ESTABLISHED_BY_USER_APPROVAL_PHASE1_DESIGN_DISPATCHED`

用户批准：2026-09-12「是否立项消除源头 int8 转置……我认为合理。同意。」

---

## 1. 动机（已核实的事实）

W8A8 线性层在当前契约（`docs/20-planning/0002-…`，**FROZEN**）下的图上形态：

```text
weight_s8 [out, K]  ──_transpose((1,0))──▶  weight_t [K, out]  ──▶  qmatmul_s8s8_s32(x[M,K], weight_t)
```

- 权重是图的**参数**，该转置节点**每次 forward 都执行**，而结果是常量；
- Q5 实测（SVE256 整网冲突扫描）：**default 配置 150 个 / full 配置 187 个 int8 转置**，
  每 forward 搬运 **474.0 / 717.1 MiB**；
- 这是**纯重复劳动**：一次准备可完成的事被放进了每次推理的热路径；
- 它同时是 **SVE256 上 W8A8 无法 lowering 的唯一违规类**（Q5 的 shadow 遍历已证明"唯一"）。

## 2. 目标与非目标

**目标**

1. 每 forward 的 int8 权重转置搬运算量从 474/717 MiB 降到 **0**（对所有后端一致）；
2. SVE256 上 W8A8 整网可直接 lowering（不再依赖 int8 layout 支持，尽管该支持本身仍有独立价值）；
3. **对外数值行为零变化**——这是硬判据，见 §5。

**非目标**

- 不改量化方案本身（码制、scale 粒度、RNE、饱和策略、累加宽度都不动）；
- 不引入新的融合 opcode；不顺手做别的图优化；
- 不改 vendor 路径的语义（AOCL/LPGEMM 的接入方式不变）。

## 3. 两个候选方案（Phase 1 必须二选一并给证据）

| | 做法 | 预期影响面 | 待证 |
|---|---|---|---|
| **c1（优先）** | **打包期就按 `[K, out]` 存权重**，图里删掉那个转置节点 | 改 packed layout + binding schema（契约 v2）；**若新布局正好是 qmatmul 期望的 `[K,N]` 连续内存，四后端的 kernel 可能完全不用改** | 四后端 kernel 的右操作数寻址/stride 假设是否接受 `[K,out]` 连续；packer 与 binding 的改动清单 |
| **c2** | 给 `qmatmul_s8s8_s32` 加 **NT 语义**（右操作数允许 `[out,K]` + 转置标志） | 改 opcode 契约 → **四后端 kernel 都要动** + R5 重验 | 是否有必要（若 c1 可行则不需要） |

**Phase 1 的产出就是"选 c1 还是 c2、以及每条后端改动点清单"**；若两者都不可行，必须给出替代方案或明确放弃。

## 4. 影响面（Phase 1 必须逐项核实并列出文件/证据）

```text
① 契约      docs/20-planning/0002-…（FROZEN）→ 升 w8a8-linear.v2；§3.3 packed layout、R 系列判据
② binding   python/pypto/portable/runtime_binding.py、qwen35_w8a8_weight*.py、packer 侧
③ 图结构    删 150/187 个 transpose 节点 → 【graph_digest 改变】
④ 后端      AVX2 / AVX-512 / SVE256 / CUDA 的打包喂参与 qmatmul 调用路径（kernel 语义尽量不动）
⑤ 证据      【最大代价】所有按 graph_digest 冻结的基线全部失效，需重做（见 §6）
```

## 5. 硬性验收（对外主张不变是底线）

1. **v1 vs v2 逐位一致**：同一输入、同一后端下，v1 图与 v2 图的**全部输出 buffer `sha256` 完全一致**
   （消除一次转置不应改变任何数值——它只是把同一份 `[K,N]` 数据提前物化）。覆盖：三 prompt × 两配置 × prefill+decode，
   以及 W8A8 四 opcode 的聚焦用例；
2. **每个后端**都要有这份逐位证据（AVX2 / AVX-512 / SVE256 / CUDA），不允许"只在一台上验过就推广"；
3. **转置确实消失**：执行 census 显示 int8 transpose 调用数为 **0**，且每 forward 搬运量按实测下降（给出前后字节数）；
4. **契约版本与 fail-closed**：v1 artifact 在 v2 运行时必须被**明确拒绝**（旧版本 fail-closed），反之亦然；
5. **既有回归不减少**：全量 pytest 收集数不减少、失败集合为空（KF-1 例外已由独立任务修）。

## 6. 重基线清单（**必须作为项目的一部分显式排期**）

以下证据挂在当前 `graph_digest`（BF16: `66dd4077` / `456ab519…d786e`；W8A8 图另有其 digest）上，
v2 合入后需要重做。Phase 1 要把它变成**可执行清单**（每条：证据路径 + 重跑命令 + 预计机时 + 是否可自动化）：

```text
- BF16 累积回归（T=1/T=5/T=18、en/zh/chat）与 artifact 版本链
- W8A8 整网 e2e（C8 的 L4 比较：Q3 已产出的 AVX-512 判定 + Q5 待产出的 SVE256 判定）
- B6 的 L1 数字（layout 原生化后的 per-op 与墙钟）
- 各后端 W8A8 opcode 套件（C1–C6 的图级部分受影响，opcode 级不受影响）
- 覆盖率/区域计数（556 = 321+187+48：形状变了，计数与 schema digest 都要重算）
- 若 binding digest 变化：`layout_digest`、`quant.scale_entry` 等元数据引用
```

## 7. 里程碑与闸门

| 阶段 | 交付 | 闸门 |
|---|---|---|
| **M1 设计/影响面（已派发）** | c1 vs c2 决策 + 逐后端改动清单 + 重基线可执行清单 + 迁移与回滚方案 | 父 agent 评审；**不合代码** |
| M2 实现（分支上） | packer/binding/图构造改动 + 后端适配 + 聚焦测试 | 分支自测 + 逐位一致（v1 vs v2） |
| M3 独立验收 | 四后端逐位、census、fail-closed、契约文档同步 | 独立 verify worktree |
| **M4 波次边界合入 + 重基线** | 合入 integration 并执行 §6 全部重跑 | **必须选在一个批次收口点**，不能插在验收中途 |

**M4 的时序约束**：本批（0040）尚有多项验收依赖当前 digest，因此 **M2/M3 可以在分支上推进，但 M4 必须等本批收口**。

## 8. 风险

| 风险 | 处置 |
|---|---|
| 新 packed 布局破坏某个后端的寻址假设 | Phase 1 逐后端核实；M3 必须四后端逐位 |
| 重基线被低估（"顺手就能重跑"） | §6 必须给出可执行清单与机时估算，作为 M4 的前置 |
| 与正在进行的工作冲突（Q10 的 SVE256 int8 layout、C8 的 L4 判定） | Q10 仍有独立价值（int8 layout 是通用能力），照常收口；C8 的 L4 判定在 v1 上完成后再考虑是否需要 v2 复跑 |
| 契约 v2 与 v1 长期并存导致混乱 | v1 明确标 deprecated + fail-closed 拒绝；v2 为唯一现行版本；ERRATA 登记切换点 |
