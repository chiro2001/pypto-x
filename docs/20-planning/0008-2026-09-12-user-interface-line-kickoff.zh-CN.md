# 用户接口线启动：U1 首切片（matmul + portable provider）

文档编号：`0008`

日期：2026-09-12（Asia/Shanghai）

状态：`STARTED_FIRST_SLICE_SCOPED`

用途：把「用户侧算子接口」这条线**从设计推进到可执行**，并把它切成不自相矛盾的开发阶梯。本文是启动决定与首切片范围冻结，不是设计文档——设计以 `docs/10-architecture/0006-…`（含 §13 v2 与 §13.9 回执）为准。

---

## 1. 启动依据

| 依据 | 状态 |
|---|---|
| `docs/10-architecture/0006-2026-09-12-cross-platform-operator-interface-guidance.zh-CN.md` | 外部评审两轮；§13 v2 采纳；§13.9 五问已回执 |
| 用户方向（2026-09-12） | "用户接口线是否可以总结、落文档、开始？你决定" → 父 agent 决定：**可以开始，但只开首切片** |
| 第一条硬约束 | 位级主张只对 portable 成立；"声明 = 执行"；fail-closed；性能一律 `UNGATED` |
| 证据生产线 | 单算子测试框架（`0007` / Q9）——本线所需的旋钮与 cost 证据由它产出，本线不自己造测量 |
| provider 来源 | vendor GEMM 选型（Q8）——本线只消费其结果，不重复调研 |

## 2. 这条线要交付的四件事（长期）

1. **契约注册表**：`OpDefinition` 作为唯一语义真源，`OpContract` 是它的用户/报告视图；
2. **策略与解析**：`ExecutionPolicy`（用户意图）+ `Capability/Provider`（平台事实）+ resolver（约束求交 + 偏好排序）；
3. **计划与报告**：`ExecutionPlan`（不可变、带 digest、运行期只验证不重选）+ `ExecutionReport`（可追溯）；
4. **发现与迁移**：`doctor` / `explain` / `plan` + 35 个环境变量的分阶段迁移。

## 3. 首切片 U1 的范围（**冻结**）

**只做一件事**：让 `matmul` 走通 `policy → resolver → plan → report`，并且**只有 portable 一个 provider**。

```text
交付物
├── pypto.execution 子模块（不改进 pypto.__init__，避免实验 API 污染稳定入口）
├── OpDefinition registry 骨架，仅登记 matmul（canonical opcode + contract version + digest）
├── ExecutionPolicy（最小字段集）：profile、target、provider.mode、numeric.requirement、
│   deterministic、fallback.mode、resources.max_threads
├── Resolver：规则式（**无 cost model**）——只按 numeric requirement / provider.mode /
│   capability 可用性求交与排序，tie-break 固定且可解释
├── ExecutionPlan：不可变、带 digest；记录 selected provider / numeric guarantee /
│   fallback chain / artifact 版本三元组
├── ExecutionReport：JSON，含 requested vs resolved policy、provider、精度类别、
│   dispatch/op 时间分解、`UNGATED` 状态
└── 一个薄入口：用 policy 调 matmul，走 plan 执行，返回结果 + 报告
```

**明确不做**（本切片）：命令行（doctor/explain/plan）、graph/region 作用域、cost model 与 tuning profile、环境变量迁移、复合算子（CompositeContract）、StateContract、自定义 provider 插件、P1–P8 里 matmul 以外的族。

> **U1-VIEW 扩展（2026-09-12 用户批准）**：在 U1 竖切之外增加 `layout.view_mode`
> 三态与零拷贝 alias proof（证明链 + AVX-512 proof-gated 执行器 + SVE256 capability
> 声明 + 对抗测试 + plan/report 字段 + `0006`/`0008` 文档）。它不是 matmul provider
> 的新实现，而是把 0006 §7.1 的 view 契约从"hard-off 警告"推进到可执行 proof；
> `matmul` 自身 contract 仍是 `must_not_alias`，只用于验证 `require` 的 fail-closed 分支。

## 4. 首切片的硬性验收

1. **`profile=portable` 与既有路径逐位一致**：同一输入下，结果与现有 native 路径全 buffer `sha256` 相同（不是"误差内相同"）；
2. **声明 = 执行**：plan 里写的 provider 必须就是运行时实际调用的那个；篡改 plan/provider 字段必须被拒；
3. **解析确定性**：同一 `(policy, capability digest, shape, dtype)` 必须产出**同一个 plan digest**（跑两次比对）；
4. **fail-closed**：未知 provider、未知 policy 字段、不适用字段（`POLICY_FIELD_NOT_APPLICABLE`）、非法组合 → 结构化错误，**不许静默忽略或降级**；
5. **报告可追溯**：能从报告回答"走了什么、为什么走、能否复现"，且缺字段即判失败（schema 校验）；
6. **不破坏既有入口**：`pypto.framework`、内部 driver 脚本、现有测试全绿、收集数不减少。

## 5. 非目标（防止这条线膨胀）

- 不做整网性能基准（L1/B3c 已覆盖）；不做功能回归（pytest 已覆盖）；
- 不引入新性能门槛（门槛是 B4）；不承诺任何"加速比"；
- 不把 vendor 路径包装成与 portable 位级一致；
- 不在首切片暴露 ISA/microkernel/blocking 之类内部参数（§13.4 的适用域规则）。

## 6. 与其它线的接口

| 线 | 关系 |
|---|---|
| 单算子测试框架（`0007` / Q9） | 本线**不自己测量**；provider 证据与（将来的）cost table 由它产出 |
| vendor GEMM 选型（Q8） | 本线**消费**其 provider 结论与绑定方式；U2 才接 vendor provider |
| N4（dispatch 残差） | 与 0006 的 P3（graph 作用域）强相关；**P3 必须排在 N4 决策之后**，本线不碰 |
| C8（W8A8 精度收口） | 数值类别（exact / bounded）与参照 gold 的来源；本线报告里的 `precision_class` 引用它 |
| B6（layout 原生化） | 已收口；`view-mode-require` 已把其 hard-off 视图别名改为 AVX-512 `proof_gated`，SVE256 声明 `alias_proof=unsupported`（见 `0006` §13.9 第 4 答 / §14.1 A-2） |

## 7. 任务阶梯（后续，不在本次派发）

| # | 任务 | 依赖 | 说明 |
|---|---|---|---|
| **U1** | **本切片（matmul + portable + plan/report）** | 无 | 本次启动 |
| **U1-VIEW** | **`view_mode=require` 提级（证明链 + 执行器启用 + 文档）** | U1 + B6 | 2026-09-12 用户批准提级；任务 `view-mode-require`，分支 `work/view-mode-require`。交付五项 proof、三态语义、AVX-512 proof-gated 零拷贝、SVE256 capability 声明、对抗测试、plan/report 字段与 `0006` 文档提级 |
| U2 | 第一个 vendor provider 绑定（oneDNN 或 AOCL/LPGEMM，取绑定干净的） | U1 + Q8 | 引入 `precision_class` 与库版本指纹 |
| U3 | `doctor` / `explain` / `plan` 三个发现命令 + capability snapshot | U1 | 把"事实/为什么/可重放"暴露给用户 |
| U4 | 35 个环境变量迁移 P0–P1（登记 → 兼容层 + 弃用警告） | U1 | 不破坏旧用户；影响 artifact 语义的 env 必须进 policy digest |
| U5 | graph/region 作用域 | **N4 决策之后** | 依赖图执行层的形态 |

## 8. 首切片的边界诚实声明

- 本切片**只证明机制成立**，不产生任何性能收益，也不改变现有数值行为；
- 报告里的时间分解在本机一律 `UNGATED`（12 vCPU KVM、无 cpufreq、窗口非静默）；
- `OpDefinition` 在本切片只登记 matmul，**不宣称**已覆盖 P1–P8；其余族按 §13.7 的顺序逐步接入。
