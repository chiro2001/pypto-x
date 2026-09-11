# C8 执行队列：W8A8 整网精度收口（参照 gold + band + 独立栈对齐）

文档编号：`0005`

日期：2026-09-11（Asia/Shanghai）

状态：`QUEUED_APPROVED_IN_PRINCIPLE`

用途：把 C8 从「口径已定」推到「可执行、可验收」的队列。本文自包含；数字与决策都来自 `configs/development_lock.yaml` 与用户 2026-09-11 的裁定。

---

## 1. 目标（一句话）

把**「我们的实现对不对」**和**「量化本身损失多大」**拆开：用**同方案、独立执行路径**的参照 gold 判实现保真度（轴 A），用**独立栈消费我们的 int8 产物**判部署一致（轴 B）。

## 2. 已决事项（用户 2026-09-11）

| 事项 | 决定 |
|---|---|
| C8 L4 口径 | 给 W8A8 建**自己的参照 gold + band**（同方法论于 BF16 gold）；对 BF16 gold 的 `2.69/2.76`、`cos 0.996` 降级为参考信息 |
| 参照平台 | **轴 A 参照留在 x86**，与既有 gold 同口径（`transformers 5.17.0` / `torch 2.14.0+cpu`）；0.8B + 十几 token 不是重活，换平台只引入第二个变量 |
| A3 定位 | **受测方**（我们的 SVE256 后端）+ **轴 B 第二平台** + **L5/L6 长序列重活**。绝不用我们自己的后端当参照 |
| A3 资源 | 用户授予**后半机**：`cpus 320–639` = NUMA node4–7 = **320 逻辑 / 160 物理核**，内存池 1 TB+（实测拓扑：640 逻辑 CPU、4 socket×80 core×2 HT、8 NUMA×80、2013 GiB 总 / 1814 GiB available） |
| A3 红线 | 不碰 chip7/NPU、不碰他人容器与进程、只写 `~/pypto-x-a3/**`、依赖装独立 conda env、大文件在 A3 侧下载、脚本以文件传输 |
| 并行策略 | 我们自己的 runtime 是 **Python 派发** → 靠**多进程铺开用例**扩展；参照/独立栈（torch/OpenMP）才靠多线程扩展 |

## 3. 判据默认（父 agent 建议值；用户未反对即按此执行，可随时覆盖）

```text
e1 判据     与 BF16 同款：逐行 argmax 硬门槛 + near-tie 例外 + max_abs ≤ 2×band + cos ≥ band_cosine − 1e-4
e2 C7 逐层  按需：C8 整网通过则 C7 降级为异常定位工具；C8 不过才逐层二分
e3 门禁层级  C8 先只做 L4（logits）；L5 用现有 5-step 链做弱化版并显式标注覆盖不足；L6 需用户定语料与阈值（契约 D5）
e4 后端覆盖 轴 A：AVX-512 + SVE256 先行；CUDA 后补。轴 B 独立栈另算
e5 产物归属  gold 数据进 _meta，摘要与 digest 进 lock；不进 patches/（补丁只含代码）
```

## 4. 队列（按依赖排序；平台无硬性并发上限，见 ERR-0005——真正的约束是资源锁）

| # | task | 依赖 | 资源 | 交付 | 验收要点 |
|---|---|---|---|---|---|
| **Q1** | `a3-c8-prep` | 无 | A3 CPU（320–639） | 独立 conda env `pypto-x-c8`（aarch64、torch CPU、`transformers` 钉同口径）；容器自带 `transformers 5.14.1` 能否加载 Qwen3.5 的探测结论；线程/进程扩展性基线表；`~/models` 权重 hash 复核；磁盘/内存余量 | env 可 import + 能 load 模型（若版本可选）+ 基线表可复跑 + 无残留进程；chip7 零触碰 |
| **Q2** | `c8-w8a8-reference-gold` | 无（等 B3c 验收让出本机计时窗口） | 本机（heavy 锁） | `tools/qwen35_reference/run_reference.py` 增加 `w8a8` fake-quant 变体；3 prompt 的 W8A8 gold；band 表；scale 一致性测试 | 量化范围逐条复刻契约 D1/D3/D4/D7/D8/D9；scale 同源（取我们 packer 产物）；band = `|fakequant − fp32|` 统计；不 import 我们任何实现模块 |
| **Q3** | `c8-l4-avx512` | Q2 | 本机 | 我们的 AVX-512 W8A8 整网结果 vs W8A8 gold 的 L4 判定（按 §3 判据）+ 与 BF16-gold 数字的对照说明 | 独立取数；near-tie 单列计数；严格 argmax 与 near-tie 必须同时可见 |
| **Q4** | `c8-axis-b-probe` | Q1（A3 侧） | 本机 / A3 / GamePC | 决策表：能否把**我们的** int8 权重+scale 喂进独立栈（x86 vLLM-CPU 0.29.0 / A3 vLLM / GamePC vLLM CUDA sm_120）；代价与差异来源分类；可行则做 1 个 prompt 的最小实证 | 明确"能/不能/要写多少适配"；若不可行要给出阻塞点证据；不下载新权重 |
| **Q5** | `c8-l4-sve256` | Q1、Q2 | A3（多进程） | 我们的 SVE256 后端整网 L4 判定 | 与 Q3 同判据；A3 只跑 CPU |
| **Q6** | `c8-l5-l6` | Q1、Q2 + **用户定语料与阈值** | A3（多线程） | L5 弱化版（现有 token 链）+ L6（若语料/阈值定下） | L5 覆盖不足必须显式标注；L6 阈值未定前不启动 |
| **Q7** | `b6-verify` | B6 实现交付并合入 integration | 本机 | B6 的独立验收（逐位一致、声明=执行、回归、UNGATED 性能） | 按冻结流水线：实现 → cherry-pick → 独立 verify worktree |

## 5. 委派规则

1. 并发：平台无硬性上限（ERR-0005）。按依赖顺序派发，**依赖不满足的跳过**（例如 Q3 必须等 Q2）；重任务由资源锁串行化，同一把锁的等待者不超过 2 个。
2. Q1 不依赖任何本机资源，**第一个空槽优先给它**。
3. 每个任务沿用统一启动协议（`docs/WORKTREE_AGENT_PLAN.zh-CN.md`）：一次 smoke、heavy 经 `run_local_heavy.sh`、返回 75/69 等待重试、不改他人 worktree、不下载权重。
4. 每个任务的验收按冻结流水线执行：实现 → cherry-pick 进 integration → 独立 `verify/<phase>` worktree → 更新 lock/ERRATA → patches → **用户批准后**才 push。

## 6. 边界（先说清楚）

- **C8 通过 ≠ W8A8 可生产**：它只证明"实现忠实于契约方案"；量化质量（相对 fp32 的真实损失）要靠 L6/perplexity 或独立评测。
- **band 是参照自身的量化噪声底**，不是行业标准阈值；对外只能声称"与同方案参照一致"，不能声称"精度达标"。
- 轴 B 的差异来源包含内核实现差异（累加顺序/epilogue 舍入/饱和策略），因此它的判据必须与轴 A 分开表述，**不得混算**。

---

## 9. 队列现状（2026-09-12 02:15 CST 快照；权威状态以 `configs/development_lock.yaml` 为准）

| # | 任务 | 状态 | 结果 / 阻塞 |
|---|---|---|---|
| Q1 | `a3-c8-prep` | ✅ 完成 | 独立 conda env `~/pypto-x-a3/envs/pypto-x-c8`（transformers 5.17.0 与 x86 gold 精确同版本）；两个加载探测 LOAD_OK；并行度基线（1 case = 1 进程 × 8 线程，16–20 并发为上限；HT 负收益）；**红旗：aarch64 无 `torch+cpu` wheel → 第二参照口径**；容器 cpuset 仅 560-639 |
| Q2 | `c8-w8a8-reference-gold` | ✅ 完成 | W8A8 参照 gold + band；**scale 交叉校验 150/150 张量、399,360 元素逐位一致（max ULP=0）**；gold 聚合 digest `22e754cf…`；两变体（`w8a8` 主参照 / `w8a8_f32ref` 诊断） |
| Q3 | `c8-l4-avx512` | 🔄 在途（持锁） | 主判定（prefill 口径）en/zh 全 PASS（严格 argmax 9/9、12/12、8/8，near-tie 0）；**zh-default 的 decode 扩展 FAIL（decode 口径下限 0.99082，实测 0.98477）**；chat 段收尾中 |
| Q4 | `c8-axis-b-probe` | ✅ 完成 | 决策表：x86 vLLM-CPU **可行**（checkpoint 重打包 200–300 行、无运行时胶水）；A3 容器不可行；A3 host 二进制可得/运行未证；**sm_120：cuBLASLt 的 int8（含融合 scale）不可用**；最硬证据=激活码一致时 vLLM(oneDNN) 与我们**逐位一致** |
| Q5 | `c8-l4-sve256` | ⛔ 阻塞（已收口） | W8A8 整网在 SVE256 **lowering 即被拒**（int8 转置 150/187 个、474/717 MiB/forward）；副产品=**A3 bf16 整网普查 PASS**（native 91–93%，RSS 峰值 68–81 GiB）→ 直接暴露 SVE256 的 int8 缺口与**不释放中间值**两个问题 |
| Q7 | `b6-verify` | ✅ 完成 | B6 独立验收 PASS_WITH_BOUNDARIES（101/101 逐位、声明=执行、fail-closed、collect +85/0 删减） |
| Q8 | `vendor-gemm-survey` | ✅ 完成 | AOCL/LPGEMM 显式 Zen4 构建 + **位级包络**（K≤1024 精确、2048≤1、3584≤4、5099≤6、133145 回绕）+ 12 形状四路实测 + **"位级的价格" 2.5–533×** + 推荐阶梯（W8A8 默认自有 kernel，AOCL 为 opt-in 提速档） |
| Q8b | `vendor-gemm-survey-2` | 🔄 在途 | aarch64：**KleidiAI 原生 per-token×per-channel 且 int32 精确**（含 `lhs_zero_point=1` 开关陷阱）；sm_120：**GemmEx int8→s32 可用且精确**，cuBLASLt 融合不可用；待补 oneDNN-aarch64 与 SVE256 对照 |
| Q9 | `op-bench-framework` | ⏳ 排队 | **依赖 U1 的 `OpDefinition` registry**（两者必须共用同一个 registry）；U1 落定后派发 |
| Q10 | `sve256-int8-layout` | 🔄 在途 | 采纳路线 **(a)**（native int8 layout，对标 B6），版本链 wire v3→4 / index v4→5 / payload 1→2 / lowering 8→9；完成后 **Q5 重跑拿 SVE256 的 L4 判定** |
| — | `sve256-liveness-release` | 🔄 在途 | 由 Q5 的 RSS 68–81 GiB 直接派生：SVE256 runtime 从不释放 SSA 中间值；头号指标=A3 RSS 前后对比 |
| — | `w8a8-v2-transpose-elimination` | 🔄 M1 在途 | 用户已批准立项（`docs/20-planning/0009-…`）；M1 只做设计/影响面与**重基线可执行清单**；**M4 合入必须等本批收口** |
| — | `launch-artifact-cache` | 🔄 在途 | 由 N4.0/ERR-0006 派生：消掉每次 launch 对同一程序做两遍 lower + 全量 plan 深比较（≈9.8 s prefill） |

### 9.1 Q5 重跑的排期与资源预算（前置条件满足后执行）

```text
前置：Q10（int8 layout 原生化）交付并合入  +  sve256-liveness-release 交付并合入
方式：6 个 prompt×配置 并行（1 case = 1 进程 × 8 线程 × 8 个不重叠物理核；A3 上 ≤20 并发）
      预fill 单 prompt 目前 242–1006 s（bf16 口径），并行后墙钟约 20–40 min 量级
内存：修复前单进程峰值 68–81 GiB；修复后预期显著下降 —— 以 sve256-liveness-release 的实测为准
判据：与 Q3 完全一致（主判定用 prefill 口径 band；decode 扩展用 decode 口径 band 并报参照自洽性）
```

### 9.2 已登记的过程教训（本队列产生）

1. **锁要分段**：单任务长时间持锁会饿死排队者（Q3 已改为按段取锁）；
2. **"改动小"不等于"不用跑全量"**：KF-1 就是这样漏掉的（见 ERR-0005 之后的流程补充：cherry-pick 后必须立刻跑一次全量 collect + 失败集合对比）；
3. **残差不是归因**：B3c 的"41–49% dispatch"实为每-launch setup（ERR-0006）——**凡是"减去法"得到的量，标注时都要写明它是残差**；
4. **"第二次实测"才能定口径**：sm_120 的 vendor int8 结论被实测收窄了两次（"不可用"→"int8+scale 融合不可用"→"cuBLASLt int8 不可用，但 GemmEx int8→s32 可用且精确"）。
