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

## 4. 队列（按依赖排序；并发上限 3）

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

1. 并发上限 3：当前占用 = B3c 验收 + C6 验收 + B6 实现；**先出槽先发**，按 Q1 → Q2 → Q3/Q4 → Q5 → Q6 的顺序，但**依赖不满足的跳过**（例如 Q3 必须等 Q2）。
2. Q1 不依赖任何本机资源，**第一个空槽优先给它**。
3. 每个任务沿用统一启动协议（`docs/WORKTREE_AGENT_PLAN.zh-CN.md`）：一次 smoke、heavy 经 `run_local_heavy.sh`、返回 75/69 等待重试、不改他人 worktree、不下载权重。
4. 每个任务的验收按冻结流水线执行：实现 → cherry-pick 进 integration → 独立 `verify/<phase>` worktree → 更新 lock/ERRATA → patches → **用户批准后**才 push。

## 6. 边界（先说清楚）

- **C8 通过 ≠ W8A8 可生产**：它只证明"实现忠实于契约方案"；量化质量（相对 fp32 的真实损失）要靠 L6/perplexity 或独立评测。
- **band 是参照自身的量化噪声底**，不是行业标准阈值；对外只能声称"与同方案参照一致"，不能声称"精度达标"。
- 轴 B 的差异来源包含内核实现差异（累加顺序/epilogue 舍入/饱和策略），因此它的判据必须与轴 A 分开表述，**不得混算**。
