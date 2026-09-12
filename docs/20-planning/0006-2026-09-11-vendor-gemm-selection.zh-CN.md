# vendor GEMM 算子选型（各平台）

文档编号：`0006`

日期：2026-09-11（Asia/Shanghai）

状态：`DECIDED_ADOPTED_PENDING_SELECTION_STUDY`

## 0. 决策（用户 2026-09-11）

1. **采纳：GEMM 一律使用各平台最优的 vendor 算子**，不再自研 GEMM 追平 vendor。
2. **W8J 的价值主张从 BF16 转到 W8A8**。依据：目标平台鲲鹏 920B 的 BF16 算力本身偏低（用户指出约为 FP16 的一半、INT8 的四分之一）→ 在 decode 的 m=1 形状上，权重字节数的削减与整数算力优势都能吃到。
3. **做一次各 vendor 的 GEMM 算子选型**（本文即选型规范）。

## 1. 三条边界（与决策同时生效，不得省略）

| 边界 | 内容 |
|---|---|
| ① portable 路径保留为契约基准 | 我们自己的内核不删，作为 `portable` provider；**"位级一致"只能对它声称**；无 vendor 库时必须能退化运行 |
| ② vendor 路径带精度契约 + 版本指纹 | 库名/版本/编译选项写进 artifact 元数据与证据；判据用"有界误差 + 版本钉死的可复现性"；库缺失或版本不符 → 退回 portable 或 fail-closed，**禁止静默换实现** |
| ③ W8A8 是主战场 | int8 整数累加精确且顺序无关（我们已实测 K 边界 ±2147479576、0 饱和）→ vendor int8 GEMM + 我们的 epilogue 在无饱和时**应当逐位一致**；必须核对库的**累加宽度与饱和策略** |

## 2. 候选矩阵（待逐个核实的初稿）

| 平台 | 候选 | 关注点 |
|---|---|---|
| x86_64（本机 12 vCPU，**Zen4 ES 样片** family 25 model 116，AVX-512 + VNNI + AVX512-BF16，无 AVX512-FP16、无 AMX） | oneDNN v3.x（vLLM CPU 现用）、**AOCL/LPGEMM**（见 §2.1）、**ZenDNN**、**libxsmm**（JIT，小 m 友好）、MKL/oneMKL、OpenBLAS、FBGEMM/QNNPACK（torch 侧） | bf16 / int8(s8s8 vs u8s8) / 运行时 per-channel scale / int32 累加 / 输出 bf16 / 线程与亲和控制 / 小 m 开销 |

### 2.1 x86 leg 追加候选：AMD AOCL / LPGEMM（2026-09-11 实测发现）

**探测结论（父 agent 现场核实，非文档引用）**

1. **便捷路径不含 W8A8**：conda-forge `aocl-blas 5.2` 可无 root 安装，但产物是纯 BLIS（`libblis-mt.so.5.2.0`）——符号表 `aocl_gemm_*` 计数 **0**、头文件 lpgemm 计数 **0**，即**未编译 LPGEMM**。
2. **LPGEMM 是 AMD BLIS 分支的 addon**（`addon/aocl_gemm/`，编译要求 GCC ≥ 11.2 或 Clang ≥ 12），文件级证据：`aocl_gemm_s8s8s32obf16.c`、`aocl_gemm_u8s8s32os32.c`、`aocl_gemm_bf16bf16f32obf16.c`、`aocl_gemm_bf16s4f32of32.c`、`aocl_batch_gemm_*`、含 `JIT/` 子目录。
3. **导出 API 与我们契约高度对齐**（`aocl_gemm_interface_apis.h`）：
   - `AOCL_GEMM_MATMUL(int8,int8,bfloat16,int32, s8s8s32obf16)` → **s8×s8 → int32 累加 → bf16 输出**；
   - `... s8s8s32obf16_sym_quant` → **对称量化专用变体**（对应我们的 per-channel 对称方案）；
   - `AOCL_GEMM_REORDER(...)` → 权重预打包（对应我们的 pack cache）；
   - 后处理 `BIAS / SCALE / MATRIX_ADD`（最多 8 个、可排序），前置 `RELU / PRELU / GELU_TANH / GELU_ERF`。
4. **对本项目的意义**：`s8s8` 免符号补偿（VNNI 家族是 `u8s8`，补偿会改数值语义）；`_sym_quant` + `SCALE` 后处理有望直接承载我们的 per-channel/per-token scale 与 epilogue。

**因此 x86 leg 的 AOCL 获取方式（用户 2026-09-11 决定）**：**从源码构建，显式选择 Zen4 配置**（本机为 Zen4 ES 样片，不允许运行期探测去猜 Zen5/其他 arch）；conda-forge 包因不含 LPGEMM 而弃用。构建后必须核对：`_sym_quant` 的 scale 语义、累加宽度、饱和策略、与 portable 路径的逐位一致性，以及构建出的 zen4 内核确实被运行时选中（而非退回 generic）。
| aarch64 鲲鹏 920B（SVE VL=32，svebf16 + svei8mm） | **oneDNN aarch64**（ACL/ SVE 后端）、**KleidiAI** 微内核、ArmPL、华为 KML/BoostKit、OpenBLAS-aarch64 | 同左；另需确认 ArmPL/KML 在鲲鹏上的**可得性与许可**、以及能否在 A3 上无需 root 安装 |
| NVIDIA RTX 5080（sm_120） | cuBLAS / **cuBLASLt**（SCALE 模式）、CUTLASS（自定义 epilogue） | 能否表达"per-channel 权重 scale + per-token 激活 scale + bf16 输出"的 epilogue；sm_120 支持矩阵 |
| AMD gfx1036 | hipBLASLt / rocBLAS(Tensile) | **静态记录即可**（我们的 AMD 运行态仍 BLOCKED_DEVICE，不实测） |
| Ascend 910C | ACLNN（`aclnnQuantMatmulV3` 等）/ CATLASS / PTO | **只做文档级记录**：chip7 是用户的，未经许可不在 NPU 上实测 |

## 3. 评估维度（每格都要有证据，不许"文档说有"就算）

1. **功能**：dtype 覆盖（bf16 / fp16 / int8）、int8 是 `s8s8` 还是 `u8s8`（后者需符号补偿）、**累加宽度**（必须 int32，或证明不会溢出）、饱和/截断策略；
2. **粒度**：权重 per-tensor / per-channel、激活 per-tensor / per-token 的**运行时** scale 支持；zero-point；能否在同一 kernel 内完成 dequant；
3. **epilogue**：bias、scale、输出 dtype（bf16/fp32）、以及"我们的 epilogue 需要单独一趟"时的额外带宽代价；
4. **可复现**：同一版本 + 同一输入是否**逐位可重复**（实测，用同一输入跑两次比对 sha256）；
5. **工程**：是否可无 root 安装/是否进 conda；包体积；版本钉死方式（wheel hash / 库文件名 + sha256）；线程数与 CPU 亲和是否可控；
6. **性能**：在我们既有的 **12 形状集**（m=1/5 × k=1024/2048/3584 × n=32…8192）上测 **bf16 与 int8**，与 `portable` 路径同基准对比；必须包含 **小 m 与极小 n** 两端的开销表现；
7. **平台事实核对**：实测确认（而非引用）鲲鹏 920B 上 `BF16 : FP16 : INT8` 的相对算力，作为 W8J 转 W8A8 的依据。

## 4. 交付物

1. **能力矩阵表**（候选 × 维度），每格注明证据来源（实测 / 官方文档 URL / 代码级阅读）；
2. **实测表**：x86 + aarch64（+ NVIDIA 若可行）× {bf16, int8} × 12 形状，含离散度与线程/进程设置；
3. **推荐阶梯**：每个平台给出 primary / secondary / fallback 三层，并写清触发条件；
4. **集成契约**：artifact 元数据新增字段草案（`gemm_provider` / `library` / `library_version` / `tolerance_class` / `accum_width` / `scale_granularity`）+ fail-closed 规则；
5. **W8A8 位级一致性探针结论**：vendor int8 GEMM + 我们的 epilogue 是否与我们现有实现逐位一致（含饱和边界用例）。

## 5. 边界与诚实声明

- 本选型**不改变**我们对外的位级主张：位级一致性仍只对 `portable` 路径声称；vendor 路径的可复现性来自**版本钉死**，不是来自"数学上必然"。
- AMD 与 Ascend 两格若只有文档证据，必须显式标注"未实测"，不得混进结论表当作已核实。
- 选型的性能数字必须与既有 W8J 基准同口径（同线程数、同 warmup、同取数方式），否则不可比。

---

## 8. 实测结果（2026-09-12，Q8 交付；本机 x86 Zen4 ES）

### 8.1 LPGEMM 构建与"显式 Zen4"验证

```text
源码      github.com/amd/blis tag 5.3.2 = 25cad99a6840855ade0a49871197f48ee0e1d317
构建      经典 ./configure（CMake 需 Fortran，本机无）+ 显式 zen4 + --disable-blis-arch-type
          + --int-size=64 --blas-int-size=32 + OpenMP
产物      libblis-mt.so.5.3.2，sha256 c7d74a3129c80129b0b560868bdbf7ad822c3777b96f64a0ba1aa7bee3063dad（17,498,040 B）
符号      27 个 aocl_gemm_* 导出；1832 条 vpdpbusd；21 个 LPGEMM zen4 kernel .o
运行期    bli_arch_query_id = 10 / bli_arch_string = "zen4"   ← 非 generic、非运行期猜测
```

### 8.2 两条路的位级结论（**本选型最重要的结果**）

| | 路 A：`s8s8s32os32`（int32 出）+ 我们自己的 epilogue | 路 B：fused `s8s8s32obf16_sym_quant` |
|---|---|---|
| 机制 | 块内（KC=2048）精确 int32，**块间 binary32 链式累加** | 单组时与我们的 epilogue 结构等价；f32 顺序为两次乘 + 组间累加 |
| 位级结论 | **不是无条件逐位一致** | 10862 元素 **0 mismatch**，但**不同构** |
| 分类 | `int32_f32block_bounded`（`bitwise` 仅当 provider=portable/自有 kernel 或 **k ≤ 1024**） | `deterministic_bounded` / `fp32_1ulp_epilogue` |
| 边界 | `K % 4 != 0` 时 reorder API 返回缓冲区大小 0 → **fail-closed** | 同左 |

偏差是一组**实测观测，不是上界**（三次修订见 ERR-0010）：`K=512/1024 → 0`（逐位）｜
`2048 → 1`｜`3584 → 4`｜`5099 → 8`｜`16696 → 75`｜`133144 → 1067`（seeds 1–12000 扫描）｜
**`133145 → 回绕到 −2147471616 且无报错`**。

U2b 及独立验收的观测口径：seeds 1–8 × {`uniform[100,127]`（A/B 独立生成器）、all-127、
`uniform[-127,127]`} × 线程 {1,2,6}（m=2,n=6）只给到 `16696→56 / 133144→457`；同一生成器扩到
seeds 1–200 后为 `16696→75（seed 147）/ 133144→841（seed 115）`，验收方另一 seed 族到
`133144→924（seed 48）`，seeds 1–1000 到 `133144→966（seed 428）`（503/1000 超过第一轮声明的
457），r2 复检扩到 seeds 1001–6000 得 `133144→1067（seed 1591，threads 1/2/6 复核）`、
seeds 6001–12000 得 `133144→1043（seed 6323）`且 `16696` 保持 75，648 用例扫描到
5099→7、16696→62、133144→678。**偏差同时依赖 K 与码值分布**
（KC=2048 的 int32 partial 在 f32 上链式累加），且 pinned zen4 内核还包含 s8→u8 转换、
int32 列和补偿、每 KC block 的 f32 转换/写回与 m/n fringe 分支，"块间纯 f32 链"模型只 25/31
用例逐位吻合。因此本选型**不提供可证明的解析上界**：vendor manifest 声明
`deviation_bound_kind=measured_snapshot`、`deviation_bound_available=false`，只给
`measured_deviation_by_k` + `measurement_scope`（值随新证据上抬，目前 133144→1067）；
需要"有证明上界"的消费者必须用
`numeric.require_proven_deviation_bound=true`，否则会拿到观测而非界。契约本身未放宽：仍是
`deterministic_bounded`、K≤1024 逐位、K≥133145 在任何 kernel 调用前 fail-closed、
`-128`/`True`/`False` 在适配层拒绝。以上观测/（若有）上界**只约束 `|AOCL_int32 - 精确 int32 参照|`**，
不覆盖反量化/epilogue 之后的输出误差，也不覆盖 scope 之外的 shape/线程/其它构建。

实现方模型（块内精确 int32 → 块间 f32 → mod 2³²）在 31 用例中 25 个逐位吻合（含回绕），6 个 mixed 用例差 2–5 → **机制已定位、细节未钉死**（如实登记为模型解释力边界）。

**三条必须由我们守的硬约束**（各带可复现探针）：

1. **`-128`**：LPGEMM 静默接受（算出 −512、无错误），我们的 kernel 返回 −1 → 码值校验必须在适配层；
2. **K 上界**：`accum_k_limit = 133144`，越界**回绕无保护** → K 上限必须继续由我们守；
3. **per-row SCALE 被 parser 明文拒绝**（stderr 报错、输出缓冲区未动）→ 这就是"路 A + 自己 epilogue（或单组 sym_quant）"的设计理由。

### 8.3 12 形状四路实测（**UNGATED**，同 W8J 口径：affinity 0–5、6 线程、rounds=5、iters 200/100、seed 20260912）

几何平均相对 oneDNN bf16 基线：

```text
aocl_int8_symquant   0.57x   ← 最快
aocl_int8_packed     0.61x
aocl_bf16_packed     0.76x
ours_bf16            6.72x
ours_int8_qmatmul   24.5x
ours_epilogue_only   0.16x   （仅 dequant epilogue，非完整 GEMM，不可与上面直接比）
```

**"位级的价格"：2.5×（m=1,n=32）… 533×（m=5,n=8192）。**

⚠ 质量声明：`m=5,k=2048,n=1024` 的 AOCL int8 三列被窗口噪声污染（矩阵 3.34–3.42 ms vs 两次复测 0.0139/0.018 ms），已标注；其它单元格跨次波动 1.3–1.6×。**本矩阵不得用于门槛**。

### 8.4 推荐阶梯（x86）

```text
W8A8 默认 = 我们自己的 kernel（位级、单线程）
     AOCL  = 显式 opt-in 的提速档，登记 tolerance_class=int32_f32block_bounded + 该形状实测界
     理由：W8A8 契约的对外主张就是 int32 逐位；默认必须位级，否则"整数域无误差"会在 K≥2048 静默失效
bf16 默认 = oneDNN；AOCL bf16 第二（0.76x）；自有 bridge 兜底
```

### 8.5 非 x86 的平台结论（**全部为文档级，未实测**）

| 平台 | 结论 |
|---|---|
| **Ascend** | `aclnnQuantMatmulV4/V5` 是文档上最贴合我们契约的候选（per-channel + per-token 运行时 scale、int32 bias 在 x1@x2 后、bf16 输出）；**NPU 属用户，未实测** |
| **aarch64（实测，2026-09-12 Q8b）** | **KleidiAI（pin `c1c9e876` / v1.31.0）**：`qai8dxp`（per-row 动态）× `qsi8cxp`（per-channel）原生路径；**int32 累加精确**（K=256–3584 反推整数 0 mismatch）；用手工构造的 packed-LHS 可达**逐位精确**；**陷阱**：RHS pack 的 `params.lhs_zero_point` 必须是**flag `1`**，否则 per-row 零偏被静默忽略（rel err 16–139）；K 精确上界 **133144**（越界回绕、`-128` 静默接受，均须我们守）；K%8≠0 只因 Release 关掉 `KAI_ASSERT` 才"能跑"，**必须我们 fail-closed**；12 形状 1×8 线程：m=1 0.0019–0.0395 ms（34–583 GOP/s）、m=5 0.0043–0.1022 ms（77–821 GOP/s）；对照 portable SVE256 单线程慢 3.3–300×。**oneDNN 3.12 aarch64 实测不支持 per-token src scale**（`attr.set_scales_mask(src, 1<<0)` → `status=3 unimplemented`），且该 conda 构建**无 ACL**。**Primary 建议 = KleidiAI**（secondary = 我们的 portable SVE256 契约基准；oneDNN 仅限 per-tensor/bf16）。集成二选一：(a) 公开 API（融合动态 per-token 量化、f32 出、`deterministic_bounded` ≤1 ulp、K 上界我们守）；(b) 私有 packed-LHS（对我们自己的码值**逐位**，但 ABI 未公开 → 需版本钉死 + 启动自检） |
| **NVIDIA sm_120（实测，2026-09-12 Q8b）** | 最终口径：**融合 scale/epilogue 的 vendor int8 不可用**（cuBLASLt：int8→s32 `status=7`、int8→bf16 与任何 scale/bias 组合 `status=15`；FP8+scale 与 bf16 对照 `status=0`，证明是 int8 特有）。**但经典 `cublasGemmEx int8→s32`（`CUBLAS_COMPUTE_32I`）可用且数值精确**（A=B=1、K=3584 → 全 3584）。因此**两趟式 vendor int8 GEMM（int32 出）+ 我们自己的 epilogue 可行**（不融合、m=1 时可能被 launch 开销主导）。C6 的 dp4a 仍是**唯一融合的 per-token×per-channel 路径**。⚠ **新红旗：C6 目前完全没有计时证据** |
| **aarch64** | KleidiAI（`qai8dxp×qsi8cxp`，int32 累加有 `smmla` 源码证据、per-row×per-channel 原生、纯 C 无线程）**最可行，但未实测**；oneDNN-ACL 的 per-token 是否被真正消费未证；**ArmPL 在 EULA 澄清前不得列入**；KML 需注册同意 EULA；OpenBLAS-aarch64 无 int8 GEMM |
| **NVIDIA sm_120** | **红灯（已由 Q4 实测确认，措辞已修正）**：cuBLASLt 的 **`A/B_SCALE` 指针** → `INVALID_VALUE(7)`（scaleType 32I/32F 均如此）；**int8 + scale → BF16 或 +bias epilogue** → `NOT_SUPPORTED(15)`。
**修正**：不是"vendor int8 不可用"，而是**"int8 + scale 融合"不可用**——**plain int8 GEMM（`GemmEx` 32I、Lt 不带 scale）是 SUPPORTED 的**。
含义：C6 的自有 dp4a 内核仍是该平台**唯一能一次做完 int8 + 我们 epilogue** 的路径；若要吃 vendor int8，必须自己拆成"GEMM + 单独 epilogue"两趟。证据：`_meta/pypto-x/c8-axis-b-probe/` |
| **AMD gfx1036** | 文档级不可达（ROCm 支持矩阵无 gfx1036、Tensile 直接 unsupported） |

### 8.6 本轮的边界与红旗

- 结论**只限定于 AOCL-BLIS 5.3.2 / 显式 Zen4 / 已测 K 值**；其它 tag、其它 config、其它库均未测；
- `libxsmm` 按"**可得但需独立 JIT 绑定任务**"上报（2.x 只暴露 `libxsmm_dispatch_gemm` JIT 入口），**未凑数**；
- 锁竞争导致构建窗口丢过一次（报告步骤 SIGPIPE），已用独立报告生成器修复并复现；
- 矩阵有一个污染单元格 + 跨次波动 → **不得用于门槛**；性能数字一律 `UNGATED`。
