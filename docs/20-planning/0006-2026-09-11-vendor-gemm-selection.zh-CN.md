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

**因此 x86 leg 必须包含两条获取路径**：(a) AMD 官方 AOCL release；(b) 自行构建 `amd/blis` 的 addon（本地需确认 GCC ≥ 11.2）。并必须核对：`_sym_quant` 的 scale 语义、累加宽度、饱和策略、以及与 portable 路径的逐位一致性。
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
