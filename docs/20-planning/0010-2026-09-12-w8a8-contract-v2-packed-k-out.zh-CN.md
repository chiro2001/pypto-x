# W8A8 契约 v2（消除源头 int8 转置）M1 设计稿：c1 = packed `[K,out]`

- 任务：`w8a8-v2-transpose-elimination-m1`（立项书 `docs/20-planning/0009-2026-09-12-w8a8-contract-v2-transpose-elimination.zh-CN.md`）
- worktree：`/home/chiro/projects/pypto/worktrees/pypto-x/w8a8-v2-transpose-elimination-m1`
- branch：`work/w8a8-v2-transpose-elimination-m1`
- base（实际）：`port/pypto-x-integration @ acece0a9264d5b4ed1496363d93ba01222d7c7c1`（立项书记载的 `8ef3559` 是控制仓 main 的立项提交；实现 worktree 的实际 base 为 `acece0a92`，比 C8/Q5 证据基线 `c2ec98f3c` 多 27 个 tools/perf 与 1 个 AVX-512 校验去重提交）
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/w8a8-v2-transpose-elimination-m1/`
- 本阶段性质：**设计/分析，只读代码，不跑 heavy，不改实现**。所有数字要么来自已存在的实测证据，要么来自本次只读 introspection（`raw/m1_introspect.py`，无权重/无编译/无 lowering）。

---

## 0. 结论摘要（给父 agent 的 6 行）

1. **推荐 c1**（打包期直接产出 `[K,out]`，图里删除 int8 transpose）。它不是"因为更简单"，而是因为 **c2 在收益上不占优、在成本上严格更大**：c1/c2 都必须让 int8 权重成为 qmatmul 的直接操作数（因此都要处理"参数直接喂 kernel"与 SVE256 int8 ingestion），但 c2 还要改 4 个后端 kernel + 标量 golden + R5 + 每后端 plan/artifact 语义。
2. **四后端 qmatmul kernel 全部不用改**（AVX2 / AVX-512 / SVE256 / CUDA dp4a）。四者都只吃"`[K,N]` 连续 row-major、行 stride=N、无 stride 入参"的右操作数；当前图里它们吃到的 transpose 输出恰好就是同一字节镜像（AVX2/AVX-512 的 transpose 是普通 row-major copy；CUDA/SVE 的 qmatmul 从未与图 transpose 同时跑通过）。
3. **但 c1 不是"零改动"**：必须补两处 non-kernel 通路——(a) **SVE256 runtime 的 int8 参数 ingestion**（现状 memoryview format='B' 会被按无符号读 + `_cast_value` 严格范围检查 → 任何负码直接失败）；(b) **CUDA 的 W8A8 graph executor 本来就不存在**（C6 只有 per-op 入口，只接受 Python list），c1 只保证 kernel 不改，整网 CUDA 是否要新增 region-direct 编排是 M2 的范围决策。
4. **契约 v2 的差异**：int8 region shape `[out,in]` → `[in,out]`；量化方案（per-output-channel、axis=0、RNE、[-127,127]）与 scale `[out]` 全不变；`quant` 映射新增 `storage_layout="k_out_rowmajor"`；`PACKED_LAYOUT_VERSION` 2→3、`BINDING_SCHEMA_VERSION` 2→3；**region 数不变**（518 / 554 / 520 / 556，556 = 321+187+48 的构成不变）；opcode 语义不变。
5. **图 digest**：所有 W8A8 图（任意 T/P、任一 scope/lm_head 组合）都变（删 150/186/151/187 个 int8 transpose + int8 参数 shape 翻转）；**BF16 图 digest 不变**（`456ab519…d786e` / `66dd4077…` 继续有效），因此 BF16 累积回归、B6 L1、CUDA C1/C2 等 **不进入重基线**。
6. **重基线**：14 条，其中必须重跑的图级重段 6 条；预计 **local heavy ≈ 2.2–2.6 h + A3 ≈ 1.5–4 h（SVE256 待 M2 单段实测后冻结）**，合计 ≈ 4–7 h 机时（不含 M2/M3 的工作量）。
7. **vendor 边界（父 agent 已裁决不共用 storage）**：aarch64 oneDNN+ACL 要 `[K,N]`+`stride(0)==1`（= v1 `[out,K]` 行主序的转置视图，与 c1 的 N 连续布局相反），KleidiAI 要私有 packed 权重布局（scale 打包进 RHS）；vendor 各自在 ingestion 一次性 repack 或直接吃源张量，c1 的"0 搬运"主张只覆盖 portable 四后端。c2 保留为"将来要求共用 storage"时的触发式方案（§3.4/§5⑥）。

---

## 1. 问题 1：当前 packed 权重到底是什么、谁产出、binding 怎么喂给图参数

### 1.1 磁盘上没有 packed 权重文件；"packed layout" 是 in-memory 契约 + 一份 metadata-only JSON

- 量化权重/codes/scales 由 **offline ingestion 在内存中生成**，不落盘：
  - `python/pypto/portable/qwen35_w8a8_weight.py:149 quantize_weight_per_output_channel` 接受 checkpoint 的 `[out,in]` 浮点字节，产出 `codes`（`[out,in]` row-major int8）与 `scales`（`[out]` fp32）；`:222` `codes[row_start*columns:row_stop*columns] = code_flat.tobytes()` 明确是 row-major `[out,in]`。
  - `python/pypto/portable/qwen35_w8a8_weight.py:251 quantize_weight_from_view` 是 driver 用的入口。
  - 真实权重 driver `python/tests/ut/pypto_x/qwen35_weighted_execution_driver.py:721 w8a8_real_parameters` 逐参数量化并在内存中建 `BufferBinding`（`:766-772`），返回字段明确 `"weight_bytes_written_to_disk": False`（`:830`）。
  - 独立复现方 `tools/qwen35_reference/packer_side_a.py`（docstring 第 1-10 行）同样只在内存中复现 codes/scales，且是唯一 import `pypto` 的进程。
- `python/pypto/portable/qwen35_weight_layout.py` 的 `build_weight_layout` 只把 packed region 的 **storage-relative offset** 与 safetensors 的 **file offset** 合并成 JSON（docstring "只读 header/offset，不读取权重字节"），它是审计/校验用的 metadata-only layout，不是权重文件。
- 结论：v1 的"磁盘布局"只有 checkpoint 的 BF16 `[out,in]`；"packed 布局"是"如果调用方要按 packed layout 落盘/绑定，字节必须长什么样"的契约。这也决定了 c1 的迁移成本 = 重跑证据，**不是数据迁移**。

### 1.2 in-memory 的 packed 布局（v1，当前实现）

- 每个量化权重 2 个 region（契约 0002 §3.1）：`<param>.int8` dtype int8、shape `[out,in]`、nbytes=out*in；`<param>.scale` dtype float32、shape `[out]`、nbytes=4*out；两者 alignment=64。
- 实际 region 由 `build_binding_schema` 从 Core 参数类型推导（`runtime_binding.py:1086`，graph digest 在 `:1387` 计算），再由 `build_packed_layout` 计算 offset（`:1467-1550`；量化条目的 region role 变成 `quant_weight`，`:1492`）。
- 由本次 introspection 实测（T=5,P=0,shape_profile=model）：

| 配置 | 图 digest | ops | int8 transpose | int8 搬运字节 | regions | 构成 |
|---|---|---:|---:|---:|---:|---|
| BF16 | `456ab519…d786e` | 6728 | 0 | 0 | 368 | 320 param + 48 state |
| W8A8 default | `07c1e4f2…` | 6728 | 150 | 497,025,024 (474.0 MiB) | 518 | 170 param + 150 quant_weight + 150 quant_scale + 48 state |
| W8A8 all | `be37f12e…` | 6728 | 186 | 497,614,848 (474.6 MiB) | 554 | 134 + 186 + 186 + 48 |
| W8A8 default+lm_head | `dc7126a5…` | 6730 | 151 | 751,304,704 (716.5 MiB) | 520 | 170 + 151 + 151 + 48 |
| W8A8 all+lm_head（=Q5 的 "full"） | `150e3725…` | 6730 | 187 | 751,894,528 (717.1 MiB) | **556** | 134 + 187 + 187 + 48 = **321+187+48** |

  - 与 Q5 brief `_meta/pypto-x/c8-l4-sve256/brief.zh-CN.md` §8 的 474.0/717.1 MiB 完全一致；region 构成与 ERR-0003 修正后的 556 一致。
  - int8 region sample（default）：`layer_00_mlp_gate_proj_weight` shape `[3584,1024]` offset 508,563,456 —— **offset 在 c1 下不会变**（region 顺序与 nbytes 不变），只有 shape 语义翻转。

### 1.3 binding 怎么把它喂给图参数

1. `build_binding_schema(program)` 按 Core 函数参数顺序建 `BindingEntry`：量化权重 entry 的 `shape` 直接来自 Core ValueRef（`_entry_from_ref`，`runtime_binding.py:736-746`），并带上 `quant` 映射（9 个冻结键，`:44-54`）；scale 是独立的 `quant_scale` 角色 entry（`:37-38`）。
2. `packed_layout.region_for(name)` 给出 `(storage_id, offset, shape, nbytes)`（`:1424-1456`）；`bind_packed_region`（`:1552-1575`）把调用方 buffer 的 `[offset, offset+nbytes)` 包成只读 `BufferBinding`。
3. `assemble_launch_request`（`:1674-1727`）先做 artifact `program_digest == schema.graph_digest` 检查（`:1701-1706`），再按 `schema.inputs` 顺序生成 `TensorDesc`（`ordered_input_descriptors`，`:1633-1662`）。
4. 后端 launch 把 `TensorDesc.handle` 变成 kernel 指针（AVX2 路径见 `backends/cpu/x86/avx2.py:358-415`；AVX-512 复用同一 helper，`avx512.py:156-158`）。**kernel 只看 shape/dtype/扁平指针，不看生产者**：graph 参数与中间结果进入同一个 environment 表（`avx2.py:1228-1236` vs `:1572-1590`）。
5. 当前 W8A8 图里，qmatmul 的右操作数是 **transpose 中间结果**（`qwen35.py:490-495`），不是参数；参数先被 transpose 读一次。c1 只是把这一步换成"qmatmul 直接读参数 region"，字节内容完全相同。

---

## 2. 问题 2：c1 下四后端 qmatmul kernel 逐项核实

### 2.1 统一结论

| 后端 | qmatmul kernel 右操作数契约 | c1 下是否需要改 |
|---|---|---|
| AVX2（widening `vpmaddwd`） | `[K,N]` 连续 row-major，行 stride **硬编码 n**；只用 1/8 字节 unaligned load | **不用改** |
| AVX-512（VNNI `vpdpbusd` + s8 符号补偿） | 同上；16 列分块、4 行 K 分块，全部 `_mm_loadu_si128` | **不用改** |
| SVE256（`sunpklo`+`mla`） | 同上；每块 8 列、predicate 尾块、`svld1_s8` unaligned | **不用改** |
| CUDA（`dp4a.s32.s32`） | 同上；行 stride=n 是编译期常量，只有 1 字节 `ld.global.u8` | **不用改** |

共同点（四后端逐一取证见 `raw/backend-*.md` 与本节行号）：

- 函数签名只有 `(left, right, output, m, k, n)`，**没有** stride/layout/producer/batch 参数；右操作数唯一寻址形式是 `right[inner*n + column]`（或等价 batch 偏移 `batch*k*n + inner*n + column`）。
- 没有任何 aligned intrinsic 作用在右操作数上：AVX2 `_mm_loadl_epi64`（`__m128i_u`，`aligned(1)`）、AVX-512 `_mm_loadu_si128`、SVE `svld1_s8`（元素粒度）、CUDA `ld.global.u8`。因此 64B region 对齐是充分条件不是必要条件。
- N 不需要是 8/16/4 的倍数：各有标量尾块（AVX2 `cpu_avx2.py:519-522`；AVX-512 `cpu_avx512.py:1642-1646`；SVE predicate `:2188-2189`；CUDA 一线程一输出元素），K 也有尾块。现有测试覆盖了非倍数形状（`test_w8a8_avx2.py:277-303`、`test_w8a8_avx512.py:336-366`、CUDA 真机 121 checks/0 FAIL）。
- runtime 侧只校验 dtype/count/descriptor 连续（AVX2 `avx2.py:1797-1802`；AVX-512 `avx512.py:2257-2264`；SVE `sve256.py:2409-2428`；CUDA `runtime.py:946-953`），**不看操作数生产者、不做 layout plan 检查**。因此"参数直接作为 inputs[1]"在结构上就是既有能力。

### 2.2 AVX2（`raw/backend-avx2.md`）

- kernel：`python/pypto/compiler/targets/cpu_avx2.py:491-530`。right 的 6 处访问全部带 `n` stride：`:511 right + index*n + block`、`:512 right + (index+1)*n + block`、`:520/521 right[index*n+block]`、`:526 right[index*n+column]`；`:498-499` 是 `-128` 全量预扫（隐含"dense k*n、无行 padding"）。
- 旧路径的 transpose 输出：`cpu_avx2.py:430-450 PTX_DEFINE_TRANSPOSE`，`output[column*rows + row] = input[row*cols + column]` —— 对 `[out,K]` 输入就是 `[K,out]` 的普通 row-major 拷贝，与 c1 字节逐位相同。
- 现成测试 `python/tests/ut/pypto_x/test_w8a8_avx2.py:106-114` 的 right 本来就是顶层函数参数；`:258-274` 断言它是 native W8A8（不是 host_reference）。
- 注意（非正确性）：int8 不在 `byte_view.py::_DTYPE_INFO`（`:29-37`、`:51-58`、`:382-403`），`BufferBinding` memoryview 会走 `_flat_storage → list → ctypes` 拷贝（`avx2.py:392-415`、`:298-299`、`:304-320`）。这是既有行为；c1 不会新增拷贝，反而省掉 transpose 的读写。

### 2.3 AVX-512（`raw/backend-avx512.md`）

- kernel：`python/pypto/compiler/targets/cpu_avx512.py:1592-1647`。`:1610` 16 列分块、`:1613` K 步进 4、右操作数 `:1616-1626` 四处 `right + (step+j)*n + column`、`:1638/1645` 标量尾块；全部 unaligned load。
- **关键排查**：B6 的 int8 native layout（`_NATIVE_LAYOUT_DTYPES` 含 int8，`cpu_avx512.py:2072-2077`）**没有被 W8A8 权重路径使用**。rank-2 `(1,0)` int8 transpose 优先命中 packed transpose（`cpu_avx512.py:1965-1967, 2011-2034`；测试钉死 `test_cpu_avx512_layout.py:303-309`），输出是普通 row-major copy（`cpu_avx512.py:960-976`，核心 `:973`）。B6 新平面只服务 reshape/view/contiguous/slice/非 (1,0) transpose（`:2105-2118`），且输出仍是按输出 row-major 的 gather-copy。
- qmatmul 分支只由 dtype/shape/matmul plan 命中（`cpu_avx512.py:2475-2511`），**不看 producer**；删 transpose 后权重路径上没有任何 layout/packed op。
- 现有测试 `test_w8a8_avx512.py:115-123` 即"right 为顶层参数"，`:336-366` 覆盖 n/k 尾块与 rank3/4。

### 2.4 SVE256（`raw/backend-sve256.md`）

- kernel：`python/pypto/compiler/targets/cpu_sve256.py:2179-2204`，右操作数唯一寻址 `:2193 right + inner*n + column`；VL=32B 时 `svcntw()=8`，`svunpklo_s16/s32` 只消费低 8 列，与 `column += svcntw()` 一致；尾块 predicate 精确，无 region 外 over-read；`svld1_s8` unaligned 安全；-128 在 ELF 侧双段预扫拒绝。
- **lowering 解封已被独立复核**：default 150 / all+lm 187 个违规**全部**是 `transpose(int8)`，其余 W8A8 op 0 违规；删掉 transpose 后 shadow census 剩余违规 = 0（白名单 `lowering/cpu/aarch64/sve/lowering.py:45-54` 已含 4 个 W8A8 op；int8 图参数不产生 plan）。参见 `_meta/pypto-x/c8-l4-sve256/brief.zh-CN.md` §2.2 与本次复核。
- **必须改的 runtime 缺口（c1 的真实阻断项）**：SVE256 的 `_storage_values`（`backends/cpu/aarch64/sve256.py:1075-1172`）对 `BufferBinding` 来的 memoryview 走最后 branch `:1162 [storage[index] for index in range(count)]`；由于 `BufferBinding` 把任何 buffer 统一 cast 成 `'B'`（`runtime_binding.py:315`）且 `byte_view` 不支持 int8（`byte_view.py:31-37/51-58/382-403`），负码会被读成 128–255，再经 `_cast_value`（`backends/cpu/tensor.py:140-155` 严格范围）在参数绑定处抛 `contains invalid int8 values`。**任何真实权重都会失败**。最小修复：把 int8 加进 `byte_view` typed view（同时消灭 list 化），或在 `_storage_values` 增加 int8 raw-buffer 分支。
- 该缺口与 c1/c2 无关：两个候选都会让 int8 权重直接成为 qmatmul 的图参数。

### 2.5 CUDA dp4a（`raw/backend-cuda.md`）

- kernel：`python/pypto/compiler/targets/cuda_w8a8.py:398-463 _render_qmatmul`。右操作数基址 `:423-427`：`right + batch*k*n + col`；主循环 `:444-445` 读取 `[%rd26+0/n/2n/3n]`；K 尾块 `:452-460`；只有 1 字节 `ld.global.u8/s8`，无 char4/向量 load、无 n%4/对齐要求。
- **但是**：当前仓库没有 CUDA W8A8 的 **graph executor**。`CudaRuntime.run_w8a8_qmatmul`（`backends/cuda/runtime.py:921-976`）只接受 Python 序列（`_w8a8_flat` `:258-268` 拒绝 bytes/memoryview/BufferBinding；`flatten_values` 只展开 list/tuple），且 `CudaW8A8Compiler` 拒绝非 4 opcode 的图（`cuda_w8a8.py:274-280`）。C6 验收明确是 per-op 真机证据（RTX 5080，121 checks/0 FAIL）。
- c1 对 CUDA 的准确结论：**kernel 0 改动**；若要把 CUDA 纳入"整网 v2 逐位"，M2 需要新增"packed region → qmatmul"的 marshalling/编排（复用 `driver.memcpy_htod_address`，不要走 Python list），否则 CUDA 的证据边界只能停留在 opcode 级（与 v1 相同）。

### 2.6 反例与替代方案

- **c1 的反例（必须写清楚）**：如果某后端的 qmatmul 需要行内有 padding 或按 64B 对齐行起点，c1 就会失效；本次逐处核实没有发现这种假设（四后端全是 unaligned load + stride=n）。唯一"看起来像反例"的是 AVX-512 的 B6 int8 layout，已核实不作用于 rank-2 `(1,0)` transpose/qmatmul。
- **c1 的硬前置**：SVE256 int8 ingestion 必须补（§2.4）；CUDA 整网必须做范围决策（§2.5）。
- **替代方案（若 M2 发现某个后端确有隐藏假设）**：保留 c1 的 packed `[K,out]`，在该后端 artifact load 时做**一次性** host repack（不是每 forward），再把 repack 后的 buffer 喂 qmatmul；这样仍满足"每 forward 搬运 0"。只有在"四后端都要求不同 layout"时才退到 c2。

---

## 3. 问题 3：c1 vs c2 决策

### 3.1 c2 的完整改动清单（基于本次逐后端核实）

| 层 | 文件（现状行号） | 改动 |
|---|---|---|
| 契约 | `docs/20-planning/0002…` §2.3/§3.3 | `qmatmul_s8s8_s32` 增加右操作数 layout/transpose 属性语义；R5 判据扩展 |
| 图构造 | `python/pypto/portable/qwen35.py:486-496` | 删除 transpose，改成 qmatmul 属性（右操作数仍 `[out,K]`） |
| 共享校验 | `python/pypto/lowering/cpu/__init__.py:1589-1644` | 接受新属性；`transpose_a/transpose_b` 当前被显式拒绝（`:1640-1644`），需新开 NT 属性；plan 构造 `lowering/cpu/vector/plan.py:1952-2007` 携带 layout |
| AVX2 | `python/pypto/compiler/targets/cpu_avx2.py:491-530` | 新增 NT 寻址（right 的 K 维连续）分支；主循环要重构（现在沿 N 向量化） |
| AVX-512 | `python/pypto/compiler/targets/cpu_avx512.py:1592-1647` | NT 分支需要按 n 行取 K 段并重排进 VNNI 的 4×16 寄存器布局（新代码量最大） |
| SVE256 | `python/pypto/compiler/targets/cpu_sve256.py:2179-2204` | NT 分支（`svld1` 沿 K 连续，行 stride=K） |
| CUDA | `python/pypto/compiler/targets/cuda_w8a8.py:423-460` | NT 寻址（对 dp4a 而言 4 个连续 K 反而连续，改动中等） |
| 标量/golden | `lowering/cpu/__init__.py` 标量路径、`backends/cpu/runtime.py:981-1023`、`portable/qwen35_w8a8_scheme.py:351-397`、`tools/qwen35_reference/w8a8_reference.py` | 全部要支持 NT，否则 R5 无 golden |
| 测试/证据 | 四后端 R5 差分 + plan digest + payload/metadata 声明 | 每后端新增 NT 用例；`w8a8_operations` metadata 增加 layout 字段 |

### 3.2 成本对比

| 维度 | c1 | c2 |
|---|---|---|
| 4 个 kernel | **0 行** | 4 处 SIMD 内核新增寻址模式 + 尾块/分块重构 |
| 标量 golden / R5 | 不变 | 改 opcode 语义 → golden + 四后端 R5 全部重验 |
| 量化器/打包器 | 码值不变，只改字节顺序（block transpose） | **不变** |
| binding schema | shape 语义翻转 + 新键 + 版本升 | 不变 |
| 契约 | packed layout 语义升 v2 | opcode 语义升 v2（面向所有后端） |
| SVE256 int8 ingestion | 必须补（共享前置） | 必须补（共享前置） |
| CUDA 整网 | 需要 M2 决策（共享前置） | 同样需要 + NT kernel |
| 每 forward 搬运 | 474/717 MiB → 0 | 474/717 MiB → 0 |
| SVE256 lowering 解封 | 是（transpose 消失） | 是（transpose 消失，qmatmul 带属性） |
| 证据失效面 | 所有 W8A8 graph_digest | 同样所有 W8A8 graph_digest（删 transpose 本身就会变） |
| 预估工作量 | 1 实现 + 1 跨后端验收 | ≈2–3× 实现 + 更大的验收矩阵（NT 的矩形/尾块组合） |

### 3.3 推荐：c1，理由与限定

**推荐 c1**，核心证据是 §2 的逐后端结论：qmatmul 消费的就是 `[K,N]` 连续 row-major，c1 让打包器直接产出同一字节镜像，等价变换、零 kernel 改动。反过来说，c2 的唯一"优势"是保留 `[out,K]` 存储语义（对未来的非 qmatmul 消费者更友好），但它要为此在 4 个后端各写一条 NT 路径并重验 R5，而收益（每 forward 搬运归零、SVE256 解封）与 c1 完全相同。

**限定条件（诚实边界）**：

1. c1 的前提是"packed bytes 与 transpose 输出逐位相同"；AVX2/AVX-512 已由源码级核实（普通 row-major transpose），CUDA/SVE256 没有 v1 整网基线可比，M3 对它们用"v2 vs 标量 golden"的口径，而不是"v1 vs v2"（见 §7 风险 4）。
2. c1 把 `[K,out]` 变成**图参数/绑定的公开语义**，未来 vendor GEMM（W8J）若要直接消费这些 region，必须知道 v2 布局；当前 vendor 路径（`framework/weight_pack.py`，PACK_VERSION=2、64×64 tile）与本变更正交，不受影响。
3. 若 M2 发现四后端中任何一处必须保持 `[out,K]`（例如某后端 kernel 被后续改成按行 64B 对齐），则退到 §2.6 的 per-backend load 时一次性 repack，而不是 c2。

### 3.4 vendor int8 布局旁证与边界（父 agent 2026-09-12 补充，已核实原文）

父 agent 转来 `axis-b-a3-runtime`（integration `73fd709f6`）的实测：aarch64 vLLM 的 int8 路径是 **oneDNN + ACL `CpuGEMMLowp`**（`_C.abi3.so` 含 dnnl/arm_compute 符号、无 `kai_`；`is_onednn_acl_supported()==true`），API 要求 **weight 为逻辑 `[K,N]` 且 `stride(0)==1`**，原文注解"即 [N,K] 行主序权重的转置视图"（`_meta/pypto-x/axis-b-a3-runtime/brief.zh-CN.md:188-191`）；否则报 `Expected b.stride(0) == 1 to be true`。

**语义更正（重要）**：`stride(0)==1` = K 连续 = **v1 packed 顺序 `[out,K]` 行主序**；c1 的 `[K,out]` 行主序是 N 连续（stride(0)=N, stride(1)=1）。四后端 qmatmul 全部硬编码 `right[i*n+j]`（§2），所以：

| 消费者 | 需要的存储字节序（逻辑 [K,N] 视角） |
|---|---|
| 四后端 portable qmatmul（c1） | `[K,N]` row-major，stride=(N,1) |
| aarch64 vLLM oneDNN+ACL | `[N,K]` row-major 的转置视图，stride=(1,K) |
| KleidiAI（Q8b 候选） | 私有 packed 权重布局（scale 打包进 RHS）（第三种） |

⇒ **没有任何单一 storage order 能同时满足三者**（除非 K/N 退化）。c1 **不会**让 oneDNN+ACL 路径"不再需要任何转置"；它把 v1 的零拷贝 `.t()` 视图换成一次性 repack。反过来，`[K,N]+stride(0)==1` 与现有四后端 kernel **不兼容**——这正是 c2 要解决的 NT 语义。

**对决策的影响**：这条旁证**不推翻 c1 推荐**，但把它的收益限定为 portable 四后端（0 kernel 改动 + SVE256 解封 + 每 forward 转置归零）。kickoff §2 明确"不改 vendor 路径的语义"，而当前 W8J/vLLM 注入的 packed 平面是 BF16/f32 的 64×64 tile（`framework/weight_pack.py`，PACK_VERSION=2），与 W8A8 int8 region 是两条路；oneDNN+ACL 目前从 vLLM 自己的权重张量取 `.t()` 视图，Q4 artifact 的互通实验不是对 portable packed 顺序的硬依赖。
**范围裁决（父 agent 2026-09-12 已采纳本设计的推荐）**：**不要求** portable packed storage 与 oneDNN+ACL 共用同一种 int8 顺序。
- 维持 c1：portable 四后端 kernel 0 改动、SVE256 解封、每 forward 转置归零；
- vendor 侧各自在 ingestion 期一次性 repack / 或直接吃源 checkpoint 张量，并把"需要的 RHS 布局"登记为 provider capability 的结构化前置条件（详见 §5⑥）；
- c2 保留为**触发式**方案：只有当未来出现"vendor 与 portable 必须共用 storage"的硬要求时才启用。

---

## 4. 契约 v2 的具体差异（字段 / 语义 / 版本号）

### 4.1 语义差异表

| 项 | v1（0002，FROZEN） | v2（本设计） |
|---|---|---|
| 源权重（checkpoint / 量化方案） | `[out,in]`，per-output-channel axis=0 | **不变** |
| `<param>.int8` region shape | `[out,in]`，`in` 连续 | **`[K,out]`（K=in_features），`out` 连续** |
| int8 字节序 | `q[o*K + k]` | **`q[k*out + o]`**（行 stride = out 字节，无 padding） |
| region nbytes / 数量 / 顺序 / alignment | out*K；2 region/权重；64B | **不变**（offset 也不变） |
| scale region | `[out]` fp32 | 不变 |
| 图形态 | `weight_s8[out,K] → transpose → qmatmul(x[M,K], w_t[K,out])` | **`qmatmul_s8s8_s32(x[M,K], w_s8[K,out])`，无 transpose** |
| opcode 语义 | qmatmul 右操作数 `[K,N]` | **不变** |
| `BindingEntry.shape` | 参数逻辑 shape | **packed/storage shape（int8 条目 = `[K,out]`）** |
| `BindingEntry.quant` 键集 | 9 键（scheme/signed/bits/axis/granularity/scale_dtype/zero_point/scale_entry/group_size） | **+ `storage_layout`（值 `"k_out_rowmajor"`）**；`axis=0` 语义仍指源 `[out,in]` 的 per-output-channel |
| 量化数值 | RNE、clamp [-127,127]、s_w=max|W|/127 | **不变**（码值/scale 逐位不变，只是排布不同） |
| `codes_sha256` / manifest digest | 对 `[out,in]` codes 取 sha | **对 packed `[K,out]` codes 取 sha（值会变）**；同时新增 `storage_shape`/`codes_layout` 字段 |
| region 计数 | 518 / 554 / 520 / 556 | **不变**（556 = 321+187+48 不变） |
| 图 digest | 07c1e4f2 / 150e3725 / … | **全部 W8A8 digest 变化**；BF16 digest 不变 |
| 数值行为 | — | 与 v1 逐位一致（packed 字节与 transpose 输出相同，累加顺序不变） |

### 4.2 版本号怎么升

1. **`PACKED_LAYOUT_VERSION` 2 → 3**（`runtime_binding.py:31`）。理由：region 的字节语义变了，这是 packed layout 直接负责的版本面。`build_packed_layout` 已有 "schema 版本不符 → reject_and_recompile" 的先例（`:1480-1486`），v3 继续沿用。
2. **`BINDING_SCHEMA_VERSION` 2 → 3**（`runtime_binding.py:30`）。理由：`quant` 映射新增冻结键（`:44-54`），`from_dict` 的版本门（`:680-684`）会明确拒绝 v2 schema；同时 `_canonical_quant_entry` 的 missing/unknown key 检查（`:199-207`）在结构层再挡一次。代价：BF16 序列化 schema 的 `schema_version` 也变成 3（全局常量），但 BF16 的 graph digest / region / offset / nbytes 都不变，且 `"schema_version": 2→3` 是同位数字，metadata-only 证据的字节数不变（需在 M2 跑一次 BF16 metadata driver 确认零差异，见 §7 风险 12）。测试里 4 处硬编码需要更新：`test_w8a8_binding.py:88,227,261,274`。
3. **`WeightLayout` JSON `layout_version` 1 → 2**（`qwen35_weight_layout.py:166`），并给每个 `quant_weight` region 增加 `storage_layout` 字段；`layout_digest`（`:212-215`，canonical JSON 的 sha256）因 shape 翻转自动变化，继续承担 fail-closed 比对。
4. **契约文档**：0002 保持 FROZEN v1（不覆盖历史）；新增 `0010-…-w8a8-contract-v2-packed-k-out.zh-CN.md` 为 v2 规格；在 0002 顶部加 "LAYOUT ITEMS SUPERSEDED BY 0010" 指针，并在 `docs/00-handoffs/ERRATA.zh-CN.md` 追加 ERR-0006（登记 v1 §3.1/§3.3 的 layout 语义被 v2 取代、切入波次与 digest 清单）。
5. **后端 artifact payload version**：**不升**。c1 不新增 native symbol、不改 kernel ABI、不改 artifact wire format；旧 artifact 由 `program_digest` 不匹配（`runtime_binding.py:1701-1706`）在装配期 fail-closed。这与 B6 为了"新增符号/执行边界"升 payload 的情形不同，需在 M2 review 时明确记录。
6. **graph digest** 没有版本常量，自然变化；建议在 graph metadata（已被 canonical JSON 覆盖）里保留 `quantized_parameters[*].storage_layout`，让 digest 同时绑定布局语义。

---

## 5. 影响面清单（立项书 §4 的 ①–⑤，逐项到文件/行）

### ① 契约
- 0002 需要被 v2 取代的部分：§1.1 逻辑 layout（`out_in_rowmajor`）、§3.1 packed layout（`[out,in]`/in 连续/64B 行起点叙述）、§3.2 binding schema（quant 键集）、§3.3 `QuantizedTensorDesc.logical_layout`、§3.5 图构建入口、§4 cache key 里的 layout 叙述、§9 R7/R8 的断言口径。
- 0002 不变的部分：§1.2–1.5 量化 scheme/体积/边界规则、§2 层覆盖、§3.4 tied lm_head 双 packing、§5 精度门禁、§6 失败回退、§8 决策 D1–D12。
- 新增 v2 验收：R15「v1 图与 v2 图在同一后端、同一输入下全部输出 buffer sha256 一致」、R16「v1 artifact / v1 schema / v1 packed payload 在 v2 下 fail-closed，反向亦然」、R17「packed bytes 无行内 padding，`region.nbytes == K*out`」、R18「四后端 qmatmul 执行模式仍为 native，且 K/N 尾块矩阵与 v1 相同」。

### ② binding / packer
| 文件:行 | 现状 | c1 改动 |
|---|---|---|
| `portable/qwen35_w8a8_weight.py:149,222,251` | 只产出 `[out,in]` codes | 增加 `storage_layout="k_out_rowmajor"` 模式：按 out-channel block 量化后 block-transpose，写出 `q[k*out+o]`；`QuantizedWeight.manifest_entry` 增加 `storage_shape`/`codes_layout`（`codes_sha256` 改为对 packed 字节） |
| `portable/qwen35.py:433-521` | `_quantized_linear_projection` 显式 transpose（:490），shape 校验 `(out,in)`（:460-465） | 删 transpose；校验改为 `(input_width, output_features)`；qmatmul 右操作数直接用 weight_s8 |
| `portable/qwen35.py:603-670` | `_decoder_parameter_manifest` / `qwen35_lm_head_quantized_manifest_entry` 产出源布局 shape | `_quantized_compact_manifest`（:753-795）与 lm_head entry（:658-670）对 int8 条目翻转 shape |
| `portable/qwen35.py:2103-2114` | `linear_parameter` 按 `[out,in]` 声明 + scale `(shape[0],)` | int8 weight 声明 `[K,out]`；scale 仍取源 out（=shape[0] before flip） |
| `portable/qwen35.py:1798-1827` | `quantized_items` 的 `axis=0` | 增 `storage_layout="k_out_rowmajor"`；`axis` 仍 0（源语义） |
| `portable/qwen35.py:2319-2322` | lm_head int8 `(v,h)` | 改为 `(h,v)`，scale `(v,)` 不变 |
| `portable/runtime_binding.py:30-31,44-54,193-241,465-476` | v2 schema / 9 键 / hard-code `logical_layout="out_in_rowmajor"` / quant weight 只要求 rank2 int8 | 版本 2→3；`QUANT_ENTRY_KEYS` 加 `storage_layout` 并校验取值；entry shape 允许/要求 `[K,out]` |
| `portable/runtime_binding.py:768-960` | trusted manifest 从 Core 参数 shape 反推 profile（`gate.shape[1]!=hidden` 等），并把 quantized 条目 dtype 重写为 int8 但**保留 shape** | 引入 `logical_shape(entry) = reversed(entry.shape) if entry.quant else entry.shape`，所有 profile/几何校验用 logical shape；int8 重写时把 shape 反转成 storage shape |
| `portable/runtime_binding.py:1467-1550` | `build_packed_layout` 从 entry.shape/nbytes 生成 region | 逻辑不变；region shape 自动变 `[K,out]`；新增断言 `region.nbytes == shape[0]*shape[1]`（顺序无关）与 v3 版本门 |
| `portable/qwen35_weight_map.py:484-489,524-529` | 要求 Core shape == manifest shape == checkpoint tensor shape | int8 条目允许 `tensor.shape == reversed(manifest_shape)`；非量化条目保持严格相等 |
| `portable/qwen35_weight_layout.py:313-420` | quantized region 只校验元素数；region.shape 原样进 JSON | 增 `storage_layout` 字段与 `source_shape`（可选）；`layout_digest` 随 shape 变化 |
| `backends/cpu/byte_view.py:31-37,51-58,382-403` | 不支持 int8 → list 化 | **c1 必改（SVE256 路径）**：int8 加入 typed view（raw bytes 快路径）；或在 `sve256._storage_values:1075-1172` 加 int8 raw 分支 |
| `python/tests/ut/pypto_x/qwen35_weighted_execution_driver.py:721-833` | `w8a8_real_parameters` 用 entry.shape 校验 source shape/建 binding | int8 条目：source shape = reversed(entry.shape)；binding 用 packed 字节 |
| `tools/qwen35_reference/packer_side_a.py` | scale 形状按 `entry.shape[0]` 校验、codes 与冻结 manifest 对拍 | 改用 source shape；对 v2 重算（codes_sha256 会变，scale sha 不变） |

### ③ 图结构 / digest
- 删除的节点数（本次 introspection，任意 T）：default 150、all 186、default+lm 151、all+lm 187 个 `transpose`（输入 dtype int8）。
- `graph_digest = sha256(program.canonical_json())`（`runtime_binding.py:1387`）。canonical JSON 至少有两类变化：op 列表少 150/186/151/187 个 transpose；int8 参数 ValueRef 的 shape 翻转。⇒ **所有 W8A8 图的 digest 必然改变**，且无法从 v1 digest 预测新值（M2 产出）。
- 已知 v1 W8A8 digest（T=5,P=0）：default `07c1e4f2c272d1cd4d3aa6fa38e62840df925bbf054d520653cc86a689fdadce`、all `be37f12e…`、default+lm `dc7126a5…`、all+lm `150e3725…`；Q3 还冻结了 T=8/T=18 的 `8b0a4e88 / 68fb6cdf / c900bbd7 / b47326dc` 等。
- **BF16 图不变**：`456ab519c9b9bc7f281103b4a327f14b4112d6751acf30c00cd612f40d6d786e`（T=5,P=0）与 `66dd4077…`（T=1/past=4096）继续有效；因此 BF16 累积回归、artifact 版本链、B6 L1、CUDA C1/C2 都不进入重基线。
- region 矩阵不变（518/554/520/556），offset 不变（nbytes 与顺序不变）；变的只有 int8 region 的 `shape`/`storage_layout` 与 `WeightLayout.layout_digest`。

### ④ 后端
见 §2 表。补充两点：
- 所有后端都保持 "`qmatmul_s8s8_s32` 在 plan 里是 native/raw-bytes 执行模式"，不会因为删 transpose 被降级为 host_reference；AVX2/AVX-512 的 execution metadata 先判 W8A8 mode，与图里的 transpose 无关。
- 需要同步的测试钉点：`test_cpu_packed_liveness.py:280`（`native_layout_reorder_operations == ("transpose",)`）、四后端 `w8a8_operations` metadata 的形状列表。这些是测试期望更新，不是 kernel 改动。

### ⑤ 证据
见 §6；核心面：所有挂 W8A8 graph_digest / region shape / `weight_scale_manifest_digest` / `codes_sha256` 的证据全部失效；BF16-only 与 opcode-only 证据可保留（但 full pytest 必须在新树上重跑一次）。

### ⑥ vendor 路径（边界，不在 c1 的改动面内）
父 agent 已裁决：**不要求 portable 与 vendor 共用同一 int8 storage**（2026-09-12；理由：kickoff §2 非目标 + 三种互不兼容的需求 + 强行共用的代价 ≥ 本立项收益）。三件事写死如下：

1. **portable 侧收益不变**：c1 对 AVX2 / AVX-512 / SVE256 / CUDA 的 **kernel 改动面 = 0**；SVE256 的 W8A8 lowering 解封（int8 transpose 消失，剩余违规 0）；每 forward 的 int8 权重 transposes 搬运 = 0（474.0/717.1 MiB → 0）。SVE256 runtime 的 int8 ingestion seam 与 CUDA 的 graph-executor 缺口是 c1 的独立前置（§2.4/§2.5），不算 vendor 改动。
2. **vendor 侧处置**：aarch64 oneDNN+ACL（要 `[K,N]`+`stride(0)==1`）与 KleidiAI（要私有 packed 权重布局，scale 打包进 RHS）**各自在 ingestion/接入期做一次性 repack，或直接消费源 checkpoint 张量**；`[K,out]`（portable）与 `[out,K]`（oneDNN/ACL 视图）之间的换算**不得**进入每 forward 热路径。接入方必须把"该 provider 需要哪种 RHS 布局（`k_out_rowmajor` / `n_k_rowmajor_view` / `private_packed`）"登记为 **provider capability 的结构化前置条件**，与 0006 的 `gemm_provider/library/tolerance_class/accum_width` 元数据字段草案同一思路，由 fail-closed 校验（布局不符即拒绝接入，不得静默改布局）。
3. **c2 的定位与触发条件**：c2 **不是被否决**，而是保留为条件方案——**当且仅当**出现"vendor 与 portable 必须共用同一种 int8 storage"这一新要求时启用（此时唯一能免 repack 的共同布局是 oneDNN+ACL 的 `[out,K]` 视图序，代价 = 4 个 NT kernel + 标量 golden + R5 + 每后端 plan/artifact 语义重验）。在此之前不做 NT，不新增双 region。
- W8J/vLLM 注入的 packed 平面（`framework/weight_pack.py`，PACK_VERSION=2、64×64 tile、`PACKED_GEMM_LAYOUT_VERSION`）是 BF16/f32 平面，与 W8A8 int8 region 无交集；c1 不动它。

---

## 6. 重基线可执行清单（立项书 §6）

> 口径：`META = /home/chiro/projects/pypto/worktrees/_meta/pypto-x`；仓库内路径相对 worktree。机时为预估（依据已有 brief 实测 + 本次盘点），heavy 一律经 `scripts/resource/run_local_heavy.sh`（local）或 A3 串行窗口。
> **重要**：M3 的"v1↔v2 逐位"与 M4 的"重基线 L4"可以共用同一批 v2 重段（v1 侧直接使用 Q3/Q2 既有 dump），不要重复跑两遍。

| # | 条目 | 证据路径（现状） | 复跑命令 | 预计机时 | 可自动化 | 资源 | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | W8A8 图 digest / transpose 计数 / region shape 现状基线 | `META/w8a8-v2-transpose-elimination-m1/logs/m1_introspect.json`（本次） | `PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python python3 META/w8a8-v2-transpose-elimination-m1/raw/m1_introspect.py` | <10 s | 是 | 无锁（无权重/无编译） | 必跑（M2 前后各一次，替换 v1 值） |
| 2 | binding schema v3 + packed layout v3 metadata（含 518/520/554/556 与 int8 shape） | `META/qwen35-w8a8-scheme-binding/r8-packed-layout-v2-driver.json`；`META/w8a8-lm-head-ingestion/raw/preflight_w8a8_{518,554,520,556}.json` | `run_local_heavy.sh … qwen35_w8a8_binding_driver.py --output <EVID>/r8-packed-layout-v2-driver.json`；lm_head preflight 同 driver 的 s0/preflight 段 | ~10–60 s | 是 | local 锁（driver 声明 model-sized metadata） | 必跑 |
| 3 | W8A8 真权重 e2e v2：AVX-512、3 prompt × {default=518, full=556}、prefill+4 decode + L4 判定 | 现状 `META/c8-l4-avx512/{raw/s3_*,raw/judgment_*}`；`META/qwen35-w8a8-graph-path` | `run_local_heavy.sh … python3 python/tests/ut/pypto_x/qwen35_weighted_execution_driver.py --stage s3 --backend avx512 --precision w8a8-linear --decode-steps 4 --prompt <p> --quantized-linear-scope <default\|all>` + lm_head 开关；判定 `tools/c8_l4_compare.py`（脚本见 `META/c8-l4-avx512/scripts/run_judgments.sh`） | **~62 min**（Q3 实测 6 段合计 61.6 min；实测各段 464–693 s） | 部分（段编排已有） | local 锁分段 + 只读权重 mmap | 必跑（L4 结论可能翻转） |
| 4 | 同上，AVX2 后端（v1 无 L4 判定，这是新增） | `META/verify-qwen35-w8a8-graph-path`（含 AVX2 聚焦） | 同 3，`--backend avx2` | **~60 min**（按 3 的量级） | 部分 | local 锁 | 必跑（§5.2 每后端逐位） |
| 5 | v1↔v2 逐位对照（AVX2/AVX-512 三 prompt×两配置×prefill+decode） | v1 侧：`META/c8-l4-avx512/raw/dumps`、`META/qwen35-w8a8-graph-path`；v2 侧 = 3/4 的 `--debug-dump` 输出 | 不新增执行：对 3/4 的 dump 与 v1 dump 做 sha256/逐 buffer 比对（脚本可复用 `tools/c8_l4_compare.py` 的自检模式） | **0**（复用 3/4） | 是 | 无 | 必做（M3 硬判据） |
| 6 | SVE256 W8A8 整网（解封 Q5）：A3 原生 + 本机判定 | 现状 `META/c8-l4-sve256` 是 `BLOCKED_AT_LOWERING_NO_L4_VERDICT` | 先做 §7 风险 1 的 int8 seam 修复；A3：`driver --stage s3 --backend sve256 --precision w8a8-linear --sve-execution-mode native …`（3 prompt × default/full）；回传 logits 后本机 `tools/c8_l4_compare.py` | **A3 ≈1.5–4 h**（BF16 对照 924/1042/1717 s；W8A8 增加 150/187 个 qmatmul runner 调用，**待 M2 先测单段再冻结预算**）+ 本机判定 <10 min | 部分 | A3 串行（cpuset 320–366 纪律）+ local 判定 | 必跑（Q5 解封） |
| 7 | W8A8 lm_head 两小配置（520/554）的 ingestion/fail-closed 矩阵 | `META/w8a8-lm-head-ingestion/raw/{preflight,s2,s3}_*`；`META/verify-w8a8-lm-head-ingestion/raw/*` | 与 2/3 共用 driver；520/554 只需 metadata + 一次小 shape preflight（真实权重 s2/s3 已在 3 的 full 路径覆盖） | ~1–2 min | 是 | local 锁 | 必跑（轻） |
| 8 | full pytest `python/tests/ut/pypto_x`（v2 树上） | `META/b6-layout-native` 的 1200/7/0 口径 | `run_local_heavy.sh … python3 -m pytest -q python/tests/ut/pypto_x` | **~6–10 min**（B6 618 s、graph-path 380 s 量级） | 是 | local 锁 | 必跑 |
| 9 | 四后端 opcode 聚焦套件（AVX2/AVX-512/SVE256 QEMU/CUDA fake） | `META/qwen35-w8a8-cpu-avx2`、`…avx512`、`…sve256`、`…gpu-kernels` | `test_w8a8_avx2.py`、`test_w8a8_avx512.py`、`test_w8a8_sve256.py`、`test_w8a8_cuda.py`（QEMU/A3 的 native 段按 §2.4 修复后加跑） | **~2–20 min**（AVX2 93 s / AVX-512 14 s / SVE 40 s QEMU / CUDA 0.6 s，含 full 上下文另计） | 是 | local（CUDA 为 GPU-only 短探测，不占 gamepc） | 跑：**核对其 native 模式与数值不变**；预期不需要重做数值结论 |
| 10 | BF16 累积回归（T=1/5/18、en/zh/chat）与 artifact 版本链 | `META/verify-bf16-cumulative-en-t5`、`META/qwen35-weighted-multiprompt` | — | 0 | — | — | **不受影响，不跑**：BF16 graph_digest 456ab519/66dd4077 未变，c1 不碰 BF16 分支（§5③） |
| 11 | B6 L1（layout 原生化 per-op + 墙钟） | `META/b6-layout-native` §1.5、`META/verify-b6-layout-native` | — | 0 | — | — | **不受影响，不跑**：B6 L1 的图就是 BF16 `456ab519`/6728 ops；int8 layout 与 W8A8 packed layout 互不相干 |
| 12 | CUDA C1/C2 回归 | `META/integration-w4-cuda-c1-final/validation.json`、`…c2-final` | — | 0 | — | — | **不受影响，不跑**（FP32/BF16，无 W8A8） |
| 13 | c8-w8a8-reference-gold 的"同源审计"（不是数值 gold） | `META/c8-w8a8-reference-gold/reference/w8a8_audit.json`（manifest `4fe99771…`；lm_head codes `5665d00d…`） | 数值 gold 不重跑；只用 `tools/qwen35_reference/packer_side_a.py` 在 v2 下重算 manifest/codes digest，并证明"unpack 后逐元素与 v1 codes 相同、scale sha 不变" | ~2–5 min（需权重 mmap） | 是 | local 锁 | 必做（审计），不改 gold 数值/`band.json` |
| 14 | 契约/ERRATA/锁/交接文档更新 | 0002、新 0010、`ERRATA.zh-CN.md`、`configs/development_lock.yaml`、`HANDOFF.zh-CN.md` | 无 heavy | 0 | — | — | 必做 |

**条目数与机时**：14 条 = 必跑 9 条（其中图级重段 5 条：3/4/6 + 2/7 轻量）+ 不受影响 3 条（10/11/12）+ 审计/文档 2 条（13/14）。
**预计总机时**：local heavy ≈ **2.2–2.6 h**（3+4+5 约 2.05 h，8 约 0.15 h，2/7/9/13 约 0.2 h）；A3 ≈ **1.5–4 h**（待 M2 实测冻结）。合计 **≈4–7 h 机时**（不含 M2 实现与 M3 独立验收的人工/编译）。

---

## 7. 迁移与回滚

### 7.1 切换方案

- **单代码路径，不设长期 v1/v2 并行**。理由：本项目不持久化 packed 权重（§1.1），v1 证据是快照不是资产；双路径会让 4 个后端 × 2 套 layout 的 kernel/测试面翻倍，且容易出现"声明 v1、字节 v2"的静默错误。
- **切换点**：M4 = 当前批次（0040）收口后的下一个波次边界。M2/M3 全部在 `work/w8a8-v2-transpose-elimination-m1` 分支完成；0040 的 C8/Q3/Q4/Q5、U1、N4 证据在 v1 上收口并冻结（特别是 Q3 的 v1 dump 要留作 M3 的 v1 对照）；M4 合入 integration 后立即执行 §6 的重基线。
- **fail-closed 矩阵（v1 → v2）**：

| v1 输入 | v2 拒绝点 | 错误类型 |
|---|---|---|
| v1 编译 artifact（metadata.program_digest = 07c1e4f2… 等） | `assemble_launch_request` `runtime_binding.py:1701-1706` | `BindingMismatchError` |
| v1 序列化 BindingSchema（schema_version=2，quant 缺 `storage_layout`） | `BindingSchema.from_dict` `:680-684` + `_canonical_quant_entry` `:199-207` | `BindingMismatchError` |
| v1 packed int8 payload（`[out,K]`）喂给 v2 entry（`[K,out]`） | `_normalize_bindings` `:1592-1599` 的 dtype/shape/nbytes 全等校验；`bind_packed_region` 用 v2 region shape | `BindingMismatchError` |
| v1 weight mapping / layout JSON | `build_weight_layout` graph_digest 校验 `qwen35_weight_layout.py:333`；`layout_digest` 随 shape 变化 | `WeightLayoutError` / digest 不一致 |
| v1 packer 审计 manifest（`4fe99771…`、`b5fa257c…`） | `packer_side_a.py` 的 manifest digest 对拍失败（显式） | 脚本非零退出 |
| v2 artifact 在 v1 代码上 | v1 的 schema digest ≠ artifact program_digest | `BindingMismatchError` |

- **cache 安全性**：artifact cache key 含 `metadata`（内含 program_digest）与 `content_digest`（`compiler/artifact.py:131-145`），新图必然 cache miss，不会命中 v1 编译产物；旧 entry 只占磁盘、不可达。

### 7.2 回滚

- M4 在 integration 上是一个可 revert 的合入点；回滚 = revert 该 merge/squash（或 reset 到 M4 前的 freeze 点，freeze 点写进 ERR-0006）。因为权重不落盘、artifact cache 内容寻址，回滚后 v1 代码+旧 cache 自然重新生效，无数据迁移/格式转换要撤销。
- 分支 `work/w8a8-v2-transpose-elimination-m1` 保留；若 M3 只在某一个后端失败，**不合并 M4**，在分支上按 §2.6 的替代方案修（per-backend 一次性 repack），不做部分合入。
- **切换点之前**：0040 的所有验收仍按 v1 执行，避免同一批证据混用两个 digest 体系。

---

## 8. 风险清单（含"我可能判断错的地方"与 M2 最小实验）

| # | 风险 | 我为什么可能错 | M2 最小实验 |
|---|---|---|---|
| 1 | **SVE256 int8 ingestion 是 c1 的硬阻断**（已由本次内存复现：`BufferBinding memoryview → _storage_values → _cast_value` 对负码抛错） | 我只做了源码级 + 内存内复现，没有跑 QEMU/A3 ELF；也可能 M2 选择 numpy/raw-address 旁路而"看起来能跑"（但会有 list 化内存风险） | 先写一个 4 元 int8 `BufferBinding` 的最小 launch，断言现状失败；补 int8 typed-view 后断言 qmatmul 输出与 scalar golden 逐位一致（QEMU 即可），并在 A3 上跑一次 |
| 2 | **CUDA 没有 W8A8 graph executor**，c1 的 CUDA 证据只能到 opcode 级 | 我把 `CudaW8A8Compiler` 的 4-opcode 限制当成范围事实；若 M2 需要整网 CUDA，这是新增编排工作，不是"kernel 不改所以 0 改动" | 决策实验：在同一 Core 图上把 qmatmul 右操作数换成 packed region（`byte_offset` slice），验证现状拒绝；若做 region-direct，用 fake driver 断言走 `memcpy_htod_address` 且逐位正确 |
| 3 | **int8 参数每次都走 Python list → ctypes 拷贝**（AVX2/AVX-512 已如此，SVE 修复后也应走 raw bytes） | 我推断 c1 不新增拷贝（今天参数也要过一次 list），但"每 forward 搬运归零"的 census 只统计 transpose 字节，可能掩盖这份残余成本 | 在 M2 用现有 driver 记录一次 launch 的 `_flat_storage` 调用/字节；建议顺手把 int8 加入 `byte_view` typed view，让 AVX2/AVX-512/SVE 共享 raw-byte 快路径（作为可选但强烈建议项） |
| 4 | **"v1 vs v2 逐位"对 SVE256/CUDA 没有 v1 整网基线** | kickoff §5.2 要求四后端都有逐位证据；但 SVE256 v1 被 lowering 拒绝、CUDA v1 无 graph executor，字面上的 v1↔v2 对照在这两个后端不存在 | 改用"v2 图 vs 标量整数 golden（int32 逐位）+ epilogue R4"作为 SVE/CUDA 的逐位口径，并在验收报告里显式标注口径差异；AVX2/AVX-512 做真正的 v1↔v2 全 buffer sha256 对照 |
| 5 | **方形权重会掩盖 [out,K]/[K,out] 写反** | 四后端 runtime 都只校验 count（K*N==N*K），不做 axis-order 校验；Qwen3.5-0.8B 恰好没有 out==in 的量化层 | 构造 synthetic square case（如 K=N=8）与判错型数据 `right[k][c]=(k*7+c*3)%127`，在四后端 opcode 测试里断言 `out[0][c]==right[k0][c]`；同时断言 `region.nbytes == K*out` 且无 padding |
| 6 | **`_expected_qwen_parameter_manifest` 的几何校验可能漏改一处**，从错误的 profile 推出"看似合法"的 manifest | 该函数有 ~10 处 shape 表达式（gate/up/down/q/k/v/o/qkv/z/out/b/a），我只逐处核对了主线；漏改通常会 fail-closed，但也可能把 out/in 交换后仍然自洽（对称尺寸不存在，风险较低） | 在四种 scope/lm_head 组合下跑 `build_binding_schema` + `qwen35_w8a8_binding_driver.py`；再加一条断言：每个 quant entry 满足 `entry.shape == tuple(reversed(logical_shape(entry)))` 且 `quant["storage_layout"]=="k_out_rowmajor"` |
| 7 | **manifest/codes digest 变化**被误当成"量化结果变了" | `codes_sha256` 覆盖的是 packed 字节；只有 `scale_sha256`/`max_abs`/`saturation_count` 才证明量化数值不变 | packer differential：v2 codes unpack 成 `[out,in]` 后与 v1 codes 逐元素相等、scale 字节 sha256 相等；记录"唯一变化的是 codes 排列与 manifest digest" |
| 8 | **lm_head 大张量的 block transpose 实现**（[248320,1024]→[1024,248320]）可能慢或吃内存 | 目前离线量化是 chunked 的（`_QUANTIZE_CHUNK_ELEMENTS`），直接按列 scatter 会 cache 不友好 | 测 step0/ingestion 时间与峰值 RSS；要求实现按 out-channel block（如 64）量化后块转置，避免整张量二次拷贝 |
| 9 | **BINDING_SCHEMA_VERSION 全局升 3 影响 BF16 序列化** | BF16 graph digest/region/offset 都不变，但 serialized schema 的 version 字段会变；若有冻结 JSON 会失效 | 在 M2 跑 BF16 `qwen35_runtime_binding_metadata_driver.py`，diff `schema_json_bytes`/`layout_json_bytes`/counts；若只有 version 整数变化（同位数）则记录后放行 |
| 10 | **vendor GEMM（W8J）未来直接消费 v2 packed 权重** | 当前 W8J 的 `PACK_VERSION=2` tile64 与本变更正交，但它若改成消费 portable W8A8 region，会默认 [out,K] | 在契约 v2 文档里写明 storage layout 字段与消费者义务；W8J 接线时把 `storage_layout` 当必检字段 |
| 11 | **重基线低估** | 我用 Q3 的 61.6 min 作为 AVX-512 重跑基准，AVX2 按同量级估；SVE256 的 A3 时间没有 v1 基线，只能给量级 | M2 完成 SVE int8 seam + default 单段后，先测 en/default 的 prefill+decode 墙钟，再冻结 A3 窗口预算；local 的三个 W8A8 段可复用 M3 的 v2 逐位运行，不重复 |
| 12 | **契约文档合入位置** | 设计文档要求"写进 `docs/`"，但实现 worktree 的 `docs/` 是 upstream pypto 的文档树，20-planning 在控制仓 `pypto_x` | 本设计稿放 `_meta/.../design-w8a8-contract-v2-m1.zh-CN.md`（报告目录，符合立项书"或在报告目录给完整设计稿"的备选）；建议父 agent 在评审后落到控制仓 `docs/20-planning/0010-2026-09-12-w8a8-contract-v2-packed-k-out.zh-CN.md`，本 worktree 不新增 docs 提交以免污染 patch 集 |
| 13 | **vendor 布局误判风险**（本轮已发生过一次） | 父 agent 最初把 `stride(0)==1` 读成"N 连续"，经复核实为"K 连续"（=v1 顺序）；类似误读可能再出现在 M2 的 provider 接入里 | M2 接入 oneDNN/ACL 前，用一条最小几何断言钉死：把 `[out,K]` row-major 权重按 `(K,N)` 声明并断言 `stride==(1,K)` 才交给 `create_onednn_scaled_mm`；把 provider 布局需求写进 capability 结构化前置条件（§5⑥），不接受隐式假设 |

### 8.1 M2 最小实验清单（汇总，按优先级）

1. **packer differential**（v1 codes vs v2 unpack；scale/absmax/saturation 不变）——纯 Python，无锁。
2. **图级 static**：v2 W8A8 图 0 个 int8 transpose、150/187 qmatmul、int8 参数 shape `[K,out]`、新 digest；AVX2/AVX-512/SVE256 静态 lower 通过（SVE 剩余违规 0）。
3. **AVX2/AVX-512 4-op direct 图**：right 为顶层参数、无 transpose，与 scalar golden 逐位；覆盖 K/N 尾块、非方阵、rank3/4；与 v1-transpose 变体 bit-exact。
4. **SVE256 int8 seam**：最小失败复现 → 修复 → QEMU 逐位；A3 一次。
5. **CUDA 范围决策**：现状拒绝 packed region 的证据 + （若做）region-direct marshalling 的 fake-driver 逐位。
6. **fail-closed 矩阵**：v1 artifact / v1 schema / v1 payload / v2-on-v1 四类各有负向测试。
7. **manifest/几何审计**：四 scope 组合的 binding schema + `storage_layout` 断言 + BF16 metadata zero-diff。
8. **region 不变性**：518/520/554/556 与 offset 不变；`region.nbytes == K*out`；无 padding。

---

## 9. 证据清单（本次产出）

```text
brief.zh-CN.md                         结论摘要（交付物 2 的主文件）
design-w8a8-contract-v2-m1.zh-CN.md    本设计稿（完整版；建议合入控制仓 docs/20-planning/0010-…）
raw/m1_introspect.py                   只读 introspection 脚本（graph digest/op/region）
logs/m1_introspect.json                实测输出（5 组配置，c1 前的基线）
logs/m1_introspect.err / 终端命令日志  本次所有只读命令的落盘
raw/backend-avx2.md                    后端逐处证据（引用/行号/结论）
raw/backend-avx512.md
raw/backend-sve256.md
raw/backend-cuda.md
raw/code-evidence.md                   契约/binding/packer 关键代码摘录与行号
raw/rebuild-inventory.md               §6 重基线清单的证据路径与命令原文（逐条）
```

**参考的既有证据**（未复制）：

```text
META/c8-l4-sve256/brief.zh-CN.md §2.2/§5/§8（150/187、474.0/717.1 MiB、选项 (c1)/(c2) 的原始分析）
META/c8-l4-avx512/brief.zh-CN.md §2.3/§7.1（Q3 的 driver 命令与 61.6 min 分段实测）
META/qwen35-w8a8-graph-path/、META/verify-qwen35-w8a8-graph-path/
META/qwen35-w8a8-scheme-binding/、META/verify-qwen35-w8a8-scheme-binding/
META/w8a8-lm-head-ingestion/、META/verify-w8a8-lm-head-ingestion/
META/qwen35-w8a8-{cpu-avx2,cpu-avx512,sve256,gpu-kernels}/ 与对应 verify 目录
META/c8-w8a8-reference-gold/（数值 gold 复用；仅审计重算）
META/b6-layout-native/、META/verify-b6-layout-native/
META/verify-bf16-cumulative-en-t5/、META/qwen35-weighted-multiprompt/
docs/00-handoffs/ERRATA.zh-CN.md（ERR-0003）
```

---

## 10. 不做的事（本阶段边界自证）

- 未修改任何实现代码；worktree `git status` 干净（仅 _meta 报告目录在仓外）。
- 未编译、未 lowering、未跑 pytest 全量、未跑 heavy、未加载权重、未申请 local 锁。
- 未接触 A3 / GamePC / A2；未 push；未改 integration/上游 master/他人 worktree。
- 唯一的 Python 执行是 `PYPTO_X_PORTABLE_ONLY=1` 下的 graph metadata introspection（无权重、无 native 库、无 CUDA）。
