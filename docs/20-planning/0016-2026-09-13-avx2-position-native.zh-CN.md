# AVX2 原生位置控制面（avx2-position-native）

- 任务：`avx2-position-native`（batch 0047 切片 B）
- 锁条目：`waves.W8.batch_0047_int8_exact_blocked_and_avx2_position_native_2026_09_13.slices.avx2_position_native`
- 实现分支：`work/avx2-position-native`（worktree `../worktrees/pypto-x/avx2-position-native`）
- 基线：集成分支 `port/pypto-x-integration` @ `393d28e0f`（tree `eebb0d584c6f1d8088637a4f7c59a484e49e5bd7`）
- 证据目录：`../worktrees/_meta/pypto-x/avx2-position-native/`
- 参照实现：SVE256 原生控制面（`python/pypto/compiler/targets/cpu_sve256.py` 的
  `control_descriptor_wire="v5:pxio-pxcp-pxbd-pxwh+fnv1a64"` 与
  `python/pypto/backends/cpu/aarch64/sve256.py` 的 `expected_position_control`）
- 不在本切片范围：AVX-512 的同一缺口（后续切片）、AVX-512 行为逐字节不变（未改动
  `cpu_avx512.py` / `backends/cpu/x86/avx512*.py`）

## 1. 语义真源

**唯一语义真源是 host reference（vector-common）**，即本切片之前 AVX2 对这些算子的
执行路径：

- `python/pypto/backends/cpu/vector/runtime.py::_execute_operation`
- `iota` → `python/pypto/lowering/cpu/__init__.py::iota_values / iota_spec`
- `compare` → `compare_values`（Python 比较运算符）
- `where` → `_where_values`（Python 真值 + `_cast_value` round-trip）

本切片**不修改**任何 host 语义。原生实现逐位复刻上述行为；差分测试以
`avx2_runtime._reference_operation`（旧 AVX2 fallback 本身）为 oracle 逐字节比对。

### 1.1 iota

- 输出 dtype 仅 `int32`/`int64`（`iota_spec` 冻结）；rank ≥ 1；`axis` 允许负值并归一化。
- `start`/`step` 为有符号 64 位整数；**序列必须完整落在输出 dtype 范围内**，否则在
  scalar validation 阶段 `CpuScalarValidationError` 拒绝（不 wrap、不静默截断）。原生
  kernel 因此可以用 int64 中间量计算并以 `__builtin_mul_overflow/__builtin_add_overflow`
  二次防御；int32 kernel 额外检查落域。
- 零长度轴（含 `(2,0,3)` 且 iota 轴为 0 长度）合法且 native：`output_count==0` 时
  kernel 直接返回 0，不读源。
- 原生描述符只携带 `axis/start/step/inner_stride/dimension/output_elements`，数值由
  `value = start + ((index / inner_stride) % dimension) * step` 得到。

### 1.2 compare

- 两个输入 dtype 必须一致（shared validator 冻结）；输出必须 `bool`。
- `bool` 输入只允许 `eq`/`ne`（ordered bool 由 shared validator 拒绝）。
- 比较语义 = Python 比较运算符：
  - NaN：`eq=False`、`ne=True`、`lt/le/gt/ge=False`（unordered 语义）；
  - `+0.0 == -0.0` 为 True；
  - bool 输入按 Python 真值比较：**任意非零字节为 True**（例如 0x02 与 0x01 相等），
    因此 `compare_bool` kernel 必须先把字节归一化为 `!=0` 再比较。
- bf16 输入按 `bf16→f32`（精确）后比较；host 侧 `bf16→f32→f64` 也是精确转换，二者
  对 NaN/±0/次正规数一致。sNaN 与 qNaN 在 compare 上只影响"是否 NaN"，结果位型一致
  （compare 输出是 bool，不复制输入位型）。
- 输出 1 字节 `0/1`（与 `_buffer_from_values("bool", ...)` 一致）。

### 1.3 where

- shared validator 冻结 `condition.dtype == "bool"`；`input`/`other`/`output` 必须同一
  dtype。**因此"float NaN 条件"在基线与新实现中都无法到达执行层**（编译期拒绝，行为
  与基线一致）。
- 可到达的条件语义 = Python 真值在 bool 存储上的投影：**只有字节 0 选 else，任何非零
  字节（1/2/255）选 then**；kernel 不得使用符号位掩码默认行为。测试用原始字节
  0/1/2/255 专门覆盖。
- `where` 的 host 路径只对**被选中的分支**做 round-trip（`_where_values` 的惰性分支 +
  `_cast_value`），未选中分支不被触碰。因此原生实现：
  - 逐位复制选中值（整型/布尔：整型复制，布尔归一化为 0/1；bool 存储的 raw 字节
    2/255 由 host 侧 `bool(value)` 归一化，原生同规则）；
  - f32：只在 **NaN 且 quiet 位为 0** 时置 quiet 位（`|=0x00400000`），保留符号与
    payload；其余位型（含 ±0、inf、次正规、qNaN）原样复制；
  - bf16：同规则在 16 位宽（`|=0x0040`）；
  - 该规则等于 host 的 f32→f64→f32 / bf16→f32→bf16 round-trip（Python float 转换在
    x86 上静默 signaling NaN）。
- `LaunchRequest.scalars` 不允许非有限值（`frozen_mapping` 拒绝 NaN/inf），故 NaN 标量
  分支不可达；标量分支测试使用有限值。
- 标量 operand/条件与零长度输出均走原生描述符（全零 strides / `output_count==0`
  直接返回），与 host reference 逐位一致。

## 2. 版本与 wire

| 项 | 旧（payload 6） | 新（payload 7） |
| --- | --- | --- |
| `ARTIFACT_PAYLOAD_VERSION` | 6 | **7** |
| `NATIVE_ABI_MARKER` / ELF `ptx_avx2_abi` | `pypto-x.cpu.avx2:7` | **`pypto-x.cpu.avx2:8`** |
| `position_control_fallback` | `host_reference` | `native`（当且仅当零回退） |
| `where` | `host_preexpanded_operations=["where"]` | `host_preexpanded_operations=[]` |
| 控制描述符 | 无 | `position_control_descriptor_wire="v1:pxio-pxcp-pxwh+fnv1a64"` |

- 控制 wire 采用与 SVE256 v5 相同的 PXIO/PXCP/PXWH 形状（rank-sized descriptor），
  AVX2 侧版本号为 `v1`；每个 descriptor 计算 `fnv1a64` 摘要并以
  `descriptor_digest="fnv1a64:xxxxxxxxxxxxxxxx"` 写入 artifact。
- 严格解码：
  - 旧 reader（393d28e0f 的 `_decode_artifact`）要求 payload version == 6 且 ELF ABI
    标记 == `:7`；新 artifact 为 7/:8，**双闸门都拒绝**（真实双树探针见证据
    `raw/payload_reader_probe.*`）。
  - 新 reader 要求 ELF 标记 == `:8`（旧 artifact 的对象被拒），并要求整个
    execution-mode 表与 CoreProgram 重新推导的结果逐键相等；payload 6 风格的
    `position_control_fallback="host_reference"`、缺失 native 条目、旧 wire、旧
    pre-expansion 列表全部 fail-closed。
  - 篡改：metadata 中 descriptor digest 改 1 个十六进制字符 → load 拒绝；native
    `.so` 改 1 bit → content digest 拒绝；把 `:8` 换回 `:7` 并重算 digest → safety
    marker 拒绝。
- `host_preexpanded_operations` 现恒为 `[]`：where 不再预展开；无法覆盖的
  iota/compare/where plan 走 vector-common host reference（其内部自行做广播索引），
  逐 op 记录在 `native_position_control_fallback_operations`，绝不再声明 pre-expansion。

## 3. 覆盖矩阵（payload 7）

原生覆盖（`mode=="native_avx2_position_control"`）：

| op | dtype / kind | predicate | kernel |
| --- | --- | --- | --- |
| iota | int32、int64 | - | `ptx_avx2_iota_i32` / `_i64` |
| compare | float32、bf16、int8/16/32/64、uint8/16/32/64 | eq/ne/lt/le/gt/ge | `ptx_avx2_compare_<suffix>` |
| compare | bool | eq、ne | `ptx_avx2_compare_bool` |
| where | float32、bf16、int8/16/32/64、uint8/16/32/64、bool | - | `ptx_avx2_where_<suffix>` |

回退（带原因，逐 op 记录）：

| 条件 | mode | reason 形态 |
| --- | --- | --- |
| rank > 16（iota/compare/where） | `host_reference` | `<op> rank <r> exceeds the native AVX2 position-control max rank 16` |
| dtype 不在集合（**compare/where 的防御性路径**） | `host_reference` | `dtype <dtype> is not in the native AVX2 ... dtype set` |
| dtype 不在集合（**iota**） | 不适用 | **该 reason 分支不可达**：iota 的非法 dtype 在 lowering 直接抛 `CpuVectorUnsupportedError`；其 schedule 校验的实际 reason 字符串是 `iota schedule is invalid`（由 `verify-0047-avx2` 复核订正） |
| 混合 dtype / 非法 predicate / 形状不可广播（防御性） | `host_reference` | 对应原因字符串 |

- shared validator 会在 lowering 阶段先拒绝 float16/float64、非 bool 条件、混合 dtype；
  这些 reason 只作为 defense-in-depth 单测（`types.SimpleNamespace` 直接调用 predicate）。
- rank ≤ 16 时原生 kernel 的 descriptor loop 是标量循环（与已验收的 SVE256 v5 控制面
  一致）；f32 compare 在"两侧 operand 都与输出形状完全连续"时走
  `_mm256_cmp_ps`/`_mm256_movemask_ps` AVX2 快路径，f32 where 在同条件时走
  `_mm256_blendv_ps` + 向量 quieting 快路径；其余情况线性扫描 descriptor。

## 4. 元数据契约（artifact 字段）

- `position_control_operations`：按 plan 记录 iota/compare/where 名字（保留重复，与
  SVE256 一致）。
- `position_control_fallback`：`"native"` 当且仅当
  `native_position_control_fallback_operations == []`。
- `position_control_mode`、`position_control_descriptor_wire`、
  `position_control_kernel_version`、`position_control_max_rank`。
- `native_position_control_operations`：每个 native plan 一条
  `{operation_index, operation, mode, kind, kernel, dtype, rank, descriptor_digest}`。
- `native_position_control_fallback_operations`：每个回退 plan 一条
  `{operation_index, operation, reason}`。
- `native_helper_symbols` 包含所有声明的 `ptx_avx2_*` kernel 符号。
- runtime 与 compiler 共用同一 predicate（`position_control_native_contract`），load
  时用 CoreProgram 重新推导整张表并要求逐键相等；dispatch 时再次用同一 predicate 取
  contract，因此不存在"声明 native 实际 host"或反向的可能。

## 5. 测试与证据

- 单元测试：`python/tests/ut/pypto_x/test_cpu_avx2_position_native.py`
  - iota 差分矩阵（负 step/多轴/零长度/int32·int64 极值）；
  - compare 全笛卡尔矩阵 6 kind × f32/bf16/i8/uint8/int16/int32/uint64/bool，
    raw-bit 语料含 sNaN（±、非零 payload）、qNaN、±0、±inf、极值、次正规；
  - where 广播组合、raw 条件字节 0/1/2/255、f32/bf16 sNaN quieting 与 ±0 保位；
  - 元数据真实性、rank>16 逐 op 回退原因、predicate fail-closed、kernel dispatch spy、
    `nm` 符号检查；
  - payload/ABI 严格性、旧 ABI 标记拒绝、单字节篡改拒绝、cache key 包含 payload 版本；
  - Qwen3.5 position-control（T=1/T=5、int32/int64）与 attention `where` 的逐位差分。
- 零漂移：`scripts/qwen_avx2_position_corpus.py` 在两个树上各自编译/执行同一 11 用例
  语料（Qwen position decode/prefill、attention where、f32/bf16 raw-bit compare/where
  矩阵、int8 compare、负 step iota），输出 buffer sha256 **0 处不一致**。
- 旧/新 reader 互拒：`scripts/artifact_emit.py` + `scripts/artifact_reader_probe.py`
  用两个真实 tree 的 decoder 做 4 向验证。
- 性能：`scripts/perf/avx2_position_microbench.py`（L0 口径，UNGATED）给出
  native vs host-reference 的 median/p95/cv 与 native-plane 计数器；结论只作
  "是否消除 host 逐元素 Python 路径"的计数证据，不作绝对性能声明。
- 规则 8：focused（本文件 + AVX2/AVX-512 回归）与全量
  `python/tests/ut/pypto_x` 结果见 `brief.zh-CN.md`。

## 6. 边界与未覆盖

- float16/float64 的 iota/compare/where：AVX2 lowering 本身不支持（与基线一致），
  不是本切片的回退项。
- 非 bool 条件的 `where`（含 float NaN 条件）：shared validator 在 lowering 拒绝，
  基线同样拒绝；本切片刻意不改变该 contract。
- rank > 16 的三种算子回退到 host reference（带原因记录）。
- AVX-512 的同缺口未在本切片处理；未触碰 AVX-512 代码与其快照。
- 本机 12 vCPU KVM 无 cpufreq：性能数字永久 UNGATED。
