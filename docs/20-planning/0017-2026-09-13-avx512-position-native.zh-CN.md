# 0017 AVX-512 位置控制原生面（payload 8）契约

- 任务：`avx512-position-native`（batch 0047 切片 C）
- 分支/工作树：`work/avx512-position-native` @ `/home/chiro/projects/pypto/worktrees/pypto-x/avx512-position-native`
- 交付基线：integration `3c2756c0e`（AVX2 切片落地后的 tip；原切片起点为 `393d28e0f`，tree `eebb0d584c6f1d8088637a4f7c59a484e49e5bd7`）
- 语义真源：`CpuScalarRuntime` / vector-common host reference，逐位一致，不允许"顺带修正"

## 1. 目标与不变量

AVX-512 后端此前把 `iota` / `compare` 记为 `host_reference` 并把 `where` 记入
`host_preexpanded_operations`。本契约把三者升级为原生控制面，同时保持：

1. 与 host reference **逐位一致**（含 NaN 位型、sNaN 静默、±0、subnormal、有符号极值、负 step、多轴/广播）。
2. artifact 元数据**逐 op** 如实描述 native / fallback；`mode == "native"` 与"零回退"互为充要。
3. 新 payload/描述符版本严格解码：旧 reader 读新 payload fail-closed；新 reader 拒绝旧
   `host_reference`-only payload，绝不把 host 路径读成 native。
4. 不修改 `cpu_avx2.py`；不做跨目标共享抽象重构；不新增环境变量。

## 2. 版本与 wire

| 项 | 旧 | 新 |
| --- | --- | --- |
| `artifact_payload_version` | 7 | **8** |
| `NATIVE_ABI_MARKER` / ELF `ptx_avx512_abi` | `pypto-x.cpu.avx512:8` | **`pypto-x.cpu.avx512:9`** |
| `control_descriptor_wire` | 无 | **`v1:pxio-pxcp-pxwh+fnv1a64`** |

描述符字节布局与已验收的 SVE256 控制 wire 完全一致（小端）：

- `PXIO` v1（64 B）：magic/version/axis/dtype/reserved/rank/output_elements/start/step/inner_stride/dimension；
- `PXCP` v1（56 B header + rank×3×u64）：两个输入 dtype、predicate、左右元素数与 strides、输出 shape；
- `PXWH` v1（64 B header + rank×4×u64）：condition/value/output dtype、三源元素数与 strides、输出 shape；
- 每个描述符附 FNV-1a 64 校验值（offset basis 与乘法常数同 SVE256 实现），native kernel 在读取任何元素前校验。

`where` 的 condition dtype 在 Core IR 校验里被限定为 `bool`；但 native kernel 对描述符允许的
所有 condition dtype 都实现了 host `_where_values` 的 Python 真值（NaN 为真、±0 为假、非零为真），
保证前端放松校验时不会与 host 分歧。

## 3. 覆盖矩阵（dtype × kind）

| kind | native dtype | 原生 kernel | 回退条件（带 reason） |
| --- | --- | --- | --- |
| `iota` | `int32`, `int64` | `ptx_avx512_iota_control` | rank=0 / rank>16 / 输出非 tensor / int32 序列越界 / int64 端点溢出（lowering 结构化拒绝） |
| `compare` | `float32`, `bf16`, `int8/uint8`, `int16/uint16`, `int32/uint32`, `int64/uint64`, `bool`（bool 仅 eq/ne） | `ptx_avx512_compare_control` | rank>16 / 输出非 tensor（scalar） |
| `where` | value dtype 全部上表 dtype；condition dtype 全部 wire dtype | `ptx_avx512_where_control` | rank>16 / 输出非 tensor（scalar） |

回退逐 op 记入 `position_control_fallback_operations`，字段为
`{operation_index, operation, kind, mode:"host_reference", dtype, reason}`；
`position_control_fallback` 取 `not_applicable | native | mixed | host_reference`。

## 4. 语义边界

### 4.1 compare

- 依赖 `compare_values` 的 Python 比较：`eq/ne/lt/le/gt/ge`；NaN 下 `==` False、`!=` True、其余 False。
- native 映射：f32/bf16 用 `_mm512_cmp_ps_mask`（bf16 精确拓宽到 f32）；整数/布尔拓宽到 64-bit
  后用 `_mm512_cmp_epi64_mask` / `_mm512_cmp_epu64_mask`；NE 使用 unordered 语义（`_CMP_NEQ_UQ`）。
- 输出 bool 为 1 字节 0/1；bool 输入先按 host 规则真值化（非零即真）再比较，避免非规范字节造成分歧。

### 4.2 where

- condition 判真 = Python 真值：bool 字节非零、整数非零、f32/bf16 `!= 0.0`（NaN 为真、±0 为假）。
- 选择用 opmask：f32/bf16 走 `_mm512_mask_blend_ps`（bf16 在 f32 域展开/回写），整数走
  `_mm512_mask_blend_epi64`，随后按输出 dtype 截断回原宽。
- sNaN 静默：host 会把选中值过一遍 Python float（`_cast_value`），sNaN 被置 quiet 位；
  native 对选中结果做同样的 quiet-NaN 变换（保留符号与 payload），bit-exact。
- 广播/标量源用 rank 对齐的 0/1 strides 描述，单 kernel 内以 native 标量 odometer 收集。

### 4.3 iota

- host 用 Python int 精确计算并在 lowering 校验 dtype 范围；越界是结构化拒绝，不是回绕。
- native 用 `__int128` 计算 `start + coordinate*step` 后按 dtype 截断，覆盖
  `start=INT64_MIN, step=INT64_MAX` 等 host 合法但 C int64 中间乘积会溢出的组合。

## 5. 严格解码与 fail-closed

- 运行时新 decoder `_decode_position_control_metadata` 在能力探测/缓存复用前执行：
  - 缺失 `control_descriptor_wire` → 明确报 "legacy host_reference payload" 拒绝；
  - wire 版本不匹配、native/fallback 表结构与 count 不一致、kernel 名不匹配、fallback 缺 reason、
    `execution_modes` 与逐 op 表矛盾 → 全部拒绝；
  - `position_control_fallback` 必须与表的存在性自洽。
- native C decoder 对每个描述符校验 magic/version/reserved/rank/ shape-product/strides 上界/FNV 校验值；
  任一项失败返回非零，Python 侧抛 `CpuAvx512ExecutionError`，不执行任何元素读写。
- 旧 reader 读新 payload：payload 8 ≠ 旧常量 7，`_validate_primary_semantics` 首查即拒绝（见证据）。

## 6. 测试与证据

- `python/tests/ut/pypto_x/test_cpu_avx512_position_native.py`：差分矩阵（6 predicate × f32/bf16/i8/int32 ×
  普通/±0/±inf/NaN 各位型/subnormal/极值）、iota 负 step/多轴/零长度/极值、where 广播与 NaN/±0 条件、
  元数据 native-iff-zero-fallback、payload/描述符攻击、与基线树逐位一致、UNGATED 计时。
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/avx512-position-native/`
  （`brief.zh-CN.md`、`raw/`、`logs/`、`scripts/`）。

## 7. 登记边界（未覆盖/不声称）

- 不启用 AVX-512 DQ/BW/VL 变体；native 面只依赖 AVX-512F（+可用 FMA），全部 kernel 在默认 variant 下可编译。
- `iota` 的 native kernel 是 native 标量循环（名称/描述符/边界检查在 ELF 内），未做 ZMM iota 向量化。
- 广播/非连续源的 lane 收集是 native 标量 odometer，不是 gather 指令；性能不作为门槛（UNGATED）。
- 未覆盖：`iota` 的 float dtype（Core IR 不支持）、`compare` 混合输入 dtype（Core IR 拒绝）、
  rank>16 与 scalar 输出的位置控制仍走 host reference（已逐 op 登记 reason）。
- 本机为 KVM guest、无 cpufreq，绝对性能永久 UNGATED；AVX-512 频率/降频不确定性不纳入判定。
