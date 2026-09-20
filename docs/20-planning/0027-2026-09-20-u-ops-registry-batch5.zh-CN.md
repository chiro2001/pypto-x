# 0027 — U 线算子注册第五批（batch5）：view 家族收口 + `copy` 别名 + W8A8 三 primitive + 27/27 桥接算子覆盖

- 任务：`u-ops-registry-batch5`（batch 0054 切片 A）
- 实现基线：integration tip **`a55a005eb`**（tree `464e3a7149cb873a048c137799e80faa41384986`，0053-A 多输出 ABI + `split` 已落地）
- 分支/worktree：`work/u-ops-registry-batch5`（`/home/chiro/projects/pypto/worktrees/pypto-x/u-ops-registry-batch5`，未 push）
- 依赖：`0019`（schema v3 契约、按 opcode 收窄的 capability 视图）、`0022`（视图语义三态与
  `may_alias`+物化）、`0023`（batch3 的契约/精度口径）、`0025`（batch4 单输出四算子）、
  `0026`（多输出 ABI 与 `split`）、`docs/20-planning/0002`/`0010`（冻结 W8A8 v1/v2 契约）
- 结论等级：**实现完成（静态 shape；portable provider 路径）**。未做性能声明；所有本机数字 **UNGATED**
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/u-ops-registry-batch5/`

## 0. 摘要

本切片按父方 0054 指派完成三件事：

1. **view 家族收口**：登记 `view` 与 `contiguous`（PIL 桥接词表最后两个未注册的真实 Core IR 拼写），
   复用 0022 冻结的视图语义（`view_obligation=may_alias`、显式物化、copy 字节数进 plan digest、
   `view_mode` 三态，`require` 无证明时结构化拒绝、绝不静默降级）；两者都是 `exact` 数据搬运。
2. **`copy` 覆盖**：独立审计确认 Core IR 拼写 `copy` 自 M1B 起就是 lowering `OP_ALIASES` 中
   `copy -> identity` 的冻结别名，`plan_operation("copy", ...)` 在父树即可规划/执行，已被冻结的
   `identity` 契约覆盖（`may_alias`+物化、copy 字节=element_count×dtype_size 进 digest、`exact`）。
   本切片**不**新建重复契约文档（避免与冻结 `identity` 契约争同一别名、避免改 29 契约 digest），
   而是把该别名归属写成 machine-readable 清单条目并新增专项测试钉住。
3. **W8A8 三 primitive**：登记 `quantize_per_token_s8`（两结果：int8 codes + fp32 per-row scales）、
   `dequantize_s8`、`dequantize_epilogue_bf16`。三者严格对齐冻结的 W8A8 v1/v2 契约
   （0002 §1.4 Q1–Q6、§3.3；0010 的 scale/布局语义）；精度声明诚实：含舍入 ⇒
   `deterministic_bounded` 且**无解析上界**，`require_proven_deviation_bound=true` 下 fail-closed，
   **不声明 exact**。

**主验收指标（硬）**：机器可读清单的 27/27 个父方桥接 Core IR 名字全部解析到已登记契约
（`copy` 为 `identity` 的 Core IR 别名，其余为 canonical）；另有 supplemental `view` 也登记，
因此非 `matmul` 的 PIL 桥接词表实际为 **28/28**。证据：`raw/batch5_checklist.json` 与
新测试文件。

执行层 opcode 契约从 29 个（含 `split`）扩到 **34 个**，新增 5 个契约文档；`copy` 不新增文档。

### 0.1 冻结兼容策略（关键设计决定）

1. **5 个新契约**沿用 registry 级 schema **v3**（`0019 §0.1`）：`contract_version="1.0"`、
   `allowed_providers=("portable",)`、`effects=pure`；29 个既有契约的 document/digest 一字节不改。
2. **capability 仍按批次冻结累积集合**：新增 `_BATCH6_SCOPED_CONTRACTS`（0054-A 的 34 契约视图，
   29 + 5 新契约）；batch1→10、batch2→16、batch3→24、batch4→28、batch5(`split`)→29 的收窄视图
   digest 与父树逐字节一致。默认 probe（`FROZEN_PORTABLE_CONTRACTS`）仍只声明 matmul/qmatmul 两个契约，
   默认 manifest digest `d48ce0df…` 不变。
3. **`copy` 不新建 canonical 契约**：lowering 的 `OP_ALIASES` 把 `copy`、`tensor.copy`、
   `core.copy` 冻结映射到 canonical `identity`；`identity` 契约的 `core_ir_opcodes` 已包含这些拼写。
   新建第二份 canonical `copy` 契约会造成同一 Core IR 拼写被两个契约声明的歧义，并使
   `plan_operation("copy")` 的既有别名决策发生计划外变化；因此本切片只做覆盖登记 + 行为钉住。
4. 不修改 `python/pypto/execution/region.py`、`python/pypto/portable/qwen35.py`、
   `python/pypto/__init__.py`、编译器 target / 后端原生实现；不新增环境变量。

## 1. 六个 opcode 的契约表

所有新契约：`allowed_providers=("portable",)`、`effects=pure`；属性白名单之外一律
`unknown_attribute`；序列型属性只接受有序 list/tuple（set/generator/mapping 结构化拒绝）。
`view`/`contiguous` 声明 `may_alias` + 允许物化；`copy` 沿用 `identity` 的同一声明；
三个 W8A8 primitive 是计算类，声明 `inputs_readonly_outputs_distinct`（`must_not_alias`）。

| opcode | 元数/结果 | shape 规则 | dtype 域 | 结果 dtype | 属性白名单 | portable executor 符号 |
|---|---|---|---|---|---|---|
| `view` | 1 in / 1 out | rank 1..4；`element_count(out)==element_count(in)`；`offset(s)` 缺省或全 0 且长度=输入 rank | 全部 portable dtype（含 bool）；in=out | = input | `shape`/`target_shape`/`out_shape`（一致，或由声明输出给出）、`offset`/`offsets`（全 0）、`inplace`（只 false）、`valid_shape`/`valid_shapes`（必须等于输出 shape） | `pypto.backends.cpu.runtime._reshape_value` |
| `contiguous` | 1 in / 1 out | rank 1..4；`shape(out)==shape(in)`；dtype 相同 | 全部 portable dtype（含 bool）；in=out | = input | `inplace`（只 false）、`valid_shape`/`valid_shapes`（必须等于输出 shape） | `pypto.backends.cpu.runtime._copy_value` |
| `copy`（别名） | 1 in / 1 out | 同 `identity`：`shape(out)==shape(in)` | 全部 portable dtype（含 bool） | = input | 同 `identity`：`inplace`（只 false）、`valid_shape`/`valid_shapes` | `pypto.backends.cpu.runtime._identity_value`（冻结 `identity` 契约） |
| `quantize_per_token_s8` | 1 in / **2 out**（多输出 ABI） | rank 2 `[rows,K]`，`rows>=1`、`K>=1`、`K<=133144`（Q5）；codes=`[rows,K]`、scales=`[rows]` | input ∈ {bf16, float32} | codes=int8；scales=float32（固定 0 zero-point） | 无（未知属性拒绝） | `pypto.backends.cpu.runtime._quantize_per_token_s8_value`（包装冻结 C1 golden） |
| `dequantize_s8` | 2 in / 1 out | codes rank 1..4 `[...,K]`，rows=`prod(leading)` 或 1；scales 形状必须恰为 `[rows]`；out 形状=codes 形状 | codes=int8；scales=float32 | float32 | 无 | `pypto.backends.cpu.runtime._dequantize_s8_value` |
| `dequantize_epilogue_bf16` | 3 in / 1 out | accumulator rank 2 `[m,n]`，`m>=1`、`n>=1`；row scales=`[m]`、column scales=`[n]`；out 形状=`[m,n]` | acc=int32；两个 scales=float32 | bf16 | 无 | `pypto.backends.cpu.runtime._dequantize_epilogue_bf16_value` |

`contract_digest`（本树，`raw/batch6_record.json`）：

```text
view                     sha256:aa15e1810a75eb90d9c3c0302b5f590316e8b0815049b69f6e5b20a604933333
contiguous               sha256:ea636d18e68808d71b17d1641627e8b91137d9dcdd752c37a08302d2cedfa49f
quantize_per_token_s8    sha256:9b2c4f1c506fff4df010288b4b6c2c309ca447d825294eb261fa9152980acaec
dequantize_s8            sha256:4e02b43cc66cd2509f8abe246770bb35785a7b895929586c1cff23a39d38355c
dequantize_epilogue_bf16 sha256:d63cc9827736c66ad148264cefede4f689c91c4db1a7ad98b9659d7099c19fd5
copy -> identity         sha256:e1a7e4f73600ede7097134af98d5f02d9a6584d5ab596b527dfce1346300b561（冻结，未改）
```

按批收窄的 capability 视图 digest（本树，`raw/batch6_record.json` / 测试内钉住）：

```text
view                     sha256:03b793a88a81710a9d190335250983d4b828aadede8321ad9a0b025c6ca5c099（34 契约）
contiguous               sha256:0671098a3086b46851772c61d6208e0fcb1f84fa18b4dd704d0bb51bb2c85071（34）
quantize_per_token_s8    sha256:01afee72a782ea54ddd838bdec7e98d40a6e0c9aaa40c96b2a223bae76caf73d（34）
dequantize_s8            sha256:ef92c0a126704f3ba42d9b1cf74148417eaeed3f3e8a796919541fcdbde2f9e0（34）
dequantize_epilogue_bf16 sha256:86fdeac8857da8bc722e6aecbcc087ba44b6cc3b427e3151705603225292ca21（34）
split                    sha256:f73c337c9fe9824a72d607a6b67fff7b5b847bd75667bf4fd8964da3ac847e43（29，不变）
```

`view`/`contiguous` 的 copy 字节数 = `element_count(output)×dtype_size(output_dtype)`，由冻结的
`request.output_shape`/`output_dtype` 复算并进入 `layout_decision`/plan digest；`view_mode=require`
时 portable trunk 无 stride ABI，结构化拒绝 `ALIAS_PROOF_UNAVAILABLE /
portable_trunk_has_no_strided_view_abi`，绝不静默降级为拷贝（`view`/`contiguous` 两者同口径）。
`layout.max_copy_bytes` 预算不足时 `forbid`/`prefer` 也拒绝（`max_copy_bytes_exceeded`）。

### 1.1 `copy` 的语义归属（审计结论）

- `plan_operation("copy", [(2,3)], ["float32"])` → plan 的 `opcode="identity"`，`contract_digest`
  等于冻结 `identity` digest `sha256:e1a7e4f7…`；`layout_decision` 为
  `may_alias` + `resolved_view_mode="materialized"` + `layout_copy_bytes=24`；`precision.class="exact"`。
- 执行输出与 `identity`（及 `_identity_value`）逐位一致；NaN/±0/±inf/次正规/bf16 位型原样保留。
- 本切片新增测试钉住上述四件事；machine-readable 覆盖清单把 `copy` 标为
  `{"opcode":"identity","via":"core_ir_alias","covered":true}`。
- **登记差异**：父方 0054 指派文中“当前 25/27、最后两个是 `contiguous` 与 `copy`”是按
  canonical 名精确匹配的口径；按契约别名归属审计，父树上 `copy` 已经可规划、真实缺口是
  `contiguous` 与 `view`（后者作为 supplemental 在本切片一并登记）。本切片如实给出两种口径，
  不把“别名已覆盖”写成“新增了第二份 copy 契约”。

### 1.2 `view` 是 supplemental 登记

父方 27 名单（见 §4）不包含 `view`；但 `_PORTABLE_OP_NAMES` 的 29 个 Core IR 拼写包含它，
0025 §9 / 0026 §9 也把 `view`/`contiguous` 列为最后两项。本切片在满足 27/27 硬指标之外
**额外**登记 `view`（与 `contiguous` 同一视图语义），因此非 `matmul` 的桥接词表实际
**28/28** 全覆盖；这是加法，不影响 27/27 清单。

## 2. 精度声明（诚实口径）

| opcode / 路径 | 类别 | bound kind | 理由 |
|---|---|---|---|
| `view` 全部 | **exact** | `exact_contract_bound`（proven，bound=0） | 只改变逻辑 shape，flat 行主序值序列逐位保留；不做算术/转换/重排 |
| `contiguous` 全部 | **exact** | 同上 | 满 shape 行主序复制，逐位保留 |
| `copy`（`identity` 路径） | **exact** | 同上 | 冻结 `identity` 契约的逐位恒等复制 |
| `quantize_per_token_s8` | `deterministic_bounded` | `deterministic_serial_order_no_registered_analytic_bound` | 每行 `s=float32(absmax/127)`、比值一次 RNE 后 clamp；scale 的 binary32 舍入与 RNE 无注册解析包络，**无解析上界**，观测绝不当界 |
| `dequantize_s8` | `deterministic_bounded` | 同上 | 每元素 `float32(code*scale)`（binary64 乘法 + 一次 binary32 RNE）；**无解析上界** |
| `dequantize_epilogue_bf16` | `deterministic_bounded` | 同上 | R4：scale 乘积 binary32、int32→binary32 乘法 binary32、最后单次 BF16 RNE；**无解析上界** |

`numeric.require_proven_deviation_bound=true` 时：

- `view`/`contiguous`/`copy`（exact）正常解析；
- 三个 W8A8 primitive **fail-closed**：`NUMERIC_GUARANTEE_UNMET /
  proven_deviation_bound_unavailable`（field `numeric.require_proven_deviation_bound`），
  portable provider 也生效。

## 3. 与冻结 W8A8 契约（0002 / 0010）的一致性

以冻结契约 0002 §1.4 / §3.3 为准逐条核对：

| 冻结规则 | 本切片实现 | 结论 |
|---|---|---|
| Q1 整行全零：`s=1.0`、`q=0`，不得 NaN/Inf | C1 golden 原样复用；tests 钉 `[0,0] -> codes 0 / scale 1.0` | **一致** |
| Q2 行内 NaN/Inf：显式报错、不静默 clamp | 执行期结构化 `OP_CONTRACT_INVALID / quantize_non_finite_input` | **一致**（reason 是本切片新增的可观测拼写） |
| Q3 clamp 到 `[-127,127]`，饱和计数上报 | clamp 域一致；饱和计数是冻结 C1 artifact 字段，两 tensor 结果的 ordered ABI **无法携带** | **数学一致；ABI 差异已登记**（§3.1） |
| Q4 RNE ties-to-even（`rne(0.5)=0`、`rne(1.5)=2`、`rne(-0.5)=0`） | 直接复用 `qwen35_w8a8_scheme` 的 RNE；tests 钉 63.5→64、-63.5→-64 | **一致** |
| Q5 `K <= floor((2^31-1)/(127*127)) = 133144`，超过显式拒绝 | 契约期 `w8a8_quantize_k_exceeds_int32_accumulation_limit`；`K=133144` 可规划 | **一致** |
| Q6 非 2-D 输入先 reshape 到 `[rows,K]` | 契约 rank 恰为 2；其他 rank 结构化 `rank_out_of_range`，不隐式 flatten | **一致**（更严格：拒绝而非隐式 reshape，符合“builder 先 reshape”的分工） |
| R4：epilogue 单次 BF16 舍入，与“FP32 乘完再 cast”逐位一致 | 直接复用 golden，contract 明确 binary32 计算顺序；tests 逐位差分 | **一致** |
| scale dtype/形状：FP32、per-row `[rows]` | codes/scales dtype 与形状固定；`dequantize_s8` scales 必须恰为 `[rows]` | **一致** |
| per-token（激活 per-row）vs per-channel（权重） | 契约只表达 per-row 激活量化/反量化；权重 per-output-channel scale 通过
epilogue 的 column scales 参与，不在 primitive 内做权重量化 | **一致**（0010 的 packed `[K,out]` 属 binding/layout 层，不是 opcode 语义） |
| codes 取值域 `[-127,127]`（-128 保留） | `dequantize_s8` 执行期拒绝 `-128`：`dequantize_code_out_of_range` | **一致**（决策相同，reason 结构化） |
| `dequantize_s8` 诊断语义 | 明确定义 `int8[..,K] + fp32[rows] -> fp32`，供单测/差分 | **一致** |

### 3.1 登记的 ABI 差异（唯一一处）

冻结契约 0002 Q3 要求把 clamp 饱和计数作为 artifact/日志字段上报。本切片的执行 ABI 是
0024/0026 的 ordered HostTensor 结果（这里恰为两个 tensor：codes、scales），没有第三个
side-channel 输出或运行时 artifact 附件，因此：

- **clamp 语义与 `[-127,127]` 域完全按冻结契约实现**（C1 golden 唯一实现），不会静默发布
  域外码点；
- **饱和计数不在本契约的返回 ABI 中**；需要该字段的调用方必须直接使用 C1 golden
  （`pypto.portable.qwen35_w8a8_scheme.quantize_per_token_s8`，其 `saturation_count` 字段保留）；
- 参考量化路径（per-row absmax）下饱和计数为 0（scale 的 binary32 舍入达不到 127.5 的
  RNE 边界），测试对此有记录；clamp helper 自身的截断/计数有独立钉住用例。
- 这是**已登记差异**，不改冻结契约、不升 opcode 契约版本；若未来需要把计数器带进执行
  ABI，应新开契约版本或多输出 ABI 扩展，不在本 slice 静默扩宽。

### 3.2 与 `docs/10-architecture/0006` 精度分类的差异（登记）

0006 §13.9（第 5 答 / B-3）把 W8A8 primitive 的精度分类冻结为：`qmatmul_s8s8_s32` = `exact`、
`quantize_per_token_s8` = `exact`（RNE + absmax + 码值域 `[-127,127]`、拒 `-128`）、
`dequantize_epilogue_bf16` = “相对契约声明的单次 RNE 序为 exact、相对其它乘序为 bounded”。
本切片**没有**沿用这两个 `exact` 标签，而是声明 `deterministic_bounded` + 无解析上界；
这是 0054 指派明确要求的“honest rounding bounded declarations”，差异登记如下：

- **语义层面完全一致**：RNE ties-to-even、per-row absmax、码域 `[-127,127]`、拒 `-128`、
  epilogue 的“先形成一次 `s_a*s_w`、再与 accumulator 相乘、最后单次 RNE”顺序，全部按 0002/0006
  与 C1 golden 实现（§3 逐条核对表）；本切片没有改任何冻结契约文档。
- **分类层面的差异**：`quantize_per_token_s8` 的 scale 计算含一次 binary32 RNE、codes 含一次
  RNE + clamp；`dequantize_s8`/`dequantize_epilogue_bf16` 含 binary32 乘法/舍入与 bf16 RNE。
  相对“实数域参考语义”而言偏差不是 0，本执行层也没有注册可证明的解析包络，因此不能声明
  `exact_contract_bound`；按本切片口径一律 `deterministic_bounded`，`require_proven_deviation_bound=true`
  时 fail-closed。0006 的 `exact` 应理解为“实现必须逐位等于声明的参考序”（可复现性/顺序冻结），
  而不是“相对数学实数的偏差为 0”；本切片在报告中同时写明 `reduction_order_id`/reference，
  但**不**把它升级为 proven bound。
- **若未来要让机器标签与 0006 的 `exact` 字样完全对齐**，需要先把“相对声明参考序的 exact”
  形式化为一个 proven `deviation_bound_kind`（并说明它不适用于实数域误差），再新开契约版本；
  在此之前，`deterministic_bounded` 是唯一诚实且 fail-closed 的选择。

### 3.3 0006 §13.2 P8 必备字段在本契约中的落点

0006 P8 的必备字段没有直接展开为 registry 的 `quantization` 字段——冻结的
`qmatmul_s8s8_s32` 契约本身只把该字段写作一行标签字符串，因此本切片沿用同一约定
（`view`/`contiguous` 的 `quantization` 为约定值 `"none"`；三个 W8A8 primitive 使用自描述标签字符串
（`s8_per_token_symmetric_absmax_rne_...`、`s8_dequantize_diagnostic_...`、`w8a8_epilogue_...`），
详细规则同时写在 `semantics`/`numeric_contract`/`precision`/reference 四个块里，全部进入 contract digest）：

| 0006 P8 字段 | 落点 |
|---|---|
| `scheme_id` | `portable_reference` 指向 `pypto.backends.cpu.w8a8_reference` → C1 golden `w8a8_linear`；`semantics.difference_vs_frozen_contract` 明确冻结来源 0002/0010 |
| signedness / bits / 保留码值 | `semantics.code_domain`、`semantics.rounding`（signed int8、`[-127,127]`、拒 `-128`） |
| scale/zero-point 粒度、dtype、来源、关联 | `semantics.granularity`/`scale_dtype`/`scale_shape`；zero-point 固定 0 |
| RNE、saturation、saturation count | `semantics.rounding`/`saturation`；RNE = C1 `rne_float_to_int`；clamp 域一致；saturation count 的 ABI 差异见 §3.1 |
| NaN/Inf、zero row、scale underflow | `semantics.nan`/`inf`/`signed_zero`/`empty` 与 `precision.error_statement`；Q1/Q2 行为由 C1 golden 判定 |
| accumulation width / K 上限 / overflow | `numeric_contract.accumulation`、`integer_overflow`、`semantics.k_bound`（K ≤ 133144） |
| logical quantized layout | `semantics` 明确 codes/scales 的 row-major HostTensor 形状；packed `[K,out]`/wire 属 binding/artifact（0010），不在 opcode 表面 |
| logical binding schema reference | 不在 opcode 契约内（binding 层）；`semantics` 明确 “primitive 不暴露 layout/stride 属性” |

`dequantize_s8` 是诊断 primitive（0002 §3.3），没有 accumulation/saturation 字段；
`dequantize_epilogue_bf16` 的 accumulation 字段写作 “int32 输入 + binary32 乘法序 + 单次 BF16 RNE”。

### 3.4 无冲突项

- 0002 的 `qmatmul_s8s8_s32` 契约与 kernel 未改动；本切片的三个 primitive 与其共享同一
  C1 golden（`pypto.backends.cpu.w8a8_reference`），不会出现第二套 RNE/clamp 实现。
- 0010 的 packed `[K,out]` 是权重 binding 布局，不是 opcode；本切片不改 `qwen35.py`/binding，
  也不在 primitive 表面暴露 layout/stride 属性（stride 由 binding 层保证）。
- `dequantize_s8` 对 NaN/Inf scale 不做额外 clamp：乘积按 IEEE-754 传播（冻结契约没有禁止性
  规定），执行层只拒绝域外 code。

## 4. 27/27 覆盖清单（machine-readable）

清单来源：`pypto.frontend.pil_core_bridge._PORTABLE_OP_NAMES` 的 29 个 Core IR 拼写；父方
0054 口径的 27 名单 = 去掉专用冻结契约 `matmul` 与 supplemental `view`。契约实现：
`pypto.execution.bridged_core_ir_coverage()`（每条含 `name/covered/opcode/via/reason/tier`），
证据 JSON：`raw/batch5_checklist.json`。

| 名字（27） | opcode | via |
|---|---|---|
| `add` `broadcast` `cast` `concat` `constant` `div` `embedding` `exp` `gather` `identity` `mul` `neg` `reduce_max` `reduce_mean` `reduce_sum` `reshape` `rsqrt` `sigmoid` `silu` `slice` `softplus` `split` `sub` `transpose` `where` | 同名 canonical 契约 | `canonical` |
| `contiguous` | `contiguous`（本切片新增） | `canonical` |
| `copy` | `identity`（冻结契约的 Core IR 别名） | `core_ir_alias` |
| **supplemental** `view` | `view`（本切片新增） | `canonical` |

结论：**27/27 covered（0 uncovered）**；`registered_opcode_count=34`；非 `matmul` 桥接词表
28/28。父树审计口径：
- 按 canonical 名精确匹配：25/27（`contiguous`、`copy` 未作为独立 canonical 名存在），
  与父方数字一致；
- 按契约别名归属：26/27（`copy` 已由 `identity` 契约覆盖，`plan_operation("copy")` 父树即可用），
  真实缺口为 `contiguous` 与 `view`；
- 本切片后两种口径都满足：27/27（清单内）+ 28/28（含 supplemental `view`）。

## 5. fail-closed 矩阵（新增触发器）

| 触发 | 错误（code / reason） | 发生层 |
|---|---|---|
| 未知 opcode / 合成名 | `OP_CONTRACT_INVALID / opcode_not_registered` | registry/resolver |
| `view` 无目标 shape（属性与声明输出都缺） | `view_target_shape_missing` | verifier |
| `view` 目标 shape 与属性冲突 / element count 不等 | `view_target_shape_mismatch` / `view_element_count_mismatch` | verifier |
| `view` offset(s) 非全 0 / 长度不等于输入 rank / 非有序容器 | `view_offsets_not_portable` / `view_offsets_invalid` / `unordered_attribute_container` | verifier |
| `view`/`contiguous` `inplace=true`、不可表示 `valid_shape` | `view_inplace_not_portable` / `contiguous_inplace_not_portable` / `*_valid_shape_not_representable` | verifier |
| `contiguous` 输出 shape/dtype 与输入不一致 | `shape_rule_violation` / `dtype_mismatch` | verifier |
| view 家族 `view_mode=require` 无零拷贝证明 | `ALIAS_PROOF_UNAVAILABLE / portable_trunk_has_no_strided_view_abi` | layout_verify |
| view 家族物化超 `layout.max_copy_bytes` | `ALIAS_PROOF_UNAVAILABLE / max_copy_bytes_exceeded` | layout_verify |
| `quantize_per_token_s8` 非 bf16/fp32、非 rank-2、空 rows/K | `dtype_outside_contract_domain` / `rank_out_of_range` / `w8a8_quantize_empty_input_not_supported` | verifier |
| `quantize_per_token_s8` `K > 133144` | `w8a8_quantize_k_exceeds_int32_accumulation_limit` | verifier |
| 声明 outputs 个数/name/role/dtype/shape/kind/unknown 字段/重复名 | `output_count_mismatch` / `output_name_invalid` / `output_role_mismatch` / `multi_output_scalar_result_not_supported` / `output_dtype_mismatch` / `output_shape_mismatch` / `output_declaration_unknown_field` / `duplicate_output_name` / `output_list_empty` | verifier |
| 多输出缓冲数量/形状/dtype 不符 | `plan_request_mismatch` / `unsupported_output_handle` | entry/provider |
| `dequantize_s8` codes/scales dtype 越域、scales 形状不是 `[rows]`、空 rows/K | `dtype_outside_contract_domain` / `w8a8_dequantize_scales_shape_mismatch` / `w8a8_dequantize_empty_input_not_supported` / `rank_out_of_range` | verifier |
| `dequantize_s8` 执行期 code ∉ `[-127,127]`（含 -128） | `OP_CONTRACT_INVALID / dequantize_code_out_of_range` | portable provider |
| `quantize_per_token_s8` 执行期 NaN/±Inf / scale 下溢 0 | `quantize_non_finite_input` / `quantize_scale_underflow` | portable provider |
| `dequantize_epilogue_bf16` dtype/rank/形状/空输入不合法 | `dtype_outside_contract_domain` / `rank_out_of_range` / `w8a8_epilogue_scales_shape_mismatch` / `w8a8_epilogue_empty_input_not_supported` | verifier |
| `dequantize_epilogue_bf16` 执行期 accumulator 越 int32 | `dequantize_accumulator_out_of_range` | portable provider（HostTensor 构造通常更早拒绝域外 int32，运行时守卫直接测试 executor 符号） |
| `require_proven_deviation_bound=true` 遇三个 bounded W8A8 路径 | `NUMERIC_GUARANTEE_UNMET / proven_deviation_bound_unavailable` | resolver |
| plan 被篡改（quantize `request.outputs[1].dtype`、view `layout_copy_bytes`） | `ARTIFACT_MISMATCH / plan_digest_mismatch` | plan 校验器 |
| report `outputs[i].digest` 被改写 | `REPORT_SCHEMA_INVALID` | report 校验器 |

## 6. 零默认变化

对照基线：父树 = integration `a55a005eb`（本树修改前同 tree）。测试内硬钉：

- **29 个既有契约 digest**：`test_the_twenty_nine_existing_contract_digests_are_byte_identical`
  逐项与父树值相等（含 matmul/qmatmul 与 `split`）；
- **默认 portable manifest digest** unchanged：`d48ce0df6d99200209a9ac561ddeb40b65f521ea8207fb7e68b928805b0c9f7c`；
- **`split` 的 29-contract 收窄视图 digest** unchanged：
  `f73c337c9fe9824a72d607a6b67fff7b5b847bd75667bf4fd8964da3ac847e43`；
- batch1–batch4 收窄视图：`exp` 10、`add` 16、`neg`/`concat`/`constant` 24、
  `div`/`identity` 28（digest 与父树相同）；新契约使用 34-contract 视图；
- `record_batch5_zero_drift.py` 在父树与本树各跑一次（默认环境与 local 锁 `OMP_NUM_THREADS=6`
  各一对），覆盖 29 个契约 digest + 10 个 capability 视图 + 30 个 frozen run
  （每个契约至少一条执行，含 matmul f32/bf16、qmatmul、`split` 双输出与 scalar 操作数）；
  `compare_zero_drift.py` 结论：**ZERO-DRIFT**（只豁免按设计增长的 `registered_opcodes`）。
- 新增 opcode 后 `registered_opcodes` 从 29 → 34，这是清单增长本身，不是既有契约漂移。

非空转证据：新测试文件在父树 collect 即 `ImportError: cannot import name
'BATCH6_CANONICAL_OPCODES'`（rc=2）。

## 7. 验证与证据

### 7.1 新增测试（`python/tests/ut/pypto_x/test_execution_ops_registry_batch5.py`，87 项）

覆盖：registry/34 契约/29 digest 钉/按批 capability 视图/27+1 清单与 PIL 词表推导；`view`/
`contiguous` 的逐 dtype 逐位差分（bool/int8/int64/f16/bf16/f32/f64）、NaN/±0/±inf/次正规/
bf16 位型（0x0001/0xFF80/0x7FC0）、整数极值、空张量与零元素；`view_mode` 三态与 copy 字节预算；
`copy` 的别名归属/位型/`require` 拒绝；`quantize_per_token_s8` 的 golden 差分、全零行、RNE tie、
NaN/Inf/scale 下溢、K 界（133144/133145）、声明 outputs 交叉检查、显式缓冲与别名拒绝、
report `outputs` 块与冻结 plan JSON 重放；`dequantize_s8` 的 rank1–4 差分、特殊 scale（±0/±inf/NaN）、
code 域（-128/+128）；`dequantize_epilogue_bf16` 的 golden 位型差分、R4 单次舍入公式、
巨大 int32 端点、运行时越界守卫；三个 bounded primitive 的
`require_proven_deviation_bound` fail-closed（exact 路径通过）；view 家族 region plan 与
`validate_region_plan(re_resolve=True)` 往返；Core IR `verify_operation`；plan/report 自包含与篡改拒绝。

- focused（默认环境，`-k execution`）：**1100 passed / 0 failed / 1734 deselected，48.01 s**
  （0053 基线同口径 1013 passed，差值 +87 全部为本切片新测试；`logs/focused_execution_default.log`）。
- focused（`OMP_NUM_THREADS=6`，未单独抢锁）：**1093 passed / 7 skipped / 0 failed**（47.06 s；
  7 个 skip 为 host-probe 守卫的冻结 digest 钉，与本批其他切片口径一致；
  `logs/focused_execution_omp6.log`）。
- 本切片文件单独运行：**87 passed**；另加 execution/w8a8-graph/cpu-scalar/core-ir/tensor-bridge/
  op-bench/composite 的合并集合 → **1240 passed / 0 failed**（58 s）。
- 规则 8 全量（local 锁、`OMP_NUM_THREADS=6`、单次调用）：见 §7.2。
- 父提交非空转：`logs/parent_a55a005eb_new_tests_error.log`。

### 7.2 规则 8 全量（local 锁）

- collect：**2834**（父树 2747 + 本切片 87）；`logs/rule8_collect.txt`。
- 运行：**2820 passed / 14 skipped / 0 failed / 0 errors，rc 0，1248.77 s（20:48）**
  （`logs/rule8_single_summary.txt`、`logs/rule8_rc.txt`、`raw/rule8_counts.json`）。
  14 个 skip 与 0053 基线一致：7 个 CUDA driver 不在本机（`test_cuda_qwen_c2.py`）
  + 7 个 host-probe 守卫的冻结 digest 钉（batch1 ×2、batch2 ×1、batch3 ×1、batch4 ×2、
  region 单算子基线 ×1）；本切片新增文件无 host-guard skip。
- 与父提交基线对照：collect 2747 → **2834**（+87 = 本切片新增测试）；passed 2733 → **2820**（+87）；
  skip 14 → 14；failed 0 → 0。没有任何既有套件回归。

## 8. 主验收指标：真实三图 region 只读规划

证据：`raw/qwen_region_missing_contracts_batch5.json`（脚本
`scripts/qwen_region_missing_contracts_batch5.py`，只读规划、不执行 provider、走 local 锁）。

| 指标 | batch4/0053 基线 | 本切片 |
|---|---|---|
| `missing_contracts` 类/次 | 1 类 / 24 次（`split`，已由 0053-A 登记） | **0 类 / 0 次**（PIL 桥接词表全部已登记） |
| `unplannable` 类/次 | 24 次 `split`（region 多输出前置检查） | **24 次 `split`**（`multi_output_not_supported_by_execution_contract`；`region.py` 未改，step 3 边界） |
| `registered_planned_total` | 4669 | **4669**（attention/gdr 整图 planned；decoder 4550 ops 中 4669 个已登记引用全部 plan 成功） |

逐图（`raw/qwen_region_missing_contracts_batch5.json`）：

| builder | ops | 状态 | missing | unplannable | region_digest |
|---|---:|---|---:|---|---|
| `build_attention_kv_cache` | 20 | planned | 0 | 0 | `92e21b47…`（不变） |
| `build_qwen35_gdr_recurrent_state` | 123 | planned | 0 | 0 | `aa413baa…`（不变） |
| `build_qwen35_text_decoder_graph` | 4550 | rejected（仅因 24 个 `split` unplannable） | 0 | 24 | — |

区域耗时 0.12 s / 0.68 s / 24.39 s，仅本机 UNGATED 观测；`registered_planned_total=4669`
与 `unplannable=24` 与 0053-A（0026 §6）完全一致，说明本切片只增登记表、未改变真实图规划决策。

注意：`view`/`contiguous` 不在三张真实图的 opcode 直方图内（真实图用 `reshape`/`transpose`/
`slice` 等）；本指标验证的是“登记表不再遮挡任何真实图 opcode”，不是“两图使用了新 opcode”。

## 9. 测试/实现层变更（严格加法）

- `operation_contracts.py`：新增 `verify_view`/`verify_contiguous`/`verify_quantize_per_token_s8`/
  `verify_dequantize_s8`/`verify_dequantize_epilogue_bf16` + 多输出声明校验 helper；
  `GENERIC_VERIFIERS` 增加 5 项；跨 lowering 校验复用既有 `verify_lowering_accepts(_multi_outputs)`。
- `op_registry.py`：`_generic_definition` 支持有序多结果 spec（单输出路径逐字节不变）；
  新增 5 个 builder 与 `BATCH6_CANONICAL_OPCODES`；新增 27+supplemental 覆盖常量与
  `bridged_core_ir_coverage()`。
- `capability.py`：新增 `_BATCH6_SCOPED_CONTRACTS`（34）与分派；旧批次视图不变。
- `provider_ids.py`：5 个新 opcode 的实现 id；`split` 的既有 fallback 拼写不变。
- `runtime.py`：新增 4 个结构化值错误子类与 3 个 W8A8 executor 符号；三个 W8A8 分支改为调用
  符号并保留 C1 golden 为唯一实现；`providers.py` 把值错误映射为
  `quantize_non_finite_input`/`quantize_scale_underflow`/`dequantize_code_out_of_range`/
  `dequantize_accumulator_out_of_range`。
- 既有测试的期望更新：`view` 从“未登记示例”改为 synthetic 名
  （`batch1/2/3/4`、`matmul`、`region planning`）；`multi_output` 测试允许第二个多输出契约。
  未改任何既有契约 document/digest，未改 region.py。

## 10. 未覆盖与下一步建议

1. **region 多输出 step 3**（`region.py`，父方另一 agent）：`quantize_per_token_s8` 是新的
   两结果 Core IR 算子，区域层仍按“多输出前置检查”记为 `unplannable`
   （`multi_output_not_supported_by_execution_contract`），与 `split` 同一依赖；ABI 级两结果
   plan/execute/report 全通。step 3 落地后应同时覆盖 `split` 与 `quantize_per_token_s8` 的
   `(op_index, output_name)` 边界。
2. **`copy` canonical 名**：当前是 `identity` 的 Core IR 别名；若某前端必须要求“独立 canonical
   copy 契约”，需要先解决 lowering 别名唯一性与 29 契约冻结问题（提案，不在本切片）。
3. **vendor / vector 原生路径**：5 个新契约 `allowed_providers=("portable",)`；AVX2/AVX-512/
   SVE/CUDA/AMD 候选结构化拒绝。W8A8 的向量/native 路径已存在于 graph-path，但未接入本 registry。
4. **饱和计数 side channel**：见 §3.1；若需要进入执行 ABI，需新开契约版本或多输出扩展。
5. **动态 shape / 运行时可变结果个数**：结构化拒绝；不在本切片。
6. **性能未测**：所有本机数字 UNGATED；region 规划耗时只是观测。
