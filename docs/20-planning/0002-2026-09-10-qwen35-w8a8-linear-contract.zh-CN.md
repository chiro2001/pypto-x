# Qwen3.5-0.8B 首期 W8A8-linear 契约（冻结稿）

- 任务：`w8a8-linear-contract`（纯规划，无实现、无权重加载、无性能测量）
- worktree：`/home/chiro/projects/pypto/worktrees/pypto-x/w8a8-linear-contract`
- branch：`work/w8a8-linear-contract`
- base：`port/pypto-x-integration @ bcf9516e6419d232338988238ffa8f79e59079c1`
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/w8a8-linear-contract/`
- 本稿状态：**契约已冻结**；第 8 节 D1–D12 已由用户于 2026-09-10 全部批准为报告建议默认值（见下方决策记录）。

## 决策记录（2026-09-10，用户批准）

用户于 2026-09-10 明确批准 **第 8 节 D1–D12 全部按报告建议默认值执行**，其中影响后续实现的关键项：

```text
D2  授权：可以使用已下载的固定 revision BF16 权重计算 W8A8 weight scale（不新增下载，不改变 0033 的权重范围）
D1  in_proj_b / in_proj_a（36 个 16x1024）首期保持 BF16
D3  不引入校准语料：weight 用自身 absmax，activation 动态 per-token
D4  lm_head（tied embedding）首期不量化
D5  L4-L6 阈值先按 5.2 节暂定值实现；拿到 BF16 整网基线后再冻结
D6  per-group 只预留字段，运行时 reject
D7  舍入模式 RNE
D8  取码区间 [-127,127]（放弃 -128）
D9  conv1d 首期保持 BF16
D10 SVE256 首期允许 widening fallback，native SDOT 另附 HWCAP/反汇编证据
D11 5 个 qlinear_w8a8_* 名字先实现为 builder 组合，不新增融合 opcode
D12 CUDA/AMD 能力不足时硬失败，需显式声明 precision="bf16" 才回退
```

实现任务仍排在 BF16 带权整网执行与 logits 对齐之后启动；第 9 节（R1–R14）为后续实现任务的验收依据。

## 0. 启动协议字段（原样）

```text
task_name=w8a8-linear-contract
worktree=/home/chiro/projects/pypto/worktrees/pypto-x/w8a8-linear-contract
branch=work/w8a8-linear-contract
base=port/pypto-x-integration @ bcf9516e6419d232338988238ffa8f79e59079c1
started_at=2026-09-10T08:03:16Z
smoke_once=true
wait_timeout_seconds=3600
poll=false
resource_lock_root=/home/chiro/projects/.resource-locks
local_heavy_policy=locked
local_heavy_runner=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh
local_min_available_mib=8192
local_safety_floor_mib=4096
local_max_cpus=6
```

### 0.1 smoke 结果（唯一一次，未重跑）

```text
script   /home/chiro/projects/pypto/pypto_x/scripts/smoke/pypto_pro_smoke.sh
target   host
exit     0
status   PASS
head     bcf9516e6419d232338988238ffa8f79e59079c1
branch   work/w8a8-linear-contract
notes    pypto/pypto_pro 源码、Git、基础工具链和 Python 语法检查通过；未加载模型
log      /home/chiro/projects/pypto/worktrees/_meta/pypto-x/w8a8-linear-contract/logs/20260910T080316Z/smoke.log
```

本次任务**没有**执行任何 `run_local_heavy.sh` 包裹的命令：全部调研为只读 `grep`/读文件，外加一个纯算术的体积估算（无 lowering、无编译、无 pytest 全量）。

## 1. 量化 scheme 冻结

### 1.1 结论表

| 项目 | 冻结值 | 依据 |
|---|---|---|
| Weight dtype | signed INT8，对称，zero-point 固定 0 | `configs/model_targets.yaml:23-30`；`docs/20-planning/0001-...md:73` |
| Weight scale 粒度 | **per-output-channel**（`[out,in]` PyTorch 布局取 `axis=0`，每行一个 scale） | `configs/model_targets.yaml:29`；`docs/20-planning/0001-...md:74` |
| Weight scale 公式 | `s_w[o] = max(|W[o,:]|) / 127`；`q = clamp(rne(W/s_w), -127, 127)` | 本稿冻结（零权重行见 §1.4） |
| Weight zero-point | 常量 0，**不落盘、不出现在 binding schema** | `model_targets.yaml:28` |
| Activation dtype | signed INT8，对称，zero-point 0，**动态** | `model_targets.yaml:31-38` |
| Activation 粒度 | **per-token / per-row**：对量化前的 2-D 视图 `[rows, K]` 每行一个 scale | `model_targets.yaml:36` |
| Activation scale 公式 | `s_a[r] = max(|A[r,:]|) / 127`，RNE，clamp `[-127,127]` | 本稿冻结 |
| scale dtype | 两者均 **FP32**；BF16 scale 在本期显式拒绝 | `model_targets.yaml:30,37` |
| GEMM 累加器 | **INT32** | `model_targets.yaml:39`；`docs/20-planning/0001-...md:77` |
| Epilogue | `out_bf16 = rne_bf16( fp32(acc_i32) * (s_a[r] * s_w[o]) )`，单次舍入 | `docs/20-planning/0001-...md:78` |
| 舍入模式 | RNE（ties-to-even），与仓库既有 BF16 RNE 策略一致 | `development_lock.yaml:261`（`bf16_fp32_accumulation_rne`） |
| 逻辑 layout | 权重行主序 `[out, in]`，`in` 连续；packing 由 target 决定 | `docs/20-planning/0001-...md:79,174` |

### 1.2 取舍理由（可复核）

1. **为什么 weight per-output-channel 而不是 per-tensor**：INT8 对称 per-tensor 在 MLP `down_proj`、GDR `out_proj` 这类输出通道动态范围差异大的层上误差不可控；per-output-channel 是 `model_targets.yaml:29` 已冻结的粒度，也是所有主流推理栈的默认。
2. **为什么 activation per-token 而不是 per-tensor**：per-tensor 需要静态校准激活范围，会把校准集依赖引入首期；per-token 动态 absmax 不依赖外部数据，只在 prefill 增加一次行内 reduce。decode（T=1）时 per-token 退化为 per-row，无需单独分支。
3. **为什么 zero-point 固定 0**：`model_targets.yaml:26-28` 与 `myplanning:73` 都已冻结对称；对称量化使 `dequantize` 保持原点对称，也让 INT8 乘加在 GPU 的 `s8×s8` 上不必做 u8 偏移补偿。
4. **为什么使用 `[-127,127]` 而不是 `[-128,127]`**：若允许 `-128`，则 `s = absmax/127` 下 `-absmax` 仍映射到 `-127`，`-128` 永远取不到；而取 `s = absmax/128` 会让 `+absmax` 无法表示。冻结 `[-127,127]` + `/127` 使正负两端都可表示且完全对称，代价是 1/256 的码点浪费，但换取“dequant 严格对称”这一可测试性质（见验收 R3）。
5. **为什么 scale 用 FP32 而不是 BF16**：BF16 scale 只有 8 位有效位，会把 `s_a*s_w` 的相对误差抬到 4e-3 量级，直接吃掉 per-channel 量化带来的收益，并且使 epilogue 误差与舍入顺序强耦合。FP32 scale 的额外体积可忽略（见 §1.5：1562 KiB / 949 MiB ≈ 0.16%）。
6. **weight scale 是否 per-group**：**本期不做**。`QuantizedTensorDesc` 预留 `granularity ∈ {per_output_channel, per_group}` 字段与 `group_size`，但 builder 在 scheme 为 per_group 时显式抛错（fail-closed），避免出现“描述符支持但后端没实现”的静默路径。是否启用见 §8 决策 D6。

### 1.3 Qwen3.5-0.8B 各线性层的适配性

权重形状来自 `python/pypto/portable/qwen35.py:487-525` 的 `_decoder_parameter_manifest`（shape 均为 `[out, in]`），模型常量来自 `qwen35.py:45-70` 的 `QWEN35_08B_SHAPE_METADATA`。

| 线性层 | 形状 `[out,in]` | 实例数 | per-channel 适配性 | 备注 |
|---|---|---:|---|---|
| `mlp.gate_proj` | `(3584, 1024)` | 24 | 好 | SwiGLU 前，K=1024 |
| `mlp.up_proj` | `(3584, 1024)` | 24 | 好 | 与 gate 共享输入 |
| `mlp.down_proj` | `(1024, 3584)` | 24 | 好 | **K=3584 为全模型最大 K** |
| `full_attention.q_proj` | `(4096, 1024)` | 6 | 好 | 含输出门（`2*qh*d`） |
| `full_attention.k_proj` | `(512, 1024)` | 6 | 中 | out=512，per-channel scale 数量少，收益有限但仍有效 |
| `full_attention.v_proj` | `(512, 1024)` | 6 | 中 | 同上 |
| `full_attention.o_proj` | `(1024, 2048)` | 6 | 好 | K=`qh*d`=2048 |
| `linear_attention.in_proj_qkv` | `(6144, 1024)` | 18 | 好 | 混合语义（Q/K/V 拼接）：**per-channel 对混合轴仍然合法**，但误差诊断必须按 Q/K/V 段分别报告 |
| `linear_attention.in_proj_z` | `(2048, 1024)` | 18 | 好 | gate 分支，数值敏感度高于 qkv |
| `linear_attention.in_proj_b` | `(16, 1024)` | 18 | **差** | 每层仅 16 个输出通道；β 进入 GDR 递推，误差会被状态放大。建议首期**保持 BF16**（见 §2 与 §8-D1） |
| `linear_attention.in_proj_a` | `(16, 1024)` | 18 | **差** | 同上，且 `A_log/dt_bias` 与它成对进入 `exp(g)` 衰减 |
| `linear_attention.out_proj` | `(1024, 2048)` | 18 | 好 | GDR 输出投影 |
| `lm_head`（与 embedding 共享） | `(248320, 1024)` | 1 | 好 | **tied weight**，见 §3.4；默认首期不量化（§8-D4） |

小通道层（`in_proj_b`/`in_proj_a`，共 36 个张量）的保守处理带来 0.56 MiB × 2 的 BF16 额外占用，相对 949 MiB 线性权重可忽略，但显著降低 GDR 递推的发散风险。

### 1.4 边界与退化规则（必须逐条实现并可测）

| 编号 | 规则 | 期望行为 |
|---|---|---|
| Q1 | 整行全零（`absmax == 0`） | `s = 1.0`，`q = 0`；**不得**产生 NaN/Inf |
| Q2 | 行内含 NaN/Inf | `quantize_per_token_s8` **显式报错**（fail-closed），不得静默 clamp |
| Q3 | clamp 饱和 | `q = -127` / `+127`，饱和计数（saturation count）必须作为 artifact/日志字段上报 |
| Q4 | RNE 舍入 | `rne(0.5) = 0`、`rne(1.5) = 2`、`rne(-0.5) = -0`，与 BF16 RNE 同一实现 |
| Q5 | K 上限 | `K ≤ floor((2^31-1)/(127*127)) = 133,144`；超过必须显式拒绝（切 INT64 累加不在本期） |
| Q6 | 非 2-D 输入 | builder 先 `reshape` 到 `[rows, K]`，量化只在 2-D 视图上定义 |

### 1.5 体积与溢出核算（本次实测计算，非估算）

对 186 个量化张量（下表同 §2）：

```text
量化张量数            186
元素总数              497,614,848
BF16 存储             949.1 MiB
INT8 存储             474.6 MiB
FP32 scale            1,562.2 KiB（0.16%）
等效压缩比            1.994x（仅线性权重，不含 embedding/norm/conv/state）

INT32 累加安全余量（K=3584，最坏 |x|=|w|=127）
  3584*127*127 = 57,806,336 < 2^31-1 = 2,147,483,647（占 2.69%）
无溢出 K 上限        133,144
```

按层类分解：

| 层类 | 形状 | 数量 | 元素 | BF16 MiB | INT8 MiB | scale KiB |
|---|---|---:|---:|---:|---:|---:|
| mlp.gate_proj | (3584,1024) | 24 | 88,080,384 | 168.00 | 84.00 | 336.0 |
| mlp.up_proj | (3584,1024) | 24 | 88,080,384 | 168.00 | 84.00 | 336.0 |
| mlp.down_proj | (1024,3584) | 24 | 88,080,384 | 168.00 | 84.00 | 96.0 |
| full_attn.q_proj | (4096,1024) | 6 | 25,165,824 | 48.00 | 24.00 | 96.0 |
| full_attn.k_proj | (512,1024) | 6 | 3,145,728 | 6.00 | 3.00 | 12.0 |
| full_attn.v_proj | (512,1024) | 6 | 3,145,728 | 6.00 | 3.00 | 12.0 |
| full_attn.o_proj | (1024,2048) | 6 | 12,582,912 | 24.00 | 12.00 | 24.0 |
| lin_attn.in_proj_qkv | (6144,1024) | 18 | 113,246,208 | 216.00 | 108.00 | 432.0 |
| lin_attn.in_proj_z | (2048,1024) | 18 | 37,748,736 | 72.00 | 36.00 | 144.0 |
| lin_attn.in_proj_b | (16,1024) | 18 | 294,912 | 0.56 | 0.28 | 1.1 |
| lin_attn.in_proj_a | (16,1024) | 18 | 294,912 | 0.56 | 0.28 | 1.1 |
| lin_attn.out_proj | (1024,2048) | 18 | 37,748,736 | 72.00 | 36.00 | 72.0 |

## 2. 层覆盖策略（混合精度边界显式清单）

### 2.1 W8A8 覆盖（186 个张量）

| 层类 | 张量数 | 语义 |
|---|---:|---|
| full-attention q/k/v/o proj | 24 | 6 层 × 4 |
| GDR `in_proj_qkv` | 18 | Q/K/V 混合投影，按段诊断 |
| GDR `in_proj_z` | 18 | 输出门 |
| GDR `out_proj` | 18 | 输出投影 |
| GDR `in_proj_b` / `in_proj_a` | 36 | **条件量化**：默认保持 BF16（§1.3），仅当 §5 单层阶梯通过时才启用 |
| FFN gate/up/down | 72 | 24 层 × 3 |
| `lm_head` | 1（tied） | **默认不量化**，opt-in（§3.4 / §8-D4） |

默认（不含 `in_proj_b/a`、不含 lm_head）量化张量数 = 24+18+18+18+72 = **150**；条件全开时 **187**（含 lm_head）。

### 2.2 必须保持 BF16/FP32 的「浮点孤岛」

依据 `configs/model_targets.yaml:41-51` 与 `docs/20-planning/0001-...md:89-98`，逐条落到现有符号：

| 孤岛 | 精度 | 现有实现锚点 |
|---|---|---|
| embedding lookup | BF16 | `qwen35.py:1593-1598`（`embedding` op） |
| RMSNorm / gated RMSNorm / L2Norm | 权重 BF16，reduce 与 eps FP32 | `qwen35.py:541-675`（`_rms_core` / `_rms_gated_core`）、`qwen35.py:289-333`（`_l2_core`） |
| SiLU / sigmoid / softplus / exp / rsqrt | FP32 计算 | `qwen35.py:1766-1775`；`lowering/cpu/__init__.py:983-1005` |
| Softmax | FP32 | `qwen35.py:889-953`（`build_stable_softmax`） |
| RoPE / MRoPE | FP32 三角，BF16 存储 | `qwen35.py:676-754`、`qwen35.py:954-995` |
| attention QKᵀ / PV 激活间 matmul | FP32 累加 | `qwen35.py:2092-2335`（`_append_full_attention_graph`） |
| KV cache | BF16，函数式 SSA | `development_lock.yaml:781`（`kv_state: functional_pure_ssa_bf16`） |
| GDR recurrent state | state/gate/scale FP32，q/k/v/beta BF16 | `development_lock.yaml:832-836`；`qwen35.py:3446-3605` |
| conv state / depthwise Conv1D | BF16 存储，FP32 累加 | `development_lock.yaml:727-732`；`qwen35.py:2609-2720` |
| residual stream | BF16 存储，两处 add 有显式 def-use | `qwen35.py:1748-1752, 1787-1791`；`development_lock.yaml:989` |
| position/iota/compare/causal mask | 整数/布尔 | `qwen35.py:1568-1591` |
| 所有 epsilon / scale 常量 | FP32 | `qwen35.py:1605-1609` |

### 2.3 混合精度边界（dtype 切换点，必须逐点可测）

```text
input_ids int64
  → [embedding]            int64  → bf16
  → hidden bf16
  → [rmsnorm]              bf16 in, fp32 reduce → bf16 out
  → [quantize_per_token_s8] bf16 → int8 + fp32 scale        ← 边界 B1（A 量化）
  → [qmatmul_s8s8_s32]     int8 × int8 → int32              ← 边界 B2（累加器）
  → [dequantize_epilogue_bf16] int32 + fp32 scale → bf16    ← 边界 B3（唯一一次舍入）
  → residual add bf16
```

- B1 之前**不得**提前降精度；B3 之后**不得**再出现第二次 bf16 舍入。验收要求：同一批输入下，`dequantize_epilogue_bf16` 的输出与「FP32 域完成乘法后单次 cast」逐位一致（R4）。
- GDR 分支内 `g/state/scale` 全程 FP32，与 `q/k/v/beta` 的 BF16 存储边界保持不变（`development_lock.yaml:832-836`）。

### 2.4 覆盖率报告要求（强制）

`docs/20-planning/0001-...md:100` 已要求同时报告两个数，本契约把它固化为必填字段：

1. `w8a8_linear_coverage` = 已量化线性层的参数量 / 全部线性层参数量（默认 150/186 张量，或按参数量加权）；
2. `whole_net_int8_compute_ratio` = INT8 MAC 数 / 总 MAC 数（分母包含 attention QK/PV、conv1d、GDR 递推的浮点 matmul）。

**禁止**用 `W8A8` 字样暗示 Norm/Softmax/GDR state/KV 已量化。

## 3. 运行时与 artifact 表示

### 3.1 存储布局（storage-relative packed layout）

现有 M1J 契约：`parameter_storage: caller_owned_noncopying_memoryview`、`packed_alignment: storage_relative_offset_64_bytes`、`physical_pointer_alignment: not_guaranteed_backend_must_check`（`development_lock.yaml:1038-1042`），实现为 `python/pypto/portable/runtime_binding.py:988-1067`（`_align_up` / `build_packed_layout`），默认对齐 `DEFAULT_PACK_ALIGNMENT = 64`（`runtime_binding.py:31`），区域类型 `PackedRegion`（`runtime_binding.py:929-951`）。

**冻结扩展**：每个量化权重张量占 **2 个 region**：

| region `storage_id` | dtype | shape | nbytes | alignment | role |
|---|---|---|---:|---:|---|
| `<param>.int8` | `int8` | `[out, in]` | `out*in` | 64 | `quant_weight` |
| `<param>.scale` | `float32` | `[out]` | `4*out` | 64 | `quant_scale` |

- zero-point **不占 region**（策略常量 0）。
- 权重行主序、`in` 连续（使 `s8` 可以用 64 字节对齐的行起点直接喂 dot 指令）。
- region 数从 368（320 参数 + 48 state，`development_lock.yaml:1031`）变为：默认 **368 − 150 + 150*2 = 518**；条件全开（含 36 个小通道层与 lm_head）为 **368 + 187 = 555**。
- 新增版本号：`PACKED_LAYOUT_VERSION` 1 → **2**（`runtime_binding.py:30`）。旧 v1 布局必须 `reject_and_recompile`（沿用 M1E/M1F/M1H 的 fail-closed 先例，`development_lock.yaml:786,888`）。

### 3.2 binding schema 扩展

现状（必须如实说明）：

- `BindingEntry` 字段为 `ordinal/name/role/dtype/shape/nbytes/storage_id/readonly`（`runtime_binding.py:347-370`），`dtype` 由 SSA `ValueRef` 类型经 `_entry_from_ref` 推导（`runtime_binding.py:467-475`）。
- 现有 role 只有 `runtime` / `parameter` / `state` / `output` / `state_output`（`runtime_binding.py:388-404, 825-829`）。
- `_DTYPE_BYTES` 已含 `int8`（`runtime_binding.py:35`），所以 **int8 buffer 的 nbytes 校验今天就能通过**。
- **硬阻塞**：`runtime_binding.py:522` 明确 `raise BindingMismatchError("Qwen model parameters must use one exact dtype")`。只要权重量化成 int8，现有 schema 构建会直接失败。

**冻结扩展方式**（最小侵入、可版本化）：

1. `BINDING_SCHEMA_VERSION` 1 → **2**（`runtime_binding.py:29`）。
2. `BindingEntry` 增加一个**可选冻结映射** `quant: Mapping[str, Any]`（默认 `None`，BF16 路径逐位不变），键固定为：
   `scheme`（`"w8a8_linear"`）、`signed`（true）、`bits`（8）、`axis`（0）、`granularity`（`per_output_channel`）、`scale_dtype`（`"float32"`）、`zero_point`（0）、`scale_entry`（对应 scale entry 的 `name`）、`group_size`（None）。
3. 新增 role **`quant_scale`**（作为 `inputs` 中的参数条目），`dtype="float32"`，`shape=[out]`。
4. `runtime_binding.py:522` 的单 dtype 约束改为：**“所有非量化参数必须同为一个精确 dtype（bf16）；量化权重必须为 int8；其 scale 必须为 float32”**，并且量化权重的 `quant.scale_entry` 必须精确指向同 schema 内存在的 `quant_scale` 条目（缺失/悬空即 fail-closed）。
5. `aliases` 继续承载官方 HF 名（`qwen35.py:492` 的 `embedding.weight` / `lm_head.weight` 先例），量化条目新增别名 `<hf_name>.qweight` / `<hf_name>.scales`（仅作为别名，不改变 storage 语义）。
6. 反序列化：`from_dict` 遇到未知 `quant` 键必须拒绝（沿用 `model.py` 的 `_ensure_keys` 风格，`core_ir/model.py:49-53`）。

**是否新增 dtype/scale 参数**：不新增顶层 dtype 参数，而是通过 ①`BindingEntry.quant` ② 独立的 `quant_scale` 条目 表达；`BufferBinding` 仍只做 `(buffer, dtype, shape)` 三元绑定（`runtime_binding.py:168-345`），因此 mmap/bytes/bytearray 的 zero-copy 路径（`development_lock.yaml:1032`）无需改动。

### 3.3 Core IR 新原语与所属 target 层

现状证据（**“没有”必须给出 grep 证据**，详见附录 A）：

- `python/pypto/core_ir/`：`model.py` 只有 `Effect/CoreType/ValueRef/Operation/Block/Region/CoreFunction/CoreProgram`（`core_ir/model.py:132,150,218,254,375,425,497,590`），无任何量化类型或算子。
- `python/pypto/lowering/`、`python/pypto/portable/`、`python/pypto/backends/` 中 `quantize|dequantize` 命中数 **0**（附录 A/E1）。
- 唯一的量化实现是 Ascend 经典前端：`python/pypto/op/quantization.py:22`（`quantize`）、`:73`（`dequantize`），直接调 `pypto_impl.Quantize/Dequantize`（`:69`、`:120`），是 CANN native、非 portable，且 `pypto/frontend/core_export.py` 中没有任何量化算子映射（附录 A/E5）。
- 现有 `matmul` op **无法**表达 `s8×s8→s32`：`lowering/cpu/__init__.py:1526-1527` 强制 `inputs[0].dtype == inputs[1].dtype == output.dtype`。

**冻结的 opcode 集合**（公共层，`pypto/core_ir` + `pypto/portable` + 各 target plan）：

| opcode | 语义 | 输入 → 输出 | 归属层 |
|---|---|---|---|
| `quantize_per_token_s8` | 动态 per-row absmax 量化，**一个 op 产出 q 与 scale** | `tensor[bf16/fp32]` → `int8[rows,K]` + `fp32[rows]` | 公共语义（core_ir alias 表 + scalar golden + vector plan + gpu plan） |
| `dequantize_s8` | `int8 + fp32 scale → fp32`（诊断/单测用） | `int8[..,K]`, `fp32[..]` → `fp32` | 公共语义 |
| `qmatmul_s8s8_s32` | INT8×INT8→INT32，rank-2/3/4 与现有 `matmul` 同 batch 规则 | `int8`, `int8` → `int32` | 公共语义 |
| `dequantize_epilogue_bf16` | `int32 + (s_a ⊗ s_w) → bf16` 单次 RNE | `int32[m,n]`, `fp32[m]`, `fp32[n]` → `bf16[m,n]` | 公共语义 |

**组合式（不下沉为 opcode）**：`qlinear_w8a8`、`qlinear_w8a8_qkv_split`、`qlinear_w8a8_swiglu`、`qlinear_w8a8_gate`、`qlinear_w8a8_lm_head`、`pack_weight_s8` 本期实现为 `python/pypto/portable/qwen35.py` 中的 **builder 函数**，内部只发射上述 4 个 opcode + 既有 `reshape/transpose/cast/split/silu/mul`。

理由（可验收）：`docs/20-planning/0001-...md:157-172` 列出的名字是目标形态；但若一上来就把 5 个 fused 名字注册成 opcode，任何后端都会被迫实现“名字存在但内部不融合”的假融合，违背 `lowering/cpu/x86/avx2/lowering.py:27-30` 注释确立的“不得静默添加算子、必须有 native symbol 或显式 fallback”的既有纪律。`pack_weight_s8` 属于**权重的离线 ingestion/artifact 阶段**操作（发生在 host 侧、不进入 Core 图），因此不应是 Core opcode。

`QuantizedTensorDesc` 放 `pypto/core_ir/model.py`（跟随 `CoreType`，`model.py:150` 附近），字段：

```text
signed: bool            # True
bits: int               # 8
axis: int               # 0（weight per-output-channel）；-1（activation per-row，仅诊断）
granularity: str        # per_output_channel | per_group（per_group 本期 reject）
group_size: Optional[int]
scale_dtype: str        # float32
zero_point_policy: str  # fixed_zero
logical_layout: str     # {"weight": "out_in_rowmajor", "activation": "row_then_k"}
```

公共 IR **不**保存 AVX/VNNI、SVE SDOT/I8MM、CUDA dp4a/imma 或 AMD 物理 packing（`docs/20-planning/0001-...md:174`）。

### 3.4 tied embedding / lm_head 的特殊处理

现状：`embedding_lm_head_shared_weight` 是**单一 storage region**（`development_lock.yaml:1042`），embedding 走 `embedding` op（`qwen35.py:1593-1598`），lm_head 走手工 `matmul`（`qwen35.py:1798-1804`），签名校验要求单个精确 dtype（`runtime_binding.py:522`）。

**冻结方案**（默认路径）：

- embedding lookup **永远**读 BF16 region（不量化），保持 `qwen35.py:1593-1598` 不变。
- `lm_head` 默认**保持 BF16**（复用同一 region）。
- 若要开启 lm_head 量化（§8-D4），必须新增 2 个 region（`embedding_lm_head_shared_weight.lm_head.int8`、`.lm_head.scale`），**不改变** embedding 的 BF16 region 与 `tied_embedding_lm_head` 的“同一份数值、两种物理 packing”语义；该模式下 `vocab=248320, K=1024`，int8 增量 237 MiB，per-channel scale 970 KiB。

### 3.5 图构建入口扩展

**硬约束**：`build_qwen35_text_decoder_graph` 当前 `raise ValueError("Qwen3.5 full text decoder graph currently requires dtype='bf16'")`（`qwen35.py:1520-1521`）。

**冻结**：新增关键字参数 `precision: str = "bf16"`（取值 `bf16` / `w8a8-linear`），而不是复用 `dtype`：

- `dtype="bf16"` 继续表示**激活与浮点孤岛**的 dtype，保持不变；
- `precision="w8a8-linear"` 时，186+ 个线性权重参数的类型改为 `int8`，并为每个权重追加 `fp32` scale 参数；
- manifest 新增字段：`precision`、`weight_dtype`、`quant_scheme`（含公式与舍入）、`quantized_parameter_names`、`bf16_island_names`、`w8a8_linear_coverage`、`whole_net_int8_compute_ratio`；
- `qwen35_text_decoder_manifest`（`qwen35.py:1289`）与 `_decoder_parameter_manifest`（`qwen35.py:473`）必须同步产出量化条目，否则 `runtime_binding.py:498-644` 的三方校验会直接失配——这正是我们想要的 fail-closed 行为。

## 4. 后端落地顺序与前置条件

顺序遵循 `configs/model_targets.yaml:65-70` 与 `HANDOFF.zh-CN.md:28`（AVX2 → AVX-512 → SVE256 → NVIDIA → AMD）。

| 阶段 | 后端 | 前置条件（必须实测取证） | 现状证据 |
|---|---|---|---|
| M3-a | **scalar reference** | 无硬件条件；必须先冻结 RNE/clamp/int32 累加/fp32 epilogue 的整数 golden | `lowering/cpu/__init__.py:34-52` 已支持 int8 dtype；`:1509-1547` matmul 校验需为新 opcode 另开分支 |
| M4-a | **AVX2** | 无 VNNI；用 `s8→s16` 拓宽 + `pmaddwd`/`madd` 组合 + int32 累加。前置 = `avx2+fma+osxsave+ymm` 探测通过 | `lowering/cpu/x86/avx2/capabilities.py:61,87-88,121-122,136`（特征集只有 avx/avx2/fma/osxsave/ymm，**无 VNNI**） |
| M4-b | **AVX-512** | 优先 VNNI；**注意 `vpdpbusd` 是 u8×s8**，s8×s8 需左侧 `xor 0x80` 偏移并补偿，或改用 `vpdpwssd`（i16 拓宽）。前置 = XCR0/opmask + BW/VL + VNNI 位 | `lowering/cpu/x86/avx512/capabilities.py:84,92,109-110,158`；`compiler/targets/cpu_avx512.py:441`（`ptx_avx512_vnni_dot_i8` C helper）、`:682,766-768`（`-mavx512vnni`）、**`:900` 明确 `"public_core_vnni": False`**；`backends/cpu/x86/avx512.py:1190-1217` 仅暴露测试入口 `run_vnni_dot_i8` |
| M4-c | **SVE256** | **SVE2=0（鲲鹏 920B 实测）**，且后端没有 SVE2 emitting lowering → 本期只允许 widening fallback（`svld1sb`/`sunpklo` 到 s16 + `svmadd`/`svmla` 到 s32）。任何 SDOT/I8MM 使用必须另附 HWCAP2 实测 + 反汇编证据 | `HANDOFF.zh-CN.md:317`（SVE=1、SVE2=0、VL=32）；`development_lock.yaml:311`（`ecs_reports_no_sve2_and_backend_has_no_sve2_emitting_lowering`）；`lowering/cpu/aarch64/sve/capabilities.py:24,79-80,94,112-113,135-136`（SVE2 独立探测、不并入 SVE256 必需特征） |
| M4-d | **NVIDIA CUDA** | 现状：`GPU_SUPPORTED_DTYPES = ("float32","bf16")`（`lowering/gpu/plan.py:43`），PTX 层只支持 int8 的 **存储 itemsize**（`compiler/targets/cuda.py:109-119`），**没有** dp4a/imma/mma 路径（附录 A/E6 grep 无命中）。前置 = 运行期 capability 记录（`TargetSpec.device`）；首期先用 `mul.wide.s16`/`mad.lo.s32` 正确性 kernel，dp4a(sm_61+)/imma(sm_75+/sm_80) 必须由 capability 探测选择，禁止假定 | `development_lock.yaml:313-316`（sm80 PTX + cc120 驱动 JIT、无 nvcc、正确性证据非性能证据） |
| M4-e | **AMD gfx1036** | 前置 = 可执行 HIP/HSA 运行态；当前 `runtime: BLOCKED_DEVICE`，且 target 的 `int8/mfma/wmma` 全部为 `"unknown"` → **必须 fail-closed 拒绝**，不得静默走 bf16 | `target/amd.py:30-38`；`development_lock.yaml:405,417,446,460`；`compiler/targets/hip.py:73-105`（30 个静态算子，无量化）、`:106`（`_SUPPORTED_DTYPES = ("float32","bf16")`） |

**跨后端一致性要求**：所有后端对 `qmatmul_s8s8_s32` 的 **int32 输出必须与 scalar golden 逐位一致**（整数运算无误差余地）；epilogue 的 fp32 结果允许 ≤1 ulp 差异，但转 bf16 必须同为 RNE。

**cache key 要求**：artifact cache key 必须纳入 `quant_scheme_digest` + 后端 int8 capability mask，避免“同一张图在无 VNNI 机器上复用有 VNNI 的 artifact”（现有版本拒绝先例：`development_lock.yaml:786,888,920,952`）。

## 5. 校准与精度验收

### 5.1 校准

| 项目 | 冻结值 |
|---|---|
| weight scale 数据来源 | **权重自身的 absmax**，不需要外部校准语料 |
| 权重读取边界 | 权重已授权下载到 `../worktrees/_meta/pypto-x/assets/qwen35-0.8b/2fc06364715b967f1860aea9cf38778875588b17/`（`docs/00-handoffs/0033-...md`），但**该授权范围是“纯文本 BF16 带权执行；不含 W8A8 实现”** → 用权重算 W8A8 scale 需 §8-D2 确认 |
| activation scale | 运行期动态计算，无校准集 |
| 可选诊断（非门禁） | 固定 8 条 prompt 上统计每层激活 absmax 分布，用于诊断 per-token 动态范围；不参与 scale 计算 |
| scale 生成物 | 每张量 `(name, max_abs, scale, saturate_count)` 的 JSON manifest，进 artifact 元数据，可复现 |

### 5.2 精度门禁（分级，先单层再整网）

| 级别 | 比较对象 | 门槛 | 说明 |
|---|---|---|---|
| L0 整数 golden | `qmatmul_s8s8_s32` vs scalar 整数参考 | **int32 逐位一致，0 mismatch** | 无误差余地；R5 |
| L1 单算子 | `quantize/dequantize` vs 手算 | clamp/RNE/全零行/NaN 用例全通过 | R3、R6 |
| L2 单层（真实权重 + 捕获激活） | 同层 BF16 输出 | cosine ≥ **0.9995**，相对 L2 ≤ **1e-2**，max|Δ| ≤ **2% × max|ref|** | 每层单独报告，`in_proj_b/a` 单独更严 |
| L3 逐层替换 | 24 层逐层开启 W8A8 的 hidden state 误差 | 每层相对误差 ≤ L2 门槛，且误差不随深度单调爆炸 | 阶梯：1 层 → 6 层（含全部 full attention）→ 24 层 |
| L4 整网 logits | 同 backend 的 BF16 基线（`validated.compare_against: transformers-bf16-reference`，`model_targets.yaml:83`） | max|Δlogits| ≤ **0.5**（暂定），mean|Δ| ≤ **0.05**（暂定），cosine ≥ 0.999 | 阈值见 §8-D5 |
| L5 生成行为 | greedy decode | top-1 一致率 ≥ **99%**（128 token）、top-5 一致率 ≥ **99.9%**、首次分叉 ≥ **32** token | 报告字段见 `model_targets.yaml:85-92` |
| L6 perplexity | 固定小语料 | Δppl ≤ **+1.0%** 或 +0.2 绝对（暂定） | 需要用户确认语料与阈值 |
| L7 GDR 专项 | recurrent state / conv state 误差 | state max|Δ| ≤ L2 门槛；不得出现 NaN/Inf 或状态发散 | `model_targets.yaml:88` 已要求 `gdr-state-error` |

**诚实声明**：L4–L6 的数字是**暂定值**，`docs/20-planning/0001-...md:270` 明确模型级精度阈值仍属“待对齐项”；BF16 参考基线由并行的 `qwen35-bf16-reference` 任务产出。本契约冻结的是**测量流程、比较对象与报告字段**，数字待 §8-D5 批准。

### 5.3 报告字段（沿用 `model_targets.yaml:85-97`，新增量化专属）

`per-op-error`、`per-layer-hidden-state-error`、`gdr-state-error`、`logits-error-and-cosine`、`top1-top5-agreement`、`greedy-first-divergence`、`perplexity-delta`、`peak-memory`、`compile-and-cache-time`、`prefill-latency`、`decode-latency`、`pypto-x-coverage`
**新增**：`w8a8_linear_coverage`、`whole_net_int8_compute_ratio`、`weight_scale_manifest_digest`、`activation_saturation_rate`、`int32_accumulation_headroom`。

### 5.4 性能

不承诺任何性能数字（`model_targets.yaml:84`；`docs/20-planning/0001-...md:257`）。W8A8 的收益/代价一律以实测记录，不得在契约里预设倍率。

## 6. 失败与回退

| 场景 | 门禁行为 |
|---|---|
| 后端缺少 int8 能力（无 VNNI、无 SVE2、CUDA cc 不足、AMD runtime blocked） | **显式拒绝**：`UnsupportedCapabilityError` / `ExplicitRejection`，错误信息必须含缺失能力名；**禁止**在 `precision="w8a8-linear"` 的图里静默执行 BF16 |
| 精度门禁不通过（L2–L6） | 该层/该 op 回退为 BF16 孤岛，并在 `bf16_island_names` 与报告中显式登记；**禁止**调整阈值以通过 |
| 量化失败（NaN/Inf、越界、scale=0 且非全零行） | artifact 不生成（fail-closed）；不允许“跳过该层继续” |
| 版本/摘要不匹配（schema v1、packed v1、旧 lowering） | `reject_and_recompile`（沿用 `development_lock.yaml:786,888,920,952`） |
| 权重与 scale 形状/字节不符 | `BindingMismatchError`，在任何 kernel 启动前失败 |
| 性能不达预期 | 不是门禁项；只记录，不改精度 |

**明确禁止的静默降精度**（全部要在报告里逐条声明“未发生”）：

1. W8A8 图内部回退 BF16 计算而不标注；
2. per-output-channel 静默退化为 per-tensor；
3. int32 累加静默换成 fp32 累加（或反之）；
4. INT8 饱和后静默 clamp 而不上报 `saturate_count`；
5. 用 `math.exp` 等近似指令替代量化路径中的精确 epilogue；
6. lm_head/embedding 仍是 BF16 却把整体标称为 W8A8（必须报告覆盖率）。

## 7. 非目标（本阶段明确不做）

1. 不实现任何量化 kernel、不做 lowering、不改运行时（本任务只是契约）。
2. 不下载/加载权重（本任务全程零权重）。
3. 不做 W8A16、W4A16/INT4、FP8、W4A4（`model_targets.yaml:53-63`）。
4. 不做 KV cache 量化、不做 GDR recurrent state / conv state 量化（同上）。
5. 不做 MoE、不做分布式、不接视觉编码器、不做 MTP/speculative decoding（`docs/20-planning/0001-...md:48-57`）。
6. 不做 per-group weight 量化（描述符预留，本期 reject）。
7. 不做非对称（zero-point ≠ 0）量化。
8. 不做性能承诺与性能门槛（`model_targets.yaml:84`）。
9. 不把 Ascend 经典前端的 `pypto.quantize/dequantize`（`op/quantization.py:22,73`）当作 portable 语义复用——它绑定 `pypto_impl` native，违反公共层边界。
10. 不修改 integration worktree、`upstream/*` master、其他 agent 的 worktree。

## 8. 待用户决策清单

| 编号 | 问题 | 建议默认 | 影响 |
|---|---|---|---|
| D1 | `in_proj_b` / `in_proj_a`（36 个 16×1024 小通道张量）首期是否量化？ | **保持 BF16**（§1.3），只在 L2 阶梯通过后另立任务开启 | 关乎 GDR 递推数值稳定性；体积代价 1.1 MiB |
| D2 | 是否授权**用已下载的 BF16 权重计算 W8A8 的 weight scale**？ | 需要显式确认：0033 授权范围写的是“纯文本 BF16 带权执行；不含 W8A8 实现” | 不确认则 W8A8 无法产生任何真实 scale，只能做合成权重单测 |
| D3 | 是否允许引入校准语料？ | **不引入**：weight scale 用权重 absmax、activation 动态 per-token，首期零语料依赖 | 若要求引入，需要指定语料、许可证、存放位置 |
| D4 | `lm_head`（tied embedding）是否首期量化？ | **默认不量化**；如需量化，走 §3.4 的双 packing 方案（+237 MiB int8） | 影响 logits 精度与 tied storage 契约 |
| D5 | L4–L6 的具体数值阈值（max|Δlogits|、top-1 一致率、首次分叉、Δppl）由谁批准？ | 先按 §5.2 暂定值实现并报告，拿到 BF16 基线后冻结 | 阈值是 W8A8 MVP 的完成定义之一（`docs/20-planning/0001-...md:263`） |
| D6 | 是否现在就在描述符里启用 per-group（group_size 64/128）？ | **本期只预留字段，运行时 reject** | 影响 artifact 布局与后端 kernel 复杂度 |
| D7 | 舍入模式确认为 RNE？ | 是（与项目既有 BF16 RNE 一致） | 若改用 half-away-from-zero，需新版本号并重跑全部 golden |
| D8 | 是否认可 `[-127,127]`（放弃 `-128`）的对称取码？ | 是 | 影响 clamp 边界与 golden 期望值 |
| D9 | conv1d 首期保持 BF16？ | 是（`docs/20-planning/0001-...md:87`） | 若它成为瓶颈，也只能新增 W8A8 实现，不新增精度种类 |
| D10 | SVE256 首期是否允许只用 widening fallback（不要求 native SDOT）？ | 允许；native SDOT 需另附 HWCAP/反汇编证据 | 影响 920B 上的 W8A8 证据强度 |
| D11 | 是否认可把 5 个 `qlinear_w8a8_*` 名字先实现为 builder 组合而非 opcode？ | 认可（§3.3 理由） | 影响 IR schema 与后端收敛节奏 |
| D12 | CUDA/AMD 在能力不足时是硬失败还是允许“显式请求 BF16 才运行”？ | 硬失败 + 显式 `precision="bf16"` 二次请求 | 影响门禁脚本与验收脚本写法 |

## 9. 可验收清单（后续实现任务的验收依据）

| 编号 | 验收项 | 证据形式 |
|---|---|---|
| R1 | `QuantizedTensorDesc` 存在于 `pypto/core_ir`，字段与 §3.3 一致，`per_group` 被 reject | 单测 + `to_dict/from_dict` 往返 |
| R2 | 4 个新 opcode 在 scalar reference 中实现，并按 §1.4 的 Q1–Q6 全通过 | 单测 + 边界用例（全零行、NaN、饱和、K 上限） |
| R3 | `[-127,127]` 对称性：`q=-127` 与 `q=127` 均可由 `±absmax` 产生 | golden 向量 |
| R4 | epilogue 单次舍入：与“FP32 乘完再 cast”逐位一致 | 差分单测 |
| R5 | `qmatmul_s8s8_s32` 与整数 golden 逐位一致（含 rank-3/4 与 batch prefix） | 差分单测 |
| R6 | 每个后端在自己的能力探测通过后复现 R5 | 每后端一组证据（日志 + 反汇编/PTX/LLVM） |
| R7 | binding schema v2：量化权重 + scale 条目齐备，悬空 `scale_entry` 被拒 | 单测（构造篡改 schema） |
| R8 | packed layout v2：region 计数与 §3.1 一致（默认 518），v1 被拒 | metadata-only driver |
| R9 | 覆盖率报告字段存在且非零：`w8a8_linear_coverage`、`whole_net_int8_compute_ratio` | manifest 断言 |
| R10 | L2 单层阶梯：逐层 cosine/相对 L2/max|Δ| 达标 | 报告表 |
| R11 | L3 逐层替换阶梯（1 → 6 → 24 层）无爆炸 | 报告表 + hidden state 误差 |
| R12 | L4–L5 整网：logits 误差、top-1/top-5、首次分叉 | 与 BF16 基线对比报告 |
| R13 | 无静默降精度：§6 的 6 条禁令逐条有负向测试（构造缺失能力/超阈值场景，期望 fail-closed） | 负向单测 |
| R14 | 未改动 BF16 路径：BF16 schema v1/packed v1 语义与既有 506–522 项测试保持一致 | 全量回归不减少 |

## 附录 A：现状调研证据（grep 与文件/行号）

### A.1 量化符号在移植层的缺席（“现状没有”的证据）

```text
命令：grep -rncE 'quantize|dequantize' python/pypto/{lowering,core_ir,portable,backends}
结果：无任何文件命中（命中数为 0）
命令：grep -rniE 'w8a8|qmatmul|qlinear' python/tests
结果：无命中
命令：grep -rn 'def quantize|def dequantize|Quantize(' python/pypto --include=*.py（排除 tests）
结果：仅 python/pypto/op/quantization.py:22 / :73 / :69（CANN native 前端）
命令：grep -rn 'dp4a|imma|mma\.sync|wmma|mfma' python/pypto/compiler/targets/*.py
结果：仅 -mfma 编译器开关（cpu_avx2.py:643,697；cpu_avx512.py:752,753），无 int8 张量指令
```

### A.2 关键文件与行号

| 主题 | 位置 |
|---|---|
| W8A8 scheme 权威定义 | `configs/model_targets.yaml:18-51` |
| 非目标 | `configs/model_targets.yaml:53-63` |
| 后端顺序 | `configs/model_targets.yaml:65-70` |
| 报告字段 | `configs/model_targets.yaml:82-97` |
| W8A8 精度契约 | `docs/20-planning/0001-...md:67-100` |
| W8A8 新增算子清单 | `docs/20-planning/0001-...md:157-174` |
| 后端 W8A8 路线 | `docs/20-planning/0001-...md:176-185` |
| M3/M4 阶段定义 | `docs/20-planning/0001-...md:227-235` |
| 待对齐项（阈值） | `docs/20-planning/0001-...md:265-270` |
| 技术决策 9/10/11 | `HANDOFF.zh-CN.md:31-33` |
| 权重授权范围 | `docs/00-handoffs/0033-2026-09-10-bf16-weight-authorization-and-wave1-dispatch.zh-CN.md` |
| M1J binding 契约 | `configs/development_lock.yaml:1038-1047` |
| M1K typed byte view 契约 | `configs/development_lock.yaml:1076-1084` |
| W6 当前任务与排队 | `configs/development_lock.yaml:1114-1121` |
| AMD runtime 阻塞 | `configs/development_lock.yaml:405,417,446,460` |
| SVE2 缺失 | `HANDOFF.zh-CN.md:317`；`development_lock.yaml:311` |
| Core IR 类型 | `python/pypto/core_ir/model.py:132,150,218,254,590` |
| 线性层构造器 | `python/pypto/portable/qwen35.py:335-371` |
| 参数 manifest | `python/pypto/portable/qwen35.py:473-525` |
| dtype 硬门禁 | `python/pypto/portable/qwen35.py:1520-1521` |
| embedding / lm_head 站点 | `python/pypto/portable/qwen35.py:1593-1598`、`1794-1810` |
| MLP 三投影 | `python/pypto/portable/qwen35.py:1762,1763,1776` |
| attention q/k/v/o | `python/pypto/portable/qwen35.py:2123,2173,2219,2322` |
| GDR qkv/z/b/a/out | `python/pypto/portable/qwen35.py:2366,2414,2426,2437,2537` |
| 现成 matmul dtype 约束 | `python/pypto/lowering/cpu/__init__.py:1526-1527` |
| CPU 支持 dtype | `python/pypto/lowering/cpu/__init__.py:34-52` |
| CPU op alias 表 | `python/pypto/lowering/cpu/__init__.py:57-260` |
| vector 支持算子 | `python/pypto/lowering/cpu/vector/plan.py:44-75` |
| GPU 支持 dtype/算子 | `python/pypto/lowering/gpu/plan.py:43-50` |
| AVX2 特征与算子 | `python/pypto/lowering/cpu/x86/avx2/capabilities.py:136`；`.../avx2/lowering.py:31-44` |
| AVX-512 特征与算子 | `python/pypto/lowering/cpu/x86/avx512/capabilities.py:84,92,109-110,158`；`.../avx512/lowering.py:41-54` |
| VNNI helper 未接公共 op | `python/pypto/compiler/targets/cpu_avx512.py:441,900`；`python/pypto/backends/cpu/x86/avx512.py:1190-1217` |
| SVE2 独立探测 | `python/pypto/lowering/cpu/aarch64/sve/capabilities.py:24,79-80,94,112-113,135-136` |
| CUDA int8 仅存储 | `python/pypto/compiler/targets/cuda.py:109-119` |
| HIP 算子/dtype | `python/pypto/compiler/targets/hip.py:73-106` |
| AMD int8 能力未知 | `python/pypto/target/amd.py:30-38` |
| binding 结构 | `python/pypto/portable/runtime_binding.py:29-31,33-51,168,347-370,373-404,467-475,522,929-951,988-1067` |
| 经典前端量化（不可移植） | `python/pypto/op/quantization.py:22,69,73,120` |

### A.3 现有可复用符号（避免重复造轮子）

`_linear_projection`（`qwen35.py:335`）、`_as_f32`/`_cast_from_f32`（`qwen35.py:218,229`）、`_constant_scalar`（`:247`）、`_decoder_parameter_entry`（`:385`）、`build_batched_matmul`（`:2990`）、`build_binding_schema`（`runtime_binding.py:759`）、`build_packed_layout`（`:998`）、`assemble_launch_request`（`:1180`）、`Artifact`（`compiler/artifact.py`）、`TargetSpec`/`CapabilitySet`（`target/spec.py:20`、`target/capabilities.py:13`）。

> 注意：`CapabilitySet` 已有 `dtypes` / `matrix_capabilities` 字段（`target/capabilities.py:13-27`），int8 与 dp4a/imma/VNNI 的能力声明应走这里，而不是新增散装布尔。

## 附录 B：建议的实现任务拆分（供后续排期，不在本任务执行）

| 任务 | 目标 | 依赖 |
|---|---|---|
| `qwen35-w8a8-scheme` | core_ir 的 `QuantizedTensorDesc` + 4 个 opcode 的 scalar golden（R1–R5） | 本契约冻结 |
| `qwen35-w8a8-binding` | binding schema v2 + packed layout v2 + 覆盖率 manifest（R7–R9） | scheme |
| `qwen35-w8a8-cpu-avx2` | AVX2 widening 路径（R6） | scheme + binding |
| `qwen35-w8a8-cpu-avx512` | 接通 VNNI（含 s8 符号补偿）或 `vpdpwssd`（R6） | avx2 |
| `qwen35-w8a8-sve256` | widening fallback；SDOT 另附证据（R6） | avx512 |
| `qwen35-w8a8-layer-ladder` | L2/L3 阶梯与报告（R10–R11） | 任一后端 |
| `qwen35-w8a8-cuda` / `qwen35-w8a8-amd-static` | 各自能力门禁下的正确性 kernel | layer-ladder |
| `qwen35-w8a8-model-validation` | L4–L6 整网对比与覆盖率报告（R12–R14） | 全部 |

## 附录 C：本任务实际做了什么 / 没做什么

- 做了：读取控制仓规范与配置；只读 grep/读 integration worktree 的实现面；确认量化实现的现状缺口；产出本契约与 `validation.json`。
- 没做：任何代码修改、任何 kernel 实现、任何 lowering/编译、任何权重下载或加载、任何性能测量、任何 heavy 命令。
- 本 worktree 保持干净（除本报告写入项目外的 `_meta` 证据目录，不在任何 Git 仓库内）。
