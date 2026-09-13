# 0018 Qwen3.5 composite 组合化（batch 0048 切片 A）契约

- 任务：`qwen35-composite-composition`（batch 0048 切片 A）
- 分支/工作树：`work/qwen35-composite-composition` @
  `/home/chiro/projects/pypto/worktrees/pypto-x/qwen35-composite-composition`
- 交付基线：integration `923a72e26`（tree `5945f3448f184bb76a9eeea1c7a5a188ebdfb927`）
- 实现 commit：`54d2f4ea9`（`work/qwen35-composite-composition`，未 push）
- 语义真源：`python/pypto/portable/qwen35.py` 的既有公共 composite；重构**不改变**
  任何公共 builder 的签名，也不改变任何已构建程序的 Core IR（逐字节）。
- 证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/qwen35-composite-composition/`

## 1. 目标与硬不变量

问题是结构性的：`qwen35.py` 的 20 个公共 `build_*` 组合器互不调用（唯一例外是
`build_causal_conv1d → build_causal_conv1d_state` 别名），而可执行整图
`build_qwen35_text_decoder_graph` 内联了全部 24 层结构，形成"公共 composite 库"
与"可执行整图"两套并行实现，存在漂移风险。

本契约的全部工作都服从一个硬不变量：**重构前后每个公开程序与整图的
`CoreProgram.canonical_json()` 逐字节相同**。特别地：

1. 公共 builder 的签名、参数顺序、返回值和 `metadata` 不变；
2. opcode、操作数顺序、SSA 名字、dtype/shape、attributes、发射顺序不变；
3. 由此 `runtime_binding` 的 graph digest 不变：整图 bf16 v3 digest
   `66dd40770dbb41251f0f7ec9b15b75b604daddc95633aabc24d9e48ffff33c6d`（4,550 ops）
   与既有 `test_qwen35_weight_ingestion.py` 的固定断言继续成立；
4. 不改 `python/pypto/execution/**`、compiler targets、权重映射/绑定对外语义；
   不新增算子、不新增环境变量、不做动态 shape、不做性能声明。

## 2. 组合机制：`_compose_program`

新增一个模块内私有原语 `_compose_program(builder, program, bindings, returns=None,
rename=None)`（`qwen35.py`，`_contract` 之后）。它是"把子程序的数据流拼接进父
builder"的唯一入口：

- `bindings`：子函数的每个参数必须绑定到父程序的 `ValueRef`，且 `CoreType`
  必须完全相等；子程序参数不会被重新声明，因此父 builder 的公共签名不变；
- `returns`：选择子程序内部的 SSA 值（缺省用子程序自身返回值），只发射从这些值
  可达的 op，保持子程序原始顺序（对公共 builder 的"最终 cast/identity"做按需求值，
  不把无关尾部 op 带进父程序）；
- `rename`：SSA 名字改写，键可以是完整值名或前缀（最长匹配优先），用于匹配父
  程序既有的命名契约、或同一子 composite 在一个父程序里实例化两次；未经显式改写
  的命名冲突由 `_Builder._reserve` 直接报错，不会静默重命名；
- 拼接的 op 保留 opcode、操作数顺序、attributes、effect 与输出类型；非 pure op
  与带嵌套 region 的 op 会被拒绝（当前组合面全部是 pure op）。

配套 `_rope_core_result_name(shape, rotary_dim, prefix)` 暴露 `_rope_core` 的 FP32
结果 SSA 名（`_result` / `_full_result`），供组合方选择内部值。

## 3. 组合分层图（重构后）

```text
build_qwen35_text_decoder_graph
├── build_qwen35_position_control ──────────────┐
│   ├── build_qwen35_position_ids               │ 顶层一次组合
│   └── build_qwen35_causal_mask_from_positions │
├── 每个 full-attention 层 (_append_full_attention_graph)
│   ├── build_rms_norm ×2        (q/k norm, 取 FP32 scale 输出)
│   ├── build_rope_rotate_half ×2(q/k rope, 取 FP32 result)
│   ├── build_batched_matmul ×2  (QK^T / PV, FP32)
│   ├── build_causal_mask        (where, FP32)
│   └── build_stable_softmax     (FP32 概率)
└── 每个 GDR 层 (_append_gdr_graph)
    ├── build_causal_conv1d_state(conv/state 尾部)
    └── build_l2_norm ×2         (q/k L2, 取 normalized)

build_qwen35_attention_kv_cache
├── build_gqa_repeat_kv ×2
├── build_batched_matmul ×2
├── build_causal_mask
└── build_stable_softmax

build_attention_subgraph
├── build_rope_rotate_half
├── build_causal_mask
├── build_stable_softmax
└── build_gqa_repeat_kv

build_decoder_subgraph
├── build_rms_norm   (取 FP32 scale)
└── build_swiglu     (FP32 packed，取 product)

build_qwen35_gated_delta_conv_state
└── build_causal_conv1d_state
```

组合是"消费子 composite 的 FP32 内部值 + 显式 rename"的方式，而不是套用公共
builder 的最终 dtype 包装：这既是逐位不变的前提（避免 bf16 round-trip），也让
子 composite 的公式只存在一份。

## 4. IR 等价性证据（P2-1）

方法：同一份捕获脚本 `scripts/capture_ir.py` 在重构前后各生成 50 条程序快照
（20 个公共 builder × dtype/shape 变体 + 7 个整图契约变体），保存 `canonical_json`
与 sha256；`scripts/compare_ir.py` 逐条比较；`scripts/structural_hashes.py` 再对
五个维度独立算哈希（op 多重集合、def-use 边、SSA 类型签名、attributes 序列、
canonical digest）。

结果（`raw/ir-diff-step7.json`、`raw/ir-structural-equivalence.json`）：

- 50/50 条 `canonical_json` **逐字节相同**（`identical=50 different=0 missing=0`）；
- 50/50 条在 op 多重集合 / def-use / 类型签名 / attributes / canonical digest
  五个维度全部相同（`entries=50 all_identical=True`）。

逐项表（节选，完整表见 `raw/ir-structural-equivalence.json`；`= ` 表示五维全同）：

| builder 快照 | ops | baseline digest | 结论 |
| --- | --- | --- | --- |
| position_control (1×5,past0,h8) | 4 | `85319081ae06343f…` | 相同 |
| position_control (2×3,past7,int32) | 4 | `1af2109e130b38…` | 相同 |
| causal_conv1d_state (2×6×5,bf16) | 29 | `00f62d13d92ad5…` | 相同 |
| gated_delta_conv_state (1×5,h8,c12) | 36 | `713b701c6106ab…` | 相同 |
| decoder_subgraph (2×8,i10,bf16) | 18 | `33708c0d3401e7…` | 相同 |
| attention_kv_cache decode T=1/past5 | 20 | `aa2bc3f16c6aa1…` | 相同 |
| attention_kv_cache prefill T=3 | 20 | `8f0a7ad6fe2b15…` | 相同 |
| attention_subgraph bf16 | 26 | `3f4eb48831b6e9…` | 相同 |
| attention_subgraph f32/int32 | 21 | `3e0dc93ba7c477…` | 相同 |
| text_decoder model b1 t1 p4096 | 4550 | `66dd40770dbb41…` | 相同（冻结 digest 保持） |
| text_decoder model b1 t5 p0 | 6728 | `456ab519c9b9bc…` | 相同 |
| text_decoder synthetic b2 t3 | 5636 | `bd47d35bf9e9ac…` | 相同 |
| text_decoder w8a8 default/all/lm_head | 4400/4922/4401 | 各自 digest 不变 | 相同 |

没有任何"无法逐位"的差异：**差异清单为空**。原因：组合原语是"IR 级拼接"，
子 composite 的 FP32 内部值在父程序中就是原来的同一组 op；rename 只改 SSA 名且
由父程序既有命名契约决定，不引入任何新的 cast/identity/op。

## 5. 执行等价性证据（P2-2）

方法：`scripts/execution_differential.py` 在同一进程、同一环境下分别加载
**重构前基线模块**（`scripts/qwen35_baseline.py`，取自 `923a72e26`，
sha256 `5ede665f7e5c417a963920ca311a93dfb3088d4e33d0114f231685b6a994406d`）
与当前模块；每个语料先断言二者 `canonical_json()` 相同，然后用同一后端各编译、
各执行一次，比较输出字节 sha256 与 `max_abs_diff`。证据：`raw/exec_*.json`、
汇总 `raw/exec_summary.json`、日志 `logs/differential2.log`（资源锁 +
cgroup 内存上限内运行）。

| 后端 | 语料数 | IR 相同 | 成功执行 | 输出逐位相同 | max_abs_diff |
| --- | --- | --- | --- | --- | --- |
| cpu-scalar 参考 | 16 | 16 | 16 | **16** | **0.0** |
| AVX2 | 11 | 11 | 11 | **11** | **0.0** |
| AVX-512 | 11 | 11 | 11 | **11** | **0.0** |
| SVE256（QEMU, VL=32B） | 11 | 11 | 11 | **11** | **0.0** |
| IR-only（不执行） | 4 | 4 | - | - | - |

语料覆盖：整图 decode（synthetic T=1/past=5）、整图 prefill（T=5/past=0）、
attention_kv_cache decode T=1/past=5 与 prefill T=3、attention_subgraph、
GQA repeat、RoPE、RMSNorm、stable softmax、causal mask、batched matmul、
decoder_subgraph、gated_delta_conv_state、causal_conv1d_state、
position_control、**GDR T=128（3843 ops，四个后端都执行）**。

结论与说明：

1. 每个后端上 baseline/current 输出**逐位相同**，`max_abs_diff = 0.0`；本切片
   不需要任何容差口径（attention 也不例外）。
2. IR-only 的 4 条（GDR T=128、model T=1/past=4096、model T=5/past=0、
   w8a8 default）只断言 IR 逐字节相同；其中 model bf16 T=1/past=4096 的
   canonical digest 就是权重映射固定的 `66dd4077…`。
3. 跨后端观察（**非本次重构引入**）：同一语料在 SVE256 上的 GDR T=128 输出与
   cpu-scalar/AVX2/AVX-512 不同，softmax(f32) 在 AVX2/AVX-512 与
   cpu-scalar/SVE 不同；baseline 模块在同后端下表现一致，说明这是既有后端
   codegen/归约顺序差异，不属于本切片的漂移。
4. GDR T=128 的 cpu-scalar 双跑峰值 child RSS 为 monitor 采样的 5,508 MiB，在资源锁
   cgroup 上限（约 7.5 GiB）内完成；与历史验收 `execute-scalar.log` 的
   `peak_rss_kib=5776328`（约 5.5 GiB）一致，未出现紧张。

## 6. 测试（P3）

新增 `python/tests/ut/pypto_x/test_qwen35_composite_composition.py`（14 个用例）：

- 对每条组合边做"精确嵌入"断言：从子 composite 程序按 `returns` 做可达切片，
  以子函数参数绑定与 rename 规则映射后，在父程序 op 序列中必须找到唯一、连续的
  同构窗口（opcode、操作数顺序/类型、输出名/类型、attributes 全等）；
- 覆盖 position_control、gated_delta_conv_state、decoder_subgraph、
  attention_kv_cache（含两个 batched matmul）、attention_subgraph、整图顶层
  position_control、整图 layer_03 full-attention 与 layer_00 GDR 的组合关系；
- 冻结 7 条重构前 canonical digest（与基线快照一致），任何 op/attribute/顺序漂移
  都会失败。

聚焦回归（资源锁内，`logs/focused.log`）：

```text
env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut \
  python3 -m pytest -q -rs -p no:cacheprovider \
  <qwen35 composites/attention/conv/gdr/parity/connectivity/runtime_binding/weight_ingestion/
   position_native 共 19 个文件>
```

结果：**246 passed, 0 failed, 0 error, rc=0, 330.02 s**（最慢单例 74.6 s：
`test_cpu_avx2_position_native::test_compare_flat_contiguous_bytes_are_bit_exact_for_f32_and_bf16`）。

规则 8 全量（`logs/rule8.log`）：

```text
env PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python:python/tests/ut \
  python3 -m pytest -q -rs -p no:cacheprovider python/tests/ut/pypto_x
```

结果：**2194 passed, 7 skipped, 0 failed, 0 error, rc=0, 1306.49 s (21:46)**；
collect ≈ 2201（2194 passed + 7 skipped）。7 个 skip 全部是 `test_cuda_qwen_c2.py`
的 CUDA Driver 设备不可用（本机无 libcuda.so），与本次重构无关。

## 7. 未替换路径与后续计划

以下路径**没有**改成复用公共 composite，均有明确原因：

| 路径 | 现状 | 原因 / 后续 |
| --- | --- | --- |
| 整图 GQA repeat (`_repeat_kv_heads`) | 保持 slice+concat | 公共 `build_gqa_repeat_kv` 是 runtime-index `gather`；整图契约刻意避免整数 index ABI。两条语义各自冻结，改动会变 IR。 |
| MLP `silu(gate)*up` | 保持内联 | 公共 `build_swiglu` 以 packed last-axis 输入 + `split` 为契约；整图 gate/up 是两次独立投影，直接组合会引入额外 split/reshape。 |
| `_gdr_recurrent_inline` / `_gdr_step` | 保持内联 | 公共 `build_qwen35_gdr_recurrent_state` 是固定 shape 的独立 builder，与整图按 layer prefix 的 SSA 展开不是同一程序结构。 |
| `_rms_gated_core` | 保持私有 core | 没有公共 composite 对应（Qwen RMSNormGated 的直接 scale + bf16 round-trip 语义）。 |
| 整图 lm_head / in_proj matmul | 保持内联 `matmul` | 只对 full-attention 内 f32×f32 的 QK^T/PV 使用了 `build_batched_matmul`；其余 matmul 与显式 transpose/cast 链绑定，单独组合收益低。 |
| `build_qwen35_attention_kv_cache` / `build_attention_subgraph` / `build_decoder_subgraph` | 独立 harness | 已由子 composite 组合而成；整图走 `_append_*` 路径，二者共享同一批子 composite。 |

后续若要继续收敛，建议顺序：先把 `build_batched_matmul` 组合推进到整图
lm_head/in_proj（需要保持现有显式 transpose 语义），再评估把整图 MLP 改成
`build_swiglu` 的 IR 契约变更（会改 graph digest，需要图契约升版）。

## 8. 边界与未做到的面

- 本切片只做"结构复用 + 等价性"，不做任何数值/性能改进；不声明性能收益。
- 只在本机 KVM + QEMU(SVE256) 上执行等价性证据；A3/GamePC 未使用。
- 模型权重的真权重 driver/整网 smoke 不在本切片重跑；整图 digest 与
  binding/权重映射契约逐字节不变，既有 weight_ingestion/runtime_binding 测试
  作为回归门（聚焦集已包含）。
- 跨后端相同输入仍存在既有的 GDR/softmax 差异（见 §5.3）；本切片只保证
  "同后端 baseline == current 逐位"，不声称跨后端逐位。
- 控制仓中的本契约文档为新增文件，未由实现 agent 提交；由父方按控制仓流程登记。
- 已观察到 Python 侧构图成本小幅上升（model bf16 整图：0.123 s → 0.147 s，
  单次 best-of-2，UNGATED）：组合需要重新实例化子 composite 程序并再次
  `verify()`。这是构建期一次性成本，不是运行时/端到端性能声明，也未做优化；
  若要消除，可让子 composite 暴露可缓存的 `_emit_*` emitter（后续工作）。
