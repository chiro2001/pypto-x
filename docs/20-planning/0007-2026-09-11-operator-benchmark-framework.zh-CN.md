# 单算子测试与性能对比框架（op-bench）

文档编号：`0007`

日期：2026-09-11（Asia/Shanghai）

状态：`APPROVED_BY_USER_DIRECTION_PENDING_IMPLEMENTATION`

用户方向（2026-09-11）：**"把单算子测试框架做好，这样才能进行算子层面的性能对比"**。

---

## 1. 为什么它排在最前面

当前所有关于"算子快不快/对不对"的结论都缺一个**可比的口径**：

| 已有的东西 | 它不解决什么 |
|---|---|
| `scripts/perf/local_l0_microbench.py`（B3c 产出） | 只覆盖 elementwise/reduction，无 GEMM、无 vendor 对比、无跨平台；且 L0 判据曾出现 0/10 case 全轮通过 |
| `qwen35_weighted_execution_driver.py`（B3c L1） | 是**整网**分解（launch/op/dispatch 三分账），不是单算子对比 |
| W8J `shape-ratio-table.json` | 是**一次性**的 12 形状对比（our vs oneDNN），不是可复用框架 |
| 各任务的 ad-hoc benchmark | 口径（线程数/warmup/轮次/是否含 pack）各不相同，**跨任务不可比** |

而下面这些决策**全都依赖单算子对比**：

- vendor GEMM 选型（Q8、0006）——四个 provider 到底谁在哪些形状上赢；
- U/P 设计里的**旋钮 evidence**（0005 §4.2 要求"无证据的旋钮不得入库"）——证据就由这个框架产出；
- **N4 vs 外包图执行**的判断（N4.0）——必须先把"算子自身的耗时"与"调度开销"分开量；
- W8J 的性能门（当前 0/12，注入被禁）——需要可复现的算子级证据才能重开；
- 跨平台对比（x86 / aarch64 鲲鹏 / NVIDIA）——需要**同一 schema** 才能横向比。

## 2. 定位与边界

**是**：单算子的可用性 / 正确性 / 性能对比框架，provider 可插拔、口径可复现、证据可追溯、跨平台同 schema。

**不是**：整网性能基准（那是 L1/B3c）、不是功能回归套件（那是 pytest）、不是性能门槛（那是 `perf_lock`/B4）。

## 3. 对象模型

```text
OpCase = (op, shape, dtype, provider, threads, binding, layout_policy, inputs_seed, extra)
```

- **op registry**（首期）：`matmul/linear`（含 BF16 / W8A8 两种精度）、`reduce_sum/mean/max`、`layout(reshape/slice/transpose/concat/split)`、`cast`、W8A8 四 primitive（qmatmul/quantize/dequantize/epilogue）。每个 op 声明：语义契约引用、合法 dtype、可用 provider、参考实现。
- **provider registry**：`portable`（我们自己的参考实现）/ `native`（我们自己的原生内核）/ `vendor:<lib>`（oneDNN / AOCL-LPGEMM / libxsmm / cuBLASLt / KleidiAI …）。每个 provider 必须声明：
  - 可用性探测方式（探测失败 → 该 provider 在报告中标 `unavailable`，**不许静默跳过**）；
  - 版本指纹（库名 + 版本 + 构建选项 + sha256）；
  - **计时语义**：是否包含权重 pack、是否包含内存分配、是否包含首次 JIT（这直接决定两 provider 能不能比）；
  - 线程语义（如何设置线程数/绑定）；
  - 精度类别（`bit_exact` / `bounded_tolerance` + band 来源）。
- **shape 套件**（三层，可分组跑）：
  1. `model`：真实模型形状（Qwen3.5-0.8B 的 12 个 linear 形状 + 真实 layout/reduce 形状）；
  2. `boundary`：m=1、n=32、K 尾（K=7/17/cmake 非倍数）、rank2–4、空轴、非单位步长 slice；
  3. `sweep`（可选）：K/N 连续扫描，用于画曲线而不是单点。

## 4. 测量协议（**与 B3c/ERR-0004 同源，不许另立一套**）

1. warmup + K 轮重复；报**中位数**与**跨轮离散度**，不报单次最优；
2. 外点判据沿用 `ERR-0004`：`|x − median| > max(3×1.4826×MAD, floor)`，`floor = max(1 µs, 0.02×median)`；`outlier_ratio > 0.10` → `UNSTABLE`；
3. **必须显式声明计时区间的边界**：计时是否包含 `pack`/`alloc`/首次 JIT；两 provider 若语义不同，必须在报告里标 `not_directly_comparable` 而不是硬比；
4. 本机（12 vCPU KVM 无 cpufreq、窗口非静默）产出的数字**一律标 `UNGATED`**，只能报中位与离散度；**不得进入 `perf_lock`**；
5. 跨平台运行纪律：本地 x86 与 A3 都经各自的 heavy 锁；A3 多进程并发按 Q1 实测（1 case = 1 进程 × 8 线程 × 8 个不重叠物理核，16–20 并发为上限）；参照侧单进程多线程按 Q1 的绑核配方。

## 5. 正确性判定（每个 case 必须带）

- 每个 case 声明**精度类别**：
  - `bit_exact`：与参考实现全 buffer `sha256` 一致（例如 W8A8 整数路径、我们的 portable ↔ native）；
  - `bounded_tolerance`：给出 band（来源必须写明，例如 C8 的 `|fakequant − fp32|` 噪声底）。
- provider 与参考不一致时：**先判"是不是语义不同"**（例如 vendor 的累加顺序/中间精度不同），再判"是不是 bug"；两者结论必须分开写。
- 参考实现优先级：`portable`（我们自己的参考路径）> 独立第三方（transformers/torch）。

## 6. 输出与可追溯

```text
machine-readable JSON：{host_fingerprint, library_versions, provider_registry_state,
                        cases: [{op, shape, dtype, provider, knobs, seconds:{median,dispersion,N},
                                 precision:{class, basis, verdict}, notes}]}
human-readable Markdown：按 op/shape 的对比表（含 provider 可用性、精度类别、UNGATED 标注）
sha256sums.txt：原始输出与脚本清单
```

**硬要求**：任何一个数字都要能追到"哪台机器 + 哪个 commit + 哪个库版本 + 哪组旋钮 + 哪份原始 JSON"。

## 7. 需复用的既有资产（不许重造）

- `scripts/perf/local_l0_microbench.py` 与 `scripts/perf/aggregate_driver_progress.py`（B3c）；
- W8J 的 12 形状与 `shape-ratio-table.json` 口径（作为 `model` 套件的首批数据）；
- Q8（vendor-gemm-survey，在途）的四路对比 harness —— **O1 应先把它的可复用部分抽出来**，而不是另写一套；
- `scripts/resource/run_local_heavy.sh`（锁与资源守卫）、`docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md` + ERR-0004。

## 8. 验收（框架自身的测试）

1. **可复现性**：同一 case 连续两次运行，报告结构一致、中位数差异落在离散度内；
2. **错误注入**：provider 不可用、库版本不符、非法旋钮组合 → 必须显式报错或标 `unavailable`，**不许静默跳过**；
3. **schema 校验**：输出 JSON 通过 schema 校验；缺字段即失败；
4. **口径对齐**：用 Q8 已测的 12 形状跑一遍，**应与 Q8 的数字量级一致**（允许平台/窗口差异，但要能解释）；
5. **跨平台**：在 x86 与 A3 至少各跑通一个 case 并产出同 schema 报告（NVIDIA 可后补）。

## 9. 非目标

- 不做整网基准（L1 已覆盖）；不做功能回归（pytest 已覆盖）；
- 不引入新的性能门槛（门槛是 B4 的事）；
- 不为了"让对比好看"而统一两个语义不同的 provider 的计时口径。
