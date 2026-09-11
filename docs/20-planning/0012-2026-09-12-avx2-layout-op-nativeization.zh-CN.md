# PyPTO-X AVX2 layout 类算子原生化（avx2-layout-native）

文档编号：`0012`

日期：2026-09-12（Asia/Shanghai）

状态：`R2_FIXED_PENDING_THE_SECOND_VERIFICATION_ROUND`（首轮独立验收 PASS_WITH_BOUNDARIES；8 条证据/流程 finding 已回修，见 §8）

用途：把 x86 AVX2 后端缺失的 layout/split/concat 原生执行面补齐，镜像 B6（AVX-512，文档 `0004`）与 SVE256 已验收的同语义实现；补对称性，不发明新设计。本文自包含，设计、版本链、覆盖矩阵、位级证据与 UNGATED 性能都在这里。

关系：B6 = AVX-512 版；本文 = AVX2 版。共同语义真源是共享的 `LayoutPlan`/`SplitPlan`/`ConcatPlan` 与 PXLD-v2 descriptor wire。

基线：`port/pypto-x-integration 77981213e`（B6 已在主干）；实现分支 `work/avx2-layout-native`；实现提交 `4e52610f3`。

---

## 1. 结论摘要

1. **AVX2 的 layout/split/concat 已原生执行**：新增 PXLD-v2 descriptor 驱动的 C 内核平面（`ptx_avx2_layout_{f32,bf16,i8}`、`split_*`、`concat_*`），reshape/view/contiguous/transpose/slice 与 split/concat 在 f32/bf16/int8 上不再走 vector-common host reference。
2. **位级一致（sha256 全 buffer）**：新增 **135** 个聚焦用例（r2；首版 134），其中 **116 次逐位差分执行**（layout 60 + layout sNaN 10 + split 21 + split sNaN 2 + concat 21 + concat sNaN 2），主口径是与**自写纯 stdlib oracle**（按 Core IR 语义逐坐标枚举；不 import 被测实现）逐字节比对，全部相等；另有 6 次与真实 vector-common reference 路径做二次交叉校验。覆盖 7 ops × f32/bf16/int8 × 空轴/退化/非单位步长 slice/rank2–4 置换/非倍数/批前缀/sNaN payload。
3. **声明=执行**：`layout_native_contract()` 同时供 artifact 元数据（`_execution_metadata`）与运行期分发使用；`_decode_artifact` 在 load 时重算整张 execution-mode 表并逐键比对。spy 用例捕获 `_call_layout_kernel/_call_split_kernel/_call_concat_kernel` 的实际 kernel 名，与 metadata 声明逐项一致。
4. **fail-closed**：payload 4→5、ABI `:5`→`:6`；旧 payload/旧 ABI/改 1 字节 descriptor/篡改 wire/版本/digest/mode/alias 全部结构化拒绝且**不写输出**；非法形状组合在 lowering 期 `CpuVectorValidationError`/`CpuVectorUnsupportedError` 拒绝。未覆盖 dtype（int32/int64/uint/bool/fp16）显式回退 host_reference，`layout_native_fallback_operations` 记录 reason，运行期 `explicit_host_fallbacks` 计数。
5. **不回归**：聚焦 **337 passed**（r2，持锁重跑；AVX2 + layout 135 + packed + W8A8 AVX2 + Qwen35 AVX2 parity + vector-common）；全量 `python/tests/ut/pypto_x` **1528 passed / 7 skipped / 0 failed**（13:48，commit `477faa357`）；收集数 base 1400 → head **1535**（+135，唯一消失的 base 用例是刻意的边界改名）。
6. **性能（UNGATED，不构成门槛）**：见 §6。`host_reference_s` 与 5 个 layout 算子的 per-op 中位均按同协议 before/after 给出；launch/op 分解单列，明确标注 12 vCPU KVM guest 无 cpufreq 与窗口噪声。
7. **未扩大范围**：未改图、未改 `LayoutPlan`/`SplitPlan`/`ConcatPlan` 语义与 digest、未改 SVE256/AVX-512/CUDA/AMD/Ascend、未动 indexing/gather/embedding、未启用 view alias 零拷贝、未动 W8A8 权重布局、未做 dispatch 压缩。

---

## 2. 方法与接口

### 2.1 C 内核（`python/pypto/compiler/targets/cpu_avx2.py::_NATIVE_SOURCE`）

- `int ptx_avx2_layout_{f32,bf16,i8}(const unsigned char *descriptor, size_t len, const void *in, void *out)`
  - 解码/校验 `PXLD` v2：magic/version/rank≤64/kind∈{copy,transpose,slice}/数组长度精确匹配/行主序 stride/permutation 唯一/slice 上界/元素数自洽；任一违反返回非 0（fail-closed）。
  - `logical_layout_copy`（reshape/view/contiguous）：AVX2 整数 lane 快速拷贝（f32 8 lane / bf16 16 lane，NaN lane 用整数位运算置 quiet bit；i8 `memcpy`），尾部标量；与 host reference 的 Python-float round-trip 逐位一致。
  - `transpose`/`slice`：输出坐标 odometer 增量维护 source offset（O(1)/元素，rank≤64），逐元素 gather + quiet（f32/bf16）。
- `int ptx_avx2_split_{f32,bf16,i8}(in, out, outer, inner, source_axis, axis_offset, axis_size)`：沿 split axis 的段为连续区间，按 outer 行拷贝；空段（`row==0` 或 `outer==0`）直接成功返回，不触碰 NULL 指针。
- `int ptx_avx2_concat_{f32,bf16,i8}(in, out, outer, inner, output_axis, axis_offset, axis_size)`：把每个输入段写到输出 axis 列。
- 全部返回 `int` status，运行期用 `_status_symbol` 检查；非 0 → `CpuAvx2ExecutionError` + `descriptor_rejections` 计数。

### 2.2 Python 侧接口

- `compiler/targets/cpu_avx2.py`：`NATIVE_LAYOUT_MODE/NATIVE_SPLIT_MODE/NATIVE_CONCAT_MODE`、`NATIVE_LAYOUT_KERNEL_VERSION`、`_NATIVE_LAYOUT_DTYPES`/`_NATIVE_LAYOUT_OPERATIONS`/`_NATIVE_LAYOUT_KERNEL_SUFFIX`、`layout_native_contract(operation, plan)`、`layout_native_fallback_reason(operation, plan)`；`_execution_metadata` 新增 `layout_native_kernel_version=1`、`layout_native_operations`、`layout_native_fallback_operations`、`layout_view_alias_mode/reason`、`native_helper_symbols`；`layout_descriptor_wire` 由 `host-reference:v1` 改为 `native:v1:pxld-v2`。
- `backends/cpu/x86/avx2.py`：`_layout_descriptor_bytes`（PXLD v2，与 SVE256/AVX-512 byte-for-byte 相同）、`_layout_descriptor_buffer`（按 `descriptor_digest` 缓存 + 8 字节对齐 buffer）、`_call_layout_kernel/_call_split_kernel/_call_concat_kernel`、`Avx2Runtime._execute_layout/_execute_split/_execute_concat`、`_split_geometry`、`layout_native_stats()`/`reset_layout_native_stats()`、`_layout_view_alias_prefix()`。
- 运行期额外校验：contract dtype ↔ plan dtype、`descriptor_digest` ↔ plan、source/output element count ↔ descriptor、split/concat 几何 ↔ descriptor；任一不符即错误，不静默换路。
- `_execute_operation` 分发顺序保持 **w8a8 → packed → layout → 既有原生/回退**，所以既有 B1 packed rank-2 transpose 的优先级不变。

### 2.3 语义对照物（只读）

`backends/cpu/x86/avx512.py` 与 `backends/cpu/aarch64/sve256.py` 的 PXLD v2 wire、`compiler/targets/cpu_avx512.py` 的 kernel 语义为唯一真源；本批以逐位差分而非目测对齐。

---

## 3. 覆盖矩阵

| 算子 | dtype | 模式 / 内核 | 说明 |
|---|---|---|---|
| reshape / view / contiguous | float32 / bf16 / int8 | `native_avx2_layout`（`layout_*`，copy kind） | 任意 rank ≤64；空轴/退化 shape 安全；AVX2 lane quiet |
| transpose | f32/bf16/int8，rank2 且 permutation=(1,0)、无 slice | `native_avx2_packed_transpose`（**既有面，判定未改**） | 保持 B1 优先；见 §7 的 sNaN 边界 |
| transpose | f32/bf16/int8，其余任意 rank/permutation（含 rank3/4） | `native_avx2_layout` | odometer gather + quiet |
| slice | f32/bf16/int8 | `native_avx2_layout` | 正步长、任意 starts/steps（含非单位步长）；空结果安全 |
| split | f32/bf16/int8 | `native_avx2_split` | 每输出一次 kernel call；sizes 可为 0 |
| concat | f32/bf16/int8 | `native_avx2_concat` | 每输入一次 kernel call；axis 段 |
| layout（任意 kind） | 其他 dtype（int32/int64/uint/bool/fp16/…） | `host_reference` | **显式回退**：metadata `layout_native_fallback_operations` 记录 reason；runtime 计数 `explicit_host_fallbacks` |
| gather / embedding | 不变 | embedding 仍走既有 packed（f32/bf16）；其余 `host_reference` | 不在本批范围（indexing wire 保持 `host-reference:v1`） |
| where / compare / iota | 不变 | `host_reference` | 不在本批范围 |
| rank > 64 / step ≤ 0 / 未知 kind | — | 构造期 validation error（fail-closed，不进入执行） | 与 `_LAYOUT_MAX_RANK` 同常量 |

**view alias（零拷贝）未启用**：`layout_view_alias_mode=disabled`，静态前缀判定 + `view_alias_candidates` 计数已实现，但 SSA liveness 别名证明未接线，输出仍为真实拷贝（与 B6 一致）。

---

## 4. 版本链与 fail-closed

| 项 | before | after |
|---|---|---|
| `ARTIFACT_PAYLOAD_VERSION` | 4 | **5** |
| `NATIVE_ABI_MARKER` / `ptx_avx2_abi` | `pypto-x.cpu.avx2:5` | **`pypto-x.cpu.avx2:6`** |
| `layout_descriptor_wire` | `host-reference:v1` | **`native:v1:pxld-v2`** |
| `indexing_descriptor_wire` | `host-reference:v1` | 不变 |
| `layout_plan_version` / `LayoutPlan` digest | 2 | 不变（descriptor 语义冻结） |
| `NATIVE_LAYOUT_KERNEL_VERSION` | — | **1** |

拒绝路径（全部结构化，不写输出）：

- 旧 payload（`artifact_payload_version != 5`）→ `CpuAvx2ArtifactError("unsupported AVX2 artifact payload version")`；
- 旧 ABI marker（`.so` 内 `ptx_avx2_abi` 不匹配）→ `CpuAvx2ArtifactError("... safety marker ...")`；
- metadata 任一 `_execution_metadata` 键被改（wire/version/mode/kernel/digest/alias）→ load 期重算比对拒绝；
- descriptor 被改 1 字节（magic/version/rank/kind/element counts/数组内容）→ C decode/validate 返回 -1，运行期 `descriptor_rejections++` 并抛错，调用方 output buffer 保持原样；
- descriptor digest 与 plan 不一致 → dispatch 期拒绝；
- 非法 dtype/形状组合 → lowering 期 `CpuVectorValidationError`/`CpuVectorUnsupportedError`；未覆盖 dtype → 显式 host fallback（不是静默）。

---

## 5. 验收证据

主证据目录：`/home/chiro/projects/pypto/worktrees/_meta/pypto-x/avx2-layout-native/`。

```text
位级:     python/tests/ut/pypto_x/test_cpu_avx2_layout.py（135 例，纯 stdlib oracle 主口径；含 r2 的全 184 descriptor 字节位置与 AVX2↔AVX-512 wire 身份回归）
声明=执行: 同文件 spy 用例 + _decode_artifact 逐键比对
fail-closed: 同文件 §fail-closed 用例（旧 payload/ABI、单字节 descriptor、wire/mode/digest/alias 篡改、非法形状）
聚焦:     logs/focused-fix.log（337 passed，r2 持锁）
全量:     logs/full-pytest-after.out（collect 数 base 1400）
性能:     raw/l1_base_{decode_t1,prefill_t5}.json(.records.json) vs raw/l1_after_*（UNGATED）
```

最终结果（持锁窗口，worktree commit `4e52610f3`）：

```text
聚焦:  337 passed（logs/focused-fix.log，r2 持锁）
全量:  1528 passed / 7 skipped / 0 failed in 13:48（logs/full-pytest-after.out，r2 持锁）
收集:  base 1400 → head 1535（+135）；逐名 diff 只有新增 135 个 + 1 个刻意改名
       （test_avx2_rank3_transpose_stays_on_the_host_reference →
        test_avx2_rank3_transpose_is_a_native_layout_reorder），无其它删减
       （logs/collect-base.out、logs/collect-after.out、raw/collect-names-diff.txt）
PXLD wire 与 AVX-512 byte-for-byte: raw/pxld-wire-identity.txt（9/9 identical）
```

---

## 6. 性能（UNGATED）

> **r2 修订（2026-09-12）**：首轮 after 3 轮混用了两种 `l1_avx2_instrument.py` 版本、before 基线窗口曾与本任务自己的迭代并发（已作废重测），且发现一段未持锁的聚焦运行。回修后 **before/after 全部改为持锁、同一 instrument 版本** 重跑，最终数字为：decode launch **28.505 → 20.400 s**、`host_reference_s` **14.670 → 7.743 s**；prefill launch **127.942 → 109.308 s**、`host_reference_s` **59.752 → 39.627 s**；per-op median（ms/call，decode / prefill）：reshape 0.0190 / 0.0230、slice 0.0260 / 0.0180、concat 0.0255 / 0.0330、split 0.0340 / 0.0410、transpose 0.0300 / 0.0180。**下表 §6.1/§6.2 的首版数字仅作历史对照，引用请用本修订与证据目录 `raw/perf-before-after.md`**。

> 本机为 12 vCPU KVM guest、无 cpufreq，且与其他 agent 任务共享窗口；以下只报中位与离散度，**不构成权威加速比，不得用于任何 ratio gate**。
> 协议：`qwen35_weighted_execution_driver.py --stage s1 --backend avx2 --precision bf16 --shape-profile model`（真实图结构与 op 数，合成权重），per-op 由证据侧 instrumented runtime 按 artifact `execution_modes` 如实计时。
> before = 导出的基线树 `77981213e` 1 轮（decode T=1 past=5 + prefill T=5 past=0）；after = 本 worktree 3 轮。

### 6.1 总账

| 段 | 轮次 | 图 op 数 | launch 中位 s | Σ(op) 中位 s | host_reference_s | 原生 s |
|---|---|---:|---:|---:|---:|---:|
| decode T=1 | before ×1 | 4550 | 28.849 | 26.047 | 15.325 | 7.489 |
| decode T=1 | after ×3 | 4550 | **20.393** | **17.806** | **7.754** | 6.720 |
| prefill T=5 | before ×1 | 6728 | 130.333 | 125.631 | 61.223 | 47.570 |
| prefill T=5 | after ×3 | 6728 | **108.325** | **103.506** | **39.393** | 48.082 |

after 三轮离散度（max/min−1）：decode launch 2.6% / host_reference_s 2.8%；prefill launch 4.3% / host_reference_s 3.4%。
launch−Σ(op) 残差（ERR-0006 口径）：decode 2.80→2.59 s，prefill 4.70→4.82 s，**基本不变**（本批不动派发层）。
after 剩余的 `host_reference_s` 全部是 AVX2 尚无原生平面的算子：`broadcast`（约 39 s/494 calls，prefill）与 `where/compare/iota`（共 <1 s）；**layout 类自身为 0**（无 `layout_native_fallback_operations`）。

### 6.2 per-op median ms/call

| 算子 | decode before | decode after | prefill before | prefill after | after mode |
|---|---:|---:|---:|---:|---|
| reshape | 1.3865 | **0.0200** | 3.8175 | **0.0230** | `native_avx2_layout` |
| slice | 4.5820 | **0.0260** | 5.8625 | **0.0190** | `native_avx2_layout` |
| concat | 4.0275 | **0.0300** | 13.3755 | **0.0325** | `native_avx2_concat` |
| split | 6.6900 | **0.0350** | 33.2835 | **0.0430** | `native_avx2_split` |
| transpose（mixed: generic+packed） | 4.5530 | **0.0300** | 4.2040 | **0.0170** | `native_avx2_layout` + `native_avx2_packed_transpose` |

> 口径：先取每轮内同 op 的 per-call 中位，再取 3 轮中位；before 只有 1 轮（同协议）。transpose 的均值仍被 rank-2 packed 权重大转置抬高，中位反映了绝大多数 generic 调用。

### 6.3 其余布局类回退计数

| 项 | decode before | decode after | prefill before | prefill after |
|---|---:|---:|---:|---:|
| `host_reference_operations` 声明条目（含重复） | 1612 | **356** | 2782 | **500** |
| 其中 distinct | broadcast/concat/reshape/slice/split/transpose/where | **broadcast/where** | 同左 | **broadcast/where** |
| `layout_native_fallback_operations` | 0（wire 还是 host-reference） | **0** | 0 | **0** |

after 的 6 轮全部在持锁窗口内完成：decode r1–r3 与 prefill r1 来自 21:24:03Z 的窗口（该窗口在 prefill r2 阶段被外部 SIGTERM 中断，resource log rc=143），prefill r2/r3 与全量回归来自 21:41:45Z 的续跑窗口（rc=0）；脚本已做成可续跑，被中断窗口的已完成轮次被复用。原始数据见 `raw/l1_after_*.json(.records.json)` 与 `raw/perf-analysis-*.txt`，汇总表 `raw/perf-before-after.md`。

---

## 7. 已知边界与未覆盖面

1. **packed rank-2 transpose 的 sNaN 语义是既有边界**：B1 packed 平面是纯元素置换，**保留** signaling-NaN payload；而 host reference 与新的 generic PXLD 平面会把 sNaN quiet 化。本批按范围裁决不动 packed kernel，因此该差异被显式记录并有专门用例（`test_rank2_packed_transpose_preserves_snan_payloads`）；f32/bf16 的 rank-2 transpose 走 packed 时与 host reference 在 sNaN payload 上不一致，普通权重/激活（无 sNaN）无差异。
2. **未覆盖 dtype**：int32/int64/uint8/uint16/uint32/uint64/bool/fp16 的 layout 仍是 host_reference + metadata reason。
3. **indexing**：gather/embedding/select 不在本批；`indexing_descriptor_wire` 仍为 `host-reference:v1`（embedding 的 packed 面不变）。
4. **view alias 零拷贝未启用**（stat-only；需要 SSA liveness 别名证明）。
5. **W8A8 权重布局未动**（W8A8 v2 M2 任务拥有）。
6. **dispatch（launch 残差）未压缩**：本批只改 op 侧；launch 残差语义见 ERR-0006。
7. **性能为 UNGATED**：本机 KVM guest 无 cpufreq、共享窗口；degrade/收益只作证据。

---

## 8. 资源与成本

- 位置：本机（12 vCPU），重活走 `scripts/resource/run_local_heavy.sh`（`--min-available-mib 8192 --safety-floor-mib 4096 --max-cpus 6`），返回 75/69 即等待重试；不涉及 GamePC / A2 / A3 / 920B / NPU / GPU。
- **诚实披露**：2026-09-11T21:30:13Z–21:35:13Z 期间，一次误调用（脚本没有 dry-run 参数，本想做语法检查却直接执行）让 after 载荷在 **未持锁** 状态下跑了约 5 分钟；当时 `local` 锁为 FREE、没有抢占任何任务。该窗口产出的 prefill r2/r3 与 pytest 输出已删除，并在随后的持锁窗口完整重跑；最终证据只包含持锁窗口数据（`logs/README-aborted-window.txt`、`logs/heavy_inner_after.aborted-window-timeline.txt`）。

---

## 8. r2 回修记录（2026-09-12，commit `477faa357`）

首轮独立验收判 **PASS_WITH_BOUNDARIES**（无功能性失败），并提出 8 条证据/流程 finding，全部回修：

| # | finding | 回修 |
|---|---|---|
| 1 | 声称"全部重活持锁"不成立（聚焦 336 例等 5 段未持锁） | `raw/lock-discipline-disclosure.md` 列全 5 段并写清 owner/处理；最终证据全部持锁重跑；受扰 baseline 作废重测 |
| 2 | `every_single_byte…` 实际只测 8 位置 | 改名 `test_all_descriptor_byte_mutations_are_rejected_except_reserved_bytes` 并扩到全 **184** 位置：180 拒绝、reserved 20–23 良性（输出 sha256 不变） |
| 3 | 跨后端 wire 身份未固化 | 新增 `test_pxld_wire_is_byte_identical_to_avx512`（7 case × 3 dtype = 21/21，digest + 逐字节） |
| 4 | after 3 轮混用两种 instrument 版本 | `raw/instrument-versions.md` 列差异；before/after 全部同版本重跑 |
| 5 | 无锁窗口残留 `.so` 被锁内复用 | 删除全部 `work/after-*` 后锁内重新编译 |
| 6 | `raw/` 无逐用例张量 sha256 | 新增 manifest 钩子；`raw/layout-case-sha256.jsonl` = 147 buffer 行 / 116 case 全 match |
| 7 | 设计文档不在提交内 | 本文件在控制仓（`2488e42`），实现提交内无副本 |
| 8 | 锁窗口 1 `rc=143` 原因未披露 | `raw/window1-termination-cause.md`：本任务自身前台 `sleep 300` 被工具清理、命中未 `setsid` 的 waiter 链；后续 waiter 全部 `setsid` |

残余边界（维持）：真实权重 s3/L4 数值、rank>4 随机置换、rank0/1、int8 sNaN、W8A8 图上 layout 计数、性能 UNGATED。
