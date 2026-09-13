# PyPTO-X 0047 波次快照：int8 可证明包络（exact-blocked）与 x86 位置控制面原生化（AVX2/AVX-512）+ U4 收尾

状态：`ALL_FOUR_SLICES_LANDED_TWO_VERDICTS_IN_AND_ONE_IN_FLIGHT`（int8 / AVX2 / U4-followup 已判，AVX-512 在跑；按自动推送政策批次内已随开发推送）
批次：`batch_0047_int8_exact_blocked_and_avx2_position_native_2026_09_13`（见 `configs/development_lock.yaml`）
撰写：parent（自动批次）

## 0. 一句话

四件事：**①** 把 AOCL int8 `s8s8s32os32` 做成**可证明偏差 0** 的 opt-in provider（K 分块使内核内所有 f32 往返恒等 + 任意精度累加 + int32 范围校验），**②③** 把 AVX2/AVX-512 的 `iota/compare/where` 从 host_reference/host 预展开改成原生（payload 6→7、7→8，控制描述符 wire v1），**④** 把 U4 验收发现的 6 个仓库范围未登记环境变量登记归零。

## 1. 冻结点

| 项 | 值 |
|---|---|
| integration tip（frozen） | **`923a72e26`**（tree `5945f3448f184bb76a9eeea1c7a5a188ebdfb927`；含 AVX-512 round-2 声明修复） |
| 切片与提交 | int8：作者 `63980fc5e` → `c91d771b9`；AVX2：作者 `460f58506`+`b64a59f99` → `2f0b31d0f`+`3c2756c0e`；U4-followup：作者 `8cf9a1b4e` → `f3d71de04`；AVX-512：作者 `8dedc5909` → `0b5b5899e`，round-2 作者 `3e378ecff` → `923a72e26` |
| 补丁数 | **275**（0046 = 269；本批 +6：int8 1 + AVX2 2 + U4-followup 1 + AVX-512 2）；`git am` 干净树复算 **275/275**（两轮均复算，最终 tree `5945f3448` = 冻结 tree） |
| 父方最终 gate | **rc=0，2180 passed / 7 skipped / 0 failed，1358.54 s**（round-2 tip `923a72e26`，collect **2187**；7 skip 全为 libcuda 缺失）；round-1 tip `0b5b5899e` 上另有 rc=0 / 2179 passed / 1313.71 s（collect 2186） |
| 父方中间 gate | rc=0，2162 passed / 7 skipped / 0 failed，1253.66 s（tip `3c2756c0e`，collect 2169 = 2059 + int8 46 + AVX2 64） |

## 2. 切片 A：int8 exact-blocked（收 ERR-0010 的"包络非上界"）

- **机制**：新 opt-in provider `vendor:aocl-lpgemm-int8-exact-blocked`；`kc = min(2048, floor(2²⁴/(max|a|·max|b|)))`，把 K 切段后逐段调用 pinned `aocl_gemm_s8s8s32os32`（alpha=1、beta=0、post_op=None），段内所有 int32↔f32 往返因 |值|≤2²⁴ 而恒等，段结果用 Python 任意精度整数累加并做 int32 范围校验。
- **envelope**：`class=exact`、`deviation_bound_kind=exact_contract_bound`、`deviation_bound_available=true`、`deviation_bound_by_k={all_k_le_limit:0, 1024:0, 1040:0, 1041:0, 2048:0, 2049:0, 16696:0, 133144:0}`；`bound_derivation` 写明 kc 规则/2²⁴ 极限/段界公式/顺序无关性；`residual_assumptions` 给出可检验项（符号+调用参数+AOCL 文档）与**撤回路径**。
- **界既充分又紧**（父方前置实验 + 验收独立复现）：all-127 下 K=1040 偏差 0，**K=1041 偏差 1**；分块后 K=16696（17 段）/K=133144（129 段）偏差 **0**，精确和 2,147,479,576 距 INT32_MAX 仅 4,071。
- **零默认变化**：新条目不在 `default_providers`/`allowed_providers`，默认策略直接跳过；60 条非 opt-in 记录在两树逐字段相等；legacy artifact digest/envelope/字符串逐字节不变；仅 `KNOWN_PROVIDER_IDS` +1。
- **准入**：`numeric.require_proven_deviation_bound=true` 时选中并成功执行（替代此前一律拒绝）；legacy 仍 `proven_deviation_bound_unavailable`；fused `sym_quant` 仍 fail-closed。
- **成本**：`ceil(K/kc)` 次内核调用 + O(MK)+O(KN) 码值扫描；性能 **UNGATED**。

## 3. 切片 B：AVX2 位置控制面原生（payload 6→7）

- **wire**：`artifact_payload_version` 6→7、ELF `ptx_avx2_abi`/`NATIVE_ABI_MARKER` `:7`→`:8`、新增 `position_control_descriptor_wire="v1:pxio-pxcp-pxwh+fnv1a64"`（逐 op descriptor digest，dispatch 前重算）。
- **覆盖**：iota（int32/int64）、compare（f32/bf16/全部整型/bool，6 predicate）、where（f32/bf16/整型/bool）全原生；rank>16 逐 op 带原因回退；`host_preexpanded_operations` 恒 `[]`（where 不再预展开）。
- **语义**：`where` 条件按 host 的 Python 真值语义（字节 0 为假、2/255 为真）；`compare` 输出 bool 一字节、NaN 规则逐 kind 一致；sNaN 仅被选中分支静默且 payload 保留。
- **零漂移**：11 用例 Qwen3.5 语料两树编译执行，raw 输出 sha256 **11/11 一致**（基线全 host_reference → 新树全 native）。
- **性能（UNGATED）**：kernel vs host reference 中位比 5.7×–75.2×；native 期 host 逐元素 0 次调用（spy 证明）；Qwen 图 launch 含 artifact load，不做模型级外推。

## 4. 切片 C：AVX-512 位置控制面原生（payload 7→8）

- **wire**：`artifact_payload_version` 7→8、`ptx_avx512_abi`/`NATIVE_ABI_MARKER` `:8`→`:9`、同一 `v1:pxio-pxcp-pxwh+fnv1a64` 控制描述符 wire（与 SVE256 控制面布局一致）；native decoder 对 magic/version/reserved/长度/FNV 严格 fail-closed。**round-2（`923a72e26`）**：payload **8→9** + `PTX_FEATURE_MASK 0x01→0x81` + `_BASE_FEATURES`/runtime 安全集/`expected_required` 补 `avx2`（`-mavx512f` 隐式启用 AVX2，compare/where 归约尾部 16 条 VEX.256；缺 AVX2 宿主在 `dlopen` 前结构化拒绝），ABI marker 保持 `:9`，编译 flag 不变。 |
- **覆盖**：iota/compare/where 对全部 wire dtype 原生（bool 限 eq/ne）；rank>16 与 scalar 输出逐 op host 回退并带稳定 reason；`mode==native` 与零回退互为充要；`python_list_elements==0` 作为"消除 host 逐元素路径"的硬断言。
- **零漂移**：跨树 45 cases 不一致 0；单元 384 compare 元素 + 88 where 选择 + 20 非 bool condition 合成 + 8 iota 全绿。
- **边界**：round-1 的「只依赖 AVX-512F」表述被独立验收的反汇编证伪（`-mavx512f` 隐式启用 AVX2，compare/where 归约尾部 16 条 VEX.256 指令）→ **round-2 已修复**（声明与运行期特性门补 `avx2`、mask `0x81`、payload 9，编译 flag 不变，数值逐位不变）；DQ/BW/VL/FMA/gather 专属指令为 0，iota 为 native 标量循环；广播/非连续 lane 收集为 native 标量 odometer；非 bool where 条件在 Core IR 校验下不可达。
- **性能（UNGATED）**：1M 元素 iota 0.625 vs 0.917 s、compare 0.482 vs 1.784 s、where 0.783 vs 2.750 s，输出逐位一致。

## 5. 切片 D：U4 收尾（登记仓库范围未登记读取）

- 6 个 `PYPTO_X_*`（`AOCL_LIB`、`AVX2_LAYOUT_SHA256_MANIFEST`、`DRIVER_PATH`、`LIBXSMM_LIB`、`VENDOR_EVIDENCE`、`WORKTREE`）登记为 `internal`，复用既有 category（**schema 仍 v1**）。
- 仓库范围（`python/`+`tools/`+`scripts/`，验收方口径：1390 个 .py）从未登记 6 名 → **0 名**（33/33）；`affects_artifact_semantics=true` 全 6 行补筛：37 项 plan/capability/wrapper/artifact/输出/report 摘要 `all_equal=true`；默认执行 digest 三树相等。
- doctor 40→46 行（预期），affected 冻结断言仅为 registry 计数与新扫描测试；新守卫测试在父提交上失败并列出同样 6 名。

## 6. 验收

| 切片 | agent | 判决 |
|---|---|---|
| int8 exact-blocked | `7095f221` | **PASS_WITH_BOUNDARIES**：12/12 声明独立复现；336 例构造差分 + **1307 例 falsification**（M/N=1、tail、`kc·p=2²⁴−1` 的 117×119、8 seeds）零界内偏差；规则 8 collect 2105 / 2098 passed（991.90 s）。边界：残余假设（非形式化）、`prepare_int32` 显式 codes 不重复码域校验、Kc 无解/溢出合法域不可达仅注入。 |
| AVX2 position-native | `32165fae` | **PASS_WITH_BOUNDARIES**：oracle 逐字节未改 + 自建第二 oracle 184 checks 0 mismatch；覆盖 spy 5 native/0 host；双树 4 向 reader + ABI 对调 + 1-bit 篡改 fail-closed；39 raw buffer 零漂移；变异测试 18/64 红；全量 collect 2169 / 2162 passed / rc=0。边界：契约两处措辞不精确（已订正 0016：iota 抛 `CpuVectorUnsupportedError`；"dtype not in set" 分支不可达）。 |
| U4 follow-up | `9e0de13f` | **PASS_WITH_BOUNDARIES**：自写扫描器复现 46 行/33 名/0 未登记，父提交恰好 6 名；37 项 digest all_equal；doctor 不变式与 canary 零泄漏；守卫测试在父提交失败；全量 collect 2170 / 2163 passed / rc=0。唯一在范围内问题 = 0014 的 +10 归因（已订正，见 ERR-0014）。 |
| AVX-512 position-native | `97503c56` | **PASS_WITH_BOUNDARIES**（round-1 @`0b5b5899e`；round-2 复验在跑）：oracle 逐字节未改 + 自写 ctypes oracle 102/102 + `CpuScalarRuntime` 20/20；metadata/覆盖 87/87；payload/metadata 篡改 33 例与描述符篡改全 fail-closed；20 例语料两树 raw sha256 20/20 一致；focused 206 passed、全量 2179 passed（1313.31 s）。**唯一实质发现**：`-mavx512f` 隐式启用 AVX2，compare/where kernel 含 16 条 AVX2 VEX.256 指令而 `required_features`/feature mask 未声明 `avx2` → **已派 round-2 修复**（声明与运行期门补 `avx2`，编译 flag 不变以保数值/性能，0017 措辞订正）。 |

## 7. 证据路径

- int8：`_meta/pypto-x/int8-exact-blocked/{brief.zh-CN.md,raw,logs}`（含父方前置实验 `raw/parent_precheck/`）；验收 `_meta/pypto-x/verify-0047-int8/`。
- AVX2：`_meta/pypto-x/avx2-position-native/`；验收 `_meta/pypto-x/verify-0047-avx2/`。
- U4 follow-up：`_meta/pypto-x/u4-followup-registry/`；验收 `_meta/pypto-x/verify-0047-u4-followup/`。
- AVX-512：`_meta/pypto-x/avx512-position-native/`；验收 `_meta/pypto-x/verify-0047-avx512/`。
- 批级：`_meta/pypto-x/batch-0047-int8-exact-and-x86-position/{logs,raw,scripts}`（含 gate 两次全量、补丁重导与 `am` 复算）。
- 补丁：`patches/pypto-x/`（274）+ `patches/README.md`（HEAD `0b5b5899e`）。
- 契约文档：`docs/20-planning/0015`（int8）、`0016`（AVX2）、`0017`（AVX-512）；`0014` 已二次订正。

## 8. 登记的边界与口径

- **int8**：exactness 是**条件命题**（"pinned 闭源内核该路径只做 int32/f32"），不是形式化证明；撤回路径写在 envelope；fused/反量化误差不在范围；−128 与 K>133144 仍拒绝；`prepare_int32` 显式 codes 入参不重复码域校验（内部 API）。
- **x86 位置控制**：AVX2 的 f32 快路径要求连续输入；AVX-512 只依赖 F（DQ/BW/VL 未用）；两者 rank>16 与部分 scalar 输出走 host；float16/float64 三种算子在 lowering 阶段即拒绝（基线同）。
- **U4**：6 名仍 `os.environ` 直读（仅登记）；`consumer_kind` 出现自由文本 `evidence_path`（待加枚举）；AOCL 正向语义仅 zen4。
- **性能**：本批全部数字 **UNGATED**（12 vCPU KVM 无 cpufreq；AVX-512 未跑真实权重整网）。
- **勘误**：ERR-0014（0014 的 +10 collect 归因被独立验收逐提交分解证伪并订正）。
