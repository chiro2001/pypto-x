# PyPTO-X 0041 波次快照：AVX2 layout 原生化、W8A8 契约 v2（packed [K,out]）、SVE256 唯一性能瓶颈、U 线 vendor provider 与 view_mode 提级

归档序号：`0041`

归档日期：2026-09-12（Asia/Shanghai；0040 推送后至本批收口）

状态：`DRAFT_PENDING_FREEZE`（收口时把"冻结点"一节补成最终值；正文事实均已在 `configs/development_lock.yaml` 与各任务证据目录登记）

本快照覆盖：AVX2 layout 原生化（B6 的 AVX2 对应件）｜W8A8 契约 v2 的 M2a 实现与 M2b 重基线（含 **Q5 SVE256 整网 L4 判定**）｜**SVE256 W8A8 每节点 O(整图) 开销修复**（A3 6 配置从外推 11.8 h 降到 39.3 min）｜U 线 **U2a**（第一个 vendor provider：AOCL bf16/f32）与 **`view_mode=require` 提级**（含独立验收抓到的 plan/report 伪造阻断）｜C7 下钻工具补齐。

---

## 1. 冻结点（**收口时填写**）

```text
upstream base        34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
integration base     195ace3eb8a521950c72ffc62f0349aed6c7ac04（0039 交接点）
integration head     TBD_AT_FREEZE（本批节点：35100030f 吞吐修复 → 3dfac2bbe U2a → fe8e6fd17 M2b 工具 → 36fa3fd3f view-mode → 修复后最终）
patches              TBD_AT_FREEZE（0040 收口时为 217）
控制仓 main          TBD_AT_FREEZE（本批提交，未推送）
```

## 2. AVX2 layout 原生化（`avx2-layout-native`，W8B 续作）

- 实现 `4e52610f3` + 证据加固 `477faa357`（135 例 / 116 次逐位差分对自写 stdlib oracle；PXLD wire 与 AVX-512 逐字节相同；fail-closed；payload 4→5、ABI `:5`→`:6`）。
- **独立验收 PASS_WITH_BOUNDARIES（r2）**：验收方另写 163 次差分 0 失配、全 184 descriptor 位置穷举（180 拒 / 4 reserved 良性）、独立 wire 9/9、sha256 manifest 双方逐键 0 diff、全量 1528/7/0。
- 性能（UNGATED）：decode launch 28.5→20.4 s、prefill 127.9→109.3 s；per-op layout 中位降到 0.02–0.04 ms/call；剩余 host_reference 仅 `broadcast`（~39 s/494 calls）与 `where/compare/iota`（<1 s）。
- 过程边界（已登记）：实现方曾有 5 段未持锁运行（含聚焦与一次 after 载荷误执行），全部作废重测；测试改名并扩到全 184 位置；新增 AVX2↔AVX-512 wire 身份回归；rc=143 定位为自身前台 `sleep` 被工具清理（walter 现用 `setsid`）。

## 3. W8A8 契约 v2（M2a）与 M2b 重基线 + Q5 SVE256 整网 L4

- **M2a（`fca4a6d9b`）**：每 forward int8 权重 transpose **150/186/151/187 → 0**；region 全表对拍（数量/顺序/storage_id/nbytes/offset 全等，仅 int8 shape 精确翻转）；packed `[K,out]` 与 v1 transpose 路径**逐位一致**（scalar/vector/AVX2/AVX-512/SVE256-QEMU + CUDA fake driver）；四后端 kernel/lowering **0 行改动**；`BINDING_SCHEMA_VERSION`/`PACKED_LAYOUT_VERSION` 2→3、`storage_layout=k_out_rowmajor`、`WeightLayout.layout_version` 1→2、**payload 不升**；旧 schema/payload/digest 全 fail-closed。
- **M2a 独立验收 PASS_WITH_BOUNDARIES**：自建 region 对拍、转置口径两路复算、真实层 3584×1024 五执行面 + lm_head 248320×1024 + CUDA fake + packer 差分全等于 numpy int32 golden；聚焦 468 / 全量 1442 passed 0 failed；仅两处文档数字勘误（已改）。
- **M2b 重基线**：AVX-512 六段 615 s、AVX2 六段 2810 s；**L4 prefill 4/4 PASS**、decode 扩展与 Q3 v1 同模式；**v1↔v2 逐位**：AVX-512 default/full 各 15 文件 765 arrays **0 mismatch**（default combined digest 与本机 v1-vs-v1 自比对相同）。
- **Q5 SVE256 整网 L4（A3 原生）**：6/6 配置 digest/op 数与静态 census 全对拍、每 forward int8 transpose=0、qmatmul 全 native、`emulated=false/vl_bytes=32`；**prefill 主判定 6/6 PASS**；两批总墙钟 **2356 s（39.3 min）**；decode 扩展 default FAIL/FAIL、full FAIL（decode band）/PASS（prefill band），与 x86 同模式。诚实边界：default/en 与 full/zh 的 decode 链在 prefill-next-token 行因 near-tie 转向（`common_prefix_match=False`，主判定仍 PASS）。

## 4. SVE256 W8A8 每节点 O(整图) 开销修复（`sve256-w8a8-op-throughput`）

- **根因两层**（与元素数无关）：① `_load_w8a8` 每节点 `_decode_for_execution`（整图 from_dict+lower+digest+sha256(ELF)，每节点 1–2 次）；② `_require_w8a8_declaration` 每 dispatch 深 thaw 全量 metadata（535,957 次递归、~0.9 s/节点）。
- **修复**：identity-keyed 有界 memo（容量 2/FIFO、`PYPTO_X_SVE_W8A8_DECODE_MEMO` 可关）+ 声明门 O(1)；wire/runner/ELF 协议与数值语义不变。
- **效果**：小 op 2.5–2.7 s → 18–22 ms（114–151×）；A3 真机 W8A8 算子从 13.59/6.75/13.51 s → **7.12/16.08/32.05 ms**（T=5）；整网 6 配置从外推 11.8 h → **39.3 min**。
- **独立验收 PASS_WITH_BOUNDARIES（52/52 机检）**：自建计数探针（node decode 2/1/2→0、thaw 535,957→0）、位级 25 例 identical、memo 纪律（命中 19 µs vs miss 2.655 s ≈1.375e5×、篡改必 miss、跨进程必重 decode、线程压测）、聚焦 474、全量 **1583 passed / 7 skipped / 0 failed**；修正了实现方对 lm_head 的外推（实测 qmatmul 44.27 s / epilogue 5.06 s → shape-aware 两批 1.03–1.25 h）。边界：memo 命中不复查盘上 ELF 摘要（可选加固 stat/重算 ~1 ms）；AVX2 无 per-op census。

## 5. U 线：U2a（第一个 vendor provider）与 `view_mode=require` 提级

- **U2a（`3dfac2bbe`）**：AOCL/LPGEMM bf16+f32 绑进 U1 执行面（probe/manifest/解析/降级语义/report schema v2）；数值仅 (5,1024,32) 差 bf16 5.96e-8 / f32 8.20e-8（保持 `deterministic_bounded`）；D1/F5 两条 U1 follow-up 一并修；**Q8 同协议 12/12 ≤2×、geomean 0.970×**（首轮 3.13× 因线程路径，方案=seed `BLIS_NUM_THREADS` + plan 冻结 `thread_mode`）；全量 UT 1609 passed 0 failed；逐调用 repack 口径 3.32× 单列为诊断（属 ingestion 预 pack 后续项）。
- **`view_mode=require` 提级**（用户 2026-09-12 批准）：五条可机检证明条件（contract view obligation / stride-shape 可表达 / 写后读 / owner-lifetime / 多 view 重叠）+ 三态语义 + AVX-512 `proof_gated` 零拷贝（与 copy 路径逐位一致）+ SVE256/AVX2 显式 `alias_proof=unsupported` + plan/report 字段 + 对抗用例。
- **独立验收抓到 3 条阻断反例**（`36fa3fd3f`）：report 层零校验；plan 层"重算 alias digest + plan_digest 的自洽伪造"被 `ExecutionPlan.from_dict` 接受并成功执行；`recompute_alias_proof_digest` 无调用路径。**数据面未被绕过**（AVX-512 执行器每次从编译 plan+SSA 重新 prove），缺口在**公开校验面**。修复中（(a)–(e) 五条校验 + 回归用例），验收方停手等修，0006 的 **provisional 标注保留**。

## 6. C7 下钻工具补齐

- `tools/diag_c7/` 的 drill-down 系列（4 个配套脚本 + `compare_step_intermediates.py`）在 cherry-pick 时只带了最后一条 → 已补齐并验证**与分支 tip 逐字节一致**（`git diff 0d98a7f28 HEAD -- tools/diag_c7/` 为空）。

## 7. 验收与规则 8

- **批次级 canonical 全量**：`36fa3fd3f` 上的一次因 harness 超时 TERM 于 82% **作废**；修复后将在**最终 tip** 上重跑一次，作为 U2a / view-mode / M2b 三方共同引用的规则 8 证据（`batch-0041-rule8`，waiter 已改 `setsid`）。
- 三个任务级独立验收：AVX2（PASS_WITH_BOUNDARIES r2）、吞吐修复（PASS_WITH_BOUNDARIES 52/52）、M2a（PASS_WITH_BOUNDARIES）、U2a（in flight）、view-mode（FAIL→修复中）。

## 8. 在途与排队（收口时更新）

`verify-u2a-vendor-provider`（跑中）｜view-mode 实现方回修 → 验收方复验｜批次规则 8 全量（等新 tip）｜W8A8 v2 M2b 待命（除共享全量外清单全清）｜`view_mode` provisional 待办。

## 9. 待用户决策

L4 decode 口径（决策包 `0011`，现已有 x86+SVE256 三平台同模式证据，推荐 A+B）｜B4 阈值冻结｜L5/L6 语料与阈值｜`view_mode` 已裁决（提级中）｜新量化方案｜W8J 注入门政策｜A3 chip7/NPU 放行。

## 10. 证据路径

```text
AVX2 layout            ../worktrees/_meta/pypto-x/avx2-layout-native（验收 verify-avx2-layout-native）
W8A8 v2 M2a/M2b        ../worktrees/_meta/pypto-x/w8a8-v2-m2（+ /m2b）
Q5 SVE256 L4           ../worktrees/_meta/pypto-x/c8-l4-sve256（+ m2b/a3）
吞吐修复               ../worktrees/_meta/pypto-x/sve256-w8a8-op-throughput（验收 verify-sve256-w8a8-op-throughput）
U2a                    ../worktrees/_meta/pypto-x/u2-vendor-provider-aocl（验收 verify-u2a-vendor-provider）
view_mode 提级         ../worktrees/_meta/pypto-x/view-mode-require（验收 verify-view-mode-require）
批次规则 8             ../worktrees/_meta/pypto-x/batch-0041-rule8
C7 下钻                ../worktrees/_meta/pypto-x/c7-decode-gap-localization
```
