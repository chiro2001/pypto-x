# PyPTO-X 0044 波次快照：report schema v4（内嵌 plan payload）与 SVE256 W8A8 memo 盘上 ELF 完整性

状态：`CLOSED_LOCALLY_VERIFICATION_IN_FLIGHT`（两条独立验收在跑；回来后按既定政策自动推送）
批次：`batch_0044_plan_payload_and_memo_integrity_2026_09_13`（见 `configs/development_lock.yaml`）
撰写：parent（自动批次）

## 0. 一句话

两条互不相关但都属"收尾欠账"的切片：**A** 把 `ExecutionReport` 升级到 schema v4——内嵌 plan payload（`resolution.plan_digest` 的 preimage），使 0043 验收残留的"`plan_digest` + decision body 协调改写"彻底闭合；**B** 把 0041 登记的边界"memo 命中不复查盘上 ELF"用每次命中的 re-hash（~0.43 ms）收成 fail-closed。

## 1. 冻结点

| 项 | 值 |
|---|---|
| integration tip（frozen 候选） | **`dcc5371cb`**（tree `42e4477ad5514395c4cac1e6a31cca5b9f62a133`） |
| 任务分支 commit | A：`e1c480be2` → cherry-pick `581510882`（T11）；B：`932980c2f` → cherry-pick `dcc5371cb`（T12，patch-id `fd91a2379…` 一致，因 B 基于 `0a63ed4dd` 而 T11 在其上，故用 patch-id 判等价） |
| 补丁数 | **256**（0043 = 254）；am 复算：256 个全部干净应用，复算 tree = 冻结 tree |
| 规则 8 全量 | **rc=0，1932 passed / 7 skipped / 0 failed，1066.72 s**（attempt 1）；collect **1939**；基线 `0a63ed4dd` = **1921**（runner 的基线目录路径未对上预置目录，父方手工补跑基线 collect 到 `raw/collect-0a63ed4dd.out`，rc=0） |

## 2. 切片 A：report schema v4（`REPORT_SCHEMA_VERSION = 4`）

- 新增**必需**顶层块 `plan_payload = {digest, document}`，`document` = plan 的 canonical payload（`resolution.plan_digest` 的 preimage），`digest = "sha256:"+sha256_text(canonical_json(document))`。
- 强制三处互等：`resolution.plan_digest == plan_payload.digest == 由 body 重算值`（`report_plan_digest_not_reconciled` / `report_plan_payload_digest_mismatch`）。
- `resolution_decision.document` 必须是 `plan_payload.document` 的**确定性投影**（新 leaf 模块 `python/pypto/execution/decision.py`，entry 与 validator 共享同一投影；不符 `report_resolution_decision_not_reconciled`，payload 不可投影 `report_plan_payload_derivation_failed`）。
- 版本 **3→4**：只收 4；v3（含 3 份冻结 verify-0043 真报告）/v2 及更旧 → `report_schema_version_legacy_rejected`；>4 → `report_schema_version_mismatch`；无静默重解释。
- **闭合**：0043 验收残余 3（`plan_digest` + decision body + 重算 decision digest）与 round-2 R5-full（runtime-fallback decision 改写）**现已拒绝**；R1/R2/R3 保持闭合、R4/R5 局部保持拒绝、0042/1c 检查与注册身份锚全保留、仍无 probe/不重跑 resolver。
- **新登记残余**：**3'** 完全协调的 alternate plan payload（payload body+digest+`plan_digest`+重推 decision+digest+全部报告副本）仍接受——内嵌 payload 无外部锚；**4** 整份报告替换为另一注册 provider 的自洽故事（无签名/attestation，需另立任务）。

## 3. 切片 B：memo 命中盘上 ELF 完整性

- 新增 `_w8a8_memo_artifact_intact()`：**每次 memo 命中**用完整 decode 同款强化读取器 `_read_native_bytes`（`O_NOFOLLOW`、单 link 常规文件）重读 ELF 并 sha256，对比 decode 时捕获的 `content_digest`；不一致/读取失败 → 丢弃条目 → 完整 decode → 规范 fail-closed 错误；原字节恢复则自动恢复服务。每次 dispatch 的声明门保持不变。
- 选 **re-hash** 而非 stat 指纹：mtime 可被 `os.utime` 回拨、同尺寸改写可落在时间戳粒度内；883,064 B re-hash 仅 ~0.43 ms。
- 同窗口数字（本机 QEMU，UNGATED）：miss 2.644 s｜**hit（含 re-hash）0.642 ms**｜hit（旧行为）12 µs｜纯 read+sha256 **0.431 ms**｜hit/miss **4,119×**；整网 QEMU smoke（4911 ops）**PASS，72.566 s**（加固前 72.494 s，+0.1%），0 int8 transpose、187 qmatmul native。
- 测试：新增 8 例（同尺寸原地改写、append、truncate、替换为另一合法 static AArch64 ELF、mtime 回拨、同内容新 inode 接受、symlink 拒、hardlink 拒）；原 6 条纪律测试未削弱；`test_w8a8_sve256.py` **54 passed**。
- **登记边界**：check→exec TOCTOU（需 fd 持有式执行 `execveat`/O_TMPFILE，属 runner/协议改动）；root/块设备/page cache 级攻击超出 userspace 读检查；SHA-256 碰撞不可行；字节逐位相同的新 inode 按设计接受。**未声称完全闭合**。以上均为本机 QEMU，A3 原生复测仍并入 M2b item 6。

## 4. 验收（独立，均在 `dcc5371cb`）

| 切片 | agent | 判决 |
|---|---|---|
| report v4 | `0eb1b90a` | 待回（重点：版本语义、payload digest 三处互等、decision 投影、无 probe、残余 3'/4 与清单完备性、全量 UT 独立复跑） |
| memo 完整性 | `a9a98cd6` | 待回（重点：七类改写是否全部检出并 fail-closed、纪律测试未削弱、命中成本、TOCTOU 边界是否已登记、全量 UT） |

父方复核（T12）：collect **1939**；`test_w8a8_sve256.py` **54 passed**；`test_execution_report_manifest_v31.py` **70 passed**；合并探针 **38/38 全拒（0 接受）**；无误伤（portable + 真 AOCL f32/bf16 + 真 runtime fallback）。

## 5. 仍为边界（汇总，逐条有最小反例）

1. 报告侧运行态/实测事实（`timing.*`、`content_digest`、`threads_used` 值、coverage 计数、`gates.*`）。
2. 未 pin 的 probe 事实（host features、target snapshot digest、AOCL fingerprint 路径/大小/线程数、线程模型数值）。
3. 报告残余 3'（完全协调的 alternate payload）与残余 4（无 attestation 的整份替换）。
4. 无外部真源字段（`request.dtype/shape`）的全副本改写；说明层。
5. memo 完整性：check→exec TOCTOU；root/块设备/page cache 级；SHA-256 碰撞。
6. AVX2 broadcast 为 scalar odometer（不称 SIMD 吞吐）；int8 包络为观测快照非上界；性能一律 UNGATED。

## 6. 证据路径

- A：`_meta/pypto-x/report-schema-v2/{brief.zh-CN.md,raw/audit_report_schema_v31.json,logs/*-v31.log,scripts/}`；独立验收 `_meta/pypto-x/verify-0044-report-v4/`。
- B：`_meta/pypto-x/sve256-w8a8-op-throughput/{brief-0044b-memo-integrity.zh-CN.md,raw/diag_after_integrity.json,logs/pytest_*}`；独立验收 `_meta/pypto-x/verify-0044-memo-integrity/`。
- 批级全量：`_meta/pypto-x/batch-0044-plan-payload-memo-hardening/{raw,logs}`（`full-pytest.out`/`full-pytest.rc`/`collect-tip.out`/`collect-0a63ed4dd.out`）。
- 补丁：`patches/pypto-x/`（256）+ `patches/README.md`（HEAD `dcc5371cb`）。
