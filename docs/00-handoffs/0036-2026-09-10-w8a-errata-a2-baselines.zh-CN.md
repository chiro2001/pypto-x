# PyPTO-X W8A 勘误收口 + Ascend A2 验收与 vLLM 基线快照

归档序号：`0036`

归档日期：2026-09-10（Asia/Shanghai）

状态：`W8A_BF16_WEIGHTED_ALIGNED_EN_ZH_CHAT_T18_NEAR_TIE_PASS_ASCEND_A2_ACCEPTANCE_PASS_W8H_W8I_VERIFIED`

本快照覆盖：ERR-0002（W8A prefill driver cos/sin 布局缺陷）修复与判据修订、W8A-C Ascend A2(910B3) 真机验收、
W8H vllm-ascend E2E 基线、W8I profiling 基线、W8G 可移植性清理收口、A2 卡占用协议落地。

---

## 1. 冻结点

```text
控制仓 / 公开主仓   chiro2001/pypto-x     HEAD 5fd3a11（本地；本快照与收口 commits 尚未 push）
实现主仓            upstream/pypto @ 34475e0d8（只读）
集成分支            port/pypto-x-integration @ dca302ef4（本地）
集成分支 base       9aae4649e
补丁集              patches/pypto-x/ 111 个（base 34475e0d8，183 文件，4.1M；本批重跑刷新）
integration 区间    W8G b3d1643c8 + c464927fa；A2 验收 0e5a51891；
                    W8H 9951a2fe7 + a65a37cc3；RoPE 修复 5f479d1a1；
                    W8I cc1cc7178 … dca302ef4
A2(910B3) 环境      1×910B3（64 GB HBM）/ CANN 9.0.0 / driver 25.2.0 / 256 vCPU / 2014 GiB
```

控制仓本批仍在 `main` 上本地 commit（docs errata / 快照与滚动文档 / configs / patches），**未 push**。

---

## 2. W8A：ERR-0002 修复与判据修订（核心）

**根因**：`python/tests/ut/pypto_x/qwen35_weighted_execution_driver.py::rope_bits_for_positions`
把 cos/sin 运行时输入按 position-major `[position][head][column]` 展平；图契约是
`(batch, heads, steps, rotary_dim)`（`python/pypto/portable/qwen35.py:1560-1561`，k 侧 `axis=1` 切 head），
扁平 buffer 必须 head-major `[head][position][column]`。T≠heads（T=5/8/18）时位置表错位。
**此前"chat(T=18) 真实数值累积分歧"结论作废**——不是模型/GDR/累加精度问题。

```text
修复 commit      d05e25734 → integration 5f479d1a1（driver-only + 聚焦测试；DRIVER_VERSION=2）
图契约零改动      (1,1,4096) digest 66dd4077… / 4,550 ops；(1,18,0) digest 001f32b7… / 13,748 ops
独立验收          verify/qwen35-t18-divergence-localization：PASS（证据见 §9）
未采纳           327b17158（--debug-f32-residual，改 qwen35.py）未合入、未采纳；
                 仅留作"未来若要求严格 18/18"的可选路线
```

**判据修订（2026-09-10 用户裁定，条件已验证）**：逐行 argmax 硬门槛 + near-tie 例外——
gold 该行 top-1/top-2 margin ≤ 0.5×band **且** ours argmax == gold top-2 token 时，记为 near-tie flip、
不判失败但必须单列计数。实测 chat row 14（0-based）：gold fp32 top1 271=19.956333160 /
top2 198=19.841325760，margin 0.115007=0.18813×band（阈值 0.5×band=0.305655），ours argmax=198=gold runner-up
→ 采纳后 chat = 17/17 有效行 + 1 near-tie = PASS。**严格口径原始结果 17/18 必须保留可见**。
其余 margin≤0.5×band 的行：chat row 3（matched）；en/zh 0 行。

替换数字（pre-fix 全部作废）：

| prompt | T | argmax（严格） | cosine | max_abs | band | 逐层 min cos | 判定 |
|---|---:|---|---:|---:|---:|---|---|
| en_continuation | 5 | 5/5 | 0.99997235 | 0.159756（0.68×band） | 0.235212 | 0.99993098（layer_12_output） | PASS |
| zh_continuation | 8 | 8/8 | 0.99993803 | 0.235296（0.85×band） | 0.277682 | 0.99993980（layer_21_output） | PASS |
| chat_zh_user | 18 | **17/18**（唯一 row 14 near-tie） | 0.99989976 | 0.416867（0.68×band） | 0.611310 | 0.99988217（layer_23_output） | 修订后 PASS |

```text
final_hidden vs gold   en 0.99994485/0.334759；zh 0.99992611/0.290360；chat 0.99984263/0.430467
decode token 链         en 11751→13→198→760→6511；zh 271→248068→271→248069→271；
                        chat 109266→6115→103724→1167→16451（token 链不变）
state                 全部有限；recurrent |state|max en 0.55–13.27 / zh 0.82–13.29 / chat 0.69–14.14
全量 ut               694 collected / 687 passed / 7 skipped（skip 全为 test_cuda_qwen_c2.py 无 CUDA 驱动环境项）
                      base 9aae4649e = 674 passed/7 skipped；新增 13 = portability 10 + 修复测试 3
```

**口径警示**：T=1/decode 在**算子层是 no-op**（单 position cos/sin 位序列逐位相同），但**端到端 decode
logits 会变**（decode 继承 prefill state；三 prompt 各自 51 个 prefill 输出中 43 个 sha256 改变，
最早为 `layer_03_new_k`）。**不得写成"decode 端到端无变化"**；旧 decode cos/max_abs 数字作废
（token 链仍有效）。详见 [`ERRATA.zh-CN.md`](ERRATA.zh-CN.md#err-0002)。

---

## 3. W8A-C：Ascend A2(910B3) 真机验收 —— **PASS**

实现 `work/qwen35-ascend-npu-acceptance @ aa43200ce` → integration `0e5a51891`；独立验收
`verify/qwen35-ascend-npu-acceptance`：**PASS**（discrepancy D1 为记账口径：`9aae4649e..0e5a51891`
区间含前置 b3d1643c8/c464927fa，待验 commit 自身仅 3 文件 +1059/-0，blob 与实现分支逐一相同）。

```text
A  PTO-ISA NPU ST     run_st.py -r npu -v a3 -t tassign -g TASSIGNTest.case1 → exit 0 / 1 test PASSED
                      （910B3，bisheng clang 15.0.5，fatobj + ACL 真机执行）
B  CANN 自带样例        mspti samples/callback_domain（aclnn Add）；样例硬编码 deviceId=4 → 107001，
                      仅改 deviceId=0 后 cmake+make+执行 exit 0（失败日志亦保留）
C  adapter 真 hook     注入式 CASK hook：真实 bisheng 编译 22,728 B / sha256 4d7f704c…；
                      live driver 9/9；真机 launch max_abs_err=0.0（验收方自构造期望 n_mismatch=0）；
                      负路径（未 lower/格式/ABI/未 load/异构 target）全部 fail-closed
独立验收               上述关键数字逐项复现；adapter UT 本机 7 passed、A2 7 passed；零漂移/脱敏 PASS
```

**known_limits 变更**（用户 2026-09-10 裁定，独立验收建议）：

```text
关闭（限定式）  ascend_cann_bisheng_npu_regression_blocked
  覆盖范围仅：① PTO-ISA 668248ec 的 tassign 单用例 NPU ST；
             ② python/tests/ut/pypto_x/ 内注入式最小 hook 的固定 64×64 f32 add（bisheng + 910B3 真机）
  明确不代表：Core IR→PTO、pypto classic/Pro JIT/OPC 链路、模型级或全量 ST 回归
新增            ascend_hook_is_test_tree_injected_minimal_f32_add_only
                ascend_stable_cann_v9_2_0_beta2_not_device_validated
                pypto_classic_pro_jit_opc_gym_not_ascend_validated
保持开启        real_native_cpp_ir_bridge_not_connected
```

`configs/upstream_lock.yaml` stable 验证状态：`driver_version = COVERED`（仅 A2 实测 driver 25.2.0，
不可外推完整矩阵）；`cann_toolkit_version = PARTIAL`（A2=9.0.0 真机 vs stable 目标 9.2.0-beta.2 未上卡）；
classic / Pro JIT / OPC / gym = NOT COVERED。

**边界**：仅 tassign 单用例、仅固定 64×64 f32 add；非 Core IR→PTO 桥、非模型级；无性能结论。

---

## 4. W8H：A2 vllm-ascend E2E 基线（实现 PASS + 独立验收 PASS）

实现 `work/qwen35-a2-vllm-ascend-e2e-baseline @ b4a3a7097`，integration `9951a2fe7 + a65a37cc3`。

```text
精度（PASS）  三 prompt 前 4 个生成 token 命中 gold；prompt token id 与 gold 完全一致；
              prefill 逐位置 argmax：en 4/4、zh 7/7、chat 17/17；decode 逐步 argmax 4/4；
              prefill 末行 top-5 集合与 gold 相同（vLLM 只给 top-5，无完整 logits → 无 cosine/max_abs）
性能（原始事实）eager：TTFT 0.202–0.204 s / TPOT 66.1–66.8 ms / decode 14.97–15.13 tok/s
              ACL graph：TTFT 0.163–0.166 s / TPOT 12.2–12.5 ms / decode 80.1–82.1 tok/s
              （协议仍为提案；batch=1、短 prompt、配置敏感，不作为绝对门槛）
环境          vllm 0.21.0 / vllm-ascend 0.21.0rc1 / CANN 9.0.0 / driver 25.2.0 / torch_npu 2.10.0 / 910B3
权重          ModelScope 官方源下载，逐文件 sha256 与冻结 manifest 全 match（含 1,746,942,600 B 主权重）
限制          vLLM 不暴露完整 logits；ACL graph 口径未做精度复跑；无并发 sweep / 长 prompt
```

**独立验收（PASS）**：任务 `verify-qwen35-a2-vllm-ascend-baselines`（agent `6418d00a`）在冻结点
`dca302ef4` 复跑并本机重算：prompt ids / 前 4 token / prefill argmax（en 4/4、zh 7/7、chat 17/17）/
decode argmax 4/4 全部一致，bf16 与 fp32 双 gold 判定相同，零漂移；性能重算与原声称一致
（eager 0.2016–0.2039 s / 66.1–66.8 ms；ACL graph 0.1625–0.1664 s / 12.19–12.48 ms），
A2 spot-check（en，R=6）eager 0.2048 s/67.9 ms、graph 0.1693 s/13.20 ms（graph TPOT ≈5.15× 优于 eager，
TTFT −17.4%），绝对值偏高 +2%~+6% 属单 prompt/宿主方差、**非阻断**。性能协议仍是提案、数字为原始事实。

---

## 5. W8I：A2 vllm-ascend profiling 基线（实现 PASS + 独立验收 PASS）

实现 `work/qwen35-a2-vllm-ascend-profiling-baseline`（`4f764edac … fb273fbf6/209e011c2`），
integration `cc1cc7178 … dca302ef4`；base `0e5a51891`。

```text
API 结论   vLLM 0.21 仅支持 --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=<abs>；
           旧 VLLM_TORCH_PROFILER_DIR 已移除 → harness 版本自适应，无回退重试
profiler 开销  ×1.65–1.71（0.8B/9B、offline/online）；这是真实开销测量，不得当作模型性能引用
              独立复核：6 个交付比值 ×1.649/×1.665/×1.667/×1.665/×1.714/×1.654；
              A2 0.8B offline 两轮 6 比值 median ×1.665、范围 [1.621,1.716]
Ascend 侧  9B MatMulV2（total_us 口径）offline 75.306% / online 75.730%；0.8B offline 33.978% / online 35.754%；
              op_statistic 自带 ratio 与 A2 源 CSV 直接解析三方一致
回传        26 CSV（2,056,724 B，sha256 全等、errors=0）；manifest 107 files / 6,431,774 B 逐 sha256 全等
              （manifest 自身不计入 107，含自身为 108 files / 6,458,618 B）
大 trace    4 目录文件数 139/161/268/332 全对、合计 ≈8.02 GiB（实现方记录 8.03）留在 A2；bytes 正漂移 17–30 KB
时钟        A2 `date -u` 比真实 UTC 快约 8 小时，跨机对齐注意
```

**独立验收（PASS）**：同一任务在 A2 复跑并复核：`ProfilerConfig`/`--profiler-config` 路径成立、
`VLLM_TORCH_PROFILER_DIR` 不存在、`api_used=config` 且 `attempts=[]`；9B 4 分片 sha256/字节全对
（19,329,393,248 B）；本地 manifest 与 26 CSV 逐 sha256 全等。非阻断 discrepancy 一并登记：
manifest 计数不含自身、trace bytes 正漂移 17–30 KB、bundle tar 已删导致 md5 不可重算（改用 sha256/摘要）、
torch_npu profiler `Incorrect schedule … RECORD` 告警（实现方与复跑均有、数据完整、判上游）、
验收 fresh profiled 与实现方 0.8B offline 的 MatMulV2 占比差异（35.12% vs 33.98%，负载不同属预期）。

---

## 6. A2 卡占用协议（2026-09-10 部署）

```text
位置         A2 容器固定目录 /root/a2-npu-lock/（README.zh-CN.md + a2_card_lock.sh）
流程         request → using → done；历史记录保留（.request/.using/.done 文件）
保护范围     **只保护 NPU 执行**；下载权重、编译、环境准备可并行进行（不需要持卡锁）
等待         抢锁失败时 180 s 轮询重试
防僵死       TTL 6 h + PID 存活判定（owner 进程消失可回收）
时钟         A2 时钟比真实 UTC 快约 8 h（按 CST 标 UTC）
入仓纪律     仓内只登记协议与目录名；端点、跳板/凭据、脚本实体只在本地私有侧，不得提交
```

**运行验证（独立验收实际使用）**：`verify-qwen35-a2-vllm-ascend-baselines` 全程用
`a2_card_lock.sh run/acquire` 包装 NPU 命令，结束态 `.using` 已释放为 `.done`、无残留 vllm 进程、
HBM 回落 3441/65536 MiB，自建临时目录已删除——协议已通过真实运行验证。
后续所有 A2 NPU 任务必须遵守。

---

## 7. W8G/W7：可移植性清理收口

```text
W8G qwen35-portability-cleanup   work @ 9749617e0（8dfec148d + 9749617e0）
                                 integration b3d1643c8 + c464927fa；独立验收 PASS（零漂移、无 discrepancy）
                                 参数化 qwen35_reference/*、evidence 模块、tools/cann 安装脚本；
                                 聚焦测试 10/10；全量新增 13 例已计入 ERR-0002 的 694 口径
W7 合并顺序约束                  将来 work/cann-toolkit-local-install 入 integration 时，
                                 丢弃同名未参数化 tools/cann/install_cann_toolkit_local.sh，
                                 保留 W8G 参数化版本（参数化版本同时含 --print-config 与 PYPTO_X_* 覆盖）
W7 README 缺口                   其 tools/cann/README.zh-CN.md 待 W7 自身合入时补一句
                                 "路径均可用 PYPTO_X_* / --* 覆盖"；本批不改 upstream 分支，仅登记
```

---

## 8. 未解决与下一步

```text
1) W8H/W8I 独立验收已完成 PASS（verify-qwen35-a2-vllm-ascend-baselines，agent 6418d00a）→ 无需返工；
   6 条非阻断 discrepancy 已登记（见 §4/§5）
2) 若未来要求 chat 严格 18/18：可选 327b17158（--debug-f32-residual，未合入、未采纳）
3) decode 第 4 步无 gold 参考行；decode 逐步 logits 数字须引用修复后 raw dump
4) W8A-C 限定式关闭后的真实缺口：Core IR→PTO 桥（W8E/E4）、stable CANN 9.2.0-beta.2 未上卡
5) W8B 硬化（AVX2 packed / SVE fallback 分类清零 / CUDA cuEvent + GEMM 基线 / 本机 L0–L1）
6) W8C W8A8-linear 实现未开始；W8D fused WY / 长序列 T=64/128；W8E CANN report（plotly）等
7) AMD 静态 compiler 未对 v3 图重跑；性能协议仍是提案；本机 G4 永久 UNGATED
```

---

## 9. 证据路径

```text
ERR-0002 验收   ../worktrees/_meta/pypto-x/verify-qwen35-t18-divergence-localization/
ERR-0002 实现   ../worktrees/_meta/pypto-x/qwen35-t18-divergence-localization/
W8A-C 验收      ../worktrees/_meta/pypto-x/verify-qwen35-ascend-npu-acceptance/
W8A-C 实现      ../worktrees/_meta/pypto-x/qwen35-ascend-npu-acceptance/
W8H 基线        ../worktrees/_meta/pypto-x/qwen35-a2-vllm-ascend-e2e-baseline/
W8I 基线        ../worktrees/_meta/pypto-x/qwen35-a2-vllm-ascend-profiling-baseline/
W8H/W8I 验收    ../worktrees/_meta/pypto-x/verify-qwen35-a2-vllm-ascend-baselines/（PASS：summary/brief/validation）
W8G 验收        ../worktrees/_meta/pypto-x/verify-qwen35-portability-cleanup/
本批收口证据    ../worktrees/_meta/pypto-x/freeze-w8a-errata-0036/（改动清单、自检日志、brief）
```
