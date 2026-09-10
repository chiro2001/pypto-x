# PyPTO-X W8A BF16 带权对齐完成 + 仓库迁移快照

归档序号：`0035`

归档日期：2026-09-10（Asia/Shanghai）

状态：`W8A_BF16_WEIGHTED_ALIGNED_EN_ZH_PASS_CHAT_T18_DIVERGENT_ASCEND_A2_ONLINE`

本快照覆盖：真权重整网对齐、GDR decay 勘误修复与独立验收、判据口径裁定、控制仓公开化迁移、A2(910B) 真机接入。

---

## 1. 冻结点

```text
控制仓 / 公开主仓    chiro2001/pypto-x         HEAD 832c38a（本地工作副本 = 该仓克隆）
私有归档             chiro2001/pypto-x-private  本地备份 ../pypto_x_private_bkp（archive=6758b76）
实现主仓             upstream/pypto @ 34475e0d8（只读）
集成分支             port/pypto-x-integration @ 9aae4649e
补丁集               patches/pypto-x/ 96 个（base 34475e0d8）
```

## 2. 真权重 BF16 对齐结果

判据（2026-09-10 用户裁定，**相对 gold 自身 fp32↔bf16 噪声底**）：

```text
band = gold 自身 fp32↔bf16 的 max_abs 与 cosine，随 prompt 变化
判据：逐行 argmax 一致（硬门槛）；max|Δlogits| ≤ 2×band；cosine ≥ band_cosine − 1e-4
```

| prompt | T | ops | argmax | cosine | max_abs | band | 逐层 min cos | decode | 判定 |
|---|---:|---:|---|---:|---:|---:|---|---|---|
| en_continuation | 5 | 6,728 | 5/5 | 0.99991364 | 0.233829 | 0.235212 | 0.99955 | 3/3 | **PASS** |
| zh_continuation | 8 | 8,348 | 8/8 | 0.99973494 | 0.463474（1.67×band） | 0.277682 | 0.99945 | 3/3 | **PASS** |
| chat_zh_user | 18 | 13,748 | 18/18 | 0.99799387 | 2.310620（3.78×band） | 0.611310 | 0.99790 | 3/3 | **FAIL（真实累积分歧）** |

- decode token 链三 prompt 全对：`11751→13→198→760→6511`、`271→248068→271→248069→271`、`109266→6115→103724→1167→16451`；
- 48 个 state 输出全部有限，recurrent `|state|max` 12.4–14.2（ERR-0001 的指数爆炸未复现）；
- chat 的分歧：**row 11 起误差急剧放大**，首个不达标层 = gold index 8（`layer_07_output`，0.99842），但 argmax/decode/state 全对 → 非结构错误；
- 算子数实测 `ops(T) = 6,728 + 540×(T−5)`（取代此前 378/步的估算）；
- 资源：T=5 prefill 132 s / T=8 377 s / T=18 932 s，峰值 RSS 3.8–4.1 GiB，单锁内 8 段重活合计约 46 min。

## 3. GDR decay 勘误（ERR-0001）与独立验收

- 根因：图 v1/v2 `_append_gdr_graph` 的 decay 门写成 `-A_log ⊙ softplus(a+dt_bias)`，官方是 `-exp(A_log) ⊙ softplus(...)`（`modeling_qwen3_5.py:619` + `:537` `A_log=log(A)`）；
- 后果：decay>1 使 FP32 state 指数爆炸（0.60→2.8e4→2.36e16），被 gated RMSNorm 掩盖，整网 logits 全错；
- 修复：`11eceeb4a` → 图契约 **v3**，graph_digest `c1a5663b…`→`66dd4077…`，ops 4,532→4,550（(1,1,4096)）；
- 独立验收 `verify-w8a-decay-fix`：**PASS_WITH_BOUNDARIES**（结构/官方语义/S2/S3/全量 pytest 674 passed/7 skipped/liveness A/B 逐位一致/packed 内核抽样逐位一致/证据审计），并纠正了 7 处表述（exp 全图 114、参数 320 而非 371、`.so` 逐字节相同等）；
- 勘误总表：`docs/00-handoffs/ERRATA.zh-CN.md`（含就地 ERRATUM 文件与滚动文档更正）。

## 4. 控制仓公开化与仓库迁移

```text
公开主仓    https://github.com/chiro2001/pypto-x         （Apache-2.0 + NOTICE；含 patches/）
私有归档    https://github.com/chiro2001/pypto-x-private  （完整私有历史）
本地备份    ../pypto_x_private_bkp  main 跟随公开 / archive 保留私有历史与第三方离线副本
```

- 工作副本已迁到公开仓克隆（`origin` = 公开仓），日常开发直接 push 公开仓；
- **两次历史清理**（force-push）：① 剔除 `references/` 第三方讲稿 PDF/HTML/TXT；② 剔除自建中继 IP/域名/服务名与 A2 访问脚本；
- **A2 访问脚本与端点永不入仓**：只保存在本地私有侧（`~/tools/a2-910b/` + `~/.ssh/a2-910b.env`，600 权限），`.gitignore` 兜底；
- 补丁集：`patches/pypto-x/`（96 个，base `34475e0d8`）+ `patches/README.md`（应用方法、验证命令、许可边界）；`export_patches.sh` 是冻结流水线的一步；
- 发布纪律：推公开仓属发布动作需用户明确指示；不得上传上游源码、权重、证据。

## 5. 落地的基础设施

| 资源 | 状态 |
|---|---|
| GamePC RTX 5080 | CUDA Toolkit（nvcc 13.3.73 + cuBLAS 13.6），Driver/PTX 回归 PASS |
| 本机 CANN 9.2.0-beta.2 | 仅 toolkit（无驱动/无 950-ops）；`cannsim`/`npusim` 可用；CA-model 最小用例跑通（审计 0007） |
| PTO-ISA CPU_SIM | 125/125 PASS（审计 0006），trace 插桩仅部分覆盖 |
| **昇腾 A2（910B3）** | **在线**：容器 256 vCPU / 2 TB 内存 / 1×910B3（64 GB HBM）/ CANN 9.0.0 / 驱动 25.2.0；经自建中继反向隧道稳定接入（脚本在本地私有侧） |
| 鲲鹏 920B ECS | 在线（2 vCPU / 2.5 GiB / SVE=1 VL=32），约定串行 |

## 6. 未解决与下一步

```text
1) chat T=18 数值累积分歧（3.78×band）→ 单列精度任务：逐 op bf16 舍入链 vs 参考 fp32 累加、
   分歧起点（row 11 / layer_07 之后）、是否需在 attention/GDR 累加点保持 fp32
2) decode 第 4 步无 gold 参考行；gold bf16 无 hidden_states_layers
3) AVX2 packed 参数级算子未做；SVE 仍有 iota/compare/broadcast/where 的 host fallback
4) GDR fused WY kernel 未做（现为顺序参考）
5) W8A8-linear 实现未开始（契约已冻结、D1–D12 已批准）
6) CANN：report 缺 plotly；npusim record 复现性未解决；IR→PTO 桥与 Ascend hooks 缺失
7) AMD 静态 compiler 未对 v3 图重跑
8) 性能协议仍是提案；本机 G4 永久 UNGATED；CUDA 无 cuEvent 计时前 kernel_seconds=null
9) A2(910B) 尚未投入 Ascend 真机验收（PTO-ISA NPU ST / CANN 样例 / adapter hooks）
```

## 7. 证据路径

```text
对齐与算法    ../worktrees/_meta/pypto-x/qwen35-weighted-multiprompt/
decay 验收    ../worktrees/_meta/pypto-x/verify-w8a-decay-fix/
实现          ../worktrees/_meta/pypto-x/qwen35-vector-runtime-packed-liveness/
CANN 仿真     ../worktrees/_meta/pypto-x/cann-simulator-minimal-probe/
CPU_SIM       ../worktrees/_meta/pypto-x/pto-isa-cpu-sim-baseline/
CUDA toolkit  ../worktrees/_meta/pypto-x/cuda-toolkit-wsl/
CANN 安装     ../worktrees/_meta/pypto-x/cann-toolkit-local-install/
```

---

## 勘误指引（2026-09-10 追加，不覆盖上文）

**ERR-0002**（W8A prefill driver cos/sin 运行时布局缺陷）使本快照 §2 的 en/zh/chat prefill 数字，
以及"chat T=18 真实累积分歧（3.78×band、row 11 起放大、首个不达标层 gold index 8）"的结论**作废**；
替换数字、near-tie 判据与受影响证据清单见 [`ERRATA.zh-CN.md`](ERRATA.zh-CN.md#err-0002) 与 `0036` 快照。
图契约 v3 的 digest/ops、decode token 链、权重映射不受影响；A2(910B) 状态已从"在线待验收"更新为
"W8A-C 真机验收 PASS（限定式关闭 blocked）"，W8H/W8I vllm-ascend 基线独立验收进行中。
