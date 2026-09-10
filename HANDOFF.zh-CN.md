# PyPTO-X 接手文档

状态：`W8A_BF16_WEIGHTED_ALIGNED_EN_ZH_PASS_CHAT_T18_DIVERGENT_ASCEND_A2_ONLINE`

最后更新：2026-09-10 20:30 CST（Asia/Shanghai）

项目根目录：`/home/chiro/projects/pypto/pypto_x`

> 首次接手请按 `AGENTS.md` 的顺序读：本文件 → `docs/00-handoffs/ERRATA.zh-CN.md` → `docs/20-planning/0003-…roadmap.zh-CN.md` → 各规范。

## 0. 恢复第一小时（零记忆 TL;DR）

本仓的设计目标是**任意全新 agent 只靠仓内文档即可接手**；若你（或新上下文）什么都不知道，按此执行：

```text
1. 读：AGENTS.md → 本文件（§1–§2）→ ERRATA.zh-CN.md → 路线图 §6/§10。
2. 跑 §12 自检。预期：integration @ 9aae4649e、96 个补丁、两把锁 FREE、CANN toolkit 在 /usr/local/Ascend。
   自检不重跑 smoke、不下载依赖、不加载权重。
3. 不要做的事：不把 chat(T=18) 说成"整网已对齐"（仍是 FAIL，§2）；不把静态 lowering 说成真机 PASS；
   不推公开仓/不改历史/不发权重证据，除非用户当次明确指示。
4. 用户说"继续"时的默认动作（按优先级）：
   a) W8A-C Ascend 真机验收（A2 910B 已在线、最高优先）——访问脚本在本地私有侧 ~/tools/a2-910b/（先读其 README）；
      若隧道不通，脚本会自动回退平台跳板，但**跳板 token 约 10 分钟失效，需要用户重新提供**；不要自己在仓内找凭据。
   b) W8A-B chat(T=18) 数值分歧攻坚（row 11 起、首个不达标层 gold index 8）。
   c) 然后才轮到 W8B/W8C（见路线图 §10 待决策）。
5. 并发与资源：活动 subagent ≤3；重任务一律经 scripts/resource/run_local_heavy.sh（返回 75/69 就等待重试，
   禁止绕过）；A2 与 920B 都是租用共享资源，约定串行。
6. 重任务需要用户批准才启动（AGENTS.md「Subagent 协议」第一条）。
7. 提交纪律：公开主仓 = 本目录（origin）；改完实现 → cherry-pick 到 integration → 独立 verify → 更新
   development_lock/快照/ERRATA → scripts/remote/export_patches.sh → git push origin main → scripts/remote/sync_private_backup.sh。
```

找不到答案时，先查 `_meta` 证据目录（`../worktrees/_meta/pypto-x/<task>/`）与 `docs/00-handoffs/` 的历史快照，
再问用户；**不要凭记忆编造数字**——所有阶段性数字都应以 `configs/development_lock.yaml` 与 0035 快照为准。

## 1. 接手摘要

**PyPTO-X（PyPTO Cross-Architecture）** 以官方 PyPTO 同仓的 Tensor/Professional 双前端为基线，把可移植语义与 Ascend 专属 dialect 分层，逐步支持：

- 鲲鹏/AArch64 CPU：scalar reference、SVE256（不开发 NEON 优化后端）；
- x86_64 CPU：scalar、AVX2、AVX-512（后续 AMX）；
- NVIDIA GPU：GPU 公共层 → CUDA/NVVM；
- AMD GPU：GPU 公共层 → HIP/ROCDL；
- Ascend CCE/CANN 保持为一个 target plugin，避免功能回退。

**当前进度一句话**：W1–W4（Core IR/ABI、CPU scalar/AVX2/AVX-512/SVE256、GPU common、CUDA C1–C2）与 AMD `gfx1036` 静态 C1–C3 已冻结；Qwen3.5-0.8B M0–M1K 已冻结；**W8A 完成"真权重 BF16 整网对齐"**——en(T=5)、zh(T=8) 按相对 dtype 带宽判据 PASS，chat(T=18) 仍有真实数值累积分歧（FAIL，见 §2）；期间修复了冻结契约级 bug ERR-0001（GDR decay 门缺 `exp(A_log)`）；控制仓已公开化；**昇腾 A2(910B) 真机已在线**，尚未投入验收。

## 2. 当前的量化状态（最重要）

```text
真权重 BF16、AVX-512、公开图 v3（graph_digest 66dd4077…，4,550 ops @ (1,1,4096)）
判据（2026-09-10 用户裁定）：相对 gold 自身 fp32↔bf16 噪声底 band
  逐行 argmax 必须一致；max|Δlogits| ≤ 2×band；cosine ≥ band_cosine − 1e-4
  band：en 0.235212 / zh 0.277682 / chat 0.611310

en(T=5)     PASS  argmax 5/5  cos 0.99991364  max_abs 0.233829（0.99×band）  逐层≥0.99955
zh(T=8)     PASS  argmax 8/8  cos 0.99973494  max_abs 0.463474（1.67×band）  逐层≥0.99945
chat(T=18)  FAIL  argmax 18/18 cos 0.99799387 max_abs 2.310620（3.78×band）  逐层≥0.99790
                  → 真实累积分歧：row 11 起放大，首个不达标层 = gold index 8（layer_07_output）
三 prompt decode token 链与 gold 全一致；48 state 全有限，recurrent |state|max 12.4–14.2
实测 ops(T) = 6,728 + 540×(T−5)；prefill wall：T=5 132 s / T=8 377 s / T=18 932 s，峰值 RSS ≤4.1 GiB
```

**结论口径**：不得把 chat(T=18) 说成"已对齐"；T≥8 的精度问题已登记为 `t8_t18_prefill_numeric_divergence_beyond_gold_dtype_band`，是**待查的数值路径问题**，不是阈值问题。

## 3. 已确定的技术决策（新增/变更部分）

1. `pypto` Tensor 与 `pypto_pro` Professional 是同一 wheel 内的并存前端；Tensor frontend 作为跨架构 Core IR 主语义入口，Pro 保留为 Ascend expert dialect。
2. 目标无关 Core IR 只表达 Tensor/Scalar/Shape/View/控制流/逻辑 Tile 与副作用；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe、CCE 语义留在 Ascend target。
3. CPU 共享 lowering，优先级 AVX2 → AVX-512 → SVE256（无 NEON）；NVIDIA/AMD 共享 GPU IR/Runtime ABI，再分别下降 CUDA/NVVM 与 HIP/ROCDL。
4. 首个实际模型固定 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期只做纯文本 BF16 与 W8A8-linear；不做 W8A16/INT4/FP8/KV-GDR state 量化；视觉编码器不入 MVP。
5. **对齐判据用相对 dtype 带宽**（§2），不再用绝对 max_abs 阈值；`decode_logits[0] ≡ prefill_next_logits`，第 k 步对齐 `decode_logits[k+1]`。
6. **仓库布局**：公开主仓 `chiro2001/pypto-x`（Apache-2.0 + NOTICE + `patches/`）是工作副本；私有归档 `chiro2001/pypto-x-private` + 本地备份 `../pypto_x_private_bkp`；**实现补丁以 `patches/` 形式发布**（96 个，base `34475e0d8`）。
7. **发布纪律**：推公开仓属发布动作需用户明确指示；不得上传上游源码、模型权重、运行证据；访问脚本/端点只留本地私有侧。
8. AMD `gfx1036` 运行态判定**架构性不可达**，用户决定长期只保留静态证据。
9. Ascend 线：本机 CANN 9.2.0-beta.2 仅 toolkit；CA-model 可跑但极慢（95 s/小算子、7.3 GiB），只能做指令级细看；真机走 A2(910B)。

## 4. 目录与 Git 状态

```text
控制仓/公开主仓   /home/chiro/projects/pypto/pypto_x   origin = chiro2001/pypto-x（公开）
私有备份          ../pypto_x_private_bkp（archive 分支含第三方离线副本；main 跟随公开）
现有 HEAD         832c38a（公开仓同步）
实现主仓          upstream/pypto @ 34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad（只读）
集成分支          port/pypto-x-integration @ 9aae4649e
任务 worktree     ../worktrees/pypto-x/<task>；证据 ../worktrees/_meta/pypto-x/<task>/
```

`upstream/*` 只读；`upstream/PTOAS/.codex/CLAUDE.md` 的 dirty 来自上游 CRLF/`.gitattributes` 不一致，不是人工改动。

## 5. 已完成的规划与规范交付

`README.md`、`AGENTS.md`、`docs/00-handoffs/`（含 `ERRATA.zh-CN.md`）、`docs/10-architecture/0001–0004`、
`docs/20-planning/0001（MVP）/0002（W8A8 契约，D1–D12 已批准）/0003（W8 并行路线图 v2，含外部评审修订）`、
`docs/WORKTREE_AGENT_PLAN`、`docs/PROJECT_LAYOUT`、`docs/RESOURCE_MATRIX`、`docs/LOCAL_RESOURCE_POLICY`、
`docs/SMOKE_TEST_SPEC`、`docs/PERF_MEASUREMENT_PROTOCOL`（提案）、`configs/*.yaml`、`patches/README.md`、
审计 `research/audits/2026/0001–0007`。

## 6. Subagent 与协作协议（要点）

- 每个 subagent：一 task、一 branch、一 worktree；启动字段必须含 `started_at`、绝对 worktree、branch、
  `smoke_once=true`、`wait_timeout_seconds=3600`、`poll=false`、`resource_lock_root=/home/chiro/projects/.resource-locks`、
  `local_heavy_policy=locked`、`local_heavy_runner=…/scripts/resource/run_local_heavy.sh`、
  `local_min_available_mib=8192`、`local_safety_floor_mib=4096`、`local_max_cpus=6`。
- **并发上限**：平台 4 槽含父 agent → 活动 subagent ≤3（推荐 1 local-heavy + 1 远端 + 1 轻任务）。
- 阶段验收必须独立：从待验收 integration HEAD 建 `verify/<phase>` worktree，源码只读，只写独占 `_meta`。
- 父 agent 不自己跑重任务；同一任务禁止同时提交两个重命令；不在锁内跑无关命令。
- 重任务经 `run_local_heavy.sh`（75/69 等待重试，禁止绕过）；大内存任务显式登记 `memory_max_mib`。
- 冻结流水线（每个阶段收口）：实现 commit → 合并 integration → 独立验收 → 更新 `development_lock` 与快照/勘误 →
  重跑 `scripts/remote/export_patches.sh` → push 公开仓 + `sync_private_backup.sh`。

## 7. 建议的下一步波次（W8 及以后）

```text
① Ascend 真机验收（A2/910B，最高优先，可立即开始）
   PTO-ISA run_st.py -r npu 最小用例 → CANN 自带样例 → backends/ascend/adapter.py 真 hook 回归
   目标：关闭 ascend_cann_bisheng_npu_regression_blocked；给 upstream_lock 的 stable lock 补 CANN 配套证据
② T≥8 精度攻坚（chat T=18 的分歧）
   逐 op bf16 舍入链 vs 参考 fp32 累加；定位 row 11 / layer_07 之后的发散来源；评估累加点保 fp32
③ W8B 硬化：AVX2 packed 参数级算子 / SVE fallback 分类清零（含 broadcast/where）/
   CUDA cuEvent 计时 + GEMM vs cuBLAS 相对门槛 / 本机 L0–L1 性能基线 → 冻结 perf 协议
④ W8C：W8A8-linear 实现（契约已冻结）：C1 scheme → C2 binding → C3 AVX2 → C4 AVX-512(VNNI) →
   C5 SVE256 → C6 CUDA/AMD → C7 layer ladder → C8 model validation
⑤ W8D/W8E：GDR fused WY、长序列 T=64/128 代价评估；CANN report（plotly）、npusim 复现性、
   IR→PTO 桥与 Ascend hooks
```

## 8. 可用资源实况

```text
本机               12 vCPU（KVM guest，无 cpufreq）/ 29 GiB 内存 / 磁盘余 ~78 GB；CANN 9.2.0-beta.2 toolkit 已装
                   重任务必须经 run_local_heavy.sh 取得全局 local 锁（默认 MemoryMax=min(启动 MemAvailable−4096, 20480) MiB）
GamePC 192.168.101.5  WSL2 24 线程 / 30 GiB（宿主 61.4 GiB）/ RTX 5080 16 GB；CUDA Toolkit（nvcc 13.3.73+cuBLAS 13.6）
                    GPU-only 短探测不申请 gamepc；host-heavy 才持锁；Linux 命令经 wsl.exe -e bash -lc
鲲鹏 920B ECS       2 vCPU / 2.5 GiB / SVE=1 VL=32、SVE2=0；约定串行；QEMU 只作功能验证
昇腾 A2(910B3)      容器 256 vCPU / 2 TB / 1×910B3（64 GB HBM）/ CANN 9.0.0 / driver 25.2.0 —— 在线
                    访问脚本与端点只在本地私有侧（~/tools/a2-910b/ + ~/.ssh/a2-910b.env），不进仓；约定串行
                    出网仅 HTTPS(gitcode/pypi)；无 22 出网、无 TUN/NET_ADMIN（VPN 不可行）
资源锁             local / gamepc 当前均 FREE；锁协议见 /home/chiro/projects/.resource-locks/README.md
```

## 9. 已验证结果（择要，完整见 `configs/development_lock.yaml`）

- W1–W4：Core IR、Target ABI、CPU scalar/AVX2/AVX-512/SVE256、GPU common、CUDA C1–C2 全部独立验收通过；
- AMD `gfx1036` 静态 C1–C3：math/reduction/rank-2..4 matmul，Qwen 静态 4,532/4,532（graph v2；v3 为 4,550，**AMD 尚未对 v3 重跑**）；
- Qwen M0–M1K：24 层无权重 decoder、3/320/48 binding、CPU/CUDA external ingestion；
- **W8A 真权重对齐**：en/zh PASS、chat T=18 FAIL（§2）；decode 链全对；state 有限；
- ERR-0001 decay 修复 + 独立验收 PASS_WITH_BOUNDARIES（674 passed/7 skipped；liveness A/B 逐位一致；packed 内核逐位一致）；
- CPU vector runtime liveness + AVX-512 packed cast/transpose/embedding：真权重整网峰值从 309.6 GiB 投影降到 3.9 GiB 实测；
- GDR T=128：五后端 + 920B 原生 PASS，关闭 `gdr_t128_not_validated`；
- 基础设施：CUDA Toolkit、本机 CANN toolkit、PTO-ISA CPU_SIM 125/125、CANN CA-model 最小用例（审计 0006/0007）。

## 10. 许可证状态

- **本仓自有内容**（docs/configs/scripts/patches）：**Apache-2.0**（`LICENSE` + `NOTICE`）；`patches/` 是对上游代码的补丁文本，应用后衍生作品仍受上游许可约束；
- 上游 CANN Open Software License 2.0 仍限制在华为 AI 处理器/软件场景；**对外发布非华为处理器衍生后端前**需取得新许可证、双许可证或明确书面例外（维护方态度积极，但尚无正式文本）；
- 发布纪律见 §3.7。本文件不构成法律意见。

## 11. 新 Agent 首次回复前必须确认

1. 已读 `AGENTS.md`、本文件、`ERRATA.zh-CN.md`、W8 路线图；
2. 理解 §2 的判据口径与 chat(T=18) 仍是 FAIL，不得表述为"整网已完全对齐"；
3. 理解仓库布局（公开主仓为工作副本、私有备份、补丁发布、发布纪律）；
4. 理解并发上限与资源锁纪律；
5. 知道 A2(910B) 已在线但尚未验收，访问脚本在本地私有侧；
6. 未经用户明确指示不推送公开仓、不发布权重/证据、不改写公开历史。

## 12. 快速自检命令

```bash
cd /home/chiro/projects/pypto/pypto_x

git status --short && git log --oneline -1 && git remote -v | head -2
git -C upstream/pypto status --short && git -C upstream/pypto rev-parse HEAD
git -C ../worktrees/pypto-x/integration log --oneline -1
scripts/worktree/status.sh
bash -n scripts/worktree/*.sh scripts/smoke/*.sh scripts/resource/*.sh scripts/remote/*.sh
python3 -c "import yaml;[yaml.safe_load(open(p)) for p in ['configs/development_lock.yaml','configs/agent_tasks.yaml','configs/upstream_lock.yaml']];print('yaml ok')"
/home/chiro/projects/.resource-locks/resource-lock status     # 只观察；取得锁必须用 run
ls patches/pypto-x/*.patch | wc -l                            # 应为 96
```

除非用户明确要求，接手自检**不重跑** smoke、不下载依赖、不加载权重。
