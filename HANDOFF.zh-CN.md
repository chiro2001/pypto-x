# PyPTO-X 接手文档

状态：`W8A_BF16_WEIGHTED_ALIGNED_EN_ZH_CHAT_T18_NEAR_TIE_PASS_ASCEND_A2_ACCEPTANCE_PASS_W8H_W8I_VERIFIED_W8C_C1_C2_VERIFIED_W8B_B5_REDUCE_BROADCAST_VERIFIED_W8B_B3A_CUDA_EVENT_TIMING_VERIFIED`

最后更新：2026-09-11 01:58 CST（Asia/Shanghai；§8 追加 920B 释放与 A2 暂停状态）

项目根目录：`/home/chiro/projects/pypto/pypto_x`

> 首次接手请按 `AGENTS.md` 的顺序读：本文件 → `docs/00-handoffs/ERRATA.zh-CN.md` → `docs/20-planning/0003-…roadmap.zh-CN.md` → 各规范。

## 0. 恢复第一小时（零记忆 TL;DR）

本仓的设计目标是**任意全新 agent 只靠仓内文档即可接手**；若你（或新上下文）什么都不知道，按此执行：

```text
1. 读：AGENTS.md → 本文件（§1–§2）→ ERRATA.zh-CN.md（含 ERR-0002）→ 路线图 §6/§10。
2. 跑 §12 自检。0036 冻结点为 integration @ dca302ef4、111 个补丁；**0037 波次后**为 integration @ 8700f7416、118 个补丁；
   两把锁 FREE、CANN toolkit 在 /usr/local/Ascend。
   自检不重跑 smoke、不下载依赖、不加载权重。
3. 不要做的事：不把 chat(T=18) 说成"真实数值累积分歧/失败"（ERR-0002 已定位为 driver 缺陷并修复；
   严格 argmax 17/18 + row 14 near-tie，按 2026-09-10 修订判据 PASS，但 17/18 必须同时可见）；
   不把 T=1/decode 写成"端到端无变化"（算子层 no-op，端到端 logits 因 state 继承而变）；
   不把静态 lowering 说成真机 PASS；不把 A2 的限定式关闭说成"PyPTO Ascend 后端已通"；
   不推公开仓/不改历史/不发权重证据，除非用户当次明确指示。
4. 用户说"继续"时的默认动作（按优先级）：
   a) 0037 波次已收口：W8B B5（reduce/broadcast 原生化，T=18 prefill 750.7s→208.2s）、
      W8B B3a（CUDA cuEvent 计时）、W8C C1+C2（W8A8 scheme/binding）全部独立验收 PASS；
      W8H/W8I 基线独立验收 PASS。
   b) 下一批候选（见 0037 §7 与路线图 §10）：W8B B1/B2/B3b/B3c/B4；W8C C3→C8；
      性能热点 matmul 65.9s / reshape 27.8s / slice 27.3s / transpose 20.7s（T=18 口径）。
   c) 待用户裁决：Framework Adapter（W8J?）、W8D/E、W8H/W8I 后续范围、本机磁盘清理。
   d) A2 NPU 任务必须遵守卡锁协议 /root/a2-npu-lock/（只保护 NPU 执行）。
5. 并发与资源：活动 subagent ≤3；重任务一律经 scripts/resource/run_local_heavy.sh（返回 75/69 就等待重试，
   禁止绕过）；A2 与 920B 都是租用共享资源，约定串行；A2 只锁 NPU 执行（下载/编译/环境准备可并行）。
6. 重任务需要用户批准才启动（AGENTS.md「Subagent 协议」第一条）。
7. 提交纪律：公开主仓 = 本目录（origin）；改完实现 → cherry-pick 到 integration → 独立 verify → 更新
   development_lock/快照/ERRATA → scripts/remote/export_patches.sh → git push origin main → scripts/remote/sync_private_backup.sh。
   （0036/0037 收口 commits 均只在本地 main，尚未 push；push 前需用户当次明确批准。）
```

找不到答案时，先查 `_meta` 证据目录（`../worktrees/_meta/pypto-x/<task>/`）与 `docs/00-handoffs/` 的历史快照，
再问用户；**不要凭记忆编造数字**——所有阶段性数字都应以 `configs/development_lock.yaml` 与 0037 快照为准。

## 1. 接手摘要

**PyPTO-X（PyPTO Cross-Architecture）** 以官方 PyPTO 同仓的 Tensor/Professional 双前端为基线，把可移植语义与 Ascend 专属 dialect 分层，逐步支持：

- 鲲鹏/AArch64 CPU：scalar reference、SVE256（不开发 NEON 优化后端）；
- x86_64 CPU：scalar、AVX2、AVX-512（后续 AMX）；
- NVIDIA GPU：GPU 公共层 → CUDA/NVVM；
- AMD GPU：GPU 公共层 → HIP/ROCDL；
- Ascend CCE/CANN 保持为一个 target plugin，避免功能回退。

**当前进度一句话**：W1–W4（Core IR/ABI、CPU scalar/AVX2/AVX-512/SVE256、GPU common、CUDA C1–C2）与 AMD `gfx1036` 静态 C1–C3 已冻结；Qwen3.5-0.8B M0–M1K 已冻结；**W8A 真权重 BF16 整网对齐已收口**——en(T=5)/zh(T=8) strict PASS，chat(T=18) 严格 argmax 17/18、row 14 near-tie，按 2026-09-10 修订判据 PASS；期间修复 ERR-0001（decay 门缺 `exp(A_log)`）与 ERR-0002（driver cos/sin 布局缺陷，此前 chat"真实累积分歧"结论作废）；控制仓已公开化；**昇腾 A2(910B3) 真机验收 PASS**（限定式关闭 blocked），W8H/W8I vllm-ascend 基线**独立验收 PASS**；**0037 波次（2026-09-11）**：W8C C1+C2（W8A8 scheme/binding，R1–R5/R7–R9）独立验收 PASS，W8B B5（AVX-512 reduce 超线性修复 + broadcast 全 native）使 T=18 chat prefill 从 750.7s 降到 208.2s（整网位级一致），W8B B3a（CUDA cuEvent 计时）三轮验收后 PASS，并发现平台级 `wsl2_cuevent_rate_offset`（raw cuEvent 偏低 4.4–9.1%，run 内标定+corrected 字段已落地）。

## 2. 当前的量化状态（最重要）

```text
真权重 BF16、AVX-512、公开图 v3（graph_digest 66dd4077…，4,550 ops @ (1,1,4096)）
判据（2026-09-10 用户裁定 + 同日 near-tie 修订）：相对 gold 自身 fp32↔bf16 噪声底 band
  逐行 argmax 与 gold 一致（硬门槛）；near-tie 例外：gold 该行 top1/top2 margin ≤ 0.5×band
  且 ours argmax == gold top-2 时，记 near-tie flip（不判失败但必须单列计数）
  max|Δlogits| ≤ 2×band；cosine ≥ band_cosine − 1e-4
  band：en 0.235212 / zh 0.277682 / chat 0.611310

en(T=5)     PASS  argmax 5/5  cos 0.99997235  max_abs 0.159756（0.68×band）  逐层≥0.99993098
zh(T=8)     PASS  argmax 8/8  cos 0.99993803  max_abs 0.235296（0.85×band）  逐层≥0.99993980
chat(T=18)  PASS（修订判据）严格 argmax 17/18（唯一 mismatch row 14）cos 0.99989976
            max_abs 0.416867（0.68×band）逐层≥0.99988217
            → row 14 near-tie：gold fp32 top1 271=19.956333160 / top2 198=19.841325760，
              margin 0.115007=0.18813×band（阈值 0.5×band=0.305655），ours argmax=198=gold runner-up
              = 17/17 有效行 + 1 near-tie；**严格 17/18 必须保留可见**
三 prompt decode token 链与 gold 全一致；state 全有限，recurrent |state|max en 0.55–13.27 / zh 0.82–13.29 / chat 0.69–14.14
实测 ops(T) = 6,728 + 540×(T−5)（图契约未变，公式仍有效）
修复后复跑窗口：window1 958.8 s / recovery 768.7 s / window2 1499.1 s，合计锁内 ≈53.8 min；
旧 per-prompt wall（132/377/932 s）与峰值 RSS 产生于旧 driver，仅作量级参考，不作为修复后冻结数字
```

**结论口径**：
- ERR-0002（driver cos/sin 布局缺陷）修复后，chat(T=18) 不存在"真实数值累积分歧"；此前定位（row 11 起、
  首个不达标层 gold index 8 / layer_07）作废。严格口径 17/18 与修订后的 near-tie 例外必须同时可见。
- **T=1/decode**：算子层布局是 no-op（单 position cos/sin 位序列逐位相同），但端到端 decode logits 因继承
  prefill state 而变化；旧 decode cos/max_abs 数字作废（token 链仍全对）。**不得写"decode 端到端无变化"**。
- 引用 A2 真机结论必须带限定：仅 PTO-ISA tassign 单用例 + 注入式最小固定 64×64 f32 add hook 真实上卡；
  不代表 Core IR→PTO、classic/Pro JIT/OPC 或模型级。

## 3. 已确定的技术决策（新增/变更部分）

1. `pypto` Tensor 与 `pypto_pro` Professional 是同一 wheel 内的并存前端；Tensor frontend 作为跨架构 Core IR 主语义入口，Pro 保留为 Ascend expert dialect。
2. 目标无关 Core IR 只表达 Tensor/Scalar/Shape/View/控制流/逻辑 Tile 与副作用；Ascend 的 UB/L0、AIC/AIV、MTE、Pipe、CCE 语义留在 Ascend target。
3. CPU 共享 lowering，优先级 AVX2 → AVX-512 → SVE256（无 NEON）；NVIDIA/AMD 共享 GPU IR/Runtime ABI，再分别下降 CUDA/NVVM 与 HIP/ROCDL。
4. 首个实际模型固定 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期只做纯文本 BF16 与 W8A8-linear；不做 W8A16/INT4/FP8/KV-GDR state 量化；视觉编码器不入 MVP。
5. **对齐判据用相对 dtype 带宽**（§2），不再用绝对 max_abs 阈值；`decode_logits[0] ≡ prefill_next_logits`，第 k 步对齐 `decode_logits[k+1]`。
6. **near-tie 例外**（2026-09-10 用户裁定）：逐行 argmax 仍为硬门槛；仅当 gold 该行 top1/top2 margin ≤ 0.5×band
   且 ours argmax == gold top-2 token 时，记 near-tie flip、不判失败但必须单列计数。严格口径结果必须同时可见。
   当前 chat 仅 row 14 适用（en/zh 无适用的 mismatch 行）。
7. **ERR-0002**：`qwen35_weighted_execution_driver.py` 的 cos/sin 布局缺陷已修复（head-major，integration `5f479d1a1`）；
   图契约零改动；旧 driver 产出的 prefill/decode logits 数字作废。`327b17158` 调试变体未合入、未采纳。
8. **仓库布局**：公开主仓 `chiro2001/pypto-x`（Apache-2.0 + NOTICE + `patches/`）是工作副本；私有归档 `chiro2001/pypto-x-private` + 本地备份 `../pypto_x_private_bkp`；**实现补丁以 `patches/` 形式发布**（111 个，base `34475e0d8`，HEAD `dca302ef4`）。
9. **发布纪律**：推公开仓属发布动作需用户明确指示；不得上传上游源码、模型权重、运行证据；访问脚本/端点只留本地私有侧。
10. AMD `gfx1036` 运行态判定**架构性不可达**，用户决定长期只保留静态证据。
11. Ascend 线：本机 CANN 9.2.0-beta.2 仅 toolkit；CA-model 可跑但极慢（95 s/小算子、7.3 GiB），只能做指令级细看；
    真机走 A2(910B3)：W8A-C 真机验收 PASS（限定式关闭 `ascend_cann_bisheng_npu_regression_blocked`，
    仅覆盖 tassign 单用例 + 注入式最小固定 64×64 f32 add hook）；**A2 NPU 任务必须走 `/root/a2-npu-lock/`**
    （只保护 NPU 执行，下载权重/编译/环境准备可并行；request→using→done + 历史保留；180 s 轮询；
    TTL 6 h + PID 僵死判定；A2 时钟约快 8 h）。

## 4. 目录与 Git 状态

```text
控制仓/公开主仓   /home/chiro/projects/pypto/pypto_x   origin = chiro2001/pypto-x（公开）
私有备份          ../pypto_x_private_bkp（archive 分支含第三方离线副本；main 跟随公开）
现有 HEAD         5fd3a11（收口前；本地 6e93dcb/5fd3a11 与 0036 收口 commits 均未 push）
实现主仓          upstream/pypto @ 34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad（只读）
集成分支          port/pypto-x-integration @ dca302ef4（包含 W8G/A2 验收/W8H/RoPE 修复/W8I）
补丁集            patches/pypto-x/ 111 个（base 34475e0d8，HEAD dca302ef4）
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
① 0037 波次已完成：W8C C1+C2、W8B B5、W8B B3a 全部独立验收 PASS（详见 0037 快照）
② 性能续作：T=18 剩余热点 matmul 65.9s / reshape 27.8s / slice 27.3s / transpose 20.7s；
   W8B B1（AVX2 packed）、B2（SVE fallback）、B3b（GEMM vs cuBLAS）、B3c（本机 L0/L1）、B4（perf freeze）
③ W8C 续作：C3 AVX2 widening → C4 AVX-512(VNNI) → C5 SVE256 → C6 CUDA/AMD 静态 →
   C7 layer ladder → C8 model validation（R6/R10–R14 未验；D5 阈值待 BF16 基线后冻结）
④ 待用户裁决：Framework Adapter（是否立 W8J；torch custom-op / vLLM plugin / HF 集成）；
   W8D/E：GDR fused WY、T=64/128 代价评估、Core IR→PTO/CCE codegen（E4）、stable CANN 9.2.0-beta.2 上卡
⑤ 平台注记：GamePC WSL2 的 cuEvent rate offset（wsl2_cuevent_rate_offset）已写入 known limits；
   CUDA 计时必须 raw + corrected + platform_flags 同报
⑥ 可选：若用户要求 chat 严格 18/18，评估任务分支 327b17158（--debug-f32-residual，未合入、未采纳）
```

## 8. 可用资源实况

```text
本机               12 vCPU（KVM guest，无 cpufreq）/ 29 GiB 内存 / 磁盘余 ~78 GB；CANN 9.2.0-beta.2 toolkit 已装
                   重任务必须经 run_local_heavy.sh 取得全局 local 锁（默认 MemoryMax=min(启动 MemAvailable−4096, 20480) MiB）
GamePC 192.168.101.5  WSL2 24 线程 / 30 GiB（宿主 61.4 GiB）/ RTX 5080 16 GB；CUDA Toolkit（nvcc 13.3.73+cuBLAS 13.6）
                    GPU-only 短探测不申请 gamepc；host-heavy 才持锁；Linux 命令经 wsl.exe -e bash -lc
鲲鹏 920B ECS       2 vCPU / 2.5 GiB / SVE=1 VL=32、SVE2=0；约定串行；QEMU 只作功能验证
                    **2026-09-11 已按用户指示释放**（实例+系统盘+EIP 删除，计费停止）；释放前 /root 工作目录
                    已归档 ../worktrees/_meta/pypto-x/ecs920b-release-archive-20260911/（758 文件 / 9.8 MB，条目核对一致）；
                    需用时以 ~/tools/ecs-920B/create.sh + setup-access.sh 重建 → **重建前 920B 线任务（B2 等）不发射**
昇腾 A2(910B3)      容器 256 vCPU / 2 TB / 1×910B3（64 GB HBM）/ CANN 9.0.0 / driver 25.2.0
                    **2026-09-11 起暂停（用户指示）**：机时预算 100 h、已用 ~6 h，暂停不消耗机时且文件保留；
                    待用户通知恢复 → **恢复前 A2 任务一律不发射**（E4/E5、W8H/W8I 后续范围、stable CANN 9.2.0-beta.2
                    上卡、classic/Pro JIT/OPC/gym 验证；另 A2 上 ~8.03 GiB profiler trace 待清理）；
                    暂停时无在途任务、卡锁已释放（.using→.done）、无残留进程
                    W8A-C 真机验收 PASS；W8H/W8I vllm-ascend 基线**独立验收 PASS**
                    **卡占用协议**：容器固定目录 /root/a2-npu-lock/（README.zh-CN.md + a2_card_lock.sh）；
                    流程 request→using→done（历史保留），180 s 轮询，TTL 6 h + PID 僵死判定；
                    **只保护 NPU 执行**（下载权重/编译/环境准备可并行）；A2 时钟约快 8 h；
                    卡锁协议已由独立验收实际使用验证（.using→.done、无残留进程、HBM 回落 3441/65536 MiB）
                    访问脚本与端点只在本地私有侧（~/tools/a2-910b/ + ~/.ssh/a2-910b.env），不进仓；约定串行
                    出网仅 HTTPS(gitcode/pypi)；无 22 出网、无 TUN/NET_ADMIN（VPN 不可行）
资源锁             local / gamepc 当前均 FREE；锁协议见 /home/chiro/projects/.resource-locks/README.md；
                   A2 卡锁仅在 A2 容器内、与 local/gamepc 锁相互独立
```

## 9. 已验证结果（择要，完整见 `configs/development_lock.yaml`）

- W1–W4：Core IR、Target ABI、CPU scalar/AVX2/AVX-512/SVE256、GPU common、CUDA C1–C2 全部独立验收通过；
- AMD `gfx1036` 静态 C1–C3：math/reduction/rank-2..4 matmul，Qwen 静态 4,532/4,532（graph v2；v3 为 4,550，**AMD 尚未对 v3 重跑**）；
- Qwen M0–M1K：24 层无权重 decoder、3/320/48 binding、CPU/CUDA external ingestion；
- **W8A 真权重对齐**：en/zh strict PASS；chat 严格 argmax 17/18 + row 14 near-tie，按修订判据 PASS（§2）；
  decode token 链全对；state 有限；
- ERR-0001 decay 修复 + 独立验收 PASS_WITH_BOUNDARIES（liveness A/B 逐位一致；packed 内核逐位一致）；
- **ERR-0002 driver cos/sin 布局缺陷修复 + 独立验收 PASS**（`5f479d1a1`；图契约零改动；694 collected/687 passed/7 skipped）；
- **W8A-C Ascend A2(910B3) 真机验收 PASS + 独立验收 PASS**：PTO-ISA tassign NPU ST、CANN mspti aclnn Add 样例、
  注入式 hook 真实 bisheng 编译 22,728 B / sha256 4d7f704c… / live 9/9 / 真机 max_abs_err=0.0；
  限定式关闭 `ascend_cann_bisheng_npu_regression_blocked`，新增 3 条限定项；
- **W8G qwen35-portability-cleanup 完成 + 独立验收 PASS**（integration c464927fa）；
- **W8H vllm-ascend E2E 基线 + W8I profiling 基线：独立验收 PASS**（复跑/重算一致；非阻断差异已登记）；
  性能协议仍为提案，profiler 开销不得当模型性能引用；
- CPU vector runtime liveness + AVX-512 packed cast/transpose/embedding：真权重整网峰值从 309.6 GiB 投影降到 3.9 GiB 实测；
- GDR T=128：五后端 + 920B 原生 PASS，关闭 `gdr_t128_not_validated`；
- 基础设施：CUDA Toolkit、本机 CANN toolkit、PTO-ISA CPU_SIM 125/125、CANN CA-model 最小用例（审计 0006/0007）；
- **W8C C1+C2 独立验收 PASS**（R1–R5/R7–R9；QuantizedTensorDesc + 4 opcode + binding v2/layout v2/coverage；
  `w8a8_linear_coverage=0.9988…`、`whole_net_int8_compute_ratio=0.9770…`，static shape model）；
- **W8B B5 独立验收 PASS**：T=18 chat prefill 750.718s→208.174s（验收复跑 233.33s），reduce_sum 5963→2.45 ms/call、
  broadcast 962/962 native；位级一致（58/315 与 43/265 两套独立用例）；全量 753/746/7/0；
- **W8B B3a 独立验收 PASS（r3，0 discrepancy）**：raw cuEvent `kernel_seconds` + 平台速率标定 +
  `kernel_seconds_corrected` + `platform_flags`；`wsl2_cuevent_rate_offset` 已进 known limits。

## 10. 许可证状态

- **本仓自有内容**（docs/configs/scripts/patches）：**Apache-2.0**（`LICENSE` + `NOTICE`）；`patches/` 是对上游代码的补丁文本，应用后衍生作品仍受上游许可约束；
- 上游 CANN Open Software License 2.0 仍限制在华为 AI 处理器/软件场景；**对外发布非华为处理器衍生后端前**需取得新许可证、双许可证或明确书面例外（维护方态度积极，但尚无正式文本）；
- 发布纪律见 §3.7。本文件不构成法律意见。

## 11. 新 Agent 首次回复前必须确认

1. 已读 `AGENTS.md`、本文件、`ERRATA.zh-CN.md`（含 ERR-0002）、W8 路线图；
2. 理解 §2 的判据口径：chat(T=18) 严格 argmax 17/18 + row 14 near-tie（修订判据 PASS），
   不得表述为"真实数值累积分歧"或"完全无保留的对齐"，17/18 必须同时可见；
3. 理解 ERR-0002：T=1 算子层 no-op，但端到端 decode 因 state 继承而变，不得写"decode 端到端无变化"；
4. 理解仓库布局（公开主仓为工作副本、私有备份、补丁 118 个、发布纪律）；
5. 理解并发上限与资源锁纪律，以及 A2 卡锁协议（/root/a2-npu-lock/，只保护 NPU 执行）；
6. 知道 A2(910B3) W8A-C 真机验收已 PASS（限定式），W8H/W8I 独立验收已 PASS；访问脚本在本地私有侧；
7. 未经用户明确指示不推送公开仓、不发布权重/证据、不改写公开历史。

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
ls patches/pypto-x/*.patch | wc -l                            # 应为 118（0037 冻结点）
```

除非用户明确要求，接手自检**不重跑** smoke、不下载依赖、不加载权重。
