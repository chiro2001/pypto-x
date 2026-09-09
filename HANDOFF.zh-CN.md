# PyPTO-X 接手文档

状态：`EXECUTION_W6_QWEN35_AVX2_AVX512_W4_CUDA_C2_COMPLETE_DECODER_CONNECTIVITY_READY`

最后更新：2026-09-09 10:17 CST（Asia/Shanghai）

项目根目录：`/home/chiro/projects/pypto/pypto_x`

## 1. 接手摘要

本项目工作名为 **PyPTO-X（PyPTO Cross-Architecture，PyPTO 跨架构后端）**。目标是以官方 PyPTO 同仓的 Tensor/Professional 双前端为研究基线，把可移植语义与 Ascend 专属 dialect 分层，并逐步支持：

- 鲲鹏/AArch64 CPU：scalar reference、SVE256/SVE2（不开发 NEON 优化后端）；
- x86_64 CPU：scalar、AVX2、AVX-512，后续 AMX；
- NVIDIA GPU：GPU 公共层 → CUDA/NVVM；
- AMD GPU：GPU 公共层 → HIP/ROCDL；
- 现有 Ascend CCE 路径保持为一个 target plugin，并避免功能回退。

目前已完成调研、架构规划、项目整理、资源盘点、上游 edge 刷新、CANN/算子生态审计、C0 控制仓治理、W1/W2/W2B、完整 W3、W4 SVE256/GPU common/CUDA C1–C2，以及 Qwen3.5-0.8B M0–M1H 与 AVX2/AVX-512 parity。integration 已包含 Core IR、Target/Compiler/Runtime ABI、CPU scalar/vector common、Clang x86 compiler/runtime、AArch64 SVE256 compiler/runtime、NVIDIA/AMD 共用的 vendor-neutral GPU IR/ABI，以及 NVIDIA CUDA Driver API + PTX JIT 后端；HIP 尚未实现。AVX2 parity 独立验收为 `474 passed`；AVX-512 parity 为 `483 passed, 7 skipped`，ZMM/opmask、FMA、BF16 `vdpbf16ps`、VNNI `vpdpbusd`、rank-3/4 matmul、非零 GDR state 与 position 大 shape 均通过。CUDA C2 独立验收为 `476 passed, 7 skipped`，RTX 5080 上 math、position、layout/indexing、rank-2/3/4 matmul、scalar SSA 与 Qwen composites 均通过。5080 GPU 由 PyPTO-X 独占；GPU-only 工作不需要 `gamepc` 锁，只有远端 host-heavy 阶段才持锁。下一步把24层 identity shell 升级为无权重、非 identity 的完整 decoder connectivity。Tensor frontend 是跨架构主入口，Pro 保留为 Ascend expert dialect，并后续提取 portable subset。

## 2. 已经确定的技术决策

1. `pypto` Tensor 与 `pypto_pro` Professional 是同一 `pypto` wheel 内的并存前端，共享部分 native IR/type substrate，但不共用完整 parser/编译/运行时链路。
2. Tensor frontend 作为跨架构 Core IR 的主语义入口；不以 classic Ascend-native 45-Pass MPMD 流水线作为新后端。Pro 保留为 Ascend expert dialect，后续只提取明确的 portable subset。
3. 建立目标无关 Core IR，仅表达 Tensor、Scalar、Shape、View、控制流、逻辑 Tile 和副作用。
4. 在 Core IR 后按 target 分叉；不把 PTO-ISA 或 CCE intrinsic 机械翻译成 NEON/AVX/CUDA/HIP。
5. 把当前 CCE 代码生成、Bisheng 编译和 CANN runtime 改造成 Ascend plugin。
6. AArch64 与 x86 共用 CPU vector lowering；开发优先级为 AVX2 → AVX-512 → SVE256，不开发 NEON 优化后端。
7. NVIDIA 与 AMD 共享 Grid/Block/Thread/Subgroup、address space、artifact 和 launch ABI；矩阵 fragment 保持 target-specific opaque。
8. 首个实际模型固定为 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`，首期验证纯文本 prefill + decode，并用 9B/27B 无权重 shape harness 补充 Dense 系列尺寸泛化。
9. 模型 MVP 只支持 BF16 和 W8A8-linear；W8A16、W4A16/INT4、FP8、KV cache/GDR state 量化均不在首期范围。
10. 视觉编码器不纳入当前模型 MVP。
11. 性能验收门槛在首批可运行后端完成后依据实测制定；当前先固定正确性、可复现性和报告字段。

详细理由见：

- [跨平台移植研究报告](research/PYPTO_PORTING_RESEARCH.zh-CN.md)
- [PyPTO 生态审计](research/audits/2026/0001-2026-09-06-pypto-ecosystem.zh-CN.md)
- [PyPTO Tensor/Professional 关系审计](research/audits/2026/0002-2026-09-06-pypto-pypto-pro-relationship.zh-CN.md)
- [算子可移植占比审计](research/audits/2026/0003-2026-09-07-operator-portability-ratio.zh-CN.md)
- [项目模块布局](docs/PROJECT_LAYOUT.zh-CN.md)
- [Worktree 与并行计划](docs/WORKTREE_AGENT_PLAN.zh-CN.md)
- [Qwen3.5-0.8B BF16/W8A8 模型 MVP](docs/20-planning/0001-2026-09-07-qwen35-08b-bf16-w8a8-mvp.zh-CN.md)

## 3. 当前目录和 Git 状态

根目录包含：

```text
AGENTS.md                         # 新 Agent 的强制工作规则
HANDOFF.zh-CN.md                  # 本文件
README.md                         # 总入口
configs/agent_tasks.yaml          # 任务 DAG、资源和启动协议
docs/                             # 架构、worktree、资源、smoke 规范
research/                         # 完整调研报告和来源清单
references/                       # 公开讲稿/活动页离线副本
scripts/worktree/                 # worktree 创建和状态工具
scripts/smoke/                    # 无模型一次 smoke
upstream/                         # 五个嵌套 Git 仓库
../worktrees/                     # 位于项目外的 linked worktree、元数据和日志
```

根目录已经初始化为轻量控制 Git 仓；五个 `upstream/*` 以固定 commit 的 submodule/gitlink 登记。`upstream/pypto` 仍是实现主仓，目前：

```text
branch = master
HEAD   = 34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
linked worktree 数 = 26（基线、integration、既有调研/W1-W4 task，以及 Qwen M1A/M1B/M1C1/M1C2a/M1C2b/M1D/M1E task worktree）
工作树 = clean
```

五个本地源码快照：

| 仓库 | 本地提交 | 本地分支 |
|---|---|---|
| `upstream/pypto` | `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad` | `master` |
| `upstream/pypto-gym` | `945a360e12592239a3549cb62d0db37af32bbc03` | `master` |
| `upstream/pto-isa` | `668248ec886447a83787200786fe6f461169b701` | `master` |
| `upstream/pypto-community` | `9f657f37ed20ce148b46fb7229c267a152a0644e` | `main` |
| `upstream/PTOAS` | `dc15ee5b9e459c025eb4f714f2f892b535d93eb0` | `master` |

`upstream/PTOAS` 更新到新 `master` 后会显示：

```text
 M .codex/CLAUDE.md
?? 3rdparty/
```

第一项是已知 CRLF normalization；`3rdparty/` 是旧 `main` 删除两个 submodule 后留下的本地内容。它们均未被删除或重置，后续在控制仓/submodule 迁移时单独归档。

## 4. 上游 edge 快照与版本策略

2026-09-06 接手检查时，`git ls-remote` 得到：

| 仓库 | 远端默认分支/HEAD | 本地状态 |
|---|---|---|
| PyPTO | `master = 34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad` | 已对齐 edge HEAD |
| PyPTO-Gym | `master = 945a360e12592239a3549cb62d0db37af32bbc03` | 已对齐 edge HEAD |
| PTO-ISA | `master = 668248ec886447a83787200786fe6f461169b701` | 已对齐 edge HEAD |
| community PyPTO | `main = 9f657f37ed20ce148b46fb7229c267a152a0644e` | 已对齐 edge HEAD |
| PTOAS | 默认分支 `master`，HEAD `dc15ee5b9e459c025eb4f714f2f892b535d93eb0` | 已切换到 `master` 并对齐 edge HEAD |

用户已授权刷新，上表五仓已与远端默认分支 HEAD 一致。它们当前只是 `edge lock` 候选，不自动等于 CANN 可运行组合。

版本策略是“CANN release family 作兼容锚点 + 每仓 exact commit SHA 作真正锁定”：

- edge lock：用于上游差异审计和接口预研；
- stable lock：必须记录 CANN/PyPTO/PTO-ISA/PyPTO-Gym 的已验证组合；
- 禁止浮动跟随 `master/main`，也禁止只记录可变 tag 名。

### 4.1 2026-09-06 生态审计结论

- CANN 9.1.0 及以后的 CANN 包已集成 PyPTO；PyPTO 不是普通的独立 PyPI 依赖。
- classic `pypto` 路径连接 Tensor/Tile IR、AICPU/AICore、CANN runtime 和 Torch/TorchNPU；`pypto_pro` 则经 AST/IR/CCECodegen 生成 CCE C++，再由 Bisheng + PTO-ISA/CANN 编译与启动。
- `pypto_pro` 当前仍是 CCE/A5 导向，parser/backend/codegen/runtime 多层存在 Ascend 语义，不能只增加一个末端 adapter。
- PyPTO-Gym 主体仍使用 classic `pypto`：115 个 `pypto_tensor` 源文件中 114 个使用 classic 前端，Pro 实现文件只有 9 个。
- Qwen3.5 GDR 当前使用 `@pypto.frontend.jit`，且 9B 接入只替换 prefill chunk GDR，不是整网 Pro 编译。
- `pypto_pro` CCE backend 约有 301 个注册项；PTO-ISA manifest 含变体/通信共 150 项；这些数量不等于跨架构公共算子数。

三个首轮调研 agent 均未修改源码。`cann-ecosystem` 与 `release-policy` host smoke 为 PASS；`operator-ecosystem` 使用 PyPTO-Gym worktree，当时通用 smoke 因假定存在 `python/pypto_pro` 而 FAIL，这是 smoke 适用范围问题，不是 Gym 源码失败。该脚本已支持识别 `src/pypto_gym`，后续 `portability-gym` smoke 已 PASS。

算子可移植审计的主结论：classic `pypto` 的语义潜在可移植率为 89.0%（排除 distributed 后 96.8%）；Pro 301 个 CCE 注册项为 36.9%；Gym 83 个去重逻辑算子数学语义为 100%，但其中 88.0% 需复杂 target lowering。三者的当前生产 lowering/实现对非 Ascend target 的直接复用率均应视为约 0%。

## 5. 已完成的规划交付

- [README 总入口](README.md)
- [项目文件布局](docs/PROJECT_LAYOUT.zh-CN.md)
- [Worktree/subagent 计划](docs/WORKTREE_AGENT_PLAN.zh-CN.md)
- [资源矩阵](docs/RESOURCE_MATRIX.zh-CN.md)
- [本机重任务资源锁策略](docs/LOCAL_RESOURCE_POLICY.zh-CN.md)
- [一次无模型 smoke 规范](docs/SMOKE_TEST_SPEC.zh-CN.md)
- [机器可读任务配置](configs/agent_tasks.yaml)
- [上游 edge/stable 版本锁](configs/upstream_lock.yaml)
- [模型目标与精度配置](configs/model_targets.yaml)
- [安全创建 worktree](scripts/worktree/create.sh)
- [查看 worktree 状态](scripts/worktree/status.sh)
- [统一 smoke 入口](scripts/smoke/pypto_pro_smoke.sh)
- [AArch64/SVE 探针](scripts/smoke/aarch64_feature_probe.c)

脚本约束：

- worktree 只能建在项目同级的 `../worktrees/` 下，不覆盖已有路径；
- 不执行自动删除、reset 或 checkout；
- 明确拒绝 `--model`、`--model-name`、`--model-path`、`--weights`；
- smoke 使用目录锁保证同一日志目录只能执行一次；
- 设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；
- smoke 状态为 `PASS`、`BLOCKED_TOOLCHAIN`、`BLOCKED_DEVICE` 或 `FAIL`。

## 6. Subagent 的固定协议

只有用户明确说进入实现阶段后才使用。每个 subagent 启动消息必须包含：

```text
task_name=<唯一任务名>
worktree=<绝对路径>
branch=<分支名>
started_at=<ISO-8601 UTC 时间>
smoke_once=true
wait_timeout_seconds=3600
poll=false
```

约束：

- `spawn_agent` 时不传 `model`/`model_name`，使用默认模型；
- 一个 subagent 对应一个 worktree，不共享未完成生成物；
- 启动后只运行一次 `scripts/smoke/pypto_pro_smoke.sh`；
- 不下载、不加载、也不要求任何 LLM 模型；
- 父 agent 派发后只调用一次 `wait_agent(timeout_ms=3600000)`，不轮询；
- subagent 返回 commit SHA、修改路径、smoke 日志、测试和风险；
- 集成由主 Agent 在 integration worktree 完成。
- 命令较多的阶段验收也使用独立 subagent：从 exact integration HEAD 创建专用验收 worktree，源码只读，只写独占 `_meta` 证据目录；验收失败返回原实现 worktree 修复，不直接修改 integration。

W1/W2/W2B 七个 task、W3 的三个 task，以及 W4 `cpu-sve256`/`gpu-common` worktree 均已提交并保持 clean；W3、SVE256 与 GPU common 已按固定协议完成并冻结。实现冻结点与 task commit 见 `configs/development_lock.yaml`。

## 7. 建议的执行波次

用户批准后，先创建：

```text
port/pypto-x-integration → ../worktrees/pypto-x/integration
```

W1 已完成以下三个 task：

1. `work/target-abi`：`TargetSpec`、Capability、CompilerBackend、RuntimeBackend、Artifact/Tensor ABI。
2. `work/core-ir`：Parser 单次生成目标无关 CoreProgram、稳定 IR dump/serialization。
3. `work/verification`：无设备/无模型测试门禁、IR snapshot 和 differential harness。

W2/W2B、完整 W3、W4 SVE256/GPU common/CUDA C1–C2、W6 M0–M1H 与 AVX2/AVX-512 parity 已完成，下一步进入：

```text
W3 parity: AVX2/AVX-512 Qwen M1A–M1H（完成）
W4 CUDA: C2 correctness 已完成 → 后续性能/fusion
W5: hip
W6: 无权重完整 decoder connectivity → 获授权后 BF16 文本模型 → W8A8-linear
```

完整文件所有权和依赖见 `configs/agent_tasks.yaml`，不要跳过 GPU common 直接让 CUDA/HIP 各自定义 ABI。

## 8. 可用资源实况

### 本地开发机

```text
x86_64
12 vCPU
Clang 22.1.8
CMake
QEMU 11.0.3
aarch64-linux-gnu-gcc/g++
/usr/aarch64-linux-gnu sysroot
```

适用于 Core IR、ABI、CPU scalar/x86 和 AArch64/QEMU 功能验证。

本机 heavy 任务受跨项目全局锁约束。full pytest、大 shape lowering/compile、并行构建或预计使用至少 6/12 CPU、4 GiB 内存的任务，必须通过：

```bash
scripts/resource/run_local_heavy.sh --task <task> --agent <agent> -- <command> [args...]
```

包装器调用 `/home/chiro/projects/.resource-locks/resource-lock run local`，默认要求 8 GiB `MemAvailable`、保留 4 GiB，并限制到最多 6 CPU；cgroup 与 Python supervisor 同时监控任务。锁忙（75）或准入不足（69）时等待，禁止裸跑。详见 `docs/LOCAL_RESOURCE_POLICY.zh-CN.md`。

### RTX 5080

```text
SSH: 192.168.101.5
Host: Windows
WSL: Ubuntu 24.04.4 / WSL2
GPU: NVIDIA GeForce RTX 5080
VRAM: 16,303 MiB
Driver: 610.62
```

SSH 默认落入 Windows `cmd`，Linux 命令必须使用：

```bash
ssh 192.168.101.5 \
  'wsl.exe -e bash -lc "uname -a; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"'
```

用户于 2026-09-09 确认 GamePC 已恢复且 RTX 5080 由 PyPTO-X 独占。GPU-only probe/执行不申请 `gamepc`；CUDA host-heavy 编译、并行构建或大量远端 CPU/内存阶段才持续持锁。恢复探测确认：RTX 5080 16,303 MiB、compute capability 12.0、KMD 610.62/CUDA UMD 13.3；没有 compute process，后续瞬时探测可用 11,389 MiB；WSL 有 24 CPU、约 30 GiB `MemAvailable`、Clang 18.1.3、CMake 3.28.3、Python 3.12.3 和 `libcuda.so.1`。没有 `nvcc`、NVRTC、CUDART、CUDA SDK headers、PyTorch 或 Triton。首个 CUDA task 可先用 Driver API + PTX JIT + Python `ctypes`，无需安装；若后续需要 Toolkit，安装仍需用户明确授权。

### AMD 6750GRE 12G

卡暂时闲置、尚未接入当前系统。本地没有 `rocminfo`/`rocm-smi`。接入后先确认 ROCm 支持、实际 `gfx` target 和 wavefront，不要预先硬编码。

### 鲲鹏与 QEMU

已通过 `~/tools/ecs-920B` 创建按量鲲鹏 920B ECS，并完成 HWCAP/HWCAP2、实际 VL、编译器、原生 runner 与热点汇编验证。实例为 openEuler 22.03、2 vCPU、HiSilicon、KVM；报告 SVE=1、SVE2=0、VL=32 bytes。实例状态以 `~/tools/ecs-920B/state.env` 为准，删除前必须由用户确认。

无真机时本地 QEMU 已验证：

```text
QEMU_CPU=max,sve256=on
sve=1
sve2=1
vl_bytes=32
```

QEMU 只证明功能路径，不可用于性能结论。

## 9. 已验证结果

项目从旧路径迁移到 `/home/chiro/projects/pypto/pypto_x` 后已经验证：

- 五个嵌套仓库都能从新路径解析；
- community/PTOAS 的五个已初始化子模块正常；
- PyPTO worktree 元数据自动更新到新路径；
- 114 个本地 Markdown 链接全部有效；
- YAML 任务配置可解析；
- 三个 shell 脚本通过 `bash -n`；
- host 无模型 smoke：`PASS`；
- QEMU SVE256/SVE2 无模型 smoke：`PASS`；
- 模型参数拒绝测试：`PASS`；
- W1 三个 task smoke 均为 `PASS`，且每个只运行一次；
- W1 同进程联合单测：`23 passed`；
- W1 package discovery：`PASS`；
- W1 integration smoke：`PASS`，同时覆盖 `python/pypto` 与 `python/pypto_pro`；
- W2 task smoke：三个 task 均各执行一次且为 `PASS`；
- W2 同一进程全量单测：`66 passed`；
- W2 隔离链 `PIL → CoreProgram → Artifact → LaunchRequest → CPU scalar → differential`：`PASS`；
- W2 Python 3.7 AST（42 个新增 Python 文件）与 package discovery：`PASS`；
- W2 integration smoke：`PASS`；
- 普通源码 import 仍在 `pypto.frontend.__init__` 触发 native 依赖，此失败已复现并由 W2B `portable-bootstrap` 门禁处理；
- W2B 全量单测：`68 passed`；
- W2B `python -S` portable direct import/mode lock/`hasattr`/typed scalar E2E：`PASS`；
- W2B Python 3.7 AST（45 个新增 Python 文件）、package discovery 和 integration smoke：`PASS`；
- 未设置 `PYPTO_X_PORTABLE_ONLY=1` 时仍进入原 native loader，不自动静默降级；
- W3 vector-common task smoke：唯一一次 host smoke `PASS`；
- W3 vector-common 全量单测：`96 passed`；
- W3 vector-common Python 3.7 AST（49 个新增 Python 文件）、compileall、package discovery：`PASS`；
- W3 vector-common integration smoke：`PASS`；
- W3 AVX2 task smoke：唯一一次 host smoke `PASS`；
- W3 AVX2 全量单测：`137 passed`，包含统一 content-addressed artifact 目录的退出码 0 回归；
- W3 AVX2 Python 3.7 AST（累计 57 个新增 Python 文件）、compileall、package discovery：`PASS`；
- 真实 host probe：Clang 22.1.8、`x86_64-pc-linux-gnu`、AVX2/FMA/OSXSAVE/XGETBV/YMM：`PASS`；
- FMA/non-FMA 反汇编：YMM `386/396`、vfmadd `16/0`、ZMM `0/0`、EVEX `0/0`；
- FMA 矩形 matmul 与 scalar differential：`PASS`，最大绝对误差约 `8.18e-9`；
- W3 AVX2 integration smoke：`PASS`；
- W3 AVX-512 task smoke：唯一一次 host smoke `PASS`；
- W3 AVX-512 全量单测：`205 passed`，统一 AVX2/AVX-512 artifact 目录下进程退出码 0；
- W3 AVX-512 Python 3.7 AST（累计 63 个新增 Python 文件）、compileall、package discovery：`PASS`；
- 真实 AVX-512 capability：F/DQ/BW/VL/VNNI/BF16 与 OS ZMM/opmask state 均通过；
- base/FMA/BF16/VNNI artifact 均有 ZMM/opmask，分别门禁 vfmadd、vdpbf16、vpdp；
- AVX2 fallback artifact：YMM 396、ZMM 0、EVEX 0；
- 全负数 reduce-max、BF16 full/tail RNE、VNNI golden 与分层 fallback：`PASS`；
- W3 AVX-512 integration smoke：`PASS`；
- W4 SVE256 task smoke：唯一一次 QEMU smoke `PASS`；
- W4 SVE256 全量单测：`264 passed`，统一 x86/SVE artifact 目录下进程退出码 0；
- W4 SVE256 Python 3.7 AST（累计 73 个新增 Python 文件）、compileall、package discovery：`PASS`；
- QEMU probe：`emulated=true`、SVE/SVE2、VL=32 bytes；VL128/VL512 直接执行安全拒绝；
- SVE 热点：`whilelo=7`、`ld1w=14`、`st1w=7`、z/p predicate 寄存器，NEON v/q=0；
- guarded canary、全负 reduce-max、BF16 raw bits/零维与 scalar differential：`PASS`；
- W4 SVE256 integration smoke：`PASS`；
- 鲲鹏 920B ECS native probe：SVE=1、SVE2=0、VL=32、`emulated=false`、GCC 10.3.1；
- ECS 原生 FP32/BF16 19 元素 masked tail、guarded canary、全负 reduce-max、9×5×10 matmul：`PASS`；
- ECS 原生热点：`whilelo=2`、`ld1w=3`、`st1w=4`、z=36、p=19、NEON v/q=0；
- native ECS 首轮暴露的默认 cross sysroot 与 `cc` compiler-name 误拒已修复，并补入回归；
- GPU common task 唯一 smoke：`BLOCKED_DEVICE`，未重跑；其余无设备门禁继续执行；
- GPU common 专属测试：`45 passed`；覆盖 1–3D geometry、32/64 subgroup、uint64 overflow、zero-work、multi-kernel geometry sequence、严格 binding/barrier/address space、canonical artifact/tamper/cache key 与 FP32/BF16/reduce/matmul oracle；
- W4 当前 integration 全量：`312 passed in 55.07s`；
- Python 3.7 AST：累计 86 个变更 Python 文件；portable direct import 11 modules、package discovery 127 packages；
- GPU common integration host smoke：`PASS`；
- integration HEAD：`e00c12a498ac806bb8f51eb58b9603fb57bc82f7`。
- 本机 heavy wrapper 已验证全局 `local` 锁可取得/释放、CPU affinity 生效、启动准入不足返回 69、动态 cgroup 配额生效，并能记录 `MemAvailable`、任务树 RSS/CPU、load 与 PSI；256 MiB cgroup 下申请 400 MiB 的受控测试仅终止任务并以 137 退出，主机正常且锁自动回到 FREE；资源日志位于 `../worktrees/_meta/pypto-x/resource-usage/`。
- Qwen3.5-0.8B M0 task：`99 passed`，wheel/package-data、Python 3.9 AST、compileall、package discovery、diff/行宽均通过；task HEAD `bb776187a1e69897a97652b6317c6b6b44e802fd`；
- PyPTO-Gym Qwen integration：HEAD `b0c1e621c886ffc4d84bdd002a6bc8c8fd5e9628`，integration smoke `PASS`；
- 920B 标准库 shape harness：12 个 prefill/decode 场景 `PASS`，Python 3.9.9，RSS 约 14.8 MiB；未安装 torch/Transformers/NumPy，未访问权重；
- M0 冻结模型结构：hidden 1024、FFN 3584、24 层，其中 18 层 Gated DeltaNet、6 层 full attention；full attention 层为 3/7/11/15/19/23；
- M0 证据目录：`../worktrees/_meta/pypto-x/integration-w6-qwen35-m0-final/`。
- Qwen M1A task commit：`5f37d771a63a506a1cdac898f215649771bf97b2`；唯一 host smoke `PASS`；
- M1A 已支持 `cast/exp/rsqrt/sigmoid/silu/softplus/reduce_mean/broadcast/where` 的 Core/bridge alias、CPU scalar golden；vector/GPU 对尚未 lower 的新原语明确拒绝；
- M1A 主线审查修正了 `rsqrt` 的负数/零 NaN/Inf 语义、`exp` 溢出、PIL pure effect 和 `expand_clone → broadcast`；
- M1A integration 全量：`329 passed in 54.33s`；Python 3.7 AST 7 files、compileall、127-package discovery、integration smoke 均 `PASS`；
- M1A integration HEAD：`8a95d5c50c58f96a5cd458a65569a11a0be30a3e`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1a-final/`。
- Qwen M1B task commit：`51873c3ebef84a30561f879446b1af2693d5517c`；唯一 host smoke `PASS`；
- M1B 已支持 portable `reshape/view/transpose/contiguous/slice/split/concat/gather/embedding` 的 Core/bridge 与 CPU scalar golden；split 使用可序列化多输出 SSA；
- M1B 兼容性边界：portable `gather` 是 index-select/embedding 语义；上游 `pypto.gather` 是 gather-elements、上游 `pypto.view` 是 offset 局部视图，两者暂不映射，避免同名错算；
- M1B integration 全量：`349 passed in 57.24s`；Python 3.7 AST 5 files、compileall、127-package discovery、integration smoke 均 `PASS`；
- M1B integration HEAD：`8e840f11344a8ac9ec5368b69a5ce0e0a66281b9`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1b-final/`。
- Qwen M1C1 task commit：`8b4fa2100cb262cb8d57f5e35aaad22c19161232`；唯一 QEMU smoke `PASS`；
- M1C1 已把 M1A 9 组原语接入 vector-common/SVE256；AVX2/AVX-512 对未实现新原语继续显式拒绝；
- M1C1 integration 全量：`360 passed in 66.92s`；Python 3.7 AST 13 files、compileall、127-package discovery、integration QEMU smoke 均 `PASS`；
- 修正后 920B native：双向 FP32/BF16 cast、五个数学原语、reduce-mean、where、guarded canary 全部 `PASS`；
- 920B `ptx_rsqrt_f32` 热点：whilelo 1、ld1w 1、st1w 1、z 14、p 8、fsqrt 1、fdivr 1、NEON v/q 0；
- M1C1 显式 fallback：`exp/sigmoid/silu/softplus` 逐 lane libm；BF16 math/where 与 cast 的数值转换逐 lane；broadcast/where 输入由 host 预展开，不能宣称完整 native broadcast lowering；
- M1C1 integration HEAD：`556a1bb4c8b48ae52b0a7e52c56048b621890dc7`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c1-final/`。
- Qwen M1C2a task commits：`0ea1b3b23396f905ec3a31827a73482fb5635629`、`015506dcaa6a5410301a6723b9bc8789da65d1fd`、`3ca34cfd6562bd34bc2c3c4acbf4db31a8855a57`；唯一 QEMU smoke `PASS`；
- M1C2a 已把 `reshape/view/transpose/contiguous/slice` 接入 vector-common/SVE256，layout descriptor 为 O(rank)，不再保存逐元素 index map；wire v3 通过 byte access + `memcpy` 解码未对齐 descriptor；
- M1C2a integration 全量：`380 passed in 84.60s`；Python 3.7 AST 6 files、compileall、127-package discovery、integration QEMU smoke 均 `PASS`；
- 920B native：FP32 transpose/slice 使用 SVE u32 indexed gather，BF16 identity copy 使用 SVE u16，BF16 reorder 为明确 scalar fallback；奇数 payload、tamper、v1 layout 拒绝和 guarded canary 均 `PASS`；
- M1C2a integration HEAD：`202b045fba9b02b4982453eac42c648ecb49dbc9`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2a-final/`。
- Qwen M1C2b task commits：`d8df8637ffe0fe3b26d45c6512278cc8cf03a066`、`eea99d332172fd06dc581531606eed4ea613173c`；唯一合规 QEMU smoke `PASS`，早期失败 pytest 已单独归档为开发日志；
- M1C2b 已把 `split/concat/gather/embedding` 接入 vector-common/SVE256；dynamic indices 保持运行时 tensor，静态 descriptor 为 O(rank + segment count)，不保存逐元素 index map；
- M1C2b integration 全量：`393 passed in 116.36s`；另有 12 个多轴/零长度 scalar differential；Python 3.7 AST 6 files、compileall、127-package discovery、integration QEMU smoke 均 `PASS`；
- 920B native：split/concat FP32/BF16、gather/embedding int32/int64、bad index/hash/descriptor、wire v1/v3 compatibility、v4 pairing 和 canary 全部 `PASS`；`ptx_index_f32` 为 `whilelo1/ld1w2/st1w1/NEON0`；
- M1C2b 明确 fallback/限制：BF16 gather/embedding 逐 lane；FP32 gather offset 为 uint32；split 当前每个输出调用一次 runner 并重复发送 source；Python runtime 仍把 tensor 暂存为 list/wire bytes；
- M1C2b integration HEAD：`31a5c46b010a621e54b8dedaa81c97a98dfc4e3e`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2b-final/`。
- Qwen M1D task commits：`2e4c4d9cc285c5fc0ce90fe352d13e51620fd8ee`、`38307d402da6f2028427d7d5086b16a8caf8f2a1`；唯一 QEMU smoke `PASS`；
- M1D 新增 `pypto.portable.qwen35`，只组合既有 primitive，覆盖 Qwen delta-weight RMSNorm、加性 epsilon L2Norm、SwiGLU、stable Softmax、partial RoPE、runtime GQA repeat-KV 与 causal mask；没有新增 fused/target opcode；
- 固定 M0 metadata 已逐字段复核：hidden 1024、FFN 3584、attention/KV heads 8/2、head_dim 256、rotary_dim 64、24 层与 6 个 full-attention 层；
- M1D integration 全量：`417 passed in 147.15s`；Python 3.7 AST 3 files、compileall、128-package discovery、portable `python -S` import、integration QEMU smoke 均 `PASS`；
- 920B native：BF16 RMSNorm/Softmax/partial RoPE/GQA、decoder 与 attention 多操作分支、guarded canary 全部 `PASS`；组合仍走现有 runner primitive，不宣称 fused kernel 或性能；
- M1D integration HEAD：`fe6b54a0f40e739d5ebed87baa5d608565171695`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1d-final/`。
- Qwen M1E task commits：`be40610255a61f8c74f2e5306dc65d1cd8a8c1d4`、`8de5714a9b067d7e52a5f53453abae7a5ac5db5b`、`f63207e26dba6eb4f90ab04e74621288d49038c5`；唯一旧 QEMU smoke 保留且未重跑；
- M1E 新增 kernel-size=4 depthwise causal Conv1D，使用 `[B,C,T] + prior_state[B,C,4] + weight[C,4] -> output,new_state` 的函数式 SSA contract；BF16 存储、FP32 累加与 SiLU，prefill/chunk/decode 共用同一图，并连接 GDR in-projection → reshape → transpose → Conv1D；
- SVE256 lowering version 升至 v4，所有主/nested iteration domain 必须 compact；旧 v3 plan/artifact 明确拒绝并要求重编译；`[1,6144,1024]` 的 29 个 plan 不再物化逐元素 chunks，small/large ELF 均为 881,928 bytes；
- M1E integration 全量：`426 passed in 113.34s`；Python 3.7 AST 94 files、compileall、128 namespace-package discovery、diff check 与 integration QEMU smoke 均 `PASS`；full suite 经 `local` 锁运行，任务 RSS 峰值 278 MiB、最低 `MemAvailable` 26,627 MiB；
- 920B native 对 integration HEAD `41523cfc3341d47abb94a8e2d97111e2948ea501` 验收：SVE=1、SVE2=0、VL=32，concat/slice/cast/broadcast/mul/add/silu、guarded canary、Conv 输出与 `new_state` carry 均 `PASS`；未加载模型权重；
- M1E integration HEAD：`41523cfc3341d47abb94a8e2d97111e2948ea501`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1e-final/`。
- Qwen M1F task commits：`d4efa2401d80ecda724763525fbc72dadd98b2ef`、`829b74380389df27198d6c1ada0d0f385d2332ed`；integration commits：`f44e1a79ce3892ab66cce8ea3398b446043982b1`、`2d7da37b9ed20e3382232e31e45f584907c55e1e`；
- M1F 将 matmul 扩展为 batch prefix 完全相等的 rank-2/3/4 语义；QKᵀ 使用显式 transpose，不引入隐式 flag 或 NumPy batch broadcasting；
- M1F 新增 Qwen fixed 8 query heads/2 KV heads/head_dim 256 的 full-attention + functional KV graph：`prior_k/prior_v/current_k/current_v -> bf16 output,new_k,new_v`，QK/Softmax/PV 使用 FP32 中间值，causal mask、GQA indices、scale/fill 仍为显式 runtime 输入；
- vector plan version 升至 v2，SVE256 lowering version 升至 v5；旧 vector v1 和 SVE v3/v4 artifact 明确拒绝。SVE batched matmul 当前由 host 按 batch/head 调用 rank-2 native runner，AVX2/AVX-512 对 batched matmul 显式拒绝；
- 独立验收 worktree `verify/qwen35-m1f` 全量 `432 passed in 120.46s`；Python 3.7 AST 95 files、compileall 199 files、128 namespace packages、diff check 与唯一 QEMU smoke 均 `PASS`；
- 大 shape prefill `T=128,past=0` 与 decode `T=1,past=128` 全部 compact、chunks 为空，plan 约 19 KiB、ELF 881,928 bytes；920B native 的 guarded self-test、rank-4→rank-2 fallback、BF16 KV concat/state carry 与输入不变性均 `PASS`；
- M1F integration HEAD：`2d7da37b9ed20e3382232e31e45f584907c55e1e`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1f-final/`。
- Qwen M1G task commits：`549e2f098d12fd357458f11b0ae16248fb4aae2c`、`a7c6a05f3b26bf461689da0483194f2c0ff16807`；integration commits：`d39e0a5be52c9a2d17ace0cd211cb93c66731140`、`962f422e18ee25fd37679687710edbd496b9f6ce`；
- M1G 固定 Qwen GDR contract：q/k/v/beta 为 BF16，g/state/scale 为 FP32，state `[B,16,128,128]` 使用 `[B,H,K,V]` 布局；beta 的 BF16 contract 来自 Qwen-specific `gdr_fwd`，不沿用其他 KDA 的 FP32 beta；
- 每 token 函数式更新为 `S_decay=exp(g)*S`、`predicted=k@S_decay`、`S_new=S_decay+(beta*k)^T@(v-predicted)`、`output=scale*q@S_new`；T=1 decode 与 T>1 chunk 使用同一静态 SSA 展开图；
- 独立验收 worktree `verify/qwen35-m1g` 全量 `438 passed in 124.37s`；Python 3.7 AST 96 files、compileall 199 files、128 namespace packages、diff check 与唯一 QEMU smoke 均 `PASS`；
- 固定 Qwen 16×128 非零 scalar/SVE differential 与 T=2 执行通过；T=64 编译得到 1,923 plans/2,307 iteration domains，全部 compact、chunks 为空，plan 约 1.87 MiB、ELF 881,928 bytes；
- 920B native 以 48 次 rank-2 runner 调用验证非零 recurrent step，prediction/state/output 最大绝对误差约 `2.70e-8`/`5.04e-11`/`8.29e-9`，guarded canary 与输入不变性 `PASS`；这是功能证据，不是性能结论；
- M1G integration HEAD：`962f422e18ee25fd37679687710edbd496b9f6ce`，证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1g-final/`。
- Qwen M1H task commits：`688c1a73ce5abf854791be28590f22eefb12b7cb`、`2e8ae4e1b768e275744fc34cc0fb21b85b9b49c1`；integration commits：`564e643ab`、`620d6f794`；
- M1H 独立验收：`455 passed`；compare 全谓词/广播/NaN/bool、int32/int64 iota、position prefill/decode、大 shape compact plan、旧 artifact 拒绝与 920B contract probe 均 `PASS`；SVE `iota/compare` 明确为 `host_reference`，24 层 manifest 只执行 identity shell；证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1h-final/`；
- CUDA C1 task commits：`272d624d51685394df371ff42e6e552f5082b208`、`167bd00c75a051de33fbd9dba18e10ab8145a0bd`；integration commits：`7eff3a056`、`2c99c4b31`；
- CUDA C1 独立验收：`470 passed`；RTX 5080 的 Driver API + PTX JIT 执行 FP32/BF16 elementwise、BF16 RNE、reduction、all-negative max、rank-2 matmul、K=0、zero-work、canary、多操作链、empty sum 和重复 launch/cleanup 均 `PASS`，前后无残留 compute process；唯一 smoke 因无 `nvcc` 按规范为 `BLOCKED_TOOLCHAIN`，不影响不依赖 Toolkit 的实测路径；证据目录 `../worktrees/_meta/pypto-x/integration-w4-cuda-c1-final/`；
- AVX2 parity task commit：`eb7e78f68fc87e92127f3e4d5b9de29478064883`；integration commit：`c0b0abead252b5d21b0dca0bc8376d1f77949b00`；独立验收 `474 passed`、focused `63 passed`，CPUID/OSXSAVE/XGETBV、YMM、FMA/non-FMA、无 ZMM/EVEX、20 个 composites、Conv/GDR/position 均 `PASS`；证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-avx2-parity-final/`；
- CUDA C2 task commit：`6f2992f91f30f565222440896a1cd12fa6290096`；integration commit：`5b0474f0db26d611526070dc4298170672e264af`；独立验收 `476 passed, 7 skipped`，GPU common vendor/ABI `45 passed`、CPU 联合回归 `260 passed`，5080 math/position/layout/indexing/batched matmul/scalar SSA/Qwen composites 与 OOB/cleanup 均 `PASS`；证据目录 `../worktrees/_meta/pypto-x/integration-w4-cuda-c2-final/`；
- AVX-512 parity task commit：`76325e3b863a4695c882c48932927d3c216c5e91`；integration commit：`63c9a4fdd6c1282aa8226b157d81dccbea6b6f21`；独立验收 `483 passed, 7 skipped`、focused `76 passed`，CPUID/OSXSAVE/XGETBV、ZMM/opmask/FMA/BF16/VNNI、rank-3/4 matmul、非零 GDR 与 position 大 shape 均 `PASS`；证据目录 `../worktrees/_meta/pypto-x/integration-w6-qwen35-avx512-parity-final/`；
- 当前 integration HEAD：`63c9a4fdd6c1282aa8226b157d81dccbea6b6f21`，工作树 clean。GPU 由 PyPTO-X 独占，`gamepc` 锁仅协调远端 heavy CPU/host-memory。

W1/W2/W2B/W3 smoke 日志位于 `../worktrees/_meta/pypto-x/`；任务日志保持只读，不重跑覆盖。

## 10. 许可证状态

维护方已经向用户表达愿意看到跨平台移植，许可证未来可能调整。这是积极信号，但本地快照的 CANN License 2.0 第 2.1/3.1 条仍限制在华为 AI 处理器/软件场景。

执行建议：

- 可以继续接口 RFC、独立设计和经授权的技术 PoC；
- 对外发布或合并非华为处理器衍生后端前，取得新许可证、双许可证或明确书面例外；
- 最理想的是把 frontend/Core IR/Target ABI 放在宽松许可证下，Ascend-specific 目录单独保留 CANN 条款。

这不是法律意见，正式发布由代码所有者和法务确认。

## 11. 新 Agent 首次回复前必须确认的事项

已对齐事项：

1. `PyPTO-X` 作为内部项目名，保留 `pypto` Tensor 与 `pypto_pro` Professional 两套公共 API。
2. worktree 默认放到项目同级的 `../worktrees/`。
3. CPU/加速器优先级为 AVX2 → AVX-512 → SVE256（无 NEON）→ NVIDIA → AMD。
4. 上游默认分支已刷新为 edge 快照，版本策略为 release family + exact SHA 双轨 lock。
5. Tensor frontend 作为跨架构主入口，Pro 作为 Ascend expert dialect + portable subset。
6. 用户已批准执行开发计划；W1/W2/W2B、完整 W3、SVE256、GPU common、CUDA C1–C2、Qwen3.5-0.8B M0–M1H 与 AVX2/AVX-512 parity 已完成。下一任务是无权重完整 decoder connectivity。5080 GPU-only 工作持续可用，host-heavy 阶段单独申请 `gamepc`。

C0 已完成：

1. 根目录控制 Git 仓与五个 upstream submodule 已正式登记，现有 linked worktree 未被重建。
2. edge SHA 已再次与远端默认分支核对一致。
3. 接手文档已按序号和日期归档到 `docs/00-handoffs/`。
4. stable lock 仍等待实际 CANN toolkit/NPU 环境做晋升验证，不影响目标无关 W1，但会门禁 Ascend 回归结论。

GPU common 已冻结 NVIDIA/AMD 共用的 Grid/Workgroup/Thread/Subgroup、address space、GPU artifact 与 launch ABI；公共层没有 NVVM/ROCDL/WMMA/MFMA 语义。CUDA C2 已用 `ctypes + libcuda.so.1 + PTX JIT` 在 5080 完成 Qwen math/layout/indexing/composites 正确性验收；它仍不是 NVVM、Tensor Core、fusion 或性能结论。PyPTO-Gym M0、PyPTO M1A–M1H 与 AVX2/AVX-512 parity 已完成并集成；下一步连接无权重 decoder connectivity，并在取得权重/参考环境授权后进入 BF16 实际模型与 W8A8-linear。M1G 仍是静态 sequential correctness graph，不是 fused WY/chunk kernel；当前 M1H decoder 仍是无权重 identity shell，SVE position/control 仍有 host fallback。鲲鹏 920B ECS 当前 ACTIVE，用于 native 正确性和汇编验收；实际权重下载/加载仍未授权。Ascend 当前只完成 adapter seam；stable CANN/NPU 回归继续保持 pending。

## 12. 快速自检命令

```bash
cd /home/chiro/projects/pypto/pypto_x

git -C upstream/pypto status --short
git -C upstream/pypto rev-parse HEAD
scripts/worktree/status.sh

bash -n \
  scripts/worktree/create.sh \
  scripts/worktree/status.sh \
  scripts/smoke/pypto_pro_smoke.sh
```

除非用户明确要求，本次接手自检不需要重新运行所有 smoke，也不要下载依赖或模型。
