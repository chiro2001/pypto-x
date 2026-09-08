# PyPTO-X 接手文档

状态：`EXECUTION_W4_GPU_COMMON_IN_PROGRESS`

最后更新：2026-09-08 13:56 CST（Asia/Shanghai）

项目根目录：`/home/chiro/projects/pypto/pypto_x`

## 1. 接手摘要

本项目工作名为 **PyPTO-X（PyPTO Cross-Architecture，PyPTO 跨架构后端）**。目标是以官方 PyPTO 同仓的 Tensor/Professional 双前端为研究基线，把可移植语义与 Ascend 专属 dialect 分层，并逐步支持：

- 鲲鹏/AArch64 CPU：scalar reference、SVE256/SVE2（不开发 NEON 优化后端）；
- x86_64 CPU：scalar、AVX2、AVX-512，后续 AMX；
- NVIDIA GPU：GPU 公共层 → CUDA/NVVM；
- AMD GPU：GPU 公共层 → HIP/ROCDL；
- 现有 Ascend CCE 路径保持为一个 target plugin，并避免功能回退。

目前已完成调研、架构规划、项目整理、资源盘点、上游 edge 刷新、CANN/算子生态审计、C0 控制仓治理、W1/W2/W2B、完整 W3，以及 W4 SVE256。integration 已包含 Core IR、Target/Compiler/Runtime ABI、CPU scalar/vector common、Clang x86 compiler/runtime 和 AArch64 SVE256 compiler/QEMU runtime；GPU 尚未实现。SVE 当前只有 QEMU 功能与汇编证据，没有鲲鹏真机结论。Tensor frontend 是跨架构主入口，Pro 保留为 Ascend expert dialect 并后续提取 portable subset。

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
linked worktree 数 = 19（基线、integration、五个只读调研、三个 W1、三个 W2、一个 W2B、三个 W3 和两个 W4 task worktree）
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

W1/W2/W2B 七个 task、W3 的三个 task 与 W4 `cpu-sve256` worktree 均已提交并保持 clean；W3 和 SVE256 已按固定协议完成并冻结，`gpu-common` 已从该冻结点创建并进入执行。实现冻结点与 task commit 见 `configs/development_lock.yaml`。

## 7. 建议的执行波次

用户批准后，先创建：

```text
port/pypto-x-integration → ../worktrees/pypto-x/integration
```

W1 已完成以下三个 task：

1. `work/target-abi`：`TargetSpec`、Capability、CompilerBackend、RuntimeBackend、Artifact/Tensor ABI。
2. `work/core-ir`：Parser 单次生成目标无关 CoreProgram、稳定 IR dump/serialization。
3. `work/verification`：无设备/无模型测试门禁、IR snapshot 和 differential harness。

W2/W2B、完整 W3 与 W4 SVE256 已完成，下一步进入：

```text
W3: cpu-vector-common → cpu-avx2 → cpu-avx512
W4: cpu-sve256 → gpu-common → cuda
W5: hip
W6: Qwen3.5-0.8B BF16 文本模型 → W8A8-linear
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

当前 WSL 可见 GPU、Git、CMake、Clang 和 Python，但没有 `nvcc`，Python 也没有 PyTorch。CUDA 工作开始前需要用户授权安装/配置工具链；smoke 本身不得下载模型。

### AMD 6750GRE 12G

卡暂时闲置、尚未接入当前系统。本地没有 `rocminfo`/`rocm-smi`。接入后先确认 ROCm 支持、实际 `gfx` target 和 wavefront，不要预先硬编码。

### 鲲鹏与 QEMU

鲲鹏 920B/SVE256 机器尚未借到。拿到后先做 HWCAP/HWCAP2、实际 VL、NUMA 和编译器探测。

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
- integration HEAD：`5e42eed7c432034db02c2d339e67789afd419ae9`。

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
6. 用户已批准执行开发计划；W1/W2/W2B、完整 W3 与 SVE256 已完成，可按同一协议继续 GPU common。

C0 已完成：

1. 根目录控制 Git 仓与五个 upstream submodule 已正式登记，现有 linked worktree 未被重建。
2. edge SHA 已再次与远端默认分支核对一致。
3. 接手文档已按序号和日期归档到 `docs/00-handoffs/`。
4. stable lock 仍等待实际 CANN toolkit/NPU 环境做晋升验证，不影响目标无关 W1，但会门禁 Ascend 回归结论。

当前在 `gpu-common` 独立 worktree 冻结 NVIDIA/AMD 共用的 Grid/Block/Thread/Subgroup、address space、GPU artifact 与 launch ABI；不得提前引入 NVVM/ROCDL vendor 语义。完成后再在 RTX 5080 上推进 CUDA。Ascend 当前只完成 adapter seam；stable CANN/NPU 回归继续保持 pending。

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
