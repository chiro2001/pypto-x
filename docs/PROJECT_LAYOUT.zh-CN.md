# PyPTO-X（PyPTO Cross-Architecture）项目文件布局

更新日期：2026-09-09（Asia/Shanghai）

## 目标

PyPTO-X 的目标是让 PyPTO_PRO 兼容鲲鹏 CPU、x86_64 CPU、NVIDIA GPU 和 AMD GPU，同时保留 `pypto_pro` Kernel API 的兼容性。本目录分成三个职责域：

项目命名约定：

- 项目/仓库工作名：`PyPTO-X` / `pypto-x`。
- 完整英文名：`PyPTO Cross-Architecture`；中文名：`PyPTO 跨架构后端`。
- 公共 Python 导入名初期继续使用 `pypto_pro`，避免破坏已有 Kernel；`PyPTO-X` 是项目和后端集合名称，不是立即替换的 Python 包名。
- 后端命名统一为 `ascend`、`cpu`、`cuda`、`hip`；CPU 变体使用 `aarch64-sve256`、`aarch64-sve2`、`x86-avx2`、`x86-avx512` 等能力名。

1. `upstream/` 保存不可直接修改的上游源码快照和资料依赖。
2. 项目同级的 `../worktrees/` 保存从 `upstream/pypto` 创建的实现分支，每个 subagent 一个独立 worktree。
3. `docs/`、`configs/`、`scripts/` 保存项目设计、资源、测试和协作工具。

根目录是轻量“控制仓”；五个 `upstream/*` 源码仓以固定 commit 的 submodule/gitlink 登记，不把约 600 MiB 的上游源码复制进控制仓历史。代码 worktree 仍以 `upstream/pypto` Git 仓库为主仓。

## 当前控制目录

```text
.
├── AGENTS.md                           # 新 Agent 的工作区规则和接手顺序
├── HANDOFF.zh-CN.md                    # 当前状态、决策、资源和下一步
├── README.md
├── .gitignore
├── configs/
│   ├── agent_tasks.yaml                 # 任务、依赖、资源和 subagent 启动约定
│   ├── model_targets.yaml               # Qwen3.5-0.8B BF16/W8A8 目标
│   └── upstream_lock.yaml               # edge/stable 上游版本锁
├── docs/
│   ├── 00-handoffs/                      # 按序号/日期保存的阶段交接快照
│   ├── 10-architecture/                  # 已评审的架构与接口 RFC
│   ├── PROJECT_LAYOUT.zh-CN.md          # 本文件
│   ├── WORKTREE_AGENT_PLAN.zh-CN.md     # worktree 拓扑和并行开发协议
│   ├── RESOURCE_MATRIX.zh-CN.md         # 机器、GPU、QEMU 和能力矩阵
│   ├── SMOKE_TEST_SPEC.zh-CN.md         # 无模型一次冒烟测试规范
│   └── 20-planning/
│       └── 0001-2026-09-07-qwen35-08b-bf16-w8a8-mvp.zh-CN.md
├── references/                          # 外部讲稿和活动页离线副本
├── research/                            # PyPTO 调研和源码清单
├── scripts/
│   ├── resource/
│   │   ├── run_local_heavy.sh            # 申请全局 local 锁、建立 cgroup 并启动监控
│   │   └── monitor_local_heavy.py         # MemAvailable/RSS/CPU/load/PSI 监督与安全停止
│   ├── smoke/
│   │   ├── pypto_pro_smoke.sh           # 统一冒烟测试入口
│   │   └── aarch64_feature_probe.c      # QEMU/native SVE 能力探针
│   └── worktree/
│       ├── create.sh                    # 安全创建单个 linked worktree
│       └── status.sh                    # 查看 worktree 和分支状态
└── upstream/                            # 上游 Git 仓库；原则上只读
    ├── pypto/
    ├── pypto-gym/
    ├── pto-isa/
    ├── pypto-community/
    └── PTOAS/

../worktrees/                             # 控制仓之外，由脚本按需创建
├── pypto-x/
│   ├── integration/
│   └── <task-name>/
├── pypto-gym/
└── _meta/
```

## PyPTO-X 源码内的目标架构

以下是建议的逻辑模块树。第一阶段不要求一次搬动所有现有文件；可以先增加 facade/adapter，再逐步迁移实现。

```text
python/pypto/                            [Tensor frontend：跨架构主入口]
├── frontend/                           [现有 parser；新增 core_export adapter]
├── core_ir/                            [新增：目标无关 Core IR facade]
│   ├── program.py
│   ├── types.py
│   ├── effects.py
│   ├── regions.py
│   └── serialization.py                [稳定 IR 快照]
├── target/                              [新增]
│   ├── spec.py                          # triple、device kind、feature、执行模型
│   ├── capabilities.py                  # dtype/op/vector/matrix/async-copy 能力
│   ├── registry.py                      # target/backend 工厂
│   └── cache_key.py                     # target + feature + ABI + shape specialization
├── lowering/                            [新增]
│   ├── core_pipeline.py                 # 目标无关 canonicalize/DCE/shape/fusion
│   ├── target_pipeline.py               # 目标 Pass 编排
│   ├── ascend/                          # MemorySpace/Pipe/AIC/AIV/CCE lowering
│   ├── cpu/                             # loop/vector/cache lowering
│   └── gpu/                             # grid/block/subgroup/address-space lowering
├── compiler/                            [新增]
│   ├── api.py                           # CompilerBackend
│   ├── artifact.py                      # Artifact、入口和 workspace 元数据
│   ├── cache.py
│   └── targets/{ascend,cpu,cuda,hip}.py
├── abi/                                 [新增：TensorDesc/LaunchRequest/RuntimeBackend]
└── backends/{ascend,cpu,cuda,hip}/      [新增：各 target runtime/compiler adapter]

python/pypto_pro/                        [Professional：Ascend expert dialect]
├── language/、ir/                       [现有 CCE/A5 语义；不作为公共 Core IR]
└── runtime/                             [后续通过 Ascend adapter 接入统一 ABI]
```

对应的 C++ 逻辑模块：

```text
framework/include/ir/                   [现有 IR；抽出目标无关子集]
framework/include/pypto_pro/
├── target/                              [新增 TargetSpec/Registry/Capability]
├── lowering/                            [新增 Core → Target]
├── backend/
│   ├── common/                          [改造：只保留真正通用的注册接口]
│   ├── ascend/                          [现有 backend_cce、SoC、CANN runtime]
│   ├── cpu/                             [新增]
│   └── gpu/                             [新增 GPU 公共 backend]
├── codegen/
│   ├── codegen_base.h                   [稳定过渡接口]
│   ├── ascend/cce/                      [现有 CCECodegen]
│   ├── cpu/                             [新增 LLVM/C++ codegen]
│   └── gpu/{common,nvvm,rocdl}/         [新增/后续]
└── runtime/                             [新增 Tensor ABI、Artifact、RuntimeBackend]
```

## 目录归属规则

### `upstream/`

- 只用于读取、对照和固定版本。
- 不在 `master`/`main` 上直接开发。
- 上游更新由协调者统一 fetch，并记录 commit；subagent 不自行改变基线。

### `../worktrees/`

- 每个 task 一个目录、一个分支、一个 subagent。
- 每个 task 使用独立 build/log/artifact 根目录，禁止共享未完成的生成文件。
- task 完成后先合并，再由协调者决定何时清理 worktree；不自动删除。

### `scripts/` 与 `configs/`

- 脚本只做可重复、可审计的创建/检查动作。
- 不在脚本中下载模型、不写入用户凭据、不执行递归删除。
- 任务的 `started_at`、worktree、branch、smoke 结果和等待策略由配置/日志记录。

## 许可证边界

建议将未来许可证拆分边界与模块边界对齐：

```text
pypto_pro/core、target ABI、CPU/GPU compiler/runtime adapter  → 争取宽松许可证
pypto_pro/targets/ascend、CCE/PTO/CANN adapter                → 保留或单独处理 CANN 许可证
```

这是工程组织建议，不替代代码所有者和法务的正式判断。
