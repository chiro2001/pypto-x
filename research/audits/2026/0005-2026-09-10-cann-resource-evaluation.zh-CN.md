# CANN 安装与仿真资源需求评估

归档序号：`0005`

评估日期：2026-09-10（Asia/Shanghai）

结论状态：`CANN_RESOURCE_EVALUATED_PROBE_RECOMMENDED`

本报告回答：要让 PyPTO-X 的 Ascend 线具备"可编译 + 可仿真（不等 NPU 硬件）"的环境，需要多少资源、放在哪台机器、还缺什么。所有数字都标注了来源；未实测的按估算标注。

## 1. 结论速览

```text
磁盘        CANN 9.1.0 官方 amd64 容器镜像压缩 4.10-4.65 GiB（实测 25 个 tag）
            安装后按 2-2.5x 估算约 10-12 GB；本机 86 GB 可用、GamePC WSL 865 GB 可用 → 磁盘不是瓶颈
CPU/内存    真正的门槛。上游 PyPTO skill 明确 CAModel 仿真需 CPU >16 核、内存 >32 GB，
            且比真机慢 100-1000x。本机 12 vCPU / 29 GiB（可用 23 GiB）不达标
机器        GamePC 硬件达标（12C/24T、61.4 GiB 物理内存、865 GB 磁盘），但 WSL 默认只分到 30 GiB
            920B ECS（2 vCPU / 2.5 GiB / 34 GB）完全不够，只能做 aarch64 交叉编译
建议        先用本机已有 docker/podman 做一次最小 cannsim 探针（拉镜像约 4.4 GiB）；
            真要跑 CAModel 再决定给 GamePC WSL 扩内存（需 wsl --shutdown）或租 16C64G ECS
```

## 2. 实测数据

### 2.1 CANN 容器镜像尺寸（Docker Hub `ascendai/cann`，2026-09-10 抽样）

```text
tag                                        amd64 压缩大小
9.1.0-beta.3-910-openeuler24.03-py3.12     4.62 GiB
9.1.0-950-ubuntu22.04-py3.10-devel         4.52 GiB
9.1.0-950-ubuntu22.04-py3.10               4.42 GiB
9.1.0-950-openeuler24.03-py3.10            4.55 GiB
9.1.0-950-ubuntu22.04-py3.11               4.46 GiB
9.1.0-950-openeuler24.03-py3.12            4.59 GiB
9.1.0-310p-ubuntu22.04-py3.11              4.10 GiB
9.1.0-910-openeuler24.03-py3.10            4.32 GiB
9.1.0-a3-ubuntu22.04-py3.11                4.31 GiB
```

`950` 对应 Ascend 950（A5）世代、Ubuntu 22.04 / openEuler 24.03、Python 3.10/3.11/3.12 都有；`devel` 比运行版大约 0.1 GiB。

### 2.2 候选机器

| 机器 | CPU | 内存 | 可用磁盘 | 判定 |
|---|---|---|---|---|
| 本机开发机（KVM guest） | 12 vCPU（无 cpufreq） | 29 GiB，可用 ~23 GiB | 86 GB | 磁盘够；**CAModel 不达标**，适合最小探针/CostModel/拉镜像 |
| GamePC（Windows + WSL2） | 12C / 24T | 宿主 61.4 GiB；WSL 当前 MemTotal 30.2 GiB | 865 GB | **唯一硬件达标**；需把 `.wslconfig` 内存提到 ~48 GB（会 `wsl --shutdown`）+ 装 docker 或直接 `.run` 安装 |
| 鲲鹏 920B ECS | 2 vCPU | 2.5 GiB | 34 GB | 不能仿真；仅可做 aarch64 交叉编译/打包 |
| 新租 ECS（x86_64 16 vCPU / 64 GB 级） | 达标 | 达标 | — | 若不想动 GamePC，最省事；价格以华为云价目为准 |

### 2.3 上游仿真资源要求（快照原文）

`upstream/pypto/.agents/skills/pypto-simulation/SKILL.md`：

```text
仿真速度比真实 NPU 慢 100-1000 倍
需充足资源：CPU >16核，内存 >32GB
生成 Chrome Tracing 流水图
```

`upstream/pypto/docs/zh/guide/programming_guide/tensor/debug/debug.md:196`：Ascend 950PR/950DT 支持两种仿真，均通过 `cannsim record` 启动——**CostModel**（任务级、粗粒度、快，出泳道图）与 **CAModel**（指令级、精度高但慢，出 `trace_core*.json`）。

## 3. 安装形态与依赖（快照证据）

- **装法**：`.run` 本地安装（root 装到 `/usr/local/Ascend/cann`，文档中另有 `ascend-toolkit/latest` 路径）、yum/apt 在线安装、官方容器镜像。
- **OS 支持面**（官方安装文档页抓取片段）：Ubuntu、openEuler、CentOS、Kylin、BCLinux、UOS V20、AntOS、AliOS；镜像侧提供 Ubuntu 22.04 与 openEuler 24.03，Python 3.10/3.11/3.12（文档仍保留 Python3.7 说明，源码构建支持 cp37 ABI）。
- **PyPTO 集成**：`upstream/pypto/docs/zh/install/build_and_install.md` 明确 **CANN 9.1.0 及之后已在包内集成 PyPTO**，装完即可用；CANN 8.5.0 对应 PyPTO 0.1.2、9.0.0 对应 0.2.0，需按版本配源码。
- **源码构建 PyPTO（仅 <9.1.0 或需要 master 时）**：`python3 build_ci.py --clean --no_isolation` → `cann-pypto_<ver>_linux-<arch>.run` → `bash ./cann-pypto_*.run --full -q --pylocal`。**并行度按 CPU 核数与可用内存（含 cgroup 上限）自动计算**，可用 `-j` / `PYPTO_BUILD_JOB_NUM` 覆盖 → 构建本身是内存敏感项。
- **仿真入口与依赖**：`cannsim record '<cmd>' -s Ascend950 -n 0 -g -o output/`，再 `cannsim report -e <dir> -o <dir>/report --core-id all`；精度仿真要求 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`（`ASCEND_HOME_PATH`）与精度仿真组件（如 `libpem_davinci.so`），并依赖可用的 `llvm-objcopy`（见 `docs/zh/guide/appendix/trouble_shooting/simulation.md` 的 F94001–F94003）。
- **PTO-ISA 侧对照**：`upstream/pto-isa/README.md:61-62` —— CPU 路径只需 Python + CMake + C++20 编译器（我们已在跑 CPU_SIM 基线）；**NPU/仿真路径需要 Linux + Ascend CANN toolkit**。
- **版本锚点**：`configs/upstream_lock.yaml` 的 `release_family=v9.2.0-beta.2`，`promotion_status=pending_cann_toolkit_and_npu_validation`（缺的正是 CANN toolkit 维度）。

## 4. 两档仿真的资源画像

| 档位 | 粒度 | 产出 | 资源 | 用途 |
|---|---|---|---|---|
| CostModel | 任务级 | `merged_swimlane.json` 泳道图 | 低（普通机器可跑小算子） | 调度与核间并行度快评 |
| CAModel | 指令级 | `trace_core*.json`（chrome://tracing） | **>16 核 / >32 GB**；比真机慢 100–1000× | 核内流水与精细调优 |

按 Qwen3.5-0.8B 这种规模预算时间时，请以"真实执行时间 × 100–1000"量级估计，并把 64 GB 内存作为更稳的配置。

## 5. 除资源之外的前置

1. **账号与协议**：CANN 社区版下载需要昇腾/华为账号并接受 EULA；商用版走支持渠道。
2. **许可证**：CANN OSL 2.0 仍限制非华为处理器用途。我们做的是 Ascend 目标的仿真/编译（华为目标用途），但**衍生后端对外发布仍需许可证澄清**（与接手文档第 10 节一致）。
3. **工程链**：CANN 只提供"能编译/能仿真 Ascend 目标"；PyPTO-X 的 Ascend 线目前只是 adapter seam（模块头明确不 import CANN，靠 `PYPTO_X_ASCEND_HOOKS` 注入）。要在仿真里跑**我们自己的图**，还需要 IR→CCE/PTO 的生成链与 hooks 实现——这是新的波次，不在本评估范围内。
4. **网络**：本机经代理可访问 `www.hiascend.com` 文档页（HTTP 200）；文档 PDF 直链返回 403，镜像拉取建议走代理。

## 6. 建议路径

```text
阶段 A（低成本探针，建议先做）
  本机 podman/docker 拉 ascendai/cann:9.1.0-950-ubuntu22.04-py3.10（4.42 GiB）
  跑一次 cannsim 最小用例/CostModel，记录真实磁盘/内存/耗时；经 heavy runner，预留约 20 GB
阶段 B（真跑 CAModel）
  GamePC WSL 扩内存到 ~48 GB（需 wsl --shutdown，用户批准）或租 16 vCPU/64 GB ECS
阶段 C（跑 PyPTO-X 自己的图）
  先补 IR→CCE/PTO 生成链与 Ascend hooks，再谈仿真验收
```

阶段 A 不改变任何现有环境，失败也只是浪费一次镜像拉取；阶段 B/C 需要用户决策与额外授权。
