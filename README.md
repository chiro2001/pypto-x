# PyPTO-X：PyPTO 跨架构后端

PyPTO-X（PyPTO Cross-Architecture）是本项目的工作名称，目标是在保留 PyPTO Tensor/Professional 双前端定位的前提下，抽取可移植 Core IR 与 target ABI，并逐步支持鲲鹏 CPU、x86_64 CPU、NVIDIA GPU 和 AMD GPU。当前已冻结 CPU scalar、SVE256、GPU common、Qwen3.5-0.8B M0–M1H、AVX2/AVX-512 Qwen parity，以及 RTX 5080 CUDA C2 math/layout/indexing/composites；下一阶段把24层 identity shell 升级为无权重、非 identity 的完整 decoder connectivity。各后端能力只以对应验收证据为准。

## 快速入口

- [新 Agent 接手文档](HANDOFF.zh-CN.md)：当前状态、冻结决策、远端漂移、资源、待确认事项和下一步。
- [交接档案索引](docs/00-handoffs/README.md)：按序号和日期保存阶段交接快照。
- [W1 公共接口 RFC](docs/10-architecture/0001-2026-09-07-w1-core-target-contract.zh-CN.md)：Tensor-first Core IR、Target/Compiler/Runtime ABI 的共同契约。
- [W2 bridge/scalar/Ascend 契约](docs/10-architecture/0002-2026-09-07-w2-bridge-scalar-ascend-contract.zh-CN.md)：Tensor bridge、CPU scalar reference 和 Ascend adapter 的验收边界。
- [portable bootstrap 契约](docs/10-architecture/0003-2026-09-07-portable-bootstrap-contract.zh-CN.md)：无 native 环境的显式轻量导入模式及默认兼容门禁。
- [W3 x86 vector 契约](docs/10-architecture/0004-2026-09-07-w3-x86-vector-contract.zh-CN.md)：CPU vector common、AVX2/AVX-512 能力探测、尾块与汇编验收。
- [早期研究报告](research/PYPTO_PORTING_RESEARCH.zh-CN.md)：源码结构、可复用边界和方案比较；其中早期路线已由接手文档/RFC 的冻结决策取代。
- [PyPTO 生态审计](research/audits/2026/0001-2026-09-06-pypto-ecosystem.zh-CN.md)：CANN 融合链路、classic/Pro 算子面、Gym/GDR 实况和版本锁定策略。
- [PyPTO/Pro 关系审计](research/audits/2026/0002-2026-09-06-pypto-pypto-pro-relationship.zh-CN.md)：两种编程模式的共享基础、独立链路和 PyPTO-X 前端决策影响。
- [算子可移植占比审计](research/audits/2026/0003-2026-09-07-operator-portability-ratio.zh-CN.md)：classic、Pro 和 Gym 的语义可移植率、当前实现复用率与 MVP 含义。
- [源码与资料清单](research/SOURCE_MANIFEST.md)：仓库 URL、固定提交、子模块、完整性验证、许可证状态和 SHA256。
- [项目文件布局](docs/PROJECT_LAYOUT.zh-CN.md)：PyPTO-X 的模块边界、源码归属和许可证分层。
- [Worktree 与 subagent 计划](docs/WORKTREE_AGENT_PLAN.zh-CN.md)：分支拓扑、任务依赖、启动参数和一小时长等待协议。
- [Qwen3.5-0.8B BF16/W8A8 模型 MVP](docs/20-planning/0001-2026-09-07-qwen35-08b-bf16-w8a8-mvp.zh-CN.md)：模型 revision、算子闭包、量化契约、后端顺序和验收标准。
- [资源矩阵](docs/RESOURCE_MATRIX.zh-CN.md)：5080 WSL、AMD 6750GRE、鲲鹏 920B 和 QEMU 的使用安排。
- [本机重任务资源锁策略](docs/LOCAL_RESOURCE_POLICY.zh-CN.md)：跨项目 `local` 锁、cgroup 限制、运行时资源监控与安全停止规则。
- [无模型冒烟测试规范](docs/SMOKE_TEST_SPEC.zh-CN.md)：每个 subagent 只执行一次的测试契约。
- [上游版本锁](configs/upstream_lock.yaml)：edge/stable 双轨仓库 SHA、release 锚点和晋升验证项。
- [开发集成锁](configs/development_lock.yaml)：integration 冻结点、task commit 与验证证据。
- [模型目标配置](configs/model_targets.yaml)：机器可读的 Qwen3.5-0.8B 与 BF16/W8A8 范围。
- [离线参考资料](references/README.md)：两份公开讲稿、可检索文本和活动页面。
- [官方 PyPTO 文档源码](upstream/pypto/docs/README.md)：固定快照内含 706 个 Markdown 文档文件及图片。

## 目录

- upstream/pypto：GitCode 官方 PyPTO submodule，包含 Tensor `pypto` 与 Professional `pypto_pro`。
- upstream/pypto-gym：官方算子、模型接入和 Qwen3.5-9B 案例。
- upstream/pto-isa：PTO Tile ISA、设备实现、CPU simulator 和文档。
- upstream/pypto-community：无共同 Git 祖先的 community implementation，含完整 simpler runtime 子模块。
- upstream/PTOAS：PTO assembler/optimizer，含两个固定测试子模块。
- research：本次分析和来源清单。
- references：外部公开资料的离线副本。

## 一句话建议

PyPTO-X 以 Tensor frontend 形成公共 Core IR，并冻结 Target/Compiler/Runtime ABI；Pro 保留为 Ascend expert dialect，并只通过显式 portable subset 接入公共层。CPU 开发优先级为 scalar reference → AVX2 → AVX-512 → SVE256，不开发 NEON 优化后端；之后再做 NVIDIA 和 AMD 共用的 GPU 中层及各自 codegen。许可证调整的积极意向应尽快落成正式文本或明确书面例外，再发布非华为处理器后端。
