# PyPTO-X Qwen3.5-0.8B M0 完成与 M1A 起点

归档序号：`0011`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M0_COMPLETE_M1A_STARTING`

## M0 冻结结果

PyPTO-Gym M0 已完成固定 revision 的纯文本无权重 shape/operator/state closure：

```text
model       = Qwen/Qwen3.5-0.8B
revision    = 2fc06364715b967f1860aea9cf38778875588b17
task HEAD   = bb776187a1e69897a97652b6317c6b6b44e802fd
integration = b0c1e621c886ffc4d84bdd002a6bc8c8fd5e9628
```

冻结的结构要点：hidden 1024、intermediate 3584、24 层，其中 18 层 Gated DeltaNet、6 层 full attention；full attention 层为 3/7/11/15/19/23；Conv1D kernel 为 4；conv state 为 `[B,6144,4]`，recurrent state 为 `[B,16,128,128]` FP32。

验证结果：

- 专属测试 `99 passed`；
- wheel/package-data、Python 3.9 AST、compileall、package discovery、diff/行宽通过；
- PyPTO-Gym integration smoke `PASS`；
- 920B 标准库 shape harness 的 12 个 prefill/decode 场景通过，RSS 约 14.8 MiB；
- 未下载、读取或加载权重，920B 未安装 torch/Transformers/NumPy。

证据目录：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m0-final/
```

## M1A 任务

首个公共原语任务为：

```text
repository  = upstream/pypto
base        = e00c12a498ac806bb8f51eb58b9603fb57bc82f7
branch      = work/qwen35-portable-primitives
worktree    = ../worktrees/pypto-x/qwen35-portable-primitives
started_at  = 2026-09-08T10:12:28Z
```

范围严格限定为 Core IR、CPU scalar golden 与验证：

```text
cast
exp / rsqrt
sigmoid / silu / softplus
reduce_mean
broadcast
where
```

M1A 不直接实现孤立 SVE kernel。完成公共语义后，M1B 再处理 shape/layout；M1C 才接入 vector-common、AVX 与 SVE256，并以 920B 进行 native 正确性和汇编验收。

## 资源与边界

- 920B 实例当前为 `ACTIVE`，继续按量运行；用于原生 SVE256 功能/汇编，不冻结性能门槛。
- RTX 5080 GamePC 关机，CUDA 继续等待资源。
- AMD 6750GRE 尚未接入。
- 不下载或加载模型权重；视觉编码器、MTP、W8A16、INT4、FP8 均不在当前范围。
- 当前 Python-list runtime 无法承载约 873M 参数；真实权重执行前仍需 typed buffer/mmap。
