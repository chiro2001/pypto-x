# PyPTO-X Qwen3.5-0.8B closure 执行起点

归档序号：`0010`

归档日期：2026-09-08（Asia/Shanghai）

状态：`QWEN35_08B_M0_CLOSURE_IN_PROGRESS`

## 决策

RTX 5080 GamePC 关机期间，CUDA 保持资源等待。用户决定保留按量鲲鹏 920B ECS，并优先推进 Qwen3.5-0.8B 纯文本算子闭包。

首个任务严格限定为 M0：

- 固定 `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` text config；
- 以 PyPTO-Gym 的 Qwen3.5-9B/27B 同构实现为拓扑证据，不复制其尺寸；
- 生成标准库、无权重、无 PyTorch/Transformers 依赖的 shape harness；
- 输出 operator、tensor shape、KV/GDR/conv state、prefill/decode closure manifest；
- 对照当前 PyPTO-X Core/scalar/vector/SVE/GPU common 能力形成 gap matrix。

## Worktree

```text
repository  = upstream/pypto-gym
base        = 945a360e12592239a3549cb62d0db37af32bbc03
integration = port/pypto-x-qwen-integration
task        = work/qwen35-08b-model
worktree    = ../worktrees/pypto-gym/qwen35-08b-model
started_at  = 2026-09-08T09:07:57Z
```

每个 subagent 仍只运行一次无模型 smoke，不下载或加载权重。

## 资源边界

- 920B ECS：openEuler 22.03、2 vCPU、4 GiB、KVM、SVE256；用于 native 正确性/汇编，不冻结性能门槛。
- ECS 当前没有 PyTorch、Transformers、NumPy；M0 不安装依赖。
- 0.8B BF16 权重裸数据约 1.63 GiB，但当前 Python-list HostTensor 的对象开销远超 4 GiB；真实权重执行前必须实现 typed buffer/mmap 路径。
- 下载/加载模型权重仍未授权；M0 只允许读取固定 revision 的小型配置/metadata。

## 后续顺序

1. M0 closure manifest 与 shape harness；
2. Core/scalar 公共 shape/math 原语；
3. SVE256 native RMSNorm、SwiGLU、RoPE、Softmax；
4. full attention + KV cache；
5. Gated DeltaNet：causal Conv1D、L2Norm、gate/decay、chunk/recurrent state；
6. 单层与 24 层无权重/合成数据连通；
7. 取得授权和 typed-buffer 内存门禁后再运行实际 BF16 权重；
8. BF16 完成后进入 W8A8-linear。
