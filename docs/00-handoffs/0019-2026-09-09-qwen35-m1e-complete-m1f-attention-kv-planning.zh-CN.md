# PyPTO-X Qwen3.5 M1E 完成与 M1F 规划

归档序号：`0019`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_08B_M1E_COMPLETE_M1F_ATTENTION_KV_PLANNING`

## M1E 冻结结果

```text
task commits      = be4061025, 8de5714a9, f63207e26
integration HEAD  = 41523cfc3341d47abb94a8e2d97111e2948ea501
```

M1E 用 portable primitive composition 实现 Qwen3.5 kernel-size=4 depthwise causal Conv1D：

```text
[B,C,T] + prior_state[B,C,4] + weight[C,4]
    -> activated_output[B,C,T], new_state[B,C,4]
```

状态转换是 CoreProgram 的函数式 SSA 多输出，不使用 hidden host mutation；prefill/chunk/decode 共用一张图。输入/state/output 为 BF16，乘加与 SiLU 前累计为 FP32，tap 顺序与 PyTorch cross-correlation 一致。GDR in-projection → reshape → transpose → Conv1D 的连接图已建立；GDR recurrent matrix state 仍未闭包。

SVE256 lowering version 已从 v3 升到 v4。v4 强制所有主 iteration、reduction nested iteration 和 matmul nested iteration 使用 compact domain；旧 v3 plan/artifact 必须拒绝并重新编译。Qwen `[1,6144,1024]` 图共 29 个操作，不物化逐元素 `VectorChunk`；small/large 静态 ELF 均为 881,928 bytes。

## 验收证据

- integration 全量：`426 passed in 113.34s`；
- Python 3.7 AST：94 files PASS；compileall、128 namespace-package discovery、diff check PASS；
- integration QEMU smoke：SVE/SVE2、VL=32 PASS，仅作功能验证；
- 920B native：SVE=1、SVE2=0、VL=32；concat/slice/cast/broadcast/mul/add/silu、guarded canary、Conv 输出与 `new_state` carry PASS；
- 无模型权重下载或加载；所有本机 full/compile 工作均经全局 `local` 锁、cgroup 和运行时监控。

证据目录：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1e-final/
```

已知限制：当前仍是 primitive composition，不是 fused Conv kernel；SiLU 逐 lane libm、host broadcast、BF16 reorder fallback 和 Python list/wire staging 仍存在。920B 是 2 vCPU KVM guest，不据此制定性能门槛。

## 下一阶段

Qwen BF16 闭包按以下顺序推进：

1. M1F：batched/transpose attention matmul + 函数式 KV cache append/read；
2. M1G：GDR recurrent matrix state/update，覆盖 chunk prefill 与 decode；
3. M1H：compare/iota/position 与无权重 24 层 text decoder graph；
4. AVX2/AVX-512 对齐 M1A–M1H 的冻结语义；
5. 获得参考环境与权重执行授权后运行 BF16 实际模型；
6. 最后进入 W8A8-linear scalar golden 与各后端实现。

用户于 2026-09-09 确认 RTX 5080 已恢复。CUDA 从“关机等待”转为“等待 `gamepc` 锁内工具链复核”；最后一次已知环境仍缺 `nvcc`/PyTorch，安装动作需要用户明确授权。CUDA backend 可在 GPU common ABI 不变的前提下与 M1F 并行。

> 后续勘误：用户在归档 `0022` 明确 RTX 5080 GPU 由 PyPTO-X 独占，GPU-only 工作不申请 `gamepc`；该锁只协调远端 heavy CPU/host-memory。

命令较多的后续阶段验收使用独立 subagent：从 exact integration HEAD 建专用验收 worktree，源码只读，统一运行 full suite、静态门禁、QEMU/真机与证据汇总；父 agent 审查并冻结结果。
