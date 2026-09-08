# PyPTO-X Qwen3.5 M1F 完成与 M1G 规划

归档序号：`0020`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_08B_M1F_COMPLETE_M1G_GDR_RECURRENT_STATE_PLANNING`

## 冻结结果

```text
task commits      = d4efa2401, 829b74380
integration HEAD  = 2d7da37b9ed20e3382232e31e45f584907c55e1e
verification      = verify/qwen35-m1f @ same exact HEAD
```

M1F 完成以下目标无关语义：

- matmul 支持 rank-2/3/4 且两侧 batch prefix 必须完全相等；首期不推断 batch broadcasting；
- QKᵀ 由显式 Core transpose 表达，不增加隐式 transpose flag；
- 固定 Qwen3.5-0.8B 的 8 query heads、2 KV heads、head_dim 256 full-attention 小图；
- `prior_k/prior_v/current_k/current_v -> bf16 output,new_k,new_v` 的函数式 KV transition；
- KV storage 保持 BF16，QK scaling、stable Softmax 与 PV 中间计算使用 FP32；
- causal mask、GQA head indices、scale 与 fill value 仍是显式 runtime 输入，等待 position/control 阶段收口。

wire/schema 兼容版本同步升级：CPU vector plan v2、SVE256 lowering v5；旧 vector v1 和 SVE v3/v4 artifact 明确拒绝并要求重编译。

## 独立验收

本阶段首次按新协议使用独立验收 subagent/worktree，源码只读，证据写入独占目录：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1f-final/
```

结果：

- 全量 `432 passed in 120.46s`；
- Python 3.7 AST 95 files、compileall 199 files、128 namespace packages、diff check PASS；
- QEMU SVE256 smoke PASS；
- prefill `T=128,past=0` 与 decode `T=1,past=128` 的所有 iteration compact、chunks 为空，plan 约 19 KiB，ELF 881,928 bytes；
- 920B native：SVE=1、SVE2=0、VL=32，guarded self-test、rank-4→逐 head rank-2 runner fallback、BF16 KV concat/state carry 与输入不变性 PASS；
- 无模型权重下载或加载，本机 heavy 命令全部经 `local` 锁和运行监控。

验收 driver 的早期试运行曾因 driver 自身的测试数据长度/字段名错误退出；修正后的同一源码 HEAD 与同一 ELF 已通过。失败日志保留用于审计，不计作实现回归。

## 真实边界与下一步

SVE batched matmul 当前是 host batch/head loop 调用 rank-2 native runner，不是 fused/native batched kernel。AVX2/AVX-512 对 batched matmul 继续显式拒绝。性能门槛仍未冻结。

下一主线 M1G 闭包 Gated DeltaNet recurrent matrix state/update，覆盖 chunk prefill 与逐 token decode；之后依次为 compare/iota/position、AVX2/AVX-512 Qwen parity、获授权后的 BF16 实际模型和 W8A8-linear。

RTX 5080 已由用户确认恢复；CUDA 等待 `gamepc` 锁释放后复核驱动、`nvcc` 与 Python/PyTorch 工具链，可与 M1G 并行。
