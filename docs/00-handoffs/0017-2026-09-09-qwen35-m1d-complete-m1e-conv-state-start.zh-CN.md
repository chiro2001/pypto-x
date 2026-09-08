# PyPTO-X Qwen3.5 M1D 完成与 M1E SVE256 起点

归档序号：`0017`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_08B_M1D_COMPLETE_M1E_SVE256_CONV_STATE_STARTED`

## M1D 冻结结果

```text
task commits      = 2e4c4d9cc, 38307d402
integration HEAD  = fe6b54a0f40e739d5ebed87baa5d608565171695
```

新增 `pypto.portable.qwen35`，只组合现有 Core primitive，覆盖 Qwen delta-weight RMSNorm、加性 epsilon L2Norm、SwiGLU、stable Softmax、partial RoPE、runtime-indexed GQA repeat-KV 与 causal mask。没有新增 composite SVE opcode。

固定 M0 契约已纠正并逐字段核对：hidden=1024、FFN=3584、24 层、attention/KV heads=8/2、head_dim=256、rotary_dim=64，full-attention 层为 3/7/11/15/19/23。

验证：

- M1D focused `24 passed`，PyPTO-X 全量 `417 passed in 147.15s`；
- Python 3.7 AST、compileall、128-package discovery、portable `python -S` import、diff check PASS；
- task 与 integration 各自唯一 QEMU smoke PASS；
- 920B native 的 BF16 RMSNorm、stable Softmax、partial RoPE、2→8 GQA、decoder/attention 多操作图与 guarded canary PASS。

真实边界：这仍是 primitive composition，不是 fused kernel；cos/sin、repeat-interleave indices 与 causal mask 由 model adapter 提供；逐 lane libm、host broadcast、BF16 reorder、split 重传和 Python list/wire staging 均未消失。

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1d-final/
```

## M1E 任务

```text
repository  = upstream/pypto
base        = fe6b54a0f40e739d5ebed87baa5d608565171695
branch      = work/qwen35-sve256-conv-state
worktree    = ../worktrees/pypto-x/qwen35-sve256-conv-state
started_at  = 2026-09-08T16:44:43Z
resource    = QEMU then Kunpeng 920B ECS
```

范围是 Qwen Gated DeltaNet 的 kernel-size=4 depthwise causal Conv1D：prefill/chunk 接受显式 prior conv state，decode 接受单 token，并都返回 `output + new_state`。优先用 slice/concat/reshape/broadcast/mul/add/silu 组合；状态演化必须是 CoreProgram 多输出，不能隐藏为 Python host mutation。
