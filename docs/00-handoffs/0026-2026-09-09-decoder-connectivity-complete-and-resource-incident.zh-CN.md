# PyPTO-X Qwen3.5 decoder connectivity 完成与资源事故闭环

归档序号：`0026`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_M1I_DECODER_CONNECTIVITY_COMPLETE_BF16_RUNTIME_BINDING_READY`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    ec60f95979a56a9646d84f163912e677e9eb08ac
parent milestone    63c9a4fdd6c1282aa8226b157d81dccbea6b6f21
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

实现与集成：

```text
recovery task branch      work/qwen35-no-weight-decoder-recovery
task commits              8a0d9c810, 910839fd0, 0e3b7b8d8, 1df74f77a
integration commits       baf826aae, fcf037efc, bf7a499a2, ec60f9597
final verification branch verify/qwen35-decoder-connectivity-r3
```

原中断 worktree `/home/chiro/projects/pypto/worktrees/pypto-x/qwen35-no-weight-decoder-connectivity` 仍保留未提交现场，未执行 reset、clean 或覆盖；它不再作为可合并实现来源。

## M1I 完成能力

`build_qwen35_text_decoder_graph()` 已从 identity shell 升级为24层参数化、无 checkpoint 值、非 identity 的 Core SSA graph。固定模型仍为：

```text
Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17
```

连接范围：

- 18层 Gated DeltaNet 与6层 full attention，full-attention 层为 `3/7/11/15/19/23`；
- embedding 与 LM head 共用一个显式 storage parameter；
- input RMSNorm、attention/GDR branch、两条 residual、SwiGLU MLP、final RMSNorm 与 LM head；
- Core `iota/compare` 生成 position ids 与 causal mask；
- 每层 KV、Conv 和 recurrent state 都以 immutable input + 新 SSA return 表达，共48个 state input 和48个 state output；
- 权重只表现为 typed function parameter，不嵌入、下载或加载数值；
- 旧 structural manifest/identity shell 通过显式 compatibility API 保留，不再由默认 graph builder 选中。

真实 decode profile `(B=1,T=1,past=4096)` 生成4,532个 Core operations。graph metadata 强制 CPU vector lowering 使用 compact iteration；调用者显式请求关闭 compact 时会被拒绝，避免按 vocabulary/KV/attention `numel` 物化 Python chunk。

## 资源事故

### 事故经过

旧实现代理在 2026-09-09 10:50 CST 左右直接启动 heredoc Python，对真实 `(B=1,T=1,past=4096)` graph 在同一进程内依次执行 scalar/vector/AVX2/AVX-512/SVE/GPU lowering。该命令：

- 没有通过 `scripts/resource/run_local_heavy.sh` 取得 `local` 锁；
- 以 `python3 -` 承载 heavy 工作，无法由独立 driver 审计；
- 工具等待30秒后返回后台 session，但代理只读取输出而丢弃 `session_id`；
- 进程继续运行时全局锁仍显示 `local FREE`。

旧 vector-common 默认路径按 lane block 创建 Python `VectorChunk`，单条真实 profile 探测已可增长到约7.8 GiB RSS；多个 backend 在同一 Python 进程中连续构建后，目标进程最终达到约23.1 GiB memory peak，并在11:51 CST 被约26 GiB 的用户 cgroup OOM killer 终止。事故进程 PID 为 `862362`；没有终止其他项目进程。

### 工程修复

恢复过程没有复用或清理旧 dirty worktree，而是从 exact integration HEAD 新建 recovery worktree并重新整理实现。后续规则已实际用于 r2 验收：

1. full suite、大 shape lowering/compile 必须通过全局 `local` 锁和项目 heavy runner；
2. 本阶段统一使用 `MemoryMax=4096 MiB`、最多4 CPU；
3. 每个 backend 使用证据目录中的独立 driver 和独立前台进程；
4. 禁止 heredoc、`python -u -`、`python3 -` 承载 heavy 工作；
5. 返回 `session_id` 时必须持续读取到终态；
6. Core graph 以 `requires_compact_iteration` 强制 vector plan 保持 symbolic/compact；
7. 原事故现场和错误根目录中的历史日志只读保留，正式结论只引用标准 `_meta` 证据。

r3 全量回归峰值 RSS 为409 MiB、最低 `MemAvailable` 为24,370 MiB。所有受管命令正常退出，`local` 锁释放。

## 三轮审查与独立验收

首轮验收没有直接放行实现。它发现 GDR `RMSNormGated` 少了上游 BF16 舍入边界：正确顺序是 FP32 normalize → normalized cast BF16 → BF16 weight product → FP32 SiLU gate → final BF16。实现修复后，主代理又修正了回归 golden 中遗漏的 BF16 product 中间 RNE。

r2 验收通过后，主代理继续逐行对照上游 decoder layer，发现第二级 residual 使用了本层 `layer_input`，而上游正确语义是 `attention_residual + mlp_output`。该问题会丢失 token-mixer 输出到 layer output 的直接残差边。修复后从新 integration HEAD 创建 r3 验收 worktree，并增加逐层 operand 审计与可区分新旧路径的非零数值 oracle。

r3 最终结果：

```text
full suite                         491 passed, 7 skipped
changed-file Python 3.7 AST        PASS
compileall/package discovery       PASS
independent RMSNormGated bit probe PASS
24-layer residual def-use audit    PASS
residual numeric oracle            PASS; fixed bits 15695, old bits 15693
```

真实 profile 的独立 lowering/compile：

```text
scalar         PASS
vector-common  PASS; all compact
AVX2           PASS; all compact
AVX-512        PASS; all compact
SVE256         PASS; all compact
GPU common     PASS
CUDA/PTX       PASS
```

拓扑等价的24层 synthetic graph 实际执行：

```text
CPU scalar              PASS
AVX2 native             PASS
AVX-512 native          PASS
QEMU SVE256             PASS
Kunpeng 920B SVE256     PASS; VL=32 bytes, SVE2=0
RTX 5080 Driver/PTX JIT PASS; compute capability 12.0
```

六条执行路径都返回51个值（logits、final hidden、position ids、48个 state），输出 finite，position ids 为 `[[0]]`，state inputs 保持不变。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-decoder-connectivity-final-r3/validation.json
../worktrees/_meta/pypto-x/integration-w6-qwen35-decoder-connectivity-final-r3/brief.zh-CN.md
```

首轮失败证据保留在：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-decoder-connectivity-final/validation.json
```

r2 的中间通过记录也保持只读，用于解释后续主审查为什么增加 residual operand oracle：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-decoder-connectivity-final-r2/validation.json
```

## 明确边界

- 未下载或加载模型权重；真实 profile 证明 graph/lowering/compile，不证明带权整网推理。
- synthetic profile 保留24层与18/6路由，但使用缩小 tensor shape，不能代表0.8B数值或性能。
- `T>1` GDR 仍是静态 sequential SSA 展开，operation 数随 token 数线性增长；尚不是 fused WY/chunk kernel。
- SVE `iota/compare` 仍是 host-reference fallback。
- 920B 是2 vCPU KVM guest；QEMU、920B 与当前 CUDA correctness kernel 都不形成性能门槛。
- CUDA 路径仍是 Driver API + PTX JIT，不是 NVVM、Tensor Core 或 fusion 结论。
- AMD/HIP、Ascend CANN/NPU stable regression 均未由本阶段闭包。

## 下一步

进入 `qwen35-bf16-runtime-binding`：先在不触碰模型文件的前提下冻结 typed buffer/mmap、parameter manifest 校验、tied-storage alias、state buffer 与 launcher 绑定契约。完成这一层后，只有在用户明确授权下载/加载固定 revision 权重并确认参考环境时，才进入 BF16 带权文本模型执行；W8A8-linear 排在 BF16 之后。
