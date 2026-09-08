# PyPTO-X Qwen3.5 M1G 完成与 M1H 规划

归档序号：`0021`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_08B_M1G_COMPLETE_M1H_POSITION_CONTROL_PLANNING`

## 冻结结果

```text
task commits      = 549e2f098, a7c6a05f3
integration HEAD  = 962f422e18ee25fd37679687710edbd496b9f6ce
verification      = verify/qwen35-m1g @ same exact HEAD
```

M1G 新增固定 Qwen3.5-0.8B GDR recurrent matrix state builder：

```text
q/k/v       BF16  [B,T,16,128]
beta        BF16  [B,T,16]       post-sigmoid
g           FP32  [B,T,16]       log-space decay
prior_state FP32  [B,16,128,128] layout [B,H,K,V]
    -> output BF16 [B,T,16,128], new_state FP32 [B,16,128,128]
```

beta dtype 依据 Qwen-specific `upstream/pypto-gym/.../qwen3_5/gdr_fwd/gdr_fwd_impl.py` 固定为 BF16，不沿用其他 KDA 实现的 FP32 beta 约定。beta 在图内显式转 FP32后进入 state update。

每 token 递推为：

```text
S_decay = exp(g_t) * S_prev
pred    = k_t @ S_decay
delta   = v_t - pred
S_new   = S_decay + transpose(beta_t * k_t) @ delta
o_t     = scale * q_t @ S_new
```

state 是函数式 SSA 输出，无 host mutation；T=1 decode 与 T>1 chunk 共用静态 sequential unroll。该路径是正确性 reference，不是上游 fused WY/chunk kernel。

## 独立验收

证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-m1g-final/validation.json
```

- 全量 `438 passed in 124.37s`；
- Python 3.7 AST 96 files、compileall 199 files、128 namespace packages、diff check 和唯一 QEMU smoke PASS；
- 小型非对称 T=2 Python/scalar/vector/SVE golden 与两次 T=1 state carry PASS；
- 固定 Qwen 16×128 非零 scalar/SVE differential、T=2 QEMU 执行 PASS；
- T=64 编译产生 1,923 plans/2,307 iteration domains，全部 compact、chunks 为空，plan 约 1.87 MiB、ELF 881,928 bytes；
- 920B native 用 48 次 rank-2 runner 调用验证非零 recurrent step，prediction/state/output 最大绝对误差约 `2.70e-8`/`5.04e-11`/`8.29e-9`；
- 本机 heavy 命令均经全局 `local` 锁和运行监控；无模型权重下载或加载。

## 边界与下一步

- SVE rank-4 matmul 是 host batch/head loop fallback；
- AVX2/AVX-512 batched matmul 仍显式拒绝；
- chunk 是静态 sequential reference，未实现 fused WY/inverse；
- 已验收 T=64，尚未验收 T=128；
- 920B 是 2 vCPU KVM guest，不形成性能门槛。

下一任务 M1H 补齐 compare/iota/position、从 position/cache offset 生成 causal mask，并连接无权重 24 层 text decoder graph。其后进入 AVX2/AVX-512 Qwen parity、获授权后的 BF16 实际模型、W8A8-linear。

RTX 5080 已恢复，但 `gamepc` 锁截至本快照仍由其他项目持有；CUDA 工具链复核继续等待，不抢占。
