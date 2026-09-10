# PyPTO-X Qwen3.5-0.8B 权重授权与 W6 带权执行波次 1 启动快照

归档序号：`0033`

归档日期：2026-09-10（Asia/Shanghai）

状态：`QWEN35_BF16_WEIGHT_AUTHORIZED_WAVE1_DISPATCHED_AMD_RUNTIME_STILL_BLOCKED_DEVICE`

## 授权

用户于 2026-09-10 明确批准 PyPTO-X 四条主线同步推进，并**首次授权下载与加载固定 revision 的 `Qwen/Qwen3.5-0.8B` 权重**：

```text
repo_id             Qwen/Qwen3.5-0.8B
revision            2fc06364715b967f1860aea9cf38778875588b17
scope               纯文本 BF16 带权执行；不接视觉编码器；不含 W8A8 实现
```

此前 `configs/development_lock.yaml` 的 `blocked_authorization` 边界（`no_weight_download_or_load_without_user_approval`）自此解除，并已更新为 `authorized_in_progress`。

## 权重资产

权重下载到控制仓之外的项目 meta 资产目录，未进入任何 Git 历史：

```text
../worktrees/_meta/pypto-x/assets/qwen35-0.8b/2fc06364715b967f1860aea9cf38778875588b17/
```

```text
model.safetensors-00001-of-00001.safetensors
  bytes   1746942600
  sha256  04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
files     13（config.json、index.json、tokenizer 系列、LICENSE、README 等）
total     1769980465 bytes（与 HF tree 列表逐项一致）
manifest  _manifest.json（父 agent 逐文件 sha256 + 字节数）
```

`config.json` 复核确认与 M0 冻结结构一致：hidden 1024、FFN 3584、24 层（`full_attention_interval=4`，即 3/7/11/15/19/23 为 full attention）、8 query heads / 2 KV heads、head_dim 256、`partial_rotary_factor=0.25`、`rope_theta=1e7`、vocab 248320、`tie_word_embeddings=true`、`rms_norm_eps=1e-6`；checkpoint 另含视觉塔与 MTP 层，均不在首期文本范围内。

## 波次 1 任务

四条主线各一个 task/branch/worktree，均从 integration HEAD `bcf9516e6419d232338988238ffa8f79e59079c1` 创建，`started_at=2026-09-10T08:03:16Z`：

| task | branch | worktree | 目标 |
|---|---|---|---|
| `qwen35-bf16-weight-ingestion` | `work/qwen35-bf16-weight-ingestion` | `../worktrees/pypto-x/qwen35-bf16-weight-ingestion` | 无依赖 safetensors 读取器、官方张量名 → 320 参数映射、真实 packed layout 与三方字节校验；不跑前向 |
| `qwen35-bf16-reference` | `work/qwen35-bf16-reference` | `../worktrees/pypto-x/qwen35-bf16-reference` | 独立 venv + CPU torch/transformers，产出固定 prompt 的官方 prefill/decode logits 与逐层 hidden states 作为 gold |
| `amd-igpu-runtime-feasibility` | `work/amd-igpu-runtime-feasibility` | `../worktrees/pypto-x/amd-igpu-runtime-feasibility` | 判定 `gfx1036` 能否获得可执行 HIP/HSA 运行态；只读探测 + 官方支持面调研，不安装任何软件 |
| `w8a8-linear-contract` | `work/w8a8-linear-contract` | `../worktrees/pypto-x/w8a8-linear-contract` | 冻结首期 W8A8-linear 契约（scheme、层覆盖、artifact/binding 扩展、后端顺序、精度验收、非目标） |

每个 subagent 遵守固定协议：启动后只执行一次统一 smoke（`target=host`）、heavy 命令必须经 `run_local_heavy.sh` 取得全局 `local` 锁、不修改 integration/上游 master/他人 worktree、Python 源码保持 3.7 AST 门槛。

## AMD 运行态复核（只读）

```text
Windows  AMD Radeon(TM) Graphics, DriverVersion=32.0.21045.5002, Status=OK
         NVIDIA GeForce RTX 5080, DriverVersion=32.0.16.1692, Status=OK
WSL      Ubuntu 24.04.4 LTS, kernel 6.6.87.2-microsoft-standard-WSL2
设备     /dev/dxg 存在；/dev/kfd 与 /dev/dri 不存在
工具     rocminfo / rocm-smi / hipcc 均不存在
```

因此 AMD runtime 仍为 `BLOCKED_DEVICE`；是否有可能通过 WSL 驱动或非官方途径解锁，交由 `amd-igpu-runtime-feasibility` 以官方支持面证据判定，本快照不做结论。

## 边界

- 本快照只记录授权、资产与任务派发；**没有任何带权前向结果**，也没有 HIP 真机执行。
- 静态 AMD C3 的 4,532/4,532 仍只是静态 lowering 证据，不因本次授权升级为运行结论。
- 权重只存在于本机资产目录，未提交、未分发；后续报告中出现权重路径时沿用同一绝对路径与 sha256。
