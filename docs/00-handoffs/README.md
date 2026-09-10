# PyPTO-X 交接档案

交接性质的文档统一使用：

```text
NNNN-YYYY-MM-DD-<阶段或主题>.zh-CN.md
```

- `NNNN` 是四位递增序号，只表示归档顺序；
- 日期使用 Asia/Shanghai 的自然日；
- 根目录 `HANDOFF.zh-CN.md` 始终保存最新的滚动接手信息；
- 本目录保存阶段快照，历史文件只追加勘误，不覆盖原结论。

## 索引

| 序号 | 日期 | 阶段 | 文档 |
|---:|---|---|---|
| 0001 | 2026-09-07 | 规划完成、执行获批 | [执行起点快照](0001-2026-09-07-execution-start.zh-CN.md) |
| 0002 | 2026-09-07 | C0/W1 完成、W2 就绪 | [W1 完成快照](0002-2026-09-07-w1-complete.zh-CN.md) |
| 0003 | 2026-09-07 | W2 完成、portable 门禁就绪 | [W2 完成快照](0003-2026-09-07-w2-complete.zh-CN.md) |
| 0004 | 2026-09-07 | portable bootstrap 完成、W3 就绪 | [W2B 完成快照](0004-2026-09-07-w2b-portable-complete.zh-CN.md) |
| 0005 | 2026-09-07 | CPU vector common 完成、AVX2 就绪 | [W3 vector-common 完成快照](0005-2026-09-07-w3-vector-common-complete.zh-CN.md) |
| 0006 | 2026-09-07 | AVX2 完成、AVX-512 就绪 | [W3 AVX2 完成快照](0006-2026-09-07-w3-avx2-complete.zh-CN.md) |
| 0007 | 2026-09-08 | AVX-512/W3 完成、SVE256 就绪 | [W3 完成快照](0007-2026-09-08-w3-avx512-complete.zh-CN.md) |
| 0008 | 2026-09-08 | SVE256 完成、GPU common 就绪 | [SVE256 完成快照](0008-2026-09-08-w4-sve256-complete.zh-CN.md) |
| 0009 | 2026-09-08 | GPU common 完成、SVE256 ECS 原生验证、CUDA 等待资源 | [GPU common 与 SVE native 完成快照](0009-2026-09-08-w4-gpu-common-sve-native-complete.zh-CN.md) |
| 0010 | 2026-09-08 | 保留 920B ECS、Qwen3.5-0.8B M0 闭包启动 | [Qwen3.5 闭包执行起点](0010-2026-09-08-qwen35-08b-closure-start.zh-CN.md) |
| 0011 | 2026-09-08 | Qwen3.5-0.8B M0 完成、M1A 公共原语启动 | [Qwen3.5 M0 完成与 M1A 起点](0011-2026-09-08-qwen35-m0-complete-m1a-start.zh-CN.md) |
| 0012 | 2026-09-08 | Qwen3.5 M1A 完成、M1B shape/layout 启动 | [Qwen3.5 M1A 完成与 M1B 起点](0012-2026-09-08-qwen35-m1a-complete-m1b-start.zh-CN.md) |
| 0013 | 2026-09-08 | Qwen3.5 M1B 完成、M1C1 SVE256 数值原语启动 | [Qwen3.5 M1B 完成与 M1C1 起点](0013-2026-09-08-qwen35-m1b-complete-m1c1-sve-start.zh-CN.md) |
| 0014 | 2026-09-08 | Qwen3.5 M1C1 完成、M1C2a SVE256 layout 启动 | [Qwen3.5 M1C1 完成与 M1C2a 起点](0014-2026-09-08-qwen35-m1c1-complete-m1c2a-layout-start.zh-CN.md) |
| 0015 | 2026-09-08 | Qwen3.5 M1C2a 完成、M1C2b SVE256 indexing 启动 | [Qwen3.5 M1C2a 完成与 M1C2b 起点](0015-2026-09-08-qwen35-m1c2a-complete-m1c2b-indexing-start.zh-CN.md) |
| 0016 | 2026-09-08 | Qwen3.5 M1C2b 完成、M1D SVE256 composites 启动 | [Qwen3.5 M1C2b 完成与 M1D 起点](0016-2026-09-08-qwen35-m1c2b-complete-m1d-composites-start.zh-CN.md) |
| 0017 | 2026-09-09 | Qwen3.5 M1D 完成、M1E SVE256 conv/state 启动 | [Qwen3.5 M1D 完成与 M1E 起点](0017-2026-09-09-qwen35-m1d-complete-m1e-conv-state-start.zh-CN.md) |
| 0018 | 2026-09-09 | 本机异常重启恢复、全局资源锁与运行监控生效 | [本机资源治理快照](0018-2026-09-09-local-resource-lock-governance.zh-CN.md) |
| 0019 | 2026-09-09 | Qwen3.5 M1E 完成、M1F attention/KV 规划、5080 恢复 | [Qwen3.5 M1E 完成与 M1F 规划](0019-2026-09-09-qwen35-m1e-complete-m1f-attention-kv-planning.zh-CN.md) |
| 0020 | 2026-09-09 | Qwen3.5 M1F 完成、独立验收生效、M1G GDR state 规划 | [Qwen3.5 M1F 完成与 M1G 规划](0020-2026-09-09-qwen35-m1f-complete-m1g-gdr-state-planning.zh-CN.md) |
| 0021 | 2026-09-09 | Qwen3.5 M1G 完成、M1H position/control 规划 | [Qwen3.5 M1G 完成与 M1H 规划](0021-2026-09-09-qwen35-m1g-complete-m1h-position-control-planning.zh-CN.md) |
| 0022 | 2026-09-09 | GamePC GPU 独占规则、5080 恢复探测、CUDA Driver/PTX 路线 | [GamePC GPU 独占与 CUDA 恢复探测](0022-2026-09-09-gamepc-gpu-exclusive-cuda-probe.zh-CN.md) |
| 0023 | 2026-09-09 | Qwen3.5 M1H 与 CUDA C1 完成、AVX parity/CUDA C2 就绪 | [M1H 与 CUDA C1 完成快照](0023-2026-09-09-qwen35-m1h-cuda-c1-complete.zh-CN.md) |
| 0024 | 2026-09-09 | AVX2 Qwen parity 与 CUDA C2 完成、AVX-512 parity 启动 | [AVX2 parity 与 CUDA C2 完成快照](0024-2026-09-09-avx2-parity-cuda-c2-complete.zh-CN.md) |
| 0025 | 2026-09-09 | AVX-512 Qwen parity 完成、无权重 decoder connectivity 就绪 | [AVX-512 parity 完成快照](0025-2026-09-09-avx512-parity-complete.zh-CN.md) |
| 0026 | 2026-09-09 | M1I decoder connectivity 完成、资源事故闭环、BF16 runtime binding 就绪 | [Decoder connectivity 与资源事故闭环](0026-2026-09-09-decoder-connectivity-complete-and-resource-incident.zh-CN.md) |
| 0027 | 2026-09-09 | M1J BF16 external runtime binding 完成、backend ingestion 就绪 | [BF16 runtime binding 完成快照](0027-2026-09-09-qwen35-bf16-runtime-binding-complete.zh-CN.md) |
| 0028 | 2026-09-10 | M1K-CPU ingestion 完成、AMD `gfx1036` 核显目标识别 | [CPU ingestion 与 AMD 核显快照](0028-2026-09-10-qwen35-bf16-cpu-ingestion-amd-igpu.zh-CN.md) |
| 0029 | 2026-09-10 | M1K-CUDA ingestion 完成、AMD `gfx1036` 静态 backend 就绪 | [CUDA ingestion 完成快照](0029-2026-09-10-qwen35-bf16-cuda-ingestion-complete.zh-CN.md) |
| 0030 | 2026-09-10 | AMD `gfx1036` 静态 C1 完成、C2 layout/indexing 就绪 | [AMDGPU 静态 C1 完成快照](0030-2026-09-10-amdgpu-gfx1036-static-c1-complete.zh-CN.md) |
| 0031 | 2026-09-10 | AMD `gfx1036` 静态 C2 完成、C3 math/reduction 就绪 | [AMDGPU 静态 C2 完成快照](0031-2026-09-10-amdgpu-gfx1036-static-c2-complete.zh-CN.md) |
| 0032 | 2026-09-10 | AMD `gfx1036` 静态 C3 完成、HIP runtime 等待设备 | [AMDGPU 静态 C3 完成快照](0032-2026-09-10-amdgpu-gfx1036-static-c3-complete.zh-CN.md) |
| 0033 | 2026-09-10 | 用户授权 Qwen3.5-0.8B 权重、W6 带权执行波次 1 派发 | [权重授权与波次 1 派发](0033-2026-09-10-bf16-weight-authorization-and-wave1-dispatch.zh-CN.md) |
| 0034 | 2026-09-10 | 波次 1 部分完成：gold 参考、AMD 运行态审计、性能协议、W8A8 契约 | [波次 1 结果与待决策](0034-2026-09-10-wave1-results-and-pending-decisions.zh-CN.md) |
