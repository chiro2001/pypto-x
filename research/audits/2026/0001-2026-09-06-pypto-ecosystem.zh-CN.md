# 0001 · 2026-09-06 · PyPTO 生态、算子库与 CANN 融合审计

状态：`COMPLETE`

后续更新：前端关系和 PyPTO-X 前端建议已由 [0002 审计](0002-2026-09-06-pypto-pypto-pro-relationship.zh-CN.md) 补充；0002 对“Pro-first”假设重新打开了决策。

## 快照

| 仓库 | 默认分支 | edge commit | 版本观察 |
|---|---|---|---|
| PyPTO | `master` | `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad` | `v9.2.0-beta.2` 后 306 commits；Python 0.2.1；CANN package 9.2.0 |
| PyPTO-Gym | `master` | `945a360e12592239a3549cb62d0db37af32bbc03` | 无 tag；package 0.1.0；`pypto>=0.2.0` |
| PTO-ISA | `master` | `668248ec886447a83787200786fe6f461169b701` | `v9.2.0-beta.2` 后 92 commits；package 仍标 9.1.0 |
| community PyPTO | `main` | `9f657f37ed20ce148b46fb7229c267a152a0644e` | 与官方 PyPTO 不是同一 Git 代码线 |
| PTOAS | `master` | `dc15ee5b9e459c025eb4f714f2f892b535d93eb0` | `vmi-v0.1.6` 后 78 commits；多条 tag 线 |

上表是 edge 快照，尚未在同一 CANN toolkit/NPU 上验证为可运行组合。

## 版本结论

采用“CANN release family 作兼容锚点 + exact commit SHA 作真正锁定”：

- `edge lock`：记录各默认分支 HEAD，用于差异审计和接口预研；
- `stable lock`：记录 release family、每仓 resolved SHA、CANN toolkit/驱动、嵌套 submodule 和测试证据；
- 不浮动跟随 `master/main`，也不只保存 tag 名。

原因：PyPTO-Gym 无 tag；PTO-ISA 版本字段与代码进度不完全同步；PTOAS 同时存在主工具链、VMI 和 CANN tag 线，并且已观察到同名远端 tag SHA 变化。

## CANN 生态中的位置

```text
CANN toolkit / driver
  ├─ runtime / acl_rt / hcomm / securec
  ├─ Bisheng CCE compiler
  └─ PTO-ISA headers
          ↑
PyPTO
  ├─ classic pypto: Tensor/Tile frontend → IR/Pass → AICPU/AICore runtime
  └─ pypto_pro: AST/IR → CCECodegen → Bisheng → CANN launch
          ↑
PyPTO-Gym
  ├─ fused operators
  ├─ torch.library / Meta / NPU wrapper
  └─ Qwen/DeepSeek/GLM/Gemma/Kimi 等模型接入
```

CANN 9.1.0 及以后的 CANN 包已集成 PyPTO。PyPTO 顶层 CMake 直接依赖 `acl_rt`、`runtime`、`hcomm`、`ascend_hal`、`ascend_dump` 等 CANN 组件，并产生 wheel、run 包及设备侧制品。

`pypto_pro/runtime/opc/pypto_compile.py` 复用真实 CANN OPC 驱动，只替换编译叶子，最终仍产生 CANN `kernel_meta`/object/JSON 制品。这是 Ascend adapter 的重要现成边界。

官方 PyPTO 当前直接消费 PTO-ISA 头文件并调用 Bisheng，未发现它直接链接 PTOAS 库。PTOAS 应先作为可选/参考工具链锁定，不应误记为官方 PyPTO 的强制构建依赖。

## 算子生态

### Classic `pypto`

`python/pypto/op/` 有 16 个主要模块，静态统计约 144 个公开顶层函数定义（含重载与校验函数，不等于唯一算子数）。覆盖 elementwise、reduction、matmul、conv、quantization、index/gather/scatter、layout/mutation、creation、distributed、random 等。

### `pypto_pro`

静态 API 口径：

- Tile/Block 约 105 个公开函数；
- VF 约 82 个公开方法；
- SIMT 约 49 个公开方法；
- CCE backend 共 301 个注册项。

这 301 项是“DSL op + CCE pipe/codegen callback”，不是现成的跨架构通用算子库。`MemorySpace`、`PipeType`、AIC/AIV、VF 和现有 SIMT 都含 Ascend/A5 语义。

### PTO-ISA

`docs/isa/manifest.yaml` 的机器口径为 138 个核心 entry + 12 个通信 entry，合计 150；README 的“90+”是去重宣传口径。它提供 Ascend A2/A3/A5 实现和 CPU simulator，但 CPU simulator 不等于 PyPTO 已有 CPU backend。

### PyPTO-Gym

源码口径：

- `pypto_tensor` 非 `__init__.py` 文件 115 个，其中 114 个使用 classic `pypto`；
- `pypto_pro` 实现文件 9 个；
- 算子测试文件约 159 个；
- 显式 `torch.library("pypto")` 注册源文件 16 个，注册名约 12 个。

覆盖 RMSNorm、RoPE、SwiGLU、GEMM/grouped GEMM、MoE routing、Flash/Sparse/Page Attention、MLA、GDR/KDA、FP8/MXFP8/QAT、KV cache 等，并包含多个大模型的融合算子和 wrapper。

Qwen3.5 GDR 的真实实现位于：

```text
src/pypto_gym/ops/pypto_tensor/qwen3_5/gdr_fwd/
src/pypto_gym/ops/pypto_tensor/qwen3_5/gdr_bwd/
```

它使用 `@pypto.frontend.jit`，而不是 `pypto_pro.language`。9B 集成仅 monkey-patch prefill chunk GDR，RMSNorm、Attention、MLP、RoPE 等并未由 PyPTO 接管。

## 对 PyPTO-X 的约束

1. 只改 `pypto_pro` 不会自动获得现有 Gym 算子和 GDR 兼容性。
2. W1 前必须确定前端策略：Pro-first 后移植 Gym，或者为 classic `pypto` 增加到 neutral Core IR 的兼容桥。
3. 现有 Pro SIMT 是 A5 Vector execution domain 的 CCE lowering，不能直接当作 CUDA/HIP 公共 GPU ABI。
4. Ascend adapter 必须保留 CANN runtime、OPC、Bisheng、PTO-ISA 和 TorchNPU 边界；非 Ascend target 不应看到这些语义。

## 证据入口

- [PyPTO README 版本配套](../../../upstream/pypto/README.md)
- [PyPTO 安装和 CANN 集成说明](../../../upstream/pypto/docs/zh/install/build_and_install.md)
- [`pypto_pro` JIT](../../../upstream/pypto/python/pypto_pro/runtime/jit.py)
- [`pypto_pro` OPC adapter](../../../upstream/pypto/python/pypto_pro/runtime/opc/pypto_compile.py)
- [PyPTO-Gym README](../../../upstream/pypto-gym/README.md)
- [PTO-ISA manifest](../../../upstream/pto-isa/docs/isa/manifest.yaml)
- [当前源码快照清单](../../SOURCE_MANIFEST.md)

## 验证记录

- `cann-ecosystem` host smoke：PASS。
- `release-policy` host smoke：PASS。
- `operator-ecosystem` smoke：FAIL，因通用脚本假定 Gym worktree 内存在 `python/pypto_pro`；这是测试适用范围问题，不是 Gym 源码回归。
- 三个调研 worktree 都没有源码修改或提交。
