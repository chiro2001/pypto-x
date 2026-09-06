# W2：Tensor bridge、CPU scalar 与 Ascend adapter 契约

状态：`FROZEN_FOR_W2`

日期：2026-09-07（Asia/Shanghai）

## 1. 波次目标

W2 把 W1 的数据契约接到三个方向：

```text
Tensor frontend/PIL ── tensor-core-bridge ── CoreProgram
                                              │
                    ┌─────────────────────────┴────────────────────────┐
                    ▼                                                  ▼
          CPU scalar compiler/runtime                      Ascend adapter seam
          （可执行正确性 reference）                       （保留既有 CCE 路径）
```

W2 不实现 SIMD/SVE/GPU，不下载模型，不宣称无真机的 Ascend 回归通过。

## 2. Tensor Core bridge

- 入口只接受 Tensor frontend 已解析的 PIL/native 结果，或显式调用一次 Tensor parser；禁止按 target 重复执行用户 Kernel。
- 优先从 `pypto.pil.pir.Function` 的稳定 Python 结构导出：参数、block、call、result 和结构化控制流。
- callable/op 名称必须规范为稳定语义名，不能把对象地址或 `repr` 写入 IR。
- 类型无法可靠推导时使用显式 `unknown` 类型并记录诊断；禁止猜测 BF16/shape。
- 未支持的 node/op 必须带路径和节点类型显式失败，不能静默跳过。
- native C++ IR 只通过独立 bridge protocol 接入；仅有文本 dump 时不得声称完成结构化 round-trip。

第一版验收至少包含纯 Python fake PIL 的 elementwise、嵌套 block、稳定命名和 unsupported-node 测试。若本机没有可导入的 PyPTO native wheel，真实 native bridge 标记为待集成，不伪造通过。

## 3. CPU scalar reference

- 位置：`pypto/lowering/cpu`、`pypto/compiler/targets/cpu.py`、`pypto/backends/cpu`。
- 接受已验证的 `CoreProgram` 和 `TargetSpec(kind="cpu")`。
- compiler 产生可复现、可校验的本地 artifact；runtime 通过公共 `LaunchRequest` 执行。
- 第一闭包：identity/copy、常量、add/sub/mul/div、neg、reduce sum/max、二维 matmul；后续模型算子不在此波次伪装支持。
- shape/dtype/op 不支持必须结构化失败；禁止 fallback 到 Ascend、PyTorch 或在线编译。
- scalar 数学实现是后续 AVX/SVE/GPU 的 correctness golden，不用于性能结论。

验收使用小型确定性输入覆盖 add、非整除 shape 的 reduce、矩形 matmul、错误 shape、未知 op，以及 artifact 篡改检测。数值结果交给 W1 differential harness 报告。

## 4. Ascend adapter seam

- 位置：`pypto/backends/ascend` 与 `pypto/compiler/targets/ascend.py`。
- 通过依赖注入/lazy import 包装现有 Pro CCECodegen、Bisheng/CANN launch；导入公共模块本身不能要求 CANN。
- 公共 ABI 不出现 UB/L0、AIC/AIV、MTE/Pipe、RegTensor/VF 等字段；这些只存在于 adapter 内部 target payload/options。
- 当 toolkit、Bisheng、PTO-ISA 或设备缺失时返回结构化 unavailable，不能降级为假 PASS。
- W2 本地只验 adapter 调用顺序、参数边界、artifact/launch 转换和 unavailable 原因；真实 CCE 编译/launch/OPC 回归仍受 stable lock 门禁。

## 5. 共享目录与并行所有权

协调者在派发前创建并提交共享父包：

```text
python/pypto/backends/__init__.py
python/pypto/compiler/targets/__init__.py
```

三个 task 不再同时修改父包或 `pyproject.toml`。各自测试文件独占；subagent 仍遵守一次 host smoke、无模型和独立 worktree 协议。

## 6. W2 完成口径

- `tensor-core-bridge`：纯 Python PIL 结构导出已验证；真实 native 接入按环境如实报告。
- `cpu-scalar`：add/reduce/matmul 通过 differential harness，可作为功能 reference。
- `ascend-adapter`：接口 seam 与 fake-hook 单测通过；只有真实 CANN/NPU 回归后才能把 Ascend 兼容状态标为 PASS。
