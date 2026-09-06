# 0002 · 2026-09-06 · PyPTO Tensor 与 PyPTO Professional/Pro 关系审计

状态：`COMPLETE`

本记录基于 CANN 开发者社区文章《PyPTO Professional模式: PyPTO-Pro易用性的来源》（2026-08-16），并与 PyPTO commit `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad` 交叉验证。原始页面通过用户提供的临时脚本访问，本项目未保存请求中的 Cookie/会话信息或原始 HTML。

## 结论

`python/pypto` 与 `python/pypto_pro` 是同一 PyPTO 产品/发行包内的并存前端：

- PyPTO Tensor / classic `pypto`：高层 Tensor/Tile 表达，框架承担多核切分、Buffer 复用、调度和底层参数；
- PyPTO Professional / `pypto_pro`：专家 Tile/Reg/SIMT/VF 表达，开发者显式控制 Tile、Buffer、流水等性能决策；
- 两者不是“旧包 → 新包”的简单废弃/替代关系，也不是两个完全独立的软件。

## 共享与分离边界

### 共享

- 同一 `pypto` wheel 同时打包 `pypto` 和 `pypto_pro` 命名空间。
- `pypto_pro/_bootstrap.py` 从 `pypto` 导入 `pypto_impl`。
- Pro IR 重用 `pypto.pypto_impl.ir` 的原生 IR/type substrate。
- 同一 C++ `pypto_impl` 扩展绑定 classic IR、Pro backend 和 Pro codegen。

这里的“共享后端 IR”应理解为共享部分原生 IR 基础，不代表已有稳定的跨厂商 Backend ABI。

### 分离

```text
classic pypto
  pypto.frontend.jit
    → classic AST/PIL/Tensor Graph/Pass
    → KernelModule/machine runtime
    → CANN

pypto_pro
  pypto_pro.language.jit
    → Pro ASTParser + Cube/Vector Program
    → CCECodegen + PTO-ISA C++
    → Bisheng
    → ctypes / torch.npu stream
```

两者拥有不同的 Python parser、算子方言、编译入口和运行时编排，不是同一 frontend 的两种语法皮肤。

## Pro 的当前定位

当前 `pypto_pro` 仍是 Ascend CCE/A5 专家路径：

- backend registry 实际只有 `BackendCCE`；
- parser 直接依赖 `BackendCCE` 查询 op pipe；
- codegen 直接把 PyPTO IR 转为 PTO-ISA C++；
- 当前 JIT 对 arch 有 A5 限制；
- MemorySpace、Pipe、UnitFlag、RegTensor、VF 和现有 SIMT 是 CCE/Ascend 语义。

因此不应把 Pro 的完整 dialect 直接当作 CPU/CUDA/HIP 公共 Core IR。

## JIT/AOT 与 CANN

文章所述 Pro JIT/AOT 基本得到源码支持：

- JIT 生成 PTO-ISA C++ 并调用 Bisheng；
- AOT/OPC 复用 CANN `asc_opc` 和 `asc_op_compile_base`；
- Pro 只替换 DSL 到 kernel 源码的叶子编译阶段；
- 产出 `kernel_meta`、object、JSON 和 TilingKey 制品。

产物流程兼容已有源码证据，但与手写 Ascend C 等价的完整边界仍需真实 CANN/NPU 验证。

## 文章与当前源码的偏差

- 文章示例的 `pl.TilingKeyField` 不在当前 `pypto_pro.language.__all__`；当前实现位于 `pypto_pro.runtime.tilingkey`。
- 文章提到的 `allow_in_graph`/`aclGraph` 能力未在当前 commit 的 Pro API 中找到对应实现，需维护方或外部 CANN 分支确认。

## 对 PyPTO-X 的影响

原先“仅以 Pro 为跨架构前端”的假设需要重新对齐。建议架构是：

```text
PyPTO Tensor frontend
  → 可移植 Tensor/Shape/Control Flow/Core IR
  → Ascend / AVX / SVE / CUDA / HIP

PyPTO Professional frontend
  ├─ portable subset → Core IR（后续明确界定）
  └─ CCE expert dialect → Ascend adapter
```

建议：

1. 保持 `pypto` 和 `pypto_pro` 两套公共 API，不让其中一个假装替代另一个。
2. 跨架构 MVP 优先从 Tensor frontend 的高层可移植语义形成 Core IR，以承接已有 Gym/GDR 生态。
3. Pro 先作为 Ascend/CANN 专家路径和回归哨兵。
4. 如果支持 Pro 跨架构，只先定义 Tensor/Scalar/Shape/基础 Tile 的 portable subset。
5. UB/L0/AIC/AIV/MTE/Pipe/RegTensor/VF/CCE SIMT 保留在 Ascend target dialect。

该建议会影响 W1 的 parser/Core IR 任务，必须先由用户确认，不能在实现中默认更改前端策略。

## 证据入口

- [PyPTO 打包配置](../../../upstream/pypto/pyproject.toml)
- [`pypto_pro` bootstrap](../../../upstream/pypto/python/pypto_pro/_bootstrap.py)
- [`pypto_pro` IR 入口](../../../upstream/pypto/python/pypto_pro/ir/__init__.py)
- [classic frontend parser](../../../upstream/pypto/python/pypto/frontend/parser/entry.py)
- [Pro kernel parser](../../../upstream/pypto/python/pypto_pro/runtime/kernel.py)
- [Pro JIT/runtime](../../../upstream/pypto/python/pypto_pro/runtime/jit.py)
- [Pro backend registry](../../../upstream/pypto/framework/src/interface/pypto_pro/backend/common/backend_registry.cpp)
- [Pro OPC adapter](../../../upstream/pypto/python/pypto_pro/runtime/opc/pypto_compile.py)

## 验证记录

- worktree：`/home/chiro/projects/pypto/worktrees/pypto-x/wiki-pypto-relation`
- commit：`34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad`
- host smoke：PASS
- 源码修改：无
- 未运行真实 CANN/NPU 测试
