# W2B：portable bootstrap 契约

状态：`DRAFT_PENDING_W2`

日期：2026-09-07（Asia/Shanghai）

## 问题

Python 导入子模块前必定执行父包 `pypto.__init__`。现有 initializer 会立即加载 native shared libraries 并触发 online build，因此源码 checkout 在无 PyPTO wheel/CANN 环境中执行：

```python
from pypto.core_ir import CoreProgram
from pypto.backends.cpu import CpuScalarRuntime
```

仍会失败。W1/W2 的隔离 namespace 单测证明了模块内部无厂商依赖，但不是用户可用的导入路径。

## 契约

- 增加显式环境开关 `PYPTO_X_PORTABLE_ONLY=1`。
- 开关未设置时，`pypto.__init__` 和 `pypto.frontend.__init__` 的导入、native loader、异常类型与顶层导出保持原样。
- 开关设置时，不导入 torch、native loader、classic runtime、Pro 或 CANN；只把父 package 初始化为可加载 portable 子模块的轻量 namespace。
- portable-only 模式支持直接导入 `pypto.core_ir`、`pypto.target`、`pypto.compiler`、`pypto.abi`、`pypto.backends.cpu` 和 `pypto.frontend.core_export`。
- classic Tensor DSL、`jit` 和 native parser 在 portable-only 模式不可伪装可用；访问时给出说明如何取消开关/安装完整 wheel 的清晰错误。
- 不允许在 initializer 捕获任意 native 错误后自动静默进入 partial package；portable 模式必须显式 opt-in。
- 不新增依赖，不下载工具链或模型，继续支持 Python 3.7。

## 验收

1. 新 Python 子进程设置开关后直接导入上述模块并完成一个 CPU scalar add。
2. 子进程没有加载 `torch`、`pypto._loader`、`pypto.pypto_impl`、`pypto_pro` 或 `torch_npu`。
3. portable-only 下访问 classic `pypto.Tensor`/`pypto.frontend.jit` 明确失败。
4. 未设置开关时，用受控 mock/探针证明原 initializer 仍尝试原 native loader，而不是降级到 portable-only。
