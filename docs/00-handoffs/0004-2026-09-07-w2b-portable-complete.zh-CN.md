# PyPTO-X W2B portable bootstrap 完成交接快照

归档序号：`0004`

归档日期：2026-09-07（Asia/Shanghai）

状态：`W2B_COMPLETE_W3_READY`

## 冻结点

```text
base        = b017b4bb3e189ba3f50ae777461f0574decf26ba
integration = 3cb7061c2592247c3c5fbb3fbfb6bc264485cdc7
branch      = port/pypto-x-integration
```

完整 commit 与验证字段见 [开发集成锁](../../configs/development_lock.yaml)。

## 已完成能力

- `PYPTO_X_PORTABLE_ONLY=1` 显式启用轻量 package initializer。
- portable-only 下直接导入 Core IR、Target、Compiler、ABI、CPU backend、core export/PIL bridge。
- 不导入 torch、native loader/pypto_impl、Pro、torch_npu 或 CANN。
- 父包初始化后删除环境变量，frontend 仍保持同一 portable 模式。
- classic Tensor/jit/dtype API 用带操作指引的 AttributeError 明确不可用，`hasattr` 语义正常。
- 开关未设置时保留原 native initializer，不自动捕获错误并降级。

## 验证证据

- task 唯一一次 host smoke：`PASS`。
- integration 全量测试：`68 passed`。
- `python -S` direct import、mode lock、forbidden-module 和反射检查：`PASS`。
- typed fake PIL → CoreProgram → CPU scalar → float32 differential：`PASS`。
- Python 3.7 AST：45 个新增 Python 文件 `PASS`。
- package discovery 与 integration smoke：`PASS`。

## 下一步

从此冻结点依次执行 [W3 x86 vector 契约](../10-architecture/0004-2026-09-07-w3-x86-vector-contract.zh-CN.md)：`cpu-vector-common` → `cpu-avx2` → `cpu-avx512`。性能门槛仍不在当前阶段冻结。
