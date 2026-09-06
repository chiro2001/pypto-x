# PyPTO-X W3 CPU vector common 完成交接快照

归档序号：`0005`

归档日期：2026-09-07（Asia/Shanghai）

状态：`W3_VECTOR_COMMON_COMPLETE_AVX2_READY`

## 冻结点

```text
base        = 3cb7061c2592247c3c5fbb3fbfb6bc264485cdc7
integration = a784bac441cb4f8564d00afdfcc4e9e17e7d8de0
branch      = port/pypto-x-integration
```

完整 commit 与验证字段见 [开发集成锁](../../configs/development_lock.yaml)。

## 已完成能力

- ISA-neutral `VectorPlan`、iteration/tail/reduction/matmul plan。
- scalar-cleanup 与 masked tail 语义。
- immutable CPU feature contract 和确定性 vector artifact。
- 独立纯 Python chunk executor，不调用 `CpuScalarRuntime`。
- target feature 只能求并集，不能被下游显式要求削弱。
- feature source 在 lower → serialize → decode/re-lower 后规范化且幂等。
- structured differential 保留 NumPy `float32` dtype，不经 `.tolist()` 丢失类型。
- portable-only 模式可直接导入 vector common。

该冻结点只证明公共 vector plan 与语义执行正确，不声称发射了真实 SIMD 指令。

## 验证证据

- task 唯一一次 host smoke：`PASS`。
- integration 全量测试：`96 passed`。
- structured float32 scalar/vector differential：`PASS`。
- Python 3.7 AST：49 个新增 Python 文件 `PASS`。
- compileall、setuptools package discovery、`git diff --check`：`PASS`。
- integration 唯一一次 smoke：`PASS`，日志为
  `../worktrees/_meta/pypto-x/integration-w3-vector-common/smoke/20260906T192442Z/smoke.log`。

## 下一步

从该冻结点创建 `cpu-avx2` 独立 worktree。验收必须覆盖真实 CPUID + OSXSAVE/XGETBV、Clang 实际编译、objdump 的 YMM/无 ZMM 证明、scalar differential、odd-tail/非对齐/feature 缺失/artifact 篡改，以及 BF16 的 FP32 accumulation + RNE；不冻结性能门槛。
