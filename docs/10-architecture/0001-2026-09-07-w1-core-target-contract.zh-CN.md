# W1：Core IR 与 Target ABI 公共契约

状态：`FROZEN_FOR_W1`

日期：2026-09-07（Asia/Shanghai）

## 1. 目标与边界

W1 只建立可测试、目标无关的最小接口，不实现 CPU/GPU 指令选择，也不改变现有 Ascend 编译链默认行为。

```text
pypto Tensor frontend
        │ export（一次）
        ▼
CoreProgram ── verify / stable JSON dump
        │
        ▼
CompilerBackend ── Artifact ── RuntimeBackend.launch(LaunchRequest)
        ▲
     TargetSpec + CapabilitySet
```

公共层允许：Tensor、Scalar、Shape、View、结构化控制流、逻辑 Tile、显式 effect。公共层禁止：UB/L0、AIC/AIV、MTE/Pipe、RegTensor/VF、CCE intrinsic、CUDA/ROCm matrix fragment 和具体寄存器宽度。

## 2. Python 模块位置

现有 `python/pypto/ir.py` 与 `python/pypto/runtime.py` 是模块文件，W1 不把它们改成包，也不做破坏性搬迁。新增模块固定为：

```text
python/pypto/core_ir/
python/pypto/target/
python/pypto/compiler/
python/pypto/abi/
python/pypto/frontend/core_export.py
```

W1 不要求在 `pypto.__init__` 顶层重导出这些实验接口；直接从上述子模块导入，避免改变现有用户路径。

## 3. Core IR 最小数据模型

- `CoreType`：kind、dtype、shape；shape 维度只使用整数或稳定的符号名。
- `ValueRef`：函数内稳定 SSA 名称与 `CoreType`。
- `Operation`：稳定 op name、输入/输出引用、规范化 attributes、effect。
- `Region`：有序 block/operation；结构化控制流以嵌套 region 表达。
- `CoreFunction`：名称、参数、返回值、函数体。
- `CoreProgram`：schema version、函数列表和目标无关 metadata。
- `Effect` 至少区分 `pure`、`read`、`write`、`read_write`、`barrier`。

序列化必须满足：键顺序与集合顺序确定、禁止进程地址/临时时间戳、相同输入多次 dump 字节一致、未知 schema version 明确失败。首版格式为 UTF-8 canonical JSON，并包含 `schema_version=1`。

Tensor parser/export adapter 的职责是把一次前端解析结果映射为 `CoreProgram`；禁止针对不同 target 重新执行用户 Python Kernel。若当前 native parser 无法在无 CANN 环境加载，可先支持显式、纯 Python 的 export input，并把 native bridge 记为后续集成点，不能伪造已连通。

## 4. Target、Compiler 与 Runtime 契约

- `TargetSpec` 是不可变数据：`kind`、`triple`、有序且去重的 features、可选 device/ABI 属性。
- `CapabilitySet` 表达 dtype、op class、vector、matrix、async-copy 等能力；查询未知能力返回 false 或明确的 unsupported 结果。
- target registry 对规范化 target key 注册工厂；重复注册默认报错，显式 replace 才允许覆盖。
- `Artifact` 至少记录格式、入口、payload/路径二选一、target、ABI version、metadata；cache key 只来自稳定字段和内容摘要。
- `CompilerBackend.lower(CoreProgram, TargetSpec)` 产生 target IR；`compile(...)` 产生 `Artifact`。
- `TensorDesc` 记录 dtype、shape、strides、device、address/handle 和只读属性，不携带厂商专属 descriptor。
- `LaunchRequest` 记录 artifact、入口、输入/输出 tensor、scalar、workspace 与 stream handle。
- `RuntimeBackend` 至少提供 `is_available`、`load`、`workspace_size`、`launch`；不可用资源给出结构化原因。

所有协议使用类型提示和抽象接口/`Protocol`，不在 W1 引入 LLVM、CANN、CUDA 或 ROCm 依赖。

## 5. 错误与兼容策略

- schema、ABI 与 artifact format 都有显式版本；不静默接受未来版本。
- unsupported capability 与 backend unavailable 分开表示。
- Ascend adapter 后续可以包装现有 CCE/Bisheng/CANN 路径，但公共接口不得反向暴露 CCE 参数。
- Pro portable subset 后续通过显式 adapter 产生同一 `CoreProgram`，不让 Pro parser 成为公共 Core IR 的唯一来源。

## 6. W1 验收

1. 新模块可在没有 CANN/CUDA/ROCm 和模型权重时导入、构造和测试。
2. Core IR canonical JSON round-trip 与 deterministic dump 测试通过。
3. Target feature normalization、registry collision、artifact cache key、ABI validation 测试通过。
4. 验证 harness 能比较 reference/backend 输出，报告 dtype/shape、最大绝对/相对误差和不匹配位置；能维护稳定 IR snapshot。
5. 每个 task 只运行一次统一 host smoke，并返回 commit SHA、日志与已知风险。
