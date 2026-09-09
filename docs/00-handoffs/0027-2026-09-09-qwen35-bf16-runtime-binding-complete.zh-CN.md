# PyPTO-X Qwen3.5 BF16 external runtime binding 完成快照

归档序号：`0027`

归档日期：2026-09-09（Asia/Shanghai）

状态：`QWEN35_M1J_BF16_RUNTIME_BINDING_COMPLETE_BACKEND_INGESTION_READY`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    0bc15d06ee167d5b6b5c62cffec88083dabb2bdf
parent milestone    ec60f95979a56a9646d84f163912e677e9eb08ac
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

实现与集成：

```text
task branch          work/qwen35-bf16-runtime-binding
task commits         801e4e6d8, 7121b2c52, a1356ed64
integration commits  55f71762e, 7cfe28133, 0bc15d06e
verification branch  verify/qwen35-bf16-runtime-binding-r2
```

## 完成能力

M1J 新增 target-independent external storage seam，不依赖 PyTorch/Transformers，也不把 tensor 转为 Python list：

- `build_binding_schema(program)` 从 exact `CoreFunction.parameters` 顺序建立 schema；
- `BufferBinding.from_buffer(...)` 持有 caller-owned contiguous bytes/memoryview/mmap slice；
- `build_packed_layout(schema)` 只计算确定性的64-byte storage-relative offsets；
- `bind_packed_region(...)` 把 packed region 绑定为 non-owning view；
- `ordered_input_descriptors(...)` 按 Core 参数 ordinal 组装 `TensorDesc`；
- `assemble_launch_request(...)` 在组装前验证 artifact entrypoint 与 `program_digest`。

Qwen3.5-0.8B 固定 contract 为：

```text
runtime inputs       3   input_ids/cos/sin
model parameters     320（包含18层各 A_log/dt_bias，共36个）
immutable states     48
regular outputs      3
state outputs        48
compact manifest     22 storage classes
packed regions       368（320 parameters + 48 input states）
packed total bytes   1,574,877,952
```

embedding 与 LM head 在 Core function 中只保留一个参数，在 packed layout 中也只有一个 storage region；两个公开 alias 不增加 descriptor 或 byte range。

真实 model profile 只计算 schema/layout JSON 和 offset，没有创建、truncate、mmap、读取或加载1.57GB backing file。metadata driver 的 RSS 增量约14.4 MiB。

## Fail-closed 门禁

schema 不信任 compact metadata 自报的结构。实现先从 exact Core 参数/state shape 与固定24层18/6路由推导 Qwen profile，再由已冻结的 manifest builder 生成唯一 expected manifest，最后逐项匹配：

```text
name / storage_id / aliases / order
dtype / shape / count
elements_per_tensor / parameter_elements / storage_bytes
top parameter_element_count / parameter_byte_count
```

exact Core、22项 compact manifest 与顶层 totals 必须三方一致。缺失、额外、重复 storage、uint64 overflow、同名不同图 artifact、缺失或篡改 digest 均在访问 backend 前拒绝。

## 两轮独立验收

首轮验收为 `FAIL`：它发现修改 compact manifest 的 `name` 或唯一 `storage_id`、同时保持算术总量不变时仍会被接受。修复将 manifest 绑定到 exact Core contract，并移除了六个无必要的公共 API alias。

r2 最终结果：

```text
focused runtime binding     15 passed
full pypto_x                506 passed, 7 skipped
Python 3.7 AST              PASS
compileall/python -S/import PASS
package/canonical exports   PASS
source worktree             clean
```

full suite 通过 `MemoryMax=4096 MiB`、最多4 CPU的 heavy runner 执行；峰值 RSS 307 MiB，最低 `MemAvailable` 24,581 MiB，`local` 锁正常释放。

小型独立 storage 测试覆盖 bytes、bytearray、read-only mmap、writable mmap、非连续 view、越界、错 offset、mutability、released view 和 caller lifetime。所有合法输入保持 zero-copy object identity。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-runtime-binding-final-r2/validation.json
../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-runtime-binding-final-r2/brief.zh-CN.md
```

首轮失败证据保留在：

```text
../worktrees/_meta/pypto-x/verify-qwen35-bf16-runtime-binding/validation.json
```

## 明确边界

- `alignment=64` 只保证 packed storage 内的相对 `byte_offset` 对齐，不保证任意 Python buffer 的实际虚拟地址对齐；native adapter 需要时必须检查指针或复制到 aligned storage。
- M1J 冻结的是 schema/layout/launcher seam，不代表 backend 已能摄取 byte memoryview/mmap。
- 当前 scalar、AVX2、CUDA runtime 对 byte memoryview 都会安全拒绝，不会静默错算；AVX-512共享AVX2 ingestion，SVE仍有自己的 wire/staging路径。
- 未下载、映射或加载模型权重，不构成 BF16 带权整网执行结论。
- packed layout 当前包含 parameter 与 input-state region；output/state-output storage 由调用方单独提供。

## 下一步

进入 `qwen35-bf16-backend-ingestion`：先为 CPU shared ingestion 冻结 BF16 byte-view 解码、指针对齐、readonly/lifetime 与无逐元素 Python list复制的规则，再接 AVX2/AVX-512、SVE256 和 CUDA host-to-device staging。所有测试继续使用小型 deterministic buffers；取得用户明确授权前不下载或加载模型权重。
