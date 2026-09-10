# PyPTO-X Qwen3.5 BF16 CPU ingestion 与 AMD 核显快照

归档序号：`0028`

归档日期：2026-09-10（Asia/Shanghai）

状态：`QWEN35_M1K_CPU_COMPLETE_CUDA_INGESTION_READY_AMD_GFX1036_BLOCKED_DEVICE`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    0bc662c6af9bdd32f587dbe2a0e9d0aec81290da
parent milestone    0bc15d06ee167d5b6b5c62cffec88083dabb2bdf
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

实现与集成：

```text
task branch          work/qwen35-bf16-cpu-buffer-ingestion
task commits         8707cb653, 78b39c62d
integration commits  e95591b94, 0bc662c6a
verification branch  verify/qwen35-bf16-cpu-buffer-ingestion-r2
```

## M1K-CPU 完成能力

新增共享 `TypedByteView`，让 M1J `BufferBinding.to_tensor_desc()` 产生的 bytes/memoryview/mmap slice 在 CPU runtime 中按 descriptor dtype/shape 解释，不再把 BF16 的两个字节误当两个元素。

- dtype：BF16、FP32、INT32、INT64；
- scalar：显式小 tensor reference 解码/写回；
- AVX2/AVX-512：经 CPython buffer protocol 借用 exporter pointer，owner/lifetime 保留到 launch 结束，不进入 list fallback；
- SVE256：在进程 wire 边界复制 raw little-endian bytes，不先转 Python float list；
- 输入：bytes、bytearray、read-only mmap、writable mmap、NumPy raw `V2/2x`（可用时）；
- 门禁：C-contiguous、exact nbytes、endianness、readonly/mutable、released view、short/long、canary。

## 两轮独立验收

首轮验收为 `FAIL`，发现：

1. 共享 `ctypes.pythonapi` function pointer 的 `argtypes` 使用模块局部 `_PyBuffer` class，`importlib.reload()` 后产生旧/新 class 指针冲突；
2. NumPy raw BF16 `dtype=V2` 的 memoryview format 实际为 `2x`，旧白名单遗漏。

修复改用 reload-stable `c_void_p` ABI signature，并在异常路径成对释放 `Py_buffer`；同时只在 `itemsize=2` 时接受 BF16 raw `2x`。

r2 结果：

```text
focused ingestion          10 passed
NumPy V2/scalar/SVE         3 passed
full pypto_x                516 passed, 7 skipped
reload/lifetime             6 reloads PASS
Python 3.7/compileall       PASS
python -S/package/diff      PASS
```

24层 synthetic external-buffer Qwen graph：

```text
inputs                      371
outputs                     51
functional state outputs    48
packed parameter/state      43,200 bytes
scalar                      PASS
AVX2                        PASS; 371 direct descriptors
AVX-512                     PASS; 371 direct descriptors
QEMU SVE256                 PASS; 371 typed inputs, 363 raw wire payloads
```

四条路径的 logits finite/nonzero、position `[0]`，所有 parameter/state/runtime input bytes 保持不变。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cpu-buffer-ingestion-final-r2/validation.json
```

明确边界：scalar 仍是小 tensor reference；SVE 整图仍有19次 compare/broadcast/where 等显式 host-reference `to_values`，不是全图 zero-copy；未运行真实权重。

## AMD 临时目标

用户确认6750GRE无法安装，AMD 路线暂时改用 GamePC 核显。只读探测得到：

```text
CPU/APU              AMD Ryzen 9 9900X
Windows adapter      AMD Radeon(TM) Graphics, DEV_13C0, status OK
Windows OpenCL       device name gfx1036
Windows Vulkan       integrated GPU
WSL devices          /dev/dxg present; /dev/kfd and /dev/dri absent
ROCm/HSA/HIP         absent
hipcc/rocminfo        absent
```

目标名冻结为 `amd-igpu-gfx1036`。`gfx1036` 来自 Windows OpenCL，不是 `rocminfo`；wavefront、BF16/INT8、MFMA/WMMA、XNACK 均为 unknown。WSL Clang 18 能识别 `amdgcn/gfx1036` 名称，但缺少 ROCm device library。

当前状态为 `BLOCKED_DEVICE`：可以开发静态 target/codegen，不能执行 HIP kernel，不能冒充6750GRE/gfx1031验收，也不能形成性能结论。

证据：

```text
../worktrees/_meta/pypto-x/gamepc-amd-igpu-probe-20260910/validation.json
../worktrees/_meta/pypto-x/gamepc-amd-igpu-probe-20260910/relocation-audit.txt
```

## 下一步

1. 实现 CUDA external byte-buffer host-to-device staging，并在 RTX 5080 上执行24层 synthetic binding graph；
2. CUDA 验收后，为 `amd-igpu-gfx1036` 实现不依赖真机 runtime 的静态 HIP/ROCDL target seam；
3. AMD WSL 获得 `/dev/kfd`、ROCm/HIP runtime 后再做 enumerate、vector-add 和完整算子正确性；
4. 取得用户明确授权后，进入 Qwen3.5-0.8B BF16 带权文本推理。
