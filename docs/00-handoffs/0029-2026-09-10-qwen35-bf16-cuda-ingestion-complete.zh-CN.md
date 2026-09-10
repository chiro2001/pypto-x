# PyPTO-X Qwen3.5 BF16 CUDA ingestion 完成快照

归档序号：`0029`

归档日期：2026-09-10（Asia/Shanghai）

状态：`QWEN35_M1K_CPU_CUDA_COMPLETE_AMD_GFX1036_STATIC_HIP_READY`

## 冻结点

```text
integration branch  port/pypto-x-integration
integration HEAD    fe6f3270b973b08be98cacfec04a6a4e9482e2b0
parent milestone    0bc662c6af9bdd32f587dbe2a0e9d0aec81290da
edge base           34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
worktree            clean
```

实现与集成：

```text
task branch          work/qwen35-bf16-cuda-buffer-ingestion
task commit          2eb57849814bd04f405699f7cf4b660afcfd7ba8
integration commit   fe6f3270b973b08be98cacfec04a6a4e9482e2b0
verification branch  verify/qwen35-bf16-cuda-buffer-ingestion
```

## 完成能力

CUDA runtime 现在能直接消费 M1J/M1K 的 typed external bytes/memoryview/mmap：

- HtoD：同步 `cuMemcpyHtoD` 使用 borrowed host address + exact nbytes；
- DtoH：同步 `cuMemcpyDtoH` 直接写入 caller-owned writable address；
- raw 路径不再执行 `_host_values → list → _pack`；
- raw 输出不再执行 `_unpack → list → _write_host`；
- `TypedByteView` 与 active `Py_buffer` 由 allocation 持有到同步拷贝和 launch 完成；
- HostTensor、普通 sequence 和显式 device pointer 兼容路径保留。

所有 external descriptor 会在创建 CUDA context、加载 module 或分配 device memory 前验证 dtype、shape、nbytes、contiguity 与 mutability。BF16、FP32、INT32、INT64，以及 bytes、bytearray、read-only mmap、writable mmap 均纳入门禁。

## 独立验收

```text
focused CUDA mock           6 passed
full pypto_x                522 passed, 7 skipped
Python 3.7 changed files    PASS
compileall/python -S        PASS
package/diff/source clean   PASS
```

扩展 mock 强制 legacy pack/unpack helper 在 raw 路径被调用时失败，并覆盖：

- exact address/nbytes、zero-size、null/negative size；
- short/long/noncontiguous/released view；
- readonly output、input immutable、output canary；
- validation view lifetime、device pointer 与 HostTensor compatibility；
- module/allocation/context 异常清理。

RTX 5080 真机结果：

```text
graph                 Qwen3.5 24-layer synthetic decoder
external inputs       371
external outputs      51
functional states     48
packed input bytes    43,200
launches              2
result                PASS
```

两次 launch 的 logits finite/nonzero、position=0、所有 external input immutable、output canary 保持；执行前后没有 CUDA compute process 残留。

正式证据：

```text
../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cuda-buffer-ingestion-final/validation.json
../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cuda-buffer-ingestion-final/brief.zh-CN.md
```

## 明确边界

- 当前仍是 Driver API + PTX JIT correctness bootstrap，不是 nvcc/NVVM/Tensor Core/fusion 或性能验收。
- 只执行 synthetic 小型无权重 storage；没有下载、映射或加载0.8B checkpoint。
- 普通 HostTensor/sequence 路径仍保留 list pack/unpack 作为兼容 fallback；只有 external raw 路径保证不进入它。
- Qwen BF16 带权执行已经具备 graph、binding 和 CPU/CUDA ingestion seam，但开始下载/加载权重仍需用户明确授权。

## 下一步

Qwen BF16 带权阶段等待授权。当前转入 `hip-gfx1036-static-backend`：为已识别的 GamePC `amd-igpu-gfx1036` 建立静态 target、artifact 和 HIP/ROCDL compiler seam；由于 WSL 无 `/dev/kfd`、ROCm/HIP runtime 和 device library，运行态必须保持 `BLOCKED_DEVICE`，不能伪造 kernel PASS。
