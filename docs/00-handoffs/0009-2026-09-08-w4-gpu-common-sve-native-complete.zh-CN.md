# PyPTO-X W4 GPU common 与 SVE256 native 完成交接快照

归档序号：`0009`

归档日期：2026-09-08（Asia/Shanghai）

状态：`W4_GPU_COMMON_COMPLETE_CUDA_WAITING_RESOURCE`

## 冻结点

```text
previous integration = 5e42eed7c432034db02c2d339e67789afd419ae9
current integration  = e00c12a498ac806bb8f51eb58b9603fb57bc82f7
branch               = port/pypto-x-integration
```

本轮 integration commit：

```text
ca1d7e783ab8c6bc0ae5a424e18cbf865fe129bb  native SVE toolchain/sysroot
c692f4f3f9d99468fb06bec727226cbc3b3c187e  GPU common initial
74c934717f2743580327c4668a2326a8f751f933  GPU ABI/simulator hardening
a266fa2a825c39fd2f36300eacb4ababb0c5cdb6  artifact/correctness gates
1c28bcc047f5e93d0c97db724fc8beaea05099c0  options/payload alignment
e00c12a498ac806bb8f51eb58b9603fb57bc82f7  native compiler metadata
```

task commit 与 integration commit 的完整对应关系见 [开发集成锁](../../configs/development_lock.yaml)。

## GPU common 已冻结能力

- 厂商无关 Grid/Workgroup/Thread/Subgroup 模型；geometry 支持 1–3D、uint64 组合溢出检查和 zero-work no-op。
- subgroup 不固定为 32；公共契约覆盖 32/64 及 partial subgroup，由 target capability 约束。
- 公共 address space、memory scope、barrier、tensor/scalar kernel argument binding。
- 多 operation 程序携带逐 plan geometry sequence，不把不同 geometry 伪装为单 kernel。
- FP32/BF16 elementwise、reduce sum/max、二维矩形 matmul；constant/identity 保留 copy dtype。
- canonical JSON artifact、稳定 digest/cache key、严格 envelope/metadata/options/plan/target round-trip 与 tamper 拒绝。
- 无设备 `GpuCommonSimulator` 使用 IEEE binary32 rounding、BF16 RNE、FP32 sequential accumulation；不调用 `CpuScalarRuntime`。
- 公共 IR/ABI 不携带 CUDA/NVVM/PTX、HIP/ROCDL/AMDGPU、WMMA/MFMA 或固定 matrix fragment。

这仍不是 CUDA/HIP backend：没有生成设备代码，没有调用 GPU driver，也没有真机吞吐结论。

## 鲲鹏 920B ECS 原生验证

用户授权通过 `~/tools/ecs-920B` 创建按量实例。能力实测：

```text
OS              = openEuler 22.03
architecture    = aarch64 little-endian
CPU vendor      = HiSilicon
vCPU            = 2
virtualization  = KVM
compiler        = GCC 10.3.1
compiler triple = aarch64-linux-gnu
SVE             = 1
SVE2            = 0
VL              = 32 bytes
```

框架 native mode 直接编译并执行 AArch64 ELF，结果：

- FP32 add，19 元素 masked tail：`PASS`；
- BF16 add，19 元素 masked tail/RNE：`PASS`；
- guarded unaligned-buffer canary：`PASS`；
- 全负数 reduce-max：`PASS`；
- 9×5×10 矩形/tail matmul：`PASS`；
- artifact：ELF64 AArch64 static、`0555`、`nlink=1`；
- 热点反汇编：`whilelo=2`、`ld1w=3`、`st1w=4`、z=36、p=19、NEON v/q=0。

真机首轮暴露并修复两项此前 QEMU 未覆盖的问题：

1. native runtime 不应默认注入 x86 cross sysroot；
2. 原生编译器可名为 `cc`，架构必须由 compiler triple 与 ELF machine 验证，不能由 basename 猜测。

ECS 是 KVM guest；以上是原生进程功能和汇编证据，不等同于裸机性能门槛。实例仍在按量运行，状态与删除目标以 `~/tools/ecs-920B/state.env` 为准；删除必须再次得到用户确认。

## 集成验证

- GPU common 专属：`45 passed`；
- SVE256 + GPU common 专属联合：`107 passed`；
- integration 全量：`312 passed in 55.07s`，exit 0；
- Python 3.7 AST：累计 86 个变更 Python 文件 `PASS`；
- portable direct import：11 modules `PASS`；
- setuptools package discovery：127 packages，GPU required packages `PASS`；
- compileall、`git diff --check`：`PASS`；
- 本轮 delta 16 个 Python 文件行宽 ≤120：`PASS`；
- integration host smoke：`PASS`，未加载模型；
- GPU task 唯一 nvidia smoke：`BLOCKED_DEVICE`，未重跑。

主要证据：

- `../worktrees/_meta/pypto-x/integration-w4-gpu-common-final/logs/20260908T075915Z-pytest-full.log`
- `../worktrees/_meta/pypto-x/integration-w4-gpu-common-final/logs/20260908T080242Z-ecs-920b-native-validation.json`
- `../worktrees/_meta/pypto-x/integration-w4-gpu-common-final/smoke/20260908T080155Z/smoke.log`
- `../worktrees/_meta/pypto-x/gpu-common/smoke/20260908T055648Z/smoke.log`

## 下一步与阻塞

1. 下一 task 是 CUDA/NVVM，不再修改已经冻结的 GPU common 契约，除非真机证据暴露缺陷。
2. 用户已确认 RTX 5080 GamePC 当前关机；关机期间不创建 CUDA task、不声称真机结果。
3. GamePC 恢复后先按 Windows SSH → `wsl.exe -e bash -lc` 只读探测；最近一次 WSL 没有 `nvcc` 和 PyTorch。
4. 安装 CUDA toolkit、驱动组件或 Python 包属于外部环境变更，必须先得到用户明确授权。
5. AMD 6750GRE 仍未接入；HIP 只能在 CUDA 之后、并在实际 gfx/wave 探测后启动。
6. 非华为后端对外发布仍受当前 CANN 许可证边界约束。
