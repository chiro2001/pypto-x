# PyPTO-X 可用资源与测试矩阵

更新日期：2026-09-11（Asia/Shanghai；2026-09-11 追加 920B 释放与 A2 暂停状态）

## 资源总览

| 资源 | 当前状态 | 主要用途 | 当前实测/已知信息 | 限制 |
|---|---|---|---|---|
| 本地开发机 | 可用；heavy 需全局 `local` 锁；M1K-CPU PASS | Core IR、ABI、CPU scalar/x86、PTO simulator、QEMU | x86_64，12 vCPU，Clang 22.1.8；M1K-CPU `516 passed, 7 skipped`，scalar/AVX2/AVX-512/QEMU SVE256 的24层 external-buffer graph 通过；2026-09-10 探测：本机是 **KVM guest**（`systemd-detect-virt=kvm`）、无 `/sys/.../cpufreq`，频率不可锁 | full pytest、大 shape lowering/compile 与并行构建必须走 `scripts/resource/run_local_heavy.sh`；默认保留 4 GiB、最多 6 CPU；**绝对性能门槛在本机永久 `UNGATED`**（见 `docs/PERF_MEASUREMENT_PROTOCOL.zh-CN.md`）；系统 libblas 是 netlib reference，唯一优化 BLAS 在 numpy 内 |
| RTX 5080（`192.168.101.5`） | 在线；GPU 由 PyPTO-X 独占；M1K-CUDA PASS | NVIDIA CUDA/PTX/NVVM 验证 | WSL2、RTX 5080 16,303 MiB、CC 12.0；2026-09-10 复测 KMD **616.92** / CUDA UMD **13.4**（文档旧值 610.62/13.3，已漂移）；Driver/PTX 连续两次执行371-input/51-output external-buffer 24层图 | GPU-only 无需 `gamepc` 锁；host-heavy 才持锁。无 `nvcc`/NVRTC/CUDART/cuBLAS/SDK headers/PyTorch/Triton；空闲时钟约 427–487 MHz vs 峰值 3090 MHz，测性能前必须 warmup 并断言时钟；当前是 correctness kernel，不是 fusion/性能结论 |
| AMD GamePC 核显 `gfx1036` | 静态 C3 PASS；HIP runtime 判定为**架构性不可达**（见审计 0004）；6750GRE 不在 PCI 总线 | AMDGPU LLVM static codegen；HIP 只能靠裸机 Linux 或换官方支持 dGPU | C3 math/reduction/batched matmul与Qwen 4,532/4,532 static lowering通过（graph v2 计数；v3 为 4,550，AMD 静态 compiler 尚未对新图重跑）；2026-09-10 可行性审计：gfx1036 不在 ROCm 10.0.0 / Radeon / WSL / HIP SDK for Windows 任何官方表内；WSL2 GPU-PV 架构下没有 `/dev/kfd`；第三方一手证据显示 ROCm 7.1.1 裸机可跑 gfx1036 kernel（wave32、2 CU、Fast F16，仅第三方参考） | 静态 LLVM/ELF与host oracle不等于HIP执行；MFMA/WMMA/INT8 dot/XNACK 仍 unknown；WSL 经 Mesa d3d12 可跑 GL compute（已实测），但那是 GLSL→DXIL 路径，不能执行我们的 AMDHSA ELF，且无 bf16/int8/subgroup 扩展；不能代表6750GRE性能 |
| 鲲鹏 920B ECS（SVE256） | **已释放（2026-09-11，用户指示）**；按需重建 | 原生 AArch64/SVE256 功能、汇编，后续受控性能探测 | openEuler 22.03、HiSilicon、2 vCPU、GCC 10.3.1、KVM；HWCAP SVE=1、SVE2=0、VL=32；M1C1–M1I 功能/contract 与24层 synthetic decoder 已通过 | ECS 是 KVM guest，不代表裸机/整机性能；`iota/compare` 是 host-reference；**已停止计费**；重建用 `~/tools/ecs-920B/create.sh` + `setup-access.sh`；释放前 /root 工作目录已归档 `_meta/pypto-x/ecs920b-release-archive-20260911/`（758 文件核对一致）；**重建前 920B 线任务（B2 等）不发射** |
| 昇腾 A2（910B3）租用环境 | **暂停（2026-09-11 用户指示）**：机时预算 100 h、已用 ~6 h，暂停不消耗机时且文件保留；**待用户通知恢复，恢复前 A2 任务一律不发射**；**W8A-C 真机验收 PASS（限定式）**；W8H/W8I vllm-ascend 基线已产出并**独立验收 PASS**；容器 256 vCPU / 2014 GiB / 300 GB；CANN 9.0.0 + 驱动 25.2.0 | Ascend 真机验收：PTO-ISA NPU ST、CANN 样例、Ascend adapter 回归、vllm-ascend E2E/性能基线 | `npu-smi` 显示 1× 910B3、Health OK；`acl.get_soc_name()=Ascend910B3`（需先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`）；容器内有 vllm 0.21.0 / vllm-ascend 0.21.0rc1 / torch_npu 2.10.0（预置环境）；bisheng clang 15.0.5 | 出网受限：HTTPS 仅 gitcode/pypi 可达、github 与一切 22 端口被封；无 `/dev/net/tun`、无 NET_ADMIN；访问走自建中继上的密钥反向隧道（端点与访问脚本只在本地私有侧，不提交进本仓）；租用共享资源 → **约定串行**；**NPU 执行必须走卡锁 `/root/a2-npu-lock/`（只保护 NPU，下载/编译/环境准备可并行）** |
| QEMU AArch64 | 可用 | AArch64/SVE/SVE2 功能和编译验证 | `qemu-aarch64` 11.0.3；已验证 `max,sve256=on` 可报告 SVE/SVE2，VL=32 bytes | 不能代表鲲鹏吞吐、缓存、内存带宽或指令时序 |

## 网络代理

用户提供的 HTTP/HTTPS 代理用于加速受限网络下的下载（2026-09-10 实测）：

```text
本机（控制机）         http://127.0.0.1:14514        （pypi/pytorch CPU index 均 200）
GamePC Windows         http://127.0.0.1:14514        （curl.exe 实测 200）
GamePC WSL（NAT 模式） http://172.28.48.1:14514      （Windows 宿主网关地址；WSL 内 127.0.0.1:14514 不通）
```

直连公网同样可用，代理只作为加速/回退。代理地址不写入仓库凭据，也不改变 smoke 的离线约定。

## 5080 WSL 连接方式

SSH 默认 shell 是 Windows `cmd`，直接执行 `uname`、`true` 等 Unix 命令会失败。统一使用：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 192.168.101.5 \
  'wsl.exe -e bash -lc "uname -a; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"'
```

2026-09-09 恢复后探测结果摘要：

```text
Linux GamePC 6.6.87.2-microsoft-standard-WSL2
NVIDIA GeForce RTX 5080, 16303 MiB, KMD 610.62, CUDA UMD 13.3
Compute capability 12.0, later probe 11389 MiB free
GPU memory used 4460 MiB, GPU util 2%, compute processes none
WSL: 24 CPU, MemAvailable about 30 GiB
Clang 18.1.3, CMake 3.28.3, Python 3.12.3
libcuda.so.1 present
nvcc/NVRTC/CUDART/CUDA SDK headers/PyTorch/Triton absent
```

建议安装阶段按顺序验证：

```bash
command -v nvidia-smi
command -v nvcc
command -v clang
command -v cmake
command -v python3
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

`nvcc`/NVRTC/PyTorch 缺失时，完整 Toolkit 路线标记 `BLOCKED_TOOLCHAIN`，但可以先用现有 `libcuda.so.1`、手写 PTX 和 Python `ctypes` 验证 CUDA Driver API 后端。不得把 Linux UAPI 的 `/usr/include/linux/cuda.h` 误认成 CUDA SDK header。

CUDA C1 已完成并由独立验收代理在 5080 上确认：FP32/BF16 elementwise、BF16 RNE、非末轴 reduction、全负 reduce-max、rank-2 matmul、K=0、zero-work/canary、多操作链、empty reduce-sum、重复 launch 与 context/module/allocation cleanup 均通过；验收前后无 compute process 残留。证据位于 `../worktrees/_meta/pypto-x/integration-w4-cuda-c1-final/validation.json`。由于 `nvcc` 缺失，统一 smoke 仍按规范标记 `BLOCKED_TOOLCHAIN`；这不否定不依赖 Toolkit 的 Driver/PTX 实测，也不构成 NVVM/Tensor Core/性能结论。

CUDA C2 在此基础上补齐 math、compare/iota/position、broadcast/where、layout/indexing、多输出 split、rank-3/4 matmul、scalar SSA 和 Qwen portable composites。独立验收为 `476 passed, 7 skipped`，GPU common vendor/ABI 定向测试 `45 passed`，CPU 联合回归 `260 passed`；5080 前后无 compute process 残留。证据位于 `../worktrees/_meta/pypto-x/integration-w4-cuda-c2-final/validation.json`。数学仍使用 PTX approximate 指令，布局/索引/batched matmul 仍是通用 correctness kernel。

M1I 在 exact integration HEAD `ec60f95979a56a9646d84f163912e677e9eb08ac` 上完成24层无权重 decoder connectivity。真实 `(B=1,T=1,past=4096)` profile 的 scalar、vector-common、AVX2、AVX-512、SVE256、GPU common 与 CUDA lowering/compile 均通过且 vector iteration domain 全部 compact；缩小但拓扑等价的24层 synthetic graph 已在 scalar、AVX2、AVX-512、QEMU SVE256、920B native 和 RTX 5080 Driver/PTX 实际执行通过。证据位于 `../worktrees/_meta/pypto-x/integration-w6-qwen35-decoder-connectivity-final-r3/validation.json`；这仍不代表带权整网推理或性能结论。

M1J 在 exact integration HEAD `0bc15d06ee167d5b6b5c62cffec88083dabb2bdf` 上冻结 external bytes/mmap binding contract。真实 profile 只计算3/320/48 input schema、368个 packed region 和1,574,877,952 bytes的 storage-relative layout metadata，没有创建、映射或读取权重文件；现有 scalar/AVX2/CUDA 对 byte memoryview 均为安全拒绝，backend ingestion 尚待下一阶段。证据位于 `../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-runtime-binding-final-r2/validation.json`。

M1K-CPU 在 exact integration HEAD `0bc662c6af9bdd32f587dbe2a0e9d0aec81290da` 上完成 typed byte-view ingestion。24层 synthetic Qwen graph 通过371个 external descriptors和43,200-byte packed storage在 scalar、AVX2、AVX-512、QEMU SVE256实际执行；AVX直接借用pointer，SVE有363个raw-wire payload并保留19个显式host-reference fallback。证据位于 `../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cpu-buffer-ingestion-final-r2/validation.json`。

M1K-CUDA 在 exact integration HEAD `fe6f3270b973b08be98cacfec04a6a4e9482e2b0` 上完成同步 borrowed-address HtoD/DtoH。raw byte views 不经 list pack/unpack；5080 上24层 external-buffer 图连续执行两次，input immutable、output canary 和 cleanup 均通过，全量 `522 passed, 7 skipped`。证据位于 `../worktrees/_meta/pypto-x/integration-w6-qwen35-bf16-cuda-buffer-ingestion-final/validation.json`。

## QEMU SVE/SVE2 验证

本地已有：

```text
/usr/bin/qemu-aarch64
/usr/bin/qemu-aarch64-static
/usr/bin/qemu-system-aarch64
/usr/bin/aarch64-linux-gnu-gcc
/usr/bin/aarch64-linux-gnu-g++
/usr/aarch64-linux-gnu
```

功能测试命令约定：

```bash
QEMU_CPU=max,sve256=on \
  qemu-aarch64 -L /usr/aarch64-linux-gnu <aarch64-test-binary>
```

这里 `VL=32` 表示 32 字节，即 256 位 SVE 向量长度。`QEMU_CPU=max` 的实现特征不等于鲲鹏 920B；QEMU 结果只能用于：

- AArch64 二进制能否启动；
- SVE/SVE2 指令和 predicate 逻辑是否正确；
- 动态 VL、尾块和 ABI 测试；
- 编译器生成代码的基本合法性。

不能用于：

- SVE 性能排名；
- 鲲鹏缓存/NUMA/带宽结论；
- 生产吞吐或功耗估计。

## 鲲鹏 920B ECS 接入结果

2026-09-08 已按接入清单完成首轮验证：

- `aarch64` little-endian、HiSilicon、2 vCPU、openEuler 22.03、KVM；
- `getauxval + prctl`：SVE=1、SVE2=0、VL=32 bytes；
- 原生 GCC triple：`aarch64-linux-gnu`；
- FP32/BF16 19 元素 masked tail、guarded canary、全负 reduce-max、9×5×10 matmul：`PASS`；
- 原生热点反汇编：`whilelo=2`、`ld1w=3`、`st1w=4`、z=36、p=19、NEON v/q=0。
- M1C2a compact layout：FP32 transpose/slice SVE indexed gather、BF16 identity SVE u16 copy、BF16 reorder scalar fallback、未对齐 descriptor、tamper 与 canary 均 `PASS`。
- M1C2b compact indexing：split/concat、int32/int64 gather/embedding、native bad-index/hash/descriptor 与 wire version pairing 均 `PASS`；BF16 gather/embedding 为明确 scalar fallback。
- M1D portable composites：BF16 RMSNorm、stable Softmax、partial RoPE、GQA、decoder/attention 多操作链与 guarded canary 均 `PASS`。
- M1E functional Conv1D/state：integration HEAD `41523cfc3` 的静态 SVE ELF 在原生机器执行 concat/slice/cast/broadcast/mul/add/silu、guarded canary、Conv 输出和 `new_state` carry 均 `PASS`；未加载模型权重。
- M1F attention/KV：integration HEAD `2d7da37b9` 的 SVE v5/vector-plan v2 ELF 执行 guarded self-test、rank-4 按 head 调用 rank-2 runner、BF16 KV concat/state carry 与输入不变性均 `PASS`；这是正确性 fallback，不是 fused batched kernel。
- M1G GDR recurrent state：integration HEAD `962f422e1` 在原生机对固定 `[1,16,128,128]` 非零 FP32 state 执行 48 次 rank-2 runner 调用，guarded self-test、输入不变性和 prediction/state/output golden 均 `PASS`；chunk 仍是 sequential reference，T=128 尚未验收。

基础证据见 `../worktrees/_meta/pypto-x/integration-w4-gpu-common-final/logs/20260908T080242Z-ecs-920b-native-validation.json`；M1C2a/M1C2b 证据分别见 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2a-final/validation.json` 与 `../worktrees/_meta/pypto-x/integration-w6-qwen35-m1c2b-final/validation.json`；M1D/M1E/M1F/M1G 证据见对应的 `integration-w6-qwen35-*-final/` 目录。

接入时发现并修复了 native runtime 错误注入 cross sysroot、以及通过 compiler basename 误拒原生 `cc` 的问题。当前结果可作为原生功能与汇编证据；性能门槛仍需单独设计、固定 affinity/频率/工作集并重复测量。

## 鲲鹏机器接入清单

拿到机器后先运行：

```bash
uname -a
uname -m
lscpu
getconf LONG_BIT
cc --version
clang --version
```

然后检查：

```text
HWCAP.SVE
HWCAP2.SVE2
实际 SVE vector length（应确认是否为 256 bit）
NUMA 节点、CPU affinity、内存带宽工具
```

测试分层：

1. `cpu-scalar` 正确性；
2. SVE256 predicate/尾块正确性；
3. SVE/SVE2 多 VL 兼容性（若机器支持）；
4. 最后才做性能和 PMU 测量。

## AMD GamePC `gfx1036` 核显接入结果

> **勘误（2026-09-10）**：本节的 `4,532`/`4,098`/`844` 等计数属于 graph contract v2；v3（GDR decay 门修复后）为 `4,550`，AMD 静态 compiler 尚未对 v3 重跑。详见 `docs/00-handoffs/ERRATA.zh-CN.md` ERR-0001。

6750GRE 实测无法安装，当前以 GamePC 核显作为临时 AMD 目标。2026-09-10 轻量探测确认：

- Windows：`AMD Radeon(TM) Graphics`、`DEV_13C0`、状态 `OK`；
- Windows OpenCL：device name `gfx1036`、preferred work-group multiple 32；
- Windows Vulkan：integrated GPU；
- WSL：`/dev/dxg` 存在，`/dev/kfd` 与 `/dev/dri` 不存在；
- WSL：无 `rocminfo`、`rocm-smi`、`hipcc`、ROCm/HSA runtime 和 device library；
- Clang 18 注册了 `amdgcn/gfx1036` 名称，但只能作为静态 target 信号。

因此当前目标名冻结为 `amd-igpu-gfx1036`，状态为 `BLOCKED_DEVICE`。在 WSL 真正提供 `/dev/kfd`、ROCm/HIP runtime 后，仍需重新取得：

- `rocminfo` 的 exact agent/gfx/wave 证据；
- FP16/BF16/INT8、MFMA/WMMA、XNACK 和 local/global memory capability；
- 最小 HIP enumerate/vector-add 真机结果。

Windows OpenCL/Vulkan 设备可见不等于 Linux HIP 可用；当前只做静态 codegen，不做执行或性能承诺。证据见 `../worktrees/_meta/pypto-x/gamepc-amd-igpu-probe-20260910/validation.json`。

静态 C1 已在 exact integration HEAD `e6e8360702d39da9f11e6352b94714c6ed23901a` 上完成 identity/add/mul/full reduce-sum/rank-2 matmul 的 FP32/BF16-storage LLVM/AMDGPU codegen。5个 artifact 都是 `elf64-amdgpu/gfx1036`，含 global load/ALU/store/bounds；Qwen metadata-only capability 为844/4,532 ops。runtime仍为 `BLOCKED_DEVICE`。证据见 `../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-final/validation.json`。

静态 C2 在 exact integration HEAD `45e8459e79ea979beb58661eb77b30a50f133f75` 上增加cast/layout/indexing/control。18类artifact和23/23 tamper通过，Qwen capability为4,098/4,532；剩余434项是math、通用reduction、batched matmul。descriptor为O(rank+segments)，未生成整网kernel。证据见 `../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c2-final-r2/validation.json`。

静态 C3 在 exact integration HEAD `bcf9516e6419d232338988238ffa8f79e59079c1` 上补齐 `exp/rsqrt/sigmoid/silu/softplus`、任意合法轴的 `reduce_sum/max/mean` 和 rank-2/3/4 exact-batch matmul。独立验收为 `577 passed, 7 skipped`；每个数学算子执行718个FP32样本及65,536个BF16 raw storage输入，10个host-adapted production LLVM reduction/matmul case exact PASS。Qwen `(B=1,T=1,past=4096)` 一次完整lower得到4,532/4,532 plans，约10.51 MB LLVM IR、约95.4 MiB峰值RSS；没有整图object/link或模型权重。证据见 `../worktrees/_meta/pypto-x/integration-w5-hip-gfx1036-static-c3-final/validation.json`。HIP运行态仍为 `BLOCKED_DEVICE`。

## 昇腾 A2（910B）租用环境接入

```text
计算资源   容器 256 vCPU / 2014 GiB 内存 / 300 GB 盘；1× Ascend 910B3（64 GB HBM）
软件栈     CANN 9.0.0（/usr/local/Ascend，含 driver 25.2.0、nnal、bishengir-compile、ccec、llvm-objcopy）
           容器内**没有** torch / torch_npu / numpy / pypto（需要时自行 pip，pypi 可达）
出网特征   443 仅 gitcode.com / pypi.org 可用；github.com 与所有 22 端口被封；无 TUN/NET_ADMIN
接入路径   A2 ──(自建中继上的密钥反向隧道，端点用真实 IP)──> 本机
           端点、跳板凭据与访问脚本**只在本地私有侧**（环境文件 + ~/tools/a2-910b/），**不得提交进本仓**
           访问脚本能力：状态检查 / 隧道重建（隧道活着时自举，不消耗平台 token）/ scp 通道
           A2 侧保活脚本每 5 s 自愈重连；重建时要确保只剩一个 supervisor（多实例会互抢监听端口）
```

**平台 token 很短期**（实测约 10 分钟），只在**首次引导**或容器重建后需要；隧道由 A2 侧 supervisor
每 5 s 自愈，之后所有运维（含重装隧道）都走隧道，不再依赖 token。历史踩坑：多个 supervisor 并存会
互相抢监听端口导致 5 s 抖动，重建隧道时要确保只剩一个 supervisor。

**注意**：中继上为此增开了**独立 sshd 实例**监听备用端口（与主 sshd 的 22 互不影响），
A2 的隧道公钥在中继侧 `authorized_keys` 中带 `restrict,port-forwarding,permitlisten=...` 限制。
具体端口、端点与配置文件路径属本地私有信息，不入仓。

**NPU 卡占用协议（2026-09-10 部署）**：

```text
位置      A2 容器固定目录 /root/a2-npu-lock/（README.zh-CN.md + a2_card_lock.sh）
流程      request → using → done；历史保留（文件即历史记录）
保护范围  **只保护 NPU 执行**；下载权重、编译、环境准备可并行进行，不需要持卡锁
等待/防僵 抢锁失败 180 s 轮询重试；TTL 6 h + PID 存活判定（owner 消失可回收）
时钟      A2 `date -u` 比真实 UTC 快约 8 h（按 CST 标 UTC），跨机对齐注意
运行验证  已由独立验收任务实际使用：全程 a2_card_lock.sh run/acquire，结束 .using→.done、
          无残留 vllm 进程、HBM 回落 3441/65536 MiB；临时目录已删除
入仓纪律  仓内只登记协议与目录名；端点/凭据/脚本实体只在本地私有侧
```

**真机验收结果（W8A-C，独立验收 PASS）**：PTO-ISA `tassign` NPU ST（910B3，exit 0 / 1 test PASSED）；
CANN mspti `callback_domain` 样例（aclnn Add，deviceId 4→0 后通过）；测试目录内注入式最小 hook
（固定 64×64 f32 add）真实 bisheng 编译 22,728 B / sha256 `4d7f704c…`、live 9/9、真机 `max_abs_err=0.0`。
`ascend_cann_bisheng_npu_regression_blocked` **限定式关闭**；新增 `ascend_hook_is_test_tree_injected_minimal_f32_add_only`、
`ascend_stable_cann_v9_2_0_beta2_not_device_validated`、`pypto_classic_pro_jit_opc_gym_not_ascend_validated`，
`real_native_cpp_ir_bridge_not_connected` 保持开启。这不代表 Core IR→PTO、classic/Pro JIT/OPC 或模型级。

## 资源分配

| 工作内容 | 首选资源 | 备用资源 | 验收结果 |
|---|---|---|---|
| Target ABI/Core IR | 本地 | 无 | IR snapshot、ABI 单测 |
| CCE 回归 | 有 Ascend/CANN 的机器 | PTO-ISA simulator | 既有 Ascend 测试不退化 |
| CPU scalar/x86 | 本地 | 无 | scalar golden、AVX 汇编检查 |
| SVE256/SVE2 | 鲲鹏 920B ECS（native） | QEMU（功能） | VL/predicate/尾块；性能结论另行冻结 |
| NVIDIA | 5080 WSL | 无 | CUDA artifact、运行结果 |
| AMD | GamePC `amd-igpu-gfx1036` | 静态 Clang amdgcn | 当前只做 gfx1036 target/codegen；HIP runtime `BLOCKED_DEVICE` |
| 9B/GDR | 后期各目标 | 5080 或鲲鹏 | 先单算子，再报告模型覆盖率 |

## 资源使用原则

- 跨项目锁的权威协议是 `/home/chiro/projects/.resource-locks/README.md`。`local` 保护本机 heavy，`gamepc` 只保护远端 heavy CPU/host-memory；对应 heavy 阶段只有 `resource-lock run` 成功才算取得。RTX 5080 GPU 由 PyPTO-X 独占，GPU-only 工作不申请 `gamepc`。
- 本机 full suite、大 shape lowering/compile、并行构建或预计使用至少一半 CPU/4 GiB 内存的任务必须经 `scripts/resource/run_local_heavy.sh`。返回 75/69 时等待，不能降级为裸跑。
- heavy runner 默认以 user cgroup 限制 MemoryHigh/MemoryMax、禁用该任务 swap、限制 CPU quota/affinity，并由 supervisor 每 2 秒检查 `MemAvailable`、任务树 RSS/CPU、load 与 PSI；资源日志写入 `../worktrees/_meta/pypto-x/resource-usage/`。
- Smoke 测试不加载模型、不需要模型路径、不产生大权重文件。
- 5080 先用于 elementwise、softmax、matmul 和 GPU ABI，不直接从 9B 端到端开始。
- SVE 已完成 QEMU 与鲲鹏 ECS native 功能验证；ECS KVM 数字暂不直接作为生产性能门槛。
- AMD 核显在 WSL 获得 `/dev/kfd` 与 ROCm/HIP 真机证据前不能把 HIP backend 标成“可运行”；最多标成“gfx1036 静态编译路径开发中”。
- A2 的 NPU 执行必须经 `/root/a2-npu-lock/a2_card_lock.sh` 串行（只锁 NPU；下载/编译/环境准备可并行）；引用 A2 结论必须带限定范围，不得写成"PyPTO Ascend 后端已通"。
- native compiler task 使用 `../worktrees/_meta/pypto-x/<task>/` 下独占的 `TMPDIR`、build、artifact 和日志目录，避免共享 `/tmp` 的容量/配额抖动，也避免不同 agent 共享未完成生成物。
