# AMD 核显 `amd-igpu-gfx1036` 可执行 HIP/HSA 运行态可行性调研报告

## 启动协议字段（原样记录）

```text
task_name=amd-igpu-runtime-feasibility
worktree=/home/chiro/projects/pypto/worktrees/pypto-x/amd-igpu-runtime-feasibility
branch=work/amd-igpu-runtime-feasibility
base=port/pypto-x-integration @ bcf9516e6419d232338988238ffa8f79e59079c1
started_at=2026-09-10T08:03:16Z
smoke_once=true
wait_timeout_seconds=3600
poll=false
resource_lock_root=/home/chiro/projects/.resource-locks
local_heavy_policy=locked
local_heavy_runner=/home/chiro/projects/pypto/pypto_x/scripts/resource/run_local_heavy.sh
local_min_available_mib=8192
local_safety_floor_mib=4096
local_max_cpus=6
```

| 项目 | 值 |
|---|---|
| smoke | `PASS`（target=host，仅执行一次，未加载模型） |
| smoke 日志 | `../_meta/pypto-x/amd-igpu-runtime-feasibility/logs/20260910T080316Z/smoke.log` |
| worktree HEAD | `bcf9516e6419d232338988238ffa8f79e59079c1`（无改动、无 commit） |
| 证据目录 | `../_meta/pypto-x/amd-igpu-runtime-feasibility/` |
| 最终状态 | `BLOCKED_DEVICE_CONFIRMED`（对 Windows+WSL2 路径）；同时新增 `BARE_METAL_LINUX_PATH_EXISTS`（第三方证据） |
| 软件安装 / 远端写操作 / 权重 | 均无 |

---

## 0 结论摘要

1. **官方支持面：`gfx1036` 不在任何一张 AMD 官方支持列表里。** ROCm 10.0.0 兼容矩阵、Radeon/Ryzen 的 Linux 与 WSL 矩阵、HIP SDK for Windows 系统要求表，全文检索 `1036` **0 命中**。AMD 在 HIP SDK 页明确写："If a GPU is not listed on this table, it is not officially supported by AMD." 而 **RDNA2 的 dGPU（gfx1030/gfx1031/gfx1032）在 Windows HIP SDK 表中已被标 ❌ Unsupported** —— gfx1036 是比"已废弃"更糟的"从未列入"。
2. **GamePC（Windows + WSL2）路径：HIP/HSA 不可行，且不是配置问题而是架构问题。** WSL2 走 WDDM GPU-PV，只提供 `/dev/dxg`（D3DKMT 通道），**在架构上不会出现原生 amdkfd 的 `/dev/kfd`**；AMD 官方 AMD SMI 文档原文已直接说明这一点。ROCm-on-WSL 依赖 `librocdxg` 桥接，而 librocdxg 的支持面是"Radeon dGPU + Ryzen AI/Strix 核显（gfx1150/gfx1151）"，**不含 Raphael/Granite Ridge（gfx1036）**。因此 `BLOCKED_DEVICE` 判定**正确**。
3. **但"gfx1036 完全跑不了 HIP"是错的。** 有两条独立的第三方一手证据表明：**裸机 Linux + ROCm 7.1.x + 发行版自带 amdgpu/KFD 驱动** 可以在 gfx1036 上建立 HSA agent 并执行 HIP kernel；其中一份 `rocminfo` 原文直接给出了本目标长期缺失的能力参数：**Wavefront Size = 32**、Compute Unit = 2、Workgroup Max Size = 1024、Max Waves Per CU = 32、Fast F16 Operation = TRUE、L1 16 KB / L2 256 KB。已知缺陷集中在调试器（rocgdb/TTMP）与 Tensile 库缺 gfx1036 产物，而非"不能执行"。
4. **本报告新增一项本机实测发现**：在 GamePC 的 WSL 里，通过 Mesa 25.2.8 的 `d3d12` 驱动经 `/dev/dxg`，**核显确实可被枚举并且真的执行了 GPU compute**（`GL_RENDERER = D3D12 (AMD Radeon(TM) Graphics)`，GL 4.6 Core，两次 dispatch 结果精确一致）。这说明"设备通道"是通的；缺的是 HIP/HSA 运行态。但这条路径是 **GLSL→DXIL/D3D12**，**不能执行我们产出的 `amdgcn-amd-amdhsa/gfx1036` ELF**。
5. **6750GRE 现状**：该卡**当前不在 PCI 总线上**（present-only 查询无此设备），Windows 里只有 `Present=False / Status=Unknown` 的幽灵 PnP 记录。所以"安装失败"首先是**硬件枚举层面**的问题，而不是纯驱动问题。是否物理在位需用户现场确认。
6. **建议**：短期不要为 gfx1036 投入运行时工程；把 AMD 证据边界明确冻结为"静态 C1–C3 + host oracle"。若确实需要 AMD 真机 HIP，**成本最低、证据最硬的路线是给 GamePC 加装一张官方支持的 AMD dGPU（或双系统装裸机 Linux）**，而不是在 WSL 里继续攻坚。

---

## 1 探测环境与方法

| 项目 | 值 |
|---|---|
| 远端 | GamePC `192.168.101.5`，SSH 默认进 Windows `cmd`，Linux 命令统一经 `wsl.exe -e bash -lc` |
| Windows | Microsoft Windows 11 `10.0.26200.9168` |
| WSL | `2.6.3.0`，内核 `6.6.87.2-1`，发行版 Ubuntu 24.04.4 LTS |
| WSL 图形栈 | Direct3D `1.611.1-81528511`，DXCore `10.0.26100.1-240331-1435.ge-release` |
| 本机 | 控制机 `/home/chiro/projects/pypto/pypto_x`（仅用于抓取官方/社区文档与读代码） |

**约束遵守情况**：未安装任何软件（无 apt/pip/driver 安装）；未执行 `wsl --update`；未修改 Windows/WSL 配置；未重启远端；未触碰 RTX 5080 的 CUDA 环境；**未在远端写入任何文件**（所有远端脚本均以 `base64(gzip)` 经 stdin 传入 `bash`/`python3` 执行）；未申请 `local`/`gamepc` 锁（全部为轻量只读探测）。

**探测命令记录**（完整原始输出见 `evidence/` 目录）：

```text
Windows: Get-PnpDevice -Class Display [-PresentOnly]
         Get-PnpDevice -PresentOnly | Where InstanceId -like 'PCI\VEN_1002*'
         Get-PnpDeviceProperty -InstanceId <id> -KeyName DEVPKEY_Device_LastArrivalDate,...
         Get-ChildItem C:\Windows\System32\lxss -Recurse
         Get-ChildItem C:\Windows\System32\DriverStore\FileRepository -Recurse -Include '*wsl*','*kfd*','*hsa*','*rocm*'
         dir C:\Windows\System32\lxss\lib
WSL:     ls -la /dev/dxg /dev/kfd /dev/dri; command -v rocminfo rocm-smi hipcc hipconfig clinfo vulkaninfo
         ldconfig -p | grep -iE 'libamdhip|libhsa-runtime|libOpenCL'; ls -d /opt/rocm*; lsmod; dmesg | grep -i dxg
         apt-cache policy rocm-hip-sdk; dpkg -l | grep -E 'mesa|vulkan'
         python3 ctypes: EGL surfaceless + Mesa d3d12 → GL_RENDERER / 扩展 / GLSL compute dispatch
```

---

## 2 问题 1：官方支持面

### 2.1 ROCm Linux 支持列表

最新正式版为 **ROCm 10.0.0**（Core SDK；消费级 Radeon/Ryzen 文档轨道为 7.2.1）。

[ROCm 10.0.0 兼容性矩阵](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) 的 "AMD APU series / Graphics model (iGPU)" 段确实**已经支持核显**，但列的是：

| LLVM target | 对应核显 |
|---|---|
| `gfx1151` | Radeon 8060S/8050S（Ryzen AI Max 300/400） |
| `gfx1150` | Radeon 890M/880M（Ryzen AI 300/400） |
| `gfx1152` | Radeon 860M/840M |
| `gfx1153` | Radeon 820M |
| `gfx1103` | Radeon 780M/760M/740M（Ryzen 200） |

**`gfx1036` 不在其中。** Ryzen AI Max / Strix Halo 是少见的 iGPU 官方支持案例：[Ryzen Linux 矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html) 写明 "Supported Architectures: gfx1150 gfx1151"。

### 2.2 ROCm on WSL 官方说明

- [WSL 支持矩阵（Radeon）](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)：ROCm 7.2.1；OS 为 Ubuntu 24.04.2 HWE / 22.04；Windows 侧要求 "Radeon Software for Windows 25.10.1 for WSL" / "Adrenalin 26.1.1 for WSL2"；**GPU 列表全部是 dGPU**（RX 9070 XT…RX 9060、RX 7900 XTX、PRO W7900/W7800/W7700），**无任何 iGPU**。
- [WSL How-to（Radeon）](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html)：Adrenalin 26.2.2 + ROCm 7.2.1 起 `librocdxg`（ROCDXG）进入生产支持，原文 "This release also marks the first time AMD is supporting Ryzen Strix and Strix Halo SKUs versus the legacy WSL solution."，即**首个被官方 WSL 支持的核显是 gfx1150/gfx1151**。
- 注意官方文档自相矛盾：[Ryzen WSL 矩阵页](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/wsl/wsl_compatibility.html) 仍停在 ROCm 6.4.2 且只列 dGPU。这属于官方文档滞后，不影响"gfx1036 不在列表内"的结论。

### 2.3 AMD HIP SDK for Windows 官方支持列表

[HIP SDK 系统要求](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html)：

- APU 页签：✅ 支持 `gfx1151` / `gfx1150`；
- Radeon 页签：✅ `gfx1201/gfx1200/gfx1100/gfx1101/gfx1102`；**RDNA2 的 `gfx1030/gfx1031/gfx1032` 已标 ❌ Unsupported**；
- `gfx1036` **根本不在表中**。原文："If a GPU is not listed on this table, it is not officially supported by AMD."

[HIP SDK 组件差异页](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/component-support.html)：Windows 侧缺少 Communication Libraries、AI Libraries（MIOpen/MIGraphX）、AI Frameworks，Runtime 闭源；HIP SDK 自称是 "a subset of the ROCm platform"。

### 2.4 `/dev/kfd` 缺失的直接原因（官方原文）

AMD 官方 [Using AMD SMI under WSL](https://rocm.docs.amd.com/projects/amdsmi/en/develop/how-to/amdsmi-wsl-mode.html) 说得最直接：

> "Under Windows Subsystem for Linux 2 (WSL2) those interfaces do not exist: there is no `/sys/class/drm` GPU tree and no `/dev/kfd`. The GPU is reached instead through the Windows WDDM display driver using the D3DKMT interface exposed by the `/dev/dxg` device node."

Linux 侧要求见 [install-on-linux prerequisites](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/prerequisites.html)：需要 amdgpu 内核驱动（含 amdkfd）提供 `/dev/kfd`。`/dev/dxg` 的来源见 [DirectX ❤ Linux](https://devblogs.microsoft.com/directx/directx-heart-linux/) 与 [GPU paravirtualization](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-paravirtualization)（WDDM GPU-PV）。

**直接原因结论**：WSL2 不是"Linux + 直通 GPU"，而是 **WDDM GPU-PV 半虚拟化**：内核里只有 `dxgkrnl`（本次实测 dmesg 确认 `hv_vmbus: registering driver dxgkrnl`），没有 `amdgpu`/`amdkfd`。因此 `/dev/kfd` 的缺失是**架构必然**，不是驱动没装好，也不是配置错误。**修 `/dev/kfd` 这条路在 WSL2 上不存在。**

### 2.5 TheRock 开发分支（唯一的官方仓库级希望，但非生产）

[TheRock SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md) 在 **Linux 与 Windows 两张表**中把 `RDNA2 | gfx1036` 都标为 ✅Build Passing / ✅Sanity Tested / ✅Release Ready。该文件原文同时声明：

> "This project is still under active development and is not yet stable for production use."
> "The official compatibility matrix should be consulted for AMD GPU support in released ROCm software, whereas the content on this page serves as a leading indicator for what will be referenced there."
> "A ✅ in the Build Passing column only indicates that a wheel or tarball is produced and published. It does **not** imply the runtime is functional on target hardware."

[RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) 的 device extra 表中也确有 `AMD Raphael iGPU | gfx1036 | device-gfx1036`（"Raphael" 正是 Ryzen 7000/9000 桌面 APU 代号）。但：

- [TheRock issue #1443](https://github.com/ROCm/TheRock/issues/1443)「[Issue]: gfx1036 Windows build fails」原文：gfx1036 target 在 rocBLAS configure 阶段失败，原因是 Tensile 不支持 gfx1036，日志为 `Unsupported GPU target: gfx1036 ... Supported targets are: gfx803;gfx900;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1010;gfx1011;...`（该 issue 现为 CLOSED）；
- RELEASES.md 里 WSL 章节仍被 HTML 注释掉：`<!-- TODO: include WSL documentation once we are satisfied with quality in nightly releases -->`。

**判定**：TheRock 的 ✅ 属 roadmap / 开发口径，**不能当作已发布的官方支持**。

### 2.6 官方支持面小结

| 平台 | gfx1036 官方 HIP/HSA | 依据 |
|---|---|---|
| 裸机 Linux | ❌ 不在列表 | ROCm 10.0.0 兼容矩阵 |
| WSL2 | ❌ 不在列表；且架构上无 `/dev/kfd` | WSL 矩阵 + AMD SMI 文档 |
| Windows 原生 | ❌ 不在列表 | HIP SDK 系统要求表 |

---

## 3 问题 2：社区 / 非官方路线现状与风险

### 3.1 `HSA_OVERRIDE_GFX_VERSION`

- **不是官方记录在案的开关**：ROCm latest / HIP / Radeon-Ryzen 三个文档站的 `searchindex.js` 中该词出现 0 次。
- **实现位于 libhsakmt**（[rocm-systems topology.c L1285–1305](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/libhsakmt/src/topology.c#L1285-L1305)）：语法 `major.minor.stepping`，**显式禁止跨 major 覆盖**（`Device is gfx%d, override requested gfx%d not allowed`），所以 gfx1036(10.x) 只能覆盖成 `10.y.z`；另有逐节点变体 `HSA_OVERRIDE_GFX_VERSION_<node_id>`；覆盖只写 `OverrideEngineId`，真实 `EngineId` 保留。
- **对 gfx1036 没有任何公开的成功实例**（`gfx1036 + HSA_OVERRIDE_GFX_VERSION=10.3` 检索 0 命中）。`10.3.0` 只是"同 major 内最合理候选"，属未验证推断。
- 典型失败模式：rocBLAS 找不到 Tensile 库 `Cannot read .../TensileLibrary.dat: No such file or directory for GPU arch : gfx1036`（[koboldcpp#676](https://github.com/LostRuins/koboldcpp/issues/676)、[sd-webui-amdgpu#551](https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu/issues/551)）；torch wheel 无 gfx1036 code object → `hipErrorInvalidKernelFile`（[unsloth#8792](https://github.com/unslothai/unsloth/issues/8792)、[#7669](https://github.com/unslothai/unsloth/issues/7669)）。
- 注意：`HSA_OVERRIDE_GFX_VERSION` 对**我们自己编译的 kernel**意义有限 —— 它影响的是运行时对 agent 上报 gfx 版本的覆盖，主要用来复用别家预编译库；对 PyPTO-X 这种自带 `--offload-arch=gfx1036` 的场景并非关键。

### 3.2 ROCmLibs 类社区库替换

[likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)：替换 Windows 侧 `%HIP_PATH%\bin\rocblas\library`（Tensile `.dat`/`.hsaco`）与 `rocblas.dll`。

- **确实覆盖 gfx1036**：最新 [v0.7.1.1（2026-05-19，对应 HIP SDK 7.1.1）](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.7.1.1) 资产含 `gfx1034-gfx1035-gfx1036.7z`。
- 风险：强版本配对（必须与已装 HIP SDK 版本严格匹配）；手工替换系统 DLL；**依赖 Windows HIP SDK 本身能在该 GPU 上工作**——而 HIP SDK 的 runtime 是闭源的且 gfx1036 不在支持表内，因此这条路**不能解决"没有 HIP 运行态"的根本问题**，只能解决"rocBLAS 库缺 gfx1036 内核"这一子问题；许可证声明含糊（"The license may also comply with this if there is any other needs"）；README 明确不建议直接用其 Linux 产物。

### 3.3 gfx1036 的专属产物与成功报告

**官方源码层面确有进展（构建目标层面）**：[rocm-libraries#2297](https://github.com/ROCm/rocm-libraries/pull/2297)（Tensile 加入 gfx1036，2025-10-27 提交、已 merged）、[TheRock#1629](https://github.com/ROCm/TheRock/pull/1629)（RDNA2 gfx103X 家族构建）、[MIOpen#3684](https://github.com/ROCm/MIOpen/pull/3684)/[#3907](https://github.com/ROCm/MIOpen/pull/3907)（gfx1036 device feature words）、[rocm-systems#5717](https://github.com/ROCm/rocm-systems/pull/5717)（gfx1036 加入 wkmi）。

**执行层**：**没有端到端成功证明**，但有两条很有分量的间接证据：

1. [legacy-rocm-build#6604](https://github.com/ROCm/legacy-rocm-build/issues/6604)（OPEN，2026-08-13，**ROCm 7.1.1**，kernel 7.1.7-200，GPU `Granite Ridge [Radeon Graphics] (rev c9) (gfx1036)`，Ryzen 9 **9950X3D**）：用户以 `hipcc -O0 -g --offload-arch=gfx1036 -o repro repro.cpp` 编译并运行，缺陷只是 **KFD 未清零 TTMP 导致 rocgdb 崩溃**（`HSA_DBG_DISPATCH_INFO_ALWAYS_VALID` 已置位但不该置位），即**执行本身是通的，坏的是调试器**。该 issue 贴出的完整 `rocminfo` 输出（本报告最有价值的能力证据，见下）。
2. [llama.cpp#14068](https://github.com/ggml-org/llama.cpp/issues/14068)：Linux 容器中 HIP 后端成功识别 `gfx1036 (0x1036), VMM: no, Wave Size: 32` 并加载 `libggml-hip.so`（后续在该 Docker/K8s 组合下崩溃，属集成问题）。

**`rocminfo` 原文（来自 #6604，第三方一手 dump，非本项目实测）**：

```text
******* Agent 2 *******
  Name:                    gfx1036
  Vendor Name:             AMD
  Feature:                 KERNEL_DISPATCH
  Profile:                 BASE_PROFILE
  Float Round Mode:        NEAR
  Max Queue Number:        128(0x80)
  Queue Type:              MULTI
  Device Type:             GPU
  Cache Info:  L1: 16(0x10) KB   L2: 256(0x100) KB
  Chip ID:                 5056(0x13c0)
  Cacheline Size:          128(0x80)
  Max Clock Freq. (MHz):   2200
  Compute Unit:            2
  SIMDs per CU:            2
  Shader Engines:          1
  Coherent Host Access:    FALSE
  Memory Properties:       APU
  Fast F16 Operation:      TRUE
  Wavefront Size:          32(0x20)
  Workgroup Max Size:      1024(0x400)
  Workgroup Max Size per Dimension: x 1024  y 1024  z 1024
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    1024(0x400)
  Grid Max Size:           4294967295
  Max fbarriers/Workgrp:   32
```

> ⚠️ **证据强度声明**：以上是**第三方**在**另一台机器**（9950X3D）上、**非官方支持配置**下的 dump，不是本项目在 9900X 上的实测。因此它**可以把 wavefront=32 记录为"有第三方一手证据"，但不能升级为本项目的能力实测记录**；MFMA/WMMA/INT8 dot 仍未证实（RDNA2 无 MFMA，WMMA 是 RDNA3+ 特性，gfx1036 两者都不该有）。

### 3.4 WSL 侧的社区结论

- WSL 上核显跑 HIP 的公开案例**只出现在 gfx1151**（Strix Halo），经 `librocdxg` 接入。
- 安装脚本常误用**已失效的 KFD 检查**（例如 [unsloth#7314](https://github.com/unslothai/unsloth/pull/7314) 修正了这个问题）——本项目 `python/pypto/backends/hip/runtime.py::probe_amd_hip_runtime` 也以 `/dev/kfd` 为门禁，属同类问题（见第 4.3 节）。
- **没有 gfx1036 在 WSL 跑 HIP 的公开报告。**

### 3.5 社区路线风险总结

| 风险 | 说明 |
|---|---|
| 许可/再分发 | 替换官方 DLL/DSO 属修改发行物；HIP SDK 组件再分发受 AMD EULA 约束 |
| 混装损坏 | 版本错配/ABI 风险；SDK 升级会覆盖替换件，必须可回滚 |
| 性能预期 | 2 CU @2200 MHz，FP32 峰值约 **1.1 TFLOPS（估算，非实测）**，共享 DDR5 带宽（理论 ~90 GB/s 且与 CPU/显示竞争）。transformer 算子是带宽瓶颈 → 定位只能是"跑通/验证正确性" |
| 误导性结论 | Windows OpenCL/Vulkan/ZLUDA 可见 ≠ HIP 可用；识别到设备 ≠ 算子正确；静态 lowering ≠ HIP 真机 PASS |

---

## 4 问题 3：要在该核显上执行 PyPTO-X 生成的 gfx1036 代码，缺一不可的清单

先说清楚**我们的产物到底是什么**（读自本 worktree）：

- `python/pypto/compiler/targets/hip.py::AmdgpuStaticCompiler` → `render_amdgcn_llvm()` 生成 `target triple = "amdgcn-amd-amdhsa"`、`define amdgpu_kernel` 的 LLVM IR；再经 `llvm-as` + `llc -march=amdgcn -mcpu=gfx1036 -filetype=obj` 得到 **ELF relocatable object**。
- 这是 **HSA 目标**（AMDHSA ABI），必须由 **HSA loader**（`libhsa-runtime64` / ROCr）经 **amdkfd** 加载执行。

因此要真正跑起来，以下组件**缺一不可**：

| # | 组件 | 作用 | 现实可行性 |
|---|---|---|---|
| 1 | **amdgpu 内核驱动（含 amdkfd）** | 提供 `/dev/kfd`、`/dev/dri`，管理 GPU 队列 | 裸机 Linux：✅ 内核自带（Raphael 支持已成熟）。**WSL2：❌ 架构上不可能**（GPU-PV 无 amdgpu） |
| 2 | **`/dev/kfd` 访问权限**（udev 规则 `KERNEL=="kfd", GROUP=..., MODE="0660"`） | 允许用户态打开 KFD | 裸机 Linux：✅ 常规配置 |
| 3 | **HSA runtime（ROCr / `libhsa-runtime64`）** | agent 枚举、code object 加载、队列派发 | 裸机 Linux：⚠️ ROCm 7.1.x 实测可枚举 gfx1036（第三方证据）；**Windows HIP SDK：Runtime 闭源且 gfx1036 不在支持表** |
| 4 | **gfx1036 的 ROCm device library / oclc bitcode** | clang 链接 LLVM IR、生成可加载 code object 所需 | ❌ 本项目 `llc -filetype=obj` 路径当前**不依赖** device library（自包含 IR），但**要链接成完整 HSA code object 并加载**就需要 HSA runtime 配套 |
| 5 | **`libamdhip64`（HIP runtime）+ `hipcc`** | HIP API/工具链门禁 | 裸机 Linux：⚠️ 需装 ROCm（非官方支持）；WSL/Windows：❌ |
| 6 | **ROCm 数学库中的 gfx1036 kernel（Tensile 等）** | rocBLAS/hipBLAS 等 | ❌ Tensile 缺 gfx1036 产物（[TheRock#1443](https://github.com/ROCm/TheRock/issues/1443)、[rocm_sdk_builder#103](https://github.com/lamikr/rocm_sdk_builder/issues/103)）；社区库替换可部分绕过（仅 Windows） |
| 7 | **（若走 WSL）`librocdxg` + Adrenalin for WSL 驱动** | 经 `/dev/dxg` 桥接 | ❌ librocdxg 支持面为 Radeon dGPU + gfx1150/gfx1151，**不含 gfx1036** |
| 8 | **固件 / MES uCode** | GPU 管理固件 | ⚠️ 无证据表明是阻塞点（#6604 的机器 uCode 28/9 正常工作），**不做结论** |

**结论**：在 GamePC 现有 Windows+WSL2 形态下，第 1、7 条**在架构上就无法满足**，因此 HIP/HSA 运行态**不可得**——这与"gfx1036 硬件能力不足"是两码事。

### 4.3 一处需要修正的项目内部假设

`python/pypto/backends/hip/runtime.py::probe_amd_hip_runtime()` 以 **`/dev/kfd` + `/dev/dri` + `libamdhip64` + `libhsa-runtime64` + `hipcc` 同时存在**为门禁。这对"原生 Linux ROCm"是对的，但对**官方支持的 ROCm-on-WSL 形态会产生假阴性**——WSL 路径用的是 `/dev/dxg` + `librocdxg`，**本来就没有 `/dev/kfd`**。当前因为 gfx1036 不在官方 WSL 列表内，这个假阴性**不影响结论**；但若将来 AMD 把某核显加入 WSL 支持，该探针需要改成"`/dev/kfd` **或** (`/dev/dxg` + `librocdxg.so`)"。建议作为低优先级 follow-up 记录。

---

## 5 问题 4：替代路线评估

### (a) 换用官方支持的 AMD dGPU

| 路线 | 最低型号（官方） | 代价 / 前置条件 |
|---|---|---|
| Linux 裸机 ROCm | **RX 7600 (gfx1102)** 或 **RX 9050/RX 9060 (gfx1200)** | 需要一台能装 Linux 的机器 + 显卡 + ROCm 安装；双 ✅ |
| WSL2 官方支持 | **RX 9060 / RX 7700** 起 | 需要 Windows 侧安装 **Adrenalin 26.2.2+ for WSL**、WSL 侧装 `librocdxg` |
| Windows 原生 HIP SDK | gfx1102 / gfx1200 等 | 注意 **RDNA2 dGPU（gfx1030/1031/1032）在 Windows 已被标 ❌** ——不要买旧 RDNA2 卡走 Windows |

**要避开的坑**：现行矩阵已取消 Full/Partial 分级；Windows HIP SDK 表是"是否列出 + ✅/⚠️/❌"三分法，**RDNA2 dGPU 在 Windows 上是死路**。

### (b) 让 6750GRE（gfx1031）复活

**本机实测（原始输出 `evidence/ghost-6750gre.txt`）**：

```text
Get-PnpDevice -Class Display -PresentOnly   →  只有 GameViewer 虚拟显示、RTX 5080、AMD Radeon(TM) Graphics
Get-PnpDevice -Class Display                →  额外存在 Present=False / Status=Unknown 的记录：
    AMD Radeon RX 6750 GRE 12GB  PCI\VEN_1002&DEV_73DF&SUBSYS_445A1DA2&REV_E5\8&523F35E&0&000000500011
    AMD Radeon RX 6750 GRE 12GB  PCI\VEN_1002&DEV_73DF&SUBSYS_445A1DA2&REV_E5\6&38E5097D&0&00000009
    NVIDIA GeForce GT 610 / AMD Radeon 760M Graphics / Intel(R) Iris(R) Xe Graphics （均为幽灵）
Get-PnpDevice -PresentOnly | Where InstanceId -like 'PCI\VEN_1002*'  →  只有 DEV_13C0（核显）与 DEV_1640（HD Audio）
DEVPKEY_Device_LastArrivalDate（6750GRE 第一个实例）= 2026-09-09 23:46:50（本地，UTC+8 → 2026-09-09T15:46:50Z）
```

**来源与判断**：

- 来源是 **Windows PnP 设备树**（`Get-PnpDevice -PresentOnly` / `Present` 属性 / `DEVPKEY_Device_LastArrivalDate`），即设备管理器背后的同一数据源；同表中还有 GT 610、760M、Iris Xe 等明显属于**其它机器/历史配置的幽灵条目**，说明 `Present=False` 的条目在本机上确实不代表当前硬件。
- 判断：**该卡当前没有被 PCI 总线枚举**（`VEN_1002` 的 present 设备只有 `DEV_13C0` 与 `DEV_1640`）。它最后一次被枚举是 **2026-09-09 23:46:50（本地时间）**，与"6750GRE 安装失败"的时间点吻合。
- 因此「安装失败」**首先是硬件枚举层面的问题**：卡要么已被拔出、要么插着但未被 BIOS/PCIe 枚举（供电、PCIe 插槽/转接、BIOS Above-4G/Resizable BAR 等），**不能简单归因为驱动或 ROCm 软件问题**。软件侧无法远程区分这两种情况。

**复活条件**：① 现场确认卡物理在位且被 BIOS 识别（`lspci` / BIOS 里能看到 `1002:73DF`）；② Windows 能稳定枚举出设备（现在是"短暂出现后消失"）；③ 再谈驱动/ROCm。**若 BIOS 层面都看不到，任何软件手段都无解。** 这一步需要**用户现场操作**，不是 agent 能做的。

### (c) Windows 原生 HIP SDK 路线

- **可行性：不可行（对 gfx1036）。** gfx1036 不在 [HIP SDK 支持表](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html) 中，且该表明确 "If a GPU is not listed on this table, it is not officially supported by AMD."
- 本机实测 `C:\Program Files\AMD\ROCm` **不存在**，未安装任何 HIP SDK；AMD 驱动包（u0203303，`amdkmdag.sys`）中也没有 WSL/KFD/HSA/ROCm 组件。
- 即便硬装 HIP SDK + 社区 ROCmLibs 替换 rocBLAS：**HIP runtime 是闭源的**，对不在支持表的 GPU 是否初始化成功无保证，且 Windows 侧缺少 MIOpen/MIGraphX 等。**不建议投入。**
- 补充：即便 Windows HIP 能跑，我们的后端产物是 **AMDHSA ELF**，走的是 Linux ROCr/HSA 路径；Windows HIP 是另一套加载路径，属于新增工程量。

### (d) 非 HIP 运行态：Vulkan / OpenCL / SPIR-V —— **本项目新增实测**

这是本次调研**唯一的一手新增实验发现**。做法：在 GamePC 的 WSL 里用 Mesa 25.2.8 的 `d3d12` gallium 驱动（`GALLIUM_DRIVER=d3d12`）经 EGL surfaceless + `/dev/dxg` 访问宿主核显。

**枚举结果（原始输出 `evidence/gl_capabilities.txt`、`evidence/gl_precision.txt`）**：

```text
VENDOR   = Microsoft Corporation
RENDERER = D3D12 (AMD Radeon(TM) Graphics)      ← 宿主 AMD 核显，经 D3D12 半虚拟化通道
VERSION  = 4.6 (Core Profile) Mesa 25.2.8-0ubuntu0.24.04.2
NUM_EXTENSIONS = 207
MAX_COMPUTE_WORK_GROUP_INVOCATIONS = 1024
MAX_COMPUTE_SHARED_MEMORY_SIZE     = 32768
MAX_SHADER_STORAGE_BLOCK_SIZE      = 134217728
MAX_SSBO_BINDINGS                  = 80
GL_ARB_compute_shader / GL_ARB_shader_storage_buffer_object / GL_ARB_shader_image_load_store /
GL_ARB_shader_atomic_counters / GL_ARB_shader_ballot / GL_ARB_gpu_shader_int64 / GL_AMD_gpu_shader_int64
```

**真实执行结果（原始输出 `evidence/gl_compute_fp32_256.txt`）**：

```text
GLSL compute shader:  layout(local_size_x = 64) in;  b[i] = a[i] * 2.0 + 1.0;
shader_compile=PASS, link_status=1, glGetError=0
first6_got  = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
first6_want = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
all_match=True
```

另一次 4096 元素、`b[i] = a[i]*3.0+1.0`、**连续两次独立 dispatch** 的测试两次均 `all_match=True`（`FP32_DETERMINISTIC_AND_CORRECT=True`）。

**精度能力（实测编译失败/成功）**：

```text
GL_EXT_shader_16bit_storage →  error: extension `GL_EXT_shader_16bit_storage' unsupported in compute shader
GL_EXT_shader_8bit_storage  →  error: extension `GL_EXT_shader_8bit_storage' unsupported in compute shader
```

即该路径**只有 32 位与 int64 算术**；**没有 FP16/BF16/INT8 原生类型**，BF16（本项目主力 dtype）必须靠位操作 + 显式 RNE 舍入软件模拟。

**能力与代价清单**：

| 维度 | 能力 | 代价 / 限制 |
|---|---|---|
| 精度 | FP32、INT32/UINT32、INT64（`GL_ARB_gpu_shader_int64`） | **无 FP16/BF16/INT8 存储与算术扩展**；BF16 需软件模拟（uint32 打包 2×bf16 + 手动 RNE） |
| 矩阵 | 无张量核心路径 | gfx1036 本身无 MFMA/WMMA；matmul 只能 FMA 展开 |
| 工作组 | 1024 invocations、32 KB shared、SSBO 128 MB 块、80 个 SSBO 绑定 | 与本项目 GPU common 的 Grid/Block/Thread 语义需要一层新的映射 |
| 同步 | `GL_ARB_shader_image_load_store`、atomic counters、`glMemoryBarrier` | 无 HSA 那样的细粒度队列/信号量语义 |
| 子组 | **`GL_KHR_shader_subgroup` 不可用**（查询返回非法值）；只有 `GL_ARB_shader_ballot` | wave32 语义无法通过子组内建暴露 |
| 需要新增的 target 层 | **GLSL compute（或 SPIR-V/DXIL）后端** + 一套 GL/EGL 运行时宿主 | 这是**与 HIP/ROCDL 完全不同的 ISA/ABI**：产物不是 AMDHSA ELF，而是 GLSL/SPIR-V，由 Mesa d3d12 → DXIL → D3D12 → Windows 驱动执行 |
| 依赖风险 | Mesa `d3d12` 驱动在 Ubuntu 24.04 是**非官方支持路径**（AMD/Microsoft 均未背书）；Ubuntu 的 `mesa-vulkan-drivers` **不含 dzn**，WSL 里也没有 OpenCL（`libOpenCL.so.1` 缺失） | 该路径可用性随 WSL/Mesa/驱动升级而变 |

> ⚠️ **最重要的边界（父 agent 特别要求明确写出）**：
> **这条 D3D12/GL-compute 路径不能执行我们产出的 gfx1036 代码。** 本项目 C1–C3 的产物是 `amdgcn-amd-amdhsa` **ELF**（AMDHSA ABI，需要 HSA loader + amdkfd）；而 GL-compute 路径执行的是 **GLSL/SPIR-V → DXIL/D3D12**，由 Windows 图形驱动编译执行，**两者 ISA、ABI、加载器、内存模型全部不同**。要走这条路，必须**新增一个完整的 target**（Core IR → GLSL/SPIR-V），并且要重新实现 BF16 模拟、布局/索引、reduction/matmul 全部 lowering，**与现有 HIP/ROCDL 路径不共享任何 codegen**。

**是否值得作为独立目标提交用户决策？——建议：暂不。** 理由：

1. 它**不能复用**已有 C1–C3 的 AMDGPU codegen，等价于"再做一个新后端"，工程量与 CUDA 后端同量级；
2. 它带来的是 **OpenGL/D3D12 依赖**（Mesa d3d12 是 Ubuntu 打包的非官方路径），工程上是"用一个不受支持的路径替换另一个不受支持的路径"；
3. 它**不能为 HIP 路线提供任何验收证据**（ISA/ABI 不同），对项目"AMD GPU：GPU 公共层 → HIP/ROCDL"的目标没有推进；
4. 若目标只是"在 AMD 硬件上做真机正确性验证"，同上成本下**买一张官方支持的 dGPU（RX 7600/RX 9060）更直接、证据更硬**。
   唯一例外场景：如果用户明确想要"零采购、纯软件、在现有 GamePC 上让 AMD 核显跑点东西"，那这条是**目前唯一真实可行**的路径（本报告已实测跑通 FP32 compute）。

### (e) 纯 static + host oracle 继续作为唯一 AMD 证据

**边界（必须保持的表述纪律）**：

- 静态 C1–C3 的 AMDGPU ELF/LLVM 产物 + host oracle 证明的是**语义正确性与目标描述符完整性**；**不等于** HIP 真机执行、不等于 wave/BF16/INT8/MFMA 能力、不等于性能。
- 现有 `577 passed, 7 skipped`、4,532/4,532 static lowering 均在 `BLOCKED_DEVICE` 下成立，文档中不得出现"HIP PASS"字样。
- 这条边界**在可预见期内是稳定且诚实的**：它为未来真机接入保留了 ABI/描述符契约，是低成本高价值的保持项。

### (f) 【新增·值得单列】裸机 Linux 路线（第三方证据支持）

这是本次调研中**最接近"让 gfx1036 真正跑 HIP"的路线**，且成本可估：

- **前置条件**：给 GamePC（或另一台 AMD 机器）装一个**裸机 Linux**（不是 WSL），发行版内核自带 amdgpu（Raphael/Granite Ridge 支持成熟），再装 **ROCm 7.1.x+**（含 2025-10 后合入的 Tensile gfx1036 支持）。
- **预期**：`/dev/kfd` 出现 → `rocminfo` 能列出 `gfx1036` agent（第三方已在 9950X3D 上证实）→ HSA loader 可加载 AMDHSA code object → **PyPTO-X 的 C1–C3 产物有真机执行的可能**。
- **风险**：非官方支持配置；已知 `rocgdb`/TTMP 缺陷（[#6604](https://github.com/ROCm/legacy-rocm-build/issues/6604)，OPEN）不影响非调试执行；Tensile/rocBLAS 的 gfx1036 库需确认；**本机是双卡（RTX 5080 + 核显），改双系统会影响 NVIDIA 独占验证资源**，需要用户权衡。
- **代价**：需要用户决策 + 现场操作（分区/双系统或另一台机器），不是 agent 能自动完成的。

---

## 6 问题 5：建议

### 下一步应该做

1. **冻结 AMD 证据边界**：AMD 侧继续以"静态 C1–C3 + host oracle"为唯一证据，文档保持 `BLOCKED_DEVICE` 表述；把本报告新增的第三方 `rocminfo`（wave32、2 CU）**登记为"第三方参考，不是本项目实测"**。
2. **低优先级 follow-up（不改结论）**：修正 `probe_amd_hip_runtime()` 的门禁语义，使其对官方 ROCm-on-WSL 形态（`/dev/dxg` + `librocdxg`，无 `/dev/kfd`）不产生假阴性。
3. **把决策权交回用户**，明确给出三条可选路线及其代价：
   - **路线 A（推荐，若需要 AMD 真机 HIP）**：给 GamePC 加装/启用一张官方支持的 AMD dGPU（**RX 7600 gfx1102 或 RX 9060 gfx1200**），Linux 或 Windows 双 ✅。**切勿买 RDNA2 dGPU 走 Windows**。
   - **路线 B（若必须用现有核显且要跑起来）**：给该机装**裸机 Linux + ROCm 7.1.x**，这是唯一能让 gfx1036 拿到 `/dev/kfd` + HSA agent 的现实路径（第三方已验证 agent 可枚举）。放弃性能预期（2 CU）。
   - **路线 C（若完全不能改硬件/系统）**：接受 AMD 信号只有静态证据；**不建议**改做 GL-compute/SPIR-V 新后端（见 5(d)）。
4. **6750GRE：先做现场硬件确认**（BIOS 是否枚举到 `1002:73DF`），再谈软件。软件侧无需任何动作。

### 下一步不该做

- ❌ **不要在 WSL 里继续找 `/dev/kfd`**：这是 GPU-PV 架构约束，官方文档已明说，不存在配置解法。
- ❌ **不要装 ROCm/HIP SDK 去赌 gfx1036**（Windows 或 WSL 都不行）；也**不要**用社区 ROCmLibs 替换库来"制造"支持 —— 它解决不了 runtime 层缺失。
- ❌ **不要把 Windows OpenCL/Vulkan 可见、或本报告的 D3D12 compute 成功，表述为 HIP 可用**。
- ❌ **不要为 gfx1036 启动 HIP runtime 实现工作**（没有可执行的运行态，写了也无法验收）。
- ❌ **不要在没有真机证据的情况下写"可以运行 HIP"**。

### 需要用户提供 / 决策

1. **6750GRE 的物理状态**：卡是否还插在机器上？BIOS/开机自检能否看到它？（这是"安装失败"归因的分水岭。）
2. **是否允许给 GamePC 装裸机 Linux（双系统）**，或使用另一台机器？——这是让 gfx1036 拿到 HIP 运行态的**唯一现实前提**。
3. **是否考虑采购一张官方支持的 AMD dGPU**（RX 7600 / RX 9060 级别）？预算与采购意愿。
4. **是否接受 AMD 侧长期只有静态证据**（即把 AMD target 定位为"ABI/描述符就绪，等硬件"）。

---

## 7 证据清单

| 文件 | 内容 |
|---|---|
| `evidence/wsl_rocm_state.txt` | WSL 设备/工具/库/内核模块/apt 状态（含时间戳） |
| `evidence/gl_capabilities.txt` | GL_RENDERER、限制值、两次 FP32 dispatch 确定性结果 |
| `evidence/gl_precision.txt` | 207 条扩展、相关扩展清单、FP16/INT8 存储扩展编译失败原文 |
| `evidence/gl_compute_fp32_256.txt` | 256 元素 compute dispatch 原始输出（`all_match=True`） |
| `evidence/ghost-6750gre.txt` | Windows PnP present-only / 幽灵设备 / `LastArrivalDate` 原始输出 |
| `validation.json` | 结构化证据（含启动协议、smoke、全部命令与结论、URL 出处） |
| `logs/20260910T080316Z/smoke.log` | 一次性 smoke 日志 |

**官方/社区来源 URL 全部在正文中以 markdown 链接给出**；无法访问的来源：`https://community.amd.com/t5/general-discussions/rocm-on-gfx1036/m-p/616823`（JS 动态渲染，抓取只得到导航骨架，内容无法证实，**未采用其任何结论**）。另：本会话 `web_fetch` 工具对 `rocm.docs.amd.com` 与 `raw.githubusercontent.com` 返回 `URL hostname resolves to a non-public IP address`，相关原文改用 `curl` 直接抓取。

---

## 8 未解决风险与不确定项

1. **gfx1036 的真机能力参数仍非本项目实测**：wave32、2 CU、L2 256 KB、Fast F16 均来自第三方的 `rocminfo` dump（另一台机器）。**MFMA/WMMA/INT8 dot/VMM/XNACK 仍为 unknown**（RDNA2 架构上本不应有 MFMA/WMMA）。
2. **裸机 Linux 路线的可行性未在本项目环境验证**：第三方证据支持"agent 可枚举 + kernel 可执行"，但 ROCm 版本、内核版本、Tensile 库完整性、以及我们 C1–C3 产物能否被 HSA loader 接受（`llc -filetype=obj` 产出的 relocatable object 是否具备可加载的 code object 元数据）**都还没有验证**。
3. **6750GRE 的物理状态未确认**：`Present=False` 只能证明"当前未被 PCI 枚举"，无法远程区分"卡不在机器上"与"卡在但未被枚举"。
4. **6750GRE 与核显的历史混淆风险**：幽灵记录里同时存在 `AMD Radeon 760M Graphics (DEV_15BF)`、`Intel Iris Xe`、`NVIDIA GT 610`，说明本机 PnP 库含其它机器的遗留条目；引用 `Get-PnpDevice -Class Display` 全量结果时**必须过滤 `Present`**。
5. **D3D12/GL 路径**依赖 Mesa `d3d12`（Ubuntu 打包的非官方路径）。本次实测的 `GL_RENDERER`、扩展集与 compute 成功结果**只代表当前 WSL `2.6.3.0` + Mesa `25.2.8` + 驱动 `32.0.21045.5002` 组合**，升级后可能失效。
6. **WSL `dxgkrnl` 启动日志异常**：dmesg 出现多次 `dxgkio_query_adapter_info: Ioctl failed: -22/-2`。本次未深究其归属（可能与被禁用的显示适配器或虚拟显示驱动有关），**不作为任何结论依据**，仅登记为观察项。
7. **性能结论一概不谈**：本次没有任何性能测量；2 CU 的数字均为估算。
