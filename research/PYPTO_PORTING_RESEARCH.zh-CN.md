# PyPTO 跨平台移植研究

更新日期：2026-09-03（Asia/Shanghai）

## 结论先行

PyPTO 不是一个可以通过替换几组 intrinsic 就完成移植的算子库。公开实现实际包含五层：Python 前端、Tensor/Tile/Block/Execution IR 与 Pass、目标代码生成、设备编译工具链、以及 Host/Device 调度运行时。当前公开代码中的“跨平台”主要指 Ascend A2/A3/A5/Kirin 的跨代兼容，而不是 Ascend、CPU、NVIDIA、AMD 之间的跨架构兼容。

技术上可以移植，但建议采用以下路线：

1. 先把目标描述、编译器驱动、运行时和算子 lowering 定义成稳定的 Target ABI。
2. 保留 Tensor/Scalar/Shape/控制流等目标无关语义；不要让新的通用 IR 继续暴露 AIC/AIV、MTE1/2/3、L0A/B/C、TPUSH/TPOP 等 Ascend 专有名词。
3. 非 Ascend 后端在目标无关 IR 后分叉；不要把 PTO-ISA 逐条机械翻译为 NEON、AVX、CUDA 或 HIP。
4. `pypto_pro` 比旧 `pypto` 更适合作为新后端的工程起点：它已有 SPMD、SIMD、SIMT、统一 IR、`CodegenBase` 和算子注册机制。但目前 `GetBackend()`、JIT 编译器和运行时仍只接 CCE，必须先解耦。
5. 鲲鹏优先做 `CPU scalar/reference -> AArch64 NEON -> SVE/SVE2`。鲲鹏 920 7260 的官方 FAQ 明确写明不支持 SVE，因此 NEON 应是首个真实硬件基线；SVE/SVE2 必须由 capability 探测选择。
6. x86_64 与 AArch64 应共享一个 CPU 后端，只在最后一级向量 lowering 和微内核派发处分叉。AMD GPU 与现有 NVIDIA 工作应共享一个 GPU 中层和 Runtime ABI，再分别下降到 ROCDL/HIP 与 NVVM/CUDA。
7. 第一项模型级验收不应宣称“整网由 PyPTO 编译”。公开 Qwen3.5-9B 案例是 Transformers 整网加一个 PyPTO Gated Delta Rule prefill 融合算子；decode 未替换。新后端先复现这种集成方式最现实，也最容易公平对比。

当前许可证是一个需要并行推进的治理事项，而不是技术研究的停止条件。维护方已向用户表示愿意看到移植，这是积极信号；在发布或合并非华为处理器后端之前，仍应以新许可证、双许可证或明确书面授权完成闭环。

## 研究范围和证据级别

本文区分三类信息：

- **已验证**：直接来自本地固定提交的源码、文档或在本机运行得到的结果。
- **上游声明**：仓库或官方平台文档中的声明，本文未在对应硬件上复测。
- **用户提供信息**：NVIDIA 9B 移植正在进行；目前未找到可公开索引的对应分支或实现，因此不对其覆盖范围和性能作推断。

源码、提交、校验值和下载位置见 [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md)。

## 1. 公开实现不是一条代码线

### 1.1 GitCode 官方实现

本地位置：[upstream/pypto](../upstream/pypto/)

这是 PyPI 元数据所指向的 `cann/pypto` 官方仓；本次研究快照的 `master` 为 `388ddce68700`。它同时包含：

- 旧 `pypto`：Tensor Graph → Tile Graph → Block Graph → Execution Graph，设备侧 MPMD 调度。
- 新 `pypto_pro`：面向 Ascend 950 的 SPMD + SIMD/SIMT Kernel DSL。
- 编译 Pass、CCE/PTO 代码生成、CANN runtime adapter、仿真与调试工具。
- 706 个 `docs/` Markdown 源文件以及完整图片资源。

旧 `pypto` 的默认 `PVC2_OOO` 策略在 [pass_manager.cpp](../upstream/pypto/framework/src/passes/pass_mgr/pass_manager.cpp) 中列出 45 个 Pass。前几项仍较接近通用 Tensor 优化，后续很快进入内存类型、图分区、AIC/AIV 子图、同步、乱序调度和设备代码生成。

`pypto_pro` 的抽象更接近其他平台：

- 外层为 SPMD block。
- 规则计算可以用二维 Tile SIMD。
- 不规则访问可以用 Grid/Block/Thread/Warp 形式的 SIMT。
- `Backend` 保存算子到 pipe/codegen callback 的注册表。
- `CodegenBase` 为目标代码生成器提供共同接口。

但当前可执行路径仍是单后端：

- [backend_cce.cpp](../upstream/pypto/framework/src/interface/pypto_pro/backend/backend_cce.cpp) 的 `GetBackend()` 固定返回 `BackendCCE::Instance()`。
- [compile_config.py](../upstream/pypto/python/pypto_pro/runtime/compile_config.py) 的 `get_jit_compile_config()` 对非 `cce` 直接抛出 `NotImplementedError`。
- [jit.py](../upstream/pypto/python/pypto_pro/runtime/jit.py) 只接受 `a2/a3/a5`，调用毕昇编译器并链接 CANN `runtime`/`profapi`。
- [cce_codegen.cpp](../upstream/pypto/framework/src/interface/pypto_pro/codegen/cce/cce_codegen.cpp) 把 SIMT 限定到 A5 Vector target。
- 四个 CCE 算子注册文件中共有 301 个 `REGISTER_BACKEND_OP` 注册点（另有 backend 主文件中的一个注册点）；完整覆盖不是一个小补丁。

### 1.2 GitHub community implementation

本地位置：[upstream/pypto-community](../upstream/pypto-community/)

`hw-native-sys/pypto` 自称 community-driven implementation，本次研究快照为 `690da78458ad`。把官方 GitCode 历史取入后执行 `git merge-base` 没有共同祖先；两边不能按“镜像”或“可直接 cherry-pick 的分支”处理。

这条实现线的优点是边界文档更完整，并已将 Ascend 代际差异集中到 `BackendHandler`：

- [backend handler 文档](../upstream/pypto-community/docs/en/dev/backend/00-backend_handler.md)
- [backend.h](../upstream/pypto-community/include/pypto/backend/common/backend.h)
- [backend_handler.h](../upstream/pypto-community/include/pypto/backend/common/backend_handler.h)

它适合作为 Target ABI 设计参考，但现有抽象仍属于“多个 Ascend 后端”，尚不是“多个硬件架构族”：

- `BackendType` 只有 `Ascend910B` 和 `Ascend950`。
- [memory_space.h](../upstream/pypto-community/include/pypto/ir/memory_space.h) 把 `Vec/Mat/Left/Right/Acc/Bias/LeftScale/RightScale` 固化进公共 IR。
- [pipe.h](../upstream/pypto-community/include/pypto/ir/pipe.h) 把 `MTE1/MTE2/MTE3/M/V/FIX` 与 `CUBE/VECTOR` 固化进公共 IR。
- `BackendHandler` 的许多 hook 是 `RequiresGMPipeBuffer`、`RequiresVtoCFractalAdapt`、`GetL0cMAlignment` 一类 Ascend 差异开关。
- Python runtime 仍硬编码 `a2a3/a2a3sim/a5/a5sim`。

所以，文档中“新增 backend 无需改 Pass/Codegen”对新增 Ascend 代际大体成立，对 CPU、CUDA、HIP 并不成立。

### 1.3 PTOAS、PTO-ISA 与 simpler

community 实现的完整链路是：

```text
Python DSL
  -> PyPTO IR / Passes
  -> InCore .pto -----------------> PTOAS -> pto-isa C++ -> AICore binary
  -> orchestration C++ ----------------------------------> AICPU binary
                                                             |
                                                             v
                                                      simpler runtime
```

- [PTOAS](../upstream/PTOAS/) 是 MLIR-based PTO assembler/optimizer，但当前 dialect、内存空间、同步和 lowering 都是 Ascend/Da Vinci 语义。
- [pto-isa](../upstream/pto-isa/) 提供 90+ Tile 指令、Ascend A2/A3/A5 实现和 CPU simulator。
- [simpler](../upstream/pypto-community/runtime/) 执行 Host ↔ AICPU ↔ AICore 任务图。

因此非 Ascend 后端不应被迫经过 PTOAS/pto-isa/simpler。最稳妥的架构是保留现有 Ascend 分支，同时为 CPU/GPU 建立独立 lowerings。

## 2. CPU simulator 的准确定位

PTO-ISA README 把 x86_64/AArch64 列为平台，但同一文档也明确将 CPU 路径定位为功能验证和调试。源码进一步说明它不是生产 CPU 后端：

- [parallel.hpp](../upstream/pto-isa/include/pto/cpu/parallel.hpp) 使用 `std::thread` 和 Clang/GCC vectorization pragma。
- [TMatmul.hpp](../upstream/pto-isa/include/pto/cpu/TMatmul.hpp) 是三重循环、元素访问和 `std::fma`，没有 NEON/SVE/AVX intrinsic。
- 在当前 12 vCPU、支持 AVX2/AVX-512 的 x86_64 主机上，`python3 tests/run_cpu.py --demo gemm --verbose --no-install` 成功；`M=32,K=16,N=32` 的最大误差为 `1.19209e-07`，实测约 `0.11 GFLOPS`，编译器还报告内层循环未能 vectorize。

它非常适合继续承担三项职责：

1. PTO/Tile 指令的语义 oracle。
2. 新后端 differential testing 的参考结果。
3. 无设备 CI 的正确性门禁。

它不应直接演变成性能后端，因为其数据布局仍为了模拟 Ascend L0/UB/NZ，并且每次并行区创建线程的模型不适合 CPU 推理热路径。

## 3. 可复用边界与必须重做的部分

| 层 | 可复用程度 | 建议 |
|---|---:|---|
| Python AST、装饰器、错误定位 | 高 | 保持用户语法，新增 target 参数，不复制每个后端的 parser |
| Scalar/Tensor/Shape/Stride/DType/控制流 IR | 高 | 定义为真正 target-neutral 的 Core IR |
| Tensor 算子语义与 golden | 高 | 复用；把支持矩阵改为 capability 查询 |
| Tile 的逻辑 shape/valid-shape/mask | 中高 | 保留逻辑语义，物理布局延迟到 target lowering |
| `Vec/Mat/Left/Right/Acc` | 低 | 下沉为 Ascend dialect；通用层使用 `global/workgroup/private/matrix_fragment` |
| `AIC/AIV` 与 TPUSH/TPOP | 低 | 下沉为 Ascend execution dialect；CPU/GPU 使用各自调度模型 |
| 45 个旧 Pass | 混合 | Tensor canonicalization 可复用；分区、内存、同步、调度需按 target pipeline 重组 |
| PTOAS/PTO-ISA | Ascend 高，其他低 | 保持 Ascend 后端；非 Ascend 不走这条链 |
| CANN adapter / simpler | 低 | 新建 Runtime ABI；分别实现 CPU/CUDA/HIP/CANN adapter |
| PyPTO-Gym golden 与模型接入 | 高 | 作为跨后端验收，但去除 `torch_npu`/NPU device 假设 |

### 为什么不能机械映射内存空间

| 现有概念 | Ascend 含义 | CPU 合理映射 | GPU 合理映射 |
|---|---|---|---|
| GM/DDR | 设备全局内存 | 普通虚拟内存/NUMA 内存 | device global memory |
| Vec/UB | 显式片上 scratchpad | 不存在同等可编程存储；只能是栈/arena/寄存器/缓存优化 | shared memory 或寄存器，取决于生命周期 |
| Mat/L1 | Cube staging buffer | packed panel / cache-resident scratch | shared memory / matrix operand fragment |
| Left/Right/Acc | L0A/L0B/L0C | 微内核内部 A/B packing 与 accumulator | Tensor Core/MFMA/WMMA 的 opaque fragment |
| MTE pipes | 独立 DMA engine | load/store loop、prefetch 或同步 memcpy | async global→shared copy pipeline（若目标支持） |
| AIC + 2×AIV | 异构 core group | 没有直接对应物 | GPU SM 内是同构线程；不能假装是两个 core 类型 |

把 CPU cache 或 GPU register 当成用户可寻址的 `L0A/L0B/L0C` 会形成错误的公共契约。公共 IR 应只表达意图和生命周期，物理空间由目标 lowering 决定；专家模式再允许进入目标专属 dialect。

## 4. 建议的目标架构

```text
                         Python APIs
                   old pypto / pypto_pro
                              |
                              v
        Core IR: Tensor + Scalar + Shape + SCF + logical Tile
                              |
                 canonicalize / fuse / tile / verify
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
   Ascend execution IR     CPU kernel IR        GPU kernel IR
   AIC/AIV/MTE/PTO         loops + vector       grid/block/thread
          |              fixed/scalable vec     subgroup + memory
          v                   |                   |
   PTOAS/CCE/simpler    +-----+------+       +----+----+
                       |            |       |         |
                    AArch64       x86_64  NVIDIA     AMD
                   NEON/SVE2    AVX/AMX   NVVM     ROCDL
```

### 4.1 Target ABI 不应继续增长为布尔开关集合

建议至少拆成四个稳定接口：

```python
@dataclass(frozen=True)
class TargetSpec:
    triple: str                    # aarch64-linux-gnu, x86_64-linux-gnu, nvptx64, amdgcn
    device_kind: str               # cpu, cuda, hip, ascend
    execution_models: frozenset[str]  # simd, simt, mpmd
    subgroup_sizes: frozenset[int]
    scalable_vector: bool
    memory_spaces: Mapping[str, MemorySpaceSpec]
    dtypes: frozenset[DType]
    op_capabilities: Mapping[str, OpCapability]
    matrix_capabilities: tuple[MatrixCapability, ...]
    async_copy: tuple[AsyncCopyCapability, ...]

class CompilerBackend(Protocol):
    def lower(self, module: CoreIR, target: TargetSpec) -> TargetIR: ...
    def compile(self, module: TargetIR, options: CompileOptions) -> Artifact: ...

class RuntimeBackend(Protocol):
    def properties(self, device: int) -> DeviceProperties: ...
    def current_stream(self, device: int) -> Stream: ...
    def load(self, artifact: Artifact) -> Module: ...
    def launch(self, kernel: Kernel, grid, block, args, stream) -> None: ...
    def synchronize(self, stream) -> None: ...
```

`TargetSpec` 是数据；真正不同的算法由 target-specific rewrite/lowering pattern 实现。应避免继续添加 `RequiresFooForA2` 一类方法，因为这种接口会把某一代硬件的 workaround 固化为所有后端都必须理解的概念。

### 4.2 推荐的编译技术

长期建议将目标无关 Kernel IR 下降到 MLIR 标准层次：

- CPU：`scf/affine/linalg/vector -> LLVM`，必要时进入 `arm_neon`、`arm_sve` 或 x86/AMX intrinsic。
- GPU 公共层：`gpu + vector`。
- NVIDIA：`nvgpu/NVVM -> NVPTX -> PTX/CUBIN`。
- AMD：`amdgpu/ROCDL -> AMDGPU -> HSACO`。

这条路线与 MLIR 已有的可伸缩 `vector<[N]xT>`、GPU address space、NVVM 和 ROCDL 基础设施吻合，也避免为 CUDA 与 HIP 各写一套完整的高层优化器。

为了尽快得到第一个可运行版本，可以先让 `CodegenBase` 生成 C++/CUDA/HIP 源码：CPU 用系统 Clang，NVIDIA 用 NVRTC，AMD 用 HIPRTC。该 PoC 应保持同一 Target ABI 和 IR 边界，之后才能替换为 MLIR lowering，而不影响前端与 runtime。

## 5. 各目标的具体路线

### 5.1 AArch64：NEON、SVE、SVE2

建议只实现一个 `cpu-aarch64` 后端，内部提供多个 code variant：

- `scalar`：正确性基线。
- `neon`：固定 128-bit，AArch64 的首个生产实现，也是鲲鹏 920 的目标。
- `sve`：向量长度不可在公共 IR 中写死；使用 predicate 做尾块。
- `sve2`：在 SVE 基础上为整数、重排、dot/matrix 类操作增加专用 pattern。

关键原则：

- Tile shape 是算法块大小，不等于 SIMD 寄存器宽度。
- SVE 内层维度必须 vector-length agnostic；用运行时 VL 与 predicate，不把 `256/512 bit` 写进模型语义。
- Linux 上通过 HWCAP/HWCAP2 或经过验证的派发库选择 variant。
- 热路径使用持久线程池并考虑 NUMA/affinity；不要沿用 CPU simulator 每个 parallel region 创建 `std::thread` 的方式。
- Elementwise/reduction 可以直接 vector lowering；GEMM 第一阶段调用 BLAS/Arm Compute Library/KleidiAI，之后再做可融合微内核。
- Google Highway 同时覆盖 NEON、SVE、SVE2、AVX2、AVX-512，并带运行时派发，适合快速 CPU PoC；但矩阵乘和复杂融合仍需单独微内核或 MLIR lowering。

### 5.2 x86_64 SIMD

x86 与 AArch64 共用 Tensor→loop→vector 的绝大部分逻辑，只把末端变体换为：

- baseline/SSE4（按最低部署要求决定）
- AVX2 + FMA
- AVX-512（细分 BF16/VNNI/FP16 capability）
- AMX（后续，仅矩阵路径）

必须同时检查 CPU 与 OS 保存扩展寄存器状态的能力，不能仅按 CPUID 位选择 AVX/AVX-512。缓存键应包含 target triple、CPU feature set、ABI 版本、dtype/layout/tile config，而不能只含 kernel 名。

### 5.3 AMD GPU

推荐先支持 datacenter CDNA，再决定是否覆盖 RDNA。两者不能用一个硬编码的 warp 大小：AMD 官方 HIP 文档给出 CDNA 通常 wave64、RDNA 通常 wave32，并明确要求 portable code 不假定 32 或 64。

第一阶段：

- 把 `pypto_pro` SIMT 的 grid/block/thread、barrier、atomic 映射到 GPU IR/HIP。
- Tensor GM → GPU global，block scratch → LDS，thread-private → VGPR/private。
- 同步 copy-in/compute/copy-out 先保证正确。
- Matmul 先调用 rocBLAS/Composable Kernel，或使用 rocWMMA；不要第一版手写 MFMA lane layout。

第二阶段：

- global→LDS async copy、软件流水和 occupancy-aware tile selection。
- 针对 gfx target 注册 MFMA/WMMA capability；matrix fragment 必须 opaque，因为不同 CDNA/RDNA 代际的 lane/VGPR layout 会变化。
- HIPRTC 或 MLIR ROCDL 生成 code object，runtime 用 HIP module/stream/event API。

### 5.4 NVIDIA 与正在进行的工作

NVIDIA 的自然映射是 SIMT + shared memory + warp32 + MMA/WGMMA；NVRTC 可完成运行时 CUDA C++→PTX，MLIR NVVM 可表达 shuffle、barrier、atomic、`cp.async`、TMA、MMA/WGMMA。

现有 NVIDIA 移植的最高价值不仅是“能跑 9B”，而是尽快抽出可由 AMD 复用的部分：

- `GpuTargetSpec`
- Grid/Block/Thread/Subgroup IR
- GPU address spaces 与 barrier 语义
- artifact/cache 格式
- Torch tensor/stream binding
- module load/launch/error/profiling ABI

CUDA 专属的 warp32、PTX、Tensor Core fragment、TMA 和 CUDA Graph 必须留在 NVIDIA lowering。否则 AMD 后端会再次重构公共层。

## 6. 选择哪条 PyPTO 作为基线

| 方案 | 优点 | 主要问题 | 结论 |
|---|---|---|---|
| 官方旧 `pypto` | 与现有 Gym、Tensor API、MPMD 和 9B GDR 直接一致 | 45-Pass 大流水线、CANN/MPMD 深度耦合，Tile/Block API 当前也未完全开放 | 作为兼容前端和模型验收，不宜直接复制整条后端 |
| 官方 `pypto_pro` | SPMD/SIMD/SIMT 与 CPU/GPU 更同构；已有 IR、backend registry、CodegenBase | 当前只支持 CCE；主要面向 A5；与旧 Tensor kernel 不同 | **推荐作为新 Kernel IR/后端解耦起点** |
| GitHub community | BackendHandler、PTOAS 边界、Pass 文档和验证体系成熟 | 与官方无共同 Git 历史；公共 IR 仍是 Ascend-native | 设计参考；是否作为代码基线由维护方明确决定 |

推荐在写大量后端代码前，让维护方回答两个问题：

1. `pypto_pro`/新 IR 是否是官方长期收敛方向？
2. community implementation 是否会并入、替代或长期独立于 GitCode 官方线？

如果答案是 `pypto_pro` 为长期主线，可采用“旧 `pypto` Tensor 前端 → neutral Core IR/Pro IR”的兼容 lowering，以保留已有模型代码，同时让新平台只实现一条后端。

## 7. 最小可行产品与验收顺序

### M0：上游和 ABI 对齐

交付物：

- 一页 Target ABI RFC。
- 目标/feature/cache key 命名约定。
- 许可证或书面授权的预期时间点。
- NVIDIA 移植作者的接口、commit、模型、dtype、硬件和覆盖算子清单。

成功标准：三方同意 CPU、NVIDIA、AMD 不各自发明一套 backend/runtime 接口。

### M1：目标无关前端与 IR

只覆盖 Scalar、Tensor、动态 shape、view/reshape/transpose、for/if、dtype cast、logical Tile/valid shape。将现有 golden 测试搬到无设备 CI。

成功标准：同一 Python kernel 对不同 target 生成结构合法的 target IR；不需要设备。

### M2：CPU reference backend

实现 contiguous FP32/FP16/BF16 的 load/store、add/sub/mul/div、cast、exp/rsqrt、sum/max、transpose/view、matmul（可先调用库），以及 Torch CPU tensor 绑定。

成功标准：hello world、elementwise、softmax、matmul 与 PyTorch golden 一致；artifact cache 可重复命中。

### M3：NEON 与 x86 SIMD

共享 vector lowering，分别产生 NEON 和 AVX2/AVX-512 变体；检查生成汇编并做 runtime dispatch。

成功标准：

- 尾块、非对齐、动态 shape 正确。
- 汇编中出现预期 SIMD 指令，而不是只靠源码里写了 pragma。
- 与 scalar reference differential test 全通过。
- elementwise/reduction 吞吐显著高于 reference；不预设一个脱离硬件的倍率门槛。

### M4：SVE/SVE2

在具备真实 SVE/SVE2 的机器或可靠模拟器上加入 scalable-vector variant。

成功标准：同一二进制/同一 IR 在至少两种 VL 设置下正确；尾块完全由 predicate 处理；无固定 VL 泄漏进缓存键之外的用户语义。

### M5：GDR 与 9B

先移植 [当前通用 GDR forward](../upstream/pypto-gym/src/pypto_gym/ops/pypto_tensor/qwen3_5/gdr_fwd/) 所需的最小算子集，而非追求全部 301 个 Pro backend op：

```text
view / assemble / full / cast
add / sub / mul / div
exp / rsqrt / sum
transpose / concat / matmul
dynamic loop + valid-shape/tail
```

先跑 Gym 的 `B=2,T=512,H=4,HV=4,D=128,BT=128`，再跑 32K 长序列、varlen、initial state。最后用 PyTorch custom op/monkey-patch 替换 Qwen3.5-9B 的 chunk-prefill GDR。

模型报告必须分别列出：

- PyPTO 覆盖的层/算子比例。
- prefill TTFT。
- decode tok/s。
- 峰值内存。
- 编译时间和 cache-hit 时间。
- 数值误差及最终 token 是否一致。

### M6：GPU 公共层与 AMD

合并 NVIDIA 工作中与厂商无关的 GPU IR/runtime，再实现 HIP/ROCDL。先复现 elementwise/softmax/matmul/GDR，之后再做 async copy、matrix core 和图捕获。

### 暂不进入 MVP

- 旧 PyPTO 全部 45-Pass 行为逐项等价。
- 全部 301 个 Pro CCE op。
- Ascend MPMD 在 GPU 上的逐机制复刻。
- 分布式通信、跨 die/superpod 语义。
- 所有 dtype/layout/量化组合。

这些项目会显著推迟第一个可验证后端，并且其中一些在 CPU/GPU 上应有不同语义。

## 8. 9B 案例应如何解读

公开的 [Qwen3.5-9B 说明](../upstream/pypto-gym/modeling/transformers/qwen3_5_9b/README.md) 已确认：

- 模型主体来自 Transformers 5.8.1。
- 通过 `sys.modules` 注入和 monkey-patch 只替换 `Qwen3_5GatedDeltaNet.forward` 的 chunk-prefill GDR。
- decode 和 full-attention 路径不受影响。
- 文档在 Ascend 910B3 上报告自然 prompt TTFT `405.6 -> 110.4 ms`（3.67x）、256-token TTFT `438.5 -> 105.1 ms`（4.17x）；decode 都约 10 tok/s。
- greedy token 可能因 BF16 微差而分叉，虽然逐 forward logits 很接近。

还有一个版本细节：2026-09-01 的 `3270199` 以“Remove repeat ops”为由删除了专门的 `qwen3_5_9b/gated_delta_rule` 目录，但模型 README 仍指向该路径。完整 Git 历史已下载，可以从删除前提交读取原实现；当前通用实现位于 `qwen3_5/gdr_fwd`。

因此，针对用户提到的 NVIDIA 9B 工作，建议先确认“run a 9B model”具体指：

1. 整网所有核心计算都经 PyPTO；
2. 整网 PyTorch/CUDA + 一个或若干 PyPTO fused op；
3. 仅模型形状的 kernel harness。

三者都很有价值，但不能用同一种覆盖率和性能表述。

## 9. 许可证并行路线

当前 [CANN Open Software License 2.0](../upstream/pypto/LICENSE) 第 2.1/3.1 条把使用、修改和衍生作品限制在华为 AI 处理器/软件场景。维护方已向用户表达欢迎移植的意向，可以支持技术预研和 RFC 沟通；这类积极反馈仍不是许可证文本，发布前最好完成以下任一项：

- 核心 frontend/IR/backend ABI 改为 Apache-2.0、BSD-3-Clause 等宽松许可证；Ascend 专属目录继续使用 CANN License。
- 对核心做 Apache-2.0 + CANN 双许可证。
- 对明确仓库、分支、贡献者、目标平台给出书面例外授权。

最利于生态的目录/许可边界是：

```text
pypto-core          permissive: frontend, neutral IR, pass/runtime/backend ABI
pypto-ascend        CANN license: CCE/PTO/AIC/AIV/CANN adapter
pypto-cpu           permissive: LLVM/Highway/KleidiAI integration
pypto-cuda          permissive + NVIDIA dependency terms
pypto-hip           permissive + ROCm dependency terms
```

这不是法律意见；正式发布应由代码所有者/法务确认。

## 10. 立即可执行的下一步

1. 把本报告和一页 Target ABI 草案发给 PyPTO 维护方及 NVIDIA 移植作者，先确认官方长期 IR 基线。
2. 获取 NVIDIA 工作的 commit 或最小设计说明，提取公共 GPU ABI；避免 AMD 再做一次拆层。
3. 在独立分支只做三个解耦改动：`GetBackend()` registry、JIT compiler registry、RuntimeBackend registry；第一版不加任何 SIMD intrinsic。
4. 增加 `cpu-scalar` target，用 PyTorch/PTO CPU simulator 做 differential test。
5. 在鲲鹏 920 上实现 NEON elementwise + reduction，并用汇编和 PMU 证明真实向量化。
6. 用当前通用 GDR golden 建立模型前的硬门禁，再接 Qwen3.5-9B prefill。

## 参考链接

PyPTO/PTO：

- PyPTO：Tile 编程与白盒优化（讲稿列表题名为“白盒编译”）：<https://syfeng.net/assets/talks/20251219-PyPTO:%20Tile%20%E7%BC%96%E7%A8%8B%E4%B8%8E%E7%99%BD%E7%9B%92%E7%BC%96%E8%AF%91.pdf>
- PTO: Ascend-Native Tile Programming Ecosystem：<https://syfeng.net/assets/talks/20260512-PTO:%20Ascend-Native%20Tile%20Programming%20Ecosystem.pdf>
- 官方 PyPTO：<https://gitcode.com/cann/pypto>
- 官方 PyPTO-Gym：<https://gitcode.com/cann/pypto-gym>
- 官方 PTO-ISA：<https://gitcode.com/cann/pto-isa>
- community PyPTO：<https://github.com/hw-native-sys/pypto>
- PTOAS：<https://github.com/hw-native-sys/PTOAS>
- F4HD 2026 PTO talk abstract：<https://sites.google.com/view/f4hd/home>

CPU：

- Arm ACLE：<https://arm-software.github.io/acle/main/acle.html>
- MLIR Vector dialect：<https://mlir.llvm.org/docs/Dialects/Vector/>
- MLIR ArmNEON：<https://mlir.llvm.org/docs/Dialects/ArmNeon/>
- MLIR ArmSVE：<https://mlir.llvm.org/docs/Dialects/ArmSVE/>
- Google Highway：<https://google.github.io/highway/en/master/README.html>
- Arm KleidiAI：<https://github.com/ARM-software/kleidiai>
- Arm Compute Library：<https://github.com/ARM-software/ComputeLibrary>
- 鲲鹏 SVE/SVE2 指南：<https://www.hikunpeng.com/document/detail/en/kunpengdevps/compilation/cm-bisheng/kunpengbisheng_30_0004.html>
- 鲲鹏硬件 FAQ：<https://www.hikunpeng.com/document/detail/en/kunpengfaq/productfaq/hardwarefaq/hardware_faq_0001.html>

GPU：

- MLIR GPU dialect：<https://mlir.llvm.org/docs/Dialects/GPU/>
- MLIR NVVM：<https://mlir.llvm.org/docs/Dialects/NVVMDialect/>
- MLIR ROCDL：<https://mlir.llvm.org/docs/Dialects/ROCDLDialect/>
- CUDA C++ Programming Guide：<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- PTX ISA：<https://docs.nvidia.com/cuda/parallel-thread-execution/>
- AMD HIP programming model：<https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html>
- AMD HIPRTC：<https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html>
- AMD Matrix Cores/MFMA：<https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/>
