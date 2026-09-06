# PyPTO 源码与资料快照清单

当前 edge 快照时间：2026-09-06 23:14（Asia/Shanghai）

本清单同时保留原始研究快照和最新 edge 快照。所有主仓库都是完整 Git clone，`git rev-parse --is-shallow-repository` 均为 `false`。2026-09-07 起，控制仓通过 `.gitmodules` 把五个现有 clone 就地登记为 submodule/gitlink；登记没有重建 `upstream/pypto`，其 linked worktree 关系保持不变。上游仍在快速更新，因此 exact commit SHA 而不是浮动分支名才是可复现标识。

## 2026-09-06 edge 快照

| 本地目录 | 上游与默认分支 | edge 提交 | 提交时间 | 跟踪文件 | 约占用 | 版本观察 |
|---|---|---|---|---:|---:|---|
| [pypto](../upstream/pypto/) | https://gitcode.com/cann/pypto.git · master | `34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad` | 2026-09-06 14:23 +08:00 | 4,648 | 117 MiB | `v9.2.0-beta.2-306-g34475e0d8`；Python 0.2.1；CANN package 9.2.0 |
| [pypto-gym](../upstream/pypto-gym/) | https://gitcode.com/cann/pypto-gym.git · master | `945a360e12592239a3549cb62d0db37af32bbc03` | 2026-09-05 16:26 +08:00 | 1,015 | 24 MiB | 无 tag；package 0.1.0；声明 `pypto>=0.2.0` |
| [pto-isa](../upstream/pto-isa/) | https://gitcode.com/cann/pto-isa.git · master | `668248ec886447a83787200786fe6f461169b701` | 2026-09-05 19:31 +08:00 | 5,041 | 66 MiB | `v9.2.0-beta.2-92-g668248ec`；package 仍标 9.1.0 |
| [pypto-community](../upstream/pypto-community/) | https://github.com/hw-native-sys/pypto.git · main | `9f657f37ed20ce148b46fb7229c267a152a0644e` | 2026-09-04 19:03 +08:00 | 1,631 | 311 MiB | 独立实现；runtime gitlink `4e4d3a4ad1e54c1db3d50e72decc025a9075bfa0` |
| [PTOAS](../upstream/PTOAS/) | https://github.com/hw-native-sys/PTOAS.git · master | `dc15ee5b9e459c025eb4f714f2f892b535d93eb0` | 2026-09-05 16:41 +08:00 | 7,180 | 122 MiB | 默认分支由 main 改为 master；`vmi-v0.1.6-78-gdc15ee5b9` |

该组合只是 `edge lock` 候选，尚未在同一 CANN toolkit/NPU 环境中完成配套验证，不得宣称为 stable 组合。建议使用“CANN release family 兼容锚点 + 每仓 exact SHA”的双轨 lock：edge 用于预研，stable 用于实现和回归。

PTOAS 远端存在可变同名 tag；例如 `vmi-v0.1.3`/`vmi-v0.1.4` 的本地与远端 SHA 不同。远端 tag 已非覆盖地抓取到 `refs/remotes/origin-tags/*`，未重写本地 tag。PTOAS 新 master 不再跟踪原有两个 `3rdparty` submodule，工作树内容保留为未跟踪归档候选，未删除。

## 2026-09-03 原始研究快照

## 主仓库

| 本地目录 | 上游与分支 | 固定提交 | 提交时间 | 跟踪文件 | 约占用 | 快照许可证状态 |
|---|---|---|---|---:|---:|---|
| [pypto](../upstream/pypto/) | https://gitcode.com/cann/pypto.git · master | 388ddce68700d8ef22e469c38c0d873031b6acf2 | 2026-09-03 09:52 +08:00 | 4,689 | 111 MiB | 根 LICENSE：CANN Open Software License 2.0 |
| [pypto-gym](../upstream/pypto-gym/) | https://gitcode.com/cann/pypto-gym.git · master | c068993f1dfab2532c909d9fd69c23d28320500e | 2026-09-02 16:18 +08:00 | 1,011 | 24 MiB | 默认 CANN 2.0；src/pypto_gym/transformers 子目录为 Apache-2.0 |
| [pto-isa](../upstream/pto-isa/) | https://gitcode.com/cann/pto-isa.git · master | 448eb959dbc67692336e3b9eb343880f7fb11060 | 2026-09-02 16:07 +08:00 | 5,031 | 66 MiB | 根 LICENSE：CANN 2.0 |
| [pypto-community](../upstream/pypto-community/) | https://github.com/hw-native-sys/pypto.git · main | 690da78458ada04184ecd287ba2383c9673d5cd9 | 2026-09-03 09:45 +08:00 | 1,610 | 301 MiB | 根 LICENSE：CANN 2.0；第三方子模块各自适用其许可证 |
| [PTOAS](../upstream/PTOAS/) | https://github.com/hw-native-sys/PTOAS.git · main | bdcb319d6ad43fe4a562e8911e05aebca228b848 | 2026-09-01 19:57 +08:00 | 7,149 | 107 MiB | 此提交未跟踪根 LICENSE，源文件头声明 CANN 2.0；需维护方确认完整许可证文件 |

2026-09-03 约 10:00 再次执行 git ls-remote 时，上述五个 commit 分别与其上游默认分支 HEAD 一致。官方 PyPTO 和 community 仓在研究期间刚有提交，已快进到检查时的远端 HEAD；报告结论已复核，不受这些小改动影响。

## 原始快照的子模块

### pypto-community

| 路径 | 上游 | 固定提交 |
|---|---|---|
| 3rdparty/libbacktrace | https://github.com/ianlancetaylor/libbacktrace.git | 6f8310e238fc3ce68f42f391cbe93fd156bb2c23 |
| 3rdparty/msgpack-c | https://github.com/msgpack/msgpack-c.git | 919908742b4fdbc575e77fe1a8657e70c9573c44 |
| runtime（simpler） | https://github.com/hw-native-sys/simpler | 15f5cbd922494c62444e75fa1585e39f86a64a78 |

### PTOAS

| 路径 | 上游 | 固定提交 | 许可证观察 |
|---|---|---|---|
| 3rdparty/PTO-Gym | https://github.com/PTO-ISA/PTO-Gym.git | a68542fdb84149d3c8be6b1be507ace625e04a90 | 根 LICENSE 为 CANN 2.0 |
| 3rdparty/VfSimulator | https://github.com/wang-chonghao/VfSimulator.git | 6183525ac1da16ce4771340b31f60d49f25a3d5e | 未发现根 LICENSE，使用/分发前需确认 |

PTOAS 的 .gitmodules 使用公开仓库的 SSH URL。本地仅在 .git/config 中把它们改写为对应 HTTPS URL，以便无 GitHub SSH 凭据时取得固定提交；没有修改受跟踪的 .gitmodules。

## 完整性与工作树状态

- 五个主仓库均通过 git fsck --no-progress --full；所有已初始化子模块也单独通过同一检查。
- pypto、pypto-gym、pto-isa 和 pypto-community 的工作树干净。PTO-ISA CPU demo 的 build 目录被上游规则忽略。
- PTOAS 显示 M .codex/CLAUDE.md。上游 blob 本身使用 CRLF，而同一提交的 .gitattributes 要求 eol=lf，Git clean filter 因此把 checkout 标成修改。工作树原始字节的 SHA-1 与 HEAD blob 相同，且忽略行尾后的 diff 为空；本次没有人工改动该文件，也没有用 reset 或 checkout 清理它。
- 官方 GitCode 实现与 GitHub community implementation 没有共同 Git 祖先。以 community 690da7845 和已取入的官方 388ddce68 比较，左右独有提交数为 1,631 / 3,397，不能把二者当作镜像分支直接 cherry-pick。

## PTO-ISA CPU simulator 复测

环境：x86_64，12 vCPU，AMD Engineering Sample，Clang 22.1.8；主机暴露 AVX2 和 AVX-512 feature。

执行命令：

    cd upstream/pto-isa
    python3 tests/run_cpu.py --demo gemm --verbose --no-install

2026-09-03 10:02 的结果：

- PASS，构建和运行总计约 2.40 秒。
- M=32、K=16、N=32，max_abs_diff=1.19209e-07。
- avg_ms=0.280177，matmul_flops=32768，约 0.116955 GFLOPS。
- Clang 明确警告 TMatmul.hpp 的请求循环未 vectorize。

这个结果只验证 CPU simulator 的语义正确性；问题规模很小，也没有生产 SIMD lowering，不能当作 CPU 后端性能基准。

## 外部离线资料

| 文件 | 字节数 | SHA256 | 来源 |
|---|---:|---|---|
| [2025-12-19-pypto-tile-whitebox-compilation.pdf](../references/2025-12-19-pypto-tile-whitebox-compilation.pdf) | 1,334,471 | 4ca9e03cd23c8603ff3173ee8f966e49688f0a9e51a850b9c64a04a17edf6fa4 | 冯思远公开讲稿 |
| [2025-12-19-pypto-tile-whitebox-compilation.txt](../references/2025-12-19-pypto-tile-whitebox-compilation.txt) | 20,364 | dfb5d307c7937d6cd3ec3e5769f586f9c8c82f18eb30989ed20af3d79a6b5e79 | 上述 PDF 的可检索文本 |
| [2026-01-28-f4hd-workshop.html](../references/2026-01-28-f4hd-workshop.html) | 155,842 | 205916f9e83eb274fd6de803e89481ee70242bda95da146df590c8bfc41bb96c | F4HD 活动页离线副本 |
| [2026-05-12-pto-ascend-native-ecosystem.pdf](../references/2026-05-12-pto-ascend-native-ecosystem.pdf) | 2,281,327 | f544f93a11a108ffcb07ca838dbefa8a27471eaca7eff7c615007e8a31cade8b | 冯思远公开讲稿 |
| [2026-05-12-pto-ascend-native-ecosystem.txt](../references/2026-05-12-pto-ascend-native-ecosystem.txt) | 33,039 | 184cd8912f7405e6915fa2a9c613b13ce91e0433b5533700e1af2e744f6f09ab | 上述 PDF 的可检索文本 |

原始 URL 与用途见 [references/README.md](../references/README.md)。PDF 分别为 27 页和 31 页；文本文件由 PDF 提取，仅用于全文检索，版式以原 PDF 为准。

## 许可证状态的解释

维护方已向用户表达愿意看到跨平台移植，这是推进 Target ABI RFC、技术 PoC 和许可证讨论的重要积极信号。但在本快照中，PyPTO、PTO-ISA 和 community 根许可证的第 2.1/3.1 条仍将衍生使用限制在华为 AI 处理器/软件场景；维护方态度或未来变更不能替代当前正式文本。

建议把发布门槛写进项目计划：在合并或发布 AArch64 CPU、x86、NVIDIA、AMD 后端前，取得新许可证、双许可证或覆盖明确仓库/分支/贡献者/目标平台的书面例外。这里仅记录源码事实，不构成法律意见。
