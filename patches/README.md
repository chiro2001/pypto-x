# PyPTO-X 实现补丁集

本目录发布 PyPTO-X 对上游 PyPTO（Tensor frontend / Portable Core IR 路径）的实现补丁，
供复核与复用。**不包含**上游源码，也不包含模型权重与运行证据。

```text
上游仓库      https://gitcode.com/cann/pypto.git
基线 base     34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
补丁 HEAD     bef73643b79a75719ca7099dc6e5480504987193
提交数        141
涉及文件      235
补丁体积      5.9M
分支（导出时） port/pypto-x-integration
```

## 应用方法

```bash
git clone https://gitcode.com/cann/pypto.git pypto && cd pypto
git checkout 34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad
git am /path/to/patches/pypto-x/*.patch        # 保留提交信息
# 或： git apply /path/to/patches/pypto-x/*.patch
```

补丁顺序见 `pypto-x/SERIES`（编号即顺序）。

## 验证

```bash
PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python python3 -m pytest -q   --confcutdir=python/tests/ut/pypto_x python/tests/ut/pypto_x
```

重活（大 shape lowering/compile、全量 pytest、并行构建）在本项目内必须经
`scripts/resource/run_local_heavy.sh` 取得跨项目 `local` 锁后执行，见
`docs/LOCAL_RESOURCE_POLICY.zh-CN.md`。

## 许可边界（重要）

- 补丁修改的上游代码遵循 **CANN Open Software License Agreement Version 2.0**；
  补丁**文本**由本项目以 Apache-2.0 提供（见根目录 `LICENSE`），
  但**应用补丁后的衍生作品仍受上游许可证约束**，包括其关于适用处理器/软件场景的条款。
- 上游另有 `pypto_pro`（Professional / Ascend expert dialect）等目录，本项目未在本补丁集中改动。
- 本目录不含权重、不含设备二进制、不含运行证据；证据摘要另见 `evidence-summary/`（若有）。

## 内容概览

补丁覆盖的波次（详见 `docs/00-handoffs/` 与 `configs/development_lock.yaml`）：
Target ABI / Core IR / CPU scalar / CPU vector（AVX2、AVX-512、SVE256）/ GPU common / CUDA C1–C2 /
Qwen3.5-0.8B M0–M1K（无权重 decoder、binding、CPU/CUDA external ingestion）/ AVX2·AVX-512 parity /
GDR T=128 验收 / CPU vector runtime liveness 与 AVX-512 packed 内核 / GDR decay 门修复（图契约 v3）。
