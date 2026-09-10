#!/usr/bin/env bash
# PyPTO-X：导出对 upstream/pypto 的实现补丁集到 patches/pypto-x/
#
# 产出：
#   patches/pypto-x/0001-*.patch ...   补丁正文（git format-patch）
#   patches/pypto-x/SERIES             补丁顺序 + 主题
#   patches/README.md                  基线 SHA、应用方法、验证命令、许可边界
#
# 用法：
#   scripts/remote/export_patches.sh [<base-sha>] [<branch>]
#   默认 base = upstream/pypto 的 edge 快照，branch = port/pypto-x-integration
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
UPSTREAM_REPO="${PYPTO_X_UPSTREAM_REPO:-$PROJECT_ROOT/upstream/pypto}"
BASE="${1:-34475e0d83c6cdc7deac2082b1b4fa81b3beb6ad}"
BRANCH="${2:-port/pypto-x-integration}"
OUT="$PROJECT_ROOT/patches/pypto-x"

[ -d "$UPSTREAM_REPO/.git" ] || { echo "找不到 upstream 仓库：$UPSTREAM_REPO" >&2; exit 1; }
git -C "$UPSTREAM_REPO" rev-parse --verify "$BASE^{commit}" >/dev/null || { echo "base 不存在：$BASE" >&2; exit 1; }
git -C "$UPSTREAM_REPO" rev-parse --verify "$BRANCH^{commit}" >/dev/null || { echo "分支不存在：$BRANCH" >&2; exit 1; }

BASE_SHA=$(git -C "$UPSTREAM_REPO" rev-parse "$BASE^{commit}")
HEAD_SHA=$(git -C "$UPSTREAM_REPO" rev-parse "$BRANCH^{commit}")
COMMITS=$(git -C "$UPSTREAM_REPO" rev-list --count "$BASE_SHA..$HEAD_SHA")
FILES=$(git -C "$UPSTREAM_REPO" diff --name-only "$BASE_SHA..$HEAD_SHA" | wc -l)

echo ">> 导出 $COMMITS 个提交（$FILES 个文件）到 $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"
git -C "$UPSTREAM_REPO" format-patch --no-signature --stat=100 -o "$OUT" "$BASE_SHA..$HEAD_SHA" >/dev/null

# SERIES：顺序 + 主题
: > "$OUT/SERIES"
for p in "$OUT"/*.patch; do
  subj=$(sed -n 's/^Subject: \[PATCH[^]]*\] //p' "$p" | awk 'NR==1 { print; exit }' || true)
  printf '%s  %s\n' "$(basename "$p")" "$subj" >> "$OUT/SERIES"
done

SIZE=$(du -sh "$OUT" | awk '{print $1}')
cat > "$PROJECT_ROOT/patches/README.md" <<EOF
# PyPTO-X 实现补丁集

本目录发布 PyPTO-X 对上游 PyPTO（Tensor frontend / Portable Core IR 路径）的实现补丁，
供复核与复用。**不包含**上游源码，也不包含模型权重与运行证据。

\`\`\`text
上游仓库      https://gitcode.com/cann/pypto.git
基线 base     $BASE_SHA
补丁 HEAD     $HEAD_SHA
提交数        $COMMITS
涉及文件      $FILES
补丁体积      $SIZE
分支（导出时） $BRANCH
\`\`\`

## 应用方法

\`\`\`bash
git clone https://gitcode.com/cann/pypto.git pypto && cd pypto
git checkout $BASE_SHA
git am /path/to/patches/pypto-x/*.patch        # 保留提交信息
# 或： git apply /path/to/patches/pypto-x/*.patch
\`\`\`

补丁顺序见 \`pypto-x/SERIES\`（编号即顺序）。

## 验证

\`\`\`bash
PYPTO_X_PORTABLE_ONLY=1 PYTHONPATH=python python3 -m pytest -q \
  --confcutdir=python/tests/ut/pypto_x python/tests/ut/pypto_x
\`\`\`

重活（大 shape lowering/compile、全量 pytest、并行构建）在本项目内必须经
\`scripts/resource/run_local_heavy.sh\` 取得跨项目 \`local\` 锁后执行，见
\`docs/LOCAL_RESOURCE_POLICY.zh-CN.md\`。

## 许可边界（重要）

- 补丁修改的上游代码遵循 **CANN Open Software License Agreement Version 2.0**；
  补丁**文本**由本项目以 Apache-2.0 提供（见根目录 \`LICENSE\`），
  但**应用补丁后的衍生作品仍受上游许可证约束**，包括其关于适用处理器/软件场景的条款。
- 上游另有 \`pypto_pro\`（Professional / Ascend expert dialect）等目录，本项目未在本补丁集中改动。
- 本目录不含权重、不含设备二进制、不含运行证据；证据摘要另见 \`evidence-summary/\`（若有）。

## 内容概览

补丁覆盖的波次（详见 \`docs/00-handoffs/\` 与 \`configs/development_lock.yaml\`）：
Target ABI / Core IR / CPU scalar / CPU vector（AVX2、AVX-512、SVE256）/ GPU common / CUDA C1–C2 /
Qwen3.5-0.8B M0–M1K（无权重 decoder、binding、CPU/CUDA external ingestion）/ AVX2·AVX-512 parity /
GDR T=128 验收 / CPU vector runtime liveness 与 AVX-512 packed 内核 / GDR decay 门修复（图契约 v3）。
EOF

echo ">> 完成：$COMMITS 个补丁，$SIZE"
ls "$OUT" | head -3
echo "   ..."
ls "$OUT" | tail -2
