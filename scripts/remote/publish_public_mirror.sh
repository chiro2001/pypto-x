#!/usr/bin/env bash
# PyPTO-X：把控制仓发布到公开主仓（默认 chiro2001/pypto-x；私有归档为 chiro2001/pypto-x-private）
#
# 做三件事（不改动私有仓）：
#   1) 克隆控制仓到临时目录，用 git-filter-repo 从**历史**中剔除第三方离线副本
#      （references/*.pdf、*.html、*.txt —— 版权归原作者，公开镜像不分发）
#   2) 应用镜像专属补丁：references/README.md 加"公开镜像说明"、.gitignore 防止再次跟踪
#   3) force-push 到公开仓（默认分支 main）
#
# 用法：
#   scripts/remote/publish_public_mirror.sh [--repo chiro2001/pypto-x] [--dry-run]
#
# 依赖：git、python3（脚本会自建 venv 安装 git-filter-repo）、可推送公开仓的 gh/凭据
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PUBLIC_REPO="chiro2001/pypto-x"
DRY_RUN=0

while (($# > 0)); do
  case "$1" in
    --repo) PUBLIC_REPO="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

EXCLUDES=(
  "references/2025-12-19-pypto-tile-whitebox-compilation.pdf"
  "references/2025-12-19-pypto-tile-whitebox-compilation.txt"
  "references/2026-05-12-pto-ascend-native-ecosystem.pdf"
  "references/2026-05-12-pto-ascend-native-ecosystem.txt"
  "references/2026-01-28-f4hd-workshop.html"
)

WORK=$(mktemp -d /tmp/pypto-x-public.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo ">> 克隆控制仓到 $WORK"
git clone -q --no-local "$PROJECT_ROOT" "$WORK/repo"

echo ">> 安装/定位 git-filter-repo"
FR_VENV="$WORK/venv"
python3 -m venv "$FR_VENV"
"$FR_VENV/bin/pip" -q install git-filter-repo

ARGS=()
for f in "${EXCLUDES[@]}"; do ARGS+=(--path "$f"); done
echo ">> 从历史中剔除 ${#EXCLUDES[@]} 个第三方离线副本"
(cd "$WORK/repo" && "$FR_VENV/bin/git-filter-repo" --force --invert-paths "${ARGS[@]}" >/dev/null)

echo ">> 应用镜像专属补丁"
python3 - "$WORK/repo" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
ref = root / "references" / "README.md"
if ref.exists():
    s = ref.read_text(encoding="utf-8")
    note = ("\n## 公开镜像说明\n\n本公开镜像**不再分发**讲稿 PDF、活动页 HTML 与其 pdftotext 摘录"
            "（著作权归原作者/发布方）；请通过上面的原始 URL 获取。私有归档仓保留离线副本以便全文检索。\n")
    if "公开镜像说明" not in s:
        ref.write_text(s.rstrip("\n") + "\n" + note, encoding="utf-8")
gi = root / ".gitignore"
s = gi.read_text(encoding="utf-8") if gi.exists() else ""
if "references/*.pdf" not in s:
    gi.write_text(s.rstrip("\n") + "\n\n# 第三方讲稿/活动页离线副本（版权归原作者；公开镜像不分发）\n"
                  "references/*.pdf\nreferences/*.html\nreferences/*.txt\n", encoding="utf-8")
PY

(cd "$WORK/repo" && git add -A && git -c user.name="Chiro" -c user.email="chiro2001@163.com" \
  commit -q -m "docs(pypto-x): exclude third-party offline copies from the public mirror" || true)

echo ">> 校验：历史中不应再有第三方副本"
if (cd "$WORK/repo" && git log --all --oneline -- 'references/*.pdf' 'references/*.html' | grep -q .); then
  echo "!! 历史中仍存在第三方副本，中止"; exit 1
fi
echo "   校验通过；最大 blob："
(cd "$WORK/repo" && git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' 2>/dev/null \
  | awk '$1=="blob"' | sort -k2 -nr | head -1 | awk '{printf "   %8.1f KB  %s\n", $2/1024, $3}')

if [ "$DRY_RUN" = "1" ]; then
  echo ">> --dry-run：不推送。镜像在 $WORK/repo（退出后删除）"; exit 0
fi

echo ">> force-push 到公开仓 $PUBLIC_REPO"
(cd "$WORK/repo" && git push --force "https://github.com/${PUBLIC_REPO}.git" HEAD:main)
echo ">> 完成：https://github.com/${PUBLIC_REPO}"
