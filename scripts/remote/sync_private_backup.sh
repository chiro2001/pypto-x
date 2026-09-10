#!/usr/bin/env bash
# PyPTO-X：把公开主仓的最新历史同步到本地私有备份，并（可选）把 archive 分支推到私有远端。
#
# 仓库布局（2026-09-10 起）：
#   工作副本   <project_root>            = 公开主仓 chiro2001/pypto-x 的克隆，日常开发都在这里
#   私有备份   <project_root>/../pypto_x_private_bkp
#              ├─ archive 分支（默认检出）：含 references/ 第三方离线副本的私有历史
#              └─ main    分支：公开主仓历史（只读参考）
#
# 用法：
#   scripts/remote/sync_private_backup.sh [--push-archive]
#   环境变量：PYPTO_X_BACKUP_DIR、PYPTO_X_PUBLIC_URL、PYPTO_X_PRIVATE_REMOTE
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BACKUP="${PYPTO_X_BACKUP_DIR:-$(dirname -- "$PROJECT_ROOT")/pypto_x_private_bkp}"
PUBLIC_URL="${PYPTO_X_PUBLIC_URL:-https://github.com/chiro2001/pypto-x.git}"
PRIVATE_REMOTE="${PYPTO_X_PRIVATE_REMOTE:-origin}"
PUSH_ARCHIVE=0
[ "${1:-}" = "--push-archive" ] && PUSH_ARCHIVE=1

[ -d "$BACKUP/.git" ] || { echo "备份仓不存在：$BACKUP（先 git clone 工作副本到该路径）" >&2; exit 1; }

echo ">> 从公开主仓拉取 main 到备份仓"
git -C "$BACKUP" fetch --no-tags "$PUBLIC_URL" main:refs/remotes/public/main
git -C "$BACKUP" update-ref refs/heads/main refs/remotes/public/main
echo "   公开 main -> $(git -C "$BACKUP" rev-parse --short refs/heads/main)"

echo ">> 校验 archive 分支仍持有第三方离线副本"
COUNT=$(git -C "$BACKUP" ls-tree -r --name-only archive -- references 2>/dev/null | grep -cE '\.(pdf|html|txt)$' || true)
echo "   archive 中 references/ 离线副本文件数：$COUNT"
if [ "${COUNT:-0}" -le 0 ]; then
  echo "!! archive 分支缺少第三方离线副本，备份不完整" >&2; exit 1
fi

if [ "$PUSH_ARCHIVE" = "1" ]; then
  echo ">> 推送 archive 到私有远端 $PRIVATE_REMOTE"
  git -C "$BACKUP" push "$PRIVATE_REMOTE" archive:archive
fi

echo ">> 完成。备份目录：$BACKUP（当前检出 $(git -C "$BACKUP" rev-parse --abbrev-ref HEAD)）"
