#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
repo="${PYPTO_SOURCE_REPO:-$project_root/upstream/pypto}"

if [[ "$repo" != /* ]]; then
    repo="$project_root/$repo"
fi
repo="$(cd "$repo" && pwd)"

git -C "$repo" worktree list --porcelain
