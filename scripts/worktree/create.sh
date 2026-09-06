#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
用法：
  create.sh --task NAME --branch BRANCH [--base REF] [--repo PATH] [--path PATH] [--started-at ISO8601]

默认：
  --repo upstream/pypto
  --base master
  --path ../worktrees/pypto-x/NAME
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"
worktree_root="$(realpath -m "$project_root/../worktrees")"
repo="$project_root/upstream/pypto"
task=""
branch=""
base="master"
path=""
started_at=""

while (($# > 0)); do
    case "$1" in
        --task)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            task="$2"
            shift 2
            ;;
        --branch)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            branch="$2"
            shift 2
            ;;
        --base)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            base="$2"
            shift 2
            ;;
        --repo)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            repo="$2"
            shift 2
            ;;
        --path)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            path="$2"
            shift 2
            ;;
        --started-at)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            started_at="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --model|--model-name|--model-path|--weights|--model-*)
            echo "错误：worktree 创建协议不接受模型参数。" >&2
            exit 2
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$task" || -z "$branch" ]]; then
    usage >&2
    exit 2
fi

if [[ ! "$task" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "错误：task 名只能包含字母、数字、下划线、点和短横线。" >&2
    exit 2
fi
if [[ "$branch" == *..* || "$branch" == /* || "$branch" == */ ]]; then
    echo "错误：branch 名不接受绝对路径或连续点。" >&2
    exit 2
fi

if [[ "$repo" != /* ]]; then
    repo="$project_root/$repo"
fi
repo="$(cd "$repo" && pwd)"
if ! git -C "$repo" rev-parse --git-dir >/dev/null 2>&1; then
    echo "错误：不是 Git 仓库：$repo" >&2
    exit 2
fi

if [[ -z "$path" ]]; then
    path="$worktree_root/pypto-x/$task"
elif [[ "$path" != /* ]]; then
    path="$project_root/$path"
fi
path="$(realpath -m "$path")"
case "$path" in
    "$worktree_root"/*) ;;
    *)
        echo "错误：worktree 必须位于 $worktree_root 下面。" >&2
        exit 2
        ;;
esac

if [[ -e "$path" || -L "$path" ]]; then
    echo "错误：目标路径已存在，不覆盖：$path" >&2
    exit 2
fi
if ! git -C "$repo" rev-parse --verify "$base^{commit}" >/dev/null 2>&1; then
    echo "错误：找不到基线引用：$base" >&2
    exit 2
fi

mkdir -p "$(dirname "$path")"
git -C "$repo" worktree add -b "$branch" "$path" "$base"

metadata_dir="$worktree_root/_meta/pypto-x/$task"
mkdir -p "$metadata_dir"
{
    printf 'task_name=%s\n' "$task"
    printf 'branch=%s\n' "$branch"
    printf 'base=%s\n' "$base"
    printf 'repository=%s\n' "$repo"
    printf 'worktree=%s\n' "$path"
    printf 'started_at=%s\n' "$started_at"
    printf 'smoke_once=true\n'
    printf 'wait_timeout_seconds=3600\n'
    printf 'poll=false\n'
} > "$metadata_dir/worktree.env"

printf 'WORKTREE_CREATED\npath=%s\nbranch=%s\nbase=%s\nmetadata=%s\n' \
    "$path" "$branch" "$base" "$metadata_dir/worktree.env"
