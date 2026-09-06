#!/usr/bin/env bash
set -u

usage() {
    cat <<'EOF'
用法：
  pypto_pro_smoke.sh \
    --agent-id NAME \
    --started-at ISO8601-UTC \
    --worktree ABS_PATH \
    --target host|qemu-aarch64|nvidia-5080|amd-6750gre|kunpeng-sve256 \
    --log-dir ABS_PATH

该脚本不接受模型参数，也不会下载模型。
EOF
}

agent_id=""
started_at=""
worktree=""
target=""
log_dir=""
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

while (($# > 0)); do
    case "$1" in
        --agent-id)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            agent_id="$2"
            shift 2
            ;;
        --started-at)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            started_at="$2"
            shift 2
            ;;
        --worktree)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            worktree="$2"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            target="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            log_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --model|--model-name|--model-path|--weights|--model-*)
            echo "错误：冒烟测试禁止模型参数。" >&2
            exit 2
            ;;
        *)
            echo "错误：未知参数 $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$agent_id" || -z "$started_at" || -z "$worktree" || -z "$target" || -z "$log_dir" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "$agent_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "错误：agent-id 含有不允许的字符。" >&2
    exit 2
fi
if [[ ! "$started_at" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
    echo "错误：started-at 必须是 YYYY-MM-DDTHH:MM:SSZ。" >&2
    exit 2
fi
case "$target" in
    host|qemu-aarch64|nvidia-5080|amd-6750gre|kunpeng-sve256) ;;
    *)
        echo "错误：不支持的 target：$target" >&2
        exit 2
        ;;
esac

if [[ "$worktree" != /* || "$log_dir" != /* ]]; then
    echo "错误：worktree 和 log-dir 必须是绝对路径。" >&2
    exit 2
fi
if [[ ! -d "$worktree" ]]; then
    echo "错误：worktree 不存在：$worktree" >&2
    exit 2
fi

mkdir -p "$log_dir"
if ! mkdir "$log_dir/.smoke-once.lock" 2>/dev/null; then
    echo "错误：该 log-dir 已执行过冒烟测试；每个 subagent 只允许一次。" >&2
    exit 2
fi
log_file="$log_dir/smoke.log"
smoke_started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log() {
    printf '%s\n' "$*" | tee -a "$log_file"
}

run_logged() {
    log "+ $*"
    "$@" 2>&1 | tee -a "$log_file"
    return "${PIPESTATUS[0]}"
}

status="FAIL"
notes=""
{
    printf 'agent_id=%s\n' "$agent_id"
    printf 'started_at=%s\n' "$started_at"
    printf 'smoke_started_at=%s\n' "$smoke_started_at"
    printf 'worktree=%s\n' "$worktree"
    printf 'target=%s\n' "$target"
} > "$log_file"

if ! git -C "$worktree" rev-parse --show-toplevel >/dev/null 2>&1; then
    notes="worktree 不是 Git 仓库"
else
    head_commit="$(git -C "$worktree" rev-parse HEAD 2>/dev/null || printf unknown)"
    branch_name="$(git -C "$worktree" branch --show-current 2>/dev/null || printf detached)"
    log "head=$head_commit"
    log "branch=$branch_name"
    if ! run_logged git -C "$worktree" diff --check; then
        notes="git diff --check 失败"
    else
        case "$target" in
            host)
                if ! command -v python3 >/dev/null 2>&1; then
                    notes="缺少 python3"
                else
                    python_source=""
                    source_name=""
                    if [[ -d "$worktree/python/pypto_pro" ]]; then
                        python_source="$worktree/python/pypto_pro"
                        source_name="pypto_pro"
                    elif [[ -d "$worktree/src/pypto_gym" ]]; then
                        python_source="$worktree/src/pypto_gym"
                        source_name="pypto_gym"
                    else
                        notes="worktree 中没有可识别的 pypto_pro 或 pypto_gym Python 源码"
                    fi
                    if [[ -n "$python_source" ]] && ! run_logged python3 -m compileall -q "$python_source"; then
                        notes="$source_name Python 语法检查失败"
                    fi
                    for tool in git python3; do
                        if ! command -v "$tool" >/dev/null 2>&1; then
                            notes="缺少基础命令：$tool"
                            break
                        fi
                    done
                    if [[ -z "$notes" ]]; then
                        status="PASS"
                        notes="$source_name 源码、Git 和 Python 语法检查通过；未加载模型"
                    fi
                fi
                ;;
            qemu-aarch64)
                qemu_bin="$(command -v qemu-aarch64 || true)"
                cc_bin="$(command -v aarch64-linux-gnu-gcc || true)"
                sysroot="${AARCH64_SYSROOT-}"
                [[ -n "$sysroot" ]] || sysroot="/usr/aarch64-linux-gnu"
                probe_dir="$log_dir/aarch64"
                probe_bin="$probe_dir/feature_probe"
                mkdir -p "$probe_dir"
                if [[ -z "$qemu_bin" || -z "$cc_bin" || ! -d "$sysroot" ]]; then
                    status="BLOCKED_TOOLCHAIN"
                    notes="缺少 qemu-aarch64、aarch64-linux-gnu-gcc 或 sysroot"
                elif ! run_logged "$cc_bin" --sysroot="$sysroot" -O2 -static \
                        -march=armv8.2-a+sve "$project_root/scripts/smoke/aarch64_feature_probe.c" \
                        -o "$probe_bin"; then
                    notes="AArch64 probe 编译失败"
                elif ! run_logged env QEMU_CPU=max,sve256=on "$qemu_bin" -L "$sysroot" "$probe_bin"; then
                    notes="QEMU AArch64 probe 执行失败"
                else
                    status="PASS"
                    notes="QEMU SVE256/SVE2 能力探针通过；不代表真实硬件性能"
                fi
                ;;
            nvidia-5080)
                if ! command -v nvidia-smi >/dev/null 2>&1; then
                    status="BLOCKED_DEVICE"
                    notes="当前 shell 中没有 nvidia-smi；5080 smoke 应在 WSL shell 执行"
                elif ! run_logged nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; then
                    notes="nvidia-smi 失败"
                elif ! command -v nvcc >/dev/null 2>&1; then
                    status="BLOCKED_TOOLCHAIN"
                    notes="GPU 可见但缺少 nvcc；未下载模型"
                else
                    status="PASS"
                    notes="NVIDIA 设备和 nvcc 可见；未加载模型"
                fi
                ;;
            amd-6750gre)
                if ! command -v rocminfo >/dev/null 2>&1; then
                    status="BLOCKED_DEVICE"
                    notes="rocminfo 不可用；AMD 6750GRE 尚未接入或 ROCm 未安装"
                elif ! run_logged rocminfo; then
                    notes="rocminfo 执行失败"
                elif ! command -v hipcc >/dev/null 2>&1; then
                    status="BLOCKED_TOOLCHAIN"
                    notes="AMD 设备可见但缺少 hipcc"
                else
                    status="PASS"
                    notes="AMD 设备和 HIP 编译器可见；未加载模型"
                fi
                ;;
            kunpeng-sve256)
                if [[ "$(uname -m)" != "aarch64" ]]; then
                    status="BLOCKED_DEVICE"
                    notes="当前机器不是 aarch64；请在鲲鹏真实机器执行"
                elif ! command -v cc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
                    status="BLOCKED_TOOLCHAIN"
                    notes="缺少 cc/clang"
                else
                    native_cc="$(command -v cc || command -v clang)"
                    probe_dir="$log_dir/native-aarch64"
                    probe_bin="$probe_dir/feature_probe"
                    mkdir -p "$probe_dir"
                    if ! run_logged "$native_cc" -O2 "$project_root/scripts/smoke/aarch64_feature_probe.c" -o "$probe_bin"; then
                        notes="native AArch64 probe 编译失败"
                    elif ! run_logged "$probe_bin"; then
                        notes="native AArch64 probe 执行失败"
                    else
                        status="PASS"
                        notes="真实 AArch64 能力探针通过；请另行确认 VL=32 bytes"
                    fi
                fi
                ;;
        esac
    fi
fi

smoke_finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    printf 'smoke_finished_at=%s\n' "$smoke_finished_at"
    printf 'head=%s\n' "${head_commit:-unknown}"
    printf 'branch=%s\n' "${branch_name:-unknown}"
    printf 'status=%s\n' "$status"
    printf 'notes=%s\n' "$notes"
} >> "$log_file"

log "status=$status"
log "notes=$notes"
log "log_file=$log_file"

case "$status" in
    PASS) exit 0 ;;
    BLOCKED_TOOLCHAIN|BLOCKED_DEVICE) exit 3 ;;
    *) exit 1 ;;
esac
