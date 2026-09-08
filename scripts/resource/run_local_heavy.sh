#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LOCK_TOOL=/home/chiro/projects/.resource-locks/resource-lock
MONITOR=${PROJECT_ROOT}/scripts/resource/monitor_local_heavy.py

usage() {
  printf '%s\n' \
    'usage: run_local_heavy.sh --task <task> --agent <agent>' \
    '  [--min-available-mib <MiB>] [--safety-floor-mib <MiB>]' \
    '  [--memory-max-mib <MiB>] [--max-cpus <count>]' \
    '  [--log-file <path>] -- <command> [args...]' >&2
  exit 64
}

positive_integer() {
  [[ $2 =~ ^[1-9][0-9]*$ ]] || {
    printf 'invalid %s: %s\n' "$1" "$2" >&2
    exit 64
  }
}

task=
agent=
min_available_mib=${PYPTO_X_LOCAL_MIN_AVAILABLE_MIB:-8192}
safety_floor_mib=${PYPTO_X_LOCAL_SAFETY_FLOOR_MIB:-4096}
memory_max_mib=${PYPTO_X_LOCAL_MEMORY_MAX_MIB:-}
max_cpus=${PYPTO_X_LOCAL_MAX_CPUS:-}
log_file=

while [[ $# -gt 0 && $1 != -- ]]; do
  case $1 in
    --task) [[ $# -ge 2 ]] || usage; task=$2; shift 2 ;;
    --agent) [[ $# -ge 2 ]] || usage; agent=$2; shift 2 ;;
    --min-available-mib) [[ $# -ge 2 ]] || usage; min_available_mib=$2; shift 2 ;;
    --safety-floor-mib) [[ $# -ge 2 ]] || usage; safety_floor_mib=$2; shift 2 ;;
    --memory-max-mib) [[ $# -ge 2 ]] || usage; memory_max_mib=$2; shift 2 ;;
    --max-cpus) [[ $# -ge 2 ]] || usage; max_cpus=$2; shift 2 ;;
    --log-file) [[ $# -ge 2 ]] || usage; log_file=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ $# -gt 0 && $1 == -- ]] || usage
shift
[[ $# -gt 0 ]] || usage
[[ ${task} =~ ^[A-Za-z0-9._-]+$ ]] || usage
[[ ${agent} =~ ^[A-Za-z0-9._/-]+$ ]] || usage
positive_integer min-available-mib "${min_available_mib}"
positive_integer safety-floor-mib "${safety_floor_mib}"
(( safety_floor_mib < min_available_mib )) || {
  printf 'safety floor must be below min available memory\n' >&2
  exit 64
}

[[ -x ${LOCK_TOOL} ]] || { printf 'resource lock is unavailable: %s\n' "${LOCK_TOOL}" >&2; exit 69; }
[[ -f ${MONITOR} ]] || { printf 'resource monitor is unavailable: %s\n' "${MONITOR}" >&2; exit 69; }
command -v systemd-run >/dev/null || { printf 'systemd-run is required for cgroup protection\n' >&2; exit 69; }

available_mib=$(awk '/^MemAvailable:/ {printf "%d\n", $2 / 1024; exit}' /proc/meminfo)
total_mib=$(awk '/^MemTotal:/ {printf "%d\n", $2 / 1024; exit}' /proc/meminfo)
online_cpus=$(getconf _NPROCESSORS_ONLN)
load1=$(awk '{print $1}' /proc/loadavg)
run_field=$(awk '{print $4}' /proc/loadavg)
runnable=${run_field%%/*}
cpu_psi_avg10=$(awk '/^some / {for (i=1; i<=NF; ++i) if ($i ~ /^avg10=/) {split($i, value, "="); print value[2]; exit}}' /proc/pressure/cpu)
[[ ${available_mib} =~ ^[0-9]+$ && ${total_mib} =~ ^[0-9]+$ ]] || exit 69
positive_integer online-cpus "${online_cpus}"
[[ ${runnable} =~ ^[0-9]+$ ]] || exit 69
if awk -v load_value="${load1}" -v cpus="${online_cpus}" -v run="${runnable}" -v psi="${cpu_psi_avg10:-0}" \
  'BEGIN {exit !(load_value > cpus * 1.5 && run > cpus && psi >= 90.0)}'; then
  printf 'local UNAVAILABLE cpu_pressure load1=%s runnable=%s online_cpus=%s cpu_psi_avg10=%s\n' \
    "${load1}" "${runnable}" "${online_cpus}" "${cpu_psi_avg10:-unknown}" >&2
  exit 69
fi

if [[ -z ${max_cpus} ]]; then
  max_cpus=$(( (online_cpus + 1) / 2 ))
fi
positive_integer max-cpus "${max_cpus}"
(( max_cpus <= online_cpus )) || {
  printf 'requested CPUs exceed online CPUs: requested=%s online=%s\n' "${max_cpus}" "${online_cpus}" >&2
  exit 69
}

safe_budget_mib=$(( available_mib - safety_floor_mib ))
(( available_mib >= min_available_mib && safe_budget_mib >= 1024 )) || {
  printf 'local UNAVAILABLE available_mib=%s required_mib=%s safety_floor_mib=%s\n' \
    "${available_mib}" "${min_available_mib}" "${safety_floor_mib}" >&2
  exit 69
}
if [[ -z ${memory_max_mib} ]]; then
  memory_max_mib=${safe_budget_mib}
  (( memory_max_mib <= 20480 )) || memory_max_mib=20480
fi
positive_integer memory-max-mib "${memory_max_mib}"
(( memory_max_mib <= safe_budget_mib )) || {
  printf 'requested MemoryMax would cross safety reserve: max_mib=%s safe_budget_mib=%s\n' \
    "${memory_max_mib}" "${safe_budget_mib}" >&2
  exit 69
}
memory_high_mib=$(( memory_max_mib * 85 / 100 ))
(( memory_high_mib >= 512 )) || memory_high_mib=512
(( memory_high_mib < memory_max_mib )) || memory_high_mib=$(( memory_max_mib - 1 ))

stamp=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -z ${log_file} ]]; then
  log_file=${PROJECT_ROOT}/../worktrees/_meta/pypto-x/resource-usage/${task}/${stamp}.log
fi
mkdir -p -- "$(dirname -- "${log_file}")"
unit_task=${task//[^A-Za-z0-9]/-}
unit=pypto-x-${unit_task:0:40}-$$

printf 'resource_preflight task=%s agent=%s available_mib=%s total_mib=%s load1=%s runnable=%s cpu_psi_avg10=%s online_cpus=%s max_cpus=%s memory_high_mib=%s memory_max_mib=%s safety_floor_mib=%s log=%s\n' \
  "${task}" "${agent}" "${available_mib}" "${total_mib}" "${load1}" "${runnable}" "${cpu_psi_avg10:-unknown}" \
  "${online_cpus}" "${max_cpus}" "${memory_high_mib}" "${memory_max_mib}" \
  "${safety_floor_mib}" "${log_file}"

exec "${LOCK_TOOL}" run local pypto-x "${task}" "${agent}" \
  --min-local-available-mib "${min_available_mib}" \
  --meta memory_policy=cgroup-dynamic-reserve \
  --meta cpu_policy=affinity-plus-quota \
  --meta memory_high_mib="${memory_high_mib}" \
  --meta memory_max_mib="${memory_max_mib}" \
  --meta safety_floor_mib="${safety_floor_mib}" \
  --meta max_cpus="${max_cpus}" \
  --meta monitor_log="${log_file}" \
  -- systemd-run --user --scope --quiet --collect \
  -p "MemoryHigh=${memory_high_mib}M" \
  -p "MemoryMax=${memory_max_mib}M" \
  -p MemorySwapMax=0 \
  -p "CPUQuota=$(( max_cpus * 100 ))%" \
  -p TasksMax=512 \
  --unit="${unit}" -- \
  python3 "${MONITOR}" \
  --log-file "${log_file}" \
  --max-cpus "${max_cpus}" \
  --min-start-available-mib "${min_available_mib}" \
  --safety-floor-mib "${safety_floor_mib}" \
  -- "$@"
