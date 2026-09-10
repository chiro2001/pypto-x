#!/usr/bin/env bash
# PyPTO-X：在租用的 Ascend（A3 / 910C 等）环境里一键开通 agent 访问
#
# 用法（在租用环境的 shell 里执行，无需下载本仓库）：
#   bash setup_a3_access.sh direct                       # 环境有公网 SSH：只装 PyPTO-X agent 公钥
#   bash setup_a3_access.sh tunnel <relay> [port]        # 环境在内网/只有出网：装公钥 + 反向隧道到中继
#   例：bash setup_a3_access.sh tunnel chiro@<relay-host> 2222
#
# 设计要点：
#   - 只做加法：追加 authorized_keys、安装 autossh、起一条反向隧道；不删不改系统其它配置
#   - 幂等：重复执行安全（公钥去重、隧道先杀旧进程再起）
#   - 默认只读探测：结束时打印连接串与基础环境信息，便于父 agent 直接接手

set -euo pipefail

AGENT_PUBKEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJOJeIsSPSBJB32woxoxbFokIDaiGxhhTRLNrAC3vVOS pypto-x-agent'

MODE="${1:-}"
RELAY="${2:-}"
TUNNEL_PORT="${3:-2222}"

usage() {
  sed -n '2,12p' "$0"
  exit 2
}

install_key() {
  umask 077
  mkdir -p "$HOME/.ssh"
  chmod 700 "$HOME/.ssh"
  touch "$HOME/.ssh/authorized_keys"
  if grep -qF "$AGENT_PUBKEY" "$HOME/.ssh/authorized_keys"; then
    echo "[key] 公钥已存在，跳过"
  else
    printf '%s\n' "$AGENT_PUBKEY" >> "$HOME/.ssh/authorized_keys"
    echo "[key] 已追加 PyPTO-X agent 公钥"
  fi
  chmod 600 "$HOME/.ssh/authorized_keys"
}

probe() {
  echo "[env] user=$(id -un) host=$(hostname) arch=$(uname -m) kernel=$(uname -r)"
  echo "[env] nproc=$(nproc 2>/dev/null || echo '?') mem=$(free -g 2>/dev/null | awk '/^Mem:/{print $2" GiB"}' || echo '?')"
  command -v npu-smi >/dev/null 2>&1 && { echo "[npu] npu-smi:"; npu-smi info 2>&1 | head -20; } || echo "[npu] 未找到 npu-smi"
  if [ -d /usr/local/Ascend ]; then
    echo "[cann] /usr/local/Ascend 存在："
    ls /usr/local/Ascend 2>/dev/null | head -5
    for f in /usr/local/Ascend/driver/version.info /usr/local/Ascend/ascend-toolkit/latest/version.cfg; do
      [ -f "$f" ] && { echo "[cann] $f:"; head -5 "$f"; }
    done
  else
    echo "[cann] 未找到 /usr/local/Ascend"
  fi
  echo "[net] 出网测试：$(timeout 8 bash -c 'exec 3<>/dev/tcp/1.1.1.1/443 && echo OK' 2>/dev/null || echo FAIL)"
}

case "$MODE" in
  direct)
    install_key
    probe
    echo
    echo "==> 连接串（交给 PyPTO-X agent）："
    echo "    ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes <user>@<公网IP> -p <SSH端口>"
    echo "    本机用户：$(id -un)   本机 IP：$(hostname -I 2>/dev/null | awk '{print $1}')"
    ;;
  tunnel)
    [ -n "$RELAY" ] || usage
    install_key
    # 中继用的专用密钥（不要把本地私钥拷进来）
    if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
      ssh-keygen -t ed25519 -N '' -f "$HOME/.ssh/id_ed25519" -C "a3-$(hostname)" >/dev/null
      echo "[relay] 已生成中继密钥"
    fi
    echo "[relay] 请把下面这行公钥加到 $RELAY 的 ~/.ssh/authorized_keys（父 agent 也可以代加）："
    cat "$HOME/.ssh/id_ed25519.pub"
    echo
    if ! command -v autossh >/dev/null 2>&1; then
      if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y -qq autossh
      elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y -q autossh
      else
        echo "[relay] 未找到 autossh 且不知道怎么装，请手动安装后重跑"; exit 1
      fi
    fi
    pkill -f "autossh.*-R ${TUNNEL_PORT}:localhost:22" 2>/dev/null || true
    sleep 1
    autossh -M 0 -f -N \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -o StrictHostKeyChecking=accept-new \
      -R "${TUNNEL_PORT}:localhost:22" "$RELAY"
    sleep 2
    if pgrep -f "autossh.*-R ${TUNNEL_PORT}:localhost:22" >/dev/null; then
      echo "[relay] 反向隧道已建立：$RELAY:${TUNNEL_PORT} -> 本机 22"
    else
      echo "[relay] 隧道未起来，请检查中继是否已授权上面的公钥"; exit 1
    fi
    probe
    echo
    echo "==> 连接串（交给 PyPTO-X agent）："
    echo "    ssh -J $RELAY -p ${TUNNEL_PORT} -o IdentitiesOnly=yes $(id -un)@127.0.0.1"
    ;;
  *)
    usage
    ;;
esac
