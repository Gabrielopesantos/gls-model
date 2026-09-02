#!/usr/bin/env bash
# vast.ai onstart script for the `medium` pretrain box.
#
# Runs as root on every instance start, before any project code is on the
# machine. So this only prepares the box:
# system packages, uv + Python 3.12, the workspace layout, and a login-shell
# environment that matches the local devenv shell.
#
# Idempotent: safe to re-run on instance reboot.
set -euo pipefail

LOG=/workspace/onstart.log
mkdir -p /workspace
exec > >(tee -a "$LOG") 2>&1
echo "=== onstart $(date -u +%FT%TZ) ==="

export DEBIAN_FRONTEND=noninteractive
GLS_ROOT="${GLS_ROOT:-/workspace/gls-model}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
PY_VERSION="${PY_VERSION:-3.12}"   # arrives via the template --env

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || true
df -h /workspace 2>&1 || true

need_pkg=()
for bin in tmux rsync rclone curl; do
	command -v "$bin" >/dev/null 2>&1 || need_pkg+=("$bin")
done
if [ "${#need_pkg[@]}" -gt 0 ]; then
	echo "installing: ${need_pkg[*]}"
	apt-get update -qq
	apt-get install -y -qq --no-install-recommends "${need_pkg[@]}"
fi

# uv: pinned install dir so every shell finds it regardless of $HOME.
export UV_INSTALL_DIR=/usr/local/bin
if ! command -v uv >/dev/null 2>&1; then
	echo "installing uv"
	curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# pyproject requires >=3.12,<3.13; provide it independent of the image python.
/usr/local/bin/uv python install "$PY_VERSION"

mkdir -p "$GLS_ROOT" "$HF_HOME"

# Login-shell environment. The template --env sets these for the container's
# main process, but docker env does not reach an interactive `ssh` login shell
# reliably - and .env (with the secrets) only exists after push.sh runs. So
# re-export here, mirroring devenv.nix enterShell, so an SSH session and `gls`
# both see GLS_ROOT + HF_TOKEN/WANDB_API_KEY/RCLONE_CONFIG_B2_*/GLS_B2_BUCKET.
cat >/etc/profile.d/gls.sh <<EOF
export GLS_ROOT="$GLS_ROOT"
export HF_HOME="$HF_HOME"
export PATH="/usr/local/bin:\$PATH"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ -f "$GLS_ROOT/.env" ]; then
	set -a
	. "$GLS_ROOT/.env"
	set +a
fi
cd "$GLS_ROOT" 2>/dev/null || true
EOF
chmod 0644 /etc/profile.d/gls.sh

echo "=== onstart done $(date -u +%FT%TZ) ==="
echo "ONSTART_COMPLETE"
