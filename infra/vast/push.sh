#!/usr/bin/env bash
# Push code + .env to a running vast.ai instance, pull the data pack from B2 on
# the box, and sync deps. Idempotent - safe to re-run.
#
#   infra/vast/push.sh <instance-id> [--no-data] [--no-sync]
#
# --no-data  skip pulling the data pack (smoke tests)
# --no-sync  skip `uv sync` on the box
#
# The data pack comes from Backblaze B2 (pulled ON the box - datacenter downlink,
# not the home uplink), so only the git tree + .env cross the SSH link.
#
# Knobs (defaults reproduce today's medium-fineweb behaviour):
#   VAST_SSH_KEY=$HOME/.ssh/vast_ed25519   identity offered to the box
#   REMOTE_ROOT=/workspace/gls-model
#   CORPUS=fineweb-edu                     data/packed/<corpus> to pull
#   B2_DATA_PREFIX=gls-data/packed         under b2:$GLS_B2_BUCKET/
#   UV_GROUPS="dev track"
#   CONFIG=configs/medium-fineweb.toml     only used in the closing hint
#   WAIT_SECS=600                          cap on the provisioning wait
#
# Run from the repo (needs vastai + VAST_API_KEY; .env supplies GLS_B2_BUCKET).
set -euo pipefail

VAST_SSH_KEY="${VAST_SSH_KEY:-$HOME/.ssh/vast_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/gls-model}"
CORPUS="${CORPUS:-fineweb-edu}"
B2_DATA_PREFIX="${B2_DATA_PREFIX:-gls-data/packed}"
UV_GROUPS="${UV_GROUPS:-dev track}"
CONFIG="${CONFIG:-configs/medium-fineweb.toml}"
WAIT_SECS="${WAIT_SECS:-600}"

DO_DATA=1
DO_SYNC=1
INSTANCE=""
for arg in "$@"; do
	case "$arg" in
		--no-data) DO_DATA=0 ;;
		--no-sync) DO_SYNC=0 ;;
		*) INSTANCE="$arg" ;;
	esac
done
: "${INSTANCE:?usage: infra/vast/push.sh <instance-id> [--no-data] [--no-sync]}"
[ -f "$VAST_SSH_KEY" ] || {
	echo "no ssh key at $VAST_SSH_KEY - generate one and register it:" >&2
	echo "  ssh-keygen -t ed25519 -f $VAST_SSH_KEY" >&2
	echo "  vastai create ssh-key \"\$(cat $VAST_SSH_KEY.pub)\"" >&2
	exit 1
}

cd "$(git rev-parse --show-toplevel)"

# ssh://user@HOST:PORT  ->  user / host / port  (direct-IP or proxy)
URL="$(vastai ssh-url "$INSTANCE")"
SSH_USER="$(printf '%s' "$URL" | sed -E 's#^ssh://([^@]+)@.*#\1#')"
SSH_HOST="$(printf '%s' "$URL" | sed -E 's#^ssh://[^@]+@([^:]+):.*#\1#')"
SSH_PORT="$(printf '%s' "$URL" | sed -E 's#.*:([0-9]+)$#\1#')"
DEST="$SSH_USER@$SSH_HOST"

# Explicit identity + IdentitiesOnly: works for direct-IP targets (which no
# ~/.ssh/config Host block matches) and never lets the agent offer a smartcard
# key first. Keepalives carry the long silent commands (rclone, uv sync).
SSH_OPTS=(
	-i "$VAST_SSH_KEY"
	-o IdentitiesOnly=yes
	-o StrictHostKeyChecking=accept-new
	-o ServerAliveInterval=30
	-o ServerAliveCountMax=6
)
SSH=(ssh -p "$SSH_PORT" "${SSH_OPTS[@]}" "$DEST")
RSH="ssh -p $SSH_PORT ${SSH_OPTS[*]}"
echo "target: $DEST:$SSH_PORT  (key: $VAST_SSH_KEY)"

echo "== waiting for onstart to finish (<= ${WAIT_SECS}s) =="
deadline=$(( $(date +%s) + WAIT_SECS ))
grace=$(( $(date +%s) + 60 ))          # tolerate ssh 255 only this long (sshd still coming up)
errf="$(mktemp)"
trap 'rm -f "$errf"' EXIT
while true; do
	set +e
	"${SSH[@]}" -o ConnectTimeout=15 'grep -q ONSTART_COMPLETE /workspace/onstart.log' 2>"$errf"
	rc=$?
	set -e
	[ "$rc" -eq 0 ] && break
	# ssh exit 255 = connection/auth failure. A brief window is a booting sshd;
	# past the grace period it is the real thing (wrong/again key) - fatal.
	if [ "$rc" -eq 255 ] && [ "$(date +%s)" -ge "$grace" ]; then
		echo "ssh cannot reach $DEST:$SSH_PORT after 60s - not a boot delay:" >&2
		sed 's/^/  ssh: /' "$errf" >&2 || true
		exit 1
	fi
	[ "$(date +%s)" -lt "$deadline" ] || { echo "onstart did not finish within ${WAIT_SECS}s" >&2; exit 1; }
	sleep 10
	[ "$rc" -eq 255 ] && echo "  ... waiting for ssh" || echo "  ... still provisioning"
done

"${SSH[@]}" "mkdir -p $REMOTE_ROOT/runs $REMOTE_ROOT/data/packed"

echo "== code (git-tracked files) =="
git ls-files -z | rsync -az --files-from=- --from0 -e "$RSH" ./ "$DEST:$REMOTE_ROOT/"

echo "== .env (secrets - SSH only) =="
rsync -az -e "$RSH" .env "$DEST:$REMOTE_ROOT/.env"
"${SSH[@]}" "chmod 600 $REMOTE_ROOT/.env"

if [ "$DO_DATA" -eq 1 ]; then
	echo "== data pack ($CORPUS, B2 -> box) =="
	"${SSH[@]}" "set -a; . $REMOTE_ROOT/.env; set +a; \
		rclone copy --transfers 16 --fast-list \
		b2:\$GLS_B2_BUCKET/$B2_DATA_PREFIX/$CORPUS \
		$REMOTE_ROOT/data/packed/$CORPUS"
fi

if [ "$DO_SYNC" -eq 1 ]; then
	groups=""
	for g in $UV_GROUPS; do groups="$groups --group $g"; done
	echo "== uv sync ($UV_GROUPS) =="
	# shellcheck disable=SC2086
	"${SSH[@]}" "cd $REMOTE_ROOT && /usr/local/bin/uv sync $groups"
	echo "== capture rented spec =="
	"${SSH[@]}" "cd $REMOTE_ROOT && /usr/local/bin/uv run gls env | tee runs/rented-env.txt"
fi

echo
echo "next:  ssh -p $SSH_PORT ${SSH_OPTS[*]} $DEST"
echo "       tmux new -s train"
echo "       uv run gls train --config $CONFIG --resume auto --wandb"
