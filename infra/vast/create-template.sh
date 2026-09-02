#!/usr/bin/env bash
# Create (or update) a vast.ai template for a GPU machine shape.
#
# A template is a machine shape, not a run: it bakes in the GPU count/VRAM/
# disk, the non-secret environment, and the onstart script. The run (which
# config, which corpus) is chosen later on the box. Re-running with the same
# shape updates that template in place rather than making a duplicate.
#
# Knobs (all optional; defaults reproduce the original 1x80GB template):
#   GPU_COUNT=1   GPU_RAM=79   DISK=120
#   VAST_IMAGE=vastai/base-image:cuda-12.9.2-auto
#   REMOTE_ROOT=/workspace/gls-model
#   PY_VERSION=3.12
#   TEMPLATE_NAME=gls-model-${GPU_COUNT}x${GPU_RAM}gb
#   EXTRA_SEARCH=            # appended to the offer filter, e.g. 'geolocation notin [CN]'
#
#   GPU_COUNT=4 DISK=200 infra/vast/create-template.sh    # a DDP box (phase-1.5)
#
# SECURITY: nothing secret goes in a template - vast.ai stores the --env string
# server-side and templates can be shared. HF_TOKEN / WANDB_API_KEY /
# RCLONE_CONFIG_B2_* reach the box only via the .env that push.sh rsyncs over
# SSH. Never add --public here.
#
# Needs VAST_API_KEY in the environment (the vastai CLI reads it directly; the
# devenv shell sources .env). Run from anywhere.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${VAST_API_KEY:?set VAST_API_KEY (it is in .env; enter the devenv shell)}"

GPU_COUNT="${GPU_COUNT:-1}"
GPU_RAM="${GPU_RAM:-80}"   # GB floor; vast's search takes GB and stores MB itself
DISK="${DISK:-120}"
VAST_IMAGE="${VAST_IMAGE:-vastai/base-image:cuda-12.9.2-auto}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/gls-model}"
PY_VERSION="${PY_VERSION:-3.12}"
TEMPLATE_NAME="${TEMPLATE_NAME:-gls-model-${GPU_COUNT}x${GPU_RAM}gb}"
EXTRA_SEARCH="${EXTRA_SEARCH:-}"

# Non-secret configuration only. PY_VERSION rides along so onstart.sh can pick
# the interpreter without being edited.
ENV_STR="-e GLS_ROOT=$REMOTE_ROOT -e HF_HOME=/workspace/hf-cache -e PY_VERSION=$PY_VERSION -e PYTHONUNBUFFERED=1 -e TOKENIZERS_PARALLELISM=false -e WANDB_PROJECT=gls-model"

# The machine shape. cuda_vers>=12.4 not >=12.9 - CUDA 12.x minor-version compat
# means the cu129 torch wheels run on any driver >=525 (see pyproject.toml's
# cu129 rationale).
SEARCH="num_gpus=$GPU_COUNT gpu_ram>=$GPU_RAM disk_space>=$DISK cuda_vers>=12.4 reliability>0.98 inet_down>=500 inet_up>=200 direct_port_count>=2 rentable=true verified=true"
[ -n "$EXTRA_SEARCH" ] && SEARCH="$SEARCH $EXTRA_SEARCH"

ONSTART="$(cat "$HERE/onstart.sh")"
README="vast.ai template for gls-model (${GPU_COUNT}x${GPU_RAM}GB). Rent an offer, then from the laptop: infra/vast/push.sh <instance-id>."

CREATOR_ID="$(vastai show user --raw | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
HASH_ID="$(vastai search templates "creator_id=$CREATOR_ID" --raw \
	| python3 -c "import sys,json; ts=[t for t in json.load(sys.stdin) if t.get('name')=='$TEMPLATE_NAME']; print(ts[0]['hash_id'] if ts else '')")"

common_args=(
	--name "$TEMPLATE_NAME"
	--image "$VAST_IMAGE"
	--disk_space "$DISK"
	--ssh --direct
	--env "$ENV_STR"
	--onstart-cmd "$ONSTART"
	--search_params "$SEARCH"
	--desc "gls-model ${GPU_COUNT}x${GPU_RAM}GB (rsync code from machine, data + checkpoints via B2)"
	--readme "$README"
)

if [ -n "$HASH_ID" ]; then
	echo "updating template $TEMPLATE_NAME ($HASH_ID)"
	vastai update template "$HASH_ID" "${common_args[@]}"
else
	echo "creating template $TEMPLATE_NAME"
	vastai create template "${common_args[@]}"
fi

echo
echo "search:  vastai search offers '$SEARCH' -o 'dph_total+'"
echo "verify:  vastai search templates \"creator_id=$CREATOR_ID\" --raw | python3 -m json.tool"
