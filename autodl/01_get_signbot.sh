#!/usr/bin/env bash
# Fetch upstream Signbot at the pinned commit and apply our patches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/.." && pwd)"
source "$HERE/env.sh"

GIT_ID=(-c user.name=signbot-autodl -c user.email=signbot-autodl@localhost)
PATCHES=("$REPO_DIR"/patches/*.patch)

# Already set up by an earlier run: only apply patches that are new since then.
if [ -d "$SIGNBOT_DIR/.git" ] && [ -n "$(git -C "$SIGNBOT_DIR" log --format=%h --grep='strip hardcoded wandb key' 2>/dev/null)" ]; then
    cd "$SIGNBOT_DIR"
    applied=$(( $(git rev-list --count "$UPSTREAM_COMMIT"..HEAD) - 1 ))
    if [ "$applied" -ge "${#PATCHES[@]}" ]; then
        echo "[skip] $SIGNBOT_DIR already has all ${#PATCHES[@]} patches"
        exit 0
    fi
    if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
        echo "[error] $SIGNBOT_DIR has local modifications; commit or stash them first" >&2
        exit 1
    fi
    echo "[update] applying patches $((applied + 1))..${#PATCHES[@]}"
    git "${GIT_ID[@]}" am -q "${PATCHES[@]:$applied}"
    git log --oneline | head -8
    exit 0
fi
if [ -e "$SIGNBOT_DIR" ] && [ -n "$(ls -A "$SIGNBOT_DIR" 2>/dev/null)" ]; then
    echo "[error] $SIGNBOT_DIR exists but is not a patched Signbot checkout; move it away first" >&2
    exit 1
fi

mkdir -p "$SIGNBOT_DIR"
cd "$SIGNBOT_DIR"
git init -q
git remote add origin "$SIGNBOT_URL"
net_turbo_on
for i in 1 2 3 4; do
    git fetch --depth 1 origin "$UPSTREAM_COMMIT" && break
    echo "fetch failed, retry $i"; sleep $((2 ** i))
done
net_turbo_off
git -c advice.detachedHead=false checkout -q -B autodl FETCH_HEAD

# The upstream train.py hardcodes the author's WANDB_API_KEY. Strip it here (not in
# a patch) so that this repository never contains the key.
sed -i "/os.environ\['WANDB_API_KEY'\]/d" legged_gym/legged_gym/scripts/train.py
git "${GIT_ID[@]}" commit -qam "strip hardcoded wandb key"
git "${GIT_ID[@]}" am -q "${PATCHES[@]}"

# SMPL-X model where smplx.create(<Signbot>/body_models, "smplx") looks for it
mkdir -p body_models/smplx
for ext in npz pkl; do
    if [ -f "$SMPLX_SRC_DIR/SMPLX_NEUTRAL.$ext" ]; then
        ln -sf "$SMPLX_SRC_DIR/SMPLX_NEUTRAL.$ext" "body_models/smplx/SMPLX_NEUTRAL.$ext"
    fi
done
[ -e body_models/smplx/SMPLX_NEUTRAL.npz ] || echo "[warn] SMPLX_NEUTRAL.npz not found in $SMPLX_SRC_DIR"

git log --oneline | head -6
echo "[ok] patched Signbot at $SIGNBOT_DIR"
