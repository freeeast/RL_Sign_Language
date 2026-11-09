# Paths used by all autodl/*.sh scripts. Override any of them before running, e.g.
#   AUTODL_TMP=/root/autodl-tmp RAW_POSES=/root/autodl-tmp/data/raw_poses bash autodl/04_data_pipeline.sh
: "${AUTODL_TMP:=/root/autodl-tmp}"

# inputs you uploaded
: "${ISAACGYM_TAR:=$AUTODL_TMP/IsaacGym_Preview_4_Package.tar}"   # only used if $ISAACGYM_DIR is missing
: "${SMPLX_SRC_DIR:=$AUTODL_TMP}"                 # contains SMPLX_NEUTRAL.npz (and/or .pkl)
: "${RAW_POSES:=$AUTODL_TMP/data/raw_poses}"      # holds val_poses.zip (Neural Sign Actors How2Sign 3D poses)
: "${POSE_ZIPS:=val_poses}"                       # zips to extract/use, space separated: "val_poses train_poses"

# outputs / code
: "${ISAACGYM_DIR:=$AUTODL_TMP/isaacgym}"
: "${SIGNBOT_DIR:=$AUTODL_TMP/Signbot}"
: "${DATA_ROOT:=$AUTODL_TMP/data/how2sign}"       # source_body/ hand_joints/ target_body/ target_linker_hand/
: "${MOTION_YAML:=motions_how2sign_val}"          # full set
: "${DEBUG_YAML:=motions_how2sign_debug}"         # small subset for smoke tests
: "${MIN_FRAMES:=30}"                             # drop very short clips

# conda envs
: "${CONDA_ENV:=signbot}"                         # python 3.8 + torch(cu118) + isaacgym
: "${DEX_ENV:=dexret}"                            # python 3.10 + dex-retargeting (pinocchio)

# upstream code pinned to the commit the patches were made against
: "${SIGNBOT_URL:=https://github.com/qiaoguanren/Signbot.git}"
: "${UPSTREAM_COMMIT:=55087d24b9c4dc4124c59c659f5e1ab89d31eeaa}"

# keep download caches off the 30GB system disk
export PIP_CACHE_DIR="$AUTODL_TMP/.cache/pip"
export CONDA_PKGS_DIRS="$AUTODL_TMP/.cache/conda_pkgs"

export AUTODL_TMP ISAACGYM_TAR SMPLX_SRC_DIR RAW_POSES POSE_ZIPS ISAACGYM_DIR SIGNBOT_DIR DATA_ROOT \
       MOTION_YAML DEBUG_YAML MIN_FRAMES CONDA_ENV DEX_ENV SIGNBOT_URL UPSTREAM_COMMIT

conda_activate() {
    local base
    base="$(conda info --base 2>/dev/null || echo /root/miniconda3)"
    # conda's activate scripts are not `set -u` safe
    local had_u=0; case $- in *u*) had_u=1; set +u;; esac
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh"
    conda activate "$1"
    if [ "$had_u" = 1 ]; then set -u; fi
}

# AutoDL "academic acceleration" for github / download.pytorch.org (no-op elsewhere)
net_turbo_on()  { if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null 2>&1 || true; fi; }
net_turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; }
