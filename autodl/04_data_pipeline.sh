#!/usr/bin/env bash
# How2Sign SMPL-X poses -> retargeted H1 + Linker-hand motions + key bodies.
#
#   bash autodl/04_data_pipeline.sh            # all steps
#   bash autodl/04_data_pipeline.sh prepare    # a single step (prepare|yaml|body|hands|finalize|keybody)
#   LIMIT=50 bash autodl/04_data_pipeline.sh   # only the first 50 clips (quick end-to-end test)
#
# Every step skips clips that are already done, so it is safe to re-run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
STEP="${1:-all}"
: "${WORKERS:=12}"   # 16 vCPU on the AutoDL instance
: "${LIMIT:=0}"

step_prepare() {   # [signbot env] SMPL-X forward pass -> source_body/ hand_joints/ index.csv
    conda_activate "$CONDA_ENV"
    cd "$SIGNBOT_DIR"
    python tools/prepare_smplx_poses.py --pose_root "$RAW_POSES" --smplx_dir "$SIGNBOT_DIR/body_models" \
        --out_root "$DATA_ROOT" --min_frames "$MIN_FRAMES" --limit "$LIMIT"
}

step_yaml() {      # [signbot env] motion list used by the body retargeting
    conda_activate "$CONDA_ENV"
    cd "$SIGNBOT_DIR"
    python tools/generate_motion_yaml.py --data_root "$DATA_ROOT" --name "$MOTION_YAML" --min_frames "$MIN_FRAMES"
}

step_body() {      # [signbot env] SMPL body -> H1 (poselib)
    conda_activate "$CONDA_ENV"
    cd "$SIGNBOT_DIR/ASE/ase/poselib"
    python retarget_motion_h1_smpl.py --yaml "$MOTION_YAML.yaml" --source_dir "$DATA_ROOT/source_body" \
        --target_dir "$DATA_ROOT/target_body" --workers "$WORKERS"
}

step_hands() {     # [dexret env] human hand -> Linker hand joint angles
    conda_activate "$DEX_ENV"
    cd "$SIGNBOT_DIR"
    python tools/retarget_hands.py --in_root "$DATA_ROOT" --out_dir "$DATA_ROOT/target_linker_hand" \
        --urdf_dir "$SIGNBOT_DIR/body_models/hands" --workers "$WORKERS"
}

step_finalize() {  # [signbot env] keep only clips with both body and hand, write full + debug yaml
    conda_activate "$CONDA_ENV"
    cd "$SIGNBOT_DIR"
    python tools/generate_motion_yaml.py --data_root "$DATA_ROOT" --name "$MOTION_YAML" \
        --min_frames "$MIN_FRAMES" --require_done
    python tools/generate_motion_yaml.py --data_root "$DATA_ROOT" --name "$DEBUG_YAML" \
        --min_frames "$MIN_FRAMES" --require_done --limit 16
    # a cached pkl of an older yaml with the same name is detected and rebuilt by MotionLib,
    # but remove it anyway to make the state obvious
    rm -f "$SIGNBOT_DIR/ASE/ase/poselib/data/pkl/$MOTION_YAML.pkl" "$SIGNBOT_DIR/ASE/ase/poselib/data/pkl/$DEBUG_YAML.pkl"
}

step_keybody() {   # [signbot env, GPU] replay motions in Isaac Gym -> <clip>_key_bodies.npy
    conda_activate "$CONDA_ENV"
    cd "$SIGNBOT_DIR/legged_gym/legged_gym/scripts"
    python gen_keybody.py --motion_name "$MOTION_YAML.yaml"
    local n_body n_kb
    n_body=$(find "$DATA_ROOT/target_body" -name '*.npy' ! -name '*_key_bodies.npy' ! -name '*_w1_dof_pos.npy' | wc -l)
    n_kb=$(find "$DATA_ROOT/target_body" -name '*_key_bodies.npy' | wc -l)
    echo "[keybody] $n_kb key body files for $n_body retargeted motions"
}

case "$STEP" in
    all)      step_prepare; step_yaml; step_body; step_hands; step_finalize; step_keybody ;;
    prepare)  step_prepare ;;
    yaml)     step_yaml ;;
    body)     step_body ;;
    hands)    step_hands ;;
    finalize) step_finalize ;;
    keybody)  step_keybody ;;
    *) echo "unknown step $STEP"; exit 1 ;;
esac
echo "[ok] step '$STEP' done"
