#!/usr/bin/env bash
# Train the Signbot (H1 + Linker hands) imitation policy.
#
#   bash autodl/05_train.sh smoke                    # 64 envs, debug yaml, 20 iters, no wandb
#   bash autodl/05_train.sh 000-01-how2sign          # full run (offline wandb unless you `wandb login`)
#   NUM_ENVS=2048 bash autodl/05_train.sh 000-01-how2sign --max_iterations 10000
#   bash autodl/05_train.sh 000-02-how2sign --resume --resumeid 000-01    # continue a run
#
# Checkpoints: $SIGNBOT_DIR/legged_gym/logs/h1/<exptid>/model_<it>.pt
# Long runs: start inside `tmux` (or use nohup) so an SSH disconnect does not kill training.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
EXPTID="${1:?usage: 05_train.sh <exptid|smoke> [extra train.py args]}"
shift || true
: "${NUM_ENVS:=4096}"

conda_activate "$CONDA_ENV"
cd "$SIGNBOT_DIR/legged_gym/legged_gym/scripts"

if [ "$EXPTID" = "smoke" ]; then
    exec python train.py 000-00-smoke --task h1_mimic_hand --motion_name "$DEBUG_YAML.yaml" \
        --debug --no_wandb --max_iterations 20 --device cuda:0 "$@"
fi
exec python train.py "$EXPTID" --task h1_mimic_hand --motion_name "$MOTION_YAML.yaml" \
    --num_envs "$NUM_ENVS" --device cuda:0 "$@"
