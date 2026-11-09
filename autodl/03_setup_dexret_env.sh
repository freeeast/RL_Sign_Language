#!/usr/bin/env bash
# Separate env for hand retargeting (dex-retargeting needs pinocchio/nlopt and a newer
# numpy than isaacgym tolerates, so upstream keeps it apart from the signbot env).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

if ! conda env list | awk '{print $1}' | grep -x "$DEX_ENV" >/dev/null; then
    conda create -y -n "$DEX_ENV" python=3.10
fi
conda_activate "$DEX_ENV"
python -m pip install -U pip setuptools wheel

# CPU torch is enough (dex-retargeting only uses torch on cpu) and is much smaller
if ! python -c "import torch" 2>/dev/null; then
    net_turbo_on
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    net_turbo_off
fi
# Signbot ships a modified dex-retargeting with Linker hand configs -> editable install
pip install "numpy<2" tqdm
pip install -e "$SIGNBOT_DIR/dex-retargeting"

python - <<EOF
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
RetargetingConfig.set_default_urdf_dir("$SIGNBOT_DIR/body_models/hands")
for h in (HandType.left, HandType.right):
    r = RetargetingConfig.load_from_file(get_default_config_path(RobotName.linker, RetargetingType.vector, h)).build()
    print(h.name, "linker hand dof:", r.optimizer.robot.dof)
print("[ok] dex-retargeting env ready")
EOF
