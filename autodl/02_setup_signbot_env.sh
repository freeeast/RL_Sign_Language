#!/usr/bin/env bash
# Conda env for Signbot: python 3.8 + torch 2.3.1 (cu118) + Isaac Gym Preview 4.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
mkdir -p "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

if ! conda env list | awk '{print $1}' | grep -x "$CONDA_ENV" >/dev/null; then
    conda create -y -n "$CONDA_ENV" python=3.8
fi
conda_activate "$CONDA_ENV"
python -m pip install -U "pip<24.1" setuptools wheel

# torch: cu118 matches the CUDA 11.8 toolkit of the AutoDL image (gymtorch is JIT-compiled
# against the local toolkit). Same torch version as the upstream README (2.3.1).
if ! python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.3.1') and torch.version.cuda=='11.8' else 1)" 2>/dev/null; then
    net_turbo_on
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
    net_turbo_off
fi

# Isaac Gym Preview 4
if [ ! -d "$ISAACGYM_DIR/python" ]; then
    [ -f "$ISAACGYM_TAR" ] || { echo "[error] $ISAACGYM_TAR not found" >&2; exit 1; }
    tar -xf "$ISAACGYM_TAR" -C "$(dirname "$ISAACGYM_DIR")"
fi
pip install -e "$ISAACGYM_DIR/python"

# Isaac Gym needs libpython3.8.so from the env at runtime
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/isaacgym_ld.sh" <<'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

pip install -e "$SIGNBOT_DIR/rsl_rl"
pip install -e "$SIGNBOT_DIR/legged_gym"
pip install -r "$HERE/requirements_signbot.txt"

python "$HERE/check_env.py" --signbot_dir "$SIGNBOT_DIR" --raw_poses "$RAW_POSES"
