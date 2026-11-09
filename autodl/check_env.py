"""Sanity checks for the Signbot environment on AutoDL.

    conda activate signbot
    python autodl/check_env.py --signbot_dir /root/autodl-tmp/Signbot --raw_poses /root/autodl-tmp/data/raw_poses
"""
import argparse
import os
import pickle
import shutil
import sys

OK, BAD = "\033[32m[ok]\033[0m", "\033[31m[!!]\033[0m"
failed = []


def check(name, fn):
    try:
        msg = fn()
        print(OK, name, "-", msg if msg else "")
    except Exception as e:
        failed.append(name)
        print(BAD, name, "-", f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signbot_dir", default="/root/autodl-tmp/Signbot")
    ap.add_argument("--raw_poses", default="/root/autodl-tmp/data/raw_poses")
    args = ap.parse_args()

    def _py():
        assert sys.version_info[:2] == (3, 8), f"python {sys.version.split()[0]}, isaacgym preview 4 needs 3.6-3.8"
        return sys.version.split()[0]
    check("python 3.8", _py)

    def _isaacgym():
        import isaacgym  # noqa: F401  (must come before torch)
        from isaacgym import gymapi, gymtorch  # noqa: F401  (gymtorch triggers the JIT build)
        return os.path.dirname(isaacgym.__file__)
    check("isaacgym import + gymtorch build", _isaacgym)

    def _torch():
        import torch
        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        return f"torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}"
    check("torch + CUDA", _torch)

    def _numpy():
        import numpy as np
        assert np.__version__.startswith("1.23"), f"numpy {np.__version__}; isaacgym needs numpy<1.24 (use 1.23.5)"
        return np.__version__
    check("numpy", _numpy)

    def _physx():
        from isaacgym import gymapi
        gym = gymapi.acquire_gym()
        p = gymapi.SimParams()
        p.up_axis = gymapi.UP_AXIS_Z
        p.use_gpu_pipeline = True
        p.physx.use_gpu = True
        sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, p)
        assert sim is not None, "create_sim returned None"
        gym.destroy_sim(sim)
        return "GPU PhysX sim created (headless)"
    check("isaac gym GPU sim", _physx)

    def _pkgs():
        import importlib
        mods = ["legged_gym", "rsl_rl", "smplx", "scipy", "yaml", "wandb", "cv2", "dill", "pydelatin", "pyfqmr"]
        for m in mods:
            importlib.import_module(m)
        return ", ".join(mods)
    check("python packages", _pkgs)

    def _patched():
        assert os.path.exists(os.path.join(args.signbot_dir, "tools", "prepare_smplx_poses.py")), \
            "tools/ missing: run autodl/01_get_signbot.sh"
        return args.signbot_dir
    check("patched Signbot checkout", _patched)

    def _smplx():
        d = os.path.join(args.signbot_dir, "body_models", "smplx")
        files = [f for f in os.listdir(d) if f.startswith("SMPLX_NEUTRAL")]
        assert any(f.endswith(".npz") for f in files), f"no SMPLX_NEUTRAL.npz in {d}"
        return ", ".join(files)
    check("SMPL-X model", _smplx)

    def _poses():
        clips, first = 0, None
        for dirpath, _, files in os.walk(args.raw_poses, followlinks=True):
            pk = sorted(f for f in files if f.endswith(".pkl"))
            if pk:
                clips += 1
                first = first or os.path.join(dirpath, pk[0])
        assert clips, f"no *.pkl under {args.raw_poses} (archive not extracted?)"
        with open(first, "rb") as f:
            d = pickle.load(f)
        keys = sorted(d.keys()) if isinstance(d, dict) else type(d).__name__
        need = {"smplx_root_pose", "smplx_body_pose", "smplx_lhand_pose", "smplx_rhand_pose", "smplx_shape"}
        missing = need - set(keys) if isinstance(d, dict) else need
        assert not missing, f"{first}: missing keys {missing}; got {keys}"
        return f"{clips} clips, e.g. {first} keys={keys}"
    check("How2Sign SMPL-X poses", _poses)

    for path in ["/", os.path.dirname(args.signbot_dir.rstrip("/"))]:
        u = shutil.disk_usage(path)
        print("[..]", f"disk {path}: {u.free / 2**30:.1f} GiB free of {u.total / 2**30:.1f} GiB")

    if failed:
        print(f"\n{len(failed)} check(s) failed: {failed}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
