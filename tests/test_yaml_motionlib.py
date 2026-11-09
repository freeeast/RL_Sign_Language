import os, sys, csv, tempfile, subprocess, yaml
SB = sys.argv[1]
d = tempfile.mkdtemp()
with open(os.path.join(d, "index.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["clip", "num_frames"])
    for i, n in enumerate([5, 50, 300, 80]): w.writerow([f"-clip{i}_1-rgb_front", n])
subprocess.check_call([sys.executable, os.path.join(SB, "tools/generate_motion_yaml.py"), "--data_root", d,
                       "--name", "_test_tmp", "--min_frames", "10", "--max_frames", "200"])
yp = os.path.join(SB, "ASE/ase/poselib/data/configs/_test_tmp.yaml")
cfg = yaml.safe_load(open(yp))
assert list(cfg["motions"])[1:] == ["-clip1_1-rgb_front", "-clip3_1-rgb_front"], cfg
assert cfg["hand_root"] == os.path.join(d, "target_linker_hand") + "/"
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "stubs"), os.path.join(SB, "ASE/ase"), os.path.join(SB, "ASE/ase/utils")]
from motion_lib import MotionLib
ml = MotionLib.__new__(MotionLib)
files, w, diff, desc, names = ml._fetch_motion_files(yp)
assert files[0] == os.path.join(d, "target_body", "-clip1_1-rgb_front.npy"), files
assert ml._resolve_hand_dir(yp) == os.path.join(d, "target_linker_hand")
ml._motion_files = files; assert ml._pkl_matches_yaml(yp)
ml._motion_files = files[:1]; assert not ml._pkl_matches_yaml(yp)
os.remove(yp)
print("yaml + MotionLib path logic OK")
