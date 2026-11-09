import os, sys, pickle, tempfile, csv
import numpy as np, torch, yaml
from scipy.spatial.transform import Rotation as R
SB = sys.argv[1]
sys.path.insert(0, os.path.join(SB, "tools"))
import prepare_smplx_poses as P
import retarget_hands as H

# 1. frame ordering
clip = "-g0iPSnQt6w_16-1-rgb_front"
names = [f"{clip}_{i}_3D.pkl" for i in range(120)]
assert [P.frame_index(n, clip) for n in sorted(names, key=lambda f: P.frame_index(f, clip))] == list(range(120))
assert sorted(names)[:3] == [f"{clip}_0_3D.pkl", f"{clip}_100_3D.pkl", f"{clip}_101_3D.pkl"]  # upstream bug
assert P.frame_index("000123.pkl", "S000001_P0000_T00") == 123
assert P.frame_index("images0042.pkl", "x") == 42
print("frame ordering ok")

# 2. load_clip on synthetic How2Sign-like dir
rng = np.random.default_rng(0)
T = 37
with tempfile.TemporaryDirectory() as d:
    cd = os.path.join(d, "val", "poses", clip); os.makedirs(cd)
    gt = []
    for i in rng.permutation(T):
        rec = {"smplx_root_pose": rng.normal(size=3).astype(np.float32), "smplx_body_pose": np.full(63, i, np.float32),
               "smplx_lhand_pose": rng.normal(size=45), "smplx_rhand_pose": rng.normal(size=45),
               "smplx_jaw_pose": np.zeros(3), "smplx_shape": rng.normal(size=10), "smplx_expr": np.zeros(10)}
        with open(os.path.join(cd, f"{clip}_{i}_3D.pkl"), "wb") as f: pickle.dump(rec, f)
    assert P.find_clip_dirs(d) == [cd]
    c, p, gaps = P.load_clip(cd)
    assert c == clip and gaps == 0 and p["body"].shape == (T, 63)
    assert np.all(p["body"][:, 0] == np.arange(T)), "frames not in numeric order"
    assert np.all(p["transl"] == 0)  # cam_trans missing -> zeros
    # same data stored as one pkl per clip (T, ...) must give identical arrays
    cf = os.path.join(d, "clips"); os.makedirs(cf)
    rec = {"smplx_root_pose": p["root"], "smplx_body_pose": p["body"], "smplx_lhand_pose": p["lhand"],
           "smplx_rhand_pose": p["rhand"], "smplx_shape": p["betas"][0]}
    with open(os.path.join(cf, "clipA.pkl"), "wb") as f: pickle.dump(rec, f)
    found = P.find_clips(d)
    assert ("clipA", os.path.join(cf, "clipA.pkl"), "clip") in found and (clip, cd, "frames") in found, found
    c2, p2, _ = P.load_clip_file(os.path.join(cf, "clipA.pkl"))
    for k in ("root", "body", "lhand", "rhand"):
        assert np.allclose(p2[k], p[k]), k
    assert p2["betas"].shape == (T, 10) and np.all(p2["transl"] == 0)
    P.inspect(d)
print("load_clip ok (per-frame and per-clip layouts)")

# 3. build_motion vs upstream loop math (fake smplx joints)
T = 25
p = {"root": rng.normal(size=(T, 3)) * .3, "body": rng.normal(size=(T, 63)) * .3, "transl": rng.normal(size=(T, 3))}
transl, go, body = P.preprocess_params(p)
joints = rng.normal(size=(T, 127, 3))
m = P.build_motion(joints, transl, go, body, 30.0, P.canonical_template())

# --- upstream reference (copied from data_process_body.py) ---
transl_c = p["transl"].copy(); go_c = p["root"].copy(); body_c = p["body"].reshape(-1, 21, 3).copy()
for i in range(T):
    transl_c[i] = transl_c[-1]; go_c[i] = go_c[-1]; body_c[i, :9] = body_c[-1, :9]
rot = R.from_euler('x', 180, degrees=True)
for i in range(T):
    go_c[i] = (rot * R.from_rotvec(go_c[i])).as_rotvec()
gp = joints[:, :22]
glv = np.concatenate([np.zeros((1, 22, 3)), np.diff(gp, axis=0) * 30], 0)[:, P.KEY_POINT_INDICES]
allp = np.concatenate([go_c[:, None], body_c], 1)
q = R.from_rotvec(allp.reshape(-1, 3)).as_quat().reshape(-1, 22, 4)[:, P.KEY_POINT_INDICES]
gav = np.zeros((T - 1, 14, 3))
for f in range(T - 1):
    for j in range(14):
        gav[f, j] = (R.from_quat(q[f + 1, j]) * R.from_quat(q[f, j]).inv()).as_rotvec() * 30
gav = np.concatenate([np.zeros((1, 14, 3)), gav], 0)
kp = gp[:, P.KEY_POINT_INDICES]; lp = np.zeros_like(kp)
for j, pj in enumerate(P.PARENT_INDICES):
    lp[:, j] = transl_c if pj == -1 else kp[:, j] - kp[:, pj]
# ------------------------------------------------------------
assert np.allclose(m["global_velocity"]["arr"], glv, atol=1e-4)
assert np.allclose(m["global_angular_velocity"]["arr"], gav, atol=1e-3)
assert np.allclose(m["rotation"]["arr"][:, 8:], q[:, 8:], atol=1e-6), "upper body rotations differ"
assert np.allclose(m["rotation"]["arr"][:, :8], [0, 0, 0, 1])
assert np.allclose(m["skeleton_tree"]["local_translation"]["arr"][1:], lp[0, 1:], atol=1e-5)
assert m["root_translation"]["arr"].shape == (T, 3)
print("build_motion matches upstream math")

# 4. hand joints & alignment helpers vs upstream
L, Rh = P.hand_joints(joints)
lref = joints[:, P.LEFT_HAND_IDXS][:, P.SMPLX_HAND_TO_PANOPTIC] - joints[:, 20:21]; lref -= lref[:, :1]
assert L.shape == (T, 21, 3) and np.allclose(L, lref, atol=1e-5)
def up_rotate(g):
    v = g[11][:2] - g[0][:2]; a = np.arctan2(v[1], v[0])
    m = np.array([[np.cos(-a), -np.sin(-a), 0], [np.sin(-a), np.cos(-a), 0], [0, 0, 1]])
    g[:, :2] = (m[:2, :2] @ g[:, :2].T).T; return g
for f in range(5):
    a = H.align_09_to_z_axis(L[f].astype(np.float64))
    assert np.allclose(H.rotate_based_on_joint_distribution(a), up_rotate(a.copy()))
    z = H.align_09_to_z_axis(L[f].astype(np.float64)); v = z[9] - z[0]
    assert np.allclose(v / np.linalg.norm(v), [0, 0, 1], atol=1e-6)
print("hand helpers ok")

# 5. vectorized _set_env_state writes through dof_state view, same as loop
N, D = 6, 55
body_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,33,34,35,36,37]
hand_idx = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32] + list(range(40, 55))
wrist_idx = [16,17,38,39]
def mk():
    st = torch.zeros(N * D, 2); return st, st.view(N, D, 2)[..., 0], st.view(N, D, 2)[..., 1]
env_ids = torch.tensor([1, 4, 5])
dp, dv, hp, ew = torch.randn(3, 21), torch.randn(3, 21), torch.randn(3, 30), torch.randn(3, 4)
s1, p1, v1 = mk()
for i, e in enumerate(env_ids):
    p1[e, body_idx] = dp[i]; v1[e, body_idx] = dv[i]; p1[e, hand_idx] = hp[i]; p1[e, wrist_idx] = ew[i]
s2, p2, v2 = mk(); rows = env_ids.view(-1, 1)
p2[rows, torch.tensor(body_idx).view(1, -1)] = dp; v2[rows, torch.tensor(body_idx).view(1, -1)] = dv
p2[rows, torch.tensor(hand_idx).view(1, -1)] = hp; p2[rows, torch.tensor(wrist_idx).view(1, -1)] = ew
assert torch.equal(s1, s2)
print("vectorized _set_env_state ok")
print("ALL OK")
