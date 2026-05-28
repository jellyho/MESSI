#!/usr/bin/env python3
"""
End-to-end pipeline: WorldPose FIFA data → robot motion training clips (GMR edition).

Replaces the slow MESSI Steps 1-3 with a single GMR-based pass:

  Old pipeline (MESSI):
    Step 1  SMPL FK → global_joint_positions   (~fast, per clip)
    Step 2  InteractionMeshRetargeter (SQP/CVXPY) → qpos  (SLOW, ~10-30 s/clip)
    Step 3  MuJoCo passive viewer replay (realtime) → body states  (SLOW, 1× realtime)
    Step 4  Resample + smooth-start

  New pipeline (this script):
    Step 1  WorldPose SMPL FK + GMR IK (mink/daqp) + headless MuJoCo FK → qpos + body states
    Step 2  Resample + smooth-start  (same as old Step 4, unchanged)

GMR is included as a git submodule at thirdparty/GMR.
No extra installation needed beyond MESSI's own dependencies + mink.

Usage (run from MESSI repo root or anywhere):

    python src/holosoma_retargeting/holosoma_retargeting/examples/run_worldpose_gmr_pipeline.py \\
        --data_dir    /path/to/FIFA/poses/ \\
        --smpl_path   /path/to/smpl_models/ \\
        --output_dir  /path/to/output/ \\
        [--robot      unitree_g1] \\
        [--fps        25] \\
        [--target_fps 50] \\
        [--workers    4] \\
        [--smooth_start]

Resume from Step 2 (skip retargeting, only re-run preprocess):

    python ... --start_step 2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make thirdparty/GMR importable (submodule, no separate pip install needed)
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).resolve().parent   # holosoma_retargeting/examples/
_MESSI_SRC = _HERE.parents[1]                  # src/
_REPO_ROOT = _MESSI_SRC.parent                 # MESSI/
_GMR_ROOT  = _REPO_ROOT / "thirdparty" / "GMR"

if not _GMR_ROOT.exists():
    sys.exit(
        f"[pipeline] GMR submodule not found at {_GMR_ROOT}.\n"
        "Run:  git submodule update --init thirdparty/GMR"
    )
if str(_GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(_GMR_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reuse MESSI's NaN-segment helpers (same logic as prep_worldpose_for_rt.py)
# ---------------------------------------------------------------------------

def _find_valid_segments(transl: np.ndarray, min_frames: int, max_nan_gap: int) -> list[tuple[int, int]]:
    valid = ~np.isnan(transl).any(axis=-1)
    if max_nan_gap > 0:
        i = 0
        while i < len(valid):
            if not valid[i]:
                j = i
                while j < len(valid) and not valid[j]:
                    j += 1
                if j - i <= max_nan_gap:
                    valid[i:j] = True
            i += 1
    segs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_frames:
                segs.append((start, i))
            start = None
    if start is not None and len(valid) - start >= min_frames:
        segs.append((start, len(valid)))
    return segs


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    flat = arr.reshape(len(arr), -1)
    for col in range(flat.shape[1]):
        x = flat[:, col]
        nans = np.isnan(x)
        if nans.any():
            idx = np.arange(len(x))
            flat[:, col] = np.interp(idx, idx[~nans], x[~nans])
    return flat.reshape(arr.shape)


# ---------------------------------------------------------------------------
# SMPL FK
# ---------------------------------------------------------------------------

@torch.no_grad()
def _smpl_fk(smpl_model, go: np.ndarray, bp: np.ndarray,
              bt: np.ndarray, tr: np.ndarray, batch: int = 256) -> np.ndarray:
    """Returns world joint positions (T, 24, 3)."""
    T = go.shape[0]
    chunks = []
    for s in range(0, T, batch):
        e = min(s + batch, T)
        bs = e - s
        out = smpl_model(
            global_orient=torch.tensor(go[s:e], dtype=torch.float32),
            body_pose=torch.tensor(bp[s:e], dtype=torch.float32),
            betas=torch.tensor(bt, dtype=torch.float32).unsqueeze(0).expand(bs, -1),
            transl=torch.tensor(tr[s:e], dtype=torch.float32),
        )
        chunks.append(out.joints[:, :24, :].detach().numpy())
    return np.concatenate(chunks, axis=0)


def _world_rotations(go: np.ndarray, bp: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """World-frame quaternions [w,x,y,z] for each SMPL joint → (T, 24, 4).

    Vectorised: processes all T frames per joint (~78x faster than T×24 loop).
    """
    from scipy.spatial.transform import Rotation as R
    bp3 = bp.reshape(len(go), 23, 3)
    world_rots: list = [None] * 24
    world_rots[0] = R.from_rotvec(go)
    for i in range(1, 24):
        world_rots[i] = world_rots[int(parents[i])] * R.from_rotvec(bp3[:, i - 1, :])
    return np.stack(
        [r.as_quat(scalar_first=True) for r in world_rots], axis=1
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Velocity recalculation (same logic as preprocess_retargeted.py)
# ---------------------------------------------------------------------------

def _recalc_qvel(qpos: np.ndarray, dt: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    T, D = qpos.shape
    J = D - 7
    qvel = np.zeros((T, 6 + J), dtype=np.float32)
    qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) / dt
    quat_sc = qpos[:, 3:7][:, [1, 2, 3, 0]]
    rots = Rotation.from_quat(quat_sc)
    qvel[1:, 3:6] = (rots[:-1].inv() * rots[1:]).as_rotvec() / dt
    qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) / dt
    return qvel


# ---------------------------------------------------------------------------
# Headless MuJoCo body-state computation (replaces Step 3's passive viewer)
# ---------------------------------------------------------------------------

def _mj_body_states(qpos_arr, mj_model, mj_data, dt):
    import mujoco
    T = len(qpos_arr)
    nb = mj_model.nbody
    body_pos_w = np.zeros((T, nb, 3), dtype=np.float32)
    body_quat_w = np.zeros((T, nb, 4), dtype=np.float32)
    body_lin_vel_w = np.zeros((T, nb, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((T, nb, 3), dtype=np.float32)
    qvel_arr = _recalc_qvel(qpos_arr, dt)
    vel_buf = np.zeros(6, dtype=np.float64)
    for t in range(T):
        mj_data.qpos[:] = qpos_arr[t]
        mj_data.qvel[:] = qvel_arr[t]
        mujoco.mj_forward(mj_model, mj_data)
        body_pos_w[t] = mj_data.xpos
        body_quat_w[t] = mj_data.xquat
        for b in range(nb):
            mujoco.mj_objectVelocity(mj_model, mj_data, mujoco.mjtObj.mjOBJ_BODY, b, vel_buf, 0)
            body_ang_vel_w[t, b] = vel_buf[0:3]
            body_lin_vel_w[t, b] = vel_buf[3:6]
    return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, qvel_arr


# ---------------------------------------------------------------------------
# Per-worker global state
# ---------------------------------------------------------------------------
_W: dict = {}


def _init_worker(robot_type: str, smpl_model_path: str) -> None:
    import mujoco
    import smplx
    from smplx.joint_names import JOINT_NAMES
    # Import directly from submodule modules — avoids loop_rate_limiters pulled in
    # by robot_motion_viewer via the top-level __init__.py
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    from general_motion_retargeting.params import ROBOT_XML_DICT

    smpl = smplx.SMPL(
        model_path=smpl_model_path,
        gender="male",
        num_betas=10,
        batch_size=1,
        ext="pkl",
    ).eval()

    gmr = GeneralMotionRetargeting(src_human="smplx", tgt_robot=robot_type, verbose=False)

    xml_path = str(ROBOT_XML_DICT[robot_type])
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
    body_names  = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY,  i) for i in range(mj_model.nbody)]

    parents_raw = smpl.parents
    parents = parents_raw.numpy() if hasattr(parents_raw, "numpy") else np.array(parents_raw)

    _W.update({
        "smpl": smpl, "gmr": gmr,
        "mj_model": mj_model, "mj_data": mj_data,
        "joint_names": joint_names, "body_names": body_names,
        "smpl_jnames": JOINT_NAMES[:len(parents)],
        "parents": parents,
    })


def _process_clip(args: tuple) -> str:
    seq_path, player_idx, seg_start, seg_end, out_path, input_fps = args
    if Path(out_path).exists():
        return f"skip  {Path(out_path).name}"

    smpl      = _W["smpl"];  gmr       = _W["gmr"]
    mj_model  = _W["mj_model"]; mj_data = _W["mj_data"]
    parents   = _W["parents"];  smpl_jnames = _W["smpl_jnames"]

    data = np.load(seq_path)
    go = _interp_nan(data["global_orient"][player_idx, seg_start:seg_end])
    bp = _interp_nan(data["body_pose"][player_idx, seg_start:seg_end])
    tr = _interp_nan(data["transl"][player_idx, seg_start:seg_end])
    bt = data["betas"][player_idx]

    joints      = _smpl_fk(smpl, go, bp, bt, tr)
    world_quats = _world_rotations(go, bp, parents)
    T           = joints.shape[0]

    # GMR retargeting — reset config between clips for temporal consistency
    gmr.setup_retarget_configuration()
    qpos_list = []
    for t in range(T):
        frame = {name: (joints[t, i], world_quats[t, i]) for i, name in enumerate(smpl_jnames)}
        qpos_list.append(gmr.retarget(frame, offset_to_ground=True))
    qpos_arr = np.stack(qpos_list, axis=0).astype(np.float32)

    dt = 1.0 / input_fps
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, qvel_arr = _mj_body_states(
        qpos_arr, mj_model, mj_data, dt
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        qpos=qpos_arr,        qvel=qvel_arr,
        joint_pos=qpos_arr,   joint_vel=qvel_arr,   # aliases for older MESSI loaders
        fps=np.float32(input_fps),
        body_pos_w=body_pos_w, body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w, body_ang_vel_w=body_ang_vel_w,
        joint_names=np.array(_W["joint_names"], dtype=object),
        body_names =np.array(_W["body_names"],  dtype=object),
    )
    return f"done  {Path(out_path).name}  (T={T})"


# ---------------------------------------------------------------------------
# Clip enumeration
# ---------------------------------------------------------------------------

def _enumerate_clips(data_dir: Path, output_dir: Path, min_frames: int, max_nan_gap: int):
    clips = []
    for seq_path in sorted(data_dir.glob("*.npz")):
        try:
            data = np.load(seq_path)
            transl = data["transl"]
        except Exception as exc:
            log.warning("Cannot read %s: %s", seq_path.name, exc)
            continue
        for p in range(transl.shape[0]):
            for s_idx, (start, end) in enumerate(
                _find_valid_segments(transl[p], min_frames, max_nan_gap)
            ):
                name = f"{seq_path.stem}_p{p:02d}_s{s_idx:02d}.npz"
                clips.append((str(seq_path), p, start, end, str(output_dir / name)))
    return clips


# ---------------------------------------------------------------------------
# Step 2: preprocess (resample + smooth-start) – runs in main process
# ---------------------------------------------------------------------------

def _run_preprocess(input_dir: Path, output_dir: Path, target_fps: int, smooth_start: bool) -> None:
    import subprocess
    preprocess_script = (
        _HERE.parent / "data_utils" / "preprocess_retargeted.py"
    )
    cmd = [
        sys.executable, str(preprocess_script),
        "--input_dir",  str(input_dir),
        "--output_dir", str(output_dir),
        "--target_fps", str(target_fps),
    ]
    if smooth_start:
        cmd.append("--smooth_start")
    print(f"\n[pipeline] $ {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[pipeline] preprocess_retargeted.py failed (exit {result.returncode})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="WorldPose → GMR (submodule) → MESSI training clips"
    )
    p.add_argument("--data_dir",    type=Path, required=True,
                   help="Directory of WorldPose .npz pose files")
    p.add_argument("--smpl_path",   type=str,  required=True,
                   help="Directory containing SMPL_MALE.pkl")
    p.add_argument("--output_dir",  type=Path, required=True,
                   help="Base output directory")
    p.add_argument("--robot",       default="unitree_g1",
                   choices=["unitree_g1", "booster_t1", "fourier_n1", "booster_k1",
                            "hightorque_hi", "kuavo_s45", "engineai_pm01"],
                   help="Target robot (default: unitree_g1)")
    p.add_argument("--fps",         type=float, default=50.0,
                   help="Input WorldPose FPS (default 50 — 50 Hz TV broadcast)")
    p.add_argument("--target_fps",  type=int,   default=50,
                   help="Target control frequency Hz for Step 2 (default 50)")
    p.add_argument("--workers",     type=int,   default=4,
                   help="Parallel worker processes for Step 1 (default 4)")
    p.add_argument("--min_frames",  type=int,   default=60)
    p.add_argument("--max_nan_gap", type=int,   default=5)
    p.add_argument("--smooth_start", action="store_true",
                   help="Prepend default-pose hold + blend-in at Step 2")
    p.add_argument("--start_step", type=int, default=1, choices=[1, 2],
                   help="Resume from this step (default 1)")
    args = p.parse_args()

    retargeted_dir   = args.output_dir / "step1_gmr_retargeted"
    preprocessed_dir = args.output_dir / "step2_preprocessed"

    # ------------------------------------------------------------------
    # Step 1 – GMR retargeting  (WorldPose → qpos + body states)
    # ------------------------------------------------------------------
    if args.start_step <= 1:
        print("\n" + "=" * 60)
        print("STEP 1: WorldPose → GMR retargeting + headless MuJoCo FK")
        print(f"        robot={args.robot}  fps={args.fps}  workers={args.workers}")
        print("=" * 60)

        retargeted_dir.mkdir(parents=True, exist_ok=True)
        clips = _enumerate_clips(
            args.data_dir, retargeted_dir, args.min_frames, args.max_nan_gap
        )
        log.info("Found %d total clips.", len(clips))

        pending = [
            (seq, p, s, e, o, args.fps)
            for (seq, p, s, e, o) in clips
            if not Path(o).exists()
        ]
        n_skip = len(clips) - len(pending)
        if n_skip:
            log.info("Skipping %d already-completed clips; %d pending.", n_skip, len(pending))

        if pending:
            n_workers = min(args.workers, len(pending))
            log.info("Launching %d worker(s) …", n_workers)
            done = failed = 0
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(args.robot, args.smpl_path),
            ) as pool:
                futures = {pool.submit(_process_clip, a): a[4] for a in pending}
                for fut in as_completed(futures):
                    try:
                        log.info(fut.result())
                        done += 1
                    except Exception:
                        import traceback
                        log.error("FAILED  %s\n%s", futures[fut], traceback.format_exc())
                        failed += 1
            log.info("Step 1 done.  Success: %d  Failed: %d", done, failed)
        else:
            log.info("All clips already processed — skipping Step 1.")

    # ------------------------------------------------------------------
    # Step 2 – Resample + optional smooth-start  (same as MESSI Step 4)
    # ------------------------------------------------------------------
    if args.start_step <= 2:
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing (resample + smooth-start)")
        print(f"        target_fps={args.target_fps}  smooth_start={args.smooth_start}")
        print("=" * 60)
        _run_preprocess(retargeted_dir, preprocessed_dir, args.target_fps, args.smooth_start)

    print(f"\n[pipeline] Done!  Final clips → {preprocessed_dir}")


if __name__ == "__main__":
    main()
