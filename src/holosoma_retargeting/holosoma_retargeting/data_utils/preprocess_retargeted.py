"""
OpenTrack-style preprocessing for MESSI retargeted motion clips.

After MESSI retargeting + data conversion (convert_data_format_mj.py), this script:

  1. Resamples qpos to the target control frequency (50 Hz) using cubic
     interpolation for joint angles and SLERP for root quaternion.
  2. Recalculates qvel from the *resampled* qpos via finite differences
     (same approach as OpenTrack's recalculate_traj_{linear,angular,joint}_velocity).
  3. [Optional] Prepends a smooth start transition: a static default-pose
     hold followed by a linear interpolation from default pose to the first
     motion frame, giving the policy a clean reset point.

Input:  .npz files produced by MESSI's convert_data_format_mj.py
        Expected keys: qpos (T, 7+J), qvel (T, 6+J), fps (scalar)
        where the free-joint part of qpos is [px, py, pz, qw, qx, qy, qz]
        and J = number of hinge joints (29 for G1-29DOF).

Output: preprocessed .npz files with the same keys at target_fps.

Usage:
    # single file
    python src/holosoma_retargeting/holosoma_retargeting/data_utils/preprocess_retargeted.py \\
        --input /path/to/converted/clip.npz \\
        --output /path/to/preprocessed/clip.npz

    # batch (whole directory)
    python src/holosoma_retargeting/holosoma_retargeting/data_utils/preprocess_retargeted.py \\
        --input_dir /path/to/converted/ \\
        --output_dir /path/to/preprocessed/ \\
        [--target_fps 50] [--smooth_start] [--smooth_hold_s 0.5] [--smooth_interp_s 0.3]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# G1 29-DOF default standing pose (matches OpenTrack's DEFAULT_QPOS)
# qpos layout: [px, py, pz, qw, qx, qy, qz, j0 … j28]
# ---------------------------------------------------------------------------

G1_DEFAULT_JOINT_POS = np.float32([
    -0.1, 0, 0, 0.3, -0.2, 0,   # left  leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    -0.1, 0, 0, 0.3, -0.2, 0,   # right leg
     0,   0, 0,                  # waist (yaw, roll, pitch)
     0.2,  0.3,  0,  1.28, 0, 0, 0,  # left  arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
     0.2, -0.3,  0,  1.28, 0, 0, 0,  # right arm
])

G1_DEFAULT_ROOT_HEIGHT = 0.79   # metres


# ---------------------------------------------------------------------------
# Quaternion utilities  (MuJoCo convention: [w, x, y, z])
# ---------------------------------------------------------------------------

def _mj_quat_to_scipy(q: np.ndarray) -> np.ndarray:
    """[w, x, y, z] → [x, y, z, w]  (scipy convention)."""
    return q[..., [1, 2, 3, 0]]

def _scipy_quat_to_mj(q: np.ndarray) -> np.ndarray:
    """[x, y, z, w] → [w, x, y, z]  (MuJoCo convention)."""
    return q[..., [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# Frequency resampling
# ---------------------------------------------------------------------------

def resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """
    Resample a (T, D) qpos array from source_fps to target_fps.

    The root quaternion (qpos[:, 3:7]) is resampled with SLERP.
    All other channels use cubic spline interpolation.

    qpos layout assumed: [px, py, pz, qw, qx, qy, qz, j0..jN]
    """
    T = qpos.shape[0]
    t_old  = np.linspace(0.0, (T - 1) / source_fps, T)
    T_new  = max(2, round(T * target_fps / source_fps))
    t_new  = np.linspace(0.0, t_old[-1], T_new)

    # --- scalar channels (position + joint angles) ---
    scalar_ids = list(range(3)) + list(range(7, qpos.shape[1]))   # px, py, pz, joints
    qpos_new = np.zeros((T_new, qpos.shape[1]), dtype=qpos.dtype)
    qpos_new[:, scalar_ids] = interp1d(
        t_old, qpos[:, scalar_ids], kind="cubic", axis=0
    )(t_new)

    # --- root quaternion (SLERP) ---
    quat_mj = qpos[:, 3:7]                      # (T, 4) [w,x,y,z]
    quat_sc = _mj_quat_to_scipy(quat_mj)        # (T, 4) [x,y,z,w]
    slerp   = Slerp(t_old, Rotation.from_quat(quat_sc))
    quat_new_sc = slerp(t_new).as_quat()        # (T_new, 4) [x,y,z,w]
    qpos_new[:, 3:7] = _scipy_quat_to_mj(quat_new_sc)

    return qpos_new


# ---------------------------------------------------------------------------
# Velocity recalculation (mirrors OpenTrack's traj_class.py helpers)
# ---------------------------------------------------------------------------

def recalculate_qvel(qpos: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute qvel (T, 6+J) from qpos (T, 7+J) using finite differences.

    qvel layout: [vx, vy, vz,  ωx, ωy, ωz,  dj0 … djN]  (MuJoCo convention)
    Root angular velocity is expressed in the *local* (body) frame.
    """
    T, D = qpos.shape
    J    = D - 7                         # number of hinge joints
    qvel = np.zeros((T, 6 + J), dtype=qpos.dtype)

    # --- linear velocity (global frame) ---
    qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) / dt

    # --- angular velocity (local / body frame) ---
    # ω_local = R_inv * (q_inv ⊗ q_next).log * 2 / dt
    # implemented via scipy Rotation
    quat_mj   = qpos[:, 3:7]
    quat_sc   = _mj_quat_to_scipy(quat_mj)        # (T, 4) [x,y,z,w]
    rots      = Rotation.from_quat(quat_sc)

    # relative rotation: r_{t+1} in the frame of r_t
    rel_rots  = rots[:-1].inv() * rots[1:]
    ang_vel_local = rel_rots.as_rotvec() / dt      # (T-1, 3)  rotvec = axis * angle

    qvel[1:, 3:6] = ang_vel_local

    # --- joint velocities ---
    qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) / dt

    return qvel


# ---------------------------------------------------------------------------
# Smooth-start transition
# ---------------------------------------------------------------------------

def build_smooth_start(
    qpos:       np.ndarray,   # (T, D) original motion at target_fps
    dt:         float,        # 1 / target_fps
    hold_s:     float = 0.5,  # seconds to hold default pose
    interp_s:   float = 0.3,  # seconds to blend default → first motion frame
) -> np.ndarray:
    """
    Prepend a two-phase transition:
      1. Static default pose for `hold_s` seconds.
      2. Linear interpolation to the first frame of `qpos` over `interp_s` seconds.

    The root position for the default pose is derived from the first frame's XY
    position but with the canonical standing height, keeping the robot in place.

    Returns the full sequence with the transition prepended: (T_total, D).
    """
    hold_frames   = max(1, round(hold_s   / dt))
    interp_frames = max(2, round(interp_s / dt))

    # Build default qpos (7 free-joint DOFs + 29 hinge DOFs = 36 for G1)
    J = qpos.shape[1] - 7
    q_default = np.zeros(qpos.shape[1], dtype=qpos.dtype)
    q_default[0] = qpos[0, 0]              # keep XY from first motion frame
    q_default[1] = qpos[0, 1]
    q_default[2] = G1_DEFAULT_ROOT_HEIGHT  # canonical standing height
    q_default[3] = 1.0                     # identity quaternion [w,x,y,z]
    q_default[7:7+J] = G1_DEFAULT_JOINT_POS[:J]

    # Phase 1: hold
    hold_block = np.tile(q_default, (hold_frames, 1))   # (hold_frames, D)

    # Phase 2: lerp from default → first motion frame
    # Scalar channels: linear interp
    alpha  = np.linspace(0.0, 1.0, interp_frames)[:, None]    # (F, 1)
    interp_block = (1.0 - alpha) * q_default[None] + alpha * qpos[[0]]

    # SLERP the quaternion part
    q0_sc  = _mj_quat_to_scipy(q_default[3:7][None])       # (1, 4)
    q1_sc  = _mj_quat_to_scipy(qpos[0, 3:7][None])         # (1, 4)
    t_slerp = np.array([0.0, 1.0])
    slerp   = Slerp(t_slerp, Rotation.from_quat(np.concatenate([q0_sc, q1_sc], axis=0)))
    quat_interp = slerp(alpha[:, 0]).as_quat()              # (F, 4) [x,y,z,w]
    interp_block[:, 3:7] = _scipy_quat_to_mj(quat_interp)

    return np.concatenate([hold_block, interp_block, qpos], axis=0)


# ---------------------------------------------------------------------------
# Main per-file processing
# ---------------------------------------------------------------------------

def process_file(
    in_path:      Path,
    out_path:     Path,
    target_fps:   float,
    smooth_start: bool,
    hold_s:       float,
    interp_s:     float,
) -> None:
    data     = np.load(in_path, allow_pickle=True)
    # Accept both "qpos" (new GMR pipeline / Step 4 convention) and
    # "joint_pos" (original MESSI Step 3 output) so both pipelines work.
    if "qpos" in data:
        qpos = data["qpos"]
    elif "joint_pos" in data:
        qpos = data["joint_pos"]
    else:
        raise KeyError(f"{in_path}: neither 'qpos' nor 'joint_pos' key found")
    qvel_orig = data.get("qvel", data.get("joint_vel", None))

    # Detect source fps
    if "fps" in data:
        source_fps = float(data["fps"])
    elif "freq" in data:
        source_fps = float(data["freq"])
    else:
        # Fall back: assume 30 Hz input
        source_fps = 30.0
        log.warning("%s: no fps key found, assuming %.0f Hz", in_path.name, source_fps)

    # --- Step 1: resample to target fps ---
    if abs(source_fps - target_fps) > 0.1:
        log.info("  resample %.1f → %.1f Hz  (T %d → %d)",
                 source_fps, target_fps, len(qpos), round(len(qpos) * target_fps / source_fps))
        qpos = resample_qpos(qpos, source_fps, target_fps)
    else:
        log.info("  already at %.1f Hz", source_fps)

    # --- Step 2: recalculate velocities ---
    dt   = 1.0 / target_fps
    qvel = recalculate_qvel(qpos, dt)

    # --- Step 3: optional smooth start ---
    if smooth_start:
        qpos_full = build_smooth_start(qpos, dt, hold_s=hold_s, interp_s=interp_s)
        qvel_full = recalculate_qvel(qpos_full, dt)
        qpos, qvel = qpos_full, qvel_full
        log.info("  smooth-start prepended  (total T=%d)", len(qpos))

    # --- save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(data)          # carry over any extra keys
    save_dict["qpos"] = qpos.astype(np.float32)
    save_dict["qvel"] = qvel.astype(np.float32)
    save_dict["fps"]  = np.float32(target_fps)
    np.savez(out_path, **save_dict)
    log.info("  saved %s  (T=%d)", out_path.name, len(qpos))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenTrack-style preprocessing for MESSI retargeted clips")

    grp_in = p.add_mutually_exclusive_group(required=True)
    grp_in.add_argument("--input",     type=Path, help="Single converted .npz")
    grp_in.add_argument("--input_dir", type=Path, help="Directory of converted .npz files")

    grp_out = p.add_mutually_exclusive_group()
    grp_out.add_argument("--output",     type=Path, help="Output path (single file mode)")
    grp_out.add_argument("--output_dir", type=Path, help="Output directory (batch mode)")

    p.add_argument("--target_fps", type=float, default=50.0,
                   help="Target control frequency in Hz (default 50)")
    p.add_argument("--smooth_start", action="store_true",
                   help="Prepend a default-pose hold + blend-in transition")
    p.add_argument("--smooth_hold_s",   type=float, default=0.5,
                   help="Seconds to hold default pose before motion (default 0.5)")
    p.add_argument("--smooth_interp_s", type=float, default=0.3,
                   help="Seconds to blend from default pose to motion (default 0.3)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is not None:
        # single-file mode
        out = args.output or args.input.with_name(args.input.stem + "_preprocessed.npz")
        log.info("Processing %s …", args.input.name)
        process_file(args.input, out, args.target_fps,
                     args.smooth_start, args.smooth_hold_s, args.smooth_interp_s)
    else:
        # batch mode
        out_dir = args.output_dir or args.input_dir.parent / (args.input_dir.name + "_preprocessed")
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(args.input_dir.glob("*.npz"))
        log.info("Found %d files in %s", len(files), args.input_dir)
        for f in files:
            log.info("Processing %s …", f.name)
            process_file(f, out_dir / f.name, args.target_fps,
                         args.smooth_start, args.smooth_hold_s, args.smooth_interp_s)
        log.info("Done → %s", out_dir)


if __name__ == "__main__":
    main()
