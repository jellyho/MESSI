"""
WorldPoseDataset → MESSI smplx-format converter.

Each WorldPoseDataset sequence contains N players tracked over T frames.
NaN entries in `transl` mark frames where a player is invisible/occluded.
This script splits each player into continuous valid segments and saves each
segment as a separate .npz clip ready for MESSI retargeting with --data_format smplx.

Output format per clip:
    global_joint_positions : (T, 22, 3)  world-frame SMPL body joints 0-21
    height                 : float        estimated subject height in metres

SMPL joint order (indices 0-21) maps directly onto MESSI's SMPLX_DEMO_JOINTS.

Requirements:
    SMPL model file:  SMPL_MALE.pkl  (register at https://smpl.is.tue.mpg.de/ and download)
    Pass the *directory* that contains the pkl via --smpl_model_path.

Usage:
    # single sequence
    python src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_worldpose_for_rt.py \\
        --seq_path /path/to/FIFA/poses/ARG_CRO_220001.npz \\
        --smpl_model_path /path/to/smpl_models/ \\
        --output_dir /path/to/messi_input/

    # whole dataset
    python src/holosoma_retargeting/holosoma_retargeting/data_utils/prep_worldpose_for_rt.py \\
        --data_dir /path/to/FIFA/poses/ \\
        --smpl_model_path /path/to/smpl_models/ \\
        --output_dir /path/to/messi_input/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMPL forward kinematics
# ---------------------------------------------------------------------------

def load_smpl(model_path: str, device: torch.device) -> object:
    """Load SMPL body model via smplx."""
    import smplx
    # smplx.SMPL expects model_path to be the directory containing SMPL_MALE.pkl
    # (smplx.create() would instead look for model_path/smpl/SMPL_MALE.pkl)
    model = smplx.SMPL(
        model_path=model_path,
        gender="male",
        num_betas=10,
        batch_size=1,
        ext="pkl",
    ).to(device)
    model.eval()
    return model


@torch.no_grad()
def run_smpl_fk(
    model,
    global_orient: np.ndarray,   # (T, 3)
    body_pose:     np.ndarray,   # (T, 69)
    betas:         np.ndarray,   # (10,)
    transl:        np.ndarray,   # (T, 3)
    device:        torch.device,
    batch_size:    int = 128,
) -> np.ndarray:
    """Run SMPL FK in mini-batches; returns joint positions (T, 22, 3)."""
    T = global_orient.shape[0]
    all_joints = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        bs  = end - start

        go  = torch.tensor(global_orient[start:end], dtype=torch.float32, device=device)
        bp  = torch.tensor(body_pose[start:end],     dtype=torch.float32, device=device)
        bt  = torch.tensor(betas,                    dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1)
        tr  = torch.tensor(transl[start:end],        dtype=torch.float32, device=device)

        out = model(global_orient=go, body_pose=bp, betas=bt, transl=tr)
        # smplx SMPL returns joints of shape (bs, 45, 3); first 24 are body joints
        joints = out.joints[:, :22, :].cpu().numpy()   # (bs, 22, 3)
        all_joints.append(joints)

    return np.concatenate(all_joints, axis=0)           # (T, 22, 3)


def estimate_height(joints: np.ndarray) -> float:
    """
    Estimate subject height as the median per-frame head-to-toe extent.
    joints: (T, 22, 3) in world coordinates.
    Head = joint 15, feet = joints 10 & 11.
    """
    head_z  = joints[:, 15, 2]                          # (T,)
    feet_z  = joints[:, [10, 11], 2].min(axis=1)        # (T,)
    heights = head_z - feet_z
    # Use the 90th percentile to be robust to crouching / jumping frames
    return float(np.percentile(heights[heights > 0.5], 90))


# ---------------------------------------------------------------------------
# Segment detection
# ---------------------------------------------------------------------------

def find_valid_segments(
    transl:        np.ndarray,   # (T, 3)
    min_frames:    int,
    max_nan_gap:   int,
) -> list[tuple[int, int]]:
    """
    Return list of (start, end) frame indices for continuous valid segments.

    A frame is invalid if transl contains NaN.
    Short NaN gaps (≤ max_nan_gap consecutive frames) are bridged so a
    brief occlusion does not fragment an otherwise long clip.
    """
    valid = ~np.isnan(transl).any(axis=-1)   # (T,) bool

    # Bridge small NaN gaps
    if max_nan_gap > 0:
        i = 0
        while i < len(valid):
            if not valid[i]:
                j = i
                while j < len(valid) and not valid[j]:
                    j += 1
                gap = j - i
                if gap <= max_nan_gap:
                    valid[i:j] = True
            i += 1

    segments: list[tuple[int, int]] = []
    seg_start: int | None = None
    for i, v in enumerate(valid):
        if v and seg_start is None:
            seg_start = i
        elif not v and seg_start is not None:
            if i - seg_start >= min_frames:
                segments.append((seg_start, i))
            seg_start = None
    if seg_start is not None and len(valid) - seg_start >= min_frames:
        segments.append((seg_start, len(valid)))

    return segments


def interpolate_nan_frames(arr: np.ndarray) -> np.ndarray:
    """
    Linear interpolation across NaN frames along axis 0.
    arr: (T, ...) — any shape after axis 0.
    """
    arr = arr.copy()
    flat = arr.reshape(len(arr), -1)
    for col in range(flat.shape[1]):
        x = flat[:, col]
        nans = np.isnan(x)
        if not nans.any():
            continue
        idx = np.arange(len(x))
        flat[:, col] = np.interp(idx, idx[~nans], x[~nans])
    return flat.reshape(arr.shape)


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_path:      Path,
    smpl_model,
    output_dir:    Path,
    device:        torch.device,
    min_frames:    int,
    max_nan_gap:   int,
    batch_size:    int,
) -> int:
    """Process one .npz sequence; returns number of clips saved."""
    data = dict(np.load(seq_path))
    seq_name = seq_path.stem

    global_orient = data["global_orient"]   # (N, T, 3)
    body_pose     = data["body_pose"]       # (N, T, 69)
    betas         = data["betas"]           # (N, 10)
    transl        = data["transl"]          # (N, T, 3)

    N = global_orient.shape[0]
    clips_saved = 0

    for player_idx in range(N):
        segments = find_valid_segments(
            transl[player_idx],
            min_frames=min_frames,
            max_nan_gap=max_nan_gap,
        )
        if not segments:
            continue

        for seg_idx, (start, end) in enumerate(segments):
            clip_name = f"{seq_name}_p{player_idx:02d}_s{seg_idx:02d}"
            out_path  = output_dir / f"{clip_name}.npz"

            if out_path.exists():
                log.debug("skip existing %s", clip_name)
                clips_saved += 1
                continue

            # Extract and fill any bridged NaN frames via interpolation
            go_seg = interpolate_nan_frames(global_orient[player_idx, start:end])
            bp_seg = interpolate_nan_frames(body_pose[player_idx, start:end])
            tr_seg = interpolate_nan_frames(transl[player_idx, start:end])
            bt_seg = betas[player_idx]       # (10,) — shape is constant

            # Run FK
            joints = run_smpl_fk(
                smpl_model, go_seg, bp_seg, bt_seg, tr_seg,
                device=device, batch_size=batch_size,
            )   # (T_seg, 22, 3)

            height = estimate_height(joints)
            if height < 0.5 or height > 2.5:
                log.warning("%s: implausible height %.2f m – skipping", clip_name, height)
                continue

            np.savez(out_path, global_joint_positions=joints.astype(np.float32),
                     height=np.float32(height))
            clips_saved += 1
            log.debug("saved %s  (T=%d, h=%.2f m)", clip_name, len(joints), height)

    return clips_saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldPoseDataset → MESSI smplx converter")

    # Input
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--seq_path", type=Path, help="Single sequence .npz")
    grp.add_argument("--data_dir", type=Path, help="Directory of .npz pose files")

    # SMPL
    p.add_argument("--smpl_model_path", type=str, required=True,
                   help="Directory containing SMPL_MALE.pkl (from smpl.is.tue.mpg.de)")
    p.add_argument("--gender", default="male", choices=["male", "female", "neutral"])

    # Output
    p.add_argument("--output_dir", type=Path, required=True)

    # Segmentation
    p.add_argument("--min_frames",  type=int, default=60,
                   help="Minimum clip length in frames (default 60 ≈ 2s at 30Hz)")
    p.add_argument("--max_nan_gap", type=int, default=5,
                   help="Bridge NaN gaps up to this many consecutive frames")

    # Performance
    p.add_argument("--batch_size", type=int, default=128, help="SMPL FK batch size")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    log.info("Loading SMPL model from %s …", args.smpl_model_path)
    smpl_model = load_smpl(args.smpl_model_path, device)

    seq_paths: list[Path]
    if args.seq_path is not None:
        seq_paths = [args.seq_path]
    else:
        seq_paths = sorted(args.data_dir.glob("*.npz"))
        log.info("Found %d sequences in %s", len(seq_paths), args.data_dir)

    total_clips = 0
    for seq_path in tqdm.tqdm(seq_paths, desc="sequences"):
        n = process_sequence(
            seq_path,
            smpl_model,
            args.output_dir,
            device,
            min_frames=args.min_frames,
            max_nan_gap=args.max_nan_gap,
            batch_size=args.batch_size,
        )
        total_clips += n
        log.info("%s → %d clips", seq_path.name, n)

    log.info("Done. Total clips saved: %d → %s", total_clips, args.output_dir)


if __name__ == "__main__":
    main()
