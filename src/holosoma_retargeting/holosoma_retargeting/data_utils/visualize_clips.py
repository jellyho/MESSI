"""
Render SMPL skeleton clips (step1_prepared .npz) to mp4 video files.

Each clip contains global_joint_positions (T, 22, 3).
Two side-by-side views are rendered per frame: front (Y-Z) and side (X-Z).

Usage (from src/holosoma_retargeting/holosoma_retargeting/):

    # single clip
    python data_utils/visualize_clips.py \\
        --input /path/to/step1_prepared/ARG_CRO_220001_p00_s00.npz \\
        --output /path/to/videos/

    # whole directory (first --max_clips clips)
    python data_utils/visualize_clips.py \\
        --input_dir /path/to/step1_prepared/ \\
        --output_dir /path/to/videos/ \\
        [--max_clips 10] [--fps 30]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SMPL 22-joint skeleton
# ---------------------------------------------------------------------------
SMPL_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),    # pelvis → hips, spine
    (1, 4), (2, 5),             # hips → knees
    (3, 6), (6, 9),             # spine chain
    (4, 7), (5, 8),             # knees → ankles
    (7, 10), (8, 11),           # ankles → feet
    (9, 12), (9, 13), (9, 14),  # spine3 → neck / collars
    (12, 15),                   # neck → head
    (13, 16), (14, 17),         # collars → shoulders
    (16, 18), (17, 19),         # shoulders → elbows
    (18, 20), (19, 21),         # elbows → wrists
]

# Joint color groups: legs / spine / arms / head
_LEG_J   = {1, 2, 4, 5, 7, 8, 10, 11}
_SPINE_J = {0, 3, 6, 9, 12}
_ARM_J   = {13, 14, 16, 17, 18, 19, 20, 21}
_HEAD_J  = {15}

def _joint_color(i: int) -> str:
    if i in _LEG_J:   return "#4fc3f7"
    if i in _SPINE_J: return "#fff176"
    if i in _ARM_J:   return "#ef9a9a"
    return "#ce93d8"   # head

_BONE_COLORS = [
    "#4fc3f7" if ({p, c} & _LEG_J)   else
    "#fff176" if ({p, c} & _SPINE_J) else
    "#ef9a9a"
    for p, c in SMPL_BONES
]


def _fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    rgba = rgba.reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def render_clip(
    joints: np.ndarray,      # (T, 22, 3)  z-up world coords
    out_path: Path,
    fps: int = 30,
    fig_size: tuple[int, int] = (1280, 480),
) -> None:
    """Render a skeleton clip to mp4."""
    T = joints.shape[0]

    # Centre the clip so it starts at origin XY
    joints = joints.copy()
    joints[:, :, :2] -= joints[0, 0, :2]

    # Axis limits — slightly padded bounding box over full sequence
    mins = joints.reshape(-1, 3).min(0)
    maxs = joints.reshape(-1, 3).max(0)
    pad  = 0.3
    xlim = (mins[0] - pad, maxs[0] + pad)
    ylim = (mins[1] - pad, maxs[1] + pad)
    zlim = (max(0, mins[2] - pad), maxs[2] + pad)

    dpi = 100
    fig_w, fig_h = fig_size
    fig = plt.figure(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi, facecolor="#1a1a2e")

    # Two 3D subplots: isometric and front view
    ax1 = fig.add_subplot(121, projection="3d", facecolor="#1a1a2e")
    ax2 = fig.add_subplot(122, projection="3d", facecolor="#1a1a2e")

    views = [
        (ax1, "Isometric",  20,  45),
        (ax2, "Front",       0,  90),
    ]
    for ax, title, elev, azim in views:
        ax.set_title(title, color="white", fontsize=9, pad=2)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel("X", color="#666"); ax.set_ylabel("Y", color="#666")
        ax.set_zlabel("Z", color="#666")
        ax.tick_params(colors="#444")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333")
        ax.yaxis.pane.set_edgecolor("#333")
        ax.zaxis.pane.set_edgecolor("#333")
        ax.grid(True, color="#2a2a4a", linewidth=0.5)
        ax.view_init(elev=elev, azim=azim)

    fig.tight_layout(pad=0.5)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, fig_size)

    # Pre-draw skeleton objects so we can update in-place each frame
    skel_objs: list[dict] = []
    for ax, _, _, _ in views:
        j = joints[0]
        # joints as scatter
        sc = ax.scatter(j[:, 0], j[:, 1], j[:, 2],
                        c=[_joint_color(i) for i in range(22)],
                        s=40, depthshade=True, zorder=5)
        # bones as line collections (one per bone for easy update)
        lines = []
        for bi, (p, c) in enumerate(SMPL_BONES):
            ln, = ax.plot(
                [j[p, 0], j[c, 0]],
                [j[p, 1], j[c, 1]],
                [j[p, 2], j[c, 2]],
                color=_BONE_COLORS[bi], linewidth=2.0, solid_capstyle="round"
            )
            lines.append(ln)
        # frame counter text
        txt = ax.text2D(0.02, 0.96, "frame 0", transform=ax.transAxes,
                        color="white", fontsize=7, va="top")
        skel_objs.append({"sc": sc, "lines": lines, "txt": txt})

    for t in range(T):
        j = joints[t]
        for obj, (ax, _, _, _) in zip(skel_objs, views):
            # update scatter
            obj["sc"]._offsets3d = (j[:, 0], j[:, 1], j[:, 2])
            # update bones
            for bi, (p, c) in enumerate(SMPL_BONES):
                obj["lines"][bi].set_data_3d(
                    [j[p, 0], j[c, 0]],
                    [j[p, 1], j[c, 1]],
                    [j[p, 2], j[c, 2]],
                )
            obj["txt"].set_text(f"frame {t}/{T}")

        frame_bgr = _fig_to_bgr(fig)
        writer.write(frame_bgr)

    writer.release()
    plt.close(fig)


def process_file(in_path: Path, out_dir: Path, fps: int) -> None:
    data   = np.load(in_path)
    joints = data["global_joint_positions"]   # (T, 22, 3)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (in_path.stem + ".mp4")
    render_clip(joints, out_path, fps=fps)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render step1_prepared clips to mp4")

    grp_in = p.add_mutually_exclusive_group(required=True)
    grp_in.add_argument("--input",     type=Path, help="Single .npz clip")
    grp_in.add_argument("--input_dir", type=Path, help="Directory of .npz clips")

    grp_out = p.add_mutually_exclusive_group()
    grp_out.add_argument("--output",     type=Path, help="Output path (single mode)")
    grp_out.add_argument("--output_dir", type=Path, help="Output directory (batch mode)")

    p.add_argument("--fps",       type=int, default=30)
    p.add_argument("--max_clips", type=int, default=None,
                   help="Limit number of clips rendered in batch mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is not None:
        out = args.output or Path("videos") / (args.input.stem + ".mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering {args.input.name} …")
        process_file(args.input, out.parent, args.fps)
        print(f"Saved → {out}")
    else:
        out_dir = args.output_dir or args.input_dir.parent / (args.input_dir.name + "_videos")
        clips   = sorted(args.input_dir.glob("*.npz"))
        if args.max_clips:
            clips = clips[:args.max_clips]
        print(f"Rendering {len(clips)} clips → {out_dir}")
        for clip in tqdm(clips, desc="clips"):
            process_file(clip, out_dir, args.fps)
        print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
