"""
Side-by-side comparison: original SMPL human skeleton (left) vs retargeted robot (right).

The retargeted .npz produced by parallel_robot_retarget.py / robot_retarget.py already
contains both `qpos` (robot) and `human_joints` (original SMPL joints used for
retargeting), so only one input file is needed.

Usage (from src/holosoma_retargeting/holosoma_retargeting/):

    python examples/compare_worldpose.py \\
        --retargeted_npz /path/to/step2_retargeted/clip_original.npz \\
        --robot_urdf models/g1/g1_29dof.urdf

Then open the URL printed in the terminal (default http://localhost:8080).
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

# ---------------------------------------------------------------------------
# SMPL (22-joint) skeleton definition
# SMPLX_DEMO_JOINTS order: Pelvis L_Hip R_Hip Spine1 L_Knee R_Knee Spine2
#   L_Ankle R_Ankle Spine3 L_Foot R_Foot Neck L_Collar R_Collar Head
#   L_Shoulder R_Shoulder L_Elbow R_Elbow L_Wrist R_Wrist
# ---------------------------------------------------------------------------
SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

SMPL_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),   # pelvis → hips, spine
    (1, 4), (2, 5),            # hips → knees
    (3, 6), (6, 9),            # spine chain
    (4, 7), (5, 8),            # knees → ankles
    (7, 10), (8, 11),          # ankles → feet
    (9, 12), (9, 13), (9, 14), # spine3 → neck / collars
    (12, 15),                  # neck → head
    (13, 16), (14, 17),        # collars → shoulders
    (16, 18), (17, 19),        # shoulders → elbows
    (18, 20), (19, 21),        # elbows → wrists
]

_BONE_PARENTS = np.array([p for p, _ in SMPL_BONES], dtype=np.int32)
_BONE_CHILDS  = np.array([c for _, c in SMPL_BONES], dtype=np.int32)

# Lateral offset so SMPL and robot don't overlap
_SMPL_OFFSET  = np.array([-1.5, 0.0, 0.0], dtype=np.float32)
_ROBOT_OFFSET = np.array([ 1.5, 0.0, 0.0], dtype=np.float32)


def _bone_segments(joints: np.ndarray) -> np.ndarray:
    """joints: (22, 3) → (N_bones, 2, 3)"""
    return np.stack([joints[_BONE_PARENTS], joints[_BONE_CHILDS]], axis=1)


def _center_xy(arr: np.ndarray) -> np.ndarray:
    """Zero-mean the XY of the first frame so both clips start at the origin."""
    origin = arr[0, 0, :2].copy()  # pelvis/base XY at frame 0
    arr = arr.copy()
    arr[:, :, :2] -= origin
    return arr


@dataclass
class CompareConfig:
    retargeted_npz: str
    """Path to _original.npz produced by robot_retarget / parallel_robot_retarget."""

    robot_urdf: str = "models/g1/g1_29dof.urdf"
    """Path to robot URDF file."""

    fps: int = 30
    """Playback FPS (overridden by fps stored in the npz if present)."""

    loop: bool = True
    """Loop the animation."""

    show_meshes: bool = True
    """Show robot URDF meshes."""

    port: int = 8080
    """Viser server port."""


def main(cfg: CompareConfig) -> None:
    # ------------------------------------------------------------------ #
    # Load data                                                            #
    # ------------------------------------------------------------------ #
    data = np.load(cfg.retargeted_npz, allow_pickle=True)

    qpos         = data["qpos"].astype(np.float32)         # (T, D)
    human_joints = data["human_joints"].astype(np.float32) # (T, 22, 3)
    fps          = int(data["fps"]) if "fps" in data else cfg.fps

    # Align clip lengths
    T = min(len(qpos), len(human_joints))
    qpos         = qpos[:T]
    human_joints = human_joints[:T]

    # Center XY so both start near the world origin
    human_joints = _center_xy(human_joints)  # (T, 22, 3)
    robot_base_xy = qpos[0, :2].copy()
    qpos = qpos.copy()
    qpos[:, :2] -= robot_base_xy

    # ------------------------------------------------------------------ #
    # Viser server + scene                                                 #
    # ------------------------------------------------------------------ #
    server = viser.ViserServer(port=cfg.port)

    # --- Labels ---
    server.scene.add_label("/smpl_label",  "SMPL (original)",  position=tuple((_SMPL_OFFSET  + [0, 0, 2.2]).tolist()))
    server.scene.add_label("/robot_label", "Robot (retargeted)", position=tuple((_ROBOT_OFFSET + [0, 0, 2.2]).tolist()))

    # --- Ground grids ---
    server.scene.add_grid("/grid_smpl",  width=3, height=3,
                          position=tuple(_SMPL_OFFSET.tolist()),  cell_color=(180, 230, 180))
    server.scene.add_grid("/grid_robot", width=3, height=3,
                          position=tuple(_ROBOT_OFFSET.tolist()), cell_color=(180, 200, 230))

    # ------------------------------------------------------------------ #
    # SMPL skeleton                                                        #
    # ------------------------------------------------------------------ #
    j0 = human_joints[0] + _SMPL_OFFSET

    joint_cloud = server.scene.add_point_cloud(
        "/smpl/joints",
        points=j0,
        colors=np.tile(np.array([[50, 220, 120]], dtype=np.uint8), (22, 1)),
        point_size=0.04,
    )
    bone_segs = server.scene.add_line_segments(
        "/smpl/bones",
        points=_bone_segments(j0),
        colors=np.tile(np.array([[80, 200, 100]], dtype=np.uint8), (len(SMPL_BONES), 1)),
        line_width=3.0,
    )

    # ------------------------------------------------------------------ #
    # Robot URDF                                                           #
    # ------------------------------------------------------------------ #
    robot_root  = server.scene.add_frame("/robot", show_axes=False)
    robot_urdf_y = yourdfpy.URDF.load(cfg.robot_urdf, load_meshes=True, build_scene_graph=True)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf_y, root_node_name="/robot")

    joint_limits = vr.get_actuated_joint_limits()
    robot_dof    = len(joint_limits)
    vr.show_visual = cfg.show_meshes

    # ------------------------------------------------------------------ #
    # GUI controls                                                         #
    # ------------------------------------------------------------------ #
    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
        play_btn     = server.gui.add_button("Play / Pause")
        fps_input    = server.gui.add_number("FPS", initial_value=fps, min=1, max=120, step=1)

    with server.gui.add_folder("Display"):
        show_mesh_cb = server.gui.add_checkbox("Robot meshes", initial_value=cfg.show_meshes)

    @show_mesh_cb.on_update
    def _(_): vr.show_visual = bool(show_mesh_cb.value)

    # ------------------------------------------------------------------ #
    # Per-frame draw helper                                                #
    # ------------------------------------------------------------------ #
    prev_rq: list[np.ndarray | None] = [None]  # quaternion continuity

    def _quat_norm(q):
        n = float(np.linalg.norm(q))
        return q / n if n > 0 else q

    def _quat_cont(prev, curr):
        q = _quat_norm(curr)
        if prev[0] is None:
            prev[0] = q
            return q
        if float(np.dot(prev[0], q)) < 0:
            q = -q
        prev[0] = q
        return q

    def _draw(frame: int) -> None:
        frame = int(np.clip(frame, 0, T - 1))

        # SMPL skeleton
        joints = human_joints[frame] + _SMPL_OFFSET
        joint_cloud.points = joints
        bone_segs.points   = _bone_segments(joints)

        # Robot base
        q   = qpos[frame]
        pos = q[:3] + _ROBOT_OFFSET
        rq  = _quat_cont(prev_rq, q[3:7])
        robot_root.position = tuple(pos.tolist())
        robot_root.wxyz     = tuple(rq.tolist())

        # Robot joints
        joints_q = q[7:7 + robot_dof]
        if len(joints_q) < robot_dof:
            joints_q = np.pad(joints_q, (0, robot_dof - len(joints_q)))
        vr.update_cfg(joints_q)

    # ------------------------------------------------------------------ #
    # Playback state + thread                                              #
    # ------------------------------------------------------------------ #
    state = {"playing": False, "f": 0.0}
    tick  = {"next": time.perf_counter()}
    prog  = {"updating": False}

    @play_btn.on_click
    def _(_):
        state["playing"] = not state["playing"]
        tick["next"] = time.perf_counter()
        prev_rq[0] = None
        state["f"] = float(frame_slider.value)

    @frame_slider.on_update
    def _(_):
        if prog["updating"]:
            return
        state["playing"] = False
        f = int(frame_slider.value)
        state["f"] = float(f)
        prev_rq[0] = None
        _draw(f)

    def _player() -> None:
        while True:
            if state["playing"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_input.value))
                dt = 1.0 / fps_val
                if now >= tick["next"]:
                    f = (state["f"] + 1.0) % T if cfg.loop else min(state["f"] + 1.0, T - 1)
                    state["f"] = f
                    frame_i = int(f)
                    _draw(frame_i)
                    prog["updating"] = True
                    frame_slider.value = frame_i
                    prog["updating"] = False
                    tick["next"] = now + dt
                else:
                    time.sleep(min(0.002, tick["next"] - now))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player, daemon=True).start()

    # Initial draw
    _draw(0)

    n_bones = len(SMPL_BONES)
    print(f"[compare] {T} frames | {fps} fps | robot_dof={robot_dof} | bones={n_bones}")
    print(f"[compare] Open  http://localhost:{cfg.port}  in your browser.")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    cfg = tyro.cli(CompareConfig)
    main(cfg)
