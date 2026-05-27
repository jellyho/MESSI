"""
End-to-end pipeline: WorldPose FIFA data → robot motion training clips.

  Step 1  data_utils/prep_worldpose_for_rt.py
          SMPL FK on FIFA pose sequences → global_joint_positions .npz clips

  Step 2  examples/parallel_robot_retarget.py
          global_joint_positions → retargeted robot qpos .npz

  Step 3  data_conversion/convert_data_format_mj.py  (per clip, headless)
          retargeted qpos → MuJoCo-format qpos/qvel .npz at target_fps

  Step 4  data_utils/preprocess_retargeted.py
          resample to target_fps + optional smooth-start transition

Run from src/holosoma_retargeting/holosoma_retargeting/:

    python examples/run_worldpose_pipeline.py --data_dir /data5/jellyho/FIFA/poses/ --smpl_model_path /data5/jellyho/Humanoid/smpl/SMPL_MALE.pkl --output_dir /data5/jellyho/Humanoid/worldpose_retarget/

Resume from a specific step (e.g. after step 2 finishes):

    python examples/run_worldpose_pipeline.py ... --start_step 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent   # holosoma_retargeting/examples/
ROOT = HERE.parent                        # holosoma_retargeting/
DATA_UTILS = ROOT / "data_utils"
DATA_CONV  = ROOT / "data_conversion"
EXAMPLES   = HERE


def run(cmd: list[str], env: dict | None = None) -> None:
    merged = {**os.environ, **(env or {})}
    print(f"\n[pipeline] $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, env=merged)
    if result.returncode != 0:
        sys.exit(f"[pipeline] Command failed (exit {result.returncode})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldPose → MESSI → RL training data pipeline")
    p.add_argument("--data_dir",        type=Path, required=True,
                   help="Directory of FIFA WorldPose .npz pose files")
    p.add_argument("--smpl_model_path", type=str,  required=True,
                   help="Directory containing SMPL_MALE.pkl")
    p.add_argument("--output_dir",      type=Path, required=True,
                   help="Base output directory; sub-dirs are created automatically")
    p.add_argument("--target_fps",      type=int,  default=50,
                   help="Target control frequency in Hz (default 50)")
    p.add_argument("--smooth_start",    action="store_true",
                   help="Prepend a default-pose hold + blend-in at step 4")
    p.add_argument("--start_step",      type=int,  default=1, choices=[1, 2, 3, 4],
                   help="Resume from this step number (1–4); earlier steps are skipped")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    prepared_dir     = args.output_dir / "step1_prepared"
    retargeted_dir   = args.output_dir / "step2_retargeted"
    converted_dir    = args.output_dir / "step3_converted"
    preprocessed_dir = args.output_dir / "step4_preprocessed"

    # ------------------------------------------------------------------
    # Step 1 – SMPL FK → global_joint_positions clips
    # ------------------------------------------------------------------
    if args.start_step <= 1:
        print("\n" + "=" * 60)
        print("STEP 1: Preparing WorldPose data (SMPL FK)")
        print("=" * 60)
        run([
            sys.executable,
            str(DATA_UTILS / "prep_worldpose_for_rt.py"),
            "--data_dir",        str(args.data_dir),
            "--smpl_model_path", args.smpl_model_path,
            "--output_dir",      str(prepared_dir),
        ])

    # ------------------------------------------------------------------
    # Step 2 – Retarget human joints to robot qpos
    # ------------------------------------------------------------------
    if args.start_step <= 2:
        print("\n" + "=" * 60)
        print("STEP 2: Retargeting to robot motion")
        print("=" * 60)
        run([
            sys.executable,
            str(EXAMPLES / "parallel_robot_retarget.py"),
            "--data-dir",                str(prepared_dir),
            "--task-type",               "robot_only",
            "--data_format",             "smplx",
            "--task-config.object-name", "ground",
            "--save_dir",                str(retargeted_dir),
        ])

    # ------------------------------------------------------------------
    # Step 3 – Convert retargeted qpos to MuJoCo qpos/qvel format
    #          Runs per clip; sets MUJOCO_GL=egl for headless rendering.
    # ------------------------------------------------------------------
    if args.start_step <= 3:
        print("\n" + "=" * 60)
        print("STEP 3: Converting to MuJoCo format (per clip)")
        print("=" * 60)
        converted_dir.mkdir(parents=True, exist_ok=True)
        npz_files = sorted(retargeted_dir.glob("*.npz"))
        if not npz_files:
            sys.exit(f"[pipeline] No .npz files found in {retargeted_dir}")

        for npz in npz_files:
            out_path = converted_dir / npz.name
            if out_path.exists():
                print(f"[pipeline] Skipping existing: {npz.name}")
                continue
            run(
                [
                    sys.executable,
                    str(DATA_CONV / "convert_data_format_mj.py"),
                    "--input_file",  str(npz),
                    "--output_fps",  str(args.target_fps),
                    "--output_name", str(out_path),
                    "--data_format", "smplx",
                    "--object_name", "ground",
                    "--once",
                ],
                env={"MUJOCO_GL": "egl"},  # headless rendering
            )

    # ------------------------------------------------------------------
    # Step 4 – Resample to target_fps + optional smooth-start
    # ------------------------------------------------------------------
    if args.start_step <= 4:
        print("\n" + "=" * 60)
        print("STEP 4: Preprocessing (resample + smooth start)")
        print("=" * 60)
        cmd = [
            sys.executable,
            str(DATA_UTILS / "preprocess_retargeted.py"),
            "--input_dir",  str(converted_dir),
            "--output_dir", str(preprocessed_dir),
            "--target_fps", str(args.target_fps),
        ]
        if args.smooth_start:
            cmd.append("--smooth_start")
        run(cmd)

    print(f"\n[pipeline] Done!  Final clips → {preprocessed_dir}")


if __name__ == "__main__":
    main()
