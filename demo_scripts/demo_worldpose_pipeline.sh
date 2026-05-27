#!/usr/bin/env bash

# End-to-end pipeline: WorldPoseDataset (FIFA soccer) → robot motion training clips
#
# Steps orchestrated by examples/run_worldpose_pipeline.py:
#   1. SMPL FK on FIFA pose sequences → global joint position clips
#   2. Retarget to G1 29-DOF joint space
#   3. Convert to MuJoCo qpos/qvel format
#   4. Resample to target_fps + optional smooth-start blend
#
# Usage:
#   bash demo_scripts/demo_worldpose_pipeline.sh
#
# Override defaults via env vars before running, e.g.:
#   WORLDPOSE_DATA=/my/poses SMPL_MODEL=/my/SMPL_MALE.pkl \
#       bash demo_scripts/demo_worldpose_pipeline.sh
#
# To resume from a specific step (e.g. skip steps 1-2 if already done):
#   START_STEP=3 bash demo_scripts/demo_worldpose_pipeline.sh

set -e

SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- CONFIG (override via env vars) ----------------------------------------
WORLDPOSE_DATA="${WORLDPOSE_DATA:-/data5/jellyho/FIFA/poses}"
SMPL_MODEL="${SMPL_MODEL:-/data5/jellyho/Humanoid/smpl/SMPL_MALE.pkl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data5/jellyho/FIFA_pipeline}"
TARGET_FPS="${TARGET_FPS:-50}"
START_STEP="${START_STEP:-1}"
SMOOTH_START="${SMOOTH_START:-1}"   # set to 0 to disable smooth-start blend

# ----------------------------------------------------------------------------

# Prevent numpy/CLARABEL from spawning internal threads — with 288 workers
# running in parallel, per-worker threading causes massive oversubscription.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Source retargeting conda env (default: hsretargeting; override with CONDA_ENV_NAME=messi)
echo "Sourcing retargeting setup..."
source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"

pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet

RETARGET_DIR="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting"
cd "$RETARGET_DIR"

SMOOTH_FLAG=""
if [ "$SMOOTH_START" = "1" ]; then
    SMOOTH_FLAG="--smooth_start"
fi

echo "Running WorldPose pipeline (start_step=$START_STEP)..."
python examples/run_worldpose_pipeline.py \
    --data_dir        "$WORLDPOSE_DATA" \
    --smpl_model_path "$SMPL_MODEL" \
    --output_dir      "$OUTPUT_DIR" \
    --target_fps      "$TARGET_FPS" \
    --start_step      "$START_STEP" \
    $SMOOTH_FLAG

echo ""
echo "Done! Training-ready clips at: $OUTPUT_DIR/step4_preprocessed"
echo ""
echo "To train MESSI WBT:"
echo "  source scripts/source_isaacsim_setup.sh"
echo "  python src/holosoma/holosoma/train_agent.py \\"
echo "      exp:g1-29dof-wbt-fast-sac \\"
echo "      logger:wandb \\"
echo "      --command.setup_terms.motion_command.params.motion_config.motion_file=\"$OUTPUT_DIR/step4_preprocessed\""
