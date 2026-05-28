#!/usr/bin/env bash
# End-to-end pipeline: WorldPose FIFA data → robot motion training clips (GMR edition)
#
# This script replaces the slow MESSI Steps 1-3 with GMR-based retargeting:
#   Step 1  WorldPose SMPL FK + GMR IK (mink/daqp) + headless MuJoCo FK
#           → per-clip qpos/qvel/body_pos_w .npz   (replaces original Steps 1-3)
#   Step 2  Resample to target_fps + optional smooth-start
#           (same as original Step 4, unchanged)
#
# Configurable via environment variables:
#   WORLDPOSE_DATA   path to WorldPose .npz pose files  (default: /scratch/jellyho/FIFA/poses)
#   SMPL_MODEL       path to directory containing SMPL_MALE.pkl
#   OUTPUT_DIR       base output directory
#   ROBOT            target robot name                  (default: unitree_g1)
#   INPUT_FPS        input frame rate of WorldPose data (default: 50)
#   TARGET_FPS       target control frequency Hz        (default: 50)
#   NUM_WORKERS      parallel worker processes          (default: $SLURM_CPUS_PER_TASK or 8)
#   START_STEP       resume from step 1 or 2            (default: 1)
#   SMOOTH_START     set to 1 to prepend default-pose blend-in (default: 0)
#
# Usage (standalone):
#   bash demo_scripts/demo_worldpose_gmr_pipeline.sh
#
# Usage (via SLURM — see scripts/slurm_worldpose_gmr.sh):
#   bash scripts/slurm_worldpose_gmr.sh

set -euo pipefail

SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- CONFIG (override via env vars) ----------------------------------------
WORLDPOSE_DATA="${WORLDPOSE_DATA:-/scratch/jellyho/FIFA/poses}"
SMPL_MODEL="${SMPL_MODEL:-/scratch/jellyho/smpl}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/jellyho/FIFA_gmr_pipeline}"
ROBOT="${ROBOT:-unitree_g1}"
INPUT_FPS="${INPUT_FPS:-50}"
TARGET_FPS="${TARGET_FPS:-50}"
# Use SLURM_CPUS_PER_TASK if available (set by srun -c), else fall back to NUM_WORKERS env var
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
START_STEP="${START_STEP:-1}"
SMOOTH_START="${SMOOTH_START:-0}"
# ----------------------------------------------------------------------------

echo "================================================================"
echo " WorldPose → GMR Pipeline (MESSI edition)"
echo "================================================================"
echo " WORLDPOSE_DATA : $WORLDPOSE_DATA"
echo " SMPL_MODEL     : $SMPL_MODEL"
echo " OUTPUT_DIR     : $OUTPUT_DIR"
echo " ROBOT          : $ROBOT"
echo " INPUT_FPS      : $INPUT_FPS"
echo " TARGET_FPS     : $TARGET_FPS"
echo " NUM_WORKERS    : $NUM_WORKERS"
echo " START_STEP     : $START_STEP"
echo " SMOOTH_START   : $SMOOTH_START"
echo "================================================================"
echo ""

# Prevent thread oversubscription: each worker uses 1 core (mink/daqp is single-threaded)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Activate retargeting conda environment
echo "[setup] Sourcing retargeting environment..."
source "$PROJECT_ROOT/scripts/source_retargeting_setup.sh"

# Make sure GMR submodule is initialised
GMR_DIR="$PROJECT_ROOT/thirdparty/GMR"
if [ ! -f "$GMR_DIR/general_motion_retargeting/__init__.py" ]; then
    echo "[setup] Initialising GMR submodule..."
    git -C "$PROJECT_ROOT" submodule update --init thirdparty/GMR
fi

# Install GMR's missing dependencies into the active env (no version conflicts).
# mink / qpsolvers[daqp] are not part of hsretargeting but install cleanly.
echo "[setup] Checking GMR dependencies (mink, qpsolvers[daqp], loop_rate_limiters)..."
pip install --quiet mink "qpsolvers[daqp]" loop_rate_limiters

# Install / ensure holosoma_retargeting package is available
pip install -e "$PROJECT_ROOT/src/holosoma_retargeting" --quiet

PIPELINE_SCRIPT="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/examples/run_worldpose_gmr_pipeline.py"

SMOOTH_FLAG=""
if [ "$SMOOTH_START" = "1" ]; then
    SMOOTH_FLAG="--smooth_start"
fi

echo ""
echo "[pipeline] Starting GMR retargeting pipeline..."
echo ""

python "$PIPELINE_SCRIPT" \
    --data_dir    "$WORLDPOSE_DATA" \
    --smpl_path   "$SMPL_MODEL" \
    --output_dir  "$OUTPUT_DIR" \
    --robot       "$ROBOT" \
    --fps         "$INPUT_FPS" \
    --target_fps  "$TARGET_FPS" \
    --workers     "$NUM_WORKERS" \
    --start_step  "$START_STEP" \
    $SMOOTH_FLAG

echo ""
echo "================================================================"
echo " Done!  Training-ready clips → $OUTPUT_DIR/step2_preprocessed"
echo "================================================================"
echo ""
echo "To train MESSI WBT with the generated clips:"
echo "  source scripts/source_isaacsim_setup.sh"
echo "  python src/holosoma/holosoma/train_agent.py \\"
echo "      exp:g1-29dof-wbt-fast-sac \\"
echo "      logger:wandb \\"
echo "      --command.setup_terms.motion_command.params.motion_config.motion_file=\"$OUTPUT_DIR/step2_preprocessed\""
