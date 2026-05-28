#!/usr/bin/env bash
# SLURM launcher for the WorldPose → GMR retargeting pipeline.
#
# Uses `srun` so that stdout streams live to your terminal.
# A timestamped log is also saved to logs/ for safety.
#
# ---- Quick start -----------------------------------------------------------
#   bash scripts/slurm_worldpose_gmr.sh
#
# ---- Override defaults via env vars ----------------------------------------
#   PARTITION=big_suma_rtx3090 NUM_CPUS=64 MEM=128G \
#       bash scripts/slurm_worldpose_gmr.sh
#
# ---- Resume from Step 2 (skip retargeting) ----------------------------------
#   START_STEP=2 bash scripts/slurm_worldpose_gmr.sh
#
# ---- Tip: run inside tmux/screen so you can detach without killing the job --
#   tmux new -s gmr
#   bash scripts/slurm_worldpose_gmr.sh
#   # Ctrl-b d  to detach;  tmux attach -t gmr  to re-attach
# ----------------------------------------------------------------------------

set -euo pipefail

SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---- SLURM resource config (override via env vars) -------------------------
PARTITION="${PARTITION:-big_suma_rtx3090}"
NUM_CPUS="${NUM_CPUS:-32}"
MEM="${MEM:-64G}"
TIME="${TIME:-08:00:00}"
JOB_NAME="${JOB_NAME:-gmr_worldpose}"

# ---- Pipeline config (passed through to demo script) -----------------------
export WORLDPOSE_DATA="${WORLDPOSE_DATA:-/scratch/jellyho/FIFA/poses}"
export SMPL_MODEL="${SMPL_MODEL:-/scratch/jellyho/smpl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/jellyho/FIFA_gmr_pipeline}"
export ROBOT="${ROBOT:-unitree_g1}"
export INPUT_FPS="${INPUT_FPS:-50}"
export TARGET_FPS="${TARGET_FPS:-50}"
export START_STEP="${START_STEP:-1}"
export SMOOTH_START="${SMOOTH_START:-0}"
# NUM_WORKERS is intentionally NOT exported here — the demo script reads
# $SLURM_CPUS_PER_TASK (set by srun -c) so it matches the actual allocation.
# ----------------------------------------------------------------------------

mkdir -p "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "================================================================"
echo " Submitting GMR WorldPose pipeline via srun"
echo "================================================================"
echo " partition : $PARTITION"
echo " cpus      : $NUM_CPUS   (= --workers inside the job)"
echo " memory    : $MEM"
echo " time      : $TIME"
echo " log file  : $LOG"
echo "================================================================"
echo " Live output below (also saved to $LOG)"
echo "================================================================"
echo ""

srun \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    -c "$NUM_CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="$JOB_NAME" \
    --gres=gpu:0 \
    bash "$PROJECT_ROOT/demo_scripts/demo_worldpose_gmr_pipeline.sh" \
    2>&1 | tee "$LOG"

echo ""
echo "[slurm] Job finished. Full log saved to: $LOG"
