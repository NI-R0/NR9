#!/bin/bash
#SBATCH --job-name=nr9_walk3d_short
#SBATCH --partition=gpu_a100_short,gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il,dev_gpu_h100,dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# GPU training with automatic resume + convergence-based auto-rescheduling.
#
# SLURM tries partitions in order; the job runs on the first with free
# resources.  Short partitions are listed first (higher chance of being
# free), then long partitions, then dev partitions.  gpu_mi300 (AMD/ROCm)
# is excluded - JAX needs CUDA.  --time=30:00 matches the short-partition
# cap; on long partitions the job simply ends earlier than their limit.
#
# Each job runs for max 25 min (--duration 25) leaving 5 min buffer for
# setup + final checkpoint save.  The script automatically detects a
# previous run directory and resumes from the latest checkpoint.
#
# After training, a convergence check runs: if the best eval reward has
# not improved by at least CONV_THRESHOLD over the last CONV_WINDOW
# evaluations, training is considered converged and no new job is
# submitted.  Otherwise a new job is automatically scheduled.
#
# Convergence parameters (override via --export):
#   CONV_WINDOW=5       - number of recent evals to compare
#   CONV_THRESHOLD=10.0 - minimum improvement to keep training
#
# Usage:
#   1st run (fresh):  sbatch --export=FRESH=1 scripts/sbatch/train_walker_3D_ball_run_short.sbatch
#   Subsequent runs: automatic (the script reschedules itself)
#
# To force a specific run to resume, set the RUN_DIR variable:
#   sbatch --export=RUN_DIR=runs/run_20250101_120000 scripts/sbatch/train_walker_3D_ball_run_short.sbatch
#
# To start fresh (skip auto-detect of old runs), set FRESH=1:
#   sbatch --export=FRESH=1 scripts/sbatch/train_walker_3D_ball_run_short.sbatch

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/NR9"
uv sync --frozen

CPUS=${SLURM_CPUS_PER_TASK:-16}
NUM_ENVS=$((CPUS * 3))
echo "Allocated ${CPUS} CPUs -> using ${NUM_ENVS} parallel envs"

CONV_WINDOW="${CONV_WINDOW:-5}"
CONV_THRESHOLD="${CONV_THRESHOLD:-10.0}"

RESUME_ARGS=""
ACTIVE_RUN=""
if [ -n "$RUN_DIR" ]; then
    echo "RUN_DIR explicitly set to: $RUN_DIR"
    RESUME_ARGS="--resume_dir $RUN_DIR"
    ACTIVE_RUN="$RUN_DIR"
elif [ "${FRESH:-0}" = "1" ]; then
    echo "FRESH=1 set - starting a new run, ignoring existing runs."
elif [ -d "runs" ]; then
    LATEST_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec sh -c '[ -f "$1/checkpoints/latest.pkl" ] && echo "$1"' _ {} \; \
        | xargs -I{} stat -c '%Y {}' {} 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')
    if [ -n "$LATEST_RUN" ]; then
        echo "Auto-detected previous run: $LATEST_RUN"
        RESUME_ARGS="--resume_dir $LATEST_RUN"
        ACTIVE_RUN="$LATEST_RUN"
    fi
fi

if [ -n "$RESUME_ARGS" ]; then
    echo "Resuming training from previous run."
else
    echo "Starting fresh training run."
fi

uv run python main.py \
  --env_domain walker_3D_ball --env_task walk \
  --episodes 500000 \
  --num_envs "$NUM_ENVS" \
  --eval_frequency "$NUM_ENVS" \
  --capacity 1000000 \
  --duration 25 \
  --curriculum \
  --phase1_threshold 100.0 \
  --phase2_threshold 350.0 \
  $RESUME_ARGS

TRAIN_EXIT=$?
echo "Training exited with code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed - NOT rescheduling. Check logs for errors."
    exit $TRAIN_EXIT
fi

if [ -z "$ACTIVE_RUN" ]; then
    ACTIVE_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec sh -c '[ -f "$1/checkpoints/latest.pkl" ] && echo "$1"' _ {} \; \
        | xargs -I{} stat -c '%Y {}' {} 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')
fi

if [ -z "$ACTIVE_RUN" ]; then
    echo "ERROR: Could not determine run directory for convergence check."
    exit 1
fi

echo "Active run directory: $ACTIVE_RUN"

echo "Running convergence check (window=$CONV_WINDOW, threshold=$CONV_THRESHOLD)..."
uv run python scripts/check_convergence.py \
  --run_dir "$ACTIVE_RUN" \
  --window "$CONV_WINDOW" \
  --threshold "$CONV_THRESHOLD"

CONV_EXIT=$?

if [ $CONV_EXIT -eq 0 ]; then
    echo "Training has converged. No new job will be scheduled."
    echo "Best model is in $ACTIVE_RUN/checkpoints/best_ckpt.pkl"
else
    echo "Training has NOT converged - scheduling next job..."
    NEW_JOB=$(sbatch --export=RUN_DIR="$ACTIVE_RUN",CONV_WINDOW="$CONV_WINDOW",CONV_THRESHOLD="$CONV_THRESHOLD" \
        scripts/sbatch/train_walker_3D_ball_run_short.sbatch)
    echo "Submitted: $NEW_JOB"
fi
