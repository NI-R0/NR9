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
# After training, a convergence check runs: if the smoothed improvement
# over the last CONV_CONVERGENCE_EPISODES episodes is below
# CONV_THRESHOLD, training is considered converged and no new job is
# submitted.  Otherwise a new job is automatically scheduled.
#
# Convergence parameters (override via --export):
#   CONV_CONVERGENCE_EPISODES=2000 - episodes to check for convergence
#   CONV_THRESHOLD=10.0         - max smoothed improvement to consider converged
#   CONV_MIN_REWARD=100.0       - don't converge below this reward
#   CONV_STAGNATION_WINDOW=20   - evals to check for stagnation
#   CONV_STAGNATION_THRESHOLD=1.0 - improvement below this over stagnation_window = stagnant
#
# Exit codes from check_convergence.py:
#   0 = converged  (stop, success)
#   1 = continue   (reschedule)
#   2 = stagnant   (stop, likely bug - no reschedule)
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
mkdir -p logs

# Environment variables for MuJoCo and to avoid subprocesses creating competing threads
export MUJOCO_GL="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

uv sync --frozen

CPUS=${SLURM_CPUS_PER_TASK:-16}
NUM_ENVS=$((CPUS * 3))
echo "Allocated ${CPUS} CPUs -> using ${NUM_ENVS} parallel envs"

CONV_CONVERGENCE_EPISODES="${CONV_CONVERGENCE_EPISODES:-2000}"
CONV_THRESHOLD="${CONV_THRESHOLD:-10.0}"
CONV_MIN_REWARD="${CONV_MIN_REWARD:-100.0}"
CONV_STAGNATION_WINDOW="${CONV_STAGNATION_WINDOW:-20}"
CONV_STAGNATION_THRESHOLD="${CONV_STAGNATION_THRESHOLD:-1.0}"

RESUME_ARGS=""
ACTIVE_RUN=""
if [ -n "$RUN_DIR" ]; then
    echo "RUN_DIR explicitly set to: $RUN_DIR"
    STATE_FILE="$RUN_DIR/checkpoints/state.pkl"
    if [ -f "$STATE_FILE" ]; then
        RESUME_ARGS="--resume $STATE_FILE"
    else
        echo "WARNING: $STATE_FILE not found - starting fresh in $RUN_DIR"
    fi
    ACTIVE_RUN="$RUN_DIR"
elif [ "${FRESH:-0}" = "1" ]; then
    echo "FRESH=1 set - starting a new run, ignoring existing runs."
elif [ -d "runs" ]; then
    LATEST_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec sh -c '[ -f "$1/checkpoints/state.pkl" ] && echo "$1"' _ {} \; \
        | xargs -I{} stat -c '%Y {}' {} 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')
    if [ -n "$LATEST_RUN" ]; then
        echo "Auto-detected previous run: $LATEST_RUN"
        RESUME_ARGS="--resume $LATEST_RUN/checkpoints/state.pkl"
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
  --phase1_threshold 200.0 \
  --phase2_threshold 300.0 \
  --phase3_threshold 700.0 \
  $RESUME_ARGS

TRAIN_EXIT=$?
echo "Training exited with code $TRAIN_EXIT"

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed - NOT rescheduling. Check logs for errors."
    exit $TRAIN_EXIT
fi

if [ -z "$ACTIVE_RUN" ]; then
    ACTIVE_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec sh -c '[ -f "$1/checkpoints/state.pkl" ] && echo "$1"' _ {} \; \
        | xargs -I{} stat -c '%Y {}' {} 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')
fi

if [ -z "$ACTIVE_RUN" ]; then
    echo "ERROR: Could not determine run directory for convergence check."
    exit 1
fi

echo "Active run directory: $ACTIVE_RUN"

echo "Running convergence check (convergence_episodes=$CONV_CONVERGENCE_EPISODES, threshold=$CONV_THRESHOLD, min_reward=$CONV_MIN_REWARD, stagnation_window=$CONV_STAGNATION_WINDOW, stagnation_threshold=$CONV_STAGNATION_THRESHOLD)..."
uv run python scripts/check_convergence.py \
  --run_dir "$ACTIVE_RUN" \
  --convergence_episodes "$CONV_CONVERGENCE_EPISODES" \
  --threshold "$CONV_THRESHOLD" \
  --min_reward "$CONV_MIN_REWARD" \
  --stagnation_window "$CONV_STAGNATION_WINDOW" \
  --stagnation_threshold "$CONV_STAGNATION_THRESHOLD"

CONV_EXIT=$?

if [ $CONV_EXIT -eq 0 ]; then
    echo "Training has converged. No new job will be scheduled."
    echo "Best model is in $ACTIVE_RUN/checkpoints/best_ckpt.pkl"
elif [ $CONV_EXIT -eq 2 ]; then
    echo "Training has STAGNATED (possible bug or fundamental issue). No new job will be scheduled."
    echo "Check logs in $ACTIVE_RUN/logs/ for details."
    echo "Best model so far is in $ACTIVE_RUN/checkpoints/best_ckpt.pkl"
else
    echo "Training has NOT converged - scheduling next job..."
    NEW_JOB=$(sbatch --export=RUN_DIR="$ACTIVE_RUN",CONV_CONVERGENCE_EPISODES="$CONV_CONVERGENCE_EPISODES",CONV_THRESHOLD="$CONV_THRESHOLD",CONV_MIN_REWARD="$CONV_MIN_REWARD",CONV_STAGNATION_WINDOW="$CONV_STAGNATION_WINDOW",CONV_STAGNATION_THRESHOLD="$CONV_STAGNATION_THRESHOLD" \
        scripts/sbatch/train_walker_3D_ball_run_short.sbatch)
    echo "Submitted: $NEW_JOB"
fi
