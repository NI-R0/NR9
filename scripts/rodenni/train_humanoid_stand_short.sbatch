#!/bin/bash
#SBATCH --job-name=nr9_humanoid_stand
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# 30-minute training chunks with automatic checkpoint resume
# and automatic rescheduling.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/NR9"
mkdir -p logs
export MUJOCO_GL="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
uv sync --frozen
NUM_ENVS=24
# --------------------------------------------------
# Find the run to resume
# --------------------------------------------------
RESUME_ARGS=""
ACTIVE_RUN=""
if [ -n "$RUN_DIR" ]; then
    echo "RUN_DIR explicitly set to: $RUN_DIR"
    STATE_FILE="$RUN_DIR/checkpoints/state.pkl"
    if [ -f "$STATE_FILE" ]; then
        RESUME_ARGS="--resume $STATE_FILE"
        ACTIVE_RUN="$RUN_DIR"
    else
        echo "ERROR: $STATE_FILE not found."
        exit 1
    fi
elif [ "${FRESH:-0}" = "1" ]; then
    echo "FRESH=1 set - starting a new run."
elif [ -d "runs" ]; then
    # Sort by run-dir name, which embeds a sortable timestamp
    # (run_{date}_{time_to_seconds}), instead of directory mtime,
    # since a directory's mtime does NOT update when a file nested
    # inside it (checkpoints/state.pkl) is merely overwritten.
    LATEST_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec test -f "{}/checkpoints/state.pkl" \; -print \
        | sort | tail -1)
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
# --------------------------------------------------
# Training
# --------------------------------------------------
uv run python main.py \
    --env_domain humanoid_custom \
    --env_task stand \
    --episodes 50000 \
    --num_envs "$NUM_ENVS" \
    --eval_frequency 50 \
    --num_eval_episodes 8 \
    --batch_size 512 \
    --capacity 500000 \
    --sgd_steps_per_learner_step 4 \
    --target_update_period 50 \
    --duration 27 \
    $RESUME_ARGS

TRAIN_EXIT=$?
echo "Training exited with code $TRAIN_EXIT"
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed - NOT rescheduling."
    exit $TRAIN_EXIT
fi
# --------------------------------------------------
# Determine active run
# --------------------------------------------------
if [ -z "$ACTIVE_RUN" ]; then
    ACTIVE_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec test -f "{}/checkpoints/state.pkl" \; -print \
        | sort | tail -1)
fi
if [ -z "$ACTIVE_RUN" ]; then
    echo "ERROR: Could not determine run directory."
    exit 1
fi
echo "Active run directory: $ACTIVE_RUN"
# --------------------------------------------------
# Schedule the next 30-minute chunk
# --------------------------------------------------
NEW_JOB=$(sbatch \
    --export=ALL,RUN_DIR="$ACTIVE_RUN" \
    scripts/sbatch/train_humanoid_stand_short.sbatch)
echo "Submitted next training job: $NEW_JOB"
