#!/bin/bash
#SBATCH --job-name=nr9_humanoid
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# 30-minute training chunks with automatic checkpoint resume
# and automatic rescheduling. Supports multiple independent,
# concurrent training chains, each pinned to its own RUN_DIR.
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
# Determine this chain's run directory
# --------------------------------------------------
# RUN_DIR is the single source of truth for a training chain
# across all its resubmissions. We decide it *now*, deterministically,
# rather than scanning `runs/` and grabbing "the latest" after the
# fact - that approach breaks as soon as more than one chain is
# active, since it can't tell which run dir belongs to which chain.
#
# - RUN_DIR passed in + checkpoint present  -> resume that chain.
# - RUN_DIR passed in + no checkpoint yet    -> fresh run, but at
#   that explicit, caller-chosen directory (lets you name chains,
#   e.g. runs/seed1, runs/seed2, and run them side by side).
# - RUN_DIR not passed in                    -> mint a fresh, unique
#   directory right now (timestamp + job id), so this new chain
#   can't collide with any other chain's directory.
RESUME_ARGS=""

if [ -z "$RUN_NAME" ]; then
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-$$}"
    echo "No RUN_DIR given - starting new run at: runs/$RUN_NAME"
else
    echo "RUN_DIR: runs/$RUN_NAME"
fi

STATE_FILE="runs/$RUN_NAME/checkpoints/state.pkl"
# Every chunk of a chain resubmits from this frozen, per-run copy of
# the script rather than the shared template at scripts/rodenni/. That
# way, editing the template later (e.g. to change hyperparameters for
# a new, separate chain) can never leak into a chain that's already
# running - each chain keeps using exactly the script version it
# started with, SBATCH directives and all.
SCRIPT_COPY="runs/$RUN_NAME/job_script.sbatch"

if [ "${FRESH:-0}" != "1" ] && [ -f "$STATE_FILE" ]; then
    echo "Found checkpoint - resuming from $STATE_FILE"
    RESUME_ARGS="--resume $STATE_FILE"
else
    if [ "${FRESH:-0}" = "1" ]; then
        echo "FRESH=1 set - ignoring any existing checkpoint at runs/$RUN_NAME."
    fi
    echo "Starting fresh training run at: runs/$RUN_NAME"
fi
ACTIVE_RUN="$RUN_NAME"

# --------------------------------------------------
# Training
# --------------------------------------------------
uv run python main.py \
    --run_name "$RUN_NAME" \
    --env_domain humanoid \
    --env_task walk \
    --episodes 50000 \
    --num_envs "$NUM_ENVS" \
    --eval_frequency 50 \
    --num_eval_episodes 8 \
    --batch_size 512 \
    --capacity 1000000 \
    --sgd_steps_per_learner_step 4 \
    --target_update_period 50 \
    --duration 27 \
    $RESUME_ARGS \

TRAIN_EXIT=$?
echo "Training exited with code $TRAIN_EXIT"
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed - NOT rescheduling."
    exit $TRAIN_EXIT
fi

mkdir -p "runs/$RUN_NAME"
# Snapshot the script as it exists right now for this chain's use
# on every future resubmission, regardless of later template edits.
cp "$0" "$SCRIPT_COPY"

echo "Active run directory: $ACTIVE_RUN"

# --------------------------------------------------
# Schedule the next 30-minute chunk for THIS chain
# --------------------------------------------------
# Always resubmit the frozen per-run copy, not the shared template,
# so this chain stays on the hyperparameters/settings it started with.
NEW_JOB=$(sbatch \
    --export=ALL,FRESH=0,RUN_NAME="$ACTIVE_RUN" \
    "$SCRIPT_COPY")
echo "Submitted next training job: $NEW_JOB"
