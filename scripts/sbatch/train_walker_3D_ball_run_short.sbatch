#!/bin/bash
#SBATCH --job-name=nr9_walk3d_short
#SBATCH --partition=gpu_a100_short,gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il,dev_gpu_h100,dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/NR9"
uv sync --frozen

CPUS=${SLURM_CPUS_PER_TASK:-16}
NUM_ENVS=$((CPUS * 3))
echo "Allocated ${CPUS} CPUs -> using ${NUM_ENVS} parallel envs"

RESUME_ARGS=""
if [ -n "$RUN_DIR" ]; then
    echo "RUN_DIR explicitly set to: $RUN_DIR"
    RESUME_ARGS="--resume_dir $RUN_DIR"
elif [ -d "runs" ]; then
    LATEST_RUN=$(find runs -maxdepth 1 -type d -name "run_*" \
        -exec sh -c '[ -f "$1/checkpoints/latest.pkl" ] && echo "$1"' _ {} \; \
        | xargs -I{} stat -c '%Y {}' {} 2>/dev/null \
        | sort -rn | head -1 | awk '{print $2}')
    if [ -n "$LATEST_RUN" ]; then
        echo "Auto-detected previous run: $LATEST_RUN"
        RESUME_ARGS="--resume_dir $LATEST_RUN"
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
  $RESUME_ARGS
