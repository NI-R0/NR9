#!/bin/bash
#SBATCH --job-name=nr9_walker_run
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

uv run python main.py \
  --env_domain walker_3D_ball --env_task walk \
  --episodes 500000 \
  --num_envs "$NUM_ENVS" \
  --eval_frequency 2 \
  --num_eval_episodes "$CPUS" \
  --capacity 1000000
