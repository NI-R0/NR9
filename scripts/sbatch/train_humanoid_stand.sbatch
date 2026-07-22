#!/bin/bash
#SBATCH --job-name=nr9_humanoid_stand
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fallback to gpu_a100_short if gpu_h100_short is unavailable:
#   sbatch --partition=gpu_a100_short scripts/train_humanoid_stand.sbatch

export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/NR9"
mkdir -p logs

# Environment variables for MuJoCo and to avoid subprocesses creating competing threads
export MUJOCO_GL="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Ensure dependencies are installed (no-op if already synced)
uv sync --frozen

# Run training with 72 parallel environments (3 envs per CPU)
uv run python main.py --env_domain humanoid_custom --env_task stand --episodes 10000 --num_envs 24 --eval_frequency 24 --sgd_steps_per_learner_step 24
