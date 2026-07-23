#!/bin/bash
#SBATCH --job-name=nr9_walker_stand
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
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

uv run python main.py --env_domain walker --env_task stand --episodes 10000 --num_envs 24 --eval_frequency 2 --num_eval_episodes 8 --sgd_steps_per_learner_step 24
