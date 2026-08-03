#!/bin/bash
#SBATCH --job-name=nr9_humanoid_stand
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
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
# --------------------------------------------------
# Training
# --------------------------------------------------
uv run python main.py --task=test --load_dir=~/NR9/runs/run_20260802_220324/ --env_domain=humanoid_custom --env_task=stand --visualize
