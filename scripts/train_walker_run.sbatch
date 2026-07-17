#!/bin/bash
#SBATCH --job-name=nr9_walker_run
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Fallback to gpu_a100_short if gpu_h100_short is unavailable:
#   sbatch --partition=gpu_a100_short scripts/train_walker_run.sbatch

# Load uv into PATH
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/NR9"

# Ensure dependencies are installed (no-op if already synced)
uv sync --frozen

# Run training with 48 parallel environments (2 envs per CPU)
uv run python main.py --env_domain walker --env_task stand --episodes 20000 --num_envs 48
