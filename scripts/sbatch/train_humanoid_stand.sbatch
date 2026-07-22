#!/bin/bash
#SBATCH --job-name=nr9_walker_run
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/NR9"
uv sync --frozen

uv run python main.py --env_domain humanoid_custom --env_task stand --episodes 10000 --num_envs 72 --eval_frequency 72
