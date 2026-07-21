#!/bin/bash
#SBATCH --job-name=nr9_walker_run
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# SLURM tries the partitions in order; the job runs on the first one with
# free resources. gpu_mi300 (AMD) is excluded (needs ROCm).
# gpu_a100_short is excluded (30 min runtime cap).

# Load uv into PATH
export PATH="$HOME/.local/bin:$PATH"

cd "$HOME/NR9"

# Ensure dependencies are installed (no-op if already synced)
uv sync --frozen

# Derive num_envs from the CPUs SLURM actually assigned to this job
# (3 envs per CPU, matching the 3:1 env-to-CPU ratio used elsewhere).
CPUS=${SLURM_CPUS_PER_TASK:-16}
NUM_ENVS=$((CPUS * 3))
echo "Allocated ${CPUS} CPUs -> using ${NUM_ENVS} parallel envs"

# Run training; num_envs adapts to the allocated CPUs automatically.
uv run python main.py \
  --env_domain walker_3D_ball --env_task walk \
  --episodes 500000 \
  --num_envs "$NUM_ENVS" \
  --eval_frequency "$NUM_ENVS" \
  --capacity 1000000
