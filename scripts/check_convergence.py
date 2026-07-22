#!/usr/bin/env python3
"""Check whether training has converged and decide whether to reschedule.

Reads ``training_stats.json`` and ``training_meta.json`` from a run
directory and compares the best eval reward of the most recent
``--window`` evaluations with the best of the ``--window`` evaluations
before that.  If the improvement is below ``--threshold``, the run is
considered converged.

Exits with code 0 if converged (stop), 1 if not converged (reschedule).

Usage:
    python scripts/check_convergence.py --run_dir runs/run_20250101_120000
    python scripts/check_convergence.py --run_dir runs/run_xxx --window 5 --threshold 10
"""

import argparse
import json
import os
import sys
from loguru import logger


def load_eval_rewards(run_dir: str) -> list[tuple[int, float]]:
    """Load (episode, mean_eval_reward) pairs from training_stats.json, sorted by episode."""
    stats_path = os.path.join(run_dir, "training_stats.json")
    if not os.path.isfile(stats_path):
        logger.error(f"No training_stats.json found in {run_dir}")
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    eval_rewards = []
    for episode_str, ep_stats in stats.items():
        if "Mean_Eval_Reward" in ep_stats:
            episode = int(episode_str)
            reward = float(ep_stats["Mean_Eval_Reward"])
            eval_rewards.append((episode, reward))

    eval_rewards.sort(key=lambda x: x[0])
    return eval_rewards


def check_convergence(run_dir: str, window: int, threshold: float) -> bool:
    """Return True if converged (improvement below threshold)."""
    eval_rewards = load_eval_rewards(run_dir)

    if len(eval_rewards) < 2 * window:
        logger.info(
            f"Not enough eval data yet: {len(eval_rewards)} evals, "
            f"need at least {2 * window} (2 × window={window}). "
            "Not converged - continue training."
        )
        return False

    recent = eval_rewards[-window:]
    previous = eval_rewards[-2 * window:-window]

    best_recent = max(r for _, r in recent)
    best_previous = max(r for _, r in previous)

    improvement = best_recent - best_previous

    logger.info(
        f"Convergence check (window={window}, threshold={threshold}):\n"
        f"  Previous {window} evals (ep {previous[0][0]}-{previous[-1][0]}): "
        f"best = {best_previous:.2f}\n"
        f"  Recent  {window} evals (ep {recent[0][0]}-{recent[-1][0]}): "
        f"best = {best_recent:.2f}\n"
        f"  Improvement: {improvement:.2f}"
    )

    if improvement < threshold:
        logger.success(
            f"Converged! Improvement {improvement:.2f} < threshold {threshold}. "
            "Stopping automatic rescheduling."
        )
        return True
    else:
        logger.info(
            f"Not converged - improvement {improvement:.2f} >= threshold {threshold}. "
            "Rescheduling."
        )
        return False


def main():
    parser = argparse.ArgumentParser(description="Check training convergence.")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the run directory to check.")
    parser.add_argument("--window", type=int, default=5,
                        help="Number of recent eval points to compare (default: 5).")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Minimum improvement to continue training (default: 10.0).")
    args = parser.parse_args()

    converged = check_convergence(args.run_dir, args.window, args.threshold)
    sys.exit(0 if converged else 1)


if __name__ == "__main__":
    main()
