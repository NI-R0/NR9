#!/usr/bin/env python3
"""Check whether training has converged, is stagnant, or should continue.

Three possible outcomes:

* **Converged (exit 0)** – The agent reached the final curriculum phase,
  achieved at least ``--min_reward``, and the improvement over the last
  ``--window`` evals is below ``--threshold``.  Stop training – success.

* **Stagnant (exit 2)** – The agent has been training for a long time
  (``--stagnation_window`` evals) without meaningful improvement
  (improvement < ``--stagnation_threshold``).  Stop training – likely a
  bug or fundamental issue.  No reschedule.

* **Continue (exit 1)** – None of the above; schedule the next job.

Exit codes
----------
0  converged  – stop, training successful
1  continue   – reschedule next job
2  stagnant   – stop, training is stuck (possible bug)
"""

import argparse
import json
import os
import sys
from loguru import logger

# Curriculum phases (must match src/train.py)
_PHASE_STAND = 0
_PHASE_APPROACH = 1
_PHASE_FULL = 2


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


def load_training_meta(run_dir: str) -> dict:
    """Load training_meta.json (phase, episode, best_eval_reward). Returns empty dict if missing."""
    meta_path = os.path.join(run_dir, "checkpoints", "training_meta.json")
    if not os.path.isfile(meta_path):
        logger.warning(f"No training_meta.json found in {meta_path}")
        return {}
    with open(meta_path) as f:
        return json.load(f)


def check_convergence(
    run_dir: str,
    window: int,
    threshold: float,
    min_reward: float,
    require_final_phase: bool,
    stagnation_window: int,
    stagnation_threshold: float,
) -> int:
    """Decide whether to stop (converged/stagnant) or continue training.

    Returns 0 (converged), 1 (continue), or 2 (stagnant).
    """
    eval_rewards = load_eval_rewards(run_dir)
    meta = load_training_meta(run_dir)
    current_phase = meta.get("phase", _PHASE_STAND)

    logger.info(
        f"Convergence check:\n"
        f"  Total evals: {len(eval_rewards)}\n"
        f"  Current phase: {current_phase} "
        f"(0=STAND, 1=APPROACH, 2=FULL)\n"
        f"  Best eval reward so far: {meta.get('best_eval_reward', '?')}"
    )

    # ------------------------------------------------------------------
    # Not enough data yet – always continue.
    # ------------------------------------------------------------------
    if len(eval_rewards) < 2 * window:
        logger.info(
            f"Not enough eval data yet: {len(eval_rewards)} evals, "
            f"need at least {2 * window} (2 × window={window}). "
            "Continue training."
        )
        return 1

    recent = eval_rewards[-window:]
    previous = eval_rewards[-2 * window:-window]

    best_recent = max(r for _, r in recent)
    best_previous = max(r for _, r in previous)
    improvement = best_recent - best_previous

    logger.info(
        f"  Previous {window} evals (ep {previous[0][0]}-{previous[-1][0]}): "
        f"best = {best_previous:.2f}\n"
        f"  Recent  {window} evals (ep {recent[0][0]}-{recent[-1][0]}): "
        f"best = {best_recent:.2f}\n"
        f"  Improvement: {improvement:.2f}"
    )

    # ------------------------------------------------------------------
    # 1) Convergence check – are we done?
    #    Requires: final curriculum phase, reward >= min_reward, and
    #    improvement below threshold.
    # ------------------------------------------------------------------
    meets_phase = not require_final_phase or current_phase >= _PHASE_FULL
    meets_reward = best_recent >= min_reward
    meets_improvement = improvement < threshold

    if meets_phase and meets_reward and meets_improvement:
        logger.success(
            f"CONVERGED! Improvement {improvement:.2f} < threshold {threshold}, "
            f"best recent reward {best_recent:.2f} >= min_reward {min_reward:.2f}, "
            f"phase {current_phase}. Stopping – training successful."
        )
        return 0

    # ------------------------------------------------------------------
    # 2) Stagnation check – long-term no improvement → abort.
    #    Only checked when NOT converged, so a stable agent at the
    #    target level is not falsely flagged as stagnant.
    # ------------------------------------------------------------------
    if len(eval_rewards) >= stagnation_window:
        stagnation_evals = eval_rewards[-stagnation_window:]
        # Compare best of the second half vs best of the first half of
        # the stagnation window.
        mid = stagnation_window // 2
        best_first_half = max(r for _, r in stagnation_evals[:mid])
        best_second_half = max(r for _, r in stagnation_evals[mid:])
        stagnation_improvement = best_second_half - best_first_half

        logger.info(
            f"  Stagnation check (window={stagnation_window}, "
            f"threshold={stagnation_threshold}):\n"
            f"    First half best  = {best_first_half:.2f}\n"
            f"    Second half best = {best_second_half:.2f}\n"
            f"    Improvement      = {stagnation_improvement:.2f}"
        )

        if stagnation_improvement < stagnation_threshold:
            logger.warning(
                f"STAGNANT! Long-term improvement {stagnation_improvement:.2f} "
                f"< stagnation threshold {stagnation_threshold} over "
                f"{stagnation_window} evals. "
                "Stopping – possible bug or fundamental issue."
            )
            return 2
    else:
        logger.info(
            f"  Stagnation check skipped: only {len(eval_rewards)} evals, "
            f"need {stagnation_window}."
        )

    # ------------------------------------------------------------------
    # 3) Not converged, not stagnant – continue training.
    # ------------------------------------------------------------------
    if not meets_phase:
        logger.info(
            f"Not converged – still in curriculum phase {current_phase} "
            f"(need phase {_PHASE_FULL}=FULL). Continue training."
        )
    elif not meets_reward:
        logger.info(
            f"Not converged – best recent reward {best_recent:.2f} "
            f"< min_reward {min_reward:.2f}. Continue training."
        )
    else:
        logger.info(
            f"Not converged – improvement {improvement:.2f} >= threshold {threshold}. "
            "Rescheduling."
        )
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Check training convergence, stagnation, or continue."
    )
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the run directory to check.")
    parser.add_argument("--window", type=int, default=5,
                        help="Number of recent eval points to compare (default: 5).")
    parser.add_argument("--threshold", type=float, default=10.0,
                        help="Minimum improvement to continue training (default: 10.0).")

    parser.add_argument("--min_reward", type=float, default=100.0,
                        help="Minimum best recent reward to allow convergence "
                             "(default: 100.0). Below this, always continue.")
    parser.add_argument("--require_final_phase", action="store_true", default=True,
                        help="Only converge in the final curriculum phase (default: True).")
    parser.add_argument("--no_require_final_phase", dest="require_final_phase",
                        action="store_false",
                        help="Allow convergence in any curriculum phase.")

    parser.add_argument("--stagnation_window", type=int, default=20,
                        help="Number of evals to check for stagnation (default: 20). "
                             "Must be >= 2× the comparison half.")
    parser.add_argument("--stagnation_threshold", type=float, default=1.0,
                        help="If improvement over stagnation_window evals is below "
                             "this, training is stagnant (default: 1.0).")
    args = parser.parse_args()

    result = check_convergence(
        run_dir=args.run_dir,
        window=args.window,
        threshold=args.threshold,
        min_reward=args.min_reward,
        require_final_phase=args.require_final_phase,
        stagnation_window=args.stagnation_window,
        stagnation_threshold=args.stagnation_threshold,
    )
    sys.exit(result)


if __name__ == "__main__":
    main()
