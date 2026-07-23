#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


# Metrics that are most useful for diagnosing MPO and the environment.
METRICS = [
    "Episode_Reward",
    "Episode_Length",
    "Mean_Eval_Reward",
    "Std_Eval_Reward",
    "loss_critic",
    "loss_policy",
    "eta",
    "alpha_mu",
    "alpha_sigma",
    "log_eta",
    "log_alpha_mean",
    "log_alpha_std",
    "kl_mu",
    "kl_sigma",
    "entropy",
    "max_weight",
    "policy_std",
    "policy_std_min",
    "policy_std_max",
    "policy_mu_min",
    "policy_mu_max",
    "q_mean",
    "q_std",
    "q_range",
    "mean_q_std_per_state",
    "mean_q_range_per_state",
    "current_q_mean",
    "target_q_mean",
    "current_q_std",
    "target_q_std",
    "sampled_action_abs_mean",
    "sampled_action_saturation_fraction",
    # Environment Reward Components
    "Mean_head_height",
    "Mean_torso_upright",
    "Mean_standing",
    "Mean_upright",
    "Mean_stand_reward",
    "Mean_small_control",
    "Mean_reward",
]


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def fmt(value):
    if value is None:
        return "-"
    if abs(value) >= 1000 or (0 < abs(value) < 0.001):
        return f"{value:.3e}"
    return f"{value:.4f}"


def find_latest_stats():
    candidates = list(Path("runs").glob("*/training_stats.json"))
    if not candidates:
        raise FileNotFoundError(
            "No runs/*/training_stats.json files found."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_rows(path):
    with open(path) as f:
        data = json.load(f)

    rows = []

    for episode, values in data.items():
        if not isinstance(values, dict):
            continue

        row = {"episode": int(episode)}

        for key, value in values.items():
            parsed = number(value)
            if parsed is not None:
                row[key] = parsed

        rows.append(row)

    return sorted(rows, key=lambda row: row["episode"])


def value_at_or_before(rows, episode, key):
    values = [
        row[key]
        for row in rows
        if row["episode"] <= episode and key in row
    ]
    return values[-1] if values else None


def print_reward_summary(rows, window):
    rewards = [
        (row["episode"], row["Episode_Reward"])
        for row in rows
        if "Episode_Reward" in row
    ]

    if not rewards:
        return

    all_values = [value for _, value in rewards]
    best_episode, best_reward = max(rewards, key=lambda x: x[1])
    recent = all_values[-window:]

    print("=== REWARD SUMMARY ===")
    print(f"best_training_reward: {fmt(best_reward)} @ episode {best_episode}")
    print(f"last_training_reward: {fmt(all_values[-1])}")
    print(f"last_{len(recent)}_reward_mean: {fmt(mean(recent))}")
    print(f"last_{len(recent)}_reward_std: {fmt(pstdev(recent))}")
    print(f"last_{len(recent)}_reward_max: {fmt(max(recent))}")


def print_eval_summary(rows):
    evaluations = [
        (row["episode"], row["Mean_Eval_Reward"])
        for row in rows
        if "Mean_Eval_Reward" in row
    ]

    if not evaluations:
        print("=== EVALUATION SUMMARY ===")
        print("No evaluation metrics found.")
        return

    best_episode, best_reward = max(evaluations, key=lambda x: x[1])

    print("=== EVALUATION SUMMARY ===")
    print(f"number_of_evaluations: {len(evaluations)}")
    print(f"best_eval_reward: {fmt(best_reward)} @ episode {best_episode}")
    print(
        f"last_eval_reward: "
        f"{fmt(evaluations[-1][1])} @ episode {evaluations[-1][0]}"
    )


def print_evaluation_table(rows, spacing=500):
    evaluations = [
        row for row in rows
        if "Mean_Eval_Reward" in row
    ]

    if not evaluations:
        return

    best_row = max(
        evaluations,
        key=lambda row: row["Mean_Eval_Reward"]
    )

    selected = []

    for row in evaluations:
        episode = row["episode"]

        if (
            not selected
            or episode - selected[-1]["episode"] >= spacing
        ):
            selected.append(row)

    required_rows = [
        evaluations[0],
        best_row,
        evaluations[-1],
    ]

    for row in required_rows:
        if row not in selected:
            selected.append(row)

    selected.sort(key=lambda row: row["episode"])

    print()
    print("=== EVALUATION CHECKPOINTS ===")
    print(f"showing approximately one evaluation every {spacing} episodes")
    print(
        "episode | eval_reward | train_reward | critic_loss | "
        "q_mean | q_range | entropy | max_weight | policy_std | "
        "head_h | upright | stand_r"
    )

    for row in selected:
        episode = row["episode"]

        print(
            f"{episode:7d} | "
            f"{fmt(row.get('Mean_Eval_Reward')):>11} | "
            f"{fmt(value_at_or_before(rows, episode, 'Episode_Reward')):>12} | "
            f"{fmt(value_at_or_before(rows, episode, 'loss_critic')):>11} | "
            f"{fmt(value_at_or_before(rows, episode, 'q_mean')):>6} | "
            f"{fmt(value_at_or_before(rows, episode, 'q_range')):>7} | "
            f"{fmt(value_at_or_before(rows, episode, 'entropy')):>7} | "
            f"{fmt(value_at_or_before(rows, episode, 'max_weight')):>10} | "
            f"{fmt(value_at_or_before(rows, episode, 'policy_std')):>10} | "
            f"{fmt(value_at_or_before(rows, episode, 'Mean_head_height')):>6} | "
            f"{fmt(value_at_or_before(rows, episode, 'Mean_torso_upright')):>6} | "
            f"{fmt(value_at_or_before(rows, episode, 'Mean_stand_reward')):>7}"
        )


def print_diagnostic_points(rows, points):
    if not rows:
        return

    first = rows[0]["episode"]
    last = rows[-1]["episode"]

    if points == 1:
        episodes = [last]
    else:
        step = (last - first) / (points - 1)
        episodes = [
            round(first + i * step)
            for i in range(points)
        ]

    keys = [
        "Episode_Reward",
        "loss_critic",
        "loss_policy",
        "eta",
        "alpha_mu",
        "alpha_sigma",
        "kl_mu",
        "kl_sigma",
        "entropy",
        "max_weight",
        "policy_std",
        "q_mean",
        "q_range",
        "mean_q_std_per_state",
        "mean_q_range_per_state",
        "current_q_mean",
        "target_q_mean",
        "current_q_std",
        "target_q_std",
        "sampled_action_saturation_fraction",
        "Mean_head_height",
        "Mean_torso_upright",
        "Mean_stand_reward",
        "Mean_small_control",
    ]

    print()
    print("=== TRAINING DIAGNOSTIC CHECKPOINTS ===")
    print("episode | " + " | ".join(keys))

    for episode in episodes:
        values = []

        for key in keys:
            values.append(
                fmt(value_at_or_before(rows, episode, key))
            )

        print(
            f"{episode:7d} | " +
            " | ".join(values)
        )


def print_final_values(rows):
    if not rows:
        return

    print()
    print("=== FINAL AVAILABLE METRICS ===")

    last = rows[-1]

    for key in METRICS:
        if key in last:
            print(f"{key}: {fmt(last[key])}")


def print_metric_summary(rows, key):
    values = [
        (row["episode"], row[key])
        for row in rows
        if key in row
    ]

    if not values:
        return

    episodes = [x[0] for x in values]
    numbers = [x[1] for x in values]

    min_index = numbers.index(min(numbers))
    max_index = numbers.index(max(numbers))

    print(
        f"{key}: "
        f"first={fmt(numbers[0])} "
        f"last={fmt(numbers[-1])} "
        f"min={fmt(numbers[min_index])}@{episodes[min_index]} "
        f"max={fmt(numbers[max_index])}@{episodes[max_index]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stats_file",
        nargs="?",
        type=Path,
        help="Path to training_stats.json. "
             "If omitted, the newest runs/*/training_stats.json is used.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Number of recent training episodes to summarize.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=8,
        help="Number of evenly spaced diagnostic checkpoints.",
    )

    parser.add_argument(
        "--eval-spacing",
        type=int,
        default=500,
        help="Approximate episode spacing between printed evaluations.",
    )

    args = parser.parse_args()

    stats_file = args.stats_file or find_latest_stats()
    rows = load_rows(stats_file)

    if not rows:
        raise RuntimeError("No usable metric rows found.")

    print(f"file: {stats_file}")
    print(
        f"episode_range: "
        f"{rows[0]['episode']}-{rows[-1]['episode']}"
    )
    print(f"rows: {len(rows)}")

    print_reward_summary(rows, args.window)
    print_eval_summary(rows)
    print_evaluation_table(rows, args.eval_spacing)
    print_diagnostic_points(rows, args.points)
    print_final_values(rows)

    print()
    print("=== METRIC RANGES ===")

    range_metrics = [
        "loss_critic",
        "loss_policy",
        "eta",
        "alpha_mu",
        "alpha_sigma",
        "kl_mu",
        "kl_sigma",
        "entropy",
        "max_weight",
        "policy_std",
        "q_mean",
        "q_range",
        "mean_q_std_per_state",
        "mean_q_range_per_state",
        "current_q_mean",
        "target_q_mean",
        "current_q_std",
        "target_q_std",
        "sampled_action_saturation_fraction",
        "Mean_head_height",
        "Mean_torso_upright",
        "Mean_stand_reward",
        "Mean_small_control",
    ]

    for key in range_metrics:
        print_metric_summary(rows, key)


if __name__ == "__main__":
    main()