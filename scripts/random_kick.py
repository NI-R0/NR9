#!/usr/bin/env python3
"""Interaktiver Random-Agent-Viewer für die Custom Environments.

Nutze mit:
    uv run python scripts/random_kick.py              # hipp_walker stand (Default)
    uv run python scripts/random_kick.py --domain cartpole_ball --task kick
    uv run python scripts/random_kick.py --list       # Alle verfügbaren Domains/Tasks auflisten
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Projektstamm (ein Verzeichnis über scripts/) zum Python-Pfad hinzufügen
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.environments.suite as suite
from dm_control import viewer


def list_tasks():
    """Alle verfügbaren (domain, task)-Kombinationen auflisten."""
    print("Verfügbare Tasks:")
    for domain, task in suite.ALL_TASKS:
        print(f"  {domain:25s}  {task}")
    print(f"\nInsgesamt: {len(suite.ALL_TASKS)} Tasks")


def main():
    parser = argparse.ArgumentParser(description="Random-Agent Viewer für Custom Environments")
    parser.add_argument("--domain", type=str, default="hipp_walker", help="Domain-Name (z.B. cartpole_ball, hipp_walker)")
    parser.add_argument("--task", type=str, default="stand", help="Task-Name (z.B. kick, stand)")
    parser.add_argument("--list", action="store_true", help="Alle verfügbaren Tasks auflisten")
    parser.add_argument("--visualize-reward", action="store_true", help="Reward als Farbe im Viewer darstellen")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    print(f"Lade Environment: domain={args.domain}, task={args.task}")
    try:
        env = suite.load(domain_name=args.domain, task_name=args.task, visualize_reward=args.visualize_reward)
    except ValueError as e:
        print(f"Fehler: {e}")
        print("\nNutze '--list' um alle verfügbaren Tasks zu sehen.")
        sys.exit(1)

    print(f"Observation spec: {env.observation_spec()}")
    print(f"Action spec: {env.action_spec()}")
    print("\nStarte Viewer. Drücke 'Escape' zum Beenden.\n")

    # Einfacher Random-Agent
    class RandomAgent:
        def __init__(self, action_spec):
            self._action_spec = action_spec

        def __call__(self, time_step):
            return np.random.uniform(
                self._action_spec.minimum,
                self._action_spec.maximum,
                self._action_spec.shape,
            )

    agent = RandomAgent(env.action_spec())

    # Environment im Viewer mit dem Agenten starten
    viewer.launch(environment_loader=env, policy=agent)


if __name__ == "__main__":
    main()