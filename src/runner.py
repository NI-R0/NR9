"""Episode runner functions shared between training and testing."""

import time
import numpy as np
from loguru import logger

from src.environment import Environment
from src.agent import MPOAgent
from src.vector_env import ParallelVectorEnv


def run_episode(
    env: Environment,
    agent: MPOAgent,
    explore: bool = True,
    visualize: bool = False,
    profile: bool = False,
):
    """Run a single episode and return (reward, steps, avg_metrics, frames)."""
    state = env.reset()
    episode_reward = 0.0
    done = False
    step = 0

    episode_metrics: dict = {}
    updates_count = 0
    timing = {"select_action": 0.0, "env_step": 0.0, "update": 0.0}

    frames = [] if visualize else None
    while not done and step < env.ep_max_steps:
        if visualize:
            frames.append(env.render())

        t0 = time.perf_counter()
        action = agent.select_action(state, explore=explore)
        if profile and hasattr(action, "block_until_ready"):
            action.block_until_ready()
        t1 = time.perf_counter()

        next_state, reward, done, _ = env.step(action)
        t2 = time.perf_counter()

        if explore:
            metrics = agent.update(state, action, reward, next_state, done)
            if profile and isinstance(metrics, dict):
                for v in metrics.values():
                    if hasattr(v, "block_until_ready"):
                        v.block_until_ready()
            t3 = time.perf_counter()
            timing["update"] += t3 - t2
            if metrics:
                updates_count += 1
                for k, v in metrics.items():
                    episode_metrics[k] = episode_metrics.get(k, 0.0) + v
        else:
            t3 = t2

        timing["select_action"] += t1 - t0
        timing["env_step"] += t2 - t1

        state = next_state
        episode_reward += reward
        step += 1

    avg_metrics: dict = {}
    if updates_count > 0:
        avg_metrics = {k: float(v) / updates_count for k, v in episode_metrics.items()}

    if profile and step > 0:
        total = timing["select_action"] + timing["env_step"] + timing["update"]
        logger.info(
            f"  Timing (episode, {step} steps, {total:.1f}s total) - "
            f"select_action: {timing['select_action']:.3f}s "
            f"({timing['select_action']/step*1000:.1f}ms/step), "
            f"env_step: {timing['env_step']:.3f}s "
            f"({timing['env_step']/step*1000:.1f}ms/step), "
            f"update: {timing['update']:.3f}s "
            f"({timing['update']/step*1000:.1f}ms/step)"
        )

    return episode_reward, step, avg_metrics, frames


def run_vectorized_episode(
    venv: ParallelVectorEnv,
    agent: MPOAgent,
    max_steps: int,
    profile: bool = False,
):
    """Run one meta-episode across all parallel environments.

    All envs step simultaneously until every env has completed at least one
    episode.  When an env finishes it auto-resets (inside
    ``ParallelVectorEnv.step``) and the terminal observation is used for the
    buffer before the new observation is carried forward.

    Returns a list of (reward, length) tuples - one per env, in order.
    """
    num_envs = venv.num_envs
    states = venv.reset()

    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    finished = [False] * num_envs
    finished_stats: list[tuple[float, int]] = [None] * num_envs

    episode_metrics: dict = {}
    updates_count = 0
    timing = {"select_action": 0.0, "env_step": 0.0, "update": 0.0}

    for step in range(max_steps):
        t0 = time.perf_counter()
        actions = agent.select_actions(states, explore=True)
        if profile and hasattr(actions, "block_until_ready"):
            actions.block_until_ready()
        t1 = time.perf_counter()

        actions_np = np.asarray(actions, dtype=np.float32)
        next_states, rewards, dones, infos = venv.step(actions_np)
        t2 = time.perf_counter()

        terminal_next_states = next_states.copy()
        for i, done in enumerate(dones):
            if done and "terminal_obs" in infos[i]:
                terminal_next_states[i] = infos[i]["terminal_obs"]

        metrics = agent.update_batch(states, actions_np, rewards, terminal_next_states, dones)
        if profile and isinstance(metrics, dict):
            for v in metrics.values():
                if hasattr(v, "block_until_ready"):
                    v.block_until_ready()
        t3 = time.perf_counter()

        timing["select_action"] += t1 - t0
        timing["env_step"] += t2 - t1
        timing["update"] += t3 - t2

        if metrics:
            updates_count += 1
            for k, v in metrics.items():
                episode_metrics[k] = episode_metrics.get(k, 0.0) + v

        for i in range(num_envs):
            ep_rewards[i] += rewards[i]
            ep_lengths[i] += 1
            if dones[i] and not finished[i]:
                finished[i] = True
                finished_stats[i] = (float(ep_rewards[i]), int(ep_lengths[i]))
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0

        states = next_states

        if all(finished):
            break

    for i in range(num_envs):
        if finished_stats[i] is None:
            finished_stats[i] = (float(ep_rewards[i]), int(ep_lengths[i]))

    avg_metrics: dict = {}
    if updates_count > 0:
        avg_metrics = {k: float(v) / updates_count for k, v in episode_metrics.items()}

    if profile:
        total = timing["select_action"] + timing["env_step"] + timing["update"]
        logger.info(
            f"  Timing (vec, {num_envs} envs, {step + 1} meta-steps, {total:.1f}s total) - "
            f"select_action: {timing['select_action']:.3f}s "
            f"({timing['select_action']/(step+1)*1000:.1f}ms/step), "
            f"env_step: {timing['env_step']:.3f}s "
            f"({timing['env_step']/(step+1)*1000:.1f}ms/step), "
            f"update: {timing['update']:.3f}s "
            f"({timing['update']/(step+1)*1000:.1f}ms/step)"
        )

    return finished_stats, avg_metrics
