"""Episode runner functions shared between training and testing."""

import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from loguru import logger

from src.environment import Environment
from src.agent import MPOAgent
from src.vector_env import ParallelVectorEnv


def run_episode(
    env: Environment,
    agent: MPOAgent,
    args: dict = None,
    explore: bool = True,
    visualize: bool = False,
    profile: bool = False,
):
    """Run a single episode and return (reward, steps, avg_metrics, frames, reward_components)."""
    state = env.reset()
    episode_reward = 0.0
    done = False
    step = 0

    episode_metrics: dict = {}
    updates_count = 0
    reward_components_sum: dict[str, float] = {}
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

        next_state, reward, done, info = env.step(action)
        t2 = time.perf_counter()

        if "reward_components" in info:
            for k, v in info["reward_components"].items():
                reward_components_sum[k] = reward_components_sum.get(k, 0.0) + v

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

    return episode_reward, step, avg_metrics, frames, reward_components_sum


def run_vectorized_episode(
    venv: ParallelVectorEnv,
    agent: MPOAgent,
    max_steps: int,
    profile: bool = False,
):
    """Run one meta-episode of exactly ``max_steps`` steps across all parallel
    environments with **CPU-GPU pipelining** via ThreadPoolExecutor.

    Pipeline layout per iteration::

        Main thread:    env_step(T) ──────────── env_step(T+1) ────────── env_step(T+2)
        Worker thread:                 update(T) ──────────────── update(T+1) ────────

    ``env_step`` (dm_control physics in 48 worker processes) runs on CPU
    in the **main thread**.  ``update_batch`` (JAX/GPU) runs on GPU in a
    **worker thread**.  Because ``env_step`` releases the GIL while
    waiting for workers (sleep-based polling), the worker thread can
    execute JAX code on the GPU simultaneously.  Wall-clock per
    meta-step is roughly ``max(env_step_time, update_time)`` instead of
    their sum.

    Every env steps for the full ``max_steps`` regardless of terminations.
    When an env terminates it auto-resets (inside ``ParallelVectorEnv.step``)
    and starts a new sub-episode immediately – no waiting for other envs.

    Returns a list of (reward, length) tuples one per env, in order.  Only
    the *last completed* sub-episode per env is reported.  If an env has not
    terminated by the final step its in-flight sub-episode is still included.
    """
    num_envs = venv.num_envs
    states = venv.reset()

    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    finished_stats: list[tuple[float, int]] = [None] * num_envs

    episode_metrics: dict = {}
    updates_count = 0
    reward_components_sum: dict[str, float] = {}
    timing = {"select_action": 0.0, "env_step": 0.0, "update": 0.0}

    # ── Pipeline: env_step on main thread, update in worker thread ────
    # Timeline per iteration:
    #   Main thread:  env_step(T) ────────────────── env_step(T+1) ────────
    #   Worker thr:                update(T) ─────── update(T+1) ──────────
    #
    # env_step() blocks the main thread for ~10s but releases the GIL
    # during sleep-based polling.  The worker thread runs update_batch()
    # on the GPU during that time.  Wall-clock ≈ max(env_step, select+update).
    executor = ThreadPoolExecutor(max_workers=1)
    prev_update_future = None

    for step in range(max_steps):
        # ── Phase 1: select_actions (GPU, ~0.3s) ──────────────────────
        t0 = time.perf_counter()
        actions = agent.select_actions(states, explore=True)
        actions_np = np.asarray(actions, dtype=np.float32)
        t1 = time.perf_counter()

        # ── Phase 2: env_step on main thread (CPU, ~10s, releases GIL) ─
        # Worker thread runs update(T-1) concurrently during this block.
        next_states, rewards, dones, infos = venv.step(actions_np)
        t2 = time.perf_counter()

        # ── Phase 3: wait for previous update to finish ────────────────
        if prev_update_future is not None:
            prev_update_future.result()

        # ── Phase 4: submit update for current step to worker ──────────
        terminal_next_states = next_states.copy()
        for i, done in enumerate(dones):
            if done and "terminal_obs" in infos[i]:
                terminal_next_states[i] = infos[i]["terminal_obs"]

        def _update(s, a, r, ns, d, ag=agent):
            return ag.update_batch(s, a, r, ns, d)

        prev_update_future = executor.submit(
            _update, states, actions_np, rewards, terminal_next_states, dones
        )
        t3 = time.perf_counter()

        timing["select_action"] += t1 - t0
        timing["env_step"] += t2 - t1
        timing["update"] += 0.0  # async, time hidden by pipeline

        # ── Episode tracking ──────────────────────────────────────────
        for i in range(num_envs):
            if "reward_components" in infos[i]:
                for k, v in infos[i]["reward_components"].items():
                    reward_components_sum[k] = reward_components_sum.get(k, 0.0) + v / num_envs
            ep_rewards[i] += rewards[i]
            ep_lengths[i] += 1
            if dones[i]:
                finished_stats[i] = (float(ep_rewards[i]), int(ep_lengths[i]))
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0

        states = next_states

    # ── Drain the last update ─────────────────────────────────────────
    if prev_update_future is not None:
        metrics = prev_update_future.result()
        if metrics:
            updates_count += 1
            for k, v in metrics.items():
                episode_metrics[k] = episode_metrics.get(k, 0.0) + v
    executor.shutdown(wait=False)

    # Collect any in-flight sub-episodes.
    for i in range(num_envs):
        if finished_stats[i] is None:
            finished_stats[i] = (float(ep_rewards[i]), int(ep_lengths[i]))

    avg_metrics: dict = {}
    if updates_count > 0:
        avg_metrics = {k: float(v) / updates_count for k, v in episode_metrics.items()}

    last_step = max_steps - 1
    if profile:
        total = timing["select_action"] + timing["env_step"]
        logger.info(
            f"  Timing (vec, {num_envs} envs, {last_step + 1} meta-steps, "
            f"{total:.1f}s total) - "
            f"select_action: {timing['select_action']:.3f}s "
            f"({timing['select_action']/(last_step+1)*1000:.1f}ms/step), "
            f"env_step: {timing['env_step']:.3f}s "
            f"({timing['env_step']/(last_step+1)*1000:.1f}ms/step), "
            f"update: overlapped (pipeline)"
        )

    return finished_stats, avg_metrics, reward_components_sum


def run_episode_with_respawn(
    env: Environment,
    agent: MPOAgent,
    args: dict,
    visualize: bool = False,
):
    """Run a test episode with the same termination/respawn logic as training."""
    max_steps = args["steps"]
    state = env.reset()
    ep_reward = 0.0
    ep_length = 0
    all_rewards: list[float] = []
    all_lengths: list[int] = []
    frames = [] if visualize else None

    for step in range(max_steps):
        if visualize:
            frame = env.render()
            frames.append(frame)

        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)

        ep_reward += reward
        ep_length += 1

        if done:
            all_rewards.append(ep_reward)
            all_lengths.append(ep_length)
            logger.info(
                f"  Sub-episode terminated at step {step + 1}/{max_steps} | "
                f"Reward: {ep_reward:.2f} | Length: {ep_length} -> respawning"
            )
            ep_reward = 0.0
            ep_length = 0
            next_state = env.reset()

        state = next_state

    if ep_length > 0:
        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
        logger.info(
            f"  Final sub-episode (no termination) | "
            f"Reward: {ep_reward:.2f} | Length: {ep_length}"
        )

    return all_rewards, all_lengths, frames