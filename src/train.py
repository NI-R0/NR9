import os
import signal
import time
import numpy as np
from loguru import logger
from src.collector import StatsCollector
from src.buffer import NStepTransitionBuffer
from src.environment import Environment
from src.agent import SoccerAgent
from src.networks import ActorNetwork, CriticNetwork
from src.vector_env import ParallelVectorEnv


def run_episode(
    env: Environment,
    agent: SoccerAgent,
    args: dict,
    explore: bool = True,
    visualize: bool = False,
    profile: bool = False,
):
    state = env.reset()
    episode_reward = 0.0
    done = False
    step = 0

    episode_metrics = {}
    avg_metrics = {}
    updates_count = 0
    reward_components_sum: dict[str, float] = {}

    timing = {"select_action": 0.0, "env_step": 0.0, "update": 0.0}

    frames = [] if visualize else None
    while not done and step < env.ep_max_steps:
        if visualize:
            frame = env.render()
            frames.append(frame)

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

    if updates_count > 0:
        avg_metrics = {k: float(v) / updates_count for k, v in episode_metrics.items()}

    if profile and step > 0:
        total = timing["select_action"] + timing["env_step"] + timing["update"]
        logger.info(
            f"  Timing (episode, {step} steps, {total:.1f}s total) - "
            f"select_action: {timing['select_action']:.3f}s "
            f"({timing['select_action'] / step * 1000:.1f}ms/step), "
            f"env_step: {timing['env_step']:.3f}s "
            f"({timing['env_step'] / step * 1000:.1f}ms/step), "
            f"update: {timing['update']:.3f}s "
            f"({timing['update'] / step * 1000:.1f}ms/step)"
        )

    return episode_reward, step, avg_metrics, frames, reward_components_sum


def run_vectorized_episode(
    venv: ParallelVectorEnv, agent: SoccerAgent, args: dict, profile: bool = False
):
    """Run one ``meta-episode'' across ``num_envs`` parallel environments.

    All envs step simultaneously for the full ``max_steps`` (e.g. 1000)
    steps.  When an env terminates early it auto-resets (inside
    ``ParallelVectorEnv.step``) and the terminal observation is used for
    the buffer before the new observation is carried forward.  All
    completed sub-episodes across all envs are aggregated into a single
    reported meta-episode, so one call = one logged episode regardless
    of how many early terminations occur.

    Returns ``(reward_mean, reward_std, length_mean, length_std, avg_metrics,
    reward_components_sum)`` aggregated over all completed sub-episodes.
    """
    num_envs = venv.num_envs
    states = venv.reset()

    # Per-env accumulators for the *current* sub-episode.
    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)

    # Collect stats from every completed sub-episode across all envs.
    all_rewards: list[float] = []
    all_lengths: list[int] = []

    episode_metrics = {}
    updates_count = 0
    reward_components_sum: dict[str, float] = {}

    timing = {"select_action": 0.0, "env_step": 0.0, "update": 0.0}

    max_steps = args["steps"]

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

        metrics = agent.update_batch(
            states, actions_np, rewards, terminal_next_states, dones
        )
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
            if "reward_components" in infos[i]:
                for k, v in infos[i]["reward_components"].items():
                    reward_components_sum[k] = (
                        reward_components_sum.get(k, 0.0) + v / num_envs
                    )
            ep_rewards[i] += rewards[i]
            ep_lengths[i] += 1
            if dones[i]:
                all_rewards.append(float(ep_rewards[i]))
                all_lengths.append(int(ep_lengths[i]))
                ep_rewards[i] = 0.0
                ep_lengths[i] = 0

        states = next_states

    # Collect any in-flight (not-yet-terminated) sub-episodes.
    for i in range(num_envs):
        if ep_lengths[i] > 0:
            all_rewards.append(float(ep_rewards[i]))
            all_lengths.append(int(ep_lengths[i]))

    rewards_arr = np.array(all_rewards, dtype=np.float32)
    lengths_arr = np.array(all_lengths, dtype=np.int32)

    avg_metrics = {}
    if updates_count > 0:
        avg_metrics = {k: float(v) / updates_count for k, v in episode_metrics.items()}

    if profile:
        total = timing["select_action"] + timing["env_step"] + timing["update"]
        logger.info(
            f"  Timing (vec, {num_envs} envs, {max_steps} meta-steps, {total:.1f}s total) - "
            f"select_action: {timing['select_action']:.3f}s "
            f"({timing['select_action'] / max_steps * 1000:.1f}ms/step), "
            f"env_step: {timing['env_step']:.3f}s "
            f"({timing['env_step'] / max_steps * 1000:.1f}ms/step), "
            f"update: {timing['update']:.3f}s "
            f"({timing['update'] / max_steps * 1000:.1f}ms/step)"
        )

    reward_mean = float(np.mean(rewards_arr))
    reward_std = float(np.std(rewards_arr))
    length_mean = float(np.mean(lengths_arr))
    length_std = float(np.std(lengths_arr))

    logger.info(
        f"  Meta-episode: {len(all_rewards)} sub-episodes over {num_envs} envs, "
        f"reward {reward_mean:.2f} ± {reward_std:.2f}, "
        f"length {length_mean:.1f} ± {length_std:.1f}"
    )

    return (
        reward_mean,
        reward_std,
        length_mean,
        length_std,
        avg_metrics,
        reward_components_sum,
    )


def _run_vectorized_evaluation(
    eval_venv: ParallelVectorEnv, agent: SoccerAgent, args: dict
):
    """Run evaluation episodes in parallel and return ``(mean, std)`` reward.

    Uses a single :class:`ParallelVectorEnv` with ``num_eval_episodes``
    environments.  Each env runs one full episode; if some envs finish
    earlier they auto-reset but we only keep the first completed episode
    per env.
    """
    num_eval = args["num_eval_episodes"]
    assert eval_venv.num_envs == num_eval, (
        f"eval_venv has {eval_venv.num_envs} envs, expected {num_eval}"
    )

    states = eval_venv.reset()
    ep_rewards = np.zeros(num_eval, dtype=np.float32)
    finished = [False] * num_eval
    finished_rewards = [None] * num_eval
    max_steps = args["steps"]

    for step in range(max_steps):
        actions = agent.select_actions(states, explore=False)
        actions_np = np.asarray(actions, dtype=np.float32)
        next_states, rewards, dones, _ = eval_venv.step(actions_np)

        for i in range(num_eval):
            if finished[i]:
                continue
            ep_rewards[i] += rewards[i]
            if dones[i]:
                finished[i] = True
                finished_rewards[i] = float(ep_rewards[i])

        states = next_states
        if all(finished):
            break

    for i in range(num_eval):
        if finished_rewards[i] is None:
            finished_rewards[i] = float(ep_rewards[i])

    rewards_arr = np.array(finished_rewards, dtype=np.float32)
    return float(np.mean(rewards_arr)), float(np.std(rewards_arr))


def _run_evaluation(
    eval_env: Environment, agent: SoccerAgent, args: dict, visualize: bool = False
):
    """Run ``num_eval_episodes`` evaluation episodes sequentially.

    Returns ``(mean_reward, std_reward)``.  Used when ``num_envs <= 1``
    or when visualization is requested.
    """
    eval_rewards = []
    for eval_episode in range(1, args["num_eval_episodes"] + 1):
        eval_reward, _, _, _, _ = run_episode(
            eval_env,
            agent,
            args,
            explore=False,
            visualize=visualize and (eval_episode == 1),
        )
        eval_rewards.append(eval_reward)
    eval_rewards_arr = np.array(eval_rewards, dtype=np.float32)
    return float(np.mean(eval_rewards_arr)), float(np.std(eval_rewards_arr))


def _handle_eval(
    episode: int, eval_env, eval_venv, agent, args, stats, buffer, agent_step_count
):
    """Run evaluation, log results, and save state/checkpoints.

    ``eval_venv`` is a :class:`ParallelVectorEnv` for vectorized
    evaluation, or ``None`` to use the sequential ``eval_env``.

    Returns the current agent step count.
    """
    logger.info(f"Starting evaluation at episode {episode}.")
    visualize = args["visualize"]
    use_vectorized_eval = eval_venv is not None

    if use_vectorized_eval:
        mean_eval_reward, std_eval_reward = _run_vectorized_evaluation(
            eval_venv, agent, args
        )
    else:
        mean_eval_reward, std_eval_reward = _run_evaluation(
            eval_env, agent, args, visualize=visualize
        )

    stats.log_stats_to_tb(
        episode,
        {
            "Mean_Eval_Reward": mean_eval_reward,
            "Eval_Reward_Std": std_eval_reward,
        },
    )
    logger.info(
        f"Mean evaluation reward over {args['num_eval_episodes']} episodes: "
        f"{mean_eval_reward:.2f} ± {std_eval_reward:.2f}"
    )

    # Only save lightweight learner-state checkpoints (no replay buffer)
    # during evaluation.  The full training state (including the buffer)
    # is saved once at the end of training to minimise I/O load.
    stats.save_checkpoint(agent.learner.state, "latest")
    if stats.update_best_checkpoint(mean_eval_reward, agent.learner.state):
        logger.info(
            f"New best mean eval reward: {stats.best_eval_reward:.2f} - checkpoint saved."
        )
    stats._write_training_meta(episode, agent_step_count)

    return agent_step_count


def train(args: dict, stats: StatsCollector):
    num_envs = args.get("num_envs", 1)
    use_vectorized = num_envs > 1

    if use_vectorized:
        venv = ParallelVectorEnv(
            domain_name=args["env_domain"],
            task_name=args["env_task"],
            max_steps=args["steps"],
            num_envs=num_envs,
            seed=args.get("seed", 42),
        )
        state_dim = venv.state_dim
        action_dim = venv.action_dim
    else:
        env = Environment(
            domain_name=args["env_domain"],
            task_name=args["env_task"],
            max_steps=args["steps"],
        )
        state_dim = env.state_dim
        action_dim = env.action_dim

    eval_env = None
    eval_venv = None
    num_eval = args["num_eval_episodes"]

    if use_vectorized and num_eval > 1:
        eval_venv = ParallelVectorEnv(
            domain_name=args["env_domain"],
            task_name=args["env_task"],
            max_steps=args["steps"],
            num_envs=num_eval,
            seed=args.get("seed", 42) + 10000,
        )
    else:
        eval_env = Environment(
            domain_name=args["env_domain"],
            task_name=args["env_task"],
            max_steps=args["steps"],
        )

    actor_net = ActorNetwork(action_dim)
    critic_net = CriticNetwork()

    buffer = NStepTransitionBuffer(
        state_dim,
        action_dim,
        capacity=args["capacity"],
        n_step=args.get("n_step", 5),
        gamma=args.get("gamma", 0.99),
    )
    if use_vectorized:
        buffer.set_num_envs(num_envs)

    learner_state = None
    episode = 0
    loaded_step_count = 0

    if args["resume"] and os.path.exists(args["resume"]):
        logger.info(f"Found existing state at {args['resume']}. Resuming...")
        (episode, learner_state, buffer, loaded_stats, loaded_step_count) = (
            stats.load_train_state(args["resume"])
        )

        # Restore serializable collector fields (loaded_stats is a dict:
        # {"stats": ..., "best_eval_reward": ...})
        stats.stats = loaded_stats["stats"]
        stats.best_eval_reward = loaded_stats["best_eval_reward"]

        # Ensure the loaded buffer's parallel-env config matches the
        # current run.  set_num_envs also reinitialises the per-env
        # n-step windows, which is safe on resume (pending partial
        # n-step transitions are discarded).
        if use_vectorized:
            buffer.set_num_envs(num_envs)

        logger.success(
            f"Successfully resumed from episode {episode} "
            f"(step_count={loaded_step_count})"
        )

    agent = SoccerAgent(
        observation_shape=state_dim,
        action_shape=action_dim,
        actor_net=actor_net,
        critic_net=critic_net,
        buffer=buffer,
        **args,
    )

    if learner_state is not None:
        agent.learner.state = learner_state
    if loaded_step_count > 0:
        agent._step_count = loaded_step_count
        logger.info(f"Restored agent step count to {loaded_step_count}.")

    logger.info("Setup complete.")

    # Log hyperparameters to TensorBoard HParams tab (once, before training).
    stats.log_hparams(args)

    duration_min = args.get("duration")
    use_duration = duration_min is not None
    max_episodes = args["episodes"]

    if use_vectorized:
        logger.info(
            f"Starting training loop for {max_episodes} episodes "
            f"with {num_envs} parallel envs. Visualization: {args['visualize']}"
        )
    elif use_duration:
        logger.info(
            f"Starting training loop (time-limited: {duration_min:.1f} min, max {max_episodes} episodes). "
            f"Visualization: {args['visualize']}"
        )
    else:
        logger.info(
            f"Starting training loop for {max_episodes} episodes. Visualization: {args['visualize']}"
        )

    profile = args.get("profile", False)
    train_start = time.perf_counter()
    time_limit_sec = duration_min * 60.0 if use_duration else None

    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.warning(
            f"Received signal {signum} - requesting graceful shutdown after current episode."
        )
        shutdown_requested = True

    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, _signal_handler)

    try:
        while True:
            episode += 1
            if episode > max_episodes:
                break
            if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                logger.info(
                    f"Time limit ({duration_min:.1f} min) reached. Stopping after {episode - 1} episodes."
                )
                break

            if use_vectorized:
                (
                    reward_mean,
                    reward_std,
                    length_mean,
                    length_std,
                    metrics,
                    reward_comp,
                ) = run_vectorized_episode(venv, agent, args, profile=profile)
                ep_stats = {
                    "Episode_Reward": reward_mean,
                    "Episode_Reward_Std": reward_std,
                    "Episode_Length": length_mean,
                    "Episode_Length_Std": length_std,
                    "Buffer_Size": len(buffer),
                    **metrics,
                }
            else:
                ep_reward, ep_length, metrics, _, reward_comp = run_episode(
                    env, agent, args, profile=profile
                )
                ep_stats = {
                    "Episode_Reward": ep_reward,
                    "Episode_Length": ep_length,
                    "Buffer_Size": len(buffer),
                    **metrics,
                }

            # Log individual reward components to TensorBoard and console
            for comp_name, comp_value in reward_comp.items():
                ep_stats[f"Reward_{comp_name}"] = comp_value

            stats.log_stats_to_tb(episode, ep_stats)
            total_label = (
                f"{duration_min:.1f}min" if use_duration else str(max_episodes)
            )
            stats.log_progress(
                episode,
                total_label,
                ep_stats,
                {"Loss": metrics.get("loss_critic", 0.0)},
            )

            if episode % args["eval_frequency"] == 0:
                _handle_eval(
                    episode,
                    eval_env,
                    eval_venv,
                    agent,
                    args,
                    stats,
                    buffer,
                    agent._step_count,
                )

            if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                logger.info(
                    f"Time limit ({duration_min:.1f} min) reached. Stopping after {episode} episodes."
                )
                break
            if shutdown_requested:
                break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

        # Save full training state (including replay buffer) once at the
        # end.  This is in the finally block so it also runs on crash /
        # signal / exception, ensuring the progress is preserved for
        # resume.
        try:
            stats.save_train_state(
                episode,
                agent.learner.state,
                buffer,
                stats,
                agent_step_count=agent._step_count,
            )
            stats.flush_stats_to_disk()
            stats.save_checkpoint(agent.learner.state, "final")
            logger.info(f"Dumped training statistics to {stats.stats_file}.")
        except Exception:
            logger.exception("Failed to save final training state.")

        if use_vectorized:
            venv.close()
        if eval_venv is not None:
            eval_venv.close()

    logger.success("Training completed successfully!")
