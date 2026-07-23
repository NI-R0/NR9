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

# Curriculum phases (must match src/environments/walker_3D_ball.py)
_PHASE_FEET = 0
_PHASE_STAND = 1
_PHASE_APPROACH = 2
_PHASE_FULL = 3


def _check_phase_advancement(current_phase: int, mean_eval_reward: float,
                             phase1_threshold: float, phase2_threshold: float,
                             phase3_threshold: float) -> int:
    """Check if the curriculum phase should advance based on eval reward.

    Returns the new phase (same as current if no advancement).
    """
    if current_phase == _PHASE_FEET and mean_eval_reward >= phase1_threshold:
        logger.info(
            f"Curriculum: advancing from FEET to STAND "
            f"(eval reward {mean_eval_reward:.2f} >= threshold {phase1_threshold:.2f})"
        )
        return _PHASE_STAND
    elif current_phase == _PHASE_STAND and mean_eval_reward >= phase2_threshold:
        logger.info(
            f"Curriculum: advancing from STAND to APPROACH "
            f"(eval reward {mean_eval_reward:.2f} >= threshold {phase2_threshold:.2f})"
        )
        return _PHASE_APPROACH
    elif current_phase == _PHASE_APPROACH and mean_eval_reward >= phase3_threshold:
        logger.info(
            f"Curriculum: advancing from APPROACH to FULL "
            f"(eval reward {mean_eval_reward:.2f} >= threshold {phase3_threshold:.2f})"
        )
        return _PHASE_FULL
    return current_phase


def _propagate_phase(phase: int, use_vectorized: bool, venv, env, eval_env):
    """Send phase update to all environments (train + eval)."""
    if use_vectorized:
        venv.set_phase(phase)
    else:
        env.set_phase(phase)
    eval_env.set_phase(phase)
    logger.info(f"Curriculum phase set to {phase} for all environments.")

def run_episode(env: Environment, agent: SoccerAgent, args: dict, explore: bool = True,
                visualize: bool = False, profile: bool = False):
    state = env.reset()
    episode_reward = 0.0
    done = False
    step = 0

    episode_metrics = {}
    avg_metrics = {}
    updates_count = 0

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


def run_vectorized_episode(venv: ParallelVectorEnv, agent: SoccerAgent, args: dict,
                           profile: bool = False):
    """Run one ``meta-episode'' across ``num_envs`` parallel environments.

    All envs step simultaneously until every env has completed at least
    one episode.  When an env finishes it auto-resets (inside
    ``ParallelVectorEnv.step``) and the terminal observation is used for
    the buffer before the new observation is carried forward.

    Returns a list of (reward, length) tuples - one per env, in order.
    """
    num_envs = venv.num_envs
    states = venv.reset()

    ep_rewards = np.zeros(num_envs, dtype=np.float32)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    finished = [False] * num_envs
    finished_stats: list[tuple[float, int]] = [None] * num_envs

    episode_metrics = {}
    updates_count = 0

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

    avg_metrics = {}
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


def train(args: dict, stats: StatsCollector):
    num_envs = args.get("num_envs", 1)
    use_vectorized = num_envs > 1
    is_resume = args.get("resume") is not None and os.path.exists(args.get("resume", ""))

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
        env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])
        state_dim = env.state_dim
        action_dim = env.action_dim

    eval_env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])

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

    if args["resume"] and os.path.exists(args["resume"]):
        logger.info(f"Found existing state at {args['resume']}. Resuming...")
        episode, learner_state, buffer, loaded_stats = stats.load_train_state(args["resume"])

        # Restore serializable collector fields (loaded_stats is a dict:
        # {"stats": ..., "best_eval_reward": ...})
        stats.stats = loaded_stats["stats"]
        stats.best_eval_reward = loaded_stats["best_eval_reward"]
        logger.success(f"Successfully resumed from episode {episode}")

    agent = SoccerAgent(
        observation_shape=state_dim,
        action_shape=action_dim,
        actor_net=actor_net,
        critic_net=critic_net,
        buffer=buffer,
        **args
    )

    if learner_state is not None:
        agent.learner.state = learner_state

    logger.info("Setup complete.")

    # Curriculum phase initialization
    use_curriculum = args.get("curriculum", False)
    if use_curriculum:
        current_phase = _PHASE_FEET
        phase1_threshold = args.get("phase1_threshold", 200.0)
        phase2_threshold = args.get("phase2_threshold", 400.0)
        phase3_threshold = args.get("phase3_threshold", 700.0)
        logger.info(f"Curriculum enabled: starting at phase {current_phase} "
                     f"(thresholds: phase1={phase1_threshold}, phase2={phase2_threshold}, phase3={phase3_threshold})")
        _propagate_phase(current_phase, use_vectorized,
                         venv if use_vectorized else None,
                         env if not use_vectorized else None,
                         eval_env)
    else:
        current_phase = _PHASE_FULL
        phase1_threshold = 0.0
        phase2_threshold = 0.0
        phase3_threshold = 0.0

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
        logger.info(f"Starting training loop for {max_episodes} episodes. Visualization: {args['visualize']}")

    if not is_resume:
        dummy_stats = {
            "Episode_Reward": 0,
            "Episode_Length": args["steps"],
            "Buffer_Size": len(buffer),
            "Episode_Loss": np.nan,
        }
        stats.log_stats_to_tb(0, dummy_stats)

    # Log hyperparameters to TensorBoard HParams tab (once, at start)
    stats.log_hparams(args)

    profile = args.get("profile", False)
    train_start = time.perf_counter()
    time_limit_sec = duration_min * 60.0 if use_duration else None

    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.warning(f"Received signal {signum} - requesting graceful shutdown after current episode.")
        shutdown_requested = True

    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, _signal_handler)

    try:
        while True:
            if use_vectorized:
                finished_stats, metrics = run_vectorized_episode(venv, agent, args, profile=profile)
                for ep_reward, ep_length in finished_stats:
                    episode += 1
                    if episode > max_episodes:
                        break
                    ep_stats = {
                        "Episode_Reward": ep_reward,
                        "Episode_Length": ep_length,
                        "Buffer_Size": len(buffer),
                        "Curriculum_Phase": current_phase,
                        **metrics,
                    }
                    stats.log_stats_to_tb(episode, ep_stats)
                    total_label = f"{duration_min:.1f}min" if use_duration else str(max_episodes)
                    stats.log_progress(episode, total_label, ep_stats, {"Loss": metrics.get("loss_critic", 0.0)})

                    if episode % args["eval_frequency"] == 0:
                        logger.info(f"Starting evaluation at episode {episode}.")
                        eval_rewards = []
                        for eval_episode in range(1, args["num_eval_episodes"] + 1):
                            eval_reward, _, _, _ = run_episode(
                                eval_env, agent, args, explore=False,
                                visualize=args["visualize"] and (eval_episode == 1),
                            )
                            eval_rewards.append(eval_reward)
                        mean_eval_reward = np.mean(eval_rewards)
                        stats.log_stats_to_tb(episode, {"Mean_Eval_Reward": mean_eval_reward})
                        logger.info(
                            f"Mean evaluation reward over {args['num_eval_episodes']} episodes: {mean_eval_reward:.2f}")
                        stats.save_train_state(episode, agent.learner.state, buffer, stats)
                        stats.flush_stats_to_disk()
                        stats.save_checkpoint(agent.learner.state, "latest")
                        if stats.update_best_checkpoint(mean_eval_reward, agent.learner.state):
                            logger.info(f"New best mean eval reward: {stats.best_eval_reward:.2f} - checkpoint saved.")

                        if use_curriculum:
                            new_phase = _check_phase_advancement(
                                current_phase, mean_eval_reward,
                                phase1_threshold, phase2_threshold,
                                phase3_threshold)
                            if new_phase != current_phase:
                                current_phase = new_phase
                                _propagate_phase(current_phase, use_vectorized,
                                                 venv if use_vectorized else None,
                                                 env if not use_vectorized else None,
                                                 eval_env)

                    if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                        logger.info(f"Time limit ({duration_min:.1f} min) reached. Stopping after {episode} episodes.")
                        break

                if episode > max_episodes:
                    break
                if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                    break
                if shutdown_requested:
                    break
            else:
                episode += 1
                if episode > max_episodes:
                    break
                if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                    logger.info(f"Time limit ({duration_min:.1f} min) reached. Stopping after {episode - 1} episodes.")
                    break
                ep_reward, ep_length, metrics, _ = run_episode(env, agent, args, profile=profile)
                ep_stats = {
                    "Episode_Reward": ep_reward,
                    "Episode_Length": ep_length,
                    "Buffer_Size": len(buffer),
                    "Curriculum_Phase": current_phase,
                    **metrics
                }

                stats.log_stats_to_tb(episode, ep_stats)

                total_label = f"{duration_min:.1f}min" if use_duration else str(max_episodes)
                stats.log_progress(episode, total_label, ep_stats, {"Loss": metrics.get("loss_critic", 0.0)})

                if episode % args["eval_frequency"] == 0:
                    logger.info(f"Starting evaluation at episode {episode}.")
                    eval_rewards = []

                    for eval_episode in range(1, args["num_eval_episodes"] + 1):
                        eval_reward, _, _, _ = run_episode(
                            eval_env,
                            agent,
                            args,
                            explore=False,
                            visualize=args["visualize"] and (eval_episode == 1)  # only vis. first eval episode
                        )
                        eval_rewards.append(eval_reward)

                    mean_eval_reward = np.mean(eval_rewards)
                    stats.log_stats_to_tb(episode, {"Mean_Eval_Reward": mean_eval_reward})
                    logger.info(
                        f"Mean evaluation reward over {args['num_eval_episodes']} episodes: {mean_eval_reward:.2f}")

                    stats.save_train_state(episode, agent.learner.state, buffer, stats)
                    stats.flush_stats_to_disk()
                    stats.save_checkpoint(agent.learner.state, "latest")
                    if stats.update_best_checkpoint(mean_eval_reward, agent.learner.state):
                        logger.info(f"New best mean eval reward: {stats.best_eval_reward:.2f} - checkpoint saved.")

                    if use_curriculum:
                        new_phase = _check_phase_advancement(
                            current_phase, mean_eval_reward,
                            phase1_threshold, phase2_threshold,
                            phase3_threshold)
                        if new_phase != current_phase:
                            current_phase = new_phase
                            _propagate_phase(current_phase, use_vectorized,
                                             venv if use_vectorized else None,
                                             env if not use_vectorized else None,
                                             eval_env)

                if shutdown_requested:
                    break
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

        if use_vectorized:
            venv.close()

    stats.save_train_state(episode, agent.learner.state, buffer, stats)
    stats.flush_stats_to_disk()
    stats.save_checkpoint(agent.learner.state, "final")
    logger.info(f"Dumped training statistics to {stats.stats_file}.")
    logger.success("Training completed successfully!")
