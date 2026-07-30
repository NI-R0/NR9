import os
import signal
import time
import numpy as np
from loguru import logger
from src.collector import StatsCollector
from src.buffer import NStepTransitionBuffer
from src.environment import Environment
from src.agent import MPOAgent
from src.networks import ActorNetwork, CriticNetwork
from src.vector_env import ParallelVectorEnv
from src.runner import run_episode, run_vectorized_episode
from src.environments.walker_3D_ball import PHASE_STAND, PHASE_APPROACH, PHASE_FULL


def _check_phase_advancement(current_phase: int, mean_eval_reward: float,
                             phase1_threshold: float, phase2_threshold: float) -> int:
    """Check if the curriculum phase should advance based on eval reward.

    Returns the new phase (same as current if no advancement).
    """
    if current_phase == PHASE_STAND and mean_eval_reward >= phase1_threshold:
        logger.info(
            f"Curriculum: advancing from STAND to APPROACH "
            f"(eval reward {mean_eval_reward:.2f} >= threshold {phase1_threshold:.2f})"
        )
        return PHASE_APPROACH
    elif current_phase == PHASE_APPROACH and mean_eval_reward >= phase2_threshold:
        logger.info(
            f"Curriculum: advancing from APPROACH to FULL "
            f"(eval reward {mean_eval_reward:.2f} >= threshold {phase2_threshold:.2f})"
        )
        return PHASE_FULL
    return current_phase


def _propagate_phase(phase: int, train_env, eval_env):
    """Send phase update to the train environment and the eval environment."""
    train_env.set_phase(phase)
    eval_env.set_phase(phase)
    logger.info(f"Curriculum phase set to {phase} for all environments.")


def _run_eval(episode: int, eval_env: Environment, agent: MPOAgent,
              args: dict, stats: StatsCollector, buffer: NStepTransitionBuffer) -> float:
    """Run evaluation episodes, checkpoint, and log.  Returns mean eval reward."""
    logger.info(f"Starting evaluation at episode {episode}.")
    eval_rewards = []
    for eval_ep in range(1, args["num_eval_episodes"] + 1):
        eval_reward, _, _, _ = run_episode(
            eval_env, agent,
            explore=False,
            visualize=args["visualize"] and (eval_ep == 1),
        )
        eval_rewards.append(eval_reward)

    mean_eval_reward = float(np.mean(eval_rewards))
    stats.log_stats_to_tb(episode, {"Mean_Eval_Reward": mean_eval_reward})
    logger.info(f"Mean evaluation reward over {args['num_eval_episodes']} episodes: {mean_eval_reward:.2f}")

    stats.save_train_state(episode, agent.learner.state, buffer, stats)
    stats.flush_stats_to_disk()
    stats.save_checkpoint(agent.learner.state, "latest")
    if stats.update_best_checkpoint(mean_eval_reward, agent.learner.state):
        logger.info(f"New best mean eval reward: {stats.best_eval_reward:.2f} - checkpoint saved.")

    return mean_eval_reward


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

    agent = MPOAgent(
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
        current_phase = PHASE_STAND
        phase1_threshold = args.get("phase1_threshold", 5.0)
        phase2_threshold = args.get("phase2_threshold", 15.0)
        logger.info(f"Curriculum enabled: starting at phase {current_phase} "
                    f"(thresholds: phase1={phase1_threshold}, phase2={phase2_threshold})")
        _propagate_phase(current_phase, venv if use_vectorized else env, eval_env)
    else:
        current_phase = PHASE_FULL
        phase1_threshold = 0.0
        phase2_threshold = 0.0

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
                finished_stats, metrics = run_vectorized_episode(venv, agent, args["steps"], profile=profile)
                for ep_reward, ep_length in finished_stats:
                    episode += 1
                    if episode > max_episodes:
                        break
                    ep_stats = {
                        "Episode_Reward": ep_reward,
                        "Episode_Length": ep_length,
                        "Buffer_Size": len(buffer),
                        **metrics,
                    }
                    stats.log_stats_to_tb(episode, ep_stats)
                    total_label = f"{duration_min:.1f}min" if use_duration else str(max_episodes)
                    stats.log_progress(episode, total_label, ep_stats, {"Loss": metrics.get("loss_critic", 0.0)})

                    if episode % args["eval_frequency"] == 0:
                        mean_eval_reward = _run_eval(episode, eval_env, agent, args, stats, buffer)
                        if use_curriculum:
                            new_phase = _check_phase_advancement(
                                current_phase, mean_eval_reward,
                                phase1_threshold, phase2_threshold)
                            if new_phase != current_phase:
                                current_phase = new_phase
                                _propagate_phase(current_phase, venv if use_vectorized else env, eval_env)

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
                ep_reward, ep_length, metrics, _ = run_episode(env, agent, profile=profile)
                ep_stats = {
                    "Episode_Reward": ep_reward,
                    "Episode_Length": ep_length,
                    "Buffer_Size": len(buffer),
                    **metrics
                }

                stats.log_stats_to_tb(episode, ep_stats)

                total_label = f"{duration_min:.1f}min" if use_duration else str(max_episodes)
                stats.log_progress(episode, total_label, ep_stats, {"Loss": metrics.get("loss_critic", 0.0)})

                if episode % args["eval_frequency"] == 0:
                    mean_eval_reward = _run_eval(episode, eval_env, agent, args, stats, buffer)
                    if use_curriculum:
                        new_phase = _check_phase_advancement(
                            current_phase, mean_eval_reward,
                            phase1_threshold, phase2_threshold)
                        if new_phase != current_phase:
                            current_phase = new_phase
                            _propagate_phase(current_phase, venv if use_vectorized else env, eval_env)

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
