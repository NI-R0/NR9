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


def _run_eval(episode: int, eval_env: Environment, eval_venv, agent: MPOAgent,
              args: dict, stats: StatsCollector) -> float:
    """Run evaluation episodes, checkpoint, and log.  Returns mean eval reward."""
    logger.info(f"Starting evaluation at episode {episode}.")
    num_eval = args["num_eval_episodes"]

    if eval_venv is not None:
        assert eval_venv.num_envs == num_eval, (
            f"eval_venv has {eval_venv.num_envs} envs, expected {num_eval}"
        )
        states = eval_venv.reset()
        ep_rewards_arr = np.zeros(eval_venv.num_envs, dtype=np.float32)
        finished = [False] * eval_venv.num_envs
        finished_rewards: list[float | None] = [None] * eval_venv.num_envs
        for _ in range(args["steps"]):
            actions = agent.select_actions(states, explore=False)
            actions_np = np.asarray(actions, dtype=np.float32)
            next_states, rewards, dones, _ = eval_venv.step(actions_np)
            for i in range(eval_venv.num_envs):
                if finished[i]:
                    continue
                ep_rewards_arr[i] += rewards[i]
                if dones[i]:
                    finished[i] = True
                    finished_rewards[i] = float(ep_rewards_arr[i])
            states = next_states
            if all(finished):
                break
        eval_rewards = [
            r if r is not None else float(ep_rewards_arr[i])
            for i, r in enumerate(finished_rewards)
        ]
    else:
        eval_rewards = []
        for eval_ep in range(1, num_eval + 1):
            eval_reward, _, _, _, _ = run_episode(
                eval_env, agent,
                explore=False,
                visualize=args["visualize"] and (eval_ep == 1),
            )
            eval_rewards.append(eval_reward)

    mean_eval_reward = float(np.mean(eval_rewards))
    std_eval_reward = float(np.std(eval_rewards))
    stats.log_stats_to_tb(episode, {"Mean_Eval_Reward": mean_eval_reward, "Eval_Reward_Std": std_eval_reward})
    logger.info(f"Mean evaluation reward over {num_eval} episodes: {mean_eval_reward:.2f} ± {std_eval_reward:.2f}")

    stats.save_checkpoint(agent.learner.state, "latest")
    if stats.update_best_checkpoint(mean_eval_reward, agent.learner.state):
        logger.info(f"New best mean eval reward: {stats.best_eval_reward:.2f} - checkpoint saved.")
    stats._write_training_meta(episode, agent._step_count)

    return mean_eval_reward


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
        env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])
        state_dim = env.state_dim
        action_dim = env.action_dim

    num_eval = args["num_eval_episodes"]
    eval_env = None
    eval_venv = None
    if use_vectorized and num_eval > 1:
        eval_venv = ParallelVectorEnv(
            domain_name=args["env_domain"],
            task_name=args["env_task"],
            max_steps=args["steps"],
            num_envs=num_eval,
            seed=args.get("seed", 42) + 10000,
        )
    else:
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
    loaded_step_count = 0

    if args["resume"] and os.path.exists(args["resume"]):
        logger.info(f"Found existing state at {args['resume']}. Resuming...")
        episode, learner_state, buffer, loaded_stats, loaded_step_count = stats.load_train_state(args["resume"])

        # Restore serializable collector fields (loaded_stats is a dict:
        # {"stats": ..., "best_eval_reward": ...})
        stats.stats = loaded_stats["stats"]
        stats.best_eval_reward = loaded_stats["best_eval_reward"]
        # Ensure loaded buffer's parallel-env config matches the current run.
        if use_vectorized:
            buffer.set_num_envs(num_envs)
        logger.success(f"Successfully resumed from episode {episode} (step_count={loaded_step_count})")

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
    if loaded_step_count > 0:
        agent._step_count = loaded_step_count
        logger.info(f"Restored agent step count to {loaded_step_count}.")

    logger.info("Setup complete.")

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

    success = False
    try:
        while True:
            episode += 1
            if episode > max_episodes:
                break
            if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                logger.info(f"Time limit ({duration_min:.1f} min) reached. Stopping after {episode - 1} episodes.")
                break

            if use_vectorized:
                finished_stats, metrics, reward_comp = run_vectorized_episode(
                    venv, agent, args["steps"], profile=profile)
                rewards = [r for r, _ in finished_stats]
                lengths = [l for _, l in finished_stats]
                ep_stats = {
                    "Episode_Reward": float(np.mean(rewards)),
                    "Episode_Reward_Std": float(np.std(rewards)),
                    "Episode_Length": float(np.mean(lengths)),
                    "Episode_Length_Std": float(np.std(lengths)),
                    "Buffer_Size": len(buffer),
                    **metrics,
                    **{f"Reward_{k}": v for k, v in reward_comp.items()},
                }
            else:
                ep_reward, ep_length, metrics, _, reward_comp = run_episode(env, agent, profile=profile)
                ep_stats = {
                    "Episode_Reward": ep_reward,
                    "Episode_Length": ep_length,
                    "Buffer_Size": len(buffer),
                    **metrics,
                    **{f"Reward_{k}": v for k, v in reward_comp.items()},
                }

            stats.log_stats_to_tb(episode, ep_stats)
            total_label = f"{duration_min:.1f}min" if use_duration else str(max_episodes)
            stats.log_progress(episode, total_label, ep_stats, {"Loss": metrics.get("loss_critic", 0.0)})

            if episode % args["eval_frequency"] == 0:
                _run_eval(episode, eval_env, eval_venv, agent, args, stats)

            if use_duration and (time.perf_counter() - train_start) >= time_limit_sec:
                logger.info(f"Time limit ({duration_min:.1f} min) reached. Stopping after {episode} episodes.")
                break
            if shutdown_requested:
                break
        success = True
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

        if success:
            try:
                stats.save_train_state(episode, agent.learner.state, buffer, stats,
                                       agent_step_count=agent._step_count)
                stats.flush_stats_to_disk()
                stats.save_checkpoint(agent.learner.state, "final")
                logger.info(f"Dumped training statistics to {stats.stats_file}.")
            except Exception:
                logger.exception("Failed to save final training state.")
        else:
            logger.error("Training loop exited via exception!")

        if use_vectorized:
            venv.close()
        if eval_venv is not None:
            eval_venv.close()

    logger.success("Training completed successfully!")
