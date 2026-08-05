import os
import numpy as np
from loguru import logger

from src.agent import MPOAgent
from src.buffer import NStepTransitionBuffer
from src.collector import StatsCollector
from src.environment import Environment
from src.networks import ActorNetwork, CriticNetwork
from src.runner import run_episode, run_episode_with_respawn
from src.serve import RemoteCheckpointReloader
from src.viewer import run_live, save_video


def test(args: dict, stats: StatsCollector):
    use_stream = bool(args.get("stream"))

    if use_stream:
        # Stream mode: initial checkpoint comes from the remote server,
        # no local --load_dir needed.
        if not args.get("live", False):
            logger.error("--stream requires --live mode.")
            return
        checkpoint_path = None
    else:
        if not args["load_dir"]:
            logger.error(
                "Test mode requires --load_dir to be set to some previous run's directory."
            )
            return

        checkpoint_path = os.path.join(
            args["load_dir"], "checkpoints", f"{args['checkpoint']}.pkl"
        )
        if not os.path.isfile(checkpoint_path):
            logger.error(f"No checkpoint found at '{checkpoint_path}'.")
            return

    env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])

    actor_net = ActorNetwork(env.action_dim)
    critic_net = CriticNetwork()

    buffer = NStepTransitionBuffer(
        env.state_dim,
        env.action_dim,
        capacity=args["capacity"],
        n_step=args.get("n_step", 5),
        gamma=args.get("gamma", 0.99),
    )

    agent = MPOAgent(
        observation_shape=env.state_dim,
        action_shape=env.action_dim,
        actor_net=actor_net,
        critic_net=critic_net,
        buffer=buffer,
        **args
    )

    if use_stream:
        # Fetch initial checkpoint from the remote server (blocks once).
        stream_url = RemoteCheckpointReloader.normalize_stream_url(args["stream"])
        reloader = RemoteCheckpointReloader(
            stream_url, agent,
            poll_interval=args.get("checkpoint_poll_interval", 0.0) or 5.0,
        )
        initial_state = reloader.fetch_initial()
        if initial_state is None:
            logger.error("Could not fetch initial checkpoint from remote server. Aborting.")
            return
        agent.learner.state = initial_state
        logger.info("Loaded initial checkpoint from remote server.")
    else:
        reloader = None
        logger.info(f"Loading checkpoint '{args['checkpoint']}' from {checkpoint_path}")
        agent.learner.state = StatsCollector.load_checkpoint_file(checkpoint_path)

    use_respawn = args.get("respawn", False)

    if args.get("live", False):
        return run_live(
            env,
            agent,
            respawn=use_respawn,
            checkpoint_path=checkpoint_path,
            poll_interval=args.get("checkpoint_poll_interval", 0.0),
            stream_url=args.get("stream"),
            reloader=reloader,
        )

    logger.info(
        f"Running {args['num_eval_episodes']} test episode(s) on "
        f"{args['env_domain']}/{args['env_task']}."
        + (" (respawn mode)" if use_respawn else "")
    )

    visualize = args["visualize"]

    episode_rewards = []
    frames = [] if visualize else None
    for episode in range(1, args["num_eval_episodes"] + 1):
        if use_respawn:
            sub_rewards, sub_lengths, ep_frames = run_episode_with_respawn(
                env, agent, args, visualize=visualize
            )
            episode_rewards.extend(sub_rewards)
            logger.info(
                f"Test episode {episode}/{args['num_eval_episodes']} | "
                f"{len(sub_rewards)} sub-episode(s), "
                f"rewards: {[f'{r:.2f}' for r in sub_rewards]}, "
                f"lengths: {sub_lengths}"
            )
            for sub_idx, sub_r in enumerate(sub_rewards):
                stats.log_stats_to_tb(
                    episode * 1000 + sub_idx,
                    {"Test_SubEpisode_Reward": sub_r},
                )
        else:
            ep_reward, ep_length, _, ep_frames, _ = run_episode(
                env, agent, args, explore=False, visualize=visualize
            )
            episode_rewards.append(ep_reward)
            logger.info(
                f"Test episode {episode}/{args['num_eval_episodes']} | "
                f"Reward: {ep_reward:.2f} | Length: {ep_length}"
            )
            stats.log_stats_to_tb(episode, {"Test_Episode_Reward": ep_reward})

        if visualize and ep_frames:
            frames.extend(ep_frames)

    if frames:
        video_path = os.path.join(args["outdir"], f"{args['checkpoint']}.mp4")
        saved_path = save_video(frames, video_path, fps=100)
        if saved_path:
            logger.success(f"Saved test visualization video to {saved_path}")

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    stats.stats["summary"] = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "num_episodes": len(episode_rewards),
        "checkpoint": checkpoint_path,
        "respawn": use_respawn,
    }
    stats.flush_stats_to_disk()

    logger.success(
        f"Testing completed. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} "
        f"over {len(episode_rewards)} episode(s)."
    )
    return mean_reward
