import os
import numpy as np
import jax
import imageio
from loguru import logger
from src.collector import StatsCollector
from src.environment import Environment
from src.learner import MPOLearner
from src.agent import SoccerAgent
from src.buffer import ReplayBuffer
from src.networks import ActorNetwork, CriticNetwork
from src.train import run_episode


def save_video(frames: list, path: str, fps: int = 30):
    try:
        imageio.mimwrite(path, frames, fps=fps)
        return path
    except Exception as e:
        gif_path = os.path.splitext(path)[0] + ".gif"
        logger.warning(f"Could not write mp4: {e}. Falling back to '{gif_path}'.")
        try:
            imageio.mimsave(gif_path, frames, fps=fps)
            return gif_path
        except Exception as e2:
            logger.warning(f"Could not write GIF recording, skipping video export: {e2}")
            return None


def test(args: dict, stats: StatsCollector):
    if not args["load_dir"]:
        logger.error("Test mode requires --load_dir to be set to some previous run's directoriy.")

    checkpoint_path = os.path.join(args["load_dir"], "checkpoints", f"{args['checkpoint']}.pkl")
    if not os.path.isfile(checkpoint_path):
        logger.error(f"No checkpoint found at '{checkpoint_path}'.")
        return

    env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])

    actor_net = ActorNetwork(env.action_dim)
    critic_net = CriticNetwork()

    buffer = ReplayBuffer(
        env.state_dim, 
        env.action_dim, 
        capacity=args["capacity"]
    )

    agent = SoccerAgent(
        observation_shape=env.state_dim,
        action_shape=env.action_dim,
        actor_net=actor_net,
        critic_net=critic_net,
        buffer=buffer,
        **args
    )

    logger.info(f"Loading checkpoint '{args['checkpoint']}' from {checkpoint_path}")
    agent.learner.state = StatsCollector.load_checkpoint_file(checkpoint_path)

    logger.info(
        f"Running {args['num_eval_episodes']} test episode(s) on "
        f"{args['env_domain']}/{args['env_task']}."
    )

    visualize = args["visualize"]

    episode_rewards = []
    frames = [] if visualize else None
    for episode in range(1, args["num_eval_episodes"] + 1):
        ep_reward, ep_length, _, ep_frames = run_episode(
            env, agent, args, explore=False, visualize=visualize
        )
        episode_rewards.append(ep_reward)
        logger.info(
            f"Test episode {episode}/{args['num_eval_episodes']} | "
            f"Reward: {ep_reward:.2f} | Length: {ep_length}"
        )
        stats.log_stats_to_tb(episode, {"Test_Episode_Reward": ep_reward})

        if visualize:
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
    }
    stats.flush_stats_to_disk()

    logger.success(
        f"Testing completed. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} "
        f"over {len(episode_rewards)} episode(s)."
    )
    return mean_reward
