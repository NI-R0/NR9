import os
import numpy as np
import jax
from loguru import logger
from src.collector import StatsCollector
from src.environment import Environment
from src.learner import MPOLearner
from src.agent import SoccerAgent
from src.buffer import ReplayBuffer
from src.networks import ActorNetwork, CriticNetwork
from src.train import run_episode


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
    args["random_key"] = jax.random.PRNGKey(args["seed"])

    learner = MPOLearner(actor_net, critic_net, env.state_dim, env.action_dim, **args)
    buffer = ReplayBuffer(env.state_dim, env.action_dim, capacity=1)  # unused: testing never trains
    agent = SoccerAgent(
        learner, buffer, args["warmup"], args["batch_size"], args["random_key"], use_ema=args["ema"]
    )

    logger.info(f"Loading checkpoint '{args['checkpoint']}' from {checkpoint_path}")
    learner.state = StatsCollector.load_checkpoint_file(checkpoint_path)

    logger.info(
        f"Running {args['num_eval_episodes']} test episode(s) on "
        f"{args['env_domain']}/{args['env_task']}."
        + (" Using EMA actor." if args["ema"] else "")
    )

    episode_rewards = []
    for episode in range(1, args["num_eval_episodes"] + 1):
        ep_reward, ep_length, _ = run_episode(
            env, agent, args, explore=False, visualize=args["visualize"]
        )
        episode_rewards.append(ep_reward)
        logger.info(
            f"Test episode {episode}/{args['num_eval_episodes']} | "
            f"Reward: {ep_reward:.2f} | Length: {ep_length}"
        )
        stats.log_stats_to_tb(episode, {"Test_Episode_Reward": ep_reward})

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
