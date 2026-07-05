import json
import numpy as np
from loguru import logger
from src.utils import setup_tensorboard, log_stats_to_tb
from src.buffer import ReplayBuffer
from src.environment import Env


def run_episode(env: Env, buffer: ReplayBuffer):
    state = env.reset()
    episode_reward = 0
    done = False
    step = 0

    while not done and step < env.ep_max_steps:
        action = np.random.randn(env.action_dim)
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)

        # TODO: agent.train

        state = next_state
        episode_reward = reward
        step += 1

    return episode_reward, step


def train(args: dict):
    max_episodes = args["episodes"]
    visualize = args["visualize"]

    # TODO: Initialize environment, networks, ...
    env = Env()
    buffer = ReplayBuffer(env.state_dim, env.action_dim, capacity=1000000)

    tb = setup_tensorboard()

    logger.info(f"Starting training loop for {max_episodes} episodes. Visualization: {visualize}")

    # TODO: make visualization toggleable

    stats = {}

    for episode in range(1, max_episodes + 1):
        ep_stats = run_episode(env)
        stats[episode] = {
            "Episode_Reward": ep_stats[0],
            "Episode_Length": ep_stats[1],
            "Buffer_Size": len(buffer)
        }
        ep_stats = stats[episode]

        log_stats_to_tb(tb, episode, ep_stats)

        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{max_episodes} | Reward: {ep_stats["Episode_Reward"]} | Buffer Size: {ep_stats["Buffer_Length"]}")

    # TODO: dump stats dict
    logger.success("Training completed successfully!")
