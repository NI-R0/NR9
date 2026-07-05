import numpy as np
from loguru import logger
from src.buffer import ReplayBuffer


def run_episode():
    pass


def train(args: dict):
    # TODO: Initialize environment, networks, ...
    state_dim = None
    action_dim = None
    max_episodes = args["episodes"]
    visualize = args["visualize"]

    buffer = ReplayBuffer(state_dim, action_dim, capacity=1000000)

    logger.info(f"Starting training for {max_episodes} episodes. Visualization: {visualize}")

    for episode in range(1, max_episodes + 1):
        ep_stats = run_episode()

        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{max_episodes} | Reward: {ep_stats["reward"]} | Buffer Size: {len(buffer)}")

    logger.success("Training completed successfully!")
