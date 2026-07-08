import json
import os
from copy import deepcopy
import numpy as np
from loguru import logger
from src.utils import setup_tensorboard, log_stats_to_tb
from src.buffer import ReplayBuffer
from src.environment import Env
from src.learner import MPOLearner
from src.networks import ActorNetwork, CriticNetwork


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
    batch_size = args["batch_size"]
    eval_interval = args["val_frequency"]
    num_eval_episodes = args["num_val_episodes"]
    run_dir = args["run_dir"]

    # Initialize env
    env = Env(domain_name="cartpole", task_name="balance")
    eval_env = deepcopy(env)

    # Initialize MPO learner components
    actor_net = ActorNetwork(env.action_dim)
    critic_net = CriticNetwork()

    learner = MPOLearner(actor_net, critic_net, env.state_dim, env.action_dim)
    buffer = ReplayBuffer(env.state_dim, env.action_dim)

    tb = setup_tensorboard(args["run_dir"])

    logger.info(f"Starting training loop for {max_episodes} episodes. Visualization: {visualize}")

    # TODO: make visualization toggleable

    stats = {}

    for episode in range(1, max_episodes + 1):
        ep_reward, ep_length = run_episode(env)
        stats[episode] = {
            "Episode_Reward": ep_reward,
            "Episode_Length": ep_length,
            "Buffer_Size": len(buffer)
        }
        ep_stats = stats[episode]

        log_stats_to_tb(tb, episode, ep_stats)

        if episode % 10 == 0:
            logger.info(
                f"Episode {episode}/{max_episodes} | Reward: {ep_stats['Episode_Reward']} | Buffer Size: {ep_stats['Buffer_Length']}")

        if episode in [1, 2, 3, 4, 5] or episode % eval_interval == 0:
            logger.info(f"Starting evaluation at episode {episode}.")
            eval_rewards = []

            for eval_episode in num_eval_episodes:
                eval_reward, _ = run_episode(
                    eval_env,
                    batch_size=batch_size,
                    explore=False,
                    visualize=visualize and (eval_episode == 0)  # only vis. first eval episode
                )
                eval_rewards.append(eval_reward)

            mean_eval_reward = np.mean(eval_rewards)
            stats[episode]["Mean_Eval_Reward"] = mean_eval_reward
            logger.info(f"Mean evaluation reward over {num_eval_episodes} episodes: {mean_eval_reward:.2f}")
            log_stats_to_tb(tb, episode, {"Mean_Eval_Reward": mean_eval_reward})

    stats_file = os.path.join(run_dir, "training_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=4)
    logger.success("Training completed successfully!")
