import numpy as np
from loguru import logger
from src.collector import StatsCollector
from src.buffer import ReplayBuffer
from src.environment import Environment
from src.agent import SoccerAgent
from src.networks import ActorNetwork, CriticNetwork

def run_episode(env: Environment, agent: SoccerAgent, args: dict, explore: bool = True,
                visualize: bool = False):
    state = env.reset()
    episode_reward = 0.0
    done = False
    step = 0

    episode_metrics = {}
    updates_count = 0

    frames = [] if visualize else None
    while not done and step < env.ep_max_steps:
        if visualize:
            frame = env.render()
            frames.append(frame)

        action = agent.select_action(state, explore=explore)
        next_state, reward, done, _ = env.step(action)

        if explore:
            metrics = agent.update(state, action, reward, next_state, done)
            if metrics:
                updates_count += 1
                for k, v in metrics.items():
                    episode_metrics[k] = episode_metrics.get(k, 0.0) + float(v)

        state = next_state
        episode_reward += reward
        step += 1

    if updates_count > 0:
        avg_metrics = {k: v / updates_count for k, v in episode_metrics.items()}
        return episode_reward, step, avg_metrics, frames

    return episode_reward, step, {}, frames


def train(args: dict, stats: StatsCollector):
    env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])
    eval_env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])

    # Initialize MPO learner components
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

    logger.info("Setup complete.")

    logger.info(f"Starting training loop for {args['episodes']} episodes. Visualization: {args['visualize']}")

    dummy_stats = {
        "Episode_Reward": 0,
        "Episode_Length": args["steps"],
        "Buffer_Size": len(buffer),
        "Episode_Loss": np.nan,
    }
    stats.log_stats_to_tb(0, dummy_stats)

    for episode in range(1, args["episodes"] + 1):
        ep_reward, ep_length, metrics, _ = run_episode(env, agent, args)
        ep_stats = {
            "Episode_Reward": ep_reward,
            "Episode_Length": ep_length,
            "Buffer_Size": len(buffer),
            **metrics
        }

        stats.log_stats_to_tb(episode, ep_stats)

        stats.log_progress(episode, args["episodes"], ep_stats, {"Loss": metrics.get("loss_critic", 0.0)})

        if episode in [4, 5, 6] or episode % args["eval_frequency"] == 0:
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
            logger.info(f"Mean evaluation reward over {args['num_eval_episodes']} episodes: {mean_eval_reward:.2f}")

            stats.flush_stats_to_disk()
            stats.save_checkpoint(agent.learner.state, "latest")
            if stats.update_best_checkpoint(mean_eval_reward, agent.learner.state):
                logger.info(f"New best mean eval reward: {stats.best_eval_reward:.2f} - checkpoint saved.")

    stats.flush_stats_to_disk()
    stats.save_checkpoint(agent.learner.state, "final")
    logger.info(f"Dumped training statistics to {stats.stats_file}.")
    logger.success("Training completed successfully!")
