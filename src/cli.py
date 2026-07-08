import argparse


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # General CLI args
    parser.add_argument("-t", "--task", type=str, choices=["train", "test"], default="train")
    parser.add_argument("-v", "--visualize", default=False, action="store_true",
                        help="Enables visualization. Does not work on headless servers or in WSL.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Sets logger output level to DEBUG.")
    parser.add_argument(
        "--outdir", default="runs", type=str,
        help="The outdir will be created in the current working directory and used by all loggers and file dumps.")
    parser.add_argument("--run_name", default=None, type=str)

    # Env-specific flags
    parser.add_argument("--env_domain", type=str, default="cartpole")
    parser.add_argument("--env_task", type=str, default="balance")

    # Training-specific CLI args
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps each episode runs for.")
    parser.add_argument("--warmup", type=int, default=200,
                        help="Number of batches to fill buffer with before starting training.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dual_lr", type=float, default=0.01)
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--tau", type=float, default=0.005)

    parser.add_argument("--eval_frequency", type=int, default=10)
    parser.add_argument(
        "--num_eval_episodes", type=int, default=5, help="Number of episodes to run evaluation for.")

    return vars(parser.parse_args())
