import os
from loguru import logger
from datetime import datetime
import jax
from src.cli import parse_args
from src.utils import setup_logger, print_run_info
from src.train import train
from src.test import test


@logger.catch
def main():
    args = parse_args()

    # Setup run directory
    run_name = args["run_name"] if args["run_name"] else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # fmt: off
    run_dir = os.path.join(os.getcwd(), args["outdir"], run_name)
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    if args["seed"] is None: args["seed"] = 42

    args["run_name"] = run_name
    args["run_dir"] = run_dir
    args["random_key"] = jax.random.PRNGKey(args["seed"])

    setup_logger("DEBUG" if args["verbose"] else "INFO", outdir=run_dir)
    logger.info(f"Using output directory at {run_dir}")

    print_run_info(args)

    try:
        train(args) if args["task"] == "train" else test(args)
    except KeyboardInterrupt:
        logger.warning("Shutting down training!")
    except Exception as e:
        logger.exception("Uncaught exception occured during training!")
        raise e


if __name__ == "__main__":
    main()
