import os
from loguru import logger
from datetime import datetime
from src.utils import setup_logger, parse_args
from src.train import train
from src.test import test


@logger.catch
def main():
    args = parse_args()

    # Setup run directory
    run_name = args["run_name"] if args["run_name"] else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # fmt: off
    run_dir = os.path.join(os.getcwd(), args["outdir"], run_name)
    os.makedirs(run_dir, exist_ok=False)
    args["run_name"] = run_name
    args["run_dir"] = run_dir

    setup_logger("DEBUG" if args["verbose"] else "INFO", outdir=run_dir)
    logger.info(f"Using output directory at {run_dir}")

    try:
        train(args) if args.task == "train" else test(args)
    except KeyboardInterrupt:
        logger.warning("Shutting down training!")
    except Exception as e:
        logger.exception("Uncaught exception occured during training!")
        raise e


if __name__ == "__main__":
    main()
