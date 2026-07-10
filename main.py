import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'

from loguru import logger
from src.cli import parse_args
from src.collector import StatsCollector
from src.train import train
from src.test import test


@logger.catch
def main():
    args = parse_args()
    stats = StatsCollector(args, level="DEBUG" if args["verbose"] else "INFO")

    try:
        train(args, stats) if args["task"] == "train" else test(args, stats)
    except KeyboardInterrupt:
        logger.warning("Shutting down training!")
    except Exception as e:
        logger.exception("Uncaught exception occured during training!")
        raise e
    finally:
        stats.close()


if __name__ == "__main__":
    main()


# TODO:
# 1. EMA for stability
# 2. Baseline using acme MPO
