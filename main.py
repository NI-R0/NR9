import argparse
from loguru import logger
from src.utils import setup_logger
from src.train import train


def parse_args():
    parser = argparse.ArgumentParser()

    # General CLI args
    parser.add_argument("-v", "--visualize", default=False, action="store_true",
                        help="Enables visualization. Does not work on headless servers or in WSL.")
    parser.add_argument("--debug", default=False, action="store_true", help="Sets logger output level to DEBUG.")
    # TODO: set central outdir used by loggers and train

    # Training-specific CLI args
    parser.add_argument("--episodes", type=int, default=1000)
    # TODO: add bs, lr, ...

    return parser.parse_args()


@logger.catch
def main():
    args = parse_args()
    setup_logger("DEBUG" if args.debug else "INFO")
    args = vars(args)

    try:
        train(args)
    except KeyboardInterrupt:
        logger.warning("Shutting down training!")
    except Exception as e:
        logger.exception("Uncaught exception occured during training!")
        raise e


if __name__ == "__main__":
    main()
