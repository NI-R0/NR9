import os
import sys
import logging
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Intercepts standard `logging` calls and routes them through `loguru`, for when external dependencies use pythons builtin logging.
    """

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logger(level: str = "INFO", outdir: str = "logs"):
    logger.remove()

    stdout_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )
    logfile_fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        # "[<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>]"
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout, format=stdout_fmt, level=level, enqueue=True
    )
    logger.add(
        os.path.join(outdir, "log_{time}.log"),
        format=logfile_fmt,
        level=level,
        enqueue=True,
        backtrace=True
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.info("Logger initialized successfully")
    return logger


def setup_tensorboard(outdir: str = "runs") -> SummaryWriter:
    logdir = os.path.join(outdir, "tensorboard")
    writer = SummaryWriter(log_dir=logdir)
    logger.info(f"Tensorboard logger initialized successfully")
    return writer


def log_stats_to_tb(writer: SummaryWriter, episode: int, stats: dict):
    for key, value in stats.items():
        writer.add_scalar(f"Metrics/{key}", value, episode)
    logger.debug(f"Added metrics to tensorboard for episode {episode}.")


def parse_args() -> dict:
    parser = argparse.ArgumentParser()

    # General CLI args
    parser.add_argument("-v", "--visualize", default=False, action="store_true",
                        help="Enables visualization. Does not work on headless servers or in WSL.")
    parser.add_argument("--debug", default=False, action="store_true", help="Sets logger output level to DEBUG.")
    parser.add_argument(
        "--outdir", default="runs", type=str,
        help="The outdir will be created in the current working directory and used by all loggers and file dumps.")
    parser.add_argument("--run_name", default=None, type=str)

    # Training-specific CLI args
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    return vars(parser.parse_args())
