import os
import sys
import logging
from datetime import datetime
from tensorboardX import SummaryWriter
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


def print_run_info(args: dict):
    msg = f"""
###############################################################################
Training Summary:
    - Run name: {args['outdir']}/{args['run_name']}
    - Environment: {args['env_domain']} (task: {args['env_task']})
    - Duration: {args['episodes']} Episodes at {args['steps']} Steps

Training Configuration:
    - Seed: {args['seed']}
    - Warmup: {args['warmup']} Steps
    - Batch Size: {args['batch_size']}
    - Learning Rate: {args['learning_rate']}
    - Dual Learning Rate: {args['dual_learning_rate']}
    - Buffer Capacity: {args['capacity']}
    - Tau: {args['tau']}

Evaluation Configuration:
    - Interval: {args['eval_frequency']}
    - Eval Duration: {args['num_eval_episodes']} Episodes
###############################################################################
        """
    logger.info(msg)
