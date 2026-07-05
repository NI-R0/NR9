import os
import sys
import logging
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
    log_path = os.path.join(os.getcwd(), outdir)
    os.makedirs(log_path, exist_ok=True)

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
        os.path.join(log_path, "log_{time}.log"),
        format=logfile_fmt,
        level=level,
        enqueue=True,
        backtrace=True
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    logger.info("Logger initialized successfully")
    return logger
