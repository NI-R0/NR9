from src.utils import setup_logger
from loguru import logger
import argparse


@logger.catch
def main():
    setup_logger()
    logger.warning("call from main.py")
    raise NotImplementedError("tesssssssssst")


if __name__ == "__main__":
    main()
