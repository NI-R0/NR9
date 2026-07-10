import os
import sys
import json
import pickle
import logging
from loguru import logger
from tensorboardX import SummaryWriter


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


class StatsCollector:
    def __init__(self, args: dict, level: str = "INFO"):
        run_name = args["run_name"] or self._default_run_name()
        self.is_test = args["task"] == "test" and args["load_dir"]

        if self.is_test:
            self.run_dir = os.path.join(os.getcwd(), args["load_dir"])
            self.outdir = os.path.join(self.run_dir, "test", run_name)

        else:
            self.run_dir = os.path.join(os.getcwd(), args["outdir"], run_name)
            self.outdir = self.run_dir

        self.log_dir = os.path.join(self.run_dir, "logs")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.stats_file = os.path.join(self.outdir, "training_stats.json")
        self.config_file = os.path.join(self.outdir, "run_config.json")

        os.makedirs(self.run_dir, exist_ok=self.is_test)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=False)

        args["run_name"] = run_name
        args["run_dir"] = self.run_dir
        args["outdir"] = self.outdir

        self._setup_logger(level)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        logger.info(f"Tensorboard logger initialized at {self.tb_dir}")

        self.stats: dict = {}
        self.best_eval_reward = -float("inf")

        if args["task"] == "test":
            return

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._dump_config(args)
        self._print_run_info(args)

    @staticmethod
    def _default_run_name() -> str:
        from datetime import datetime
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _setup_logger(self, level: str):
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
        tag = "test" if self.is_test else "train"
        logger.add(
            os.path.join(self.log_dir, f"{tag}_{{time}}.log"),
            format=logfile_fmt,
            level=level,
            enqueue=True,
            backtrace=True
        )

        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        logger.info("Logger initialized successfully")

    def _dump_config(self, args: dict):
        with open(self.config_file, "w") as f:
            json.dump(args, f, indent=4, default=str)

    def _print_run_info(self, args: dict):
        msg = f"""
###############################################################################
Training Summary:
    - Run name: {self.run_dir}
    - Environment: {args['env_domain']} (task: {args['env_task']})
    - Duration: {args['episodes']} Episodes at {args['steps']} Steps

Training Configuration:
    - Seed: {args['seed']}
    - Warmup: {args['warmup']} Steps
    - Batch Size: {args['batch_size']}
    - Learning Rate: {args['lr']}
    - Dual Learning Rate: {args['dual_lr']}
    - Buffer Capacity: {args['capacity']}
    - Tau: {args['tau']}

Evaluation Configuration:
    - Interval: {args['eval_frequency']}
    - Eval Duration: {args['num_eval_episodes']} Episodes
###############################################################################
        """
        logger.info(msg)

    # Public methods ####################################

    def log_stats_to_tb(self, episode: int, stats: dict):
        self.stats.setdefault(episode, {}).update(stats)
        for key, value in stats.items():
            self.writer.add_scalar(f"Metrics/{key}", value, episode)
        logger.debug(f"Added metrics to tensorboard for episode {episode}.")

    def log_progress(self, episode: int, total_episodes: int, ep_stats: dict,
                     extra_metrics: dict | None = None):
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in (extra_metrics or {}).items())
        logger.info(
            f"Episode [{episode}/{total_episodes}] - Reward: {ep_stats['Episode_Reward']:.2f} "
            f"| Buffer Size: {ep_stats['Buffer_Size']}"
            + (f" | {metrics_str}" if metrics_str else "")
        )

    def flush_stats_to_disk(self):
        with open(self.stats_file, "w") as f:
            json.dump(self.stats, f, indent=4)

    def save_checkpoint(self, state, name: str) -> str:
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return path

    def load_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        return self.load_checkpoint_file(path)

    @staticmethod
    def load_checkpoint_file(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def update_best_checkpoint(self, eval_reward: float, state) -> bool:
        improved = eval_reward > self.best_eval_reward
        if improved:
            self.best_eval_reward = eval_reward
            self.save_checkpoint(state, "best_ckpt")
        return improved

    def close(self):
        self.writer.close()
