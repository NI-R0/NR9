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
        self.is_resume = args["task"] == "train" and args.get("resume_dir") is not None

        if self.is_test:
            self.run_dir = os.path.join(os.getcwd(), args["load_dir"])
            self.outdir = os.path.join(self.run_dir, "test", run_name)
        elif self.is_resume:
            self.run_dir = os.path.join(os.getcwd(), args["resume_dir"])
            self.outdir = self.run_dir
        else:
            self.run_dir = os.path.join(os.getcwd(), args["outdir"], run_name)
            self.outdir = self.run_dir

        self.log_dir = os.path.join(self.run_dir, "logs")
        self.tb_dir = os.path.join(self.run_dir, "tensorboard")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.stats_file = os.path.join(self.outdir, "training_stats.json")
        self.config_file = os.path.join(self.outdir, "run_config.json")

        if self.is_resume:
            if not os.path.isdir(self.run_dir):
                raise FileNotFoundError(
                    f"Resume directory '{self.run_dir}' does not exist. "
                    "Cannot resume from a non-existent run."
                )
        else:
            os.makedirs(self.run_dir, exist_ok=self.is_test)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.outdir, exist_ok=True)

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

        if self.is_resume:
            self._load_training_meta()
            self._load_stats()
        else:
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
        duration = args.get("duration")
        if duration is not None:
            duration_str = f"{duration:.1f} min (max {args['episodes']} episodes)"
        else:
            duration_str = f"{args['episodes']} Episodes at {args['steps']} Steps"

        msg = f"""
###############################################################################
Training Summary:
    - Run name: {self.run_dir}
    - Environment: {args['env_domain']} (task: {args['env_task']})
    - Duration: {duration_str}

Training Configuration:
    - Seed: {args['seed']}
    - Warmup: {args['warmup']} Steps
    - Batch Size: {args['batch_size']}
    - Learning Rate: {args['lr']}
    - Critic Learning Rate: {args['critic_lr']}
    - Dual Learning Rate: {args['dual_lr']}
    - Buffer Capacity: {args['capacity']}
    - Gamma: {args['gamma']}
    - Epsilon (E-step): {args['epsilon']}
    - Epsilon Mean (M-step): {args['epsilon_mean']}
    - Epsilon Std (M-step): {args['epsilon_std']}
    - Sample K: {args['sample_k']}
    - N-step: {args['n_step']}
    - SGD steps/learner step: {args['sgd_steps_per_learner_step']}
    - Target update period: {args['target_update_period']}
    - Grad norm clip: {args['grad_norm_clip']}

Evaluation Configuration:
    - Interval: {args['eval_frequency']}
    - Eval Duration: {args['num_eval_episodes']} Episodes
###############################################################################
        """
        logger.info(msg)

    # Public methods ####################################

    def save_training_meta(self, episode: int, phase: int = 0):
        """Persist training-level metadata (episode count, best eval reward, phase).

        Called alongside checkpoint saves so that a resumed run can pick
        up exactly where the previous run left off.
        """
        meta = {
            "episode": episode,
            "best_eval_reward": self.best_eval_reward,
            "phase": phase,
        }
        path = os.path.join(self.checkpoint_dir, "training_meta.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=4)

    def _load_training_meta(self):
        """Load training-level metadata from a previous run."""
        path = os.path.join(self.checkpoint_dir, "training_meta.json")
        if not os.path.isfile(path):
            logger.warning(
                f"No training_meta.json found in {self.checkpoint_dir}. "
                "Resuming without episode count / best eval reward - "
                "episode counter starts at 0, best eval reward at -inf."
            )
            return

        with open(path) as f:
            meta = json.load(f)
        self._resumed_episode = meta.get("episode", 0)
        self.best_eval_reward = meta.get("best_eval_reward", -float("inf"))
        self._resumed_phase = meta.get("phase", 0)
        logger.info(
            f"Resuming from episode {self._resumed_episode} "
            f"with best eval reward {self.best_eval_reward:.2f} "
            f"and phase {self._resumed_phase}."
        )

    @property
    def resumed_episode(self) -> int:
        """Episode count from a previous run, or 0 for a fresh start."""
        return getattr(self, "_resumed_episode", 0)

    @property
    def resumed_phase(self) -> int:
        """Curriculum phase from a previous run, or 0 for a fresh start."""
        return getattr(self, "_resumed_phase", 0)

    def _load_stats(self):
        """Load previously saved training stats so they are preserved on resume."""
        if os.path.isfile(self.stats_file):
            with open(self.stats_file) as f:
                self.stats = json.load(f)
            logger.info(f"Loaded {len(self.stats)} episodes of stats from {self.stats_file}.")
        else:
            logger.warning(f"No stats file found at {self.stats_file} - starting with empty stats.")

    def log_stats_to_tb(self, episode: int, stats: dict):
        self.stats.setdefault(episode, {}).update(stats)
        for key, value in stats.items():
            self.writer.add_scalar(f"Metrics/{key}", value, episode)
        logger.debug(f"Added metrics to tensorboard for episode {episode}.")

    def log_progress(self, episode: int, total_episodes: int | str, ep_stats: dict,
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
