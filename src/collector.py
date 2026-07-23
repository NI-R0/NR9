import os
import sys
import json
import pickle
import logging
import cloudpickle
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
        # When resuming, derive the run name from the checkpoint path so
        # the same run directory is reused instead of creating a new one.
        resume_path = args.get("resume")
        if resume_path and os.path.exists(resume_path):
            # resume path is .../<run_name>/checkpoints/state.pkl
            run_dir_from_resume = os.path.dirname(os.path.dirname(resume_path))
            run_name = os.path.basename(run_dir_from_resume)
        else:
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

        if self.is_test:
            os.makedirs(self.run_dir, exist_ok=True)
        else:
            os.makedirs(self.run_dir, exist_ok=bool(resume_path))

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

    def log_stats_to_tb(self, episode: int, stats: dict):
        self.stats.setdefault(episode, {}).update(stats)
        for key, value in stats.items():
            self.writer.add_scalar(f"Metrics/{key}", value, episode)
        logger.debug(f"Added metrics to tensorboard for episode {episode}.")

    def log_hparams(self, args: dict):
        """Log hyperparameters to TensorBoard HParams tab.

        Must be called once at the start of training (before any metrics).
        The final metric (Mean_Eval_Reward) is used as the HParams metric.
        """
        hparam_keys = [
            "env_domain", "env_task", "steps",
            "seed", "warmup", "batch_size", "lr", "critic_lr", "dual_lr",
            "capacity", "gamma", "epsilon", "epsilon_mean", "epsilon_std",
            "sample_k", "n_step", "sgd_steps_per_learner_step",
            "target_update_period", "grad_norm_clip", "update_every",
            "num_envs", "eval_frequency", "num_eval_episodes",
            "curriculum", "phase1_threshold", "phase2_threshold",
            "phase3_threshold",
        ]
        hparams = {}
        for k in hparam_keys:
            if k in args:
                val = args[k]
                if isinstance(val, bool):
                    hparams[k] = int(val)
                elif val is None:
                    continue
                else:
                    hparams[k] = val
        metric = {"Mean_Eval_Reward": 0.0}
        self.writer.add_hparams(hparams, metric)
        logger.info(f"Logged {len(hparams)} hyperparameters to TensorBoard.")

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

    def save_train_state(self, episode: int, learner_state, buffer, collector,
                         current_phase: int = 0, agent_step_count: int = 0):
        """Save a full training checkpoint to disk (atomic write).

        Only serializable collector fields (``stats`` dict and
        ``best_eval_reward``) are stored – the ``SummaryWriter`` and
        logger objects are *not* picklable because they contain
        ``multiprocessing.Queue`` instances.

        ``current_phase`` is stored so curriculum progress survives resume.
        ``agent_step_count`` is stored so the warmup/update_every timing
        is preserved across restarts.
        """
        tmp_path = os.path.join(self.checkpoint_dir, "state.tmp")
        path = os.path.join(self.checkpoint_dir, "state.pkl")
        state = {
            "episode": episode,
            "learner_state": learner_state,
            "buffer": buffer,
            "collector": {
                "stats": collector.stats,
                "best_eval_reward": collector.best_eval_reward,
            },
            "current_phase": current_phase,
            "agent_step_count": agent_step_count,
        }
        with open(tmp_path, "wb") as f:
            cloudpickle.dump(state, f)

        os.replace(tmp_path, path)
        logger.debug(f"Full training state saved to {path}.")

    @staticmethod
    def load_train_state(filepath: str):
        with open(filepath, "rb") as f:
            state = cloudpickle.load(f)
        return (state["episode"], state["learner_state"], state["buffer"],
                state["collector"], state.get("current_phase", 0),
                state.get("agent_step_count", 0))

    def close(self):
        self.writer.close()
