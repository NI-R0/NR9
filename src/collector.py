import os
import sys
import json
import logging
import time
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

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


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

        # I/O timing tracking (populated when profile=True)
        self.io_timings: dict[str, list[float]] = {}
        self._profile = False

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

        logger.add(sys.stdout, format=stdout_fmt, level=level, enqueue=True)
        tag = "test" if self.is_test else "train"
        logger.add(
            os.path.join(self.log_dir, f"{tag}_{{time}}.log"),
            format=logfile_fmt,
            level=level,
            enqueue=True,
            backtrace=True,
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
    - Environment: {args["env_domain"]} (task: {args["env_task"]})
    - Duration: {duration_str}

Training Configuration:
    - Seed: {args["seed"]}
    - Warmup: {args["warmup"]} Steps
    - Batch Size: {args["batch_size"]}
    - Learning Rate: {args["lr"]}
    - Critic Learning Rate: {args["critic_lr"]}
    - Dual Learning Rate: {args["dual_lr"]}
    - Buffer Capacity: {args["capacity"]}
    - Gamma: {args["gamma"]}
    - Epsilon (E-step): {args["epsilon"]}
    - Epsilon Mean (M-step): {args["epsilon_mean"]}
    - Epsilon Std (M-step): {args["epsilon_std"]}
    - Sample K: {args["sample_k"]}
    - N-step: {args["n_step"]}
    - SGD steps/learner step: {args["sgd_steps_per_learner_step"]}
    - Target update period: {args["target_update_period"]}
    - Grad norm clip: {args["grad_norm_clip"]}

Evaluation Configuration:
    - Interval: {args["eval_frequency"]}
    - Eval Duration: {args["num_eval_episodes"]} Episodes
###############################################################################
        """
        logger.info(msg)

    # Public methods ####################################

    def set_profile(self, enabled: bool):
        """Enable I/O timing tracking for profiling."""
        self._profile = enabled

    def _record_io(self, name: str, duration: float):
        if self._profile:
            self.io_timings.setdefault(name, []).append(duration)

    def get_io_summary(self) -> dict[str, dict]:
        """Return a summary of I/O timings (count, total_s, mean_ms).

        Only populated when profiling is enabled.
        """
        summary = {}
        for name, timings in self.io_timings.items():
            summary[name] = {
                "count": len(timings),
                "total_s": sum(timings),
                "mean_ms": (sum(timings) / len(timings) * 1000) if timings else 0.0,
            }
        return summary

    def log_io_summary(self):
        """Log a summary of I/O timings (call at end of training)."""
        summary = self.get_io_summary()
        if not summary:
            return
        logger.info("I/O timing summary:")
        for name, s in sorted(summary.items(), key=lambda x: -x[1]["total_s"]):
            logger.info(
                f"  {name}: {s['count']} calls, "
                f"{s['total_s']:.3f}s total, "
                f"{s['mean_ms']:.1f}ms/call"
            )

    def log_stats_to_tb(self, episode: int, stats: dict):
        self.stats.setdefault(episode, {}).update(stats)
        t0 = time.perf_counter()
        for key, value in stats.items():
            self.writer.add_scalar(f"Metrics/{key}", value, episode)
        self._record_io("tb_add_scalar", time.perf_counter() - t0)
        logger.debug(f"Added metrics to tensorboard for episode {episode}.")

    def log_hparams(self, args: dict):
        """Log hyperparameters to TensorBoard HParams tab.

        Must be called once at the start of training (before any metrics).
        The final metric (Mean_Eval_Reward) is used as the HParams metric.
        """
        hparam_keys = [
            "env_domain",
            "env_task",
            "steps",
            "seed",
            "warmup",
            "batch_size",
            "lr",
            "critic_lr",
            "dual_lr",
            "capacity",
            "gamma",
            "epsilon",
            "epsilon_mean",
            "epsilon_std",
            "sample_k",
            "n_step",
            "sgd_steps_per_learner_step",
            "target_update_period",
            "grad_norm_clip",
            "update_every",
            "num_envs",
            "eval_frequency",
            "num_eval_episodes",
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

    def log_progress(
        self,
        episode: int,
        total_episodes: int | str,
        ep_stats: dict,
        extra_metrics: dict | None = None,
    ):
        metrics_str = ", ".join(
            f"{k}: {v:.4f}" for k, v in (extra_metrics or {}).items()
        )
        logger.info(
            f"Episode [{episode}/{total_episodes}] - Reward: {ep_stats['Episode_Reward']:.2f} "
            f"| Buffer Size: {ep_stats['Buffer_Size']}"
            + (f" | {metrics_str}" if metrics_str else "")
        )

        # Log reward component breakdown
        reward_comp = {k: v for k, v in ep_stats.items() if k.startswith("Reward_")}
        if reward_comp:
            comp_str = ", ".join(
                f"{k.replace('Reward_', '')}: {v:+.3f}"
                for k, v in sorted(reward_comp.items())
            )
            logger.info(f"  Reward breakdown → {comp_str}")

    def flush_stats_to_disk(self):
        t0 = time.perf_counter()
        with open(self.stats_file, "w") as f:
            json.dump(self.stats, f, indent=4)
        self._record_io("flush_stats_to_disk", time.perf_counter() - t0)

    def save_checkpoint(self, state, name: str) -> str:
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            cloudpickle.dump(state, f)
        self._record_io(f"save_checkpoint({name})", time.perf_counter() - t0)
        return path

    def load_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        return self.load_checkpoint_file(path)

    @staticmethod
    def load_checkpoint_file(path: str):
        with open(path, "rb") as f:
            return cloudpickle.load(f)

    def update_best_checkpoint(self, eval_reward: float, state) -> bool:
        improved = eval_reward > self.best_eval_reward
        if improved:
            self.best_eval_reward = eval_reward
            self.save_checkpoint(state, "best_ckpt")
        return improved

    def _write_training_meta(self, episode: int, agent_step_count: int = 0):
        """Write training_meta.json with current episode + best reward.

        This is read by the checkpoint server (``--task serve``) and
        external tools (e.g. check_convergence.py) without unpickling
        the full state.
        """
        meta_path = os.path.join(self.checkpoint_dir, "training_meta.json")
        meta = {
            "episode": episode,
            "best_eval_reward": self.best_eval_reward,
            "agent_step_count": agent_step_count,
        }
        t_meta = time.perf_counter()
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self._record_io("save_train_meta", time.perf_counter() - t_meta)

    def save_train_state(
        self, episode: int, learner_state, buffer, collector, agent_step_count: int = 0
    ):
        """Save a full training checkpoint to disk (atomic write).

        Only serializable collector fields (``stats`` dict and
        ``best_eval_reward``) are stored – the ``SummaryWriter`` and
        logger objects are *not* picklable because they contain
        ``multiprocessing.Queue`` instances.

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
            "agent_step_count": agent_step_count,
        }
        t0 = time.perf_counter()
        with open(tmp_path, "wb") as f:
            cloudpickle.dump(state, f)
        os.replace(tmp_path, path)
        self._record_io("save_train_state", time.perf_counter() - t0)
        logger.debug(f"Full training state saved to {path}.")

        self._write_training_meta(episode, agent_step_count)

    @staticmethod
    def load_train_state(filepath: str):
        with open(filepath, "rb") as f:
            state = cloudpickle.load(f)
        return (
            state["episode"],
            state["learner_state"],
            state["buffer"],
            state["collector"],
            state.get("agent_step_count", 0),
        )

    def close(self):
        self.writer.close()
