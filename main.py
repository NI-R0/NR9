import os
import subprocess
import cProfile
import pstats
from loguru import logger
from src.cli import parse_args
from src.collector import StatsCollector
from src.train import train
from src.test import test

def _nvidia_gpu_available() -> bool:
    """Check whether an NVIDIA GPU with working drivers is present."""
    if not any(os.path.exists(f"/dev/nvidia{i}") for i in range(4)):
        return False
    try:
        return subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        ).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


if _nvidia_gpu_available():
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
else:
    os.environ["JAX_PLATFORMS"] = "cpu"
    n_devices = os.cpu_count() or 1
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_force_host_platform_device_count={n_devices}"
    )



@logger.catch
def main():
    args = parse_args()
    stats = StatsCollector(args, level="DEBUG" if args["verbose"] else "INFO")

    profiler = None
    try:
        if args["profile"]:
            profiler = cProfile.Profile()
            profiler.enable()

        train(args, stats) if args["task"] == "train" else test(args, stats)
    except KeyboardInterrupt:
        logger.warning("Shutting down training!")
    except Exception as e:
        logger.exception("Uncaught exception occured during training!")
        raise e
    finally:
        if profiler is not None:
            profiler.disable()
            prof_path = os.path.join(stats.run_dir, "profile.prof")
            profiler.dump_stats(prof_path)
            logger.info(f"Profile saved to {prof_path}")

            summary = pstats.Stats(profiler).strip_dirs().sort_stats("cumulative")
            logger.info("Top 30 functions by cumulative time:")
            summary.print_stats(30)

        stats.close()


if __name__ == "__main__":
    main()


# TODO:
# 1. EMA for stability
# 2. Baseline using acme MPO
