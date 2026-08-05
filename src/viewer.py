import os
import time
import threading

import dm_env
import imageio
import numpy as np
from dm_control.viewer import application
from loguru import logger

from src.agent import MPOAgent
from src.collector import StatsCollector
from src.environment import Environment
from src.serve import RemoteCheckpointReloader


class _AutoResetWrapper:
    """Wraps a raw dm_control Environment for the interactive viewer."""

    def __init__(self, raw_env, respawn: bool):
        self._env = raw_env
        self._respawn = respawn

    @property
    def physics(self):
        return self._env.physics

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def step(self, action):
        timestep = self._env.step(action)

        if not self._respawn:
            return timestep

        # Determine whether the episode should end.  The raw dm_control
        # env only signals last() via its own time-limit.
        done = timestep.last()
        if not done:
            task = getattr(self._env, "task", None)
            if task is not None and hasattr(task, "should_terminate"):
                if task.should_terminate(self._env.physics):
                    done = True
                    logger.debug(
                        "Respawn: should_terminate=True "
                        f"(height={self._env.physics.torso_height():.2f}, "
                        f"non_foot_touch={self._env.physics.non_foot_touch():.2f})"
                    )

        if not done:
            return timestep

        logger.info(
            f"Respawn: sub-episode ended (last={timestep.last()}) -> resetting."
        )
        # Preserve the physics simulation time so the viewer's
        # time-based step loop doesn't "catch up" from 0 after reset
        # (which would cause a multi-second freeze).
        sim_time = self._env.physics.data.time
        new_ts = self._env.reset()
        self._env.physics.data.time = sim_time
        # Return a MID timestep so the viewer keeps running.
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=0.0,
            discount=1.0,
            observation=new_ts.observation,
        )


class _CheckpointReloader:
    def __init__(
        self,
        checkpoint_path: str,
        agent: MPOAgent,
        poll_interval: float,
    ):
        self._path = checkpoint_path
        self._agent = agent
        self._poll_interval = poll_interval
        self._last_mtime: float | None = None
        self._last_check: float = 0.0
        self._lock = threading.Lock()

        if os.path.isfile(self._path):
            self._last_mtime = os.path.getmtime(self._path)

    def maybe_reload(self):
        """Check if the checkpoint file changed and reload if so.
        """
        if self._poll_interval <= 0:
            return

        now = time.monotonic()
        if now - self._last_check < self._poll_interval:
            return
        self._last_check = now

        if not os.path.isfile(self._path):
            return

        mtime = os.path.getmtime(self._path)
        if self._last_mtime is not None and mtime <= self._last_mtime:
            return

        with self._lock:
            try:
                new_state = StatsCollector.load_checkpoint_file(self._path)
                self._agent.learner.state = new_state
                self._last_mtime = mtime
                logger.success(
                    f"Hot-swapped checkpoint: {self._path} "
                    f"(mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})"
                )
            except Exception:
                logger.exception(
                    f"Failed to reload checkpoint from {self._path} - "
                    "keeping old weights."
                )


def _format_title(episode, reward):
    """Build a compact window title showing checkpoint origin."""
    parts = ["NR9 Viewer"]
    if episode is not None:
        parts.append(f"Ep {episode}")
    if reward is not None:
        parts.append(f"R {reward:.1f}")
    return " | ".join(parts)


def save_video(frames: list, path: str, fps: int = 30):
    try:
        imageio.mimwrite(path, frames, fps=fps)
        return path
    except Exception as e:
        gif_path = os.path.splitext(path)[0] + ".gif"
        logger.warning(f"Could not write mp4: {e}. Falling back to '{gif_path}'.")
        try:
            imageio.mimsave(gif_path, frames, fps=fps)
            return gif_path
        except Exception as e2:
            logger.warning(f"Could not write GIF recording, skipping video export: {e2}")
            return None


def run_live(
    env: Environment,
    agent: MPOAgent,
    respawn: bool = False,
    checkpoint_path: str | None = None,
    poll_interval: float = 0.0,
    stream_url: str | None = None,
    reloader: RemoteCheckpointReloader | None = None,
):
    """Launch the interactive dm_control viewer with the trained agent.
    """
    # Use Application directly (instead of viewer.launch) so we can
    # update the window title dynamically via set_title.
    initial_ep = reloader.checkpoint_episode if reloader else None
    initial_rw = reloader.checkpoint_reward if reloader else None
    app = application.Application(title=_format_title(initial_ep, initial_rw))

    def _on_swap(episode, reward):
        title = _format_title(episode, reward)
        logger.info(f"on_swap callback fired: episode={episode}, reward={reward} -> '{title}'")
        try:
            app._window.set_title(title)
        except Exception as e:
            logger.warning(f"Could not set window title to '{title}': {e}")

    if stream_url and reloader is not None:
        # Reuse the reloader created by test() for the initial fetch.
        reloader._on_swap = _on_swap
        reloader.start()
    elif stream_url:
        url = RemoteCheckpointReloader.normalize_stream_url(stream_url)
        effective_interval = poll_interval if poll_interval > 0 else 5.0
        reloader = RemoteCheckpointReloader(
            url, agent, effective_interval, on_swap=_on_swap
        )
        reloader.start()
    elif checkpoint_path and poll_interval > 0:
        reloader = _CheckpointReloader(checkpoint_path, agent, poll_interval)
        logger.info(
            f"Local checkpoint hot-swap enabled: polling '{checkpoint_path}' "
            f"every {poll_interval:.1f}s"
        )

    def policy(timestep):
        if reloader is not None:
            reloader.maybe_reload()
        obs = env._flatten_observation(timestep.observation)
        action = agent.select_action(obs, explore=False)
        return np.asarray(action, dtype=np.float32)

    raw_env = _AutoResetWrapper(env.env, respawn=respawn)

    logger.info(
        "Launching interactive viewer. Close the window to exit."
        + (" (respawn enabled)" if respawn else "")
        + (f" (hot-swap every {poll_interval:.0f}s)" if reloader else "")
    )
    try:
        app.launch(environment_loader=raw_env, policy=policy)
    finally:
        if isinstance(reloader, RemoteCheckpointReloader):
            reloader.stop()
