import os
import io
import time
import threading
import http.server
from typing import Optional
import numpy as np
import jax
import imageio
import dm_env
from PIL import Image
from loguru import logger
from dm_control import viewer
from src.collector import StatsCollector
from src.environment import Environment
from src.learner import MPOLearner
from src.agent import SoccerAgent
from src.buffer import NStepTransitionBuffer
from src.networks import ActorNetwork, CriticNetwork
from src.train import run_episode


def run_episode_with_respawn(
    env: Environment,
    agent: SoccerAgent,
    args: dict,
    visualize: bool = False,
):
    """Run a test episode with the same termination/respawn logic as training.

    Unlike ``run_episode``, which stops at the first termination, this
    function keeps stepping until ``max_steps`` is reached.  When the env
    terminates (done=True) it is reset and the episode continues —
    mirroring the auto-reset behaviour of ``run_vectorized_episode``
    during training.

    All completed sub-episodes are tracked individually so you can see
    how many terminations/respawns occur and what reward/length each
    sub-episode achieves.

    Returns ``(all_rewards, all_lengths, frames)``.
    """
    max_steps = args["steps"]
    state = env.reset()
    ep_reward = 0.0
    ep_length = 0
    all_rewards: list[float] = []
    all_lengths: list[int] = []
    frames = [] if visualize else None

    for step in range(max_steps):
        if visualize:
            frame = env.render()
            frames.append(frame)

        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)

        ep_reward += reward
        ep_length += 1

        if done:
            all_rewards.append(ep_reward)
            all_lengths.append(ep_length)
            logger.info(
                f"  Sub-episode terminated at step {step + 1}/{max_steps} | "
                f"Reward: {ep_reward:.2f} | Length: {ep_length} -> respawning"
            )
            ep_reward = 0.0
            ep_length = 0
            next_state = env.reset()

        state = next_state

    # Collect any in-flight (not-yet-terminated) sub-episode.
    if ep_length > 0:
        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
        logger.info(
            f"  Final sub-episode (no termination) | "
            f"Reward: {ep_reward:.2f} | Length: {ep_length}"
        )

    return all_rewards, all_lengths, frames


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


class _AutoResetWrapper:
    """Wraps a raw dm_control Environment for the interactive viewer.

    When ``--respawn`` is active, the wrapper intercepts every
    ``step()`` call.  If the underlying environment signals termination
    (either via its own time-limit or via the task's ``should_terminate``
    method), the wrapper resets the environment immediately and returns
    a **MID** ``TimeStep`` with the *new* observation and a zero reward.

    This keeps the dm_control viewer running indefinitely — every
    termination is followed by an invisible respawn instead of the
    default "EPISODE TERMINATED" freeze.

    When respawn is disabled the wrapper is transparent and delegates
    every call directly to the underlying environment.
    """

    def __init__(self, raw_env, respawn: bool):
        self._env = raw_env
        self._respawn = respawn

    # --- dm_control viewer interface ---

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
        # env only signals ``last()`` via its own time-limit.  Custom
        # early-termination (``should_terminate``, e.g. fall detection)
        # is NOT checked by the raw env — so we check it ourselves here,
        # mirroring the logic in ``Environment.step``.
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

        # --- Respawn path: reset and return a MID timestep ---
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


def run_live(env: Environment, agent: SoccerAgent, respawn: bool = False):
    """Launch the interactive dm_control viewer with the trained agent.

    The viewer calls the policy function on each timestep. We flatten the
    observation for the agent and convert its JAX output back to numpy.

    When ``respawn`` is ``True``, the environment auto-resets on every
    termination (fall or time-limit) so the viewer runs continuously.
    """
    def policy(timestep):
        obs = env._flatten_observation(timestep.observation)
        action = agent.select_action(obs, explore=False)
        return np.asarray(action, dtype=np.float32)

    raw_env = _AutoResetWrapper(env.env, respawn=respawn)

    logger.info(
        "Launching interactive viewer. Close the window to exit."
        + (" (respawn enabled)" if respawn else "")
    )
    viewer.launch(environment_loader=raw_env, policy=policy)


class _CheckpointReloader:
    """Polls a checkpoint file for changes and reloads the agent state.

    Thread-safe: the env-loop thread calls :meth:`maybe_reload` which
    checks the file's mtime and, if changed, loads the new state into
    ``agent.learner.state``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        agent: SoccerAgent,
        poll_interval: float,
    ):
        self._path = checkpoint_path
        self._agent = agent
        self._poll_interval = poll_interval
        self._last_mtime: float | None = None
        self._last_check: float = 0.0
        self._lock = threading.Lock()

        # Record initial mtime (without reloading, since test() already
        # loaded the checkpoint once).
        if os.path.isfile(self._path):
            self._last_mtime = os.path.getmtime(self._path)

    def maybe_reload(self):
        """Check if the checkpoint file changed and reload if so.

        Should be called from the env-loop thread on every step (or
        every few steps).  Uses ``_poll_interval`` to avoid stat-ing
        the file too frequently.
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


class _MJPEGStreamHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler that streams JPEG frames as MJPEG."""

    # Shared frame buffer set by run_live_stream().
    _frame_buffer: Optional[list[bytes]] = None
    _frame_lock: Optional[threading.Lock] = None
    _fps: int = 30

    def do_GET(self):
        if self.path != "/" and self.path != "/stream":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        frame_idx = 0
        while True:
            if _MJPEGStreamHandler._frame_buffer is None:
                time.sleep(0.1)
                continue

            with _MJPEGStreamHandler._frame_lock:
                if frame_idx >= len(_MJPEGStreamHandler._frame_buffer):
                    time.sleep(1.0 / max(_MJPEGStreamHandler._fps, 1))
                    continue
                jpg_bytes = _MJPEGStreamHandler._frame_buffer[frame_idx]
                # Trim old frames to avoid unbounded memory growth.
                if frame_idx > 64:
                    del _MJPEGStreamHandler._frame_buffer[:frame_idx]
                    frame_idx = 0
            frame_idx += 1

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg_bytes)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpg_bytes)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, format, *args):
        # Suppress default HTTP logging to keep the console clean.
        pass


def run_live_stream(
    env: Environment,
    agent: SoccerAgent,
    checkpoint_path: str,
    port: int,
    poll_interval: float,
    respawn: bool = False,
):
    """Run the agent in the env and stream frames via HTTP MJPEG.

    Designed for headless clusters: uses ``env.render()`` (offscreen
    EGL/OSMesa) and serves JPEG frames over HTTP.  View in a local
    browser via SSH port forwarding::

        ssh -L <PORT>:localhost:<PORT> user@cluster
        # then open http://localhost:<PORT> in your browser

    The checkpoint file is polled every ``poll_interval`` seconds.  When
    the file changes (e.g. training saved a new ``latest.pkl``), the
    agent's weights are hot-swapped without restarting the stream.

    Args:
        env: The Environment wrapper (must support ``render()``).
        agent: The SoccerAgent with loaded weights.
        checkpoint_path: Path to the checkpoint .pkl file to watch.
        port: TCP port for the HTTP server.
        poll_interval: Seconds between checkpoint file checks (0 = off).
        respawn: If True, auto-reset the env on termination.
    """
    # --- Shared frame buffer ---
    frame_buffer: list[bytes] = []
    frame_lock = threading.Lock()
    _MJPEGStreamHandler._frame_buffer = frame_buffer
    _MJPEGStreamHandler._frame_lock = frame_lock
    _MJPEGStreamHandler._fps = 30

    # --- HTTP server in background thread ---
    server = http.server.ThreadingHTTPServer(
        ("0.0.0.0", port), _MJPEGStreamHandler
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info(
        f"MJPEG stream ready on http://localhost:{port}  "
        f"(connect via SSH port forwarding: ssh -L {port}:localhost:{port})"
    )

    # --- Checkpoint reloader ---
    reloader = _CheckpointReloader(checkpoint_path, agent, poll_interval)

    # --- Env loop ---
    state = env.reset()
    ep_reward = 0.0
    ep_length = 0
    step = 0

    logger.info("Starting live stream env loop. Press Ctrl+C to stop.")

    try:
        while True:
            reloader.maybe_reload()

            frame = env.render(height=360, width=480)

            # Encode to JPEG
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            jpg_bytes = buf.getvalue()

            with frame_lock:
                frame_buffer.append(jpg_bytes)
                # Keep buffer bounded
                if len(frame_buffer) > 128:
                    del frame_buffer[:64]

            action = agent.select_action(state, explore=False)
            next_state, reward, done, _ = env.step(action)

            ep_reward += reward
            ep_length += 1
            step += 1

            if done:
                if respawn:
                    logger.info(
                        f"Respawn at step {step} | "
                        f"reward: {ep_reward:.2f} | length: {ep_length}"
                    )
                    ep_reward = 0.0
                    ep_length = 0
                    next_state = env.reset()
                else:
                    logger.info(
                        f"Episode ended at step {step} | "
                        f"reward: {ep_reward:.2f} | length: {ep_length} -> resetting"
                    )
                    ep_reward = 0.0
                    ep_length = 0
                    next_state = env.reset()

            state = next_state

            # Pace the loop to ~30 FPS to avoid burning 100% CPU.
            time.sleep(1.0 / 30.0)

    except KeyboardInterrupt:
        logger.info("Live stream stopped by user.")
    finally:
        server.shutdown()
        server_thread.join(timeout=2)
        logger.info("HTTP server shut down.")


def test(args: dict, stats: StatsCollector):
    if not args["load_dir"]:
        logger.error("Test mode requires --load_dir to be set to some previous run's directoriy.")

    checkpoint_path = os.path.join(args["load_dir"], "checkpoints", f"{args['checkpoint']}.pkl")
    if not os.path.isfile(checkpoint_path):
        logger.error(f"No checkpoint found at '{checkpoint_path}'.")
        return

    env = Environment(domain_name=args["env_domain"], task_name=args["env_task"], max_steps=args["steps"])

    actor_net = ActorNetwork(env.action_dim)
    critic_net = CriticNetwork()

    buffer = NStepTransitionBuffer(
        env.state_dim,
        env.action_dim,
        capacity=args["capacity"],
        n_step=args.get("n_step", 5),
        gamma=args.get("gamma", 0.99),
    )

    agent = SoccerAgent(
        observation_shape=env.state_dim,
        action_shape=env.action_dim,
        actor_net=actor_net,
        critic_net=critic_net,
        buffer=buffer,
        **args
    )

    logger.info(f"Loading checkpoint '{args['checkpoint']}' from {checkpoint_path}")
    agent.learner.state = StatsCollector.load_checkpoint_file(checkpoint_path)

    use_respawn = args.get("respawn", False)
    live_stream_port = args.get("live_stream", 0)

    if args.get("live", False):
        run_live(env, agent, respawn=use_respawn)
        return

    if live_stream_port > 0:
        run_live_stream(
            env=env,
            agent=agent,
            checkpoint_path=checkpoint_path,
            port=live_stream_port,
            poll_interval=args.get("checkpoint_poll_interval", 10.0),
            respawn=use_respawn,
        )
        return

    logger.info(
        f"Running {args['num_eval_episodes']} test episode(s) on "
        f"{args['env_domain']}/{args['env_task']}."
        + (" (respawn mode)" if use_respawn else "")
    )

    visualize = args["visualize"]

    episode_rewards = []
    frames = [] if visualize else None
    for episode in range(1, args["num_eval_episodes"] + 1):
        if use_respawn:
            sub_rewards, sub_lengths, ep_frames = run_episode_with_respawn(
                env, agent, args, visualize=visualize
            )
            episode_rewards.extend(sub_rewards)
            logger.info(
                f"Test episode {episode}/{args['num_eval_episodes']} | "
                f"{len(sub_rewards)} sub-episode(s), "
                f"rewards: {[f'{r:.2f}' for r in sub_rewards]}, "
                f"lengths: {sub_lengths}"
            )
            for sub_idx, sub_r in enumerate(sub_rewards):
                stats.log_stats_to_tb(
                    episode * 1000 + sub_idx,
                    {"Test_SubEpisode_Reward": sub_r},
                )
        else:
            ep_reward, ep_length, _, ep_frames, _ = run_episode(
                env, agent, args, explore=False, visualize=visualize
            )
            episode_rewards.append(ep_reward)
            logger.info(
                f"Test episode {episode}/{args['num_eval_episodes']} | "
                f"Reward: {ep_reward:.2f} | Length: {ep_length}"
            )
            stats.log_stats_to_tb(episode, {"Test_Episode_Reward": ep_reward})

        if visualize and ep_frames:
            frames.extend(ep_frames)

    if frames:
        video_path = os.path.join(args["outdir"], f"{args['checkpoint']}.mp4")
        saved_path = save_video(frames, video_path, fps=100)
        if saved_path:
            logger.success(f"Saved test visualization video to {saved_path}")

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    stats.stats["summary"] = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "num_episodes": len(episode_rewards),
        "checkpoint": checkpoint_path,
        "respawn": use_respawn,
    }
    stats.flush_stats_to_disk()

    logger.success(
        f"Testing completed. Mean reward: {mean_reward:.2f} +/- {std_reward:.2f} "
        f"over {len(episode_rewards)} episode(s)."
    )
    return mean_reward
