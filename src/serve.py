"""Headless MJPEG stream server with checkpoint hot-swap.

Runs the agent in the environment, renders frames via offscreen EGL,
and serves them over HTTP as an MJPEG stream.  Designed for headless
clusters: view in a local browser via SSH / VS Code port forwarding.

The checkpoint file is polled periodically; when it changes (e.g.
training saved a new ``latest.pkl``), the agent's weights are
hot-swapped without restarting the stream.

Usage::

    MUJOCO_GL=egl uv run python main.py --task serve \
        --load_dir runs/run_20260730_102646 --checkpoint latest \
        --serve_port 8080 --checkpoint_poll_interval 5 --respawn \
        --env_domain walker_3D_ball --env_task run
"""

import io
import os
import time
import threading
import http.server
from typing import Optional

from PIL import Image
from loguru import logger

from src.collector import StatsCollector
from src.environment import Environment
from src.agent import SoccerAgent
from src.buffer import NStepTransitionBuffer
from src.networks import ActorNetwork, CriticNetwork


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

        if os.path.isfile(self._path):
            self._last_mtime = os.path.getmtime(self._path)

    def maybe_reload(self):
        """Check if the checkpoint file changed and reload if so.

        Uses ``_poll_interval`` to avoid stat-ing the file too
        frequently.  Safe to call on every step.
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

    _frame_buffer: Optional[list[bytes]] = None
    _frame_lock: Optional[threading.Lock] = None
    _fps: int = 30

    def do_GET(self):
        if self.path not in ("/", "/stream"):
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
        pass


def serve(args: dict, stats: StatsCollector):
    """Start a headless MJPEG stream with checkpoint hot-swap.

    Expects the same ``args`` dict as :func:`src.test.test`.  Relevant
    keys: ``load_dir``, ``checkpoint``, ``serve_port``,
    ``checkpoint_poll_interval``, ``respawn``, ``env_domain``,
    ``env_task``, ``steps``, and the agent hyper-parameters.
    """
    if not args["load_dir"]:
        logger.error("Serve mode requires --load_dir to be set.")
        return

    checkpoint_path = os.path.join(
        args["load_dir"], "checkpoints", f"{args['checkpoint']}.pkl"
    )
    if not os.path.isfile(checkpoint_path):
        logger.error(f"No checkpoint found at '{checkpoint_path}'.")
        return

    port = args.get("serve_port", 2324)
    poll_interval = args.get("checkpoint_poll_interval", 10.0)
    respawn = args.get("respawn", False)

    # --- Build env + agent (same setup as test()) ---
    env = Environment(
        domain_name=args["env_domain"],
        task_name=args["env_task"],
        max_steps=args["steps"],
    )

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
        **args,
    )

    logger.info(f"Loading checkpoint '{args['checkpoint']}' from {checkpoint_path}")
    agent.learner.state = StatsCollector.load_checkpoint_file(checkpoint_path)

    _run_stream(env, agent, checkpoint_path, port, poll_interval, respawn)


def _run_stream(
    env: Environment,
    agent: SoccerAgent,
    checkpoint_path: str,
    port: int,
    poll_interval: float,
    respawn: bool,
):
    """Run the env loop and serve frames over HTTP MJPEG.

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
    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _MJPEGStreamHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    logger.info(
        f"MJPEG stream ready on http://localhost:{port}  "
        f"(forward port, then open in browser)"
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
