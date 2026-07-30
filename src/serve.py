"""Headless checkpoint server for remote hot-swap.

Runs on the cluster (or any machine with the latest checkpoint) and
serves checkpoint state over a minimal HTTP API so that a local machine
running ``--task test --live --stream`` can poll for new weights and
hot-swap them into the agent without restarting the viewer.

Endpoints
---------

``GET /check``
    Returns JSON ``{"mtime": <float>}`` — the modification time of the
    checkpoint file (Unix timestamp).  ``null`` if no checkpoint exists
    yet.

``GET /checkpoint``
    Returns the raw ``cloudpickle`` bytes of the checkpoint
    ``TrainingState``.  The client deserialises with
    :func:`StatsCollector.load_checkpoint_file`.

``GET /health``
    Returns JSON ``{"status": "ok"}`` — useful for verifying the port
    forward is working.

Usage (on the cluster)::

    uv run python main.py --task serve \
        --load_dir runs/run_20260730_102646 --checkpoint latest \
        --serve_port 2324 \
        --env_domain walker_3D_ball --env_task run
"""

import http.server
import json
import os
import time
from loguru import logger

from src.collector import StatsCollector


def _checkpoint_path(load_dir: str, checkpoint_name: str) -> str:
    return os.path.join(load_dir, "checkpoints", f"{checkpoint_name}.pkl")


class _CheckpointHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler exposing /check, /checkpoint, and /health."""

    # Set as class attributes by ``serve()``.
    _checkpoint_path: str = ""

    def do_GET(self):
        if self.path == "/health":
            self._json_response({"status": "ok"})
        elif self.path == "/check":
            self._handle_check()
        elif self.path == "/checkpoint":
            self._handle_checkpoint()
        else:
            self.send_error(404)

    # ------------------------------------------------------------------

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_check(self):
        path = _CheckpointHandler._checkpoint_path
        if not os.path.isfile(path):
            self._json_response({"mtime": None})
            return
        self._json_response({"mtime": os.path.getmtime(path)})

    def _handle_checkpoint(self):
        path = _CheckpointHandler._checkpoint_path
        if not os.path.isfile(path):
            self.send_error(404, "No checkpoint file available")
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ------------------------------------------------------------------

    def log_message(self, fmt, *args):
        pass


def serve(args: dict, stats: StatsCollector):
    """Start the headless checkpoint server.

    Relevant ``args`` keys: ``load_dir``, ``checkpoint``, ``serve_port``.
    """
    if not args["load_dir"]:
        logger.error("Serve mode requires --load_dir to be set.")
        return

    ckpt_path = _checkpoint_path(args["load_dir"], args["checkpoint"])
    if not os.path.isfile(ckpt_path):
        logger.error(f"No checkpoint found at '{ckpt_path}'.")
        return

    port = args.get("serve_port", 2324)

    _CheckpointHandler._checkpoint_path = ckpt_path

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), _CheckpointHandler)

    logger.info(
        f"Checkpoint server ready on http://0.0.0.0:{port}\n"
        f"  Serving: {ckpt_path}\n"
        f"  Endpoints: /check, /checkpoint, /health\n"
        f"Forward the port (SSH / VS Code) and connect locally with --stream."
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Checkpoint server stopped by user.")
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Local-side remote reloader
# ---------------------------------------------------------------------------

class RemoteCheckpointReloader:
    """Polls a remote checkpoint server and hot-swaps agent weights.

    Instead of watching a local file (like :class:`_CheckpointReloader`),
    this class queries an HTTP endpoint that serves the latest
    checkpoint from the cluster.  The local viewer keeps running
    uninterrupted; when a newer checkpoint is available, the weights are
    fetched and swapped into ``agent.learner.state``.

    Used by :func:`src.test.run_live` when ``--stream`` is set.
    """

    def __init__(
        self,
        stream_url: str,
        agent,
        poll_interval: float,
    ):
        self._url = stream_url.rstrip("/")
        self._agent = agent
        self._poll_interval = poll_interval
        self._last_mtime: float | None = None
        self._last_check: float = 0.0

    @staticmethod
    def normalize_stream_url(raw: str) -> str:
        """Normalise the ``--stream`` value into a full ``http://`` URL.

        Accepts:
        - ``http://localhost:2324``
        - ``localhost:2324``
        - ``2324`` (host defaults to ``localhost``)
        """
        raw = raw.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw.rstrip("/")
        if ":" in raw:
            return f"http://{raw}".rstrip("/")
        return f"http://localhost:{raw}".rstrip("/")

    def maybe_reload(self):
        """Poll the remote server and hot-swap if a newer checkpoint exists."""
        if self._poll_interval <= 0:
            return

        now = time.monotonic()
        if now - self._last_check < self._poll_interval:
            return
        self._last_check = now

        try:
            check_url = f"{self._url}/check"
            logger.debug(f"Polling remote checkpoint: {check_url}")
            import urllib.request

            with urllib.request.urlopen(check_url, timeout=10) as resp:
                payload = json.loads(resp.read().decode())

            mtime = payload.get("mtime")
            if mtime is None:
                logger.debug("Remote server has no checkpoint file yet.")
                return

            mtime = float(mtime)
            if self._last_mtime is not None and mtime <= self._last_mtime:
                return  # no change

            logger.info(f"New checkpoint available (mtime={mtime}). Fetching...")
            with urllib.request.urlopen(
                f"{self._url}/checkpoint", timeout=30
            ) as resp:
                raw_bytes = resp.read()

            import cloudpickle

            new_state = cloudpickle.loads(raw_bytes)
            self._agent.learner.state = new_state
            self._last_mtime = mtime
            logger.success(
                f"Hot-swapped remote checkpoint "
                f"(mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})"
            )

        except Exception:
            logger.exception(
                "Failed to poll / fetch remote checkpoint - keeping old weights."
            )
