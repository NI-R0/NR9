"""
Implemented by: Jason Dietrich
Note: Implemented with the help of Claude!
"""

import http.server
import json
import os
import threading
import time
import urllib.request
import cloudpickle
from loguru import logger


def _checkpoint_path(load_dir: str, checkpoint_name: str) -> str:
    return os.path.join(load_dir, "checkpoints", f"{checkpoint_name}.pkl")


def _load_training_meta(load_dir: str) -> dict:
    """Read training_meta.json next to the checkpoints, if it exists.
    """
    meta_path = os.path.join(load_dir, "checkpoints", "training_meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


class _CheckpointHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler exposing /check, /checkpoint, and /health."""

    # Set as class attributes by ``serve()``.
    _checkpoint_path: str = ""
    _load_dir: str = ""

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
        meta = _load_training_meta(_CheckpointHandler._load_dir)
        self._json_response({
            "mtime": os.path.getmtime(path),
            "episode": meta.get("episode"),
            "best_eval_reward": meta.get("best_eval_reward"),
        })

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


def serve(args: dict, stats=None):
    """Start the headless checkpoint server.
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
    _CheckpointHandler._load_dir = args["load_dir"]

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
    """Polls a remote checkpoint server in a background thread and hot-swaps agent weights."""

    def __init__(
        self,
        stream_url: str,
        agent,
        poll_interval: float,
        on_swap=None,
    ):
        self._url = stream_url.rstrip("/")
        self._agent = agent
        self._poll_interval = poll_interval
        self._on_swap = on_swap
        self._last_mtime: float | None = None

        # Latest checkpoint metadata (episode, best_eval_reward) for display.
        self.checkpoint_episode: int | None = None
        self.checkpoint_reward: float | None = None

        # Inter-thread communication: the background thread stores a
        # freshly downloaded state here; the main thread picks it up.
        self._pending_state = None
        self._pending_mtime: float | None = None
        self._pending_episode: int | None = None
        self._pending_reward: float | None = None
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def normalize_stream_url(raw: str) -> str:
        """
        Normalize --stream to a full http:// URL (accepts full URL, host:port, or bare port)
        """
        raw = raw.strip()
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw.rstrip("/")
        if ":" in raw:
            return f"http://{raw}".rstrip("/")
        return f"http://localhost:{raw}".rstrip("/")

    def fetch_initial(self):
        """Synchronously fetch the first checkpoint from the server."""
        try:
            with urllib.request.urlopen(
                f"{self._url}/checkpoint", timeout=30
            ) as resp:
                raw_bytes = resp.read()
            new_state = cloudpickle.loads(raw_bytes)

            # Get mtime + metadata so the background thread knows the initial version
            try:
                with urllib.request.urlopen(
                    f"{self._url}/check", timeout=10
                ) as resp:
                    payload = json.loads(resp.read().decode())
                self._last_mtime = payload.get("mtime")
                self.checkpoint_episode = payload.get("episode")
                self.checkpoint_reward = payload.get("best_eval_reward")
            except Exception:
                pass

            logger.success("Fetched initial checkpoint from remote server.")
            return new_state
        except Exception:
            logger.exception("Failed to fetch initial checkpoint from remote server.")
            return None

    def start(self):
        """Start the background polling thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(
            f"Background checkpoint poller started: '{self._url}' "
            f"every {self._poll_interval:.1f}s"
        )

    def stop(self):
        """Signal the background thread to stop and wait for it."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

    def _poll_loop(self):
        """Background loop: poll /check, download /checkpoint if newer."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self._poll_interval)
            if self._stop_event.is_set():
                break

            try:
                with urllib.request.urlopen(
                    f"{self._url}/check", timeout=10
                ) as resp:
                    payload = json.loads(resp.read().decode())

                mtime = payload.get("mtime")
                if mtime is None:
                    logger.debug("Remote server has no checkpoint file yet.")
                    continue

                mtime = float(mtime)
                if self._last_mtime is not None and mtime <= self._last_mtime:
                    continue  # no change

                episode = payload.get("episode")
                reward = payload.get("best_eval_reward")

                logger.info(f"New checkpoint available (mtime={mtime}). Fetching...")
                with urllib.request.urlopen(
                    f"{self._url}/checkpoint", timeout=30
                ) as resp:
                    raw_bytes = resp.read()

                new_state = cloudpickle.loads(raw_bytes)

                with self._lock:
                    self._pending_state = new_state
                    self._pending_mtime = mtime
                    self._pending_episode = episode
                    self._pending_reward = reward

                logger.success(
                    f"Downloaded new checkpoint "
                    f"(mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))})"
                )

            except Exception:
                logger.exception(
                    "Failed to poll / fetch remote checkpoint - keeping old weights."
                )

    def maybe_reload(self):
        """Non-blocking swap of pending state if the background thread has a newer checkpoint."""
        if self._pending_state is None:
            return

        with self._lock:
            if self._pending_state is None:
                return
            self._agent.learner.state = self._pending_state
            self._last_mtime = self._pending_mtime
            self.checkpoint_episode = self._pending_episode
            self.checkpoint_reward = self._pending_reward
            self._pending_state = None
            self._pending_mtime = None
            self._pending_episode = None
            self._pending_reward = None

        logger.success("Hot-swapped remote checkpoint into agent.")

        if self._on_swap is not None:
            try:
                self._on_swap(self.checkpoint_episode, self.checkpoint_reward)
            except Exception:
                logger.exception("on_swap callback failed.")
