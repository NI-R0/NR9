"""Tests for the checkpoint server and remote reloader protocol.

These tests start the HTTP checkpoint server on a random port, write a
fake checkpoint file, and verify that ``RemoteCheckpointReloader``
correctly detects and fetches new versions — all in background threads
without blocking the caller.
"""

import cloudpickle
import http.server
import json
import os
import tempfile
import threading
import time

from src.serve import _CheckpointHandler, RemoteCheckpointReloader


def _start_server(checkpoint_path: str, load_dir: str = "", port: int = 0) -> tuple[http.server.HTTPServer, int, threading.Thread]:
    """Start the checkpoint server in a background thread.

    Returns ``(server, actual_port, thread)``.
    """
    _CheckpointHandler._checkpoint_path = checkpoint_path
    _CheckpointHandler._load_dir = load_dir
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), _CheckpointHandler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual_port, thread


def _make_fake_checkpoint(path: str, value: int):
    """Write a pickled object to *path*."""
    with open(path, "wb") as f:
        cloudpickle.dump({"value": value}, f)


class _FakeAgent:
    """Minimal stand-in for SoccerAgent with a learner.state attribute."""

    class _Learner:
        state = None

    def __init__(self):
        self.learner = self._Learner()


class TestCheckpointServer:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ckpt_path = os.path.join(self._tmpdir, "latest.pkl")
        self._server = None
        self._thread = None

    def teardown_method(self):
        if self._server is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)

    def test_health_endpoint(self):
        self._server, port, self._thread = _start_server(self._ckpt_path)
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
            assert resp.status == 200

    def test_check_returns_null_when_no_file(self):
        self._server, port, self._thread = _start_server(self._ckpt_path)
        import urllib.request
        import json

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/check", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        assert data["mtime"] is None

    def test_check_returns_mtime_when_file_exists(self):
        _make_fake_checkpoint(self._ckpt_path, 42)
        expected_mtime = os.path.getmtime(self._ckpt_path)
        self._server, port, self._thread = _start_server(self._ckpt_path)
        import urllib.request
        import json

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/check", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        assert data["mtime"] is not None
        assert abs(data["mtime"] - expected_mtime) < 1.0

    def test_checkpoint_endpoint_returns_bytes(self):
        _make_fake_checkpoint(self._ckpt_path, 99)
        self._server, port, self._thread = _start_server(self._ckpt_path)
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/checkpoint", timeout=5) as resp:
            raw = resp.read()
        loaded = cloudpickle.loads(raw)
        assert loaded == {"value": 99}

    def test_checkpoint_endpoint_404_when_no_file(self):
        self._server, port, self._thread = _start_server(self._ckpt_path)
        import urllib.request

        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/checkpoint", timeout=5)
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_check_returns_episode_and_reward_metadata(self):
        """When training_meta.json exists, /check returns episode + best_eval_reward."""
        _make_fake_checkpoint(self._ckpt_path, 42)
        # training_meta.json lives in <load_dir>/checkpoints/
        ckpt_dir = os.path.join(self._tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Move checkpoint into checkpoints/ so paths line up
        real_ckpt = os.path.join(ckpt_dir, "latest.pkl")
        os.replace(self._ckpt_path, real_ckpt)
        self._ckpt_path = real_ckpt
        meta = {"episode": 566, "best_eval_reward": 57.0, "agent_step_count": 12345}
        with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
            json.dump(meta, f)

        self._server, port, self._thread = _start_server(
            self._ckpt_path, load_dir=self._tmpdir
        )
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/check", timeout=5) as resp:
            data = json.loads(resp.read().decode())
        assert data["mtime"] is not None
        assert data["episode"] == 566
        assert data["best_eval_reward"] == 57.0


class TestRemoteCheckpointReloader:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ckpt_path = os.path.join(self._tmpdir, "latest.pkl")
        self._server = None
        self._thread = None
        self._reloader = None

    def teardown_method(self):
        if self._reloader is not None:
            self._reloader.stop()
        if self._server is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)

    def test_maybe_reload_noop_when_no_pending(self):
        """maybe_reload should be a no-op when nothing has been downloaded."""
        reloader = RemoteCheckpointReloader("http://127.0.0.1:9999", _FakeAgent(), 0.01)
        reloader.maybe_reload()  # should not raise

    def test_fetch_initial(self):
        _make_fake_checkpoint(self._ckpt_path, 42)
        # Set up training_meta.json so fetch_initial populates episode/reward.
        ckpt_dir = os.path.join(self._tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        real_ckpt = os.path.join(ckpt_dir, "latest.pkl")
        os.replace(self._ckpt_path, real_ckpt)
        self._ckpt_path = real_ckpt
        with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
            json.dump({"episode": 566, "best_eval_reward": 57.0}, f)

        self._server, port, self._thread = _start_server(
            self._ckpt_path, load_dir=self._tmpdir
        )

        agent = _FakeAgent()
        self._reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=1.0
        )
        state = self._reloader.fetch_initial()
        assert state == {"value": 42}
        assert agent.learner.state is None  # not swapped yet, just returned
        assert self._reloader.checkpoint_episode == 566
        assert self._reloader.checkpoint_reward == 57.0

    def test_fetch_initial_failure(self):
        # No server running, should return None.
        reloader = RemoteCheckpointReloader("http://127.0.0.1:9999", _FakeAgent(), 1.0)
        state = reloader.fetch_initial()
        assert state is None

    def test_background_swap_on_new_checkpoint(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        agent = _FakeAgent()
        self._reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.05
        )
        self._reloader.start()

        # Wait for background thread to download and then swap.
        self._wait_for_state(self._reloader, agent, expected={"value": 1}, timeout=5.0)
        assert agent.learner.state == {"value": 1}

    def test_background_swap_on_updated_checkpoint(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        agent = _FakeAgent()
        self._reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.05
        )
        self._reloader.start()

        self._wait_for_state(self._reloader, agent, expected={"value": 1}, timeout=5.0)

        # Write a new checkpoint with a different mtime.
        time.sleep(0.1)
        _make_fake_checkpoint(self._ckpt_path, 2)

        self._wait_for_state(self._reloader, agent, expected={"value": 2}, timeout=5.0)

    def test_on_swap_callback_called(self):
        """on_swap callback should fire with (episode, reward) after a swap."""
        _make_fake_checkpoint(self._ckpt_path, 1)
        # Set up training_meta.json so the server returns episode/reward.
        ckpt_dir = os.path.join(self._tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        real_ckpt = os.path.join(ckpt_dir, "latest.pkl")
        os.replace(self._ckpt_path, real_ckpt)
        self._ckpt_path = real_ckpt
        with open(os.path.join(ckpt_dir, "training_meta.json"), "w") as f:
            json.dump({"episode": 100, "best_eval_reward": 42.5}, f)

        self._server, port, self._thread = _start_server(
            self._ckpt_path, load_dir=self._tmpdir
        )

        swap_calls = []

        def on_swap(episode, reward):
            swap_calls.append((episode, reward))

        agent = _FakeAgent()
        self._reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.05, on_swap=on_swap
        )
        self._reloader.start()

        self._wait_for_state(self._reloader, agent, expected={"value": 1}, timeout=5.0)

        assert len(swap_calls) >= 1
        assert swap_calls[0] == (100, 42.5)

    def test_no_double_swap_when_unchanged(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        agent = _FakeAgent()
        self._reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.05
        )
        self._reloader.start()

        self._wait_for_state(self._reloader, agent, expected={"value": 1}, timeout=5.0)

        # Wait several poll cycles — file unchanged, state should stay.
        time.sleep(0.3)
        assert agent.learner.state == {"value": 1}

    def test_stop_joins_thread(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", _FakeAgent(), poll_interval=0.05
        )
        reloader.start()
        time.sleep(0.1)
        reloader.stop()
        assert reloader._thread is None

    def test_normalize_stream_url(self):
        assert RemoteCheckpointReloader.normalize_stream_url("2324") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("localhost:2324") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("http://localhost:2324") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("http://localhost:2324/") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("10.0.0.1:8080") == "http://10.0.0.1:8080"

    @staticmethod
    def _wait_for_state(reloader, agent, expected, timeout=5.0):
        """Poll maybe_reload until agent.learner.state == expected or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reloader.maybe_reload()
            time.sleep(0.02)
            if agent.learner.state == expected:
                return
        assert False, f"State never reached {expected}, got {agent.learner.state}"
