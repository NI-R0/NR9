"""Tests for the checkpoint server and remote reloader protocol.

These tests start the HTTP checkpoint server on a random port, write a
fake checkpoint file, and verify that ``RemoteCheckpointReloader``
correctly detects and fetches new versions.
"""

import cloudpickle
import http.server
import os
import tempfile
import threading
import time

from src.serve import _CheckpointHandler, RemoteCheckpointReloader


def _start_server(checkpoint_path: str, port: int = 0) -> tuple[http.server.HTTPServer, int, threading.Thread]:
    """Start the checkpoint server in a background thread.

    Returns ``(server, actual_port, thread)``.
    """
    _CheckpointHandler._checkpoint_path = checkpoint_path
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


class TestRemoteCheckpointReloader:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ckpt_path = os.path.join(self._tmpdir, "latest.pkl")
        self._server = None
        self._thread = None

    def teardown_method(self):
        if self._server is not None:
            self._server.shutdown()
            self._thread.join(timeout=2)

    def test_no_swap_when_poll_interval_zero(self):
        reloader = RemoteCheckpointReloader("http://127.0.0.1:9999", _FakeAgent(), 0.0)
        # Should be a no-op, no exception even though server doesn't exist.
        reloader.maybe_reload()

    def test_swap_on_new_checkpoint(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        agent = _FakeAgent()
        reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.01
        )
        reloader.maybe_reload()
        assert agent.learner.state == {"value": 1}

    def test_no_swap_when_unchanged(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        agent = _FakeAgent()
        reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.01
        )
        reloader.maybe_reload()
        assert agent.learner.state == {"value": 1}

        # Second poll — file unchanged, should NOT re-fetch.
        # (We can't easily verify "no fetch" without mocking, but state
        # should remain the same.)
        time.sleep(0.02)
        reloader.maybe_reload()
        assert agent.learner.state == {"value": 1}

    def test_swap_on_updated_checkpoint(self):
        _make_fake_checkpoint(self._ckpt_path, 1)
        self._server, port, self._thread = _start_server(self._ckpt_path)

        agent = _FakeAgent()
        reloader = RemoteCheckpointReloader(
            f"http://127.0.0.1:{port}", agent, poll_interval=0.01
        )
        reloader.maybe_reload()
        assert agent.learner.state == {"value": 1}

        # Write a new checkpoint with a different mtime.
        time.sleep(0.05)
        _make_fake_checkpoint(self._ckpt_path, 2)

        time.sleep(0.02)
        reloader.maybe_reload()
        assert agent.learner.state == {"value": 2}

    def test_normalize_stream_url(self):
        assert RemoteCheckpointReloader.normalize_stream_url("2324") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("localhost:2324") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("http://localhost:2324") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("http://localhost:2324/") == "http://localhost:2324"
        assert RemoteCheckpointReloader.normalize_stream_url("10.0.0.1:8080") == "http://10.0.0.1:8080"
