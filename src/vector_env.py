"""Parallel vectorized environment using multiprocessing + shared memory.

Each worker process owns a single ``dm_control`` environment instance.
Communication uses **pure shared memory** — no pipes for per-step data:

- **Shared-memory NumPy arrays** carry actions, observations, rewards,
  dones, terminal observations, AND command/status signals.
- Workers poll their command slot; master sets ``CMD_STEP`` and waits
  for all workers to set ``ready``.  No pickling, no pipe I/O per step.
- A fixed-size reward-components buffer stores per-step reward breakdowns
  directly in shared memory.  Keys are communicated once during init.

This eliminates ~2.6M pipe send/recv calls per 5-min run (72 envs ×
18k steps × 2), reducing IPC overhead from ~164s to ~5-10s.
"""

import json
import struct
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from loguru import logger

# Command codes stored in shared memory (int32).
_CMD_IDLE = 0
_CMD_STEP = 1
_CMD_RESET = 2
_CMD_CLOSE = 3


def _worker_fn(
    env_idx: int,
    domain_name: str,
    task_name: str,
    max_steps: int,
    seed: int,
    shm_names: dict,
    state_dim: int,
    action_dim: int,
    num_envs: int,
    num_reward_keys: int,
):
    """Worker process: owns one Environment, polls shared-memory command slot.

    No pipes — the worker reads its command from ``command_buf[env_idx]``,
    performs the action, writes results to shared memory, and sets
    ``ready_buf[env_idx] = True``.  The master clears the ready flag
    before issuing the next command.
    """
    import sys
    import traceback as _tb

    # ── Attach to shared memory FIRST so we can signal errors ──
    from multiprocessing import shared_memory as _shm_mod

    shm_objects = {k: _shm_mod.SharedMemory(name=v) for k, v in shm_names.items()}

    command_buf = np.ndarray(
        (num_envs,), dtype=np.int32, buffer=shm_objects["command"].buf
    )
    ready_buf = np.ndarray(
        (num_envs,), dtype=np.int32, buffer=shm_objects["ready"].buf
    )

    try:
        from src.environment import Environment

        env = Environment(domain_name=domain_name, task_name=task_name, max_steps=max_steps)
        np.random.seed(seed)

        # Data buffers (same layout as main process).
        action_buf = np.ndarray(
            (num_envs, action_dim), dtype=np.float32, buffer=shm_objects["action"].buf
        )
        next_state_buf = np.ndarray(
            (num_envs, state_dim), dtype=np.float32, buffer=shm_objects["next_state"].buf
        )
        reward_buf = np.ndarray(
            (num_envs,), dtype=np.float32, buffer=shm_objects["reward"].buf
        )
        done_buf = np.ndarray(
            (num_envs,), dtype=np.int64, buffer=shm_objects["done"].buf
        )
        terminal_obs_buf = np.ndarray(
            (num_envs, state_dim), dtype=np.float32, buffer=shm_objects["terminal_obs"].buf
        )
        reward_comp_buf = np.ndarray(
            (num_envs, num_reward_keys), dtype=np.float32, buffer=shm_objects["reward_comp"].buf
        )

        # Discover reward component keys from the environment (done once).
        reward_keys: list[str] = []
        _ = env.reset()
        _action = np.zeros(env.action_spec.shape, dtype=np.float32)
        _timestep = env.step(_action)
        task = getattr(env, "task", None)
        if task is not None and hasattr(task, "_reward_components"):
            reward_keys = list(task._reward_components.keys())
        actual_num_keys = len(reward_keys)

        # Write reward keys into the key buffer (JSON string, null-terminated).
        key_shm = shm_objects["reward_comp_keys"]
        keys_json = json.dumps(reward_keys)
        key_array = np.ndarray(key_shm.size, dtype=np.uint8, buffer=key_shm.buf)
        encoded = keys_json.encode("utf-8")
        key_array[: len(encoded)] = encoded

        # Signal to master that this worker is fully initialized and ready.
        ready_buf[env_idx] = 1

    except Exception:
        # Worker init failed — write error info to stderr so it shows up
        # in the job log, then exit to let master detect us as dead.
        err_msg = (
            f"Worker {env_idx} failed to initialise: "
            f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}\n"
            f"{''.join(_tb.format_exception(*sys.exc_info()))}"
        )
        # Write to a file so master can read it if needed.
        err_file = f"/tmp/worker_{env_idx}_init_error.log"
        try:
            with open(err_file, "w") as f:
                f.write(err_msg)
        except Exception:
            pass
        print(err_msg, file=sys.stderr, flush=True)
        ready_buf[env_idx] = -1  # Signal failure to master
        sys.exit(1)

    try:
        import time as _time

        while True:
            # ── Phase 1: Wait for a non-IDLE command ──────────────────
            # Re-read command from shared memory every iteration so we
            # don't spin on a stale local copy.
            cmd = command_buf[env_idx]
            if cmd == _CMD_IDLE:
                _time.sleep(0.0001)  # 0.1ms yield — prevents 100% CPU spin
                continue
            if cmd == _CMD_CLOSE:
                break

            # ── Phase 2: Execute command ──────────────────────────────
            if cmd == _CMD_STEP:
                action = action_buf[env_idx].copy()
                state, reward, done, info = env.step(action)

                # Reward components → shared memory
                if "reward_components" in info and actual_num_keys > 0:
                    rc = info["reward_components"]
                    for ki, key in enumerate(reward_keys):
                        if ki < num_reward_keys:
                            reward_comp_buf[env_idx, ki] = float(rc.get(key, 0.0))
                else:
                    reward_comp_buf[env_idx, :] = 0.0

                if done:
                    terminal_obs_buf[env_idx] = state
                    state = env.reset()

                next_state_buf[env_idx] = state
                reward_buf[env_idx] = reward
                done_buf[env_idx] = int(done)
                ready_buf[env_idx] = 1
            elif cmd == _CMD_RESET:
                state = env.reset()
                next_state_buf[env_idx] = state
                ready_buf[env_idx] = 1
            else:
                continue  # Unknown command, re-poll

            # ── Phase 3: Wait for master to acknowledge (IDLE) ───────
            # Master reads data, sets command back to IDLE.  This
            # prevents the worker from re-executing the same command.
            while command_buf[env_idx] != _CMD_IDLE:
                pass  # Tight spin — master resets IDLE quickly
            ready_buf[env_idx] = 0

    except KeyboardInterrupt:
        pass
    finally:
        for shm in shm_objects.values():
            try:
                shm.close()
            except Exception:
                pass


class ParallelVectorEnv:
    """Runs ``num_envs`` dm_control environments in separate processes.

    Pure shared-memory communication: commands, status flags, and all
    data arrays live in shared memory.  No per-step pipe I/O.
    """

    # Upper bound on reward-component keys per environment.
    # walker_3D_ball/kick has ~15 keys; raise if needed.
    MAX_REWARD_KEYS = 32

    def __init__(self, domain_name: str, task_name: str, max_steps: int,
                 num_envs: int, seed: int = 42):
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")

        # Probe state/action dims from a temporary env.
        from src.environment import Environment
        probe = Environment(domain_name=domain_name, task_name=task_name,
                            max_steps=max_steps)
        state_dim = int(np.prod(probe.state_dim))
        action_dim = int(np.prod(probe.action_dim))
        del probe

        self._state_dim = state_dim
        self._action_dim = action_dim

        # ── Shared memory allocations ──────────────────────────────────
        shm_specs = {
            "action":         num_envs * action_dim * 4,
            "next_state":     num_envs * state_dim * 4,
            "reward":         num_envs * 4,
            "done":           num_envs * 8,        # int64
            "terminal_obs":   num_envs * state_dim * 4,
            "command":        num_envs * 4,        # int32
            "ready":          num_envs * 4,        # int32
            "reward_comp":    num_envs * self.MAX_REWARD_KEYS * 4,
        }
        self._shm_segments: dict[str, shared_memory.SharedMemory] = {}
        for name, size in shm_specs.items():
            self._shm_segments[name] = shared_memory.SharedMemory(
                create=True, size=size
            )

        # Reward-component keys buffer (text).
        self._shm_segments["reward_comp_keys"] = shared_memory.SharedMemory(
            create=True, size=1024
        )

        shm_names = {k: v.name for k, v in self._shm_segments.items()}

        # ── NumPy views into shared memory (main process) ──────────────
        self._action_buf = np.ndarray(
            (num_envs, action_dim), dtype=np.float32,
            buffer=self._shm_segments["action"].buf,
        )
        self._next_state_buf = np.ndarray(
            (num_envs, state_dim), dtype=np.float32,
            buffer=self._shm_segments["next_state"].buf,
        )
        self._reward_buf = np.ndarray(
            (num_envs,), dtype=np.float32,
            buffer=self._shm_segments["reward"].buf,
        )
        self._done_buf = np.ndarray(
            (num_envs,), dtype=np.int64,
            buffer=self._shm_segments["done"].buf,
        )
        self._terminal_obs_buf = np.ndarray(
            (num_envs, state_dim), dtype=np.float32,
            buffer=self._shm_segments["terminal_obs"].buf,
        )
        self._command_buf = np.ndarray(
            (num_envs,), dtype=np.int32,
            buffer=self._shm_segments["command"].buf,
        )
        self._ready_buf = np.ndarray(
            (num_envs,), dtype=np.int32,
            buffer=self._shm_segments["ready"].buf,
        )
        self._reward_comp_buf = np.ndarray(
            (num_envs, self.MAX_REWARD_KEYS), dtype=np.float32,
            buffer=self._shm_segments["reward_comp"].buf,
        )

        # Read reward keys (written by worker 0 during init).
        self._reward_keys: list[str] = []

        # ── Spawn worker processes ─────────────────────────────────────
        self.processes: list[mp.Process] = []
        for i in range(num_envs):
            p = ctx.Process(
                target=_worker_fn,
                args=(
                    i,
                    domain_name,
                    task_name,
                    max_steps,
                    seed + i,
                    shm_names,
                    state_dim,
                    action_dim,
                    num_envs,
                    self.MAX_REWARD_KEYS,
                ),
                daemon=True,
            )
            p.start()
            self.processes.append(p)

        # Wait for ALL workers to initialise (they set ready_buf[env_idx] = 1
        # after Environment creation + reward key discovery).  Each worker
        # needs ~1-3s for spawn + dm_control import + env creation.
        import time
        deadline = time.monotonic() + 120.0  # 2 min for 48 workers
        while time.monotonic() < deadline:
            if np.all(self._ready_buf[: self.num_envs]):
                break
            elapsed = time.monotonic() - (deadline - 120.0)
            ready_count = int(np.sum(self._ready_buf[: self.num_envs]))
            if int(elapsed) % 5 == 0:
                logger.info(
                    f"Waiting for workers to initialise: {ready_count}/{num_envs} ready "
                    f"({elapsed:.1f}s elapsed)"
                )
            time.sleep(0.05)
        else:
            not_ready = np.where(self._ready_buf[: self.num_envs] == 0)[0]
            raise TimeoutError(
                f"Workers {not_ready.tolist()} did not become ready within 120.0s. "
                f"Check that dm_control and the environment can be loaded."
            )

        # Clear ready flags now that all workers are confirmed initialised.
        self._ready_buf[:] = 0

        # Read reward keys from worker 0.
        key_buf = np.ndarray(
            self._shm_segments["reward_comp_keys"].size,
            dtype=np.uint8,
            buffer=self._shm_segments["reward_comp_keys"].buf,
        )
        key_bytes = bytes(key_buf)
        null_idx = key_bytes.find(b"\x00")
        if null_idx >= 0:
            key_bytes = key_bytes[:null_idx]
        if key_bytes:
            self._reward_keys = json.loads(key_bytes.decode("utf-8"))

        self.state_dim = (state_dim,)
        self.action_dim = (action_dim,)
        self.action_min = None
        self.action_max = None

        logger.debug(
            f"ParallelVectorEnv initialized: num_envs={num_envs}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"reward_keys={self._reward_keys}"
        )

    # ── Synchronisation helpers ────────────────────────────────────────

    def _wait_all_ready(self, timeout: float = 60.0) -> None:
        """Busy-wait until all workers set their ready flag.

        Uses a tight loop with a short sleep to avoid wasting CPU, but
        checks frequently enough for low latency.
        """
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if np.all(self._ready_buf[: self.num_envs]):
                return
            time.sleep(0.0001)  # 0.1ms — tight enough for responsiveness
        not_ready = np.where(self._ready_buf[: self.num_envs] == 0)[0]
        raise TimeoutError(
            f"Workers {not_ready.tolist()} did not become ready "
            f"within {timeout:.1f}s"
        )

    def _clear_ready(self) -> None:
        """Clear ready flags and reset commands to IDLE for all envs.

        Must set command to IDLE *before* clearing ready, so that workers
        blocked in Phase 3 (waiting for IDLE after step/reset) can proceed.
        """
        self._command_buf[: self.num_envs] = _CMD_IDLE
        self._ready_buf[: self.num_envs] = 0

    # ── Public API ─────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations."""
        self._command_buf[: self.num_envs] = _CMD_RESET
        self._wait_all_ready()
        self._command_buf[: self.num_envs] = _CMD_IDLE  # Acknowledge workers
        return self._next_state_buf.copy()

    def step(self, actions: np.ndarray):
        """Step all environments with the given batched actions.

        Returns:
            next_states: (N, state_dim)
            rewards: (N,)
            dones: (N,)  — bool array
            infos: list[dict]  — reward_components + terminal_obs
        """
        self._action_buf[:] = actions
        self._command_buf[: self.num_envs] = _CMD_STEP
        self._wait_all_ready()
        self._clear_ready()

        next_states = self._next_state_buf.copy()
        rewards = self._reward_buf.copy()
        dones = self._done_buf[: self.num_envs].astype(np.bool_)

        # Build info dicts from shared-memory reward components.
        infos: list[dict] = []
        for i in range(self.num_envs):
            info: dict = {}
            if self._reward_keys:
                rc = {}
                for ki, key in enumerate(self._reward_keys):
                    if ki < self.MAX_REWARD_KEYS:
                        val = self._reward_comp_buf[i, ki]
                        if val != 0.0:
                            rc[key] = val
                if rc:
                    info["reward_components"] = rc
            if dones[i]:
                info["terminal_obs"] = self._terminal_obs_buf[i].copy()
            infos.append(info)

        return next_states, rewards, dones, infos

    def close(self) -> None:
        """Signal all workers to shut down and join them."""
        self._command_buf[: self.num_envs] = _CMD_CLOSE
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)

        # Unlink shared-memory segments.
        for shm in self._shm_segments.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass