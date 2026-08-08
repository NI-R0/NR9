"""Parallel vectorized environment using multiprocessing + shared memory.

Each worker process owns a single ``dm_control`` environment instance.
Communication uses **pure shared memory** — no pipes for per-step data:

- **Shared-memory NumPy arrays** carry actions, observations, rewards,
  dones, terminal observations, AND command/status signals.
- Workers **poll a shared step_counter** that the master increments before
  each batch.  A single int64 write replaces 48× Event.set()/clear() calls.
- A fixed-size reward-components buffer stores per-step reward breakdowns
  directly in shared memory.  Keys are communicated once during init.

This eliminates ~2.6M pipe send/recv calls per 5-min run (72 envs x
18k steps x 2), reducing IPC overhead from ~164s to ~5-10s.
The Event-based signaling overhead (~970s per 25-min run for 48 envs) is
eliminated by step-counter polling (~0.5ms yield vs SemLock syscalls).
"""

import json
import os
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
    log_dir: str,
    use_icm: bool = False,
    icm_intrinsic_scale: float = 1.0,
    icm_lr: float = 5e-4,
    icm_hidden_sizes: tuple = (64, 32),
):
    """Worker process: owns one Environment, polls shared step counter for commands.

    No pipes, no per-worker Events.  Workers read a shared-memory
    ``step_counter`` that the master increments before each batch.
    This eliminates the 48× Event.set() overhead per step (~5.5ms saved).
    """
    import sys
    import os as _os
    import traceback as _tb
    import time as _time

    # ── Force JAX to CPU-only in worker processes ──
    # The main process holds the GPU; workers spawning with "spawn" inherit
    # CUDA context and exhaust GPU memory when JAX initializes (even for
    # jax.random.PRNGKey).  Setting this env var BEFORE any jax import
    # forces all JAX operations to CPU, which is what we want for the
    # lightweight ICM forward model.
    _os.environ["JAX_PLATFORM_NAME"] = "cpu"
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # ── Attach to shared memory FIRST so we can signal errors ──
    from multiprocessing import shared_memory as _shm_mod

    shm_objects = {k: _shm_mod.SharedMemory(name=v) for k, v in shm_names.items()}

    command_buf = np.ndarray(
        (num_envs,), dtype=np.int32, buffer=shm_objects["command"].buf
    )
    ready_buf = np.ndarray(
        (num_envs,), dtype=np.int32, buffer=shm_objects["ready"].buf
    )
    step_counter = np.ndarray(
        (1,), dtype=np.int64, buffer=shm_objects["step_counter"].buf
    )

    try:
        from src.environment import Environment

        env = Environment(
            domain_name=domain_name,
            task_name=task_name,
            max_steps=max_steps,
            use_icm=use_icm,
            icm_intrinsic_scale=icm_intrinsic_scale,
            icm_lr=icm_lr,
            icm_hidden_sizes=icm_hidden_sizes,
            icm_seed=seed + env_idx,
        )
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
        if use_icm:
            reward_keys.append("icm_reward")
        actual_num_keys = len(reward_keys)

        # Write reward keys into the key buffer (JSON string, null-terminated).
        key_shm = shm_objects["reward_comp_keys"]
        keys_json = json.dumps(reward_keys)
        key_array = np.ndarray(key_shm.size, dtype=np.uint8, buffer=key_shm.buf)
        encoded = keys_json.encode("utf-8")
        key_array[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

        # Signal to master that this worker is fully initialized and ready.
        ready_buf[env_idx] = 1

    except Exception:
        # Worker init failed -- write error to log file
        err_msg = (
            f"Worker {env_idx} failed to initialise: "
            f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}\n"
            f"{''.join(_tb.format_exception(*sys.exc_info()))}"
        )
        err_file = _os.path.join(log_dir, f"worker_{env_idx}_init_error.log")
        try:
            with open(err_file, "w") as f:
                f.write(err_msg)
        except Exception:
            pass
        print(err_msg, file=sys.stderr, flush=True)
        ready_buf[env_idx] = -1  # Signal failure to master
        sys.exit(1)

    try:
        last_seen_counter = -1

        while True:
            # ── Phase 1: Poll shared step_counter ─────────────────────
            # The master increments step_counter before writing new commands.
            # We poll until we see a new value, yielding CPU time when stuck.
            while True:
                current_counter = int(step_counter[0])
                if current_counter != last_seen_counter:
                    last_seen_counter = current_counter
                    break
                _time.sleep(0.0005)  # 0.5ms yield — dm_control physics takes ~200ms/step

            cmd = command_buf[env_idx]
            if cmd == _CMD_CLOSE:
                break

            # ── Phase 2: Execute command ─────────────────────────────
            if cmd == _CMD_STEP:
                action = action_buf[env_idx].copy()
                state, reward, done, info = env.step(action)

                # Reward components -- shared memory
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
            # else: spurious wake-up, re-block

    except KeyboardInterrupt:
        pass
    except Exception:
        # Runtime crash -- log it so the master can diagnose timeouts
        err_msg = (
            f"Worker {env_idx} runtime crash: "
            f"{type(sys.exc_info()[1]).__name__}: {sys.exc_info()[1]}\n"
            f"{''.join(_tb.format_exception(*sys.exc_info()))}"
        )
        err_file = _os.path.join(log_dir, f"worker_{env_idx}_error.log")
        try:
            with open(err_file, "w") as f:
                f.write(err_msg)
        except Exception:
            pass
        print(err_msg, file=sys.stderr, flush=True)
        raise
    finally:
        for shm in shm_objects.values():
            try:
                shm.close()
            except Exception:
                pass


class ParallelVectorEnv:
    """Runs ``num_envs`` dm_control environments in separate processes.

    Pure shared-memory communication with **step-counter polling**.
    Workers read a shared ``step_counter`` that the master increments
    before each batch.  This eliminates the per-worker Event overhead
    (~5.5ms/step for 48× Event.set/clear).
    """

    # Upper bound on reward-component keys per environment.
    # walker_3D_ball/kick has ~15 keys; raise if needed.
    MAX_REWARD_KEYS = 32

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        max_steps: int,
        num_envs: int,
        seed: int = 42,
        use_icm: bool = False,
        icm_intrinsic_scale: float = 1.0,
        icm_lr: float = 5e-4,
        icm_hidden_sizes: tuple[int, ...] = (64, 32),
    ):
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
        self._use_icm = use_icm

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
            "step_counter":   8,                   # int64 — single shared counter
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
        self._step_counter = np.ndarray(
            (1,), dtype=np.int64,
            buffer=self._shm_segments["step_counter"].buf,
        )
        self._step_counter[0] = 0

        # Read reward keys (written by worker 0 during init).
        self._reward_keys: list[str] = []

        # ── Spawn worker processes ─────────────────────────────────────
        self.processes: list[mp.Process] = []

        # Log directory for worker init errors (spawn context doesn't support
        # stdout/stderr redirection, so workers write to files directly).
        self._worker_log_dir = f"/tmp/vector_env_logs_{os.getpid()}"
        os.makedirs(self._worker_log_dir, exist_ok=True)

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
                    self._worker_log_dir,
                    use_icm,
                    icm_intrinsic_scale,
                    icm_lr,
                    icm_hidden_sizes,
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
            ready_mask = self._ready_buf[: self.num_envs] == 1
            if np.all(ready_mask):
                break
            failed_mask = self._ready_buf[: self.num_envs] == -1
            failed_count = int(np.sum(failed_mask))
            if failed_count > 0:
                # Read error logs from failed workers
                failed_workers = np.where(failed_mask)[0].tolist()
                error_details = []
                for w in failed_workers:
                    log_path = os.path.join(self._worker_log_dir, f"worker_{w}_init_error.log")
                    try:
                        with open(log_path) as f:
                            error_details.append(f"Worker {w}: {f.read().strip()}")
                    except Exception:
                        error_details.append(f"Worker {w}: (no log found)")
                msg = (
                    f"{failed_count} worker(s) failed to initialise: "
                    + "; ".join(error_details)
                )
                raise RuntimeError(msg) from None
            elapsed = time.monotonic() - (deadline - 120.0)
            ready_count = int(np.sum(ready_mask))
            if int(elapsed) % 5 == 0:
                logger.info(
                    f"Waiting for workers to initialise: {ready_count}/{num_envs} ready "
                    f"({elapsed:.1f}s elapsed)"
                )
            time.sleep(0.05)
        else:
            not_ready = np.where(self._ready_buf[: self.num_envs] != 1)[0]
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

    def _signal_all(self) -> None:
        """Wake all workers by incrementing the shared step counter.

        Workers poll ``step_counter`` and wake when they see a new value.
        This replaces 48× Event.set()/clear() (SemLock syscalls) with a
        single shared-memory int64 write.
        """
        self._step_counter[0] += 1

    def _wait_all_ready(self, timeout: float = 60.0) -> None:
        """Wait until all workers set their ready flag.

        Uses a lightweight busy-loop with 1ms yield when stuck.  Workers
        should respond quickly after detecting the step_counter change.
        """
        import time
        deadline = time.monotonic() + timeout
        ready_count = 0
        while time.monotonic() < deadline:
            if np.all(self._ready_buf[: self.num_envs]):
                return
            # Only sleep if NO progress has been made for a while
            new_ready = int(np.sum(self._ready_buf[: self.num_envs]))
            if new_ready == ready_count:
                time.sleep(0.001)  # 1ms yield only when stuck
            ready_count = new_ready
        not_ready = np.where(self._ready_buf[: self.num_envs] == 0)[0]

        # Diagnose: check process states for hung workers
        diag_parts = []
        for idx in not_ready:
            p = self.processes[idx]
            exit_code = p.exit_code
            alive = p.is_alive()
            diag_parts.append(
                f"Worker[{idx}]: alive={alive}, exit_code={exit_code}, pid={p.pid}"
            )

        # Check for worker error logs (runtime crashes)
        import glob as _glob
        runtime_errors = []
        for logfile in _glob.glob(
            os.path.join(self._worker_log_dir, "worker_*_error.log")
        ):
            try:
                with open(logfile) as f:
                    runtime_errors.append(f.read().strip())
            except Exception:
                pass

        detail = "; ".join(diag_parts)
        if runtime_errors:
            detail += "\nWorker error logs:\n" + "\n".join(runtime_errors)

        raise TimeoutError(
            f"Workers {not_ready.tolist()} did not become ready "
            f"within {timeout:.1f}s. Diagnostics: {detail}"
        )

    def _clear_ready(self) -> None:
        """Clear ready flags and reset commands to IDLE for all envs."""
        self._command_buf[: self.num_envs] = _CMD_IDLE
        self._ready_buf[: self.num_envs] = 0

    # ── Public API ─────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations."""
        self._command_buf[: self.num_envs] = _CMD_RESET
        self._signal_all()
        self._wait_all_ready()
        self._command_buf[: self.num_envs] = _CMD_IDLE  # Acknowledge workers
        return self._next_state_buf.copy()

    def step(self, actions: np.ndarray):
        """Step all environments with the given batched actions.

        Returns:
            next_states: (N, state_dim)
            rewards: (N,)
            dones: (N,)  -- bool array
            infos: list[dict]  -- reward_components + terminal_obs
        """
        self._action_buf[:] = actions
        self._command_buf[: self.num_envs] = _CMD_STEP
        self._signal_all()          # wake all workers
        self._wait_all_ready()      # block until all done
        self._clear_ready()         # clear for next step

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
        self._signal_all()
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