import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from multiprocessing.connection import wait, Connection
from loguru import logger


def _worker_fn(remote, parent_remote, domain_name, task_name, max_steps, seed,
               shm_names, state_dim, action_dim, num_envs, env_idx):
    """Worker process: owns one Environment, handles step/reset commands."""
    parent_remote.close()

    from src.environment import Environment
    from multiprocessing import shared_memory as _shm_mod

    env = Environment(domain_name=domain_name, task_name=task_name, max_steps=max_steps)
    np.random.seed(seed)

    # Attach to existing shared-memory segments by name.
    shm_objects = {k: _shm_mod.SharedMemory(name=v) for k, v in shm_names.items()}

    # Reconstruct NumPy views into the shared-memory buffers.
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
        (num_envs,), dtype=np.bool_, buffer=shm_objects["done"].buf
    )
    terminal_obs_buf = np.ndarray(
        (num_envs, state_dim), dtype=np.float32, buffer=shm_objects["terminal_obs"].buf
    )

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                action = action_buf[env_idx].copy()
                state, reward, done, info = env.step(action)
                if done:
                    # Write terminal obs to shared memory; signal via info
                    # without the large array (avoids pickling over pipe).
                    terminal_obs_buf[env_idx] = state
                    info["terminal_obs"] = True  # flag, not the array
                    state = env.reset()
                else:
                    info["terminal_obs"] = False
                next_state_buf[env_idx] = state
                reward_buf[env_idx] = reward
                done_buf[env_idx] = done
                remote.send(("step_done", info))
            elif cmd == "reset":
                state = env.reset()
                next_state_buf[env_idx] = state
                remote.send(("reset_done", None))
            elif cmd == "get_spaces":
                remote.send((env.state_dim, env.action_dim, env.action_spec.minimum,
                             env.action_spec.maximum))
            elif cmd == "close":
                remote.close()
                break
            else:
                raise ValueError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        for shm in shm_objects.values():
            try:
                shm.close()
            except Exception:
                pass
        remote.close()


class ParallelVectorEnv:
    """Runs n=num_envs dm_control environments in separate processes."""

    def __init__(self, domain_name: str, task_name: str, max_steps: int,
                 num_envs: int, seed: int = 42):
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        self.remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.Process] = []

        # Probe state/action dims from a temporary env.
        from src.environment import Environment
        probe = Environment(domain_name=domain_name, task_name=task_name,
                            max_steps=max_steps)
        state_dim = int(np.prod(probe.state_dim))
        action_dim = int(np.prod(probe.action_dim))
        del probe

        self._state_dim = state_dim
        self._action_dim = action_dim

        # Allocate shared-memory buffers for inter-process data transfer.
        shm_specs = {
            "action": (num_envs * action_dim * 4, (num_envs, action_dim), np.float32),
            "next_state": (num_envs * state_dim * 4, (num_envs, state_dim), np.float32),
            "reward": (num_envs * 4, (num_envs,), np.float32),
            "done": (num_envs * 1, (num_envs,), np.bool_),
            "terminal_obs": (num_envs * state_dim * 4, (num_envs, state_dim), np.float32),
        }
        self._shm_segments: dict[str, shared_memory.SharedMemory] = {}
        for name, (size, _, _) in shm_specs.items():
            self._shm_segments[name] = shared_memory.SharedMemory(create=True, size=size)

        # Dict of segment names for passing to worker processes.
        shm_names = {k: v.name for k, v in self._shm_segments.items()}

        # Local NumPy views into shared memory (main process side).
        self._action_buf = np.ndarray(
            shm_specs["action"][1], dtype=shm_specs["action"][2], buffer=self._shm_segments["action"].buf
        )
        self._next_state_buf = np.ndarray(
            shm_specs["next_state"][1], dtype=shm_specs["next_state"][2], buffer=self._shm_segments["next_state"].buf
        )
        self._reward_buf = np.ndarray(
            shm_specs["reward"][1], dtype=shm_specs["reward"][2], buffer=self._shm_segments["reward"].buf
        )
        self._done_buf = np.ndarray(
            shm_specs["done"][1], dtype=shm_specs["done"][2], buffer=self._shm_segments["done"].buf
        )
        self._terminal_obs_buf = np.ndarray(
            shm_specs["terminal_obs"][1],
            dtype=shm_specs["terminal_obs"][2],
            buffer=self._shm_segments["terminal_obs"].buf
        )

        self._remote_to_idx: dict[Connection, int] = {}
        for i in range(num_envs):
            parent_remote, child_remote = ctx.Pipe()
            p = ctx.Process(
                target=_worker_fn,
                args=(child_remote, parent_remote, domain_name, task_name,
                      max_steps, seed + i,
                      shm_names, state_dim, action_dim, num_envs, i),
                daemon=True,
            )
            p.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self._remote_to_idx[parent_remote] = i
            self.processes.append(p)

        self.state_dim = (state_dim,)
        self.action_dim = (action_dim,)
        self.action_min = None
        self.action_max = None

        logger.debug(
            f"ParallelVectorEnv initialized: num_envs={num_envs}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}"
        )

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations."""
        for remote in self.remotes:
            remote.send(("reset", None))
        wait([r for r in self.remotes])
        for remote in self.remotes:
            remote.recv()  # ("reset_done", None)
        return self._next_state_buf.copy()

    def step(self, actions: np.ndarray):
        """
        Step all environments with the given batched actions.
        Writes actions into the shared action buffer, signals all workers
        to step in parallel, then collects results from shared memory.
        """
        self._action_buf[:] = actions

        # Fire all step commands in parallel
        for remote in self.remotes:
            remote.send(("step", None))

        # Collect responses in parallel using wait(), then map back to env index
        infos: list[dict] = [None] * self.num_envs  # type: ignore[assignment]
        remaining = list(self.remotes)
        while remaining:
            ready = wait(remaining, timeout=None)
            for remote in ready:
                _, info = remote.recv()
                idx = self._remote_to_idx[remote]
                infos[idx] = info
                remaining.remove(remote)

        next_states = self._next_state_buf.copy()
        rewards = self._reward_buf.copy()
        dones = self._done_buf.copy()

        # Attach terminal_obs from shared memory for envs that terminated.
        for i, info in enumerate(infos):
            if info.pop("terminal_obs", False):
                info["terminal_obs"] = self._terminal_obs_buf[i].copy()

        return next_states, rewards, dones, infos

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for remote in self.remotes:
            remote.close()

        # Unlink shared-memory segments.
        for shm in self._shm_segments.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
