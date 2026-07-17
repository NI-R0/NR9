"""Parallel vectorized environment using multiprocessing.

Each worker process owns a single ``dm_control`` environment instance.
The main process sends batched actions and receives batched results via
pipes.  When an env terminates it auto-resets and returns the terminal
observation in ``info["terminal_obs"]`` so the caller can store it in the
replay buffer before using the new observation for the next step.
"""

import numpy as np
import multiprocessing as mp
from loguru import logger


def _worker_fn(remote, parent_remote, domain_name, task_name, max_steps, seed):
    """Worker process: owns one Environment, handles step/reset commands."""
    parent_remote.close()

    # Import here so each process gets its own MuJoCo state.
    from src.environment import Environment

    env = Environment(domain_name=domain_name, task_name=task_name, max_steps=max_steps)
    np.random.seed(seed)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                state, reward, done, info = env.step(data)
                if done:
                    # Store terminal observation for the buffer, then auto-reset.
                    info["terminal_obs"] = state
                    state = env.reset()
                remote.send((state, reward, done, info))
            elif cmd == "reset":
                state = env.reset()
                remote.send(state)
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
        remote.close()


class ParallelVectorEnv:
    """Runs ``num_envs`` dm_control environments in separate processes.

    All environments share the same domain/task but are otherwise
    independent (different random seeds, auto-reset on done).
    """

    def __init__(self, domain_name: str, task_name: str, max_steps: int,
                 num_envs: int, seed: int = 42):
        self.num_envs = num_envs
        ctx = mp.get_context("spawn")
        self.remotes: list[mp.connection.Connection] = []
        self.processes: list[mp.Process] = []

        for i in range(num_envs):
            parent_remote, child_remote = ctx.Pipe()
            p = ctx.Process(
                target=_worker_fn,
                args=(child_remote, parent_remote, domain_name, task_name,
                      max_steps, seed + i),
                daemon=True,
            )
            p.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(p)

        # Query state/action dims from first worker.
        self.remotes[0].send(("get_spaces", None))
        self.state_dim, self.action_dim, self.action_min, self.action_max = \
            self.remotes[0].recv()

        logger.debug(
            f"ParallelVectorEnv initialized: num_envs={num_envs}, "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}"
        )

    def reset(self) -> np.ndarray:
        """Reset all environments and return stacked observations."""
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs).astype(np.float32)

    def step(self, actions: np.ndarray):
        """Step all environments with the given batched actions.

        Returns:
            next_states: (N, state_dim) — observation for the *next* step
                         (auto-reset obs if the env was done).
            rewards: (N,)
            dones: (N,)
            infos: list[dict] — ``info["terminal_obs"]`` present when done.
        """
        for i, remote in enumerate(self.remotes):
            remote.send(("step", actions[i]))

        results = [remote.recv() for remote in self.remotes]
        next_states = np.stack([r[0] for r in results]).astype(np.float32)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=bool)
        infos = [r[3] for r in results]
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
