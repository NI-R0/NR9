"""Replay buffer that stores n-step transitions.

Incoming 1-step transitions are accumulated into n-step transitions
following the ``rlax.n_step_bootstrapped_returns`` convention:

    state_t, action_t, sum_{i=0}^{n-1} gamma^i * r_{t+i},
    next_state_{t+n}, done_{t+n}, discount_{t+n}

The buffer keeps a rolling window of the last ``n_step`` raw
transitions per episode trajectory.  As soon as a full n-step window is
available, the aggregated transition is committed to the circular
replay storage.  When ``done`` is encountered, all remaining partial
windows are flushed (with appropriate discounting and done flags).

**Vectorized batch operations** are used for ``add_many`` to avoid
Python-loop overhead when feeding from 48+ parallel envs.
"""

import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger


class NStepTransitionBuffer:
    def __init__(self, state_shape: tuple[int], action_shape: tuple[int],
                 capacity: int = 100_000, n_step: int = 5, gamma: float = 0.99):
        self._capacity = capacity
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._state_dim = int(np.prod(state_shape))
        self._action_dim = int(np.prod(action_shape))
        self._n_step = n_step
        self._gamma = gamma
        self._size = 0
        self._pos = 0

        # Precompute gamma powers for discounted reward aggregation.
        self._gamma_powers = np.power(gamma, np.arange(n_step + 1), dtype=np.float32)

        # ── Circular storage (contiguous arrays) ──────────────────────
        self._states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, self._action_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._discounts = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        self._num_envs = 1
        self._windows: list[list[dict]] = [[]]

        logger.debug(
            f"NStepTransitionBuffer initialized: capacity={capacity}, "
            f"n_step={n_step}, gamma={gamma}, state_shape={state_shape}"
        )

    def __len__(self):
        return self._size

    @property
    def n_step(self) -> int:
        return self._n_step

    def set_num_envs(self, num_envs: int):
        """Configure the number of parallel env trajectories."""
        self._num_envs = num_envs
        self._windows = [[] for _ in range(num_envs)]

    def add(self, state, action, reward, next_state, done, env_id=0):
        """Add a single 1-step transition (non-vectorized path)."""
        window = self._windows[env_id]
        window.append({
            "state": np.asarray(state, dtype=np.float32).ravel(),
            "action": np.asarray(action, dtype=np.float32).ravel(),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float32).ravel(),
            "done": float(done),
        })

        if len(window) >= self._n_step:
            self._commit_nstep(window)

        if done:
            while len(window) > 0:
                self._commit_nstep(window)
            window.clear()

    def add_many(self, states, actions, rewards, next_states, dones):
        """Add transitions from a batch of parallel environments.

        Vectorized path: processes all envs in one pass with minimal
        Python overhead.  Only creates dict objects for windows that
        actually need to commit an n-step transition.
        """
        num = self._num_envs

        # ── Fast path: append raw data to all windows ────────────────
        # Avoid per-item np.asarray calls by slicing the input arrays directly.
        states_flat = np.asarray(states, dtype=np.float32)  # (N, D)
        actions_flat = np.asarray(actions, dtype=np.float32)  # (N, A)
        next_states_flat = np.asarray(next_states, dtype=np.float32)  # (N, D)
        rewards_flat = np.asarray(rewards, dtype=np.float32)  # (N,)
        dones_flat = np.asarray(dones, dtype=np.float32)  # (N,)

        for i in range(num):
            self._windows[i].append({
                "state": states_flat[i],
                "action": actions_flat[i],
                "reward": rewards_flat[i],
                "next_state": next_states_flat[i],
                "done": dones_flat[i],
            })

        # ── Commit n-step transitions where ready ────────────────────
        # Only touch windows that have enough entries (most will after
        # warmup, but early episodes may not).
        ready_mask = np.array(
            [len(w) >= self._n_step for w in self._windows],
            dtype=bool
        )
        done_mask = dones_flat.astype(bool)

        # Process windows ready for n-step commit
        for i in np.where(ready_mask)[0]:
            self._commit_nstep(self._windows[i])

        # Flush done windows
        for i in np.where(done_mask)[0]:
            w = self._windows[i]
            while len(w) > 0:
                self._commit_nstep(w)
            w.clear()

    def _commit_nstep(self, window):
        """Commit the oldest n-step (or shorter if flushing) transition."""
        n = len(window)
        first = window[0]
        last = window[-1]

        discounted_reward = np.dot(
            self._gamma_powers[:n],
            np.array([t["reward"] for t in window], dtype=np.float32),
        )

        discount = self._gamma_powers[n]
        done = last["done"]

        self._states[self._pos] = first["state"]
        self._actions[self._pos] = first["action"]
        self._next_states[self._pos] = last["next_state"]
        self._rewards[self._pos] = discounted_reward
        self._discounts[self._pos] = discount
        self._dones[self._pos] = done

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

        window.pop(0)

    def next(self, key, batch_size):
        """Samples a random batch of n-step transitions.

        Uses the provided JAX PRNG key for reproducible sampling.  Indices
        are generated on-device then converted to a Python list to avoid a
        GPU→CPU array-deref sync point.  Each array is transferred
        individually via ``jnp.asarray`` simple host→device copies that
        are cheaper than concatenating on CPU and then slicing on GPU.
        """
        indices = jax.random.randint(key, (batch_size,), 0, self._size).tolist()

        return {
            "state": jnp.asarray(self._states[indices]),
            "action": jnp.asarray(self._actions[indices]),
            "next_state": jnp.asarray(self._next_states[indices]),
            "reward": jnp.asarray(self._rewards[indices]),
            "discount": jnp.asarray(self._discounts[indices]),
            "done": jnp.asarray(self._dones[indices]),
        }