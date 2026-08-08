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

**Prioritized Experience Replay (PER)** — Schaul et al. 2015:
When ``use_per=True``, transitions are sampled proportional to their
|TD-error|^alpha.  Importance Sampling (IS) weights correct the bias
introduced by non-uniform sampling.  The TD-error is written back by
the learner via ``update_priorities()``.
"""

import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger


class _SegmentTree:
    """Simple segment tree for efficient priority lookup and update.

    Supports:
    - ``__setitem__(index, value)``: O(log n)
    - ``sample(prob)``: O(log n) — find leaf by cumulative probability
    - ``total``: O(1) sum of all priorities
    - ``__getitem__(index)``: O(log n)
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._tree_size = 2 * capacity

    def __setitem__(self, index: int, value: float):
        i = index + self._capacity  # leaf offset
        self._tree[i] = max(value, 1e-8)  # avoid zero priorities
        i //= 2
        while i > 0:
            self._tree[i] = self._tree[2 * i] + self._tree[2 * i + 1]
            i //= 2

    def __getitem__(self, index: int):
        return float(self._tree[index + self._capacity])

    @property
    def total(self) -> float:
        if self._capacity == 0:
            return 0.0
        return self._tree[1]

    def sample(self, prob: float) -> int:
        """Find the leaf index whose cumulative probability covers ``prob``."""
        i = 1  # root
        while i < self._capacity:
            i = 2 * i if prob <= self._tree[i] else 2 * i + 1
        return i - self._capacity


class NStepTransitionBuffer:
    def __init__(self, state_shape: tuple[int], action_shape: tuple[int],
                 capacity: int = 100_000, n_step: int = 5, gamma: float = 0.99,
                 use_per: bool = False, per_alpha: float = 0.6, per_beta: float = 0.4):
        self._capacity = capacity
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._state_dim = int(np.prod(state_shape))
        self._action_dim = int(np.prod(action_shape))
        self._n_step = n_step
        self._gamma = gamma
        self._size = 0
        self._pos = 0
        self._use_per = use_per

        # PER hyper-parameters
        self._per_alpha = per_alpha
        self._per_beta = per_beta

        # Precompute gamma powers for discounted reward aggregation.
        self._gamma_powers = np.power(gamma, np.arange(n_step + 1), dtype=np.float32)

        # ── Circular storage (contiguous arrays) ──────────────────────
        self._states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, self._action_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, self._state_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._discounts = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        # ── PER storage ───────────────────────────────────────────────
        if use_per:
            self._tree = _SegmentTree(capacity)
            self._priorities = np.ones(capacity, dtype=np.float32)  # uniform at init
        else:
            self._tree = None
            self._priorities = None

        self._num_envs = 1
        self._windows: list[list[dict]] = [[]]

        logger.debug(
            f"NStepTransitionBuffer initialized: capacity={capacity}, "
            f"n_step={n_step}, gamma={gamma}, state_shape={state_shape}, "
            f"PER={'ON (α=' + str(per_alpha) + ', β=' + str(per_beta) + ')' if use_per else 'OFF'}"
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

        if self._use_per:
            # New transitions start with max priority (will be updated by learner)
            max_p = self._priorities.max() if self._size > 0 else 1.0
            self._priorities[self._pos] = max_p
            self._tree[self._pos] = max_p

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

        window.pop(0)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for a batch of transitions after learning.

        Called by the learner after computing TD-errors for a batch.
        Priority = |TD-error| + ε raised to α.

        Parameters
        ----------
        indices : np.ndarray
            Buffer indices of the transitions (length batch_size).
        td_errors : np.float32
            Absolute TD-error per transition (length batch_size).
        """
        if not self._use_per:
            return
        priorities = (np.abs(td_errors) + 1e-8) ** self._per_alpha
        for idx, p in zip(indices, priorities):
            self._priorities[idx] = p
            self._tree[idx] = p

    def next(self, key, batch_size):
        """Sample a batch of n-step transitions.

        With PER enabled: samples proportional to priority^α and returns
        importance sampling (IS) weights for gradient correction.
        Without PER: uniform random sampling.

        Returns
        -------
        dict with keys: state, action, next_state, reward, discount, done
            plus (with PER): weights (IS correction), indices (buffer positions)
        """
        if self._use_per:
            # Probability-proportional sampling via segment tree
            probs = np.random.rand(batch_size) * self._tree.total
            indices = np.array([self._tree.sample(p) for p in probs], dtype=np.int64)
            # Clamp to valid range
            indices = np.clip(indices, 0, self._size - 1)

            # Importance sampling weights: (N * P(i))^(-β)
            # Normalised to [0, 1] for numerical stability
            sampling_probs = self._priorities[indices] / (self._tree.total + 1e-8)
            is_weights = (self._size * sampling_probs + 1e-8) ** (-self._per_beta)
            max_weight = float(is_weights.max())
            is_weights = (is_weights / max_weight).astype(np.float32)
        else:
            # Non-PER: use numpy random fallback when key is None
            if key is None:
                indices = np.random.randint(0, self._size, batch_size).tolist()
            else:
                indices = jax.random.randint(key, (batch_size,), 0, self._size).tolist()

        is_weights = np.ones(batch_size, dtype=np.float32)

        return {
            "state": jnp.asarray(self._states[indices]),
            "action": jnp.asarray(self._actions[indices]),
            "next_state": jnp.asarray(self._next_states[indices]),
            "reward": jnp.asarray(self._rewards[indices]),
            "discount": jnp.asarray(self._discounts[indices]),
            "done": jnp.asarray(self._dones[indices]),
            "weights": jnp.asarray(is_weights) if self._use_per else jnp.ones(batch_size, dtype=jnp.float32),
            "indices": jnp.asarray(indices),
        }