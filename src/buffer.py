import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger


class NStepTransitionBuffer:
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
    """

    def __init__(self, state_shape: tuple[int], action_shape: tuple[int],
                 capacity: int = 100_000, n_step: int = 5, gamma: float = 0.99):
        self._capacity = capacity
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._n_step = n_step
        self._gamma = gamma
        self._size = 0
        self._pos = 0

        # Circular storage for committed n-step transitions
        self._states = np.zeros((capacity, *self._state_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *self._state_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, *self._action_shape), dtype=np.float32)
        # n-step discounted reward
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        # Remaining discount: gamma^n (or less at episode end)
        self._discounts = np.zeros((capacity,), dtype=np.float32)
        # True if the n-step window ended because of a terminal state
        self._dones = np.zeros((capacity,), dtype=np.float32)

        # Rolling window of raw transitions — one list per parallel env.
        # For backward compatibility (single-env), we use a single window
        # when ``num_envs == 1`` and select by ``env_id``.
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
        """Add a single 1-step transition; commits n-step transitions as they become available."""
        window = self._windows[env_id]
        window.append({
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float32),
            "done": float(done),
        })

        # Once we have n_step transitions, commit the n-step transition
        if len(window) >= self._n_step:
            self._commit_nstep(window)

        if done:
            # Flush all remaining partial windows
            while len(window) > 0:
                self._commit_nstep(window)
            window.clear()

    def add_many(self, states, actions, rewards, next_states, dones):
        """Add transitions from a batch of parallel environments.

        ``dones`` may be True for some envs and False for others; each
        env's n-step window is tracked independently.
        """
        for i in range(self._num_envs):
            self.add(states[i], actions[i], rewards[i], next_states[i],
                     dones[i], env_id=i)

    def _commit_nstep(self, window):
        """Commit the oldest n-step (or shorter if flushing) transition."""
        n = len(window)
        first = window[0]
        last = window[-1]

        # Discounted reward sum: r_0 + gamma*r_1 + ... + gamma^{n-1}*r_{n-1}
        discounted_reward = 0.0
        for i, trans in enumerate(window):
            discounted_reward += (self._gamma ** i) * trans["reward"]

        # Remaining discount = gamma^n
        discount = self._gamma ** n
        done = last["done"]

        self._states[self._pos] = first["state"]
        self._actions[self._pos] = first["action"]
        self._next_states[self._pos] = last["next_state"]
        self._rewards[self._pos] = discounted_reward
        self._discounts[self._pos] = discount
        self._dones[self._pos] = done

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

        # Slide window
        window.pop(0)

    def next(self, key, batch_size):
        """Samples a random batch of n-step transitions.

        Gathers all sampled arrays into a single contiguous NumPy buffer
        before transferring to GPU, so only **one** ``jnp.asarray`` call
        is issued instead of six — eliminating five redundant JAX staging
        operations per sample call.
        """
        # Use NumPy RNG to avoid a GPU→CPU sync point inside the training loop.
        indices = np.random.randint(0, self._size, size=batch_size)

        # Slice all arrays with fancy indexing (NumPy, stays on CPU).
        s = self._states[indices]          # (B, *state_shape)
        a = self._actions[indices]         # (B, *action_shape)
        ns = self._next_states[indices]    # (B, *state_shape)
        r = self._rewards[indices]         # (B,)
        d = self._discounts[indices]       # (B,)
        dn = self._dones[indices]          # (B,)

        # Concatenate into one flat buffer: each 2-D array contributes
        # B * prod(shape) elements, each 1-D array contributes B elements.
        flat = np.concatenate([
            s.reshape(batch_size, -1),
            a.reshape(batch_size, -1),
            ns.reshape(batch_size, -1),
            r.reshape(batch_size, -1),
            d.reshape(batch_size, -1),
            dn.reshape(batch_size, -1),
        ], axis=1)  # shape: (B, total_cols)

        # Single CPU→GPU transfer + JAX staging operation.
        flat_gpu = jnp.asarray(flat)

        # Unpack on GPU — pure views, no extra transfers.
        off = 0
        def _take(n_cols, shape):
            nonlocal off
            sl = flat_gpu[:, off:off + n_cols]
            off += n_cols
            return sl.reshape(batch_size, *shape)

        state = _take(int(np.prod(self._state_shape)), self._state_shape)
        action = _take(int(np.prod(self._action_shape)), self._action_shape)
        next_state = _take(int(np.prod(self._state_shape)), self._state_shape)
        reward = _take(1, (1,)).squeeze(-1)
        discount = _take(1, (1,)).squeeze(-1)
        done = _take(1, (1,)).squeeze(-1)

        return {
            "state": state,
            "action": action,
            "next_state": next_state,
            "reward": reward,
            "discount": discount,
            "done": done,
        }
