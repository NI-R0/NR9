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

        # Rolling window of raw transitions (list of dicts)
        self._window: list[dict] = []

        logger.debug(
            f"NStepTransitionBuffer initialized: capacity={capacity}, "
            f"n_step={n_step}, gamma={gamma}, state_shape={state_shape}"
        )

    def __len__(self):
        return self._size

    @property
    def n_step(self) -> int:
        return self._n_step

    def add(self, state, action, reward, next_state, done):
        """Add a single 1-step transition; commits n-step transitions as they become available."""
        self._window.append({
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float32),
            "done": float(done),
        })

        # Once we have n_step transitions, commit the n-step transition
        if len(self._window) >= self._n_step:
            self._commit_nstep()

        if done:
            # Flush all remaining partial windows
            while len(self._window) > 0:
                self._commit_nstep()
            self._window = []

    def _commit_nstep(self):
        """Commit the oldest n-step (or shorter if flushing) transition."""
        n = len(self._window)
        first = self._window[0]
        last = self._window[-1]

        # Discounted reward sum: r_0 + gamma*r_1 + ... + gamma^{n-1}*r_{n-1}
        discounted_reward = 0.0
        for i, trans in enumerate(self._window):
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
        self._window.pop(0)

    def next(self, key, batch_size):
        """Samples a random batch of n-step transitions."""
        # Use NumPy RNG to avoid a GPU→CPU sync point inside the training loop.
        indices = np.random.randint(0, self._size, size=batch_size)

        return {
            "state": jnp.asarray(self._states[indices]),
            "action": jnp.asarray(self._actions[indices]),
            "next_state": jnp.asarray(self._next_states[indices]),
            "reward": jnp.asarray(self._rewards[indices]),
            # remaining discount (gamma^n) — learner multiplies Q_target by this
            "discount": jnp.asarray(self._discounts[indices]),
            "done": jnp.asarray(self._dones[indices]),
        }
