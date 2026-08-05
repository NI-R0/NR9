import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger


class NStepTransitionBuffer:
    """
    Circular eplay buffer that stores n-step transitions.
    Accumulates incoming 1-step transitions n-step transitions.
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

        # Precompute gamma powers for discounted reward aggregation.
        self._gamma_powers = np.power(gamma, np.arange(n_step + 1), dtype=np.float32)

        self._states = np.zeros((capacity, *self._state_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *self._state_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, *self._action_shape), dtype=np.float32)
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
        """Add a single 1-step transition; commits n-step transitions as they become available."""
        window = self._windows[env_id]
        window.append({
            "state": np.asarray(state, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": float(reward),
            "next_state": np.asarray(next_state, dtype=np.float32),
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

        ``dones`` may be True for some envs and False for others; each
        env's n-step window is tracked independently.
        """
        # Fast path: append to all windows in one pass, then commit where possible
        for i in range(self._num_envs):
            self._windows[i].append({
                "state": np.asarray(states[i], dtype=np.float32),
                "action": np.asarray(actions[i], dtype=np.float32),
                "reward": float(rewards[i]),
                "next_state": np.asarray(next_states[i], dtype=np.float32),
                "done": float(dones[i]),
            })
            window = self._windows[i]
            if len(window) >= self._n_step:
                self._commit_nstep(window)
            if dones[i]:
                while len(window) > 0:
                    self._commit_nstep(window)
                window.clear()

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
        """Sample a batch using the JAX PRNG key"""
        indices = jax.random.randint(key, (batch_size,), 0, self._size).tolist()
        return {
            "state": jnp.asarray(self._states[indices]),
            "action": jnp.asarray(self._actions[indices]),
            "next_state": jnp.asarray(self._next_states[indices]),
            "reward": jnp.asarray(self._rewards[indices]),
            "discount": jnp.asarray(self._discounts[indices]),
            "done": jnp.asarray(self._dones[indices]),
        }
