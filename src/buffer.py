import numpy as np
import jax
import jax.numpy as jnp
from loguru import logger


class ReplayBuffer:
    def __init__(self, state_shape: tuple[int], action_shape: tuple[int], capacity: int = 100000):
        self._capacity = capacity
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._size = 0
        self._pos = 0

        self._states = np.zeros((capacity, *self._state_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *self._state_shape), dtype=np.float32)
        self._actions = np.zeros((capacity, *self._action_shape), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

        logger.debug(f"ReplayBuffer initialized with capacity {capacity} and state shape {state_shape}")

    def __len__(self):
        return self._size

    def add(self, state, action, reward, next_state, done):
        self._states[self._pos] = np.asarray(state, dtype=np.float32)
        self._next_states[self._pos] = np.asarray(next_state, dtype=np.float32)
        self._actions[self._pos] = np.asarray(action, dtype=np.float32)
        self._rewards[self._pos] = reward
        self._dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def next(self, key, batch_size):
        """
        Samples a random batch of experienced transitions.
        Returns a dict of JAX arrays ready for model input.
        """
        indices = jax.random.randint(key, (batch_size,), 0, self._size)
        indices = np.asarray(indices)

        return {
            "state": jnp.asarray(self._states[indices]),
            "action": jnp.asarray(self._actions[indices]),
            "next_state": jnp.asarray(self._next_states[indices]),
            "reward": jnp.asarray(self._rewards[indices]),
            "done": jnp.asarray(self._dones[indices]),
        }
