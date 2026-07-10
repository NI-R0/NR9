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

        self._states = jnp.zeros((capacity, *self._state_shape))
        self._next_states = jnp.zeros((capacity, *self._state_shape))
        self._actions = jnp.zeros((capacity, *self._action_shape))
        self._rewards = jnp.zeros((capacity,))
        self._dones = jnp.zeros((capacity,))

        logger.debug(f"ReplayBuffer initialized with capacity {capacity} and state shape {state_shape}")

    def __len__(self):
        return self._size

    def add(self, state, action, reward, next_state, done):
        self._states = self._states.at[self._pos].set(state)
        self._next_states = self._next_states.at[self._pos].set(next_state)
        self._actions = self._actions.at[self._pos].set(action)
        self._rewards = self._rewards.at[self._pos].set(reward)
        self._dones = self._dones.at[self._pos].set(done)

        self._pos = (self._pos + 1) % self._capacity
        self._size = jnp.minimum(self._size + 1, self._capacity)

    def next(self, key, batch_size):
        """
        Samples a random batch of experienced transitions.
        """
        indices = jax.random.randint(key, (batch_size,), 0, self._size)
        return {
            "state": self._states[indices],
            "action": self._actions[indices],
            "next_state": self._next_states[indices],
            "reward": self._rewards[indices],
            "done": self._dones[indices],
        }
