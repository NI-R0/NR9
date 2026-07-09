import numpy as np
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
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

        logger.debug(f"ReplayBuffer initialized with capacity {capacity} and state shape {state_shape}")

    def __len__(self):
        return self._size

    def add(self, state, action, reward, next_state, done):
        self._states[self._pos] = np.asarray(state)
        self._next_states[self._pos] = np.asarray(next_state)
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = done

        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def next(self, batch_size):
        """
        Samples a random batch of experienced transitions.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self._states[indices],
            self._actions[indices],
            self._next_states[indices],
            self._rewards[indices],
            self._dones[indices],
        )
