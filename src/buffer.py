import pickle
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
        for i in range(self._num_envs):
            self.add(states[i], actions[i], rewards[i], next_states[i],
                     dones[i], env_id=i)

    def _commit_nstep(self, window):
        """Commit the oldest n-step (or shorter if flushing) transition."""
        n = len(window)
        first = window[0]
        last = window[-1]

        discounted_reward = 0.0
        for i, trans in enumerate(window):
            discounted_reward += (self._gamma ** i) * trans["reward"]

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

        window.pop(0)

    def save_state(self, path: str):
        """Save the replay buffer contents and internal state to a pickle file."""
        state = {
            "states": self._states[:self._size],
            "next_states": self._next_states[:self._size],
            "actions": self._actions[:self._size],
            "rewards": self._rewards[:self._size],
            "discounts": self._discounts[:self._size],
            "dones": self._dones[:self._size],
            "size": self._size,
            "pos": self._pos,
            "num_envs": self._num_envs,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.debug(f"Replay buffer saved to {path} ({self._size} transitions).")

    def load_state(self, path: str):
        """Load replay buffer contents and internal state from a pickle file."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        n = state["size"]
        self._states[:n] = state["states"]
        self._next_states[:n] = state["next_states"]
        self._actions[:n] = state["actions"]
        self._rewards[:n] = state["rewards"]
        self._discounts[:n] = state["discounts"]
        self._dones[:n] = state["dones"]
        self._size = n
        self._pos = state["pos"]
        self._num_envs = state.get("num_envs", 1)
        self._windows = [[] for _ in range(self._num_envs)]
        logger.info(f"Replay buffer loaded from {path} ({n} transitions).")

    def next(self, key, batch_size):
        """Samples a random batch of n-step transitions.

        Uses NumPy RNG to avoid a GPU→CPU sync point inside the training
        loop.  Each array is transferred individually via ``jnp.asarray``
        simple host→device copies that are cheaper than concatenating on
        CPU and then slicing on GPU.
        """
        indices = np.random.randint(0, self._size, size=batch_size)

        return {
            "state": jnp.asarray(self._states[indices]),
            "action": jnp.asarray(self._actions[indices]),
            "next_state": jnp.asarray(self._next_states[indices]),
            "reward": jnp.asarray(self._rewards[indices]),
            "discount": jnp.asarray(self._discounts[indices]),
            "done": jnp.asarray(self._dones[indices]),
        }
