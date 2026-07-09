import numpy as np
import jax
import jax.numpy as jnp
from src.learner import MPOLearner
from src.buffer import ReplayBuffer
from jax.random import PRNGKey


class SoccerAgent:
    def __init__(self, learner: MPOLearner, buffer: ReplayBuffer, warmup: int, batch_size: int, random_key: PRNGKey):
        self.learner = learner
        self.buffer = buffer

        self.batch_size = batch_size
        self.warmup = warmup * batch_size
        self.random_key = random_key

    def train_step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.warmup:
            batch = self.buffer.next(self.batch_size)
            self.learner.state, info = self.learner._update_step(self.learner.state, batch)
            return info
        return None

    def select_action(self, state, explore):
        state = jnp.asarray(state)[None, :]

        params = self.learner.state.params_actor
        dist = self.learner.actor_net.apply(params, state)

        if explore:
            key, subkey = jax.random.split(self.learner.state.random_key)
            self.learner.state = self.learner.state._replace(random_key=key)
            action = dist.sample(seed=subkey)
        else:
            action = dist.loc

        return np.array(action[0])
