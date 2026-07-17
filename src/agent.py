import numpy as np
import jax
import jax.numpy as jnp
from src.learner import MPOLearner
from src.actor import MPOActor
from src.buffer import NStepTransitionBuffer
from jax.random import PRNGKey


class SoccerAgent:
    def __init__(
        self, 
        observation_shape, 
        action_shape, 
        actor_net, 
        critic_net, 
        buffer: NStepTransitionBuffer,
        **kwargs
    ):
        self.random_key = jax.random.PRNGKey(kwargs.get("seed", 42))

        self.learner = MPOLearner(
            actor_net=actor_net,
            critic_net=critic_net,
            observation_shape=observation_shape,
            action_shape=action_shape,
            random_key=self.random_key,
            **kwargs
        )

        self.actor = MPOActor(actor_net)
        self.buffer = buffer
        self.warmup = kwargs.get("warmup", 1000)
        self.batch_size = kwargs.get("batch_size", 256)
        # Update every N environment steps (Acme-style rhythm control).
        # The learner internally performs sgd_steps_per_learner_step gradient
        # steps on the same batch each time update() is called.
        self.update_every = kwargs.get("update_every", 1)
        self._step_count = 0

    def select_action(self, observation, explore=True):

        action, self.random_key = self.actor.select_action(
            params=self.learner.state.params_actor,
            observation=observation,
            key=self.random_key,
            explore=explore
        )

        return action

    def update(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self._step_count += 1

        if len(self.buffer) > self.warmup and (self._step_count % self.update_every == 0):
            self.random_key, sample_key = jax.random.split(self.random_key)
            batch = self.buffer.next(sample_key, self.batch_size)
            self.learner.state, metrics = self.learner._update_step(self.learner.state, batch)
            return metrics
        return {}