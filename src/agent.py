import jax
import numpy as np
from src.learner import MPOLearner
from src.actor import MPOActor
from src.buffer import NStepTransitionBuffer


class MPOAgent:
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

    def select_actions(self, observations, explore=True):
        """Select actions for a batch of observations (vectorized envs)."""
        actions, self.random_key = self.actor.select_actions(
            params=self.learner.state.params_actor,
            observations=observations,
            key=self.random_key,
            explore=explore,
        )
        return actions

    def update(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self._step_count += 1

        if len(self.buffer) > self.warmup and (self._step_count % self.update_every == 0):
            self.random_key, sample_key = jax.random.split(self.random_key)
            batch = self.buffer.next(sample_key, self.batch_size)
            self.learner.state, metrics = self.learner._update_step(self.learner.state, batch)

            # PER: write TD-errors back to buffer for priority update
            if self.buffer._use_per and "indices" in batch and "weights" in batch:
                # Compute TD-error from critic loss (|Q - target_Q|)
                # We can extract it from the batch: compute current Q vs target Q
                self._update_priorities_from_batch(batch)

            return metrics
        return {}

    def _update_priorities_from_batch(self, batch):
        """Compute TD-errors and write them back to the buffer."""
        from jax import numpy as jnp
        # Current Q-values
        current_q = self.learner.critic_net.apply(
            self.learner.state.params_critic,
            batch["state"],
            batch["action"],
        )
        # Target Q-values (bootstrap from target networks)
        dist_next = self.learner.actor_net.apply(
            self.learner.state.target_params_actor, batch["next_state"]
        )
        next_actions = jnp.tanh(dist_next.sample(seed=self.random_key))
        _, self.random_key = jax.random.split(self.random_key)
        next_q = self.learner.critic_net.apply(
            self.learner.state.target_params_critic,
            batch["next_state"],
            next_actions,
        )
        target_q = batch["reward"] + batch["discount"] * (1.0 - batch["done"]) * next_q
        td_errors = jnp.abs(current_q - target_q)

        # Convert to numpy for buffer update
        indices = np.asarray(batch["indices"])
        errors = np.asarray(td_errors)
        self.buffer.update_priorities(indices, errors)

    def update_batch(self, states, actions, rewards, next_states, dones):
        """Add transitions from all parallel envs and optionally run a learner step.

        Each env contributes one transition.  ``self._step_count`` is
        incremented by ``num_envs`` so that ``update_every`` still refers
        to total environment steps (not meta-steps).
        """
        self.buffer.add_many(states, actions, rewards, next_states, dones)
        previous_count = self._step_count
        self._step_count += self.buffer._num_envs

        should_update = (self._step_count // self.update_every) > (previous_count // self.update_every)

        if len(self.buffer) > self.warmup and should_update:
            self.random_key, sample_key = jax.random.split(self.random_key)
            batch = self.buffer.next(sample_key, self.batch_size)
            self.learner.state, metrics = self.learner._update_step(self.learner.state, batch)

            # PER: write TD-errors back to buffer for priority update
            if self.buffer._use_per and "indices" in batch and "weights" in batch:
                self._update_priorities_from_batch(batch)

            return metrics
        return {}
