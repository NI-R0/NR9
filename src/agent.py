import jax
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
        """Select actions for a batch of observations (parallel envs)."""
        actions, self.random_key = self.actor.select_actions(
            params=self.learner.state.params_actor,
            observations=observations,
            key=self.random_key,
            explore=explore,
        )
        return actions

    def _run_learner_step(self):
        if len(self.buffer) <= self.warmup:
            return {}
        self.random_key, sample_key = jax.random.split(self.random_key)
        batch = self.buffer.next(sample_key, self.batch_size)
        self.learner.state, metrics = self.learner._update_step(self.learner.state, batch)
        return metrics

    def update(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self._step_count += 1
        if self._step_count % self.update_every == 0:
            return self._run_learner_step()
        return {}

    def update_batch(self, states, actions, rewards, next_states, dones):
        """Add transitions from all parallel envs and optionally run a learner step."""
        self.buffer.add_many(states, actions, rewards, next_states, dones)
        previous_count = self._step_count
        # Increment step_count by num_envs so update_every refers to total env steps
        self._step_count += self.buffer._num_envs
        if (self._step_count // self.update_every) > (previous_count // self.update_every):
            return self._run_learner_step()
        return {}
