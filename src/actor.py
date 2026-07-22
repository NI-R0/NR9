import jax
import jax.numpy as jnp

class MPOActor:
    def __init__(self, actor_net):
        self.actor_net = actor_net
        self._jit_select_action = jax.jit(self._select_action_fn)

    def _select_action_fn(self, params, observation, key, explore):
        key, subkey = jax.random.split(key)
        dist = self.actor_net.apply(params, observation)

        def sampled():
            return dist.sample(seed=subkey)
        def deterministic():
            return dist.mode()

        action = jax.lax.cond(explore, sampled, deterministic)
        return action, key

    def select_action(self, params, observation, key, explore=True):
        if observation.ndim == 1:
            observation = observation[None, :]

        action, new_key = self._jit_select_action(params, observation, key, explore)

        return jnp.squeeze(action, axis=0), new_key

    def select_actions(self, params, observations, key, explore=True):
        """Select actions for a batch of observations (vectorized envs).

        Args:
            observations: (N, obs_dim) array - always 2-D.

        Returns:
            actions: (N, action_dim) array.
            new_key: updated JAX PRNG key.
        """
        action, new_key = self._jit_select_action(params, observations, key, explore)
        return action, new_key