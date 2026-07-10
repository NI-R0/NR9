import jax

class MPOActor:
    def __init__(self, actor_net):
        self.actor_net = actor_net
        self._jit_select_action = jax.jit(self._select_action_fn)
    
    def _select_action_fn(self, params, observation, key, explore):
        dist = self.actor_net.apply(params, observation)

        def sampled():
            return dist.sample(seed=key)
        def deterministic():
            return dist.mode()

        action = jax.lax.cond(explore, sampled, deterministic)
        return action
    
    def select_action(self, params, observation, key, explore=True):
        if observation.ndim == 1:
            observation = observation[None, :]
        
        action = self._jit_select_action(params, observation, key, explore)

        return jax.device_get(action[0])