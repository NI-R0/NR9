import typing
import jax
import jax.numpy as jnp
import optax
import distrax
from functools import partial
from src.networks import ActorNetwork, CriticNetwork


class TrainingState(typing.NamedTuple):
    params_actor: optax.Params
    params_critic: optax.Params
    target_params_actor: optax.Params
    target_params_critic: optax.Params
    dual_params: optax.Params
    opt_state_actor: optax.OptState
    opt_state_critic: optax.OptState
    opt_state_dual: optax.OptState
    steps: jax.Array
    random_key: jax.Array


def _clip_log_dual_params(dual_params: dict) -> dict:
    """Clip dual parameters in log-space to ``max(-18, log_x)`` (Acme)."""
    return {
        "log_eta": jnp.maximum(dual_params["log_eta"], -18.0),
        "log_alpha_mean": jnp.maximum(dual_params["log_alpha_mean"], -18.0),
        "log_alpha_std": jnp.maximum(dual_params["log_alpha_std"], -18.0),
    }


def _kl_diag_per_dim(dist_old: distrax.MultivariateNormalDiag,
                     dist_new: distrax.MultivariateNormalDiag) -> jax.Array:
    """Per-dimension KL divergence between two diagonal Gaussians.

    Returns shape ``(batch, dim)`` so that each action dimension can be
    constrained independently (Acme per-dim constraining).
    """
    mu_old, std_old = dist_old.loc, dist_old.scale_diag
    mu_new, std_new = dist_new.loc, dist_new.scale_diag

    var_old = std_old ** 2
    var_new = std_new ** 2
    kl = (jnp.log(std_new) - jnp.log(std_old)
          + (var_old + (mu_old - mu_new) ** 2) / (2.0 * var_new)
          - 0.5)
    return kl


class MPOLearner:
    def __init__(self,
                 actor_net: ActorNetwork,
                 critic_net: CriticNetwork,
                 observation_shape: tuple,
                 action_shape: tuple,
                 random_key,
                 lr=5e-4,
                 critic_lr=None,
                 dual_lr=None,
                 **kwargs):

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.gamma = kwargs.get("gamma", 0.99)

        critic_lr = critic_lr if critic_lr is not None else lr
        dual_lr = dual_lr if dual_lr is not None else lr

        self.config = {
            "epsilon": kwargs.get("epsilon", 0.1),
            "epsilon_mean": kwargs.get("epsilon_mean", 0.0025),
            "epsilon_std": kwargs.get("epsilon_std", 1e-6),
            "sample_k": kwargs.get("sample_k", 20),
            "sgd_steps_per_learner_step": kwargs.get("sgd_steps_per_learner_step", 8),
            "target_update_period": kwargs.get("target_update_period", 100),
            "grad_norm_clip": kwargs.get("grad_norm_clip", 40.0),
        }

        key_actor, key_critic, key_state = jax.random.split(random_key, 3)
        dummy_obs = jnp.zeros((1, *observation_shape))
        dummy_act = jnp.zeros((1, *action_shape))

        params_actor = self.actor_net.init(key_actor, dummy_obs)
        params_critic = self.critic_net.init(key_critic, dummy_obs, dummy_act)

        action_dim = action_shape[0]
        dual_params = {
            "log_eta": jnp.array(jnp.log(10.0)),
            "log_alpha_mean": jnp.full((action_dim,), jnp.log(10.0)),
            "log_alpha_std": jnp.full((action_dim,), jnp.log(1000.0)),
        }

        self.opt_actor = optax.chain(
            optax.clip_by_global_norm(self.config["grad_norm_clip"]),
            optax.adam(lr),
        )
        self.opt_critic = optax.chain(
            optax.clip_by_global_norm(self.config["grad_norm_clip"]),
            optax.adam(critic_lr),
        )
        self.opt_dual = optax.chain(
            optax.clip_by_global_norm(self.config["grad_norm_clip"]),
            optax.adam(dual_lr),
        )

        self.state = TrainingState(
            params_actor=params_actor,
            params_critic=params_critic,
            target_params_actor=params_actor,
            target_params_critic=params_critic,
            dual_params=dual_params,
            opt_state_actor=self.opt_actor.init(params_actor),
            opt_state_critic=self.opt_critic.init(params_critic),
            opt_state_dual=self.opt_dual.init(dual_params),
            steps=jnp.array(0),
            random_key=key_state
        )

    def _critic_loss(self,
                     params_critic,
                     target_params_critic,
                     target_params_actor,
                     batch,
                     key):

        distribution_next = self.actor_net.apply(target_params_actor, batch["next_state"])
        next_actions = distribution_next.sample(seed=key)
        next_actions = jnp.tanh(next_actions)
        next_q = self.critic_net.apply(target_params_critic, batch["next_state"], next_actions)
        next_q = jnp.clip(next_q, -100.0, 100.0)
        target_q = batch["reward"] + batch["discount"] * (1.0 - batch["done"]) * next_q
        target_q = jnp.clip(target_q, -200.0, 200.0)
        current_q = self.critic_net.apply(params_critic, batch["state"], batch["action"])

        return jnp.mean(jnp.square(current_q - jax.lax.stop_gradient(target_q)))

    def _compute_weights(self, params_critic, dist_target, batch, eta, key):
        states = batch["state"]
        k = self.config["sample_k"]

        sampled_actions = dist_target.sample(seed=key, sample_shape=(k,))
        tanh_actions = jnp.tanh(sampled_actions)
        vmapped_critic = jax.vmap(self.critic_net.apply, in_axes=(None, None, 0))
        q_values = vmapped_critic(params_critic, states, tanh_actions)
        q_values = q_values.T
        q_values = jax.lax.stop_gradient(q_values)
        max_q = jnp.max(q_values, axis=1, keepdims=True)
        log_weights = (q_values - max_q) / jnp.maximum(eta, 1e-8)

        sampled_actions = jnp.swapaxes(sampled_actions, 0, 1)

        return log_weights, max_q, sampled_actions

    def _dual_loss(self, log_eta, log_weights, max_q, epsilon):
        """E-step dual loss (temperature eta)."""
        eta = jnp.exp(log_eta)

        k = log_weights.shape[1]
        log_avg_exp = jax.nn.logsumexp(log_weights, axis=1) - jnp.log(k)

        return eta * epsilon + jnp.mean(max_q.squeeze(axis=-1) + eta * log_avg_exp)

    def _policy_and_dual_loss(self,
                              params_actor,
                              dual_params,
                              distribution_old,
                              batch,
                              log_weights,
                              max_q,
                              sampled_actions):

        weights = jax.lax.stop_gradient(jax.nn.softmax(log_weights, axis=1))
        distribution_current = self.actor_net.apply(params_actor, batch["state"])
        dist_expanded = distrax.MultivariateNormalDiag(
            loc=distribution_current.loc[:, None, :],
            scale_diag=distribution_current.scale_diag[:, None, :]
        )

        log_probs = dist_expanded.log_prob(sampled_actions)
        loss_policy = -jnp.mean(jnp.sum(weights * log_probs, axis=1))

        dist_fixed_stddev = distrax.MultivariateNormalDiag(
            loc=distribution_current.loc,
            scale_diag=distribution_old.scale_diag,
        )

        dist_fixed_mean = distrax.MultivariateNormalDiag(
            loc=distribution_old.loc,
            scale_diag=distribution_current.scale_diag,
        )

        kl_mean_per_dim = _kl_diag_per_dim(distribution_old, dist_fixed_stddev)
        kl_std_per_dim = _kl_diag_per_dim(distribution_old, dist_fixed_mean)

        kl_mean = jnp.mean(kl_mean_per_dim, axis=0)
        kl_std = jnp.mean(kl_std_per_dim, axis=0)

        log_eta = dual_params["log_eta"]
        alpha_mean = jnp.exp(dual_params["log_alpha_mean"])
        alpha_std = jnp.exp(dual_params["log_alpha_std"])

        loss_eta = self._dual_loss(log_eta, log_weights, max_q, self.config["epsilon"])

        epsilon_mean = self.config["epsilon_mean"]
        epsilon_std = self.config["epsilon_std"]

        loss_alpha_mean = jnp.sum(
            alpha_mean * (epsilon_mean - jax.lax.stop_gradient(kl_mean)))
        loss_alpha_std = jnp.sum(
            alpha_std * (epsilon_std - jax.lax.stop_gradient(kl_std)))

        loss_actor = (
            loss_policy
            + jnp.sum(jax.lax.stop_gradient(alpha_mean) * kl_mean)
            + jnp.sum(jax.lax.stop_gradient(alpha_std) * kl_std)
        )

        return loss_actor + loss_eta + loss_alpha_mean + loss_alpha_std

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(self, state: TrainingState, batch):
        """Performs one full learner step with sgd_steps_per_learner_step gradient updates.

        1. Critic update (N-step Bellman target).
        2. E-Step: compute action weights from critic Q-values.
        3. M-Step: update actor + dual variables with per-dim KL constraints.
        4. Periodic hard target update every target_update_period steps.
        5. Dual parameter clipping (max(-18, log_x)).
        """
        def sgd_step(carry, _):
            state = carry
            key, key_critic, key_sample = jax.random.split(state.random_key, 3)

            dist_sample = self.actor_net.apply(state.target_params_actor, batch["state"])
            dist_old = self.actor_net.apply(state.params_actor, batch["state"])

            def critic_loss_fn(p):
                return self._critic_loss(
                    p,
                    state.target_params_critic,
                    state.target_params_actor,
                    batch,
                    key_critic,
                )

            loss_c, grads_critic = jax.value_and_grad(critic_loss_fn)(state.params_critic)
            updates_c, opt_state_c = self.opt_critic.update(grads_critic, state.opt_state_critic)
            params_critic = optax.apply_updates(state.params_critic, updates_c)

            eta = jnp.exp(state.dual_params["log_eta"])
            log_weights, max_q, sampled_actions = self._compute_weights(
                state.params_critic, dist_sample, batch, eta, key_sample
            )

            def actor_dual_loss_fn(p_actor, p_dual):
                return self._policy_and_dual_loss(
                    p_actor, p_dual, dist_old, batch,
                    log_weights, max_q, sampled_actions,
                )

            grads_actor, grads_dual = jax.grad(actor_dual_loss_fn, argnums=(0, 1))(
                state.params_actor, state.dual_params
            )

            updates_a, opt_state_a = self.opt_actor.update(grads_actor, state.opt_state_actor)
            params_actor = optax.apply_updates(state.params_actor, updates_a)
            updates_d, opt_state_d = self.opt_dual.update(grads_dual, state.opt_state_dual)
            dual_params = optax.apply_updates(state.dual_params, updates_d)
            dual_params = _clip_log_dual_params(dual_params)

            new_state = state._replace(
                params_actor=params_actor,
                params_critic=params_critic,
                dual_params=dual_params,
                opt_state_actor=opt_state_a,
                opt_state_critic=opt_state_c,
                opt_state_dual=opt_state_d,
                random_key=key,
            )
            return new_state, loss_c

        state, loss_c = jax.lax.scan(
            sgd_step, state, None, length=self.config["sgd_steps_per_learner_step"]
        )

        loss_c = loss_c[-1]

        period = self.config["target_update_period"]
        steps = state.steps + 1
        do_update = (steps % period) == 0
        target_params_actor = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_update, new, old),
            state.params_actor, state.target_params_actor)
        target_params_critic = jax.tree_util.tree_map(
            lambda new, old: jnp.where(do_update, new, old),
            state.params_critic, state.target_params_critic)

        new_state = state._replace(
            target_params_actor=target_params_actor,
            target_params_critic=target_params_critic,
            steps=steps,
        )

        return new_state, {"loss_critic": loss_c}
