import typing
import jax
import jax.numpy as jnp
import optax
import distrax
from functools import partial
from src.networks import ActorNetwork, CriticNetwork


class TrainingState(typing.NamedTuple):
    params_actor: optax.Params          # Actor params
    params_critic: optax.Params         # Critic params
    target_params_actor: optax.Params   # Target Actor params
    target_params_critic: optax.Params  # Target Critic params
    dual_params: optax.Params           # Log_eta, log_alpha_mu, log_alpha_sigma
    opt_state_actor: optax.OptState     # Optimizer state for actor
    opt_state_critic: optax.OptState    # Optimizer state for critic
    opt_state_dual: optax.OptState      # Optimizer state for dual variables
    steps: jax.Array                    # Training step counter
    random_key: jax.Array               # RNG key for sampling/noise


class MPOLearner:
    def __init__(self,
                 actor_net: ActorNetwork,
                 critic_net: CriticNetwork,
                 observation_shape: tuple,
                 action_shape: tuple,
                 random_key,
                 learning_rate=3e-4,
                 dual_learning_rate=1e-2,
                 tau=0.005):

        self.actor_net = actor_net
        self.critic_net = critic_net
        self.tau = tau

        # MPO hyperparameters
        self.config = {
            "epsilon": 0.1,         # KL constraint for E-step
            "epsilon_mean": 0.001,  # KL constraint for M-step (mean)
            "epsilon_std": 0.0001,  # KL constraint for M-step (std)
            "sample_k": 20          # Action samples for E-step
        }

        # RNG keys
        key_actor, key_critic, key_state = jax.random.split(random_key, 3)
        dummy_obs = jnp.zeros((1, *observation_shape))
        dummy_act = jnp.zeros((1, *action_shape))

        # Network params
        params_actor = self.actor_net.init(key_actor, dummy_obs)
        params_critic = self.critic_net.init(key_critic, dummy_obs, dummy_act)

        # Dual variables
        # log-space dual parameters avoids crashing exp(Q/eta)
        dual_params = {
            "log_eta": jnp.array([0.0]),
            "log_alpha_mean": jnp.array([0.0]),
            "log_alpha_std": jnp.array([0.0]),
        }

        # Optimizers
        self.opt_actor = optax.adam(learning_rate)
        self.opt_critic = optax.adam(learning_rate)
        self.opt_dual = optax.adam(dual_learning_rate)

        # Build initial training state
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
        # Sample next action from target actor
        distribution_next = self.actor_net.apply(target_params_actor, batch["next_state"])
        next_actions = distribution_next.sample(seed=key)

        # Get target Q-values
        next_q = self.critic_net.apply(target_params_critic, batch["next_state"], next_actions)

        # Bellman target: y = r + gamma * Q_target(s', a')
        # Using batch["done"] to calculate bellman target for final episode
        gamma = 0.99
        target_q = batch["reward"] + gamma * (1.0 - batch["done"]) * next_q

        # Current Q-value prediction
        current_q = self.critic_net.apply(params_critic, batch["state"], batch["action"])

        # Return MSE loss
        return jnp.mean(jnp.square(current_q - jax.lax.stop_gradient(target_q)))

    def _compute_weights(self, params_critic, dist_target, batch, eta, key):
        states = batch["state"]
        k = self.config["sample_k"]

        # Sample K actions from target actor for each state
        sampled_actions = dist_target.sample(seed=key, sample_shape=(k,))

        # Vectorize critic over K dimensions
        vmapped_critic = jax.vmap(self.critic_net.apply, in_axes=(None, None, 0))
        q_values = vmapped_critic(params_critic, states, sampled_actions)
        q_values = jnp.squeeze(q_values, axis=-1).T
        q_values = jax.lax.stop_gradient(q_values)

        # Compute weights via temperature eta
        max_q = jnp.max(q_values, axis=1, keepdims=True)
        log_weights = (q_values - max_q) / jnp.maximum(eta, 1e-8)

        sampled_actions = jnp.swapaxes(sampled_actions, 0, 1)

        return log_weights, max_q, sampled_actions

    def _dual_loss(self, log_eta, log_weights, max_q, epsilon):
        eta = jnp.exp(log_eta)

        k = log_weights.shape[1]
        log_avg_exp = jax.nn.logsumexp(log_weights, axis=1) - jnp.log(k)

        return eta * epsilon + jnp.mean(max_q.squeeze() + eta * log_avg_exp)

    def _policy_and_dual_loss(self,
                              params_actor,
                              dual_params,
                              distribution_old,
                              batch,
                              log_weights,
                              max_q,
                              sampled_actions):
        weights = jax.nn.softmax(log_weights, axis=1)

        # Get distribution
        distribution_current = self.actor_net.apply(params_actor, batch["state"])

        # Weighted log-likehood loss
        # Minimize negative to maximize
        log_probs = distribution_current.log_prob(sampled_actions)
        loss_policy = -jnp.mean(jnp.sum(weights * log_probs, axis=1))

        # Calculate KL constraints mean and std individually
        dist_mu = distrax.MultivariateNormalDiag(
            loc=distribution_current.loc,
            scale_diag=distribution_old.scale_diag
        )
        dist_sigma = distrax.MultivariateNormalDiag(
            loc=distribution_old.loc,
            scale_diag=distribution_current.scale_diag
        )

        kl_mu = jnp.mean(distribution_old.kl_divergence(dist_mu))
        kl_sigma = jnp.mean(distribution_old.kl_divergence(dist_sigma))

        # Dual variables loss
        log_eta = dual_params["log_eta"]
        alpha_mu = jnp.exp(dual_params["log_alpha_mean"])
        alpha_sigma = jnp.exp(dual_params["log_alpha_std"])

        # Stop gradient on KL so actor params aren't affected
        loss_eta = self._dual_loss(log_eta, log_weights, max_q, self.config["epsilon"])
        loss_alpha_mu = alpha_mu * (self.config["epsilon_mean"] -
                                    jax.lax.stop_gradient(kl_mu))
        loss_alpha_sigma = alpha_sigma * (self.config["epsilon_std"] -
                                          jax.lax.stop_gradient(kl_sigma))

        # Penalize actor if KL > epsilon
        # (stop gradient on alpha so dual params are not affected)
        loss_actor = loss_policy + \
            jax.lax.stop_gradient(alpha_mu) * kl_mu + \
            jax.lax.stop_gradient(alpha_sigma) * kl_sigma

        return loss_actor + loss_eta + loss_alpha_mu + loss_alpha_sigma

    @partial(jax.jit, static_argnums=(0,))
    def _update_step(self, state: TrainingState, batch):
        """
        Performs the actual MPO algorithm. The steps are the following:

        1. Calculate the loss of the critic.
            - let target actor predict action in the next state
            - let target critic evaluate the value of that next action
            - calculate the Bellman target 'Reward + (Gamma * Next_Q_Value)
            - let online critic evaluate the Q-value of the current state-action pair
            - calculate the mean squared error between the prediction of the online 
            critic and the bellman target
        2. Perform gradient descent to update the weights of the online critic.<br></br>

        3. E-step - Determine weights for the actions according to the critic.
            - let online critic evaluate the Q-values for states and actions of the batch
            - apply temperature by deviding the Q-values by eta
                - large eta -> similar weights, uncertainty
                - small eta -> aggressive weights, high certainty
            - exponentiate to get weights that are probability-like and normalize

        4. M-step - Update actor parameters based on weights from the E-step, while ensuring
        policy does not change to much.
            - let online actor calculate the log-probability of the actions
            - mulitply by weights to get learning signals
            - calculate Kullback-Leibler (KL) divergences for KL mean and KL std seperately
            - calculate alpha (mean, std) to act like a penalty multiplier
            - combine losses to calculate gradients for actor parameters and alphas

        5. Soft target update & making the new TrainingState
            - perform a weighted average based weight update
            - build the new TrainingState dictionary"""
        key, key_critic, key_sample = jax.random.split(state.random_key, 3)
        dist_target = self.actor_net.apply(state.target_params_actor, batch["state"])

        # Update critic
        def critic_loss_fn(p):
            return self._critic_loss(
                p,
                state.target_params_critic,
                state.target_params_actor,
                batch,
                key_critic
            )

        loss_c, grads_critic = jax.value_and_grad(critic_loss_fn)(state.params_critic)
        updates_c, opt_state_c = self.opt_critic.update(grads_critic, state.opt_state_critic)
        params_critic = optax.apply_updates(state.params_critic, updates_c)

        # E-Step
        eta = jnp.exp(state.dual_params["log_eta"])
        log_weights, max_q, sampled_actions = self._compute_weights(
            params_critic,
            dist_target,
            batch,
            eta,
            key_sample
        )

        # M-Step
        def actor_dual_loss_fn(p_actor, p_dual):
            return self._policy_and_dual_loss(
                p_actor,
                p_dual,
                dist_target,
                batch,
                log_weights,
                max_q,
                sampled_actions
            )

        grads_actor, grads_dual = jax.grad(actor_dual_loss_fn, argnums=(0, 1))(
            state.params_actor,
            state.dual_params
        )

        updates_a, opt_state_a = self.opt_actor.update(grads_actor, state.opt_state_actor)
        params_actor = optax.apply_updates(state.params_actor, updates_a)
        updates_d, opt_state_d = self.opt_dual.update(grads_dual, state.opt_state_dual)
        dual_params = optax.apply_updates(state.dual_params, updates_d)

        # Soft update target networks
        target_params_critic = optax.incremental_update(
            params_critic,
            state.target_params_critic,
            self.tau
        )
        target_params_actor = optax.incremental_update(
            params_actor,
            state.target_params_actor,
            self.tau
        )

        # Build new state and return it
        new_state = state._replace(
            params_actor=params_actor,
            params_critic=params_critic,
            target_params_actor=target_params_actor,
            target_params_critic=target_params_critic,
            dual_params=dual_params,
            opt_state_actor=opt_state_a,
            opt_state_critic=opt_state_c,
            opt_state_dual=opt_state_d,
            steps=state.steps + 1,
            random_key=key
        )

        return new_state, {"loss_critic": loss_c}
