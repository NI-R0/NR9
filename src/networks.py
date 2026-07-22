import flax.linen as nn
import jax
import jax.numpy as jnp
import distrax


class ActorNetwork(nn.Module):
    """Actor network with diagonal Gaussian output (Acme-style).

    Uses a MultivariateNormalDiag distribution instead of full covariance.
    The initial scale is set to ``init_scale`` to match Acme's
    ``MultivariateNormalDiagHead(init_scale=0.7)``.
    """
    action_dim: tuple[int]
    init_scale: float = 0.7

    @nn.compact
    def __call__(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        dim = self.action_dim[0]

        # MLP backbone: 100-100 (Control Suite setting from the paper)
        x = nn.Dense(features=100)(obs)
        x = nn.elu(x)

        x = nn.Dense(features=100)(x)
        x = nn.elu(x)

        # Mean head: unbounded (paper does not bound the mean)
        mu = nn.Dense(features=dim)(x)

        # Scale head: produces log-std parameters, shifted so that the
        # initial scale is approximately ``init_scale``.  We use a fixed
        # bias of ``log(init_scale)`` and a zero-initialized weight so that
        # at initialisation the output is exactly ``init_scale``.
        initial_log_std = jnp.log(
            jnp.expm1(self.init_scale)
        )

        log_std = nn.Dense(
            features=dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(initial_log_std),
        )(x)
        
        scale = jnp.clip(
            jax.nn.softplus(log_std),
            1e-4,
            1.0
        )

        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        inputs = jnp.concatenate([obs, action], axis=-1)

        # MLP backbone: 200-200 (Control Suite setting from the paper)
        x = nn.Dense(features=200)(inputs)
        x = nn.elu(x)

        x = nn.Dense(features=200)(x)
        x = nn.elu(x)

        q_value = nn.Dense(features=1)(x)

        return jnp.squeeze(q_value, axis=-1)
