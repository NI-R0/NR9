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

        x = nn.Dense(features=400)(obs)
        x = nn.elu(x)
        x = nn.Dense(features=400)(x)
        x = nn.elu(x)
        x = nn.Dense(features=400)(x)
        x = nn.elu(x)

        mu = nn.Dense(features=dim)(x)

        log_std = nn.Dense(
            features=dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(jnp.log(self.init_scale)),
        )(x)
        scale = jax.nn.softplus(log_std) + 1e-6

        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        inputs = jnp.concatenate([obs, action], axis=-1)

        x = nn.Dense(features=400)(inputs)
        x = nn.elu(x)
        x = nn.Dense(features=400)(x)
        x = nn.elu(x)
        x = nn.Dense(features=400)(x)
        x = nn.elu(x)

        q_value = nn.Dense(features=1)(x)

        return jnp.squeeze(q_value, axis=-1)
