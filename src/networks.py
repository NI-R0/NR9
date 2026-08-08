import flax.linen as nn
import jax
import jax.numpy as jnp
import distrax


class MLP(nn.Module):
    """Shared MLP trunk: N hidden layers of the same width, ELU activations."""
    hidden_sizes: tuple[int, ...] = (400, 400, 400)

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for size in self.hidden_sizes:
            x = nn.Dense(features=size)(x)
            x = nn.elu(x)
        return x


class ActorNetwork(nn.Module):
    """Actor network with diagonal Gaussian output (Acme-style).

    Uses a MultivariateNormalDiag distribution instead of full covariance.
    The initial scale is set to ``init_scale`` to match Acme's
    ``MultivariateNormalDiagHead(init_scale=0.7)``.
    """
    action_dim: tuple[int]
    init_scale: float = 1.0

    @nn.compact
    def __call__(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        dim = self.action_dim[0]
        x = MLP()(obs)
        mu = nn.Dense(features=dim)(x)

        initial_log_std = jnp.log(jnp.expm1(self.init_scale))
        log_std = nn.Dense(
            features=dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(initial_log_std),
        )(x)
        scale = jnp.clip(jax.nn.softplus(log_std), 1e-2, 1.0)

        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=scale)


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = MLP()(jnp.concatenate([obs, action], axis=-1))
        return jnp.squeeze(nn.Dense(features=1)(x), axis=-1)
