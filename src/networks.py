import flax.linen as nn
import jax
import jax.numpy as jnp
import distrax


class ActorNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        # Shared MLP Backbone
        x = nn.Dense(features=256)(obs)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        x = nn.Dense(features=256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        x = nn.Dense(features=256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        # Mean Head: Bounded between -1 and 1 via tanh
        mu = nn.Dense(features=self.action_dim)(x)
        mu = jnp.tanh(mu)

        # Scale Head: Add 1e-5 to prevent 0-division or NaN logs.
        log_sigma = nn.Dense(features=self.action_dim)(x)
        sigma = nn.softplus(log_sigma) + 1e-5

        return distrax.MultivariateNormalDiag(loc=mu, scale_diag=sigma)


class CriticNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        inputs = jnp.concatenate([obs, action], axis=-1)

        # MLP Backbone
        x = nn.Dense(features=256)(inputs)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        x = nn.Dense(features=256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        x = nn.Dense(features=256)(x)
        x = nn.LayerNorm()(x)
        x = nn.elu(x)

        q_value = nn.Dense(features=1)(x)

        return jnp.squeeze(q_value, axis=-1)
