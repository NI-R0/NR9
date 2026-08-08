"""Intrinsic Curiosity Module (ICM) with forward state prediction.

A small online forward model predicts the next observation from the current
observation.  The prediction error (root MSE) is used as an **intrinsic
reward signal**: novel/unpredictable states yield high intrinsic reward,
familiar states yield near-zero intrinsic reward.

The forward model is a tiny MLP trained with JAX (CPU-only) via automatic
differentiation.  It runs inside each dm_control worker process independently
— no shared state, no GPU dependency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import numpy as np


class ForwardModel:
    """Online forward state predictor for ICM intrinsic rewards.

    Predicts ``next_obs`` from ``obs`` using a small MLP.  Trains itself
    with every call to ``update`` and returns the prediction error as
    the intrinsic reward signal.

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the flattened observation vector.
    hidden_sizes : tuple[int, ...]
        Hidden layer sizes of the MLP (default: ``(64, 32)``).
    lr : float
        Learning rate for the forward model optimizer.
    intrinsic_scale : float
        Multiplicative factor applied to the prediction error before
        returning it as intrinsic reward.  Higher values amplify the
        curiosity signal.
    clip_grad : float
        Global gradient norm clipping threshold for stability.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        lr: float = 5e-4,
        intrinsic_scale: float = 1.0,
        clip_grad: float = 1.0,
        seed: int = 42,
    ):
        self._obs_dim = obs_dim
        self._intrinsic_scale = float(intrinsic_scale)
        self._num_layers = len(hidden_sizes) + 1  # hidden + output

        dims = [obs_dim] + list(hidden_sizes) + [obs_dim]
        self._dims = dims

        # ---- Forward MLP ----
        def _forward(params: dict, obs: jax.Array) -> jax.Array:
            """MLP: dense → ELU → dense → … → dense (linear output)."""
            x = obs
            for i in range(self._num_layers):
                x = x @ params[f"W{i}"] + params[f"b{i}"]
                if i < self._num_layers - 1:
                    x = jax.nn.elu(x)
            return x

        self._forward_fn = _forward

        # ---- Initialize weights (Xavier uniform) ----
        init_rng = jax.random.PRNGKey(seed)
        init_params = {}
        for i in range(self._num_layers):
            fan_in, fan_out = dims[i], dims[i + 1]
            scale = jnp.sqrt(2.0 / (fan_in + fan_out))
            w_key, init_rng = jax.random.split(init_rng)
            init_params[f"W{i}"] = jax.random.normal(w_key, (fan_in, fan_out)) * scale
            init_params[f"b{i}"] = jnp.zeros(fan_out)

        # ---- Optimizer ----
        self._tx = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.adam(lr),
        )
        self._opt_state = self._tx.init(init_params)

        # ---- JIT-compiled update function ----
        @jax.jit
        def _step(params: dict, opt_state: optax.OptState,
                   obs: jax.Array, next_obs: jax.Array) -> tuple[dict, optax.OptState, jax.Array]:
            """One training step: predict, compute error, update weights."""
            pred = self._forward_fn(params, obs)
            error = pred - next_obs
            loss = jnp.mean(error ** 2)
            grad_fn = jax.grad(lambda p: jnp.mean((self._forward_fn(p, obs) - next_obs) ** 2))
            grads = grad_fn(params)
            updates, new_opt = self._tx.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            rmse = jnp.sqrt(loss)
            return new_params, new_opt, rmse

        self._step = _step
        self._params = init_params

    def update(self, obs: np.ndarray, next_obs: np.ndarray) -> float:
        """Train forward model and return intrinsic reward (prediction error).

        Parameters
        ----------
        obs : np.ndarray
            Current flattened observation.
        next_obs : np.ndarray
            Next flattened observation.

        Returns
        -------
        float
            Intrinsic reward: ``intrinsic_scale * prediction_error``.
            High for novel/unpredictable transitions, near zero for familiar ones.
        """
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        next_obs_jax = jnp.asarray(next_obs, dtype=jnp.float32)

        self._params, self._opt_state, rmse = self._step(
            self._params, self._opt_state, obs_jax, next_obs_jax
        )

        intrinsic_reward = float(rmse) * self._intrinsic_scale
        return intrinsic_reward

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict next observation from current observation (no training)."""
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)
        pred = self._forward_fn(self._params, obs_jax)
        return np.asarray(pred)