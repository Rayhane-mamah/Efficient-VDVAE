import jax
import jax.numpy as jnp
from optax._src import base
from optax._src import numerics
from optax._src.transform import _update_moment, _bias_correction, ScaleByAdamState


def _update_infinite_moment(updates, moments, decay, eps):
    """Compute the exponential moving average of the infinite moment."""
    return jax.tree_map(
        lambda g, t: jnp.maximum(decay * t, jnp.abs(g) + eps), updates, moments)  # max(β2 · ut−1, |gt|)


def scale_by_adamax(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
) -> base.GradientTransformation:
    """Rescale updates according to the Adamax algorithm.

    References:
      [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Args:
      b1: decay rate for the exponentially weighted average of grads.
      b2: decay rate for the exponentially weighted average of squared grads.
      eps: term added to the denominator to improve numerical stability.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)  # First moment
        nu = jax.tree_map(jnp.zeros_like, params)  # Infinite moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = _update_moment(updates, state.mu, b1, 1)
        nu = _update_infinite_moment(updates, state.nu, b2, eps)  # No bias correction for infinite moment
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = _bias_correction(mu, b1, count_inc)
        updates = jax.tree_map(
            lambda m, v: m / v, mu_hat, nu)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)
