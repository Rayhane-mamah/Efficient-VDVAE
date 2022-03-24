import jax.numpy as jnp
from jax import random
from flax import linen as nn
from hparams import HParams

try:
    from .conv2d import Conv2D, uniform_init

except (ImportError, ValueError):
    from model.conv2d import Conv2D, uniform_init

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class GaussianLatentLayer(nn.Module):
    num_variates: int

    @nn.compact
    def __call__(self, key, x, prior_stats=None, return_sample=True, temperature=None):
        x = Conv2D(filters=self.num_variates * 2,
                   kernel_size=1,
                   strides=1,
                   padding='SAME',
                   name='projection')(x)

        if hparams.model.distribution_base == 'std':
            mean, std, stats = _std_mode(x, prior_stats)
        elif hparams.model.distribution_base == 'logstd':
            mean, std, stats = _logstd_mode(x, prior_stats)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        if temperature is not None:
            std = std * temperature

        if return_sample:
            eps = random.normal(key=key, shape=mean.shape, dtype=mean.dtype)
            z = eps * std + mean
            return z, stats

        return stats


def beta_softplus(x, beta):
    r"""Softplus activation function with additional beta parameter.

      Computes the element-wise function

      .. math::
        \mathrm{softplus}(x, beta) = \frac{1}{beta} * \log(1 + e^(beta * x))

      Args:
        x : input array
        beta : parameter of the softplus. Controls the smoothness of transition from 0 gradient to 1 gradient
      """
    return 1 / beta * jnp.logaddexp(beta * x, 0)


def _std_mode(x, prior_stats):
    """The model predicts std, softplus is used to ensure std is positive and avoid overflow by std denominator in KL div"""
    mean, std = jnp.split(x, indices_or_sections=2, axis=-1)
    std = beta_softplus(std, beta=hparams.model.gradient_smoothing_beta)  # beta=ln(2.) yields std of 1 at x=0 and is less sharp than beta=1

    if prior_stats is not None:
        mean = mean + prior_stats[0]
        std = std * prior_stats[1]

    stats = [mean, std]
    return mean, std, stats


def _logstd_mode(x, prior_stats):
    """The model predicts logstd, which we smooth turn into std"""
    mean, logstd = jnp.split(x, indices_or_sections=2, axis=-1)

    if prior_stats is not None:
        mean = mean + prior_stats[0]
        logstd = logstd + prior_stats[1]

    std = jnp.exp(hparams.model.gradient_smoothing_beta * logstd)
    stats = [mean, logstd]
    return mean, std, stats
