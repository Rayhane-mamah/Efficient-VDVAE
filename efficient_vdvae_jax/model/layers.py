import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.linen.initializers import glorot_uniform, zeros
from typing import Any, Union
import numpy as np

from hparams import HParams

try:
    from .conv2d import Conv2D, stable_init
    from .latent_layers import GaussianLatentLayer

except (ImportError, ValueError):
    from model.conv2d import Conv2D, stable_init
    from model.latent_layers import GaussianLatentLayer

hparams = HParams.get_hparams_by_name("efficient_vdvae")

Array = Any


class UnpoolLayer(nn.Module):
    filters: int
    strides: int

    @nn.compact
    def __call__(self, x):
        B, H, W, _ = x.shape
        x = Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='SAME', name='conv')(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        x = jax.image.resize(x, shape=[B, H * self.strides, W * self.strides, x.shape[-1]], method='nearest')

        scale_bias_param = self.param('scale_bias',
                                      zeros,
                                      (1, H * self.strides, W * self.strides, x.shape[-1]))
        x = x + jnp.asarray(scale_bias_param, dtype=x.dtype)

        return x


class PoolLayer(nn.Module):
    filters: int
    strides: int

    @nn.compact
    def __call__(self, x):
        x = Conv2D(filters=self.filters, kernel_size=self.strides, strides=self.strides, padding='SAME', name='conv')(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        return x


class ProjectionWrapper(nn.Module):
    """Wrapper on top of Conv2D for when the filter_size is not available in the setup stage.
     Maps an input 'x' to the shape 'filters' using a 1x1 conv.
     Both 'x' and 'filters' are provided during the call and not during the initialization of the layer.

     Mainly used to overcome the limitation of flax.linen to provide the inputs' shapes during the setup stage."""

    @nn.compact
    def __call__(self, x, filters):
        return Conv2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='SAME',
            kernel_init=stable_init(scale=np.sqrt(1. / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)))),
            name=self.name
        )(x)


class ResidualConvCell(nn.Module):
    """Optionally residual conv cell, the output is always input_filters * output_ratio.
    If residual=True then output_ratio must be equal to 1."""
    n_layers: int
    bottleneck_ratio: int
    kernel_size: int
    init_scaler: float
    residual: bool = True
    output_ratio: Union[int, float] = 1

    @nn.compact
    def __call__(self, dropout_rng, inputs, training):
        if self.residual:
            # The output filters need to be equal to input filters for residual connection
            assert self.output_ratio == 1

        filters = int(inputs.shape[-1] * self.output_ratio)
        bottleneck_filters = int(inputs.shape[-1] * self.bottleneck_ratio)

        x = inputs

        for i in range(self.n_layers + 2):
            x = nn.swish(x)
            x = Conv2D(filters=filters if i == self.n_layers + 1 else bottleneck_filters,
                       kernel_size=1 if (i == 0 or i == self.n_layers + 1) and hparams.model.use_1x1_conv else self.kernel_size,
                       strides=1, padding='SAME',
                       kernel_init=stable_init(scale=self.init_scaler) if i == self.n_layers + 1 else glorot_uniform(), name=f'conv_{i}')(x)

        if self.residual:
            outputs = inputs + x
        else:
            outputs = x

        return outputs


class LevelBlockUp(nn.Module):
    n_blocks: int
    n_layers: int
    filters: int
    bottleneck_ratio: int
    kernel_size: int
    strides: int
    skip_filters: int

    @nn.compact
    def __call__(self, key, x, use_skip, training):
        # Pre-skip block
        dropout_rng = random.split(key, num=self.n_blocks)
        for i in range(self.n_blocks):
            x = ResidualConvCell(n_layers=self.n_layers,
                                 bottleneck_ratio=self.bottleneck_ratio,
                                 kernel_size=self.kernel_size,
                                 init_scaler=np.sqrt(1. / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))) if hparams.model.stable_init else 1.,
                                 name=f'residual_block_{i}')(dropout_rng[i], x, training)

        # Skip connection from bottom-up used to compute z
        if use_skip:
            skip_output = Conv2D(filters=self.skip_filters, kernel_size=1, strides=1, padding='SAME', name='skip_projection')(x)
        else:
            skip_output = x

        # In downsample last, this block is the last block of the bigger resolution
        if self.strides > 1:
            x = PoolLayer(self.filters, self.strides, name='pooling_layer')(x)

        return x, skip_output


class LevelBlockDown(nn.Module):
    n_blocks: int
    n_layers: int
    filters: int
    bottleneck_ratio: int
    kernel_size: int
    strides: int
    latent_variates: int

    def setup(self):
        if self.strides > 1:
            self.unpool_layer = UnpoolLayer(filters=self.filters, strides=self.strides, name='unpooling_layer')

        residual_block = []
        for i in range(self.n_blocks):
            residual_block.append(ResidualConvCell(
                n_layers=self.n_layers,
                bottleneck_ratio=self.bottleneck_ratio,
                kernel_size=self.kernel_size,
                init_scaler=np.sqrt(1. / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))) if hparams.model.stable_init else 1.,
                name=f'residual_block_{i}'
            ))

        self.residual_block = residual_block

        self.posterior_net = ResidualConvCell(
            n_layers=self.n_layers,
            bottleneck_ratio=self.bottleneck_ratio * 0.5,  # the input is 2* filters (concat of skip_x and y)
            kernel_size=self.kernel_size,
            init_scaler=1.,
            residual=False,
            output_ratio=0.5,  # the input is 2* filters
            name='posterior_net'
        )
        self.prior_net = ResidualConvCell(
            n_layers=self.n_layers,
            bottleneck_ratio=self.bottleneck_ratio,
            kernel_size=self.kernel_size,
            init_scaler=0. if hparams.model.initialize_prior_weights_as_zero else 1.,
            residual=False,
            output_ratio=2,
            name='prior_net'
        )

        self.prior_layer = GaussianLatentLayer(
            num_variates=self.latent_variates,
            name='prior_gaussian_layer'
        )
        self.posterior_layer = GaussianLatentLayer(
            num_variates=self.latent_variates,
            name='posterior_gaussian_layer'
        )

        self.z_projection = ProjectionWrapper(name='z_projection')

    def sampler(self, latent_fn, key, y, prior_stats=None, temperature=None):
        z, dist = latent_fn(key, y, prior_stats=prior_stats, temperature=temperature)
        return z, dist

    def get_analytical_distribution(self, latent_fn, key, y, prior_stats=None):
        dist = latent_fn(key, y, prior_stats=prior_stats, return_sample=False)
        return dist

    def __call__(self, key, x_skip, y, variate_mask, training):
        # In upsample first, this block is the first block of the bigger resolution
        if self.strides > 1:
            y = self.unpool_layer(y)

        key, pr_dropout_key, po_dropout_key, *re_dropout_key = random.split(key, num=3 + len(self.residual_block))
        kl_residual, y_prior_kl = jnp.split(self.prior_net(pr_dropout_key, y, training), indices_or_sections=2, axis=-1)

        y_post = jnp.concatenate([y, x_skip], axis=-1)
        y_post = self.posterior_net(po_dropout_key, y_post, training)

        # Prior under expected value of q(z<i|x)
        if variate_mask is not None:
            # Sample z from the prior distribution
            z_prior_kl, prior_kl_dist = self.sampler(self.prior_layer, key, y_prior_kl)
        else:
            prior_kl_dist = self.get_analytical_distribution(self.prior_layer, key, y_prior_kl)

        # Sample posterior under expected value of q(z<i|x)
        z_post, posterior_dist = self.sampler(self.posterior_layer, key, y_post,
                                              prior_stats=prior_kl_dist if hparams.model.use_residual_distribution else None)

        if variate_mask is not None:
            # Only used in inference mode to prune turned-off variates
            # Use posterior sample from meaningful variates, and prior sample from "turned-off" variates
            # The NLL should be similar to using z_post without masking if the mask is good (not very destructive)
            # variate_mask automatically broadcasts to [batch_size, H, W, n_variates]
            z_post = variate_mask * z_post + (1. - variate_mask) * z_prior_kl

        # Residual with prior
        y = y + kl_residual

        # Project z and merge back into main stream
        proj_z_post = self.z_projection(z_post, filters=y.shape[-1])
        y = y + proj_z_post

        # Residual block
        for i, layer in enumerate(self.residual_block):
            y = layer(re_dropout_key[i], y, training)

        return y, posterior_dist, prior_kl_dist

    def sample_from_prior(self, key, y, training, temperature):
        # In upsample first, this block is the first block in the bigger resolution
        if self.strides > 1:
            y = self.unpool_layer(y)

        key, pr_dropout_key, *re_dropout_key = random.split(key, num=2 + len(self.residual_block))

        kl_residual, y_prior = jnp.split(self.prior_net(pr_dropout_key, y, training), indices_or_sections=2, axis=-1)
        z, _ = self.sampler(self.prior_layer, key, y_prior, temperature=temperature)

        y = y + kl_residual

        proj_z = self.z_projection(z, filters=y.shape[-1])
        y = y + proj_z

        # Residual block
        for i, layer in enumerate(self.residual_block):
            y = layer(re_dropout_key[i], y, training)

        return y, z
