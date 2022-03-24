import jax.numpy as jnp
from jax import random
from flax.linen.initializers import zeros
from flax import linen as nn
from hparams import HParams

try:
    from .layers import LevelBlockUp, LevelBlockDown, ResidualConvCell
    from .conv2d import Conv2D
    from ..utils.utils import compute_latent_dimension

except (ImportError, ValueError):
    from model.layers import LevelBlockUp, LevelBlockDown, ResidualConvCell
    from model.conv2d import Conv2D
    from utils.utils import compute_latent_dimension

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class BottomUp(nn.Module):

    @nn.compact
    def __call__(self, key, x, training):
        # Input conv
        x = Conv2D(filters=hparams.model.input_conv_filters,
                   kernel_size=hparams.model.input_kernel_size,
                   strides=1,
                   padding='SAME',
                   name='input_conv')(x)

        # Bottom up blocks
        skip_list = []
        for i, stride in enumerate(hparams.model.up_strides):
            key, downsample_key, *res_keys = random.split(key, num=hparams.model.up_n_blocks_per_res[i] + 2)

            for j in range(hparams.model.up_n_blocks_per_res[i]):
                x, _ = LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                    n_layers=hparams.model.up_n_layers[i],
                                    filters=hparams.model.up_filters[i],
                                    bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                    kernel_size=hparams.model.up_kernel_size[i],
                                    strides=1,
                                    skip_filters=hparams.model.up_skip_filters[i],
                                    name=f'block_up_level_{i}_{j}')(res_keys[j], x, use_skip=False, training=training)

            x, skip_out = LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                       n_layers=hparams.model.up_n_layers[i],
                                       filters=hparams.model.up_filters[i],
                                       bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                       kernel_size=hparams.model.up_kernel_size[i],
                                       strides=stride,
                                       skip_filters=hparams.model.up_skip_filters[i],
                                       name=f'block_up_level_{i}_downsample')(downsample_key, x, use_skip=True, training=training)

            skip_list.append(skip_out)

        return skip_list[::-1]  # Reverse to match order of top-down model


class TopDown(nn.Module):
    def setup(self):
        H = W = compute_latent_dimension()

        # 1, H // strides, W // strides, C_dec
        self.h = self.param('trainable_h',
                            zeros,
                            (1, H, W, hparams.model.down_filters[0]))

        upsample_blocks_list = []
        block_sets_list = []
        for i, stride in enumerate(hparams.model.down_strides):
            upsample_blocks_list.append(
                LevelBlockDown(n_blocks=hparams.model.down_n_blocks[i],
                               n_layers=hparams.model.down_n_layers[i],
                               filters=hparams.model.down_filters[i],
                               bottleneck_ratio=hparams.model.down_mid_filters_ratio[i],
                               kernel_size=hparams.model.down_kernel_size[i],
                               strides=stride,
                               latent_variates=hparams.model.down_latent_variates[i],
                               name=f'block_down_level_{i}_upsample')
            )

            block_sets_list.append(
                [LevelBlockDown(n_blocks=hparams.model.down_n_blocks[i],
                                n_layers=hparams.model.down_n_layers[i],
                                filters=hparams.model.down_filters[i],
                                bottleneck_ratio=hparams.model.down_mid_filters_ratio[i],
                                kernel_size=hparams.model.down_kernel_size[i],
                                strides=1,
                                latent_variates=hparams.model.down_latent_variates[i],
                                name=f'block_down_level_{i}_{j}')
                 for j in range(hparams.model.down_n_blocks_per_res[i])]
            )

        self.upsample_blocks_list = upsample_blocks_list
        self.block_sets_list = block_sets_list

        self.output_layer = Conv2D(
            filters=hparams.data.channels if hparams.data.dataset_source == 'binarized_mnist' else hparams.model.num_output_mixtures * (3 * hparams.data.channels + 1),
            kernel_size=hparams.model.output_kernel_size,
            strides=1,
            padding='SAME',
            name='output_conv'
        )

    def __call__(self, key, skip_list, variate_masks, training):
        y = jnp.tile(self.h, [skip_list[0].shape[0], 1, 1, 1])

        posterior_dist_list = []
        prior_kl_dist_list = []
        layer_idx = 0
        for i, (upsample_block, block_set, skip_input) in enumerate(zip(self.upsample_blocks_list, self.block_sets_list, skip_list)):
            key, upsample_key, *res_keys = random.split(key, num=len(block_set) + 2)
            y, posterior_dist, prior_kl_dist = upsample_block(upsample_key, skip_input, y, variate_mask=variate_masks[layer_idx], training=training)
            layer_idx += 1

            resolution_posterior_dist = [posterior_dist]
            resolution_prior_kl_dist = [prior_kl_dist]
            for j, block in enumerate(block_set):
                y, posterior_dist, prior_kl_dist = block(res_keys[j], skip_input, y, variate_mask=variate_masks[layer_idx], training=training)
                layer_idx += 1

                resolution_posterior_dist.append(posterior_dist)
                resolution_prior_kl_dist.append(prior_kl_dist)

            posterior_dist_list += resolution_posterior_dist
            prior_kl_dist_list += resolution_prior_kl_dist

        y = self.output_layer(y)
        return y, posterior_dist_list, prior_kl_dist_list

    def sample_from_prior(self, key, batch_size, training, temperatures):
        y = jnp.tile(self.h, [batch_size, 1, 1, 1])

        prior_zs = []
        for i, (upsample_block, block_set, temperature) in enumerate(zip(self.upsample_blocks_list, self.block_sets_list, temperatures)):
            # Temperatures are assumed to be equal in-between skip connections.
            key, upsample_key, *res_keys = random.split(key, num=len(block_set) + 2)
            y, z = upsample_block.sample_from_prior(upsample_key, y, training=training, temperature=temperature)

            level_z = [z]
            for j, block in enumerate(block_set):
                y, z = block.sample_from_prior(res_keys[j], y, training=training, temperature=temperature)
                level_z.append(z)

            prior_zs += level_z

        y = self.output_layer(y)
        return y, prior_zs
