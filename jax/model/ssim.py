import jax
import jax.lax as lax
import jax.numpy as jnp
from flax import linen as nn
from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class StructureSimilarityIndexMap:
    def __init__(self, image_channels, denormalizer, unnormalized_max=255., filter_size=11):
        self.denormalizer = denormalizer
        self.ssim = SSIM(image_channels=image_channels, max_val=unnormalized_max, filter_size=filter_size)

    def __call__(self, targets, outputs, global_batch_size):
        if hparams.data.dataset_source == 'binarized_mnist':
            return 0

        else:
            @jax.jit
            def compute_ssim(t, o):
                t = self.denormalizer(t).astype(jnp.float32)
                o = self.denormalizer(o).astype(jnp.float32)

                assert t.shape == o.shape
                per_example_ssim = self.ssim(t, o)

                mean_axis = tuple(range(1, len(per_example_ssim.shape)))
                per_example_ssim = jnp.mean(per_example_ssim, axis=mean_axis)
                assert len(per_example_ssim.shape) == 1

                return jnp.sum(per_example_ssim) / global_batch_size

            return compute_ssim(targets, outputs)


class SSIM:
    def __init__(self, image_channels, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        self.max_val = max_val

        filter_size = filter_size
        filter_sigma = jnp.float32(filter_sigma)

        self.k1 = k1
        self.k2 = k2

        self.compensation = 1.

        self.kernel = SSIM._fspecial_gauss(filter_size, filter_sigma, image_channels)

    @staticmethod
    def _fspecial_gauss(filter_size, filter_sigma, image_channels):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        coords = jnp.arange(filter_size, dtype=filter_sigma.dtype)
        coords -= (filter_size - 1.) / 2.

        g = jnp.square(coords)
        g *= -0.5 / jnp.square(filter_sigma)

        g = jnp.reshape(g, newshape=(1, -1)) + jnp.reshape(g, newshape=(-1, 1))
        g = jnp.reshape(g, newshape=[1, -1])  # For tf.nn.softmax().
        g = nn.softmax(g)
        g = jnp.reshape(g, newshape=[filter_size, filter_size, 1, 1])
        return jnp.tile(g, [1, 1, 1, image_channels])  # kH, kW, 1, input_filters * multiplier  for depthwise conv

    def _apply_filter(self, x):
        shape = x.shape  # B, H, W, C
        x = jnp.reshape(x, newshape=(-1,) + shape[-3:])
        # to be a depthwise conv, use n_groups equal to input filters
        y = lax.conv_general_dilated(lhs=x, rhs=self.kernel, window_strides=(1, 1), padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                     feature_group_count=x.shape[-1])  # B, H, W, C
        return jnp.reshape(y, newshape=shape[:-3] + y.shape[1:])

    def _compute_luminance_contrast_structure(self, x, y):
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # SSIM luminance measure is
        # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
        mean0 = self._apply_filter(x)
        mean1 = self._apply_filter(y)
        num0 = mean0 * mean1 * 2.0
        den0 = jnp.square(mean0) + jnp.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)

        # SSIM contrast-structure measure is
        #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
        # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
        #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
        #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
        num1 = self._apply_filter(x * y) * 2.0
        den1 = self._apply_filter(jnp.square(x) + jnp.square(y))
        c2 *= self.compensation
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        # SSIM score is the product of the luminance and contrast-structure measures.
        return luminance, cs

    def _compute_one_channel_ssim(self, x, y):
        luminance, contrast_structure = self._compute_luminance_contrast_structure(x, y)
        return jnp.mean(luminance * contrast_structure, axis=(-3, -2))

    def __call__(self, targets, outputs):
        ssim_per_channel = self._compute_one_channel_ssim(targets, outputs)
        return jnp.mean(ssim_per_channel, axis=-1)
