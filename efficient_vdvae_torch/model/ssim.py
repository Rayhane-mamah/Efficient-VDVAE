"""Multi-GPU supported ssim function (heavily inspired by original tf.image implementation):
https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/ops/image_ops_impl.py#L3236
"""
import torch
import torch.nn.functional as F

import numpy as np

try:
    from ..utils.utils import get_same_padding

except (ImportError, ValueError):
    from utils.utils import get_same_padding


class SSIM(torch.nn.Module):
    def __init__(self, image_channels, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
        super(SSIM, self).__init__()
        self.max_val = max_val

        self.k1 = k1
        self.k2 = k2
        self.filter_size = filter_size

        self.compensation = 1.

        self.kernel = SSIM._fspecial_gauss(filter_size, filter_sigma, image_channels)

    @staticmethod
    def _fspecial_gauss(filter_size, filter_sigma, image_channels):
        """Function to mimic the 'fspecial' gaussian MATLAB function."""
        coords = torch.arange(0, filter_size, dtype=torch.float32)
        coords -= (filter_size - 1.0) / 2.0

        g = torch.square(coords)
        g *= -0.5 / np.square(filter_sigma)

        g = torch.reshape(g, shape=(1, -1)) + torch.reshape(g, shape=(-1, 1))
        g = torch.reshape(g, shape=(1, -1))  # For tf.nn.softmax().
        g = F.softmax(g, dim=-1)
        g = torch.reshape(g, shape=(1, 1, filter_size, filter_size))
        return torch.tile(g, (image_channels, 1, 1, 1)).cuda()  # out_c, in_c // groups, h, w

    def _apply_filter(self, x):
        shape = list(x.size())
        x = torch.reshape(x, shape=[-1] + shape[-3:])  # b , c , h , w
        y = F.conv2d(x, weight=self.kernel, stride=1, padding=(self.filter_size - 1) // 2,
                     groups=x.shape[1])  # b, c, h, w
        return torch.reshape(y, shape[:-3] + list(y.size()[1:]))

    def _compute_luminance_contrast_structure(self, x, y):
        c1 = (self.k1 * self.max_val) ** 2
        c2 = (self.k2 * self.max_val) ** 2

        # SSIM luminance measure is
        # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
        mean0 = self._apply_filter(x)
        mean1 = self._apply_filter(y)
        num0 = mean0 * mean1 * 2.0
        den0 = torch.square(mean0) + torch.square(mean1)
        luminance = (num0 + c1) / (den0 + c1)

        # SSIM contrast-structure measure is
        #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
        # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
        #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
        #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
        num1 = self._apply_filter(x * y) * 2.0
        den1 = self._apply_filter(torch.square(x) + torch.square(y))
        c2 *= self.compensation
        cs = (num1 - num0 + c2) / (den1 - den0 + c2)

        # SSIM score is the product of the luminance and contrast-structure measures.
        return luminance, cs

    def _compute_one_channel_ssim(self, x, y):
        luminance, contrast_structure = self._compute_luminance_contrast_structure(x, y)
        return (luminance * contrast_structure).mean(dim=(-2, -1))

    def forward(self, targets, outputs):
        ssim_per_channel = self._compute_one_channel_ssim(targets, outputs)
        return ssim_per_channel.mean(dim=-1)
