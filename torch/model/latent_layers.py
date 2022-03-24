import numpy as np
import torch
import torch.nn as nn

try:
    from ..utils.utils import get_same_padding
    from .conv2d import Conv2d

except (ImportError, ValueError):
    from utils.utils import get_same_padding
    from model.conv2d import Conv2d

from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class GaussianLatentLayer(nn.Module):
    def __init__(self, in_filters, num_variates, min_std=np.exp(-2)):
        super(GaussianLatentLayer, self).__init__()

        self.projection = Conv2d(
            in_channels=in_filters,
            out_channels=num_variates * 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )

        self.min_std = min_std
        self.softplus = torch.nn.Softplus(beta=np.log(2))

    def forward(self, x, temperature=None, prior_stats=None, return_sample=True):
        x = self.projection(x)
        mean, std = torch.chunk(x, chunks=2, dim=1)  # B, C, H, W

        std = self.softplus(std)
        if prior_stats is not None:
            mean = mean + prior_stats[0]
            std = std * prior_stats[1]

        stats = [mean, std]
        if temperature is not None:
            std = std * temperature

        if return_sample:
            z, mean, std = calculate_z(mean, std)
            return z, stats
        return stats


@torch.jit.script
def calculate_z(mean, std):
    eps = torch.empty_like(mean, device=torch.device('cuda')).normal_(0., 1.)
    z = eps * std + mean
    return z, mean, std
