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


def _std_mode(x, prior_stats, softplus):
    mean, std = torch.chunk(x, chunks=2, dim=1)  # B, C, H, W
    std = softplus(std)

    if prior_stats is not None:
        mean = mean + prior_stats[0]
        std = std * prior_stats[1]

    stats = [mean, std]
    return mean, std, stats


def _logstd_mode(x, prior_stats):
    mean, logstd = torch.chunk(x, chunks=2, dim=1)

    if prior_stats is not None:
        mean = mean + prior_stats[0]
        logstd = logstd + prior_stats[1]

    std = torch.exp(hparams.model.gradient_smoothing_beta * logstd)
    stats = [mean, logstd]

    return mean, std, stats


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
        self.softplus = torch.nn.Softplus(beta=hparams.model.gradient_smoothing_beta)

    def forward(self, x, temperature=None, prior_stats=None, return_sample=True):
        x = self.projection(x)

        if hparams.mode.distribution_base == 'std':
            mean, std, stats = _std_mode(x, prior_stats, self.softplus)
        elif hparams.mode.distribution_base == 'logstd':
            mean, std, stats = _logstd_mode(x, prior_stats)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

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
