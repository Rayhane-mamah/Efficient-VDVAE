import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
import typing
import numpy as np
from PIL import Image
from hparams import HParams
import torch_optimizer as optim
from typing import List

try:
    from .losses import KLDivergence
except (ImportError, ValueError):
    from model.losses import KLDivergence

hparams = HParams.get_hparams_by_name("efficient_vdvae")


@torch.jit.script
def calculate_std_loss(p: List[torch.Tensor], q: List[torch.Tensor]):
    q_std = q[1]
    p_std = p[1]
    term1 = (p[0] - q[0]) / q_std
    term2 = p_std / q_std
    loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    return loss


def calculate_logstd_loss(p, q):
    q_logstd = q[1]
    p_logstd = p[1]

    p_std = torch.exp(hparams.model.gradient_smoothing_beta * p_logstd)
    inv_q_std = torch.exp(-hparams.model.gradient_smoothing_beta * q_logstd)

    term1 = (p[0] - q[0]) * inv_q_std
    term2 = p_std * inv_q_std
    loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
    return loss


class KLDivergenceStats(KLDivergence):
    """
    Compute Kl div per dimension (variate)

    """

    def avg_loss_compute(self, p, q, global_batch_size):
        if hparams.model.distribution_base == 'std':
            loss = calculate_std_loss(p, q)
        elif hparams.model.distribution_base == 'logstd':
            loss = calculate_logstd_loss(p, q)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        mean_axis = list(range(2, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)
        n_mean_elems = np.prod([loss.size()[a] for a in mean_axis])  # heads * h * w  or h * w
        avg_per_example_loss = per_example_loss / n_mean_elems
        assert len(per_example_loss.shape) == 2

        # [n_variates,]
        avg_loss = torch.sum(avg_per_example_loss, dim=0) / (
                global_batch_size * np.log(2))  # divide by ln(2) to convert to KL rate (average space bits/dim)

        return avg_loss

    def forward(self, p_dist, q_dist, global_batch_size):
        kls = []

        for p, q in zip(p_dist, q_dist):
            kl_loss = self.avg_loss_compute(p=p, q=q, global_batch_size=global_batch_size)
            kls.append(kl_loss)

        return torch.stack(kls, dim=0)  # n_layers, n_variates
