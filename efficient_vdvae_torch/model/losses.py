import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from .ssim import SSIM
    from ..utils.utils import scale_pixels, effective_pixels
except (ImportError, ValueError):
    from model.ssim import SSIM
    from utils.utils import scale_pixels, effective_pixels

from hparams import HParams
from typing import List
from torch.distributions.bernoulli import Bernoulli

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class BernoulliLoss(nn.Module):
    def __init__(self):
        super(BernoulliLoss, self).__init__()

    def forward(self, targets, logits, global_batch_size):
        targets = targets[:, :, 2:30, 2:30]
        logits = logits[:, :, 2:30, 2:30]

        # recon = torch.squeeze(torch.maximum(logits, torch.as_tensor(0.)) - logits * targets + torch.log(torch.as_tensor(1.) + torch.exp(-torch.abs(logits))), dim=1)
        loss_value = Bernoulli(logits=logits)
        recon = loss_value.log_prob(targets)
        mean_axis = list(range(1, len(recon.size())))
        per_example_loss = - torch.sum(recon, dim=mean_axis)
        avg_per_example_loss = per_example_loss / (
            np.prod([recon.size()[i] for i in mean_axis]))  # B
        scalar = global_batch_size * effective_pixels()
        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(
            2))  # divide by ln(2) to convert to bit range (for visualization purposes only)
        if hparams.data.dataset_source == 'binarized_mnist':
            # Convert from bpd to nats for comparison
            avg_loss = avg_loss * np.log(2.) * effective_pixels()
        model_means, log_scales = None, None
        return loss, avg_loss, model_means, log_scales


def _compute_inv_stdv(logits):
    softplus = nn.Softplus(beta=hparams.model.output_gradient_smoothing_beta)
    if hparams.model.output_distribution_base == 'std':
        scales = torch.maximum(softplus(logits),
                               torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))
        inv_stdv = 1. / scales  # Not stable for sharp distributions
        log_scales = torch.log(scales)

    elif hparams.model.output_distribution_base == 'logstd':
        log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
        inv_stdv = torch.exp(-hparams.model.output_gradient_smoothing_beta * log_scales)
    else:
        raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')

    return inv_stdv, log_scales


class DiscMixLogistic(nn.Module):
    def __init__(self):
        super(DiscMixLogistic, self).__init__()

        # Only works for when images are [0., 1.] normalized for now
        self.num_classes = 2. ** hparams.data.num_bits - 1.
        self.min_pix_value = scale_pixels(0.)
        self.max_pix_value = scale_pixels(255.)

    def forward(self, targets, logits, global_batch_size):
        # Shapes:
        #    targets: B, C, H, W
        #    logits: B, M * (3 * C + 1), H, W

        assert len(targets.shape) == 4
        B, C, H, W = targets.size()
        assert C == 3  # only support RGB for now

        targets = targets.unsqueeze(2)  # B, C, 1, H, W

        logit_probs = logits[:, :hparams.model.num_output_mixtures, :, :]  # B, M, H, W
        l = logits[:, hparams.model.num_output_mixtures:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(B, hparams.data.channels, 3 * hparams.model.num_output_mixtures, H, W)  # B, C, 3 * M, H, W

        model_means = l[:, :, :hparams.model.num_output_mixtures, :, :]  # B, C, M, H, W

        inv_stdv, log_scales = _compute_inv_stdv(
            l[:, :, hparams.model.num_output_mixtures: 2 * hparams.model.num_output_mixtures, :, :])

        model_coeffs = torch.tanh(
            l[:, :, 2 * hparams.model.num_output_mixtures: 3 * hparams.model.num_output_mixtures, :,
            :])  # B, C, M, H, W

        # RGB AR
        mean1 = model_means[:, 0:1, :, :, :]  # B, 1, M, H, W
        mean2 = model_means[:, 1:2, :, :, :] + model_coeffs[:, 0:1, :, :, :] * targets[:, 0:1, :, :, :]  # B, 1, M, H, W
        mean3 = model_means[:, 2:3, :, :, :] + model_coeffs[:, 1:2, :, :, :] * targets[:, 0:1, :, :, :] + model_coeffs[
                                                                                                          :, 2:3, :, :,
                                                                                                          :] * targets[
                                                                                                               :, 1:2,
                                                                                                               :, :,
                                                                                                               :]  # B, 1, M, H, W
        means = torch.cat([mean1, mean2, mean3], dim=1)  # B, C, M, H, W
        centered = targets - means  # B, C, M, H, W

        plus_in = inv_stdv * (centered + 1. / self.num_classes)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.num_classes)
        cdf_min = torch.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255 (before scaling)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min  # B, C, M, H, W

        mid_in = inv_stdv * centered
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        broadcast_targets = torch.broadcast_to(targets, size=[B, C, hparams.model.num_output_mixtures, H, W])
        log_probs = torch.where(broadcast_targets == self.min_pix_value, log_cdf_plus,
                                torch.where(broadcast_targets == self.max_pix_value, log_one_minus_cdf_min,
                                            torch.where(cdf_delta > 1e-5,
                                                        torch.log(torch.clamp(cdf_delta, min=1e-12)),
                                                        log_pdf_mid - np.log(self.num_classes / 2))))  # B, C, M, H, W

        log_probs = torch.sum(log_probs, dim=1) + F.log_softmax(logit_probs, dim=1)  # B, M, H, W
        negative_log_probs = -torch.logsumexp(log_probs, dim=1)  # B, H, W

        mean_axis = list(range(1, len(negative_log_probs.size())))
        per_example_loss = torch.sum(negative_log_probs, dim=mean_axis)  # B
        avg_per_example_loss = per_example_loss / (
                np.prod([negative_log_probs.size()[i] for i in mean_axis]) * hparams.data.channels)  # B

        assert len(per_example_loss.size()) == len(avg_per_example_loss.size()) == 1

        scalar = global_batch_size * hparams.data.target_res * hparams.data.target_res * hparams.data.channels

        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(avg_per_example_loss) / (global_batch_size * np.log(
            2))  # divide by ln(2) to convert to bit range (for visualization purposes only)

        return loss, avg_loss, model_means, log_scales


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


class KLDivergence(nn.Module):
    def forward(self, p, q, global_batch_size):
        if hparams.model.distribution_base == 'std':
            loss = calculate_std_loss(p, q)
        elif hparams.model.distribution_base == 'logstd':
            loss = calculate_logstd_loss(p, q)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        mean_axis = list(range(1, len(loss.size())))
        per_example_loss = torch.sum(loss, dim=mean_axis)
        n_mean_elems = np.prod([loss.size()[a] for a in mean_axis])  # heads * h * w  or h * w * z_dim
        avg_per_example_loss = per_example_loss / n_mean_elems

        assert len(per_example_loss.shape) == 1

        scalar = global_batch_size * effective_pixels()

        loss = torch.sum(per_example_loss) / scalar
        avg_loss = torch.sum(avg_per_example_loss) / (
                global_batch_size * np.log(2))  # divide by ln(2) to convert to KL rate (average space bits/dim)

        return loss, avg_loss


class StructureSimilarityIndexMap(nn.Module):
    def __init__(self, image_channels, unnormalized_max=255., filter_size=11):
        super(StructureSimilarityIndexMap, self).__init__()
        self.ssim = SSIM(image_channels=image_channels, max_val=unnormalized_max, filter_size=filter_size)

    def forward(self, targets, outputs, global_batch_size):
        if hparams.data.dataset_source == 'binarized_mnist':
            return 0.
        targets = targets * 127.5 + 127.5
        outputs = outputs * 127.5 + 127.5

        assert targets.size() == outputs.size()
        per_example_ssim = self.ssim(targets, outputs)
        mean_axis = list(range(1, len(per_example_ssim.size())))
        per_example_ssim = torch.sum(per_example_ssim, dim=mean_axis)

        loss = torch.sum(per_example_ssim) / global_batch_size
        return loss
