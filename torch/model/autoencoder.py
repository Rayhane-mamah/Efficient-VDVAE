import torch
import torch.nn as nn
import numpy as np
from hparams import HParams

try:
    from .layers import LevelBlockUp, LevelBlockDown, ResidualConvCell
    from ..utils.utils import one_hot, get_same_padding, compute_latent_dimension
    from .conv2d import Conv2d
    from ..utils.utils import scale_pixels
except (ImportError, ValueError):
    from model.layers import LevelBlockUp, LevelBlockDown, ResidualConvCell
    from utils.utils import one_hot, get_same_padding, compute_latent_dimension
    from model.conv2d import Conv2d
    from utils.utils import scale_pixels

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class BottomUp(torch.nn.Module):
    def __init__(self):
        super(BottomUp, self).__init__()

        in_channels_up = [hparams.model.input_conv_filters] + hparams.model.up_filters[0:-1]

        self.levels_up = nn.ModuleList([])
        self.levels_up_downsample = nn.ModuleList([])

        for i, stride in enumerate(hparams.model.up_strides):
            elements = nn.ModuleList([])
            for j in range(hparams.model.up_n_blocks_per_res[i]):
                elements.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                              n_layers=hparams.model.up_n_layers[i],
                                              in_filters=in_channels_up[i],
                                              filters=hparams.model.up_filters[i],
                                              bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                              kernel_size=hparams.model.up_kernel_size[i],
                                              strides=1,
                                              skip_filters=hparams.model.up_skip_filters[i],
                                              use_skip=False)])

            self.levels_up.extend([elements])

            self.levels_up_downsample.extend([LevelBlockUp(n_blocks=hparams.model.up_n_blocks[i],
                                                           n_layers=hparams.model.up_n_layers[i],
                                                           in_filters=in_channels_up[i],
                                                           filters=hparams.model.up_filters[i],
                                                           bottleneck_ratio=hparams.model.up_mid_filters_ratio[i],
                                                           kernel_size=hparams.model.up_kernel_size[i],
                                                           strides=stride,
                                                           skip_filters=hparams.model.up_skip_filters[i],
                                                           use_skip=True)])

        self.input_conv = Conv2d(in_channels=hparams.data.channels,
                                 out_channels=hparams.model.input_conv_filters,
                                 kernel_size=hparams.model.input_kernel_size,
                                 stride=(1, 1),
                                 padding='same')

    def forward(self, x):
        x = self.input_conv(x)

        skip_list = []

        for i, (level_up, level_up_downsample) in enumerate(zip(self.levels_up, self.levels_up_downsample)):
            for layer in level_up:
                x, _ = layer(x)

            x, skip_out = level_up_downsample(x)
            skip_list.append(skip_out)

        skip_list = skip_list[::-1]

        return skip_list


class TopDown(torch.nn.Module):
    def __init__(self):
        super(TopDown, self).__init__()
        self.min_pix_value = scale_pixels(0.)
        self.max_pix_value = scale_pixels(255.)

        H = W = compute_latent_dimension()

        # 1,  C_dec, H // strides, W // strides
        self.trainable_h = torch.nn.Parameter(data=torch.empty(size=(1, hparams.model.down_filters[0], H, W)),
                                              requires_grad=True)
        nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')

        in_channels_down = [hparams.model.down_filters[0]] + hparams.model.down_filters[0:-1]
        self.levels_down, self.levels_down_upsample = nn.ModuleList([]), nn.ModuleList([])

        for i, stride in enumerate(hparams.model.down_strides):
            self.levels_down_upsample.extend([LevelBlockDown(
                n_blocks=hparams.model.down_n_blocks[i],
                n_layers=hparams.model.down_n_layers[i],
                in_filters=in_channels_down[i],
                filters=hparams.model.down_filters[i],
                bottleneck_ratio=hparams.model.down_mid_filters_ratio[i],
                kernel_size=hparams.model.down_kernel_size[i],
                strides=stride,
                skip_filters=hparams.model.up_skip_filters[::-1][i],
                latent_variates=hparams.model.down_latent_variates[i],
                first_block=i == 0,
                last_block=False
            )])

            self.levels_down.extend([nn.ModuleList(
                [LevelBlockDown(n_blocks=hparams.model.down_n_blocks[i],
                                n_layers=hparams.model.down_n_layers[i],
                                in_filters=hparams.model.down_filters[i],
                                filters=hparams.model.down_filters[i],
                                bottleneck_ratio=hparams.model.down_mid_filters_ratio[i],
                                kernel_size=hparams.model.down_kernel_size[i],
                                strides=1,
                                skip_filters=hparams.model.up_skip_filters[::-1][i],
                                latent_variates=hparams.model.down_latent_variates[i],
                                first_block=False,
                                last_block=i == len(hparams.model.down_strides) - 1 and j ==
                                           hparams.model.down_n_blocks_per_res[i] - 1)
                 for j in range(hparams.model.down_n_blocks_per_res[i])])])

        self.output_conv = Conv2d(in_channels=hparams.model.down_filters[-1],
                                  out_channels=1 if hparams.data.dataset_source == 'binarized_mnist' else
                                  hparams.model.num_output_mixtures * (3 * hparams.data.channels + 1),
                                  kernel_size=hparams.model.output_kernel_size,
                                  stride=(1, 1), padding='same')

    def sample(self, logits):
        if hparams.data.dataset_source == 'binarized_mnist':
            return self._sample_from_bernoulli(logits)
        else:
            return self._sample_from_mol(logits)

    def _sample_from_bernoulli(self, logits):
        logits = logits[:, :, 2:30, 2:30]
        probs = torch.sigmoid(logits)
        return torch.Tensor(logits.size()).bernoulli_(probs)  # B, C, H, W

    def _compute_scales(self, logits):
        softplus = nn.Softplus(beta=hparams.model.gradient_smoothing_beta)
        if hparams.model.distribution_base == 'std':
            scales = torch.maximum(softplus(logits), torch.as_tensor(np.exp(hparams.loss.min_mol_logscale)))

        elif hparams.model.distribution_base == 'logstd':
            log_scales = torch.maximum(logits, torch.as_tensor(np.array(hparams.loss.min_mol_logscale)))
            scales = torch.exp(hparams.model.gradient_smoothing_beta * log_scales)

        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        return scales

    def _sample_from_mol(self, logits):
        B, _, H, W = logits.size()  # B, M*(3*C+1), H, W,

        logit_probs = logits[:, :hparams.model.num_output_mixtures, :, :]  # B, M, H, W
        l = logits[:, hparams.model.num_output_mixtures:, :, :]  # B, M*C*3 ,H, W
        l = l.reshape(B, hparams.data.channels, 3 * hparams.model.num_output_mixtures, H, W)  # B, C, 3 * M, H, W

        model_means = l[:, :, :hparams.model.num_output_mixtures, :, :]  # B, C, M, H, W
        scales = self._compute_scales(
            l[:, :, hparams.model.num_output_mixtures: 2 * hparams.model.num_output_mixtures, :, :])  # B, C, M, H, W
        model_coeffs = torch.tanh(
            l[:, :, 2 * hparams.model.num_output_mixtures: 3 * hparams.model.num_output_mixtures, :,
            :])  # B, C, M, H, W

        # Gumbel-max to select the mixture component to use (per pixel)
        gumbel_noise = -torch.log(-torch.log(
            torch.Tensor(logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, H, W
        logit_probs = logit_probs / hparams.synthesis.output_temperature + gumbel_noise
        lambda_ = one_hot(torch.argmax(logit_probs, dim=1), logit_probs.size()[1], dim=1)  # B, M, H, W

        lambda_ = lambda_.unsqueeze(1)  # B, 1, M, H, W

        # select logistic parameters
        means = torch.sum(model_means * lambda_, dim=2)  # B, C, H, W
        scales = torch.sum(scales * lambda_, dim=2)  # B, C, H, W
        coeffs = torch.sum(model_coeffs * lambda_, dim=2)  # B, C,  H, W

        # Samples from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()
        x = means + scales * hparams.synthesis.output_temperature * (
                torch.log(u) - torch.log(1. - u))  # B, C,  H, W

        # Autoregressively predict RGB
        x0 = torch.clamp(x[:, 0:1, :, :], min=self.min_pix_value, max=self.max_pix_value)  # B, 1, H, W
        x1 = torch.clamp(x[:, 1:2, :, :] + coeffs[:, 0:1, :, :] * x0, min=self.min_pix_value,
                         max=self.max_pix_value)  # B, 1, H, W
        x2 = torch.clamp(x[:, 2:3, :, :] + coeffs[:, 1:2, :, :] * x0 + coeffs[:, 2:3, :, :] * x1,
                         min=self.min_pix_value,
                         max=self.max_pix_value)  # B, 1, H, W

        x = torch.cat([x0, x1, x2], dim=1)  # B, C, H, W
        return x

    def forward(self, skip_list, variate_masks):
        y = torch.tile(self.trainable_h, (skip_list[0].size()[0], 1, 1, 1))
        posterior_dist_list = []
        prior_kl_dist_list = []

        layer_idx = 0
        for i, (level_down_upsample, level_down, skip_input) in enumerate(
                zip(self.levels_down_upsample, self.levels_down, skip_list)):
            y, posterior_dist, prior_kl_dist, = level_down_upsample(skip_input, y,
                                                                    variate_mask=variate_masks[layer_idx])
            layer_idx += 1

            resolution_posterior_dist = [posterior_dist]
            resolution_prior_kl_dist = [prior_kl_dist]

            for j, layer in enumerate(level_down):
                y, posterior_dist, prior_kl_dist = layer(skip_input, y, variate_mask=variate_masks[layer_idx])
                layer_idx += 1

                resolution_posterior_dist.append(posterior_dist)
                resolution_prior_kl_dist.append(prior_kl_dist)

            posterior_dist_list += resolution_posterior_dist
            prior_kl_dist_list += resolution_prior_kl_dist

        y = self.output_conv(y)

        return y, posterior_dist_list, prior_kl_dist_list,

    def sample_from_prior(self, batch_size, temperatures):
        with torch.no_grad():
            y = torch.tile(self.trainable_h, (batch_size, 1, 1, 1))

            prior_zs = []
            for i, (level_down_upsample, level_down, temperature) in enumerate(
                    zip(self.levels_down_upsample, self.levels_down, temperatures)):
                y, z = level_down_upsample.sample_from_prior(y, temperature=temperature)

                level_z = [z]
                for _, layer in enumerate(level_down):
                    y, z = layer.sample_from_prior(y, temperature=temperature)
                    level_z.append(z)

                prior_zs += level_z  # n_layers * [batch_size,  n_variates H, W]
            y = self.output_conv(y)

        return y, prior_zs
