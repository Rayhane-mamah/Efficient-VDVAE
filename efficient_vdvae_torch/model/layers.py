from hparams import HParams
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

try:
    from .latent_layers import GaussianLatentLayer
    from ..utils.utils import get_same_padding
    from .conv2d import Conv2d
except (ImportError, ValueError):
    from model.latent_layers import GaussianLatentLayer
    from utils.utils import get_same_padding
    from model.conv2d import Conv2d

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class Interpolate(nn.Module):
    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Unpoolayer(nn.Module):
    def __init__(self, in_filters, filters, strides):
        super(Unpoolayer, self).__init__()
        self.filters = filters

        if isinstance(strides, int):
            self.strides = (strides, strides)
        else:
            self.strides = strides

        ops = [Conv2d(in_channels=in_filters, out_channels=filters, kernel_size=(1, 1), stride=(1, 1),
                      padding='same'),
               nn.LeakyReLU(negative_slope=0.1),
               Interpolate(scale=self.strides)]

        self.register_parameter('scale_bias', None)

        self.ops = nn.Sequential(*ops)

    def reset_parameters(self, inputs):
        B, C, H, W = inputs.shape
        self.scale_bias = nn.Parameter(torch.zeros(size=(1, C, H, W), device='cuda'), requires_grad=True)

    def forward(self, x):
        x = self.ops(x)
        if self.scale_bias is None:
            self.reset_parameters(x)
        x = x + self.scale_bias
        return x


class PoolLayer(nn.Module):
    def __init__(self, in_filters, filters, strides):
        super(PoolLayer, self).__init__()
        self.filters = filters

        if isinstance(strides, int):
            strides = (strides, strides)

        ops = [Conv2d(in_channels=in_filters, out_channels=filters,
                      kernel_size=strides, stride=strides, padding='same'),
               nn.LeakyReLU(negative_slope=0.1)]

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class ResidualConvCell(nn.Module):
    def __init__(self, n_layers, in_filters, bottleneck_ratio, kernel_size, init_scaler
                 , residual=True, use_1x1=True, output_ratio=1.0):
        super(ResidualConvCell, self).__init__()

        self.residual = residual
        self.output_ratio = output_ratio
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if self.residual:
            assert self.output_ratio == 1

        output_filters = int(in_filters * output_ratio)
        bottlneck_filters = int(in_filters * bottleneck_ratio)

        convs = [nn.SiLU(inplace=False),
                 nn.Conv2d(in_channels=in_filters,
                           out_channels=bottlneck_filters,
                           kernel_size=(1, 1) if use_1x1 else kernel_size,
                           stride=(1, 1),
                           padding='same')]

        for _ in range(n_layers):
            convs.append(nn.SiLU(inplace=False))
            convs.append(Conv2d(in_channels=bottlneck_filters,
                                out_channels=bottlneck_filters,
                                kernel_size=kernel_size,
                                stride=(1, 1),
                                padding='same'))

        convs += [nn.SiLU(inplace=False),
                  Conv2d(in_channels=bottlneck_filters,
                         out_channels=output_filters,
                         kernel_size=(1, 1) if use_1x1 else kernel_size,
                         stride=(1, 1),
                         padding='same')]

        convs[-1].weight.data *= init_scaler

        self.convs = nn.Sequential(*convs)

    def forward(self, inputs):
        x = inputs
        x = self.convs(x)

        if self.residual:
            outputs = inputs + x
        else:
            outputs = x
        return outputs


class LevelBlockUp(nn.Module):
    def __init__(self, n_blocks, n_layers, in_filters, filters, bottleneck_ratio,
                 kernel_size, strides, skip_filters, use_skip):
        super(LevelBlockUp, self).__init__()
        self.strides = strides
        self.use_skip = use_skip
        self.residual_block = nn.Sequential(*[
            ResidualConvCell(
                n_layers=n_layers,
                in_filters=in_filters,
                bottleneck_ratio=bottleneck_ratio,
                kernel_size=kernel_size,
                init_scaler=np.sqrt(1. / float(sum(hparams.model.down_n_blocks_per_res) + len(
                    hparams.model.down_strides))) if hparams.model.stable_init else 1.,
                use_1x1=hparams.model.use_1x1_conv
            )
            for _ in range(n_blocks)
        ])

        if self.use_skip:
            self.skip_projection = Conv2d(
                in_channels=in_filters, out_channels=skip_filters, kernel_size=(1, 1), stride=(1, 1),
                padding='same'
            )
        if self.strides > 1:
            self.pool = PoolLayer(in_filters, filters, strides)

    def forward(self, x):
        # Pre-skip block
        x = self.residual_block(x)

        # Skip connection from bottom-up used to compute z
        if self.use_skip:
            skip_output = self.skip_projection(x)
        else:
            skip_output = x

        if self.strides > 1:
            x = self.pool(x)
        return x, skip_output


class LevelBlockDown(nn.Module):
    def __init__(self, n_blocks, n_layers, in_filters, filters, bottleneck_ratio, kernel_size,
                 strides, skip_filters, latent_variates, first_block, last_block):
        super(LevelBlockDown, self).__init__()

        self.first_block = first_block
        self.last_block = last_block

        self.strides = strides
        self.filters = filters

        assert not (self.first_block and self.last_block)
        if self.strides > 1:
            self.unpool = Unpoolayer(in_filters, filters, strides)
            in_filters = filters

        self.residual_block = nn.Sequential(*[
            ResidualConvCell(
                n_layers=n_layers,
                in_filters=in_filters,
                bottleneck_ratio=bottleneck_ratio,
                kernel_size=kernel_size,
                init_scaler=np.sqrt(1. / float(sum(hparams.model.down_n_blocks_per_res) + len(
                    hparams.model.down_strides))) if hparams.model.stable_init else 1.,
                use_1x1=hparams.model.use_1x1_conv
            )
            for _ in range(n_blocks)
        ])

        self.posterior_net = nn.Sequential(ResidualConvCell(
            n_layers=n_layers,
            in_filters=in_filters + skip_filters,
            bottleneck_ratio=bottleneck_ratio * 0.5,
            kernel_size=kernel_size,
            init_scaler=1.,
            residual=False,
            use_1x1=hparams.model.use_1x1_conv,
            output_ratio=0.5  # Assuming skip_filters == in_filters
        ))

        self.prior_net = nn.Sequential(ResidualConvCell(
            n_layers=n_layers,
            in_filters=in_filters,
            bottleneck_ratio=bottleneck_ratio,
            kernel_size=kernel_size,
            init_scaler=0. if hparams.model.initialize_prior_weights_as_zero else 1.,
            residual=False,
            use_1x1=hparams.model.use_1x1_conv,
            output_ratio=2.0
        ))

        self.prior_layer = GaussianLatentLayer(in_filters=in_filters,
                                               num_variates=latent_variates
                                               )
        self.posterior_layer = GaussianLatentLayer(in_filters=in_filters,
                                                   num_variates=latent_variates
                                                   )
        self.latent_embeddings = None

        self.z_projection = Conv2d(
            in_channels=latent_variates,
            out_channels=filters,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding='same'
        )
        self.z_projection.weight.data *= np.sqrt(
            1. / float(sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)))

    def sampler(self, latent_fn, y, prior_stats=None, temperature=None):
        z, dist = latent_fn(y, prior_stats=prior_stats, temperature=temperature)
        return z, dist

    def get_analytical_distribution(self, latent_fn, y, prior_stats=None):
        dist = latent_fn(y, prior_stats=prior_stats, return_sample=False)
        return None, dist

    def sample_from_weights(self, latent_fn, y, attention_weights, latent_embeddings):
        z = latent_fn.extract_and_project_memory(attention_weights, latent_embeddings)

        B, _, H, W = y.size()
        z = z.reshape(B, z.size()[1], H, W)
        return z

    def forward(self, x_skip, y, variate_mask=None):

        if self.strides > 1:
            y = self.unpool(y)
        y_prior_kl = self.prior_net(y)
        kl_residual, y_prior_kl = torch.chunk(y_prior_kl, chunks=2, dim=1)  # B, C, H, W

        y_post = torch.cat([y, x_skip], dim=1)
        y_post = self.posterior_net(y_post)

        # Prior under expected value of q(z<i|x)
        if variate_mask is None:
            z_prior_kl, prior_kl_dist = self.get_analytical_distribution(self.prior_layer, y_prior_kl)

        else:
            z_prior_kl, prior_kl_dist = self.sampler(self.prior_layer, y_prior_kl)

        # Samples posterior under expected value of q(z<i|x)
        z_post, posterior_dist = self.sampler(
            self.posterior_layer,
            y_post,
            prior_stats=prior_kl_dist if hparams.model.use_residual_distribution else None)

        if variate_mask is not None:
            variate_mask = torch.Tensor(variate_mask)[None, :, None, None].cuda()
            # Only used in inference mode to prune turned-off variates
            # Use posterior sample from meaningful variates, and prior sample from "turned-off" variates
            # The NLL should be similar to using z_post without masking if the mask is good (not very destructive)
            # variate_mask automatically broadcasts to [batch_size, H, W, n_variates]
            z_post = variate_mask * z_post + (1. - variate_mask) * z_prior_kl

        # Residual with prior
        y = y + kl_residual

        # Project z and merge back into main stream
        z_post = self.z_projection(z_post)
        y = y + z_post

        # Residual block
        y = self.residual_block(y)

        return y, posterior_dist, prior_kl_dist

    def sample_from_prior(self, y, temperature):
        if self.strides > 1:
            y = self.unpool(y)

        y_prior = self.prior_net(y)

        kl_residual, y_prior = torch.chunk(y_prior, chunks=2, dim=1)
        y = y + kl_residual

        z, _ = self.sampler(self.prior_layer, y_prior, temperature=temperature)
        proj_z = self.z_projection(z)

        y = y + proj_z

        # Residual block
        y = self.residual_block(y)

        return y, z
