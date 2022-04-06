import jax
from jax import random
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import linen as nn

from hparams import HParams

try:
    from .ssim import SSIM
    from .schedules import GammaSchedule, LogisticBetaSchedule, LinearBetaSchedule
    from ..utils.normalizer import Normalizer
    from ..utils.utils import get_effective_n_pixels
    from .latent_layers import beta_softplus

except (ImportError, ValueError):
    from model.ssim import SSIM
    from model.schedules import GammaSchedule, LogisticBetaSchedule, LinearBetaSchedule
    from utils.normalizer import Normalizer
    from utils.utils import get_effective_n_pixels
    from model.latent_layers import beta_softplus

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class Loss:
    def __init__(self):
        self.reconstruction_loss = ReconstructionLayer()
        self.kldiv_loss = KLDivergence()
        self.gamma_schedule = GammaSchedule(max_steps=hparams.loss.gamma_max_steps)

        if hparams.loss.variation_schedule == 'None':
            self.kldiv_schedule = lambda x: 1.

        elif hparams.loss.variation_schedule == 'Logistic':
            self.kldiv_schedule = LogisticBetaSchedule(
                activation_step=hparams.loss.vae_beta_activation_steps,
                growth_rate=hparams.loss.vae_beta_growth_rate
            )

        elif hparams.loss.variation_schedule == 'Linear':
            self.kldiv_schedule = LinearBetaSchedule(
                anneal_start=hparams.loss.vae_beta_anneal_start,
                anneal_steps=hparams.loss.vae_beta_anneal_steps,
                beta_min=hparams.loss.vae_beta_min,
            )

        else:
            raise NotImplementedError(f'KL beta schedule {hparams.loss.variation_schedule} not known!!')

    def compute_loss(self, targets, logits, posterior_dist_list, prior_kl_dist_list, step_n, variate_masks, global_batch_size):
        if variate_masks is None:
            variate_masks = [None] * len(posterior_dist_list)
        else:
            assert len(variate_masks) == len(posterior_dist_list) == len(prior_kl_dist_list)

        recon_loss = self.reconstruction_loss.compute_loss(
            targets=targets,
            logits=logits,
            global_batch_size=global_batch_size
        )

        def get_kl_list(p_dist, q_dist, variate_mask):
            kls = []
            res = []

            for p, q, vm in zip(p_dist, q_dist, variate_mask):
                kl_loss, resolution = self.kldiv_loss.compute_loss(
                    p=p,
                    q=q,
                    variate_mask=vm,
                    global_batch_size=global_batch_size
                )

                kls.append(kl_loss)
                res.append(resolution)

            return jnp.stack(kls, axis=0), jnp.stack(res, axis=0)

        kl_losses, resolutions = get_kl_list(posterior_dist_list, prior_kl_dist_list, variate_masks)

        if hparams.loss.use_gamma_schedule:
            kl_div = self.gamma_schedule(kl_losses,
                                         resolutions,
                                         step_n=step_n)
        else:
            kl_div = jnp.sum(kl_losses)

        loss = recon_loss + self.kldiv_schedule(step_n) * kl_div

        # Return KL divergence (without schedule) for logging purposes
        return loss, kl_div

    def compute_metrics(self, targets, predictions, posterior_dist_list, prior_kl_dist_list, kl_div, variate_masks, global_batch_size):
        if variate_masks is None:
            variate_masks = [None] * len(posterior_dist_list)
        else:
            assert len(variate_masks) == len(posterior_dist_list) == len(prior_kl_dist_list)

        avg_recon_loss, means, log_stds = self.reconstruction_loss.compute_metrics(
            targets=targets,
            logits=predictions,
            global_batch_size=global_batch_size
        )

        def get_kl_list(p_dist, q_dist, variate_mask):
            kls = []

            for p, q, vm in zip(p_dist, q_dist, variate_mask):
                kl_loss = self.kldiv_loss.compute_metrics(
                    p=p,
                    q=q,
                    variate_mask=vm,
                    global_batch_size=global_batch_size
                )
                kls.append(kl_loss)

            return jnp.stack(kls, axis=0)

        avg_kl_divs = get_kl_list(posterior_dist_list, prior_kl_dist_list, variate_masks)

        metrics = {
            'avg_recon_loss': lax.psum(avg_recon_loss, axis_name='shards'),
            'avg_kl_divs': lax.psum(avg_kl_divs, axis_name='shards'),
            'kl_div': lax.psum(kl_div / jnp.log(2.), axis_name='shards'),  # divide by ln(2) to convert to bits/dim (logging only)
            'means': means,
            'log_stds': log_stds
        }

        return metrics


class ReconstructionLayer:
    def __init__(self):
        # Only works for when images are [-1., 1.] normalized for now
        self.num_classes = 2. ** hparams.data.num_bits - 1.
        normalizer = Normalizer(use_tf=False)
        self.min_pix_value = normalizer(0, reduce_bits=True)
        self.max_pix_value = normalizer(255, reduce_bits=True)

    def _compute_scales(self, logits):
        if hparams.model.output_distribution_base == 'std':
            scales = jnp.maximum(beta_softplus(logits, beta=hparams.model.output_gradient_smoothing_beta),
                                 jnp.exp(hparams.loss.min_mol_logscale))

        elif hparams.model.output_distribution_base == 'logstd':
            log_scales = jnp.maximum(logits, hparams.loss.min_mol_logscale)
            scales = jnp.exp(hparams.model.output_gradient_smoothing_beta * log_scales)

        else:
            raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')

        return scales

    def _sample_from_mol(self, key, logits):
        B, H, W, _ = logits.shape

        logit_probs = logits[:, :, :, :hparams.model.num_output_mixtures]  # B, H, W, M
        l = logits[:, :, :, hparams.model.num_output_mixtures:]  # B, H, W, M*C*3
        l = jnp.reshape(l, newshape=(B, H, W, hparams.model.num_output_mixtures * 3, hparams.data.channels))  # B, H, W, M*3, C

        model_means = l[:, :, :, :hparams.model.num_output_mixtures, :]  # B, H, W, M, C
        scales = self._compute_scales(l[:, :, :, hparams.model.num_output_mixtures: 2 * hparams.model.num_output_mixtures, :])  # B, H, W, M, C
        model_coeffs = nn.tanh(
            l[:, :, :, 2 * hparams.model.num_output_mixtures: 3 * hparams.model.num_output_mixtures,
            :])  # B, H, W, M, C

        # Split RNGKey
        log_key, mix_key = random.split(key)

        # Gumbel-max to select the mixture component to use (per pixel)
        gumbel_noise = -jnp.log(-jnp.log(
            random.uniform(key=mix_key, shape=logit_probs.shape, minval=1e-5, maxval=1. - 1e-5,
                           dtype=logit_probs.dtype)))  # B, H, W, M
        logit_probs = logit_probs / hparams.synthesis.output_temperature + gumbel_noise
        lambda_ = jnn.one_hot(jnp.argmax(logit_probs, axis=-1), num_classes=logit_probs.shape[-1],
                              dtype=logit_probs.dtype, axis=-1)  # B, H, W, M
        lambda_ = jnp.expand_dims(lambda_, axis=-1)  # B, H, W, M, 1

        # select logistic parameters
        means = jnp.sum(model_means * lambda_, axis=-2)  # B, H, W, C
        scales = jnp.sum(scales * lambda_, axis=-2)  # B, H, W, C
        coeffs = jnp.sum(model_coeffs * lambda_, axis=-2)  # B, H, W, C

        # Samples from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = random.uniform(key=log_key, shape=means.shape, minval=1e-5, maxval=1. - 1e-5, dtype=means.dtype)
        x = means + scales * hparams.synthesis.output_temperature * (
                jnp.log(u) - jnp.log(1. - u))  # B, H, W, C

        # Autoregressively predict RGB
        x0 = jnp.clip(x[..., 0:1], a_min=self.min_pix_value, a_max=self.max_pix_value)  # B, H, W, 1
        x1 = jnp.clip(x[..., 1:2] + coeffs[..., 0:1] * x0, a_min=self.min_pix_value, a_max=self.max_pix_value)  # B, H, W, 1
        x2 = jnp.clip(x[..., 2:3] + coeffs[..., 1:2] * x0 + coeffs[..., 2:3] * x1, a_min=self.min_pix_value, a_max=self.max_pix_value)  # B, H, W, 1

        x = jnp.concatenate([x0, x1, x2], axis=-1)  # B, H, W, C
        return x

    def _sample_from_bernoulli(self, key, logits):
        """Even though the function is called 'sample' we're not really sampling from the bernoulli distributions
        For MNIST we rather turn the logits to sigmoid and show the smooth images of probabilities instead of hard binaries"""
        # Crop to remove padding
        logits = logits[:, 2:-2, 2:-2, :]
        probs = nn.sigmoid(logits / hparams.synthesis.output_temperature)
        return probs

    def sample(self, key, logits):
        if hparams.data.dataset_source == 'binarized_mnist':
            return self._sample_from_bernoulli(key, logits)
        else:
            return self._sample_from_mol(key, logits)

    def _compute_inv_stdv(self, logits):
        if hparams.model.output_distribution_base == 'std':
            scales = jnp.maximum(beta_softplus(logits, beta=hparams.model.output_gradient_smoothing_beta),
                                 jnp.exp(hparams.loss.min_mol_logscale))
            inv_stdv = 1. / scales  # Not stable for sharp distributions
            log_scales = jnp.log(scales)

        elif hparams.model.output_distribution_base == 'logstd':
            log_scales = jnp.maximum(logits, hparams.loss.min_mol_logscale)
            inv_stdv = jnp.exp(-hparams.model.output_gradient_smoothing_beta * log_scales)

        else:
            raise ValueError(f'distribution base {hparams.model.output_distribution_base} not known!!')

        return inv_stdv, log_scales

    def _get_mol_negative_log_probs(self, targets, logits):
        assert len(targets.shape) == 4
        B, H, W, C = targets.shape
        assert C == 3  # only support RGB for now

        targets = jnp.expand_dims(targets, axis=3)  # B, H, W, 1, C

        logit_probs = logits[:, :, :, :hparams.model.num_output_mixtures]  # B, H, W, M
        l = logits[:, :, :, hparams.model.num_output_mixtures:]  # B, H, W, M*C*3
        l = jnp.reshape(l, newshape=(B, H, W, hparams.model.num_output_mixtures * 3, hparams.data.channels))  # B, H, W, M*3, C

        model_means = l[:, :, :, :hparams.model.num_output_mixtures, :]  # B, H, W, M, C
        inv_stdv, log_scales = self._compute_inv_stdv(l[:, :, :, hparams.model.num_output_mixtures: 2 * hparams.model.num_output_mixtures, :])  # B, H, W, M, C
        model_coeffs = jnp.tanh(l[:, :, :, 2 * hparams.model.num_output_mixtures: 3 * hparams.model.num_output_mixtures, :])  # B, H, W, M, C

        # RGB AR
        mean1 = model_means[..., 0:1]  # B, H, W, M, 1
        mean2 = model_means[..., 1:2] + model_coeffs[..., 0:1] * targets[..., 0:1]  # B, H, W, M, 1
        mean3 = model_means[..., 2:3] + model_coeffs[..., 1:2] * targets[..., 0:1] + model_coeffs[..., 2:3] * targets[..., 1:2]  # B, H, W, M, 1
        means = jnp.concatenate([mean1, mean2, mean3], axis=-1)  # B, H, W, M, C
        centered = targets - means  # B, H, W, M, C

        # inv_stdv = jnp.exp(-log_scales)
        plus_in = inv_stdv * (centered + 1. / self.num_classes)
        cdf_plus = nn.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.num_classes)
        cdf_min = nn.sigmoid(min_in)

        log_cdf_plus = plus_in - nn.softplus(plus_in)  # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = -nn.softplus(min_in)  # log probability for edge case of 255 (before scaling)

        # probability for all other cases
        cdf_delta = cdf_plus - cdf_min  # B, H, W, M, C

        mid_in = inv_stdv * centered
        # log probability in the center of the bin, to be used in extreme cases
        # (not actually used in this code)
        log_pdf_mid = mid_in - log_scales - 2. * nn.softplus(mid_in)

        # Treat edge pixels (0 and 255 before scaling) differently from the middle range.
        broadcast_targets = jnp.broadcast_to(targets, shape=[targets.shape[0], targets.shape[1], targets.shape[2], hparams.model.num_output_mixtures, targets.shape[-1]])
        log_probs = jnp.where(broadcast_targets == self.min_pix_value, log_cdf_plus,
                              jnp.where(broadcast_targets == self.max_pix_value, log_one_minus_cdf_min,
                                        jnp.where(cdf_delta > 1e-5,
                                                  jnp.log(jnp.maximum(cdf_delta, 1e-12)),  # 1e-12 is ugly trick to get around backprop through jnp.where NaNs
                                                  log_pdf_mid - jnp.log(self.num_classes / 2))))  # B, H, W, M, C

        log_probs = jnp.sum(log_probs, axis=-1) + nn.log_softmax(logit_probs, axis=-1)  # B, H, W, M
        negative_log_probs = -logsumexp(log_probs, axis=-1)  # B, H, W

        return negative_log_probs, model_means, log_scales

    def _get_bernoulli_log_probs(self, targets, logits):
        assert len(targets.shape) == 4
        B, H, W, C = targets.shape
        assert C == 1  # Only support single channel for now

        # Stable Negative Log Probs (sigmoid cross entropy): max(x, 0) - x * z + log(1 + exp(-abs(x))) where x = logits and z = targets.
        # Derivation in: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        negative_log_probs = jnp.squeeze(jnp.maximum(logits, 0) - logits * targets + jnp.log(1 + jnp.exp(-jnp.abs(logits))), axis=-1)  # B, H, W

        # Remove MNIST padding
        negative_log_probs = negative_log_probs[:, 2:30, 2:30]
        return negative_log_probs, None, None

    def _get_negative_log_probs(self, targets, logits):
        return self._get_bernoulli_log_probs(targets, logits) if hparams.data.dataset_source == 'binarized_mnist' else self._get_mol_negative_log_probs(targets, logits)

    def compute_loss(self, targets, logits, global_batch_size):
        negative_log_probs, _, _ = self._get_negative_log_probs(targets, logits)

        mean_axis = (1, 2)
        per_example_loss = jnp.sum(negative_log_probs, axis=mean_axis)  # B

        assert len(per_example_loss.shape) == 1

        loss = jnp.sum(per_example_loss) / (global_batch_size * get_effective_n_pixels())

        return loss

    def compute_metrics(self, targets, logits, global_batch_size):
        negative_log_probs, means, log_stds = self._get_negative_log_probs(targets, logits)

        mean_axis = (1, 2)
        avg_per_example_loss = jnp.sum(negative_log_probs, axis=mean_axis) / (jnp.prod(jnp.asarray([negative_log_probs.shape[i] for i in mean_axis])) * hparams.data.channels)  # B

        assert len(avg_per_example_loss.shape) == 1

        avg_loss = jnp.sum(avg_per_example_loss) / (global_batch_size * jnp.log(2.))  # divide by ln(2) to convert to bits/dim (for visualization purposes only)
        return avg_loss, means, log_stds


class KLDivergence:
    def _gauss_logstd_based_kl(self, p, q):
        p_mean, p_logstd = p
        q_mean, q_logstd = q

        p_std = jnp.exp(hparams.model.gradient_smoothing_beta * p_logstd)
        inv_q_std = jnp.exp(-hparams.model.gradient_smoothing_beta * q_logstd)  # 1 / q_std

        term1 = jnp.square((p_mean - q_mean) * inv_q_std)
        term2 = jnp.square(p_std * inv_q_std)

        # KL = 0.5 [(p_mu - q_mu) ** 2 / q_sigma ** 2 + p_sigma ** 2 / q_sigma ** 2 - 1 - 2 log(p_sigma / q_sigma)]
        loss = 0.5 * (term1 + term2) - 0.5 - p_logstd + q_logstd  # batch, h, w, n_gauss
        return loss

    def _gauss_std_based_kl(self, p, q):
        p_mean, p_std = p
        q_mean, q_std = q

        inv_q_std = 1. / q_std
        p_std_on_q_std = p_std * inv_q_std

        term1 = jnp.square((p_mean - q_mean) * inv_q_std)
        term2 = jnp.square(p_std_on_q_std)

        # KL = 0.5 [(p_mu - q_mu) ** 2 / q_sigma ** 2 + p_sigma ** 2 / q_sigma ** 2 - 1 - 2 log(p_sigma / q_sigma)]
        loss = 0.5 * (term1 + term2) - 0.5 - jnp.log(p_std_on_q_std)  # batch, h, w, n_gauss
        return loss

    def _get_kl_loss(self, p, q):
        if hparams.model.distribution_base == 'std':
            loss = self._gauss_std_based_kl(p, q)
        elif hparams.model.distribution_base == 'logstd':
            loss = self._gauss_logstd_based_kl(p, q)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        mean_axis = (1, 2, 3)

        return loss, mean_axis

    def compute_loss(self, p, q, variate_mask, global_batch_size):
        loss, mean_axis = self._get_kl_loss(p, q)

        if variate_mask is not None:
            loss = loss * variate_mask

        # mean_axis = tuple(range(1, len(loss.shape)))
        per_example_loss = jnp.sum(loss, axis=mean_axis)

        n_mean_elems = jnp.prod(jnp.asarray([loss.shape[a] for a in mean_axis]))  # heads * h * w  or h * w * n_variates

        assert len(per_example_loss.shape) == 1

        loss = jnp.sum(per_example_loss) / (global_batch_size * get_effective_n_pixels())

        return loss, n_mean_elems

    def compute_metrics(self, p, q, variate_mask, global_batch_size):
        loss, mean_axis = self._get_kl_loss(p, q)

        if variate_mask is not None:
            loss = loss * variate_mask

        # mean_axis = tuple(range(1, len(loss.shape)))
        per_example_loss = jnp.sum(loss, axis=mean_axis)

        n_mean_elems = jnp.prod(jnp.asarray([loss.shape[a] for a in mean_axis]))  # heads * h * w  or h * w * n_variates
        avg_per_example_loss = per_example_loss / n_mean_elems

        avg_loss = jnp.sum(avg_per_example_loss) / (global_batch_size * jnp.log(2.))  # divide by ln(2) to convert to KL rate (average space bits/dim)
        return avg_loss
