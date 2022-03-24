import jax.lax as lax
import jax.numpy as jnp

from hparams import HParams

try:
    from .losses import KLDivergence

except (ImportError, ValueError):
    from model.losses import KLDivergence

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class KLDivergenceStats(KLDivergence):
    def _get_kl_loss(self, p, q):
        """ Compute KL div per dimension (variate)"""
        if hparams.model.distribution_base == 'std':
            loss = self._gauss_std_based_kl(p, q)
        elif hparams.model.distribution_base == 'logstd':
            loss = self._gauss_logstd_based_kl(p, q)
        else:
            raise ValueError(f'distribution base {hparams.model.distribution_base} not known!!')

        mean_axis = (1, 2)

        return loss, mean_axis

    def compute_loss(self, p, q, global_batch_size):
        raise NotImplementedError("Don't use this class to compute loss! Call compute_metrics instead!!")

    def _compute_metrics(self, p, q, global_batch_size):
        loss, mean_axis = self._get_kl_loss(p, q)

        per_example_loss = jnp.sum(loss, axis=mean_axis)  # [batch_size, n_variates]

        n_mean_elems = jnp.prod(jnp.asarray([loss.shape[a] for a in mean_axis]))  # heads * h * w  or h * w
        avg_per_example_loss = per_example_loss / n_mean_elems

        # [n_variates, ]
        avg_loss = jnp.sum(avg_per_example_loss, axis=0) / (global_batch_size * jnp.log(2.))  # divide by ln(2) to convert to KL rate (average space bits/dim)
        return avg_loss

    def compute_metrics(self, posterior_dist_list, prior_kl_dist_list, global_batch_size):
        def get_kl_list(p_dist, q_dist):
            kls = []

            for p, q in zip(p_dist, q_dist):
                kl_loss = self._compute_metrics(
                    p=p,
                    q=q,
                    global_batch_size=global_batch_size
                )
                kls.append(kl_loss)

            return jnp.stack(kls, axis=0)  # [n_layers, n_variates]

        per_variate_avg_divs = get_kl_list(posterior_dist_list, prior_kl_dist_list)

        metrics = {'per_variate_avg_divs': lax.psum(per_variate_avg_divs, axis_name='shards')}
        return metrics
