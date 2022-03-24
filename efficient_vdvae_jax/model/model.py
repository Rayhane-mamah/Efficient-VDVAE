import jax.random as random
from flax import linen as nn
from hparams import HParams

try:
    from .schedules import LogisticBetaSchedule, NoamSchedule, \
        NarrowExponentialDecay, ConstantLearningRate, GammaSchedule, LinearBetaSchedule, NarrowCosineDecay
    from .autoencoder import TopDown, BottomUp
    from .losses import ReconstructionLayer

except (ImportError, ValueError):
    from model.schedules import LogisticBetaSchedule, NoamSchedule, \
        NarrowExponentialDecay, ConstantLearningRate, GammaSchedule, LinearBetaSchedule, NarrowCosineDecay
    from model.autoencoder import TopDown, BottomUp
    from model.losses import ReconstructionLayer

hparams = HParams.get_hparams_by_name("efficient_vdvae")


class UniversalAutoEncoder(nn.Module):
    def setup(self):
        assert hparams.model.up_strides == hparams.model.down_strides[::-1]

        self.bottom_up = BottomUp(name='bottom_up')
        self.top_down = TopDown(name='top_down')

    def __call__(self, key, x, training, variate_masks=None):
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

        bu_key, td_key = random.split(key)

        # Bottom-up model
        skip_list = self.bottom_up(bu_key, x, training=training)

        # Top-down model
        logits, posterior_dist_list, prior_kl_dist_list = self.top_down(
            td_key, skip_list, variate_masks=variate_masks, training=training
        )
        return logits, posterior_dist_list, prior_kl_dist_list

    def sample_from_prior(self, key, batch_size, training, temperatures):
        reconstruction_layer = ReconstructionLayer()
        key, sample_key = random.split(key, num=2)
        logits, prior_zs = self.top_down.sample_from_prior(key, batch_size, training, temperatures=temperatures)
        return reconstruction_layer.sample(key=sample_key, logits=logits), prior_zs
