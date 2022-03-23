from torch import nn
from hparams import HParams

try:
    from .autoencoder import TopDown, BottomUp
except (ImportError, ValueError):
    from model.autoencoder import TopDown, BottomUp

hparams = HParams.get_hparams_by_name("global_local_memcodes")


class UniversalAutoEncoder(nn.Module):
    def __init__(self):
        super(UniversalAutoEncoder, self).__init__()

        self.bottom_up = BottomUp()
        self.top_down = TopDown()

    def forward(self, x, variate_masks=None):
        """
        x: (batch_size, time, H, W, C). In train, this is the shifted version of the target
        In slow synthesis, it would be the concatenated previous outputs
        """
        if variate_masks is None:
            variate_masks = [None] * (sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides))
        else:
            assert len(variate_masks) == sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

        skip_list = self.bottom_up(x)
        outputs, posterior_dist_list, prior_kl_dist_list = self.top_down(skip_list, variate_masks=variate_masks)

        return outputs, posterior_dist_list, prior_kl_dist_list
