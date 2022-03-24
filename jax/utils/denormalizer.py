import numpy as np
from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def min_max(x):
    scale = shift = (2 ** 8 - 1) / 2
    x = x * scale + shift  # [0, 255]
    x = np.round(x).astype(np.uint8)
    return x


class Denormalizer:
    def __init__(self):
        # Always denormalize to 8 bits for metrics and plots
        self.denormalize_fn = min_max

    def __call__(self, x):
        return self.denormalize_fn(x)
