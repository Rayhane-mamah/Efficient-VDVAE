import tensorflow as tf
import numpy as np

from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def reduce_bits_fn(x, use_tf):
    if use_tf:
        x = tf.math.floor(x / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (8 - hparams.data.num_bits)  # [0, 255]
    else:
        x = np.floor(x / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (8 - hparams.data.num_bits)
    return x


def min_max(x, reduce_bits, use_tf):
    if reduce_bits:
        x = reduce_bits_fn(x, use_tf)

    scale = shift = (2 ** 8 - 1) / 2
    return (x - shift) / scale  # [-1, 1]


class Normalizer:
    def __init__(self, use_tf=True):
        self.normalize_fn = min_max
        self.use_tf = use_tf

    def __call__(self, x, reduce_bits):
        return self.normalize_fn(x, reduce_bits, use_tf=self.use_tf)
