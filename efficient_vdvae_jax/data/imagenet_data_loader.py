import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
from sklearn.utils import shuffle
from hparams import HParams
import jax
from flax import jax_utils

try:
    from ..utils.normalizer import Normalizer
    from ..utils.utils import compute_latent_dimension
except (ImportError, ValueError):
    from utils.normalizer import Normalizer
    from utils.utils import compute_latent_dimension

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def load_and_shard_tf_batch(xs, global_batch_size):
    local_device_count = jax.local_device_count()

    def _prepare(x):
        return x.reshape((local_device_count, global_batch_size // local_device_count) + x.shape[1:])

    return jax.tree_map(_prepare, xs)


def create_synthesis_imagenet_dataset():
    if hparams.synthesis.synthesis_mode == 'reconstruction':
        test_images = read_imagenet_images(hparams.data.synthesis_data_path)

        test_data = tf.data.Dataset.from_tensor_slices(test_images)

        test_data = test_data.interleave(
            lambda x: data_prep(x, False),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

        test_data = test_data.batch(
            hparams.synthesis.batch_size,
            drop_remainder=True
        )
        test_data = test_data.prefetch(tf.data.AUTOTUNE)

        test_data = tfds.as_numpy(test_data)
        test_data = map(lambda x: load_and_shard_tf_batch(x, hparams.synthesis.batch_size), test_data)
        test_data = jax_utils.prefetch_to_device(test_data, 10)
        return test_data

    elif hparams.synthesis.synthesis_mode == 'div_stats':
        train_data = read_imagenet_images(hparams.data.train_data_path)

        n_train_samples = train_data.shape[0]

        train_data = tf.data.Dataset.from_tensor_slices(train_data)

        # Take a subset of the data
        train_data = train_data.shuffle(n_train_samples)
        train_data = train_data.take(int(hparams.synthesis.div_stats_subset_ratio * n_train_samples))

        # Preprocess subset and prefect to device
        train_data = train_data.interleave(
            lambda x: data_prep(x, False),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False)

        train_data = train_data.batch(
            hparams.synthesis.batch_size,
            drop_remainder=True
        )
        train_data = train_data.prefetch(tf.data.AUTOTUNE)

        train_data = tfds.as_numpy(train_data)
        train_data = map(lambda x: load_and_shard_tf_batch(x, hparams.synthesis.batch_size), train_data)
        train_data = jax_utils.prefetch_to_device(train_data, 10)
        return train_data

    elif hparams.synthesis.synthesis_mode == 'encoding':
        train_data = read_imagenet_images(hparams.data.train_data_path)
        train_filenames = make_toy_filenames(train_data)

        train_data = tf.data.Dataset.from_tensor_slices((train_data, train_filenames))

        train_data = train_data.interleave(
            named_data_prep,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        train_data = train_data.batch(
            hparams.synthesis.batch_size,
            drop_remainder=True
        )

        train_data = train_data.prefetch(tf.data.AUTOTUNE)

        train_data = tfds.as_numpy(train_data)
        train_data = map(lambda x: (load_and_shard_tf_batch(x[0], hparams.synthesis.batch_size), x[1]), train_data)
        return train_data

    else:
        return None


def named_data_prep(img, filename):
    inputs = data_prep(img, flip=False, return_targets=False)
    return tf.data.Dataset.from_tensor_slices(tensors=(inputs, filename[tf.newaxis]))


def data_prep(img, flip, return_targets=True):
    # Random flip
    if flip and hparams.data.random_horizontal_flip:
        img = tf.image.random_flip_left_right(img)

    # Normalize and possibly reduce bits
    normalizer = Normalizer()
    inputs = normalizer(img, reduce_bits=True)
    targets = normalizer(img, reduce_bits=True)

    if return_targets:
        # [1, H, W, C]
        return tf.data.Dataset.from_tensor_slices(tensors=(inputs[tf.newaxis, ...], targets[tf.newaxis, ...]))
    else:
        return inputs[tf.newaxis, ...]


def make_toy_filenames(data):
    return [f'image_{i}' for i in range(data.shape[0])]


def read_imagenet_images(path):
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
    data = np.concatenate([np.load(f)['data'] for f in files], axis=0)  # [samples, C * H * W]
    data = data.reshape([data.shape[0], hparams.data.channels, hparams.data.target_res, hparams.data.target_res])  # [samples, C, H, W]
    data = data.transpose([0, 2, 3, 1])  # [samples, H, W, C]

    assert data.shape[1] == data.shape[2] == hparams.data.target_res
    assert data.shape[3] == hparams.data.channels

    data = shuffle(data)
    return data


def create_imagenet_datasets():
    train_images = read_imagenet_images(hparams.data.train_data_path)
    val_images = read_imagenet_images(hparams.data.val_data_path)

    n_train_samples = train_images.shape[0]
    n_val_samples = val_images.shape[0]

    train_data = tf.data.Dataset.from_tensor_slices(train_images).cache()
    val_data = tf.data.Dataset.from_tensor_slices(val_images).cache()

    # Repeat data across epochs
    train_data = train_data.repeat()
    val_data = val_data.repeat()

    # Shuffle samples with a buffer of the size of the dataset
    train_data = train_data.shuffle(n_train_samples)
    val_data = val_data.shuffle(n_val_samples)

    train_data = train_data.interleave(
        lambda x: data_prep(x, True),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    val_data = val_data.interleave(
        lambda x: data_prep(x, False),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    # cache, Batch, prefetch
    train_data = train_data.batch(
        hparams.train.batch_size,
        drop_remainder=True
    )
    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    val_data = val_data.batch(
        hparams.val.batch_size,
        drop_remainder=True
    )
    val_data = val_data.prefetch(tf.data.AUTOTUNE)

    train_data = tfds.as_numpy(train_data)
    val_data = tfds.as_numpy(val_data)

    train_data = map(lambda x: load_and_shard_tf_batch(x, hparams.train.batch_size), train_data)
    train_data = jax_utils.prefetch_to_device(train_data, 5)

    val_data = map(lambda x: load_and_shard_tf_batch(x, hparams.val.batch_size), val_data)
    val_data = jax_utils.prefetch_to_device(val_data, 1)
    return train_data, val_data
