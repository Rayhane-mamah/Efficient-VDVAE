import tensorflow as tf
import tensorflow_datasets as tfds
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


def create_synthesis_mnist_dataset():
    if hparams.synthesis.synthesis_mode == 'reconstruction':
        _, _, test_images = download_mnist_datasets()

        test_data = tf.data.Dataset.from_tensor_slices(test_images)

        test_data = test_data.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(tensors=(x[tf.newaxis, ...], x[tf.newaxis, ...])),
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

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
        train_data, _, _ = download_mnist_datasets()

        n_train_samples = train_data.shape[0]

        train_data = tf.data.Dataset.from_tensor_slices(train_data)

        # Take a subset of the data
        train_data = train_data.shuffle(n_train_samples)
        train_data = train_data.take(int(hparams.synthesis.div_stats_subset_ratio * n_train_samples))

        # Preprocess subset and prefect to device
        train_data = train_data.interleave(
            data_prep,
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
        train_data, _, _ = download_mnist_datasets()
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
    inputs = data_prep(img, return_targets=False)
    return tf.data.Dataset.from_tensor_slices(tensors=(inputs, filename[tf.newaxis]))


def data_prep(img, use_tf=True, return_targets=True):
    # Binarize (random bernoulli)
    if use_tf:
        # Used repeatedly during training on a per-sample basis
        noise = tf.random.uniform(shape=img.shape, minval=0., maxval=1., dtype=img.dtype)
        targets = inputs = tf.cast(noise <= img, tf.float32)

        if return_targets:
            # [1, H, W, C]
            return tf.data.Dataset.from_tensor_slices(tensors=(inputs[tf.newaxis, ...], targets[tf.newaxis, ...]))
        else:
            return inputs[tf.newaxis, ...]
    else:
        # Used once for val and test data on a per-data basis
        noise = np.random.uniform(low=0., high=1., size=img.shape).astype(img.dtype)
        data = (noise <= img).astype(np.float32)

        # [n_samples, H, W, C]
        return data


def download_mnist_datasets():
    # Ignore labels
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[..., np.newaxis] / 255.  # (60000, 28, 28, 1)
    x_test = x_test[..., np.newaxis] / 255.  # (10000, 28, 28, 1)

    # make images of size 32x32
    x_train = np.pad(x_train, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))  # (60000, 32, 32, 1)
    x_test = np.pad(x_test, pad_width=((0, 0), (2, 2), (2, 2), (0, 0)))  # (60000, 32, 32, 1)

    x_train = shuffle(x_train)
    x_test = shuffle(x_test, random_state=101)  # Fix this seed to not overlap val and test between train and inference runs

    x_val = x_test[:len(x_test) // 2]  # 5000
    x_test = x_test[len(x_test) // 2:]  # 5000

    x_val = data_prep(x_val, use_tf=False)
    x_test = data_prep(x_test, use_tf=False)
    return x_train, x_val, x_test


def make_toy_filenames(data):
    return [f'image_{i}' for i in range(data.shape[0])]


def create_mnist_datasets():
    train_images, val_images, _ = download_mnist_datasets()

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
        data_prep,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )

    val_data = val_data.interleave(
        lambda x: tf.data.Dataset.from_tensor_slices(tensors=(x[tf.newaxis, ...], x[tf.newaxis, ...])),
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
