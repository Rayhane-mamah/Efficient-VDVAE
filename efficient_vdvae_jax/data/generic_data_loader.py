import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
from sklearn.utils import shuffle
from hparams import HParams
from PIL import Image
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


def read_image_and_resize(image_file):
    return np.asarray(
        Image.open(image_file).convert("RGB").resize((hparams.data.target_res, hparams.data.target_res),
                                                     resample=Image.BILINEAR)).astype(np.uint8)


def create_synthesis_generic_dataset():
    if hparams.synthesis.synthesis_mode == 'reconstruction':
        test_data = FileReader(path=hparams.data.synthesis_data_path, shuffle_=False)

        test_data = tf.data.Dataset.from_generator(test_data.generator,
                                                   output_types=tf.string,
                                                   output_shapes=tf.TensorShape([]))

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
        train_data = FileReader(path=hparams.data.train_data_path, shuffle_=False)

        n_train_samples = len(train_data.files)

        train_data = tf.data.Dataset.from_generator(train_data.generator,
                                                    output_types=tf.string,
                                                    output_shapes=tf.TensorShape([]))

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
        train_data = NamedFileReader(path=hparams.data.train_data_path)

        train_data = tf.data.Dataset.from_generator(train_data.generator,
                                                    output_types=(tf.string, tf.string),
                                                    output_shapes=(tf.TensorShape([]), tf.TensorShape([])))

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


def named_data_prep(img_file, filename):
    inputs = data_prep(img_file, flip=False, return_targets=False)
    return tf.data.Dataset.from_tensor_slices(tensors=(inputs, filename[tf.newaxis]))


def data_prep(img_file, flip, return_targets=True):
    # Read image
    img = tf.numpy_function(read_image_and_resize, inp=[img_file], Tout=[tf.uint8], name='read_image_resize')
    img = tf.ensure_shape(img, shape=[1, hparams.data.target_res, hparams.data.target_res, hparams.data.channels])  # 1, H, W, C

    # Random flip
    if flip and hparams.data.random_horizontal_flip:
        img = tf.image.random_flip_left_right(img)

    # Normalize and possibly reduce bits
    normalizer = Normalizer()
    inputs = normalizer(img, reduce_bits=True)

    if return_targets:
        targets = normalizer(img, reduce_bits=True)
        return tf.data.Dataset.from_tensor_slices(tensors=(inputs, targets))
    else:
        return inputs


def create_generic_datasets():
    train_data = FileReader(path=hparams.data.train_data_path, shuffle_=True)
    val_data = FileReader(path=hparams.data.val_data_path, shuffle_=True)

    n_train_samples = len(train_data.files)
    n_val_samples = len(val_data.files)

    train_data = tf.data.Dataset.from_generator(train_data.generator,
                                                output_types=tf.string,
                                                output_shapes=tf.TensorShape([])).cache()
    val_data = tf.data.Dataset.from_generator(val_data.generator,
                                              output_types=tf.string,
                                              output_shapes=tf.TensorShape([])).cache()

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


class FileReader:
    def __init__(self, path, shuffle_):
        self.shuffle_ = shuffle_
        self.files = [os.path.join(path, f) for f in os.listdir(path)]
        print(path, len(self.files))

    def _get_item(self, idx):
        return self.files[idx]

    def generator(self):
        if self.shuffle_:
            self.files = shuffle(self.files)

        for idx in range(len(self.files)):
            yield self._get_item(idx)


class NamedFileReader:
    def __init__(self, path):
        self.filenames = sorted(os.listdir(path))
        self.files = [os.path.join(path, f) for f in self.filenames]

        print(path, len(self.files))

    def _get_item(self, idx):
        return self.files[idx]

    def _get_filename(self, idx):
        return self.filenames[idx]

    def generator(self):
        for idx in range(len(self.files)):
            yield self._get_item(idx), self._get_filename(idx)
