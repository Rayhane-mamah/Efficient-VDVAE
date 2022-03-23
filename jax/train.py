# import os

# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# os.environ['JAX_LOG_COMPILES'] = '1'

import tensorflow as tf

# Prevent tensorflow from using GPUs
tf.config.set_visible_devices([], device_type='GPU')

from hparams import HParams

# Initialize hparams before any other imports
hparams = HParams('.', name="global_local_memcodes")

from jax.config import config

# Set default precision for dots and conv ops to TF32
config.update("jax_default_matmul_precision", 'tensorfloat32')

import jax
from jax.flatten_util import ravel_pytree
import random as pyrandom
import jax.random as random
import jax.numpy as jnp
from numpy.random import seed

try:
    from .model.model import UniversalAutoEncoder
    from .model.optimizers import get_optimizer
    from .model.schedules import get_lr_schedule
    from .utils.utils import create_tb_writer, assert_CUDA_and_hparams_gpus_are_equal, create_checkpoint_dir, load_checkpoint_if_exists, get_l2_mask_from_params
    from .data.generic_data_loader import create_generic_datasets
    from .data.cifar10_data_loader import create_cifar10_datasets
    from .data.mnist_data_loader import create_mnist_datasets
    from .data.imagenet_data_loader import create_imagenet_datasets
    from .utils.train_helpers import train
    from .utils.ema_train_state import EMATrainState

except (ImportError, ValueError):
    from model.model import UniversalAutoEncoder
    from model.optimizers import get_optimizer
    from model.schedules import get_lr_schedule
    from utils.utils import create_tb_writer, assert_CUDA_and_hparams_gpus_are_equal, create_checkpoint_dir, load_checkpoint_if_exists, get_l2_mask_from_params
    from data.generic_data_loader import create_generic_datasets
    from data.cifar10_data_loader import create_cifar10_datasets
    from data.mnist_data_loader import create_mnist_datasets
    from data.imagenet_data_loader import create_imagenet_datasets
    from utils.train_helpers import train
    from utils.ema_train_state import EMATrainState


def main():
    print("Initializing...")
    # n_devices sanity check
    assert_CUDA_and_hparams_gpus_are_equal()

    # Set numpy seed and create jax rng
    seed(hparams.run.seed)  # Numpy seed.
    rng = random.PRNGKey(hparams.run.seed)  # Jax seed.
    pyrandom.seed(hparams.run.seed)  # Python random seed.

    print("Creating model...")
    # Create model
    model = UniversalAutoEncoder()

    print(f"Loading {hparams.data.dataset_source} datasets...")
    # Load datasets
    if hparams.data.dataset_source in ('ffhq', 'celebAHQ', 'celebA'):
        train_data, val_data = create_generic_datasets()
    elif hparams.data.dataset_source == 'cifar-10':
        train_data, val_data = create_cifar10_datasets()
    elif hparams.data.dataset_source == 'imagenet':
        train_data, val_data = create_imagenet_datasets()
    elif hparams.data.dataset_source == 'binarized_mnist':
        train_data, val_data = create_mnist_datasets()
    else:
        raise ValueError(f'dataset source {hparams.data.dataset_source} not known!')

    # Book Keeping
    writer_train, logdir = create_tb_writer(mode='train')
    writer_val, _ = create_tb_writer(mode='val')
    checkpoint_dir = create_checkpoint_dir()

    print("Creating/Loading training state...")
    # Initialize the model and create training state
    print("\tCreating RNG and toy init data..")
    rng, sample_key, init_key = random.split(rng, num=3)
    init_data = jnp.ones((hparams.init.batch_size, hparams.data.target_res, hparams.data.target_res, hparams.data.channels), jnp.float32)

    print("\tInitializing model params...")
    # jit_init = jax.jit(model.init, static_argnums=3)
    init_params = model.init(init_key, sample_key, init_data, True)['params']
    flat_counts, _ = ravel_pytree(jax.tree_map(lambda w: jnp.prod(jnp.asarray(w.shape)), init_params))
    n_params = jnp.sum(flat_counts) / 1000000
    print(f'\t\tModel trainable parameters: {n_params:.3f}m')

    print("\tCreating schedule...")
    learning_rate_schedule = get_lr_schedule(decay_scheme=hparams.optimizer.learning_rate_scheme,
                                             init_lr=hparams.optimizer.learning_rate,
                                             warmup_steps=hparams.optimizer.warmup_steps,
                                             decay_steps=hparams.optimizer.decay_steps,
                                             decay_rate=hparams.optimizer.decay_rate,
                                             decay_start=hparams.optimizer.decay_start,
                                             min_lr=hparams.optimizer.min_learning_rate)
    print("\tCreating optimizer...")
    l2_mask = get_l2_mask_from_params(init_params)
    optimizer = get_optimizer(type=hparams.optimizer.type,
                              learning_rate=learning_rate_schedule,
                              beta_1=hparams.optimizer.beta1,
                              beta_2=hparams.optimizer.beta2,
                              epsilon=hparams.optimizer.epsilon,
                              use_weight_decay=hparams.loss.use_weight_decay,
                              l2_weight=hparams.loss.l2_weight,
                              l2_mask=l2_mask)

    print("\tLoading state if exists...")
    state = EMATrainState.create(
        apply_fn=model.apply,
        params=init_params,
        ema_params=init_params,
        tx=optimizer,
        ema_decay=hparams.train.ema_decay
    )

    # Load checkpoint if exists
    state = load_checkpoint_if_exists(checkpoint_dir, state, replace_params_with_emaparams=hparams.train.resume_from_ema)

    print("Beginning training...")
    # Training loop
    train(rng=rng,
          state=state,
          train_data=train_data,
          val_data=val_data,
          tb_writer_train=writer_train,
          tb_writer_val=writer_val,
          checkpoint_dir=checkpoint_dir,
          lr_schedule=learning_rate_schedule)


if __name__ == '__main__':
    main()
