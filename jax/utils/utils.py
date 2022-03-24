"""Utility functions throughout the project for: GPU mangement, checkpointing, TB, plots, etc."""

import os
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax import traverse_util
from flax.training import checkpoints
from trax.jaxboard import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def assert_CUDA_and_hparams_gpus_are_equal():
    try:
        assert hparams.run.num_gpus == jax.device_count('gpu')
    except RuntimeError:
        assert hparams.run.num_gpus == 0


def create_checkpoint_dir(working_dir='.'):
    return os.path.join(working_dir, f'checkpoints-{hparams.run.name}')


def get_logdir(working_dir='.'):
    return os.path.join(working_dir, f'logs-{hparams.run.name}')


def create_tb_writer(mode):
    logdir = get_logdir()
    tbdir = os.path.join(logdir, mode)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tbdir)

    return writer, logdir


def get_effective_n_pixels():
    if hparams.data.dataset_source == 'binarized_mnist':
        return 28 * 28 * hparams.data.channels
    else:
        return hparams.data.target_res * hparams.data.target_res * hparams.data.channels


def compute_latent_dimension():
    assert np.prod(hparams.model.down_strides) == np.prod(hparams.model.up_strides)

    return hparams.data.target_res // np.prod(hparams.model.down_strides)


def plot_image(writer, outputs, targets, step, denormalizer):
    step = step
    if hparams.data.dataset_source == 'binarized_mnist':
        assert targets.shape[-1] == outputs.shape[-1] == 1
        targets = (targets[2:-2, 2:-2, :] * 255).astype(jnp.uint8)  # Target is still not cropped
        outputs = (outputs * 255).astype(jnp.uint8)
    else:
        assert targets.shape[-1] == outputs.shape[-1] == 3
        targets = denormalizer(targets)
        outputs = denormalizer(outputs)

    writer.image(tag=f"{step}/Original_{step}", image=targets, step=step)
    writer.image(tag=f"{step}/Generated_{step}", image=outputs, step=step)
    writer.flush()


def tensorboard_log(writer, global_step, losses, outputs, targets, means=None, log_scales=None, lr_schedule=None, updates=None,
                    global_norm=None):
    mode = 'train' if updates is not None else 'val'
    global_step = global_step

    for key, value in losses.items():
        writer.scalar(tag=f"Losses/{key}", value=value, step=global_step)

    writer.histogram(tag="Distributions/target", values=targets, bins=20, step=global_step)
    writer.histogram(tag="Distributions/output", values=jnp.clip(outputs, a_min=-1., a_max=1.), bins=20, step=global_step)

    if means is not None:
        assert log_scales is not None

        writer.histogram(tag='OutputLayer/means', values=means, bins=30, step=global_step)
        writer.histogram(tag='OutputLayer/log_scales', values=log_scales, bins=30, step=global_step)

    if mode == 'train':
        # Get the learning rate from the optimizer
        writer.scalar(tag="Schedules/learning_rate", value=lr_schedule(global_step), step=global_step)

        assert global_norm is not None
        writer.scalar(tag="Mean_Max_Updates/Global_norm", value=global_norm, step=global_step)

        if updates != {}:
            for layer, update in updates.items():
                writer.scalar(tag="Updates/{}".format(layer), value=update, step=global_step)

            max_updates = jnp.max(jnp.array(list(updates.values()), dtype=jnp.float32))
            writer.scalar(tag="Mean_Max_Updates/Max_updates", value=max_updates, step=global_step)

    writer.flush()


def save_checkpoint(checkpoint_dir, state):
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(checkpoint_dir, state, step=step, keep=hparams.run.max_allowed_checkpoints)


def load_checkpoint_if_exists(checkpoint_dir, state, replace_params_with_emaparams):
    if checkpoints.latest_checkpoint(checkpoint_dir) is not None:
        print(f"\tLoading checkpoint state: {checkpoints.latest_checkpoint(checkpoint_dir)}")
        state = checkpoints.restore_checkpoint(checkpoint_dir, state)

        if replace_params_with_emaparams:
            # Affect ema params to trainable parameters
            state = state.replace(params=state.ema_params)

        return state

    print(f"\tCouldn't find state checkpoints to load at: {checkpoint_dir}")
    return state


def get_l2_mask_from_params(params):
    flat_params = {'/'.join(k): v for k, v in traverse_util.flatten_dict(unfreeze(params)).items()}
    mask = {k: not ('bias' in k or 'input_conv' in k or 'output_conv' in k or 'ln' in k or 'trainable_h' in k
                    or 'embedding_table' in k or 'projection' in k or 'trainable_prior' in k) for k in flat_params.keys()}

    mask = traverse_util.unflatten_dict({tuple(k.split('/')): v for k, v in mask.items()})
    return freeze(mask)


def get_variate_masks(stats):
    thresh = np.quantile(stats, 1. - hparams.synthesis.variate_masks_quantile)
    return stats > thresh


def _denormalize(image, denormalizer):
    if hparams.data.dataset_source == 'binarized_mnist':
        assert image.shape[-1] == 1
        image = (image * 255).astype(jnp.uint8)
        image = jnp.tile(image, [1, 1, 3])
    else:
        assert image.shape[-1] == 3
        image = denormalizer(image)
    return image


def write_image_to_disk(filepath, image, denormalizer):
    assert len(image.shape) == 3
    image = _denormalize(image, denormalizer)

    im = Image.fromarray(np.array(image).astype(np.uint8))
    im.save(filepath, format='png')


def write_grid_to_disk(filepath, metric_grid, image_grid, denormalizer):
    assert len(metric_grid.shape) == 1
    assert len(image_grid.shape) == 4
    image_grid = _denormalize(image_grid, denormalizer)

    fig = plt.figure(figsize=(12, 8))
    n_cols = 3
    n_rows = int(np.ceil(len(image_grid) / n_cols))
    for i, (metric, image) in enumerate(zip(metric_grid, image_grid)):
        fig.add_subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'gen') if i == 0 else plt.title(f'{metric:.4f}↑')
    plt.tight_layout()
    plt.savefig(filepath, format='png', dpi=300)
    plt.close()


def transpose_dicts(dct):
    d = defaultdict(dict)
    for key1, inner in dct.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return d
