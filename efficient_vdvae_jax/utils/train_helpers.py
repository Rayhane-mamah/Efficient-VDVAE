import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
from flax import jax_utils
from jax.flatten_util import ravel_pytree
import time
from functools import partial

from hparams import HParams

try:
    from .utils import tensorboard_log, plot_image, load_checkpoint_if_exists, save_checkpoint, get_effective_n_pixels
    from .denormalizer import Denormalizer
    from ..model.losses import Loss, ReconstructionLayer
    from ..model.ssim import StructureSimilarityIndexMap

except (ImportError, ValueError):
    from utils.utils import tensorboard_log, plot_image, load_checkpoint_if_exists, save_checkpoint, get_effective_n_pixels
    from utils.denormalizer import Denormalizer
    from model.losses import Loss, ReconstructionLayer
    from model.ssim import StructureSimilarityIndexMap

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def compute_global_norm(grads):
    norms, _ = ravel_pytree(jax.tree_map(jnp.linalg.norm, grads))
    return jnp.linalg.norm(norms)


def clip_by_gradnorm(grads, clip_norm):
    global_norm = compute_global_norm(grads)

    if hparams.optimizer.clip_gradient_norm:
        factor = clip_norm / jnp.maximum(global_norm, clip_norm)
        return jax.tree_map((lambda x: x * factor), grads), global_norm

    return grads, global_norm


def safe_update(state, grads, global_norm, clip_value):
    """Safe gradient updates, by skipping the step if grads are too big (gradient skip)"""

    def update(_):
        return state.apply_gradients(grads=grads)

    def do_nothing(_):
        return state

    # Update only if global_norm is not NaN and is smaller than clip_value
    state = jax.lax.cond(global_norm < clip_value, update, do_nothing, operand=None)
    skip_bool = jnp.logical_or(global_norm >= clip_value, jnp.isnan(global_norm))  # Increment if bigger than clip value or =NaN
    return state, jnp.int32(skip_bool)


@partial(jax.jit, static_argnums=1)
def reshape_to_single_device(inputs_tree, global_batch_size):
    # n_gpus, batch // n_gpus, H, W, C -> batch, H, W, C
    return jax.tree_map(lambda x: jnp.reshape(x, newshape=(global_batch_size,) + x.shape[2:]), inputs_tree)


@jax.jit
def sample_outputs_from_logits(key, logits):
    reconstruction_layer = ReconstructionLayer()
    return reconstruction_layer.sample(key=key, logits=logits)


def forward_fn(params, rng, state, inputs, targets, global_batch_size, training):
    loss_ = Loss()

    def loss_fn(*args, **kwargs):
        return loss_.compute_loss(*args, **kwargs, global_batch_size=global_batch_size)

    def metrics_fn(*args, **kwargs):
        return loss_.compute_metrics(*args, **kwargs, global_batch_size=global_batch_size)

    logits, posterior_dist_list, prior_kl_dist_list = state.apply_fn({'params': params},
                                                                     rng,
                                                                     inputs,
                                                                     training=training)

    loss, kl_div = loss_fn(
        targets, logits, posterior_dist_list, prior_kl_dist_list, state.step, None
    )

    metrics = metrics_fn(
        targets, logits, posterior_dist_list, prior_kl_dist_list, kl_div, None
    )
    return loss, (logits, metrics)


@jax.jit
def change_unit_of_metrics(metrics):
    """Change order of metrics from bpd to nats for binarized mnist only"""
    if hparams.data.dataset_source == 'binarized_mnist':
        # Convert from bpd to nats for comparison
        metrics['kl_div'] = metrics['kl_div'] * jnp.log(2.) * get_effective_n_pixels()
        metrics['avg_kl_divs'] = jax.tree_map(lambda x: x * jnp.log(2.) * get_effective_n_pixels(), metrics['avg_kl_divs'])
        metrics['avg_recon_loss'] = metrics['avg_recon_loss'] * jnp.log(2.) * get_effective_n_pixels()
    return metrics


def train_step(rng, state, train_inputs, train_targets, skip_counter):
    # Forward step + gradient computation
    grad_fn = jax.value_and_grad(forward_fn, has_aux=True, argnums=0)
    (_, (logits, metrics)), grads = grad_fn(state.params, rng, state, train_inputs, train_targets, hparams.train.batch_size, training=True)

    # For MNIST only, change metrics from bits/dim to nats
    metrics = change_unit_of_metrics(metrics)

    # Multi-GPU aggregation
    grads = lax.psum(grads, axis_name='shards')

    # Optional Gradient Clipping
    grads, global_norm = clip_by_gradnorm(grads, clip_norm=hparams.optimizer.gradient_clip_norm_value)

    # Gradient skipping for large gradnorms
    if hparams.optimizer.gradient_skip:
        state, skip_delta = safe_update(state, grads, global_norm, hparams.optimizer.gradient_skip_threshold)
    else:
        state = state.apply_gradients(grads=grads)
        skip_delta = jnp.int32(0)

    skip_counter += skip_delta
    return state, logits, metrics, global_norm, skip_counter


def eval_step(rng, state, val_inputs, val_targets, batch_size):
    # Forward step
    _, (logits, metrics) = forward_fn(state.params, rng, state, val_inputs, val_targets, batch_size, training=False)

    # For MNIST only, change metrics from bits/dim to nats
    metrics = change_unit_of_metrics(metrics)
    return logits, metrics


def train(rng, state, train_data, val_data, tb_writer_train, tb_writer_val, checkpoint_dir, lr_schedule):
    initial_step = int(state.step)  # get the init step number before replicating the state
    print(f'\tStarting from step {initial_step}...\n')
    devices = jax.devices('gpu') if hparams.run.num_gpus >= 1 else jax.devices()
    state = jax_utils.replicate(state, devices=devices)

    denormalizer = Denormalizer()
    loss = Loss()

    start_time = time.time()
    successive_steps = 1

    # Create parallel train and eval steps
    skip_counter = jax_utils.replicate(jnp.int32(0), devices=devices)

    p_train_step = jax.pmap(
        fun=jax.jit(train_step),
        axis_name='shards',
    )
    p_eval_step = jax.pmap(
        fun=jax.jit(eval_step, static_argnums=4),  # per_gpu, batch_size is static argument to avoid re-compilation
        static_broadcasted_argnums=4,  # Define batch_size as a static argument for pmap so we don't have to duplicate it for all GPUs
        axis_name='shards',
    )

    # Wrap entire training code in jax mesh to designate device partition
    for global_step, (train_inputs, train_targets) in zip(range(initial_step, hparams.train.total_train_steps), train_data):
        # Train step
        rng, *train_step_rng = random.split(rng, num=jax.local_device_count() + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, devices)
        state, train_logits, train_metrics, global_norm, skip_counter = p_train_step(
            train_step_rng,
            state,
            train_inputs,
            train_targets,
            skip_counter
        )

        # Only print once in few steps, this allows asynchronous dispatch to queue multiple successive updates during compute
        if (global_step % hparams.train.logging_interval_in_steps == 0 or global_step % hparams.train.checkpoint_and_eval_interval_in_steps == 0
                or global_step == 0 or global_step == hparams.train.total_train_steps - 1):
            # Block code until all jitted functions are done running
            train_metrics['avg_kl_divs'].block_until_ready()
            end_time = time.time()
            step_time = (end_time - start_time) / successive_steps

            train_metrics = jax.tree_map(lambda x: x[0], train_metrics)
            global_norm = global_norm[0]
            # logging
            train_avg_kl = jnp.sum(train_metrics['avg_kl_divs'])
            train_nelbo = train_metrics['kl_div'] + train_metrics['avg_recon_loss']
            # n_active_groups IS MERELY A HEURISTIC FOR DEBUGGING, IT DOES NOT LITERALLY MEAN A HIERARCHY GROUP IS USEFUL
            n_active_groups = jnp.sum(jnp.asarray([v >= hparams.metrics.latent_active_threshold for v in train_metrics['avg_kl_divs']], dtype=jnp.float32))
            print(f'global_step: {global_step:08d} | Time/global_step (sec): {step_time:.4f} | '
                  f'Reconstruction Loss: {train_metrics["avg_recon_loss"]:.4f} | '
                  f'KL loss: {train_metrics["kl_div"]:.4f} | NELBO: {train_nelbo:.4f} | '
                  f'average KL loss: {train_avg_kl:.4f} | Beta: {loss.kldiv_schedule(global_step):.4f} | '
                  f'N° active groups: {n_active_groups:.4f} | Global Norm: {global_norm:.4f} | '
                  f'Skips: {skip_counter[0]}  ', end='\r')

        if global_step % hparams.train.checkpoint_and_eval_interval_in_steps == 0 or global_step == 0 or global_step == hparams.train.total_train_steps - 1:
            print()
            rng, train_rng, val_rng = random.split(rng, num=3)
            train_losses, train_targets, train_outputs, train_means, train_log_scales = evaluate(p_eval_step=p_eval_step, rng=train_rng, state=state, val_data=train_data,
                                                                                                 global_step=global_step, data_name='train_data')
            val_losses, val_targets, val_outputs, val_means, val_log_scales = evaluate(p_eval_step=p_eval_step, rng=val_rng, state=state, val_data=val_data,
                                                                                       global_step=global_step, data_name='val_data')

            # Save checkpoint (only if better than best)
            print(f'Saving checkpoint for global_step {global_step}..')
            save_checkpoint(checkpoint_dir, state)

            # Tensorboard logging
            print('Logging to Tensorboard..')
            train_losses['skips_count'] = skip_counter[0] / hparams.train.total_train_steps  # Log % of cumulative skipped steps

            tensorboard_log(tb_writer_train, global_step, train_losses, train_outputs,
                            train_targets,
                            means=train_means,
                            log_scales=train_log_scales,
                            global_norm=global_norm,
                            lr_schedule=lr_schedule,
                            updates={})
            tensorboard_log(tb_writer_val, global_step, val_losses, val_outputs, val_targets,
                            means=val_means, log_scales=val_log_scales)

            # Save artifacts
            plot_image(tb_writer_train, train_outputs[0], train_targets[0], global_step, denormalizer=denormalizer)
            plot_image(tb_writer_val, val_outputs[0], val_targets[0], global_step, denormalizer=denormalizer)
            print('Resuming..')

        # Start timer for next logging_interval steps
        if (global_step % hparams.train.logging_interval_in_steps == 0 or global_step % hparams.train.checkpoint_and_eval_interval_in_steps == 0
                or global_step == 1):
            start_time = time.time()
            successive_steps = 1
        else:
            successive_steps += 1

    print(f'Finished training after {global_step} steps!')


def evaluate(p_eval_step, rng, state, val_data, global_step, data_name):
    print(f"Beginning evaluation on {data_name}...")
    if data_name == 'train_data':
        batch_size = hparams.train.batch_size
    elif data_name == 'val_data':
        batch_size = hparams.val.batch_size
    else:
        raise ValueError(f'Data name {data_name} not known!!')

    denormalizer = Denormalizer()
    loss = Loss()
    ssim_metric = StructureSimilarityIndexMap(hparams.data.channels, denormalizer=denormalizer)

    val_metrics = {
        'avg_recon_loss': 0,
        'avg_kl_divs': 0,
        'kl_div': 0,
    }
    val_ssims = 0

    # Validate for correct number of val steps (n_val_samples / val_batch_size)
    # For training where batch size is smaller, we validate on less samples to not waste time, assuming the subsample is large enough and representative.
    for val_step, (val_inputs, val_targets) in zip(range(hparams.val.n_samples_for_validation // hparams.val.batch_size), val_data):
        rng, val_sample_key, *val_step_rng = random.split(rng, num=jax.local_device_count() + 2)
        val_step_rng = jnp.asarray(val_step_rng)
        val_logits, val_batch_metrics = p_eval_step(
            val_step_rng,
            state,
            val_inputs,
            val_targets,
            batch_size
        )

        val_batch_metrics = jax.tree_map(lambda x: x[0], val_batch_metrics)

        val_targets, val_logits = reshape_to_single_device(inputs_tree=(val_targets, val_logits), global_batch_size=batch_size)
        val_outputs = sample_outputs_from_logits(key=val_sample_key, logits=val_logits)
        val_ssim = ssim_metric(val_targets, val_outputs, global_batch_size=batch_size)

        val_ssims += val_ssim
        for k, v in val_metrics.items():
            val_metrics[k] += val_batch_metrics[k]

    for k, v in val_metrics.items():
        val_metrics[k] = v / (val_step + 1)

    val_ssim = val_ssims / (val_step + 1)

    val_avg_kl = jnp.sum(val_metrics['avg_kl_divs'])
    val_nelbo = val_metrics['kl_div'] + val_metrics['avg_recon_loss']
    val_losses = {'reconstruction_loss': val_metrics["avg_recon_loss"],
                  'kl_div': val_metrics['kl_div'],
                  'average_kl_div': val_avg_kl,
                  'variational_beta': loss.kldiv_schedule(global_step),
                  'ssim': val_ssim,
                  'nelbo': val_nelbo}

    val_losses.update({f'latent_kl_{i}': v for i, v in enumerate(val_metrics['avg_kl_divs'])})

    print(f'Validation Stats for global_step {global_step} on {data_name} | Reconstruction: {val_metrics["avg_recon_loss"]:.6f} | KL: {val_metrics["kl_div"]:.6f} | '
          f'NELBO {val_nelbo:.6f} | SSIM: {val_ssim:.6f}')

    return val_losses, val_targets, val_outputs, val_batch_metrics['means'], val_batch_metrics['log_stds']
