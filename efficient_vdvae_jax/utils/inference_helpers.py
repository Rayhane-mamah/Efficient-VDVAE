import jax
import jax.random as random
import jax.numpy as jnp
from functools import partial
import numpy as np
from flax import jax_utils
import os
from time import time
import pickle
from hparams import HParams

try:
    from . import temperature_functions
    from .utils import get_variate_masks, write_image_to_disk, write_grid_to_disk, transpose_dicts
    from ..model.model import UniversalAutoEncoder
    from ..model.losses import Loss
    from ..model.div_stats_utils import KLDivergenceStats
    from ..model.ssim import StructureSimilarityIndexMap
    from .train_helpers import sample_outputs_from_logits, reshape_to_single_device, change_unit_of_metrics
    from .denormalizer import Denormalizer

except (ImportError, ValueError):
    from utils import temperature_functions
    from utils.utils import get_variate_masks, write_image_to_disk, write_grid_to_disk, transpose_dicts
    from model.model import UniversalAutoEncoder
    from model.losses import Loss
    from model.div_stats_utils import KLDivergenceStats
    from model.ssim import StructureSimilarityIndexMap
    from utils.train_helpers import sample_outputs_from_logits, reshape_to_single_device, change_unit_of_metrics
    from utils.denormalizer import Denormalizer

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def forward_fn(params, rng, model, inputs, targets, global_batch_size, training, variate_masks):
    loss_ = Loss()

    # @jax.vmap
    def loss_fn(*args, **kwargs): return loss_.compute_loss(*args, **kwargs, global_batch_size=global_batch_size)

    # @jax.vmap
    def metrics_fn(*args, **kwargs):
        return loss_.compute_metrics(*args, **kwargs, global_batch_size=global_batch_size)

    logits, posterior_dist_list, prior_kl_dist_list = model.apply({'params': params},
                                                                  rng,
                                                                  inputs,
                                                                  training,
                                                                  variate_masks)
    loss, kl_div = loss_fn(
        targets, logits, posterior_dist_list, prior_kl_dist_list,
        max(hparams.loss.vae_beta_anneal_steps, hparams.loss.gamma_max_steps) * 10., variate_masks
    )

    metrics = metrics_fn(
        targets, logits, posterior_dist_list, prior_kl_dist_list, kl_div,
        variate_masks
    )
    return logits, metrics


def reshape_distribution(dist_list, variate_masks):
    # variate_masks shape: [n_layers, n_variates]
    # dist_list shape: n_layers, 2 x [batch_size, H, W, n_variates]
    # H, W and variates may be different between layers
    def reshape(dist, variate_mask):
        # Reshape
        dist = jnp.array(dist)  # [2, batch_size, H, W, n_variates]

        # Only take effective variates
        dist = dist[:, :, :, :, variate_mask]  # [2, batch_size, H, W, n_variates_subset]

        dist = jnp.transpose(dist, axes=[1, 2, 3, 4, 0])  # [batch_size, H, W, n_variates_subset, 2]
        return dist  # [batch_size, H, W, n_variates_subset, 2]

    # Collect all layers into: n_layers_subset x [batch_size, H, W, n_variates_subset, 2]
    # If the mask states all variates of a layer are not effective, we don't collect any latents from that layer.
    dist_list = {i: reshape(dist, variate_mask) for i, (dist, variate_mask) in enumerate(zip(dist_list, variate_masks)) if variate_mask.any()}
    return dist_list


def filter_z(zs, variate_masks):
    # zs shape: n_layers x [batch_size, H, W, n_variates]
    # variate_masks shape: [n_layers, n_variates]
    # H, W and variates may be different between layers
    def _filter(z, variate_mask):
        # Only take effective variates
        z = z[:, :, :, variate_mask]  # [batch_size, H, W, n_variates_subset]
        return z

    # Collect all layers into: n_layers_subset x [batch_size, H, W, n_variates_subset]
    # If the mask states all variates of a layer are not effective, we don't collect any latents from that layer.
    zs = {i: _filter(z, variate_mask) for i, (z, variate_mask) in enumerate(zip(zs, variate_masks)) if variate_mask.any()}

    return zs


def compute_stats_step(params, rng, inputs):
    """Per variate average KL divergence computation step.
    Runs the forward pass of the inference model and compute the average KL divergence per variate.
    These stats will be used later to prune the latent space.
    """
    model = UniversalAutoEncoder()
    stats_ = KLDivergenceStats()

    def metrics_fn(*args, **kwargs):
        return stats_.compute_metrics(*args, **kwargs, global_batch_size=hparams.synthesis.batch_size)

    _, posterior_dist_list, prior_kl_dist_list = model.apply({'params': params},
                                                             rng,
                                                             inputs,
                                                             False)

    metrics = metrics_fn(posterior_dist_list, prior_kl_dist_list)
    return metrics


def encode_step(params, rng, inputs, variate_masks):
    """Encoding step.
    Runs samples throught the inference model to generate q_phi(z|x).
    Since the latent distribution is large in its dense form, it is automatically masked to prune the turned-off variates.
    A pruning mask must be pre-computed before using this function.
    """
    model = UniversalAutoEncoder()
    # Slow but safe encode step that reuses the training forward function
    logits, posterior_dist_list, prior_kl_dist_list = model.apply({'params': params}, rng, inputs, False, variate_masks=variate_masks)

    posterior_dist_list = reshape_distribution(posterior_dist_list, variate_masks=variate_masks)  # n_layers_subset x [batch_size, H, W, n_variates_subset, 2]
    return posterior_dist_list


def generation_step(params, rng, temperatures):
    """Generation step.
    Generates random samples from the prior distribution.
    Characteristics of the features in the images are purely controlled by RNG.
    """
    model = UniversalAutoEncoder()
    outputs, prior_zs = model.apply({'params': params}, rng, hparams.synthesis.batch_size, False, temperatures,
                                    method=model.sample_from_prior)

    return outputs


def reconstruction_step(params, rng, inputs, targets, variate_masks):
    """Reconstruction step.
    Reconstructs input images and measures metrics (reconstruction, KL, etc).
    Mainly used to compute the Negative ELBO on the test set.
    If variate_masks is not None, this function will prune the latent space hierarchically
     and compute the Negative ELBO of the pruned latent space.
    """
    model = UniversalAutoEncoder()
    rng, sample_rng = random.split(rng, num=2)
    logits, metrics = forward_fn(params, rng, model, inputs, targets, hparams.synthesis.batch_size, False, variate_masks=variate_masks)

    # For MNIST only, change metrics from bits/dim to nats
    metrics = change_unit_of_metrics(metrics)

    # Sample pixels from the output logits
    outputs = sample_outputs_from_logits(sample_rng, logits)

    return outputs, metrics


def _reconstruction_mode(rng, test_dataset, params, devices, ssim_metric, denormalizer, artifacts_folder, latents_folder):
    artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/reconstructed')
    os.makedirs(artifacts_folder, exist_ok=True)

    if hparams.synthesis.mask_reconstruction:
        if not os.path.isfile(os.path.join(latents_folder, 'div_stats.npy')):
            raise FileNotFoundError('Run div_stats mode before trying to do masked reconstruction!')

        div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
        variate_masks = get_variate_masks(div_stats).astype(np.float32)
    else:
        variate_masks = None

    p_reconstruction_step = jax.pmap(
        fun=jax.jit(partial(reconstruction_step, variate_masks=variate_masks)),
        axis_name='shards',
    )

    ssims = 0.
    kls = 0.
    recons = 0.
    nelbos = 0.
    sample_i = 0
    for step, (inputs, targets) in enumerate(test_dataset):
        # Reconstruction step
        start_time = time()
        rng, *recon_step_rng = random.split(rng, num=jax.local_device_count() + 1)
        recon_step_rng = jax.device_put_sharded(recon_step_rng, devices)
        outputs, metrics = p_reconstruction_step(params, recon_step_rng, inputs, targets)
        metrics['kl_div'].block_until_ready()
        step_time = time() - start_time

        # Compute metrics we care about
        targets, outputs = reshape_to_single_device(inputs_tree=(targets, outputs), global_batch_size=hparams.synthesis.batch_size)
        metrics = jax.tree_map(lambda x: x[0], metrics)
        kl = metrics['kl_div']
        recon = metrics['avg_recon_loss']
        nelbo = kl + recon
        ssim_per_batch = ssim_metric(targets, outputs, global_batch_size=hparams.synthesis.batch_size)

        # Sum for aggregation
        ssims += ssim_per_batch
        nelbos += nelbo
        kls += kl
        recons += recon

        # Save images to disk
        for batch_i, (target, output) in enumerate(zip(targets, outputs)):
            if hparams.synthesis.save_target_in_reconstruction:
                write_image_to_disk(
                    os.path.join(artifacts_folder, f'target-{sample_i:04d}.png'),
                    target,
                    denormalizer=denormalizer
                )
            write_image_to_disk(
                os.path.join(artifacts_folder, f'image-{sample_i:04d}.png'),
                output,
                denormalizer=denormalizer
            )

            sample_i += 1

        print(f'Step: {step:04d} | Time/Step (sec): {step_time:.4f} | NELBO: {nelbo:.4f} | SSIM: {ssim_per_batch:.4f}    ', end='\r')

    nelbo = nelbos / (step + 1)
    kl = kls / (step + 1)
    recon = recons / (step + 1)
    ssim = ssims / (step + 1)

    print()
    print()
    print('===================== Total Average Stats ======================')
    print(f'NELBO: {nelbo:.6f} | Recon: {recon:.6f} | KL: {kl:.6f} | SSIM: {ssim:.6f}')


def _generation_mode(rng, params, devices, denormalizer, artifacts_folder, latents_folder):
    artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/generated')
    os.makedirs(artifacts_folder, exist_ok=True)

    p_generation_step = jax.pmap(
        fun=jax.jit(generation_step),
        axis_name='shards',
    )

    # Generation supports runs with several temperature configs to avoid rebuilding each time
    for temp_i, temperature_setting in enumerate(hparams.synthesis.temperature_settings):
        print(f'Generating for temperature setting {temp_i:01d}')
        # Make per layer temperatures of the setting
        if isinstance(temperature_setting, list):
            # Use defined list of temperatures
            assert len(temperature_setting) == len(hparams.model.down_strides)
            temperatures = temperature_setting

        elif isinstance(temperature_setting, float):
            # Use the same float valued temperature for all layers
            temperatures = [temperature_setting] * len(hparams.model.down_strides)

        elif isinstance(temperature_setting, tuple):
            # Fallback to function defined temperature. Function params are defined with 3 arguments in a tuple
            assert len(temperature_setting) == 3
            temp_fn = getattr(temperature_functions, temperature_setting[0])(temperature_setting[1], temperature_setting[2], n_layers=len(hparams.model.down_strides))
            temperatures = [temp_fn(layer_i) for layer_i in range(len(hparams.model.down_strides))]

        else:
            raise ValueError(f'Temperature Setting {temperature_setting} not interpretable!!')
        # Replicate temperatures across devices
        temperatures = jax_utils.replicate(jnp.array(temperatures, dtype=jnp.float32), devices=devices)

        sample_i = 0
        for step in range(hparams.synthesis.n_generation_batches):
            # Generation step
            start_time = time()
            rng, *gen_step_rng = random.split(rng, num=jax.local_device_count() + 1)
            gen_step_rng = jax.device_put_sharded(gen_step_rng, devices)
            outputs = p_generation_step(params, gen_step_rng, temperatures)
            outputs.block_until_ready()
            step_time = time() - start_time

            outputs = reshape_to_single_device(inputs_tree=outputs, global_batch_size=hparams.synthesis.batch_size)  # [batch_size, H, W, C]
            outputs = np.array(outputs)

            # Save images
            for output in outputs:
                write_image_to_disk(
                    os.path.join(artifacts_folder, f'setup-{temp_i:01d}-image-{sample_i:04d}.png'),
                    output,
                    denormalizer=denormalizer
                )

                sample_i += 1

            print(f'Step: {step:04d} | Time/Step (sec): {step_time:.4f}      ', end='\r')
        print()


def _compute_per_dimension_divergence_stats(rng, train_dataset, params, devices, latents_folder):
    stats_filepath = os.path.join(latents_folder, 'div_stats.npy')

    p_compute_stats_step = jax.pmap(
        fun=jax.jit(compute_stats_step),
        axis_name='shards'
    )

    per_dim_divs = None
    for step, (inputs, _) in enumerate(train_dataset):
        # Stats computation step
        start_time = time()
        rng, *stats_step_rng = random.split(rng, num=jax.local_device_count() + 1)
        stats_step_rng = jax.device_put_sharded(stats_step_rng, devices)
        metrics = p_compute_stats_step(params, stats_step_rng, inputs)
        metrics = jax.tree_map(lambda x: x[0], metrics)
        metrics['per_variate_avg_divs'].block_until_ready()
        step_time = time() - start_time

        if per_dim_divs is None:
            per_dim_divs = metrics['per_variate_avg_divs']
        else:
            per_dim_divs += metrics['per_variate_avg_divs']

        print(f'Step: {step:04d} | Time/Step (sec): {step_time:.4f}    ', end='\r')

    print()
    print(f'Stats computation finished after {step + 1} iterations!')

    per_dim_divs /= (step + 1)
    np.save(stats_filepath, np.array(per_dim_divs, dtype=np.float32))


def _encoding_mode(rng, train_dataset, params, devices, latents_folder):
    # Load div stats from disk and create variate_masks
    if not os.path.isfile(os.path.join(latents_folder, 'div_stats.npy')):
        raise FileNotFoundError('Run div_stats mode before trying to do encoding!')

    div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
    variate_masks = get_variate_masks(div_stats)

    p_encode_step = jax.pmap(
        fun=jax.jit(partial(encode_step, variate_masks=variate_masks)),
        axis_name='shards',
    )

    encodings = {'images': {}, 'latent_codes': {}}
    for step, (inputs, filenames) in enumerate(train_dataset):
        # Reconstruction step
        start_time = time()
        rng, *encode_step_rng = random.split(rng, num=jax.local_device_count() + 1)
        encode_step_rng = jax.device_put_sharded(encode_step_rng, devices)
        posterior_dist = p_encode_step(params, encode_step_rng, inputs)
        posterior_dist[list(posterior_dist.keys())[0]].block_until_ready()
        step_time = time() - start_time

        # Compute metrics we care about
        # shape: n_layers_subset x [global_batch_size, H, W, n_variates_subset, 2]
        posterior_dist = reshape_to_single_device(inputs_tree=posterior_dist, global_batch_size=hparams.synthesis.batch_size)
        # Bring down data to CPU/RAM and reshape to n_layers_subset, global_batch_size x [H, W, n_variates_subset, 2]
        assert len(filenames) == posterior_dist[list(posterior_dist.keys())[0]].shape[0]
        posterior_dist = jax.tree_map(lambda x: {name.decode('utf-8'): xa for name, xa in zip(filenames, list(np.array(x)))}, posterior_dist)

        # Save new files of each layer
        if encodings['latent_codes'] == {}:
            # Put first batch
            encodings['latent_codes'] = posterior_dist
        else:
            # Update files of each layer
            assert posterior_dist.keys() == encodings['latent_codes'].keys()
            for layer_key, layer_dict in posterior_dist.items():
                encodings['latent_codes'][layer_key].update(layer_dict)

        # Set filename - latent mappings
        log_inputs = jnp.reshape(inputs, newshape=(hparams.synthesis.batch_size,) + inputs.shape[2:])
        assert len(filenames) == len(log_inputs)
        # assert len(posterior_dist[0]) == n_layers_subset
        for filename, input_image in zip(filenames, np.array(log_inputs)):
            encodings['images'][filename.decode('utf-8')] = input_image

        print(f'Step: {step:04d} | Time/Step (sec): {step_time:.4f}    ', end='\r')

    print()
    print('Inefficiently transposing the encoded latents...')
    encodings['latent_codes'] = transpose_dicts(encodings['latent_codes'])

    assert encodings['images'].keys() == encodings['latent_codes'].keys()  # All layers share the same filename keys
    print(f'Encoding finished after {step + 1} iterations!')
    with open(os.path.join(latents_folder, f'encodings_seed_{hparams.run.seed}.pkl'), 'wb') as handle:
        pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def general_inference(rng, params, artifacts_folder, latents_folder, dataset, mode):
    devices = jax.devices()
    params = jax_utils.replicate(params, devices=devices)
    denormalizer = Denormalizer()
    ssim_metric = StructureSimilarityIndexMap(hparams.data.channels, denormalizer=denormalizer)

    if mode == 'reconstruction':
        _reconstruction_mode(rng, dataset, params, devices, ssim_metric, denormalizer, artifacts_folder, latents_folder)

    elif mode == 'generation':
        _generation_mode(rng, params, devices, denormalizer, artifacts_folder, latents_folder)

    elif mode == 'encoding':
        _encoding_mode(rng, dataset, params, devices, latents_folder)

    elif mode == 'div_stats':
        _compute_per_dimension_divergence_stats(rng, dataset, params, devices, latents_folder)

    else:
        raise ValueError(f'Unknown Mode {mode}')


def synthesize(rng, params, dataset, logdir):
    artifacts_folder = os.path.join(logdir, 'synthesis-images')
    latents_folder = os.path.join(logdir, 'latents')
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(latents_folder, exist_ok=True)

    general_inference(rng, params, artifacts_folder, latents_folder, dataset, mode=hparams.synthesis.synthesis_mode)
