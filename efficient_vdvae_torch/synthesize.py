import pickle

from hparams import HParams

hparams = HParams('.', name="efficient_vdvae")
import torch
from numpy.random import seed
import random
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

try:
    from .utils.utils import assert_CUDA_and_hparams_gpus_are_equal, create_checkpoint_manager_and_load_if_exists, \
        get_logdir, get_variate_masks, transpose_dicts
    from .data.generic_data_loader import synth_generic_data, encode_generic_data, stats_generic_data
    from .data.cifar10_data_loader import synth_cifar_data, encode_cifar_data, stats_cifar_data
    from .data.imagenet_data_loader import synth_imagenet_data, encode_imagenet_data, stats_imagenet_data
    from .data.mnist_data_loader import synth_mnist_data, encode_mnist_data, stats_mnist_data
    from .model.def_model import UniversalAutoEncoder
    from .model.model import reconstruction_step, generation_step, encode_step
    from .model.losses import StructureSimilarityIndexMap
    from .utils import temperature_functions
    from .model.div_stats_utils import KLDivergenceStats

except(ImportError, ValueError):
    from utils.utils import assert_CUDA_and_hparams_gpus_are_equal, create_checkpoint_manager_and_load_if_exists, \
        get_logdir, get_variate_masks, transpose_dicts
    from data.generic_data_loader import synth_generic_data, encode_generic_data, stats_generic_data
    from data.cifar10_data_loader import synth_cifar_data, encode_cifar_data, stats_cifar_data
    from data.imagenet_data_loader import synth_imagenet_data, encode_imagenet_data, stats_imagenet_data
    from data.mnist_data_loader import synth_mnist_data, encode_mnist_data, stats_mnist_data
    from model.def_model import UniversalAutoEncoder
    from model.model import reconstruction_step, generation_step, encode_step
    from model.losses import StructureSimilarityIndexMap
    from utils import temperature_functions
    from model.div_stats_utils import KLDivergenceStats

assert_CUDA_and_hparams_gpus_are_equal()
device = torch.device('cuda')

# Fix random seeds
torch.manual_seed(hparams.run.seed)
torch.manual_seed(hparams.run.seed)
torch.cuda.manual_seed(hparams.run.seed)
torch.cuda.manual_seed_all(hparams.run.seed)  # if you are using multi-GPU.
seed(hparams.run.seed)  # Numpy module.
random.seed(hparams.run.seed)  # Python random module.
torch.manual_seed(hparams.run.seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def write_image_to_disk(filepath, image):
    assert len(image.shape) == 3
    if hparams.data.dataset_source == ' binarized_mnist':
        assert image.shape[0] == 1
        image += 255.
    else:
        assert image.shape[0] == 3
        image = np.round(image * 127.5 + 127.5)
        image = image.astype(np.uint8)
        image = np.transpose(image, (1, 2, 0))
    im = Image.fromarray(image)
    im.save(filepath, format='png')


def reconstruction_mode(artifacts_folder, latents_folder, test_dataset, model, ssim_metric):
    artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/reconstructed')
    os.makedirs(artifacts_folder, exist_ok=True)

    if hparams.synthesis.mask_reconstruction:
        div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
        variate_masks = get_variate_masks(div_stats).astype(np.float32)
    else:
        variate_masks = None

    nelbos, ssims = 0., 0.
    sample_i = 0
    with torch.no_grad():
        for step, inputs in enumerate(test_dataset):
            inputs = inputs.to(device, non_blocking=True)
            outputs, reconstruction_loss, kl_div = reconstruction_step(model, inputs, variates_masks=variate_masks)

            targets = inputs
            nelbo = reconstruction_loss + kl_div
            ssim_per_batch = ssim_metric(targets, outputs, global_batch_size=hparams.synthesis.batch_size)
            ssims += ssim_per_batch
            nelbos += nelbo

            # Save images to disk
            for batch_i, (target, output) in enumerate(zip(targets, outputs)):
                if hparams.synthesis.save_target_in_reconstruction:
                    write_image_to_disk(
                        os.path.join(artifacts_folder, f'target-{sample_i:04d}.png'),
                        target.detach().cpu().numpy())
                write_image_to_disk(
                    os.path.join(artifacts_folder, f'image-{sample_i:04d}.png'),
                    output.detach().cpu().numpy())

                sample_i += 1
            print(f'Step: {step:04d}  | NELBO: {nelbo:.4f} | Reconstruction: {reconstruction_loss:.4f} | '
                  f'kl_div: {kl_div:.4f}| SSIM: {ssim_per_batch:.4f}    ', end='\r')

        nelbo = nelbos / (step + 1)
        ssim = ssims / (step + 1)
        print()
        print()
        print('===========================================')
        print(f'NELBO: {nelbo:.6f} | SSIM: {ssim:.6f}')


def generation_mode(artifacts_folder, latents_folder, model):
    artifacts_folder = artifacts_folder.replace('synthesis-images', 'synthesis-images/generated')
    os.makedirs(artifacts_folder, exist_ok=True)

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
            temp_fn = getattr(temperature_functions, temperature_setting[0])(temperature_setting[1],
                                                                             temperature_setting[2], n_layers=len(
                    hparams.model.down_strides))
            temperatures = [temp_fn(layer_i) for layer_i in range(len(hparams.model.down_strides))]

        else:
            raise ValueError(f'Temperature Setting {temperature_setting} not interpretable!!')

        sample_i = 0
        for step in range(hparams.synthesis.n_generation_batches):
            outputs, prior_zs = generation_step(model, temperatures=temperatures)
            for output in outputs:
                write_image_to_disk(os.path.join(artifacts_folder, f'setup-{temp_i:01d}-image-{sample_i:04d}.png'),
                                    output.detach().cpu().numpy())

                sample_i += 1

            print(f'Step: {step:04d} ', end='\r')
            print()


def compute_per_dimension_divergence_stats(dataset, model, latents_folder):
    kl_stats = KLDivergenceStats()
    stats_filepath = os.path.join(latents_folder, 'div_stats.npy')

    per_dim_divs = None
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(dataset)):
            inputs = inputs.to(device, non_blocking=True)
            predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
            kl_div = kl_stats(posterior_dist_list, prior_kl_dist_list, hparams.synthesis.batch_size)

            if per_dim_divs is None:
                per_dim_divs = kl_div
            else:
                per_dim_divs += kl_div

    per_dim_divs /= (step + 1)
    np.save(stats_filepath, per_dim_divs.detach().cpu().numpy())


def reshape_distribution(dist_list, variate_mask):
    """
    :param dist_list: n_layers, 2*  [ batch_size n_variates, H , W]
    :return: Tensors  of shape batch_size, H, W ,n_variates, 2
    H, W , n_variates will be different from each other in the list depending on which layer you are in.
    """
    dist = torch.stack(dist_list, dim=0)  # 2, batch_size, n_variates, H ,W
    dist = dist[:, :, variate_mask, :, :]  # Only take effective variates
    dist = torch.permute(dist, (1, 3, 4, 2, 0))  # batch_size, H ,W ,n_variates (subset), 2
    # dist = torch.unbind(dist, dim=0)  # Return a list of tensors of length batch_size
    return dist


def encoding_mode(latents_folder, dataset, model):
    train_codes_folder = os.path.join(latents_folder, 'train-code')
    os.makedirs(train_codes_folder, exist_ok=True)

    if not os.path.isfile(os.path.join(latents_folder, 'div_stats.npy')):
        raise FileNotFoundError('No div_stats found')
    # Load div stats from disk
    div_stats = np.load(os.path.join(latents_folder, 'div_stats.npy'))
    variate_masks = get_variate_masks(div_stats)

    encodings = {'images': {}, 'latent_codes': {}}
    model = model.eval()
    print('Starting Encoding mode \n')
    with torch.no_grad():
        for step, (inputs, filenames) in enumerate(tqdm(dataset)):
            inputs = inputs.to(device, non_blocking=True)
            # posterior_dist_list : n_layers, 2 (mean,std), batch_size, n_variates, H, W (List of List of Tensors)
            posterior_dist_list = reconstruction_step(model, inputs, variates_masks=variate_masks, mode='encode')

            # If the mask states all variables of a layer are not effective we don't collect any latents from that layer
            # n_layers , batch_size, [H, W, n_variates, 2]
            dist_dict = {}
            for i, (dist_list, variate_mask) in enumerate(zip(posterior_dist_list, variate_masks)):
                if variate_mask.any():
                    x = reshape_distribution(dist_list, variate_mask).detach().cpu().numpy()
                    v = {name: xa for name, xa in zip(filenames, list(x))}
                    dist_dict[i] = v

            if encodings['latent_codes'] == {}:
                # Put first batch
                encodings['latent_codes'] = dist_dict
            else:
                # Update files of each layer
                assert dist_dict.keys() == encodings['latent_codes'].keys()
                for layer_key, layer_dict in dist_dict.items():
                    encodings['latent_codes'][layer_key].update(layer_dict)

            inputs = inputs.detach().cpu().numpy()
            assert len(filenames) == len(inputs)
            for filename, input_image in zip(filenames, inputs):
                encodings['images'][filename] = input_image

    encodings['latent_codes'] = transpose_dicts(encodings['latent_codes'])
    print('Saving Encoded File')
    assert encodings['images'].keys() == encodings['latent_codes'][0].keys()
    with open(os.path.join(latents_folder, f'encodings_seed_{hparams.run.seed}.pkl'), 'wb') as handle:
        pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def synthesize(model, data, logdir, mode):
    ssim_metric = StructureSimilarityIndexMap(image_channels=hparams.data.channels)
    artifacts_folder = os.path.join(logdir, 'synthesis-images')
    latents_folder = os.path.join(logdir, 'latents')
    os.makedirs(artifacts_folder, exist_ok=True)
    os.makedirs(latents_folder, exist_ok=True)

    if mode == 'reconstruction':
        reconstruction_mode(artifacts_folder, latents_folder, data, model, ssim_metric)
    elif mode == 'generation':
        generation_mode(artifacts_folder, latents_folder, model)
    elif mode == 'encoding':
        encoding_mode(latents_folder, data, model)
    elif mode == 'div_stats':
        compute_per_dimension_divergence_stats(data, model, latents_folder)
    else:
        raise ValueError(f'Unknown Mode {mode}')


def synth_data():
    if hparams.data.dataset_source in ['ffhq', 'celebAHQ', 'celebA']:
        return synth_generic_data()
    elif hparams.data.dataset_source == 'cifar-10':
        return synth_cifar_data()
    elif hparams.data.dataset_source == 'binarized_mnist':
        return synth_mnist_data()
    elif hparams.data.dataset_source == 'imagenet':
        return synth_imagenet_data()
    else:
        raise ValueError(f'Dataset {hparams.data.dataset_source} is not included.')


def encode_data():
    if hparams.data.dataset_source in ['ffhq', 'celebAHQ', 'celebA']:
        return encode_generic_data()
    elif hparams.data.dataset_source == 'cifar-10':
        return encode_cifar_data()
    elif hparams.data.dataset_source == 'binarized_mnist':
        return encode_mnist_data()
    elif hparams.data.dataset_source == 'imagenet':
        return encode_imagenet_data()
    else:
        raise ValueError(f'Dataset {hparams.data.dataset_source} is not included.')


def stats_data():
    if hparams.data.dataset_source in ['ffhq', 'celebAHQ', 'celebA']:
        return stats_generic_data()
    elif hparams.data.dataset_source == 'cifar-10':
        return stats_cifar_data()
    elif hparams.data.dataset_source == 'binarized_mnist':
        return stats_mnist_data()
    elif hparams.data.dataset_source == 'imagenet':
        return stats_imagenet_data()
    else:
        raise ValueError(f'Dataset {hparams.data.dataset_source} is not included.')


def main():
    if hparams.run.num_gpus > 1 :
        raise ValueError("Inference mode in Pytorch only supports single GPU")
    model = UniversalAutoEncoder()
    model = model.to(device)
    _ = model(torch.ones((1, hparams.data.channels, hparams.data.target_res, hparams.data.target_res)).cuda())
    # count_parameters(model)
    checkpoint, checkpoint_path = create_checkpoint_manager_and_load_if_exists(rank=0)

    if hparams.synthesis.load_ema_weights:
        assert checkpoint['ema_model_state_dict'] is not None
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print('EMA model is loaded')
    else:
        assert checkpoint['model_state_dict'] is not None
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Model Checkpoint is loaded')
    print(checkpoint_path)

    if hparams.synthesis.synthesis_mode == 'reconstruction':
        data_loader = synth_data()
    elif hparams.synthesis.synthesis_mode == 'encoding':
        data_loader = encode_data()
    elif hparams.synthesis.synthesis_mode == 'div_stats':
        data_loader = stats_data()
    elif hparams.synthesis.synthesis_mode == 'generation':
        data_loader = None
    else:
        raise ValueError(f'Unknown Mode {hparams.synthesis.synthesis_mode}')

    # Synthesis using pretrained model
    logdir = get_logdir()
    synthesize(model, data_loader, logdir, mode=hparams.synthesis.synthesis_mode)


if __name__ == '__main__':
    main()
