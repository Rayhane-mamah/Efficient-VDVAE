import torch
import os
import numpy as np
from hparams import HParams
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def get_logdir():
    return f'logs-{hparams.run.name}'


def transpose_dicts(dct):
    d = defaultdict(dict)
    for key1, inner in dct.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return d


def get_variate_masks(stats):
    thresh = np.quantile(stats, 1 - hparams.synthesis.variates_masks_quantile)
    return stats > thresh


def scale_pixels(img):
    img = np.floor(img / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (8 - hparams.data.num_bits)
    shift = scale = (2 ** 8 - 1) / 2
    img = (img - shift) / scale  # Images are between [-1, 1]
    return img


def effective_pixels():
    if hparams.data.dataset_source == 'binarized_mnist':
        return 28 * 28 * hparams.data.channels
    else:
        return hparams.data.target_res * hparams.data.target_res * hparams.data.channels


def one_hot(indices, depth, dim):
    indices = indices.unsqueeze(dim)
    size = list(indices.size())
    size[dim] = depth
    y_onehot = torch.zeros(size, device=torch.device('cuda'))
    y_onehot.zero_()
    y_onehot.scatter_(dim, indices, 1)

    return y_onehot


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def assert_CUDA_and_hparams_gpus_are_equal():
    print('Running on: ', torch.cuda.device_count(), ' GPUs')
    assert hparams.run.num_gpus == torch.cuda.device_count()


def load_checkpoint_if_exists(checkpoint_path, rank):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(rank))
    except FileNotFoundError:
        checkpoint = {'global_step': -1,
                      'model_state_dict': None,
                      'ema_model_state_dict': None,
                      'optimizer_state_dict': None,
                      'scheduler_state_dict': None}
    return checkpoint


def create_checkpoint_manager_and_load_if_exists(model_directory='.', rank=0):
    checkpoint_path = os.path.join(model_directory, f'checkpoints-{hparams.run.name}')
    checkpoint = load_checkpoint_if_exists(checkpoint_path, rank)

    return checkpoint, checkpoint_path


def get_logdir():
    return f'logs-{hparams.run.name}'


def create_tb_writer(mode):
    logdir = get_logdir()
    tbdir = os.path.join(logdir, mode)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(tbdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tbdir)

    return writer, logdir


def get_same_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    # Reverse order for F.pad
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("Can't have the stride and dilation rate over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]
        if p % 2 == 0:
            p = (p // 2, p // 2)
        else:
            p = (int(np.ceil(p / 2)), int(np.floor(p / 2)))

        p_ += p

    return tuple(p_)


def get_valid_padding(n_dims=2):
    p_ = (0,) * 2 * n_dims
    return p_


def get_causal_padding(kernel_size, strides, dilation_rate, n_dims=2):
    p_ = []
    for i in range(n_dims - 1, -1, -1):
        if strides[i] > 1 and dilation_rate[i] > 1:
            raise ValueError("can't have the stride and dilation over 1")
        p = (kernel_size[i] - strides[i]) * dilation_rate[i]

        p_ += (p, 0)

    return p_


def compute_latent_dimension():
    assert np.prod(hparams.model.down_strides) == np.prod(hparams.model.up_strides)

    return hparams.data.target_res // np.prod(hparams.model.down_strides)
