import copy

from hparams import HParams
import os
import numpy as np

hparams = HParams('.', name="efficient_vdvae")

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from numpy.random import seed
import random

try:
    from .model.model import train, get_optimizer
    from .model.def_model import UniversalAutoEncoder
    from .utils.utils import assert_CUDA_and_hparams_gpus_are_equal, \
        create_checkpoint_manager_and_load_if_exists, create_tb_writer
    from .data.generic_data_loader import train_val_data_generic, create_filenames_list
    from .data.cifar10_data_loader import train_val_data_cifar10
    from .data.mnist_data_loader import train_val_data_mnist
    from .data.imagenet_data_loader import train_val_data_imagenet
except (ImportError, ValueError):
    from model.model import train, get_optimizer
    from utils.utils import assert_CUDA_and_hparams_gpus_are_equal, \
        create_checkpoint_manager_and_load_if_exists, create_tb_writer
    from data.generic_data_loader import train_val_data_generic, create_filenames_list
    from data.cifar10_data_loader import train_val_data_cifar10
    from data.imagenet_data_loader import train_val_data_imagenet
    from data.mnist_data_loader import train_val_data_mnist
    from model.def_model import UniversalAutoEncoder

local_rank = int(os.environ["LOCAL_RANK"])

assert_CUDA_and_hparams_gpus_are_equal()

# set the device
device = torch.device(local_rank)

dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(local_rank)

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


def cleanup():
    dist.destroy_process_group()


def main():
    model = UniversalAutoEncoder()
    model.to(device)
    with torch.no_grad():
        _ = model(torch.ones((1, hparams.data.channels, hparams.data.target_res, hparams.data.target_res)).cuda())

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Train step generator trainable params {:.3f}m.'.format(
        np.sum([np.prod(v.size()) for v in model_parameters]) / 1000000))

    ema_model = copy.deepcopy(model)

    checkpoint, checkpoint_path = create_checkpoint_manager_and_load_if_exists(rank=local_rank)

    optimizer, schedule = get_optimizer(model=model,
                                        type=hparams.optimizer.type,
                                        learning_rate=hparams.optimizer.learning_rate,
                                        beta_1=hparams.optimizer.beta1,
                                        beta_2=hparams.optimizer.beta2,
                                        epsilon=hparams.optimizer.epsilon,
                                        weight_decay_rate=hparams.loss.l2_weight if hparams.loss.use_weight_decay else 0.,
                                        decay_scheme=hparams.optimizer.learning_rate_scheme,
                                        warmup_steps=hparams.optimizer.warmup_steps,
                                        decay_steps=hparams.optimizer.decay_steps,
                                        decay_rate=hparams.optimizer.decay_rate,
                                        decay_start=hparams.optimizer.decay_start,
                                        min_lr=hparams.optimizer.min_learning_rate,
                                        last_epoch=torch.tensor(checkpoint['global_step']),
                                        checkpoint=checkpoint)

    if checkpoint['model_state_dict'] is not None:
        if hparams.train.resume_from_ema:
            print('Resuming from EMA model')
            model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            print('Loaded Model Checkpoint')
            model.load_state_dict(checkpoint['model_state_dict'])

    if checkpoint['ema_model_state_dict'] is not None:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        print('EMA Loaded from checkpoint')
    else:
        ema_model.load_state_dict(model.state_dict())
        print('Copy EMA from model')


    model = model.to(device)
    if ema_model is not None:
        ema_model = ema_model.to(device)
        ema_model.requires_grad_(False)

    model = DistributedDataParallel(model)

    if hparams.data.dataset_source in ['ffhq', 'celebaHQ', 'celeba', 'ffhq1024']:
        train_files, train_filenames = create_filenames_list(hparams.data.train_data_path)
        val_files, val_filenames = create_filenames_list(hparams.data.val_data_path)
        train_loader, val_loader = train_val_data_generic(train_files, train_filenames, val_files, val_filenames,
                                                          hparams.run.num_gpus, local_rank)
    elif hparams.data.dataset_source == 'cifar10':
        train_loader, val_loader = train_val_data_cifar10(hparams.run.num_gpus, local_rank)
    elif hparams.data.dataset_source == 'binarized_mnist':
        train_loader, val_loader = train_val_data_mnist(hparams.run.num_gpus, local_rank)
    elif hparams.data.dataset_source == 'imagenet':
        train_loader, val_loader = train_val_data_imagenet(hparams.run.num_gpus, local_rank)
    else:
        raise ValueError(f'Dataset{hparams.data.dataset_source} is not included.')

    # Book Keeping
    writer_train, logdir = create_tb_writer(mode='train')
    writer_val, _ = create_tb_writer(mode='val')

    # Train model
    train(model, ema_model, optimizer, schedule, train_loader, val_loader, checkpoint['global_step'], writer_train,
          writer_val, checkpoint_path, device, local_rank)
    cleanup()


if __name__ == '__main__':
    main()
