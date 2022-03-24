import torch.utils.data
import tensorflow as tf
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.utils import shuffle
from hparams import HParams
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def download_cifar10_datasets():
    # Ignore labels
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    x_train = shuffle(x_train)
    x_test = shuffle(x_test,
                     random_state=101)  # Fix this seed to not overlap val and test between train and inference runs

    x_val = x_test[:len(x_test) // 2]  # 5000
    x_test = x_test[len(x_test) // 2:]  # 5000
    return x_train.astype(np.uint8), x_val.astype(np.uint8), x_test.astype(np.uint8)


class Normalize(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Normalized image
        """
        img = np.asarray(img)
        img_dtype = img.dtype

        img = np.floor(img / np.uint8(2 ** (8 - hparams.data.num_bits))) * 2 ** (8 - hparams.data.num_bits)
        img = img.astype(img_dtype)

        return Image.fromarray(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class MinMax(object):
    def __call__(self, img):
        """
        :param img: PIL): Image

        :return: Tensor
        """
        img = np.asarray(img)
        img = img / (2 ** hparams.data.num_bits - 1)

        return torch.tensor(img).permute(2, 0, 1).contiguous().float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


if hparams.data.random_horizontal_flip:
    train_transform = transforms.Compose([
        Normalize(),
        transforms.RandomHorizontalFlip(),
        MinMax(),
    ])
else:
    train_transform = transforms.Compose([
        Normalize(),
        MinMax(),
    ])

valid_transform = transforms.Compose([
    MinMax(),
])


def make_toy_filenames(data):
    return [f'image_{i}' for i in range(data.shape[0])]


class CifarDataset(torch.utils.data.Dataset):
    def __init__(self, images, mode):

        self.mode = mode
        self.images = shuffle(images)
        self.filenames = make_toy_filenames(images)

    def __getitem__(self, idx):
        if self.mode in ['train', 'test']:
            img = self.images[idx]
            img = train_transform(img)
            return img

        elif self.mode in ['val', 'div_stats']:
            img = self.images[idx]
            img = valid_transform(img)
            return img
        elif self.mode == 'encode':
            filename = self.filenames[idx]
            img = self.images[idx]
            img = valid_transform(img)
            return img, filename
        else:
            raise ValueError(f'Unknown Mode {self.mode}')

    def __len__(self):
        if self.mode in ['train', 'test', 'encode']:
            return len(self.images)
        elif self.mode == 'val':
            return hparams.val.n_samples_for_validation
        elif self.mode == 'div_stats':
            return round(len(self.images) * hparams.synthesis.div_stats_subset_ratio)


def train_val_data_cifar10(world_size, rank):
    train_data, valid_data, _ = download_cifar10_datasets()
    train_data = CifarDataset(train_data, mode='train')
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(sampler=train_sampler,
                                               dataset=train_data,
                                               batch_size=hparams.train.batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=2,
                                               drop_last=True, prefetch_factor=3)
    valid_data = CifarDataset(valid_data, mode='val')
    val_sampler = DistributedSampler(valid_data, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(sampler=val_sampler,
                                             dataset=valid_data,
                                             batch_size=hparams.val.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2,
                                             drop_last=True, prefetch_factor=3)

    return train_loader, val_loader


def synth_cifar_data():
    _, _, test_data = download_cifar10_datasets()
    test_data = CifarDataset(test_data, mode='test')
    synth_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus,
        drop_last=False)
    return synth_loader


def encode_cifar_data():
    train_data, _, _ = download_cifar10_datasets()
    train_data = CifarDataset(train_data, mode='test')
    encode_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus,
        drop_last=False)
    return encode_loader


def stats_cifar_data():
    train_data, _, _ = download_cifar10_datasets()
    train_data = CifarDataset(train_data, mode='div_stats')
    stats_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus,
        drop_last=False)
    return stats_loader
