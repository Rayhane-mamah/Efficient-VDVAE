import torch.utils.data
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.utils import shuffle
from hparams import HParams
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def read_imagenet_images(path):
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))]
    data = np.concatenate([np.load(f)['data'] for f in files], axis=0) # [samples, C * H * W]
    data = data.reshape(
        [data.shape[0], hparams.data.channels, hparams.data.target_res, hparams.data.target_res])  # [samples, C, H, W]

    data = data.transpose([0, 2, 3, 1])  # [samples, H, W, C]
    assert data.shape[1] == data.shape[2] == hparams.data.target_res
    assert data.shape[3] == hparams.data.channels

    data = shuffle(data)
    print('Number of Images:', len(data))
    print('Path: ', path)
    return data


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
        shift = scale = (2 ** 8 - 1) / 2
        img = (img - shift) / scale  # Images are between [-1, 1]
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
    Normalize(),
    MinMax(),
])


def make_toy_filenames(data):
    return [f'image_{i}' for i in range(data.shape[0])]


class imagenet_dataset(torch.utils.data.Dataset):
    def __init__(self, images, mode):
        self.filenames = make_toy_filenames(images)
        self.mode = mode
        self.images = shuffle(images)

    def __getitem__(self, idx):
        if self.mode in 'train':
            img = self.images[idx]
            img = train_transform(img)
            return img

        elif self.mode in ['val', 'test', 'div_stats']:
            img = self.images[idx]
            img = valid_transform(img)
            return img

        elif self.mode == 'encode':
            img = self.images[idx]
            filename = self.filenames[idx]
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


def train_val_data_imagenet(world_size, rank):
    train_images = read_imagenet_images(hparams.data.train_data_path)
    train_imagenet = imagenet_dataset(train_images, mode='train')
    train_sampler = DistributedSampler(train_imagenet, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(sampler=train_sampler,
                                               dataset=train_imagenet,
                                               batch_size=hparams.train.batch_size // hparams.run.num_gpus,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=2,
                                               drop_last=True, prefetch_factor=3)

    val_images = read_imagenet_images(hparams.data.val_data_path)
    val_imagenet = imagenet_dataset(val_images, mode='val')
    val_sampler = DistributedSampler(val_imagenet, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(sampler=val_sampler,
                                             dataset=val_imagenet,
                                             batch_size=hparams.val.batch_size // hparams.run.num_gpus,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2,
                                             drop_last=True, prefetch_factor=3)

    return train_loader, val_loader


def synth_imagenet_data():
    synth_images = read_imagenet_images(hparams.data.synthesis_data_path)
    synth_images = imagenet_dataset(synth_images, mode='test')
    synth_loader = torch.utils.data.DataLoader(
        dataset=synth_images,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus,
        drop_last=False)
    return synth_loader


def encode_imagenet_data():
    images = read_imagenet_images(hparams.data.train_data_path)
    images = imagenet_dataset(images, mode='encode')
    encode_loader = torch.utils.data.DataLoader(
        dataset=images,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus,
        drop_last=False)
    return encode_loader


def stats_imagenet_data():
    images = read_imagenet_images(hparams.data.train_data_path)
    images = imagenet_dataset(images, mode='div_stats')
    stats_loader = torch.utils.data.DataLoader(
        dataset=images,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus,
        drop_last=False)
    return stats_loader
