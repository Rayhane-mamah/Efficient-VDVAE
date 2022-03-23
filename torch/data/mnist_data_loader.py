import numpy as np
from sklearn.utils import shuffle
from hparams import HParams
import tensorflow as tf
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

hparams = HParams.get_hparams_by_name("global_local_memcodes")


def download_mnist_datasets():
    # Ignore labels
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train[:, np.newaxis, :, :]  # (60000, 28, 28, 1)
    x_test = x_test[:, np.newaxis, :, :]  # (10000, 28, 28, 1)
    x_train, x_test = x_train / 255., x_test / 255.

    # make images of size 32x32
    x_train = np.pad(x_train, pad_width=((0, 0), (0, 0), (2, 2), (2, 2)))  # (60000, 1, 32, 32)
    x_test = np.pad(x_test, pad_width=((0, 0), (0, 0), (2, 2), (2, 2)))  # (60000, 1, 32, 32)

    x_train = shuffle(x_train)
    x_test = shuffle(x_test,
                     random_state=101)  # Fix this seed to not overlap val and test between train and inference runs

    x_val = x_test[:len(x_test) // 2]  # 5000
    x_test = x_test[len(x_test) // 2:]  # 5000

    x_val, x_test = torch.tensor(x_val), torch.tensor(x_test)
    x_val = torch.Tensor(x_val.size()).bernoulli_(x_val)  # fix binarization
    x_test = torch.tensor(x_test.size()).bernoulli_(x_test)

    return x_train, x_val, x_test


class Binarize(object):
    def __call__(self, img):
        """
        :param numpy array

        :return: Tensor
        """
        img = torch.Tensor(img)
        return torch.Tensor(img.size()).bernoulli_(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


transform = transforms.Compose([
    Binarize()
])


def make_toy_filenames(data):
    return [f'image_{i}' for i in range(data.shape[0])]


class mnist_dataset(torch.utils.data.Dataset):
    def __init__(self, images, mode):
        self.images = shuffle(images)
        self.mode = mode
        self.filenames = make_toy_filenames(images)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img = self.images[idx]
            img = transform(img)
            return img
        elif self.mode in ['val', 'div_stats']:
            img = self.images[idx]
            return img
        elif self.mode == 'encode':
            filename = self.filenames[idx]
            img = self.images[idx]
            return img, filename
        else:
            raise ValueError(f'Unknown Mode {self.mode}')

    def __len__(self):
        if self.mode == 'div_stats':
            return round(len(self.images) * hparams.synthesis.div_stats_subset_ratio)
        else:
            return len(self.images)


def train_val_data_mnist(world_size, rank):
    train_images, val_images, _ = download_mnist_datasets()
    train_mnist = mnist_dataset(train_images, mode='train')
    train_sampler = DistributedSampler(train_mnist, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(sampler=train_sampler,
                                               dataset=train_mnist,
                                               batch_size=hparams.train.batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               num_workers=2,
                                               drop_last=True, prefetch_factor=3)

    val_mnist = mnist_dataset(val_images, mode='val')
    val_sampler = DistributedSampler(val_mnist, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(sampler=val_sampler,
                                             dataset=val_mnist,
                                             batch_size=hparams.val.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=2,
                                             drop_last=True, prefetch_factor=3)

    return train_loader, val_loader


def synth_mnist_data():
    _, _, test_images = download_mnist_datasets()
    synth_mnist = mnist_dataset(test_images, 'val')
    synth_loader = torch.utils.data.DataLoader(
        dataset=synth_mnist,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus)
    return synth_loader


def encode_mnist_data():
    train_data, _, _ = download_mnist_datasets()
    encode_mnist = mnist_dataset(train_data, 'encode')
    encode_loader = torch.utils.data.DataLoader(
        dataset=encode_mnist,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus)
    return encode_loader


def stats_mnist_data():
    train_data, _, _ = download_mnist_datasets()
    stats_mnist = mnist_dataset(train_data, 'div_stats')
    stats_loader = torch.utils.data.DataLoader(
        dataset=stats_mnist,
        batch_size=hparams.synthesis.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=hparams.run.num_cpus)
    return stats_loader
