import os
import pandas as pd
from PIL import Image
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

dump_folder = '../../tmp/'


def read_and_crop(image_filename):
    return Image.open(os.path.join(dump_folder, 'img_align_celeba', image_filename)).convert("RGB").crop((15, 40, 178 - 15, 218 - 30))


def save(path, image_filename, image):
    image.save(os.path.join(path, image_filename))


def process_one_file(image_filename, split):
    output_folder = os.path.join('../../celeba', split)
    os.makedirs(output_folder, exist_ok=True)

    image = read_and_crop(image_filename)
    save(output_folder, image_filename, image)


def get_partitioned_filenames():
    partition_list = pd.read_csv(os.path.join(dump_folder, 'list_eval_partition.txt'), sep=' ', names=['filenames', 'partition'])
    train_filenames = partition_list[partition_list.partition == 0]
    val_filenames = partition_list[partition_list.partition == 1]
    test_filenames = partition_list[partition_list.partition == 2]
    return train_filenames.filenames, val_filenames.filenames, test_filenames.filenames


def main():
    splits = ['train', 'val', 'test']
    all_filenames = get_partitioned_filenames()

    for split, split_filenames in zip(splits, all_filenames):
        print(f'Processing {split} data..')
        executor = ProcessPoolExecutor(max_workers=256)
        futures = [executor.submit(partial(process_one_file, filename, split)) for filename in split_filenames]
        _ = [future.result() for future in tqdm(futures)]


if __name__ == '__main__':
    main()
