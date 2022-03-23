import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
from sklearn.utils import shuffle
import shutil

input_dir = sys.argv[1]
output_dir = sys.argv[2]
n_val_files = int(sys.argv[3])


def get_partitioned_filenames():
    filenames = os.listdir(input_dir)
    filenames = shuffle(filenames)

    train_filenames = filenames[:-n_val_files]
    val_filenames = filenames[-n_val_files:]

    return train_filenames, val_filenames


def process_one_file(filename, split):
    output_folder = os.path.join(output_dir, split)
    os.makedirs(output_folder, exist_ok=True)

    shutil.move(os.path.join(input_dir, filename), os.path.join(output_folder, filename))


def main():
    splits = ['train_data', 'val_data']
    all_filenames = get_partitioned_filenames()

    for split, split_filenames in zip(splits, all_filenames):
        print(f'Processing {split}..')
        executor = ProcessPoolExecutor(max_workers=256)
        futures = [executor.submit(partial(process_one_file, filename, split)) for filename in split_filenames]
        _ = [future.result() for future in tqdm(futures)]


if __name__ == '__main__':
    main()
