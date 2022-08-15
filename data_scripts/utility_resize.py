from concurrent.futures import ProcessPoolExecutor
from functools import partial
from sklearn.utils import shuffle
from PIL import Image
import os
from random import randint
import argparse
from tqdm import tqdm
import numpy as np


def _process_one_image(filename, in_path, out_path, resolution, resize_type, repeat_n):
    filepath = os.path.join(in_path, filename)
    image = Image.open(filepath).convert('RGB')

    if resize_type == 'reshape':
        out_image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        assert np.asarray(out_image).shape == (resolution, resolution, 3)
        out_image.save(os.path.join(out_path, filename))

    elif resize_type == 'center_crop':
        # Crop the image in the center to the minimum dimension
        # or the desired resolution if the latter is smaller than the image dimensions
        width, height = image.size
        min_dim = min(width, height, resolution)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = (width + min_dim) // 2
        bottom = (height + min_dim) // 2

        out_image = image.crop((left, top, right, bottom))

        # If the min dim is smaller than the resolution, we have to upsample
        if min_dim != resolution:
            out_image = out_image.resize((resolution, resolution), Image.Resampling.LANCZOS)

        assert np.asarray(out_image).shape == (resolution, resolution, 3)
        out_image.save(os.path.join(out_path, filename))

    elif resize_type == 'random_crop':
        # Random crop the image to the minimum dimension or the desired resolution
        # if the latter is smaller than the image dimensions
        width, height = image.size
        min_dim = min(width, height, resolution)

        for idx in range(repeat_n):
            randw = randint(0, width - min_dim)
            randh = randint(0, height - min_dim)

            out_image = image.crop((randw, randh, randw + min_dim, randh + min_dim))

            # If the min dim is smaller than the resolution, we have to upsample
            if min_dim != resolution:
                out_image = out_image.resize((resolution, resolution), Image.Resampling.LANCZOS)

            assert np.asarray(out_image).shape == (resolution, resolution, 3)
            out_image.save(os.path.join(out_path, filename.replace('.png', f'_re{idx:02d}.png')))
    else:
        raise ValueError(f'Provided resize type value not known!! Please verify: {resize_type}')


def preprocess(in_path, out_path, resolution, resize_type, repeat_n):
    print('Preprocessing...')
    os.makedirs(out_path, exist_ok=True)

    if not os.path.exists(in_path):
        raise ValueError(f'Provided Input path doesn\'t exist, please verify your input: {in_path}')

    futures = []
    executor = ProcessPoolExecutor(max_workers=256)

    filenames = shuffle(os.listdir(in_path))

    for idx, filename in enumerate(filenames):
        futures.append(
            executor.submit(
                partial(
                    _process_one_image,
                    filename,
                    in_path,
                    out_path,
                    resolution,
                    resize_type,
                    repeat_n
                )
            )
        )

    return [future.result() for future in tqdm(futures)]


def run():
    parser = argparse.ArgumentParser(description='Run Custom Data Preprocessor.')

    parser.add_argument('--in-path', help='Input directory that has all the images of the custom dataset.', required=True)
    parser.add_argument('--out-path', help='Output directory that will have all the images of the custom dataset, preprocessed.', required=True)
    parser.add_argument('--resolution', type=int, help='Desired resolution of the preprocessed dataset.', required=True)
    parser.add_argument('--resize-type', help='How to reach the desired resolution. One of (center_crop, random_crop, reshape).', required=True)
    parser.add_argument('--repeat-n', type=int,
                        help='Oversampling ratio for random crop. i.e: number of images to be created from each sample using random crop',
                        default=4, required=False)
    args = parser.parse_args()

    preprocess(in_path=args.in_path, out_path=args.out_path, resolution=args.resolution, resize_type=args.resize_type,
               repeat_n=args.repeat_n)


if __name__ == '__main__':
    run()
