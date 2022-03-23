#!/bin/sh

mkdir -p ../dataset_dumps
mkdir -p ../datasets

if [ "$1" = "imagenet32" ]; then
  if [ ! -f ../dataset_dumps/Imagenet32_train_npz.zip ]; then
    echo "downloading imagenet32 training data"
    wget https://image-net.org/data/downsample/Imagenet32_train_npz.zip -P ../dataset_dumps

  fi

  if [ ! -f ../dataset_dumps/Imagenet32_val_npz.zip ]; then
    echo "downloading imagenet32 validation data"
    wget https://image-net.org/data/downsample/Imagenet32_val_npz.zip -P ../dataset_dumps

  fi

  echo "processing imagenet32"
  unzip ../dataset_dumps/Imagenet32_train_npz.zip -d ../dataset_dumps
  unzip ../dataset_dumps/Imagenet32_val_npz.zip -d ../dataset_dumps

  mkdir -p ../datasets/imagenet_32
  mv ../dataset_dumps/Imagenet32_train_npz ../datasets/imagenet_32/train_data
  mv ../dataset_dumps/Imagenet32_val_npz ../datasets/imagenet_32/val_data

elif [ "$1" = "imagenet64" ]; then
  if [ ! -f ../dataset_dumps/Imagenet64_train_part1_npz.zip ]; then
    echo "downloading imagenet64 training part 1 data"
    wget https://image-net.org/data/downsample/Imagenet64_train_part1_npz.zip -P ../dataset_dumps

  fi

  if [ ! -f ../dataset_dumps/Imagenet64_train_part2_npz.zip ]; then
    echo "downloading imagenet64 training part 2 data"
    wget https://image-net.org/data/downsample/Imagenet64_train_part2_npz.zip -P ../dataset_dumps

  fi

  if [ ! -f ../dataset_dumps/Imagenet64_val_npz.zip ]; then
    echo "downloading imagenet64 validation data"
    wget https://image-net.org/data/downsample/Imagenet64_val_npz.zip -P ../dataset_dumps

  fi

  echo "processing imagenet64"
  unzip ../dataset_dumps/Imagenet64_train_part1_npz.zip -d ../dataset_dumps
  unzip ../dataset_dumps/Imagenet64_train_part2_npz.zip -d ../dataset_dumps
  unzip ../dataset_dumps/Imagenet64_val_npz.zip -d ../dataset_dumps

  mkdir -p ../datasets/imagenet_64
  mv ../dataset_dumps/Imagenet64_train_part1_npz ../datasets/imagenet_64/train_data
  mv ../dataset_dumps/Imagenet64_train_part2_npz/* ../datasets/imagenet_64/train_data
  mv ../dataset_dumps/Imagenet64_val_npz ../datasets/imagenet_64/val_data

elif [ "$1" = "celeba" ]; then
  # Since automatic downloading from the source Google drive with wget/TF API/Pytorch API can be buggy
  # we require a manual download to make sure it will work
  # data: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
  # partitions: https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg
  if [ ! -f ../dataset_dumps/img_align_celeba.zip ]; then
    echo "CelebA dataset file not found under <project_path>/dataset_dumps/!
    Please manually download it from: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ \n
    Please make sure the file is under <project_path>/dataset_dumps/img_align_celeba.zip"
    exit 0

  fi

  if [ ! -f ../dataset_dumps/list_eval_partition.txt ]; then
    echo "CelebA partition file not found under <project_path>/dataset_dumps/!
    Please manually download it from: https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg \n
    Please make sure the file is under <project_path>/dataset_dumps/list_eval_partition.txt"
    exit 0

  fi

  echo "processing celeba"
  unzip ../dataset_dumps/img_align_celeba.zip -d ../dataset_dumps
  CUDA_VISIBLE_DEVICES=-1 python preprocess_celeba64.py

elif [ "$1" = "celebahq" ]; then
  # For convenience, we download the filtered dataset from this work:
  # https://github.com/suvojit-0x55aa/celebA-HQ-dataset-download
  if [ ! -f ../dataset_dumps/data1024x1024.zip ]; then
    echo "downloading celebahq"
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies
    --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-LFFkFKNuyBO1sjkM4t_AArIXr3JAOyl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    &id=1-LFFkFKNuyBO1sjkM4t_AArIXr3JAOyl" -O ../dataset_dumps/data1024x1024.zip && rm -rf /tmp/cookies.txt

  fi

  echo "processing celebahq"
  unzip ../dataset_dumps/data1024x1024.zip -d ../dataset_dumps

  CUDA_VISIBLE_DEVICES=-1 python random_split.py ../dataset_dumps/data1024x1024 ../datasets/celebAHQ 3000

elif [ "$1" = "ffhq" ]; then
  if [ ! -f ../dataset_dumps/images1024x1024.zip ]; then
    echo "downloading ffhq"
    wget https://archive.org/download/ffhq-dataset/images1024x1024.zip -P ../dataset_dumps

  fi

  echo "processing ffhq"
  unzip ../dataset_dumps/images1024x1024.zip -d ../dataset_dumps
  rm ../dataset_dumps/images1024x1024/LICENSE.txt

  CUDA_VISIBLE_DEVICES=-1 python random_split.py ../dataset_dumps/images1024x1024 ../datasets/ffhq 7000

else
  echo "Argument $1 not recognized! Accepted arguments: imagenet32, imagenet64, celeba, celebahq, ffhq"


echo "Finished installing $1 datasets!"
echo "If your disk space is running low, feel free to remove the contents of <project_path>/dataset_dumps."
fi
