<div align="center"> <h1>The Official Pytorch and JAX implementation of "Efficient-VDVAE: Less is more" <a href="https://arxiv.org/abs/2203.13751">Arxiv preprint</a></h1> </div>  
  
<div align="center">    
  <a>Louay&nbsp;Hazami</a>     
  &emsp; <b>&middot;</b> &emsp;    
  <a>Rayhane&nbsp;Mama</a>     
  &emsp; <b>&middot;</b> &emsp;    
  <a>Ragavan&nbsp;Thurairatnam</a>    
</div>    
<br>    
<br>   

 [![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/) [![](https://img.shields.io/badge/cs.LG-arXiv%3A2203.13751-%23B31B1B)](https://arxiv.org/abs/2203.13751) [![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/LICENSE)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-binarized-mnist)](https://paperswithcode.com/sota/image-generation-on-binarized-mnist?p=efficient-vdvae-less-is-more) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-cifar-10)](https://paperswithcode.com/sota/image-generation-on-cifar-10?p=efficient-vdvae-less-is-more)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-imagenet-32x32)](https://paperswithcode.com/sota/image-generation-on-imagenet-32x32?p=efficient-vdvae-less-is-more)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-imagenet-64x64)](https://paperswithcode.com/sota/image-generation-on-imagenet-64x64?p=efficient-vdvae-less-is-more)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-celeba-64x64)](https://paperswithcode.com/sota/image-generation-on-celeba-64x64?p=efficient-vdvae-less-is-more)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-celeba-256x256)](https://paperswithcode.com/sota/image-generation-on-celeba-256x256?p=efficient-vdvae-less-is-more)   [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-ffhq-256-x-256)](https://paperswithcode.com/sota/image-generation-on-ffhq-256-x-256?p=efficient-vdvae-less-is-more)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-vdvae-less-is-more/image-generation-on-ffhq-1024-x-1024)](https://paperswithcode.com/sota/image-generation-on-ffhq-1024-x-1024?p=efficient-vdvae-less-is-more) 

 
[Efficient-VDVAE](https://arxiv.org/abs/2203.13751) is a memory and compute efficient very deep hierarchical VAE. It converges faster and is more stable than current hierarchical VAE models. It also achieves SOTA likelihood-based performance on several image datasets.    
    
<p align="center">  
    <img src="images/unconditional_samples.png" width="1200">  
</p>  
  
## Pre-trained model checkpoints  
  
We provide checkpoints of pre-trained models on MNIST, CIFAR-10, Imagenet 32x32, Imagenet 64x64, CelebA 64x64, CelebAHQ 256x256 (5-bits and 8-bits), FFHQ 256x256 (5-bits and 8bits), CelebAHQ 1024x1024 and FFHQ 1024x1024 in the links in the table below. All provided models are the ones trained for table 4 of the [paper](https://arxiv.org/abs/2203.13751).

<table align="center">
    <thead align="center">
        <tr>
            <th rowspan=2 align="center"> Dataset </th>
            <th colspan=2 align="center"> Pytorch </th>
            <th colspan=2 align="center"> JAX </th>
            <th rowspan=2 align="center"> Negative ELBO </th>
        </tr>
        <tr>
	        <th align="center"> Logs </th>
	        <th align="center"> Checkpoints </th>
	        <th align="center"> Logs </th>
	        <th align="center"> Checkpoints </th>
        </tr>
    </thead>
    <tbody align="center">
        <tr>
            <td align="center">MNIST</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/binarized_mnist_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/binarized_mnist_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/binarized_mnist_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/binarized_mnist_baseline_checkpoints.zip">link</a></td>
            <td align="center">79.09 nats</td>
        </tr>
        <tr>
            <td align="center">CIFAR-10</td>
            <td align="center">Queued</td>
            <td align="center">Queued</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/cifar10_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/cifar10_baseline_checkpoints.zip">link</a></td>
            <td align="center">2.87 bits/dim</td>
        </tr>
        <tr>
            <td align="center">Imagenet 32x32</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/imagenet32_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/imagenet32_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/imagenet32_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/imagenet32_baseline_checkpoints.zip">link</a></td>
            <td align="center">3.58 bits/dim</td>
        </tr>
        <tr>
            <td align="center">Imagenet 64x64</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/imagenet64_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/imagenet64_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/imagenet64_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/imagenet64_baseline_checkpoints.zip">link</a></td>
            <td align="center">3.30 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebA 64x64</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebA64_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebA64_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebA64_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebA64_baseline_checkpoints.zip">link</a></td>
            <td align="center">1.83 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebAHQ 256x256 (5-bits)</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebAHQ256_5bits_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebAHQ256_5bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebAHQ256_5bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebAHQ256_5bits_baseline_checkpoints.zip">link</a></td>
            <td align="center">0.51 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebAHQ 256x256 (8-bits)</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebahq256_8bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebahq256_8bits_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebAHQ256_8bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebAHQ256_8bits_baseline_checkpoints.zip">link</a></td>
            <td align="center">1.35 bits/dim</td>
        </tr>
        <tr>
            <td align="center">FFHQ 256x256 (5-bits)</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq256_5bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq256_8bits_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/ffhq256_5bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/ffhq256_5bits_baseline_checkpoints.zip">link</a></td>
            <td align="center">0.53 bits/dim</td>
        </tr>
        <tr>
            <td align="center">FFHQ 256x256 (8-bits)</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq256_8bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq256_8bits_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/ffhq256_8bits_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/ffhq256_8bits_baseline_checkpoints.zip">link</a></td>
            <td align="center">2.17 bits/dim</td>
        </tr>
        <tr>
            <td align="center">CelebAHQ 1024x1024</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebAHQ1024_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/celebAHQ1024_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebAHQ1024_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/celebAHQ1024_baseline_checkpoints.zip">link</a></td>
            <td align="center">1.01 bits/dim</td>
        </tr>
        <tr>
            <td align="center">FFHQ 1024x1024</td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq1024_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/Pytorch/ffhq1024_baseline_checkpoints.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/ffhq1024_baseline_logs.zip">link</a></td>
            <td align="center"><a href="https://storage.googleapis.com/dessa-public-files/efficient_vdvae/JAX/ffhq1024_baseline_checkpoints.zip">link</a></td>
            <td align="center">2.30 bits/dim</td>
        </tr>
    </tbody>
</table>
 
### Notes: 

- Downloading from the *"Checkpoints"* link will download the minimal required files to resume training/do inference. The minimal files are the model checkpoint file and the saved hyper-parameters of the run (explained further below).
- Downloading from the *"Logs"* link will download additional pre-training logs such as tensorboard files or saved images from training. *"Logs"* also holds the saved hyper-parameters of the run.
- Downloaded *"Logs"* and/or *"Checkpoints"* should be always unzipped in their implementation folder (`efficient_vdvae_torch` for Pytorch checkpoints and `efficient_vdvae_jax` for JAX checkpoints).
- Some of the model checkpoints are missing in either Pytorch or JAX for the moment. We will update them soon.
  
## Pre-requisites   
To run this codebase, you need:  
  
- Machine that runs a linux based OS (tested on Ubuntu 20.04 (LTS))  
- GPUs (preferably more than 16GB)  
- [Docker](https://docs.docker.com/engine/install/ubuntu/)  
- Python 3.7 or higher  
- CUDA 11.1 or higher (can be installed from [here](https://developer.nvidia.com/cuda-11.1.0-download-archive))  

We recommend running all the code below inside a `Linux screen` or any other terminal multiplexer, since some commands can take hours/days to finish and you don't want them to die when you close your terminal.

### Note:

- If you're planning on running the JAX implementation, the installed JAX must use exactly the same CUDA and Cudnn versions installed. Our default [Dockerfile](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/build/Dockerfile) assumes the code will run with CUDA 11.4 or newer and should be changed otherwise. For more details, refer to [JAX
 installation](https://github.com/google/jax#installation).

## Installation  
  
To create the docker image used in both the Pytorch and JAX implementations:  
  
```  
cd build  
docker build -t efficient_vdvae_image .  
```  

### Note:
- If using JAX library on ampere architecture GPUs, it's possible to face a random GPU hanging problem when training on multiple GPUs 
([issue](https://github.com/google/jax/issues/8475)). In that case, we provide an 
[alternative docker image](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/build/ampere_Dockerfile) with an older version of JAX to bypass the issue until a solution is 
found. 

All code executions should be done within a docker container. To start the docker container, we provide a utility script:  
  
```  
sh docker_run.sh  # Starts the container and attaches terminal
cd /workspace/Efficient-VDVAE  # Inside docker container
```  
## Setup datasets  
  
All datasets can be automatically downloaded and pre-processed from the convenience script we provide:

```
cd data_scripts
sh download_and_preprocess.sh <dataset_name>
```

### Notes:
- `<dataset_name>` can be one of `(imagenet32, imagenet64, celeba, celebahq, ffhq)`. MNIST and CIFAR-10 datasets will get automatically downloaded later when training the model, and they do no require any dataset setup.
- For the `celeba` dataset, a manual download of `img_align_celeba.zip` and  `list_eval_partition.txt` files is necessary. Both files should be placed under `<project_path>/dataset_dumps/`.
- `img_align_celeba.zip` download [link](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ).
- `list_eval_partition.txt` download [link](https://drive.google.com/file/d/0B7EVK8r0v71pY0NSMzRuSXJEVkk/view?usp=sharing&resourcekey=0-i4TGCi_51OtQ5K9FSp4EDg).
  
## Setting the hyper-parameters  
  
In this repository, we use [hparams](https://github.com/Rayhane-mamah/hparams) library (already included in the Dockerfile) for hyper-parameter management:  
  
- Specify all run parameters (number of GPUs, model parameters, etc) in one `.cfg` file
- Hparams evaluates any expression used as "value" in the `.cfg` file. "value" can be any basic python object `(floats, strings, lists, etc)` or any python basic expression `(1/2, max(3, 7), etc.)` as long as the evaluation does not require any library importations or does not rely on other values from the `.cfg`.
- Hparams saves the configuration of previous runs for reproducibility, resuming training, etc.  
- All hparams are saved by name, and re-using the same name will recall the old run instead of making a new one.  
- The `.cfg` file is split into sections for readability, and all parameters in the file are accessible as class attributes in the codebase for convenience.  
- The HParams object keeps a global state throughout all the scripts in the code.  
  
We highly recommend having a deeper look into how this library works by reading the [hparams library documentation](https://github.com/Rayhane-mamah/hparams), the [parameters description](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/jax/hparams.cfg) and figures 4 and 5 in the [paper](https://arxiv.org/abs/2203.13751) before trying to run Efficient-VDVAE.  

We have heavily tested the robustness and stability of our approach, so changing the model/optimization hyper-parameters for memory load reduction should not introduce any drastic instabilities as to make the model untrainable. That is of course as long as the changes don't negate the important stability points we describe in the paper.

## Training the Efficient-VDVAE  
  
To run Efficient-VDVAE in Torch:  
  
```  
cd efficient_vdvae_torch  
# Set the hyper-parameters in "hparams.cfg" file  
# Set "NUM_GPUS_PER_NODE" in "train.sh" file  
sh train.sh  
```  
  
To run Efficient-VDVAE in JAX:  
  
```  
cd efficient_vdvae_jax  
# Set the hyper-parameters in "hparams.cfg" file  
python train.py  
```  
  
If you want to run the model with less GPUs than available on the hardware, for example 2 GPUs out of 8:  
  
```  
CUDA_VISIBLE_DEVICES=0,1 sh train.sh  # For torch  
CUDA_VISIBLE_DEVICES=0,1 python train.py  # For JAX  
```  
  
Models automatically create checkpoints during training. To resume a model from its last checkpoint, set its *`<run.name>`* in *`hparams.cfg`* file and re-run the same training commands.  
  
Since training commands will save the hparams of the defined run in the `.cfg` file. If trying to restart a pre-existing run (by re-using its name in `hparams.cfg`), we provide a convenience script for resetting saved runs:  
  
```  
cd efficient_vdvae_torch  # or cd efficient_vdvae_jax  
sh reset.sh <run.name>  # <run.name> is the first field in hparams.cfg  
```  

### Note:  
  
- To make things easier for new users, we provide example `hparams.cfg` files that can be used under the [egs](https://github.com/Rayhane-mamah/Efficient-VDVAE/tree/main/egs) folder. Detailed description of the role of each parameter is also inside [hparams.cfg](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/jax/hparams.cfg).
- Hparams in [egs](https://github.com/Rayhane-mamah/Efficient-VDVAE/tree/main/egs) are to be viewed **only** as guiding examples, they are not meant to be exactly similar to pre
-trained checkpoints or experiments done in the paper.
- While the example hparams under the naming convention `..._baseline.cfg` are not exactly the hparams of `C2` models in the paper (pre-trained checkpoints), they are easier to 
  design models that achieve the same performance and can be treated as equivalents to `C2` models.
  
## Monitoring the training process  
  
While writing this codebase, we put extra emphasis on verbosity and logging. Aside from the printed logs on terminal (during training), you can monitor the training progress and keep track of useful metrics using [Tensorboard](https://www.tensorflow.org/tensorboard):  
  
```  
# While outside efficient_vdvae_torch or efficient_vdvae_jax  
# Run outside the docker container
tensorboard --logdir . --port <port_id> --reload_multifile True  
```  

In the browser, navigate to `localhost:<port_id>` to visualize all saved metrics.

If Tensorboard is not installed (outside the docker container):
```
pip install --upgrade tensorboard
```
  
## Inference with the Efficient-VDVAE  
  
Efficient-VDVAE support multiple inference modes:  
  
- "reconstruction": Encodes then decodes the test set images and computes test NLL and SSIM.  
- "generation": Generates random images from the prior distribution. Randomness is controlled by the `run.seed` parameter.  
- "div_stats": Pre-computes the average KL divergence stats used to determine turned-off variates (refer to section 7 of the [paper](https://arxiv.org/abs/2203.13751)). Note: This mode needs to be run before "encoding" mode and before trying to do masked "reconstruction" (Refer to [hparams.cfg](https://github.com/Rayhane-mamah/Efficient-VDVAE/blob/main/jax/hparams.cfg) for a detailed description).  
- "encoding": Extracts the latent distribution from the inference model, pruned to the quantile defined by `synthesis.variates_masks_quantile` parameter. This latent distribution is usable in downstream tasks.  
  
To run the inference:  
  
```  
cd efficient_vdvae_torch  # or cd efficient_vdvae_jax  
# Set the inference mode in "logs-<run.name>/hparams-<run.name>.cfg"  
# Set the same <run.name> in "hparams.cfg"  
python synthesize.py  
```  
  
### Notes:  
- Since training a model with a name *`<run.name>`* will save that configuration under *`logs-<run.name>/hparams-<run.name>.cfg`* for reproducibility and error reduction. Any changes that one wants to make during inference time need to be applied on the saved hparams file (*`logs-<run.name>/hparams-<run.name>.cfg`*) instead of the main file *`hparams.cfg`*.  
- The torch implementation currently doesn't support multi-GPU inference. The JAX implementation does.  
  
## Using custom datasets

If you want to train the networks on your custom datasets, you need the following requisites:

- A folder with all the data samples saved as any image extension (png, jpg, etc), readable by PIL.
- The data should be split across at least two folders for train and val datasets. Any splitting strategy of your choice should work. (Not mandatory but it's highly discouraged to 
  evaluate on the train data).
- All data images must be square shaped, preferably in powers of 2. e.g: 64x64, 128x128 etc.

To use your custom dataset (in both training and inference), you only need to modify the `data` section of your `hparams.cfg` file. Specifically set `dataset_source = custom` 
then change the data paths and image metadata. 

For an example custom dataset of resolution `512` and grey scale, the `data` section of my `hparams.cfg` would look like:

```
[data]
# Data section: Defines the dataset parameters
# To change a dataset to run the code on:
#   - Change the data.dataset_source to reflect which dataset you're trying to run.
#           This controls which data loading scripts to use and how to normalize
#   - Change the paths. For all datasets but binarized_mnist and cifar-10, define where the data lives on disk.
#   - Change the metadata: Define the image resolution, the number of channels and the color bit-depth of the data.

# Dataset source. Can be one of ('binarized_mnist', 'cifar-10', 'imagenet', 'celebA', 'celebAHQ', 'ffhq', 'custom')
dataset_source = 'custom'

# Data paths. Not used for (binarized_mnist, cifar-10)
train_data_path = '../datasets/my_custom_data/train_data/'
val_data_path = '../datasets/my_custom_data/val_data/'
synthesis_data_path = '../datasets/my_custom_data/synthesis_data/'

# Image metadata
# Image resolution of the dataset (High and Width, assumed square)
target_res = 512
# Image channels of the dataset (Number of color channels)
channels = 1
# Image color depth in the dataset (bit-depth of each color channel)
num_bits = 8.
# Whether to do a random horizontal flip of images when loading the data (no applicable to MNIST)
random_horizontal_flip = True
```

Obviously, also change the model section of the `hparams.cfg` to create a model that works well with your data resolution. When in doubt, get inspired by the example hparams in the 
[egs](https://github.com/Rayhane-mamah/Efficient-VDVAE/tree/main/egs) folder.

### Notes:

- If your custom dataset isn't split between train and val, you can use the standalone utility script we provide:
```
cd data_scripts
python random_split.py <input_directory> <output_directory> <num_val_samples>
```
- Splitting the data with this script will create two subfolders `<output_directory>/train_data` and `<output_directory>/val_data` which can be used in `hparams.cfg`.
- If your custom dataset isn't square shaped, you can use the standalone utility script we provide:
```
cd data_scripts
python utility_resize.py --in-path=<input_directory> --out-path=<output_directory> --resolution=<my_res> --resize-type=<my_resize_type>
```
- `--resize-type` can be one of `(center_crop, random_crop, reshape)` and defiles how to make your dataset square shaped.
- For `random_crop` resize type, there is an extra `--repeat-n` argument that defines how many images to create from each initial non square shaped sample (by randomly cropping it).
- For more information about the utility resize script, refer to the code or run:
```
python utility_resize.py --help
```
- An example preprocessing for a non-split dataset of desired resolution of `512` with `random crop` and a number of validation samples of `10000`:
```
cd data_scripts
python utility_resize.py --in-path=/raw/data/path --out-path=/resized/data/path --resolution=512 --resize-type=random_crop --repeat-n=4
python random_split.py /resized/data/path /preprocessed/data/path 
```

## Potential TODOs

- [x] Make data loaders Out-Of-Core (OOC) in Pytorch
- [x] Make data loaders Out-Of-Core (OOC) in JAX
- [x] Update pre-trained model checkpoints
- [x] Add support for custom datasets
- [ ] Add Fréchet-Inception Distance (FID) and Inception Score (IS) as measures for sample quality performance.
- [ ] Improve the format of the encoded dataset used in downstream tasks (output of `encoding` mode, if there is a need)
- [ ] Write a `decoding` mode API (if needed).

## Bibtex  
  
If you happen to use this codebase, please cite our paper:

```
@article{hazami2022efficient,
  title={Efficient-VDVAE: Less is more},
  author={Hazami, Louay and Mama, Rayhane and Thurairatnam, Ragavan},
  journal={arXiv preprint arXiv:2203.13751},
  year={2022}
}
```