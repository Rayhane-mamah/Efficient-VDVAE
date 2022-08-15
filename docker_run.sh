#!/bin/sh

HOST_DATA=`readlink -e ./`;
CONTAINER_DATA=/workspace/Efficient-VDVAE;
nvidia-docker run -it --shm-size=128g -v $HOST_DATA:$CONTAINER_DATA efficient_vdvae_image