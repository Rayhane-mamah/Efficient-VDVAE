#!/usr/bin/env bash

HOST_DATA=./;
CONTAINER_DATA=/workspace;
nvidia-docker run -it --shm-size=128g -v $HOST_DATA:$CONTAINER_DATA efficient_vdvae_image