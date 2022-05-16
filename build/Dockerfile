FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN pip install --upgrade pip

#install hparams
RUN pip install --upgrade git+https://github.com/Rayhane-mamah/hparams

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
#pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
RUN pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install JAX extensions
RUN pip install --upgrade optax flax trax

ENV NUMBA_CACHE_DIR /tmp/