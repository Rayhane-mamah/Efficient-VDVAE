FROM nvcr.io/nvidia/pytorch:22.02-py3

RUN pip install --upgrade pip

#install hparams
RUN pip install --upgrade git+https://github.com/Rayhane-mamah/hparams

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
#
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install --upgrade optax flax trax

ENV NUMBA_CACHE_DIR /tmp/
