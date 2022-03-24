import jax.numpy as jnp
from jax import dtypes
import jax.random as random
from flax import linen as nn
from flax.linen.initializers import glorot_uniform, zeros
from typing import Iterable, Union, Tuple, Callable, Any

from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any

default_kernel_init = glorot_uniform()


def stable_init(scale, dtype=jnp.float_):
    glorot_init = glorot_uniform()

    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        initial_values = glorot_init(key, shape, dtype=dtype)
        return initial_values * scale

    return init


def uniform_init(minval=0., maxval=1., dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.uniform(key, shape, dtype, minval=minval, maxval=maxval)

    return init


class Conv2D(nn.Module):
    """A Conv2D special case of the general jax Conv class."""
    filters: int
    kernel_size: Union[int, Iterable[int]]
    strides: Union[None, int, Iterable[int]] = 1
    padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
    input_dilation: Union[None, int, Iterable[int]] = 1
    kernel_dilation: Union[None, int, Iterable[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        return nn.Conv(
            features=self.filters,
            kernel_size=(self.kernel_size, self.kernel_size) if isinstance(self.kernel_size, int) else self.kernel_size,
            strides=(self.strides, self.strides) if isinstance(self.strides, int) else self.strides,
            padding=self.padding,
            input_dilation=(self.input_dilation, self.input_dilation) if isinstance(self.input_dilation, int) else self.input_dilation,
            kernel_dilation=(self.kernel_dilation, self.kernel_dilation) if isinstance(self.kernel_dilation, int) else self.kernel_dilation,
            feature_group_count=self.feature_group_count,
            use_bias=self.use_bias,
            dtype=self.dtype,
            precision=self.precision,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name='conv'
        )(inputs)
