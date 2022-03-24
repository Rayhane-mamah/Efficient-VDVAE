from typing import Any, Callable, Optional, Union
from optax._src.base import Params, identity
from optax._src import combine
from optax._src.alias import _scale_by_learning_rate, ScalarOrSchedule
from optax._src import transform

from hparams import HParams

try:
    from .adamax import scale_by_adamax
    from .schedules import NarrowCosineDecay

except (ImportError, ValueError):
    from model.adamax import scale_by_adamax
    from model.schedules import NarrowCosineDecay

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def get_optimizer(type, learning_rate, beta_1, beta_2, epsilon, use_weight_decay, l2_weight, l2_mask):
    if type == 'Radam':
        opt = Radam(
            learning_rate=learning_rate,
            b1=beta_1,
            b2=beta_2,
            eps=epsilon,
            use_weight_decay=use_weight_decay,
            l2_weight=l2_weight,
            l2_mask=l2_mask
        ).make()

    elif type == 'Adam':
        opt = Adam(
            learning_rate=learning_rate,
            b1=beta_1,
            b2=beta_2,
            eps=epsilon,
            use_weight_decay=use_weight_decay,
            l2_weight=l2_weight,
            l2_mask=l2_mask
        ).make()

    elif type == 'Adamax':
        opt = Adamax(
            learning_rate=learning_rate,
            b1=beta_1,
            b2=beta_2,
            eps=epsilon,
            use_weight_decay=use_weight_decay,
            l2_weight=l2_weight,
            l2_mask=l2_mask
        ).make()

    else:
        raise ValueError(f'Optimizer {type} not known!!')

    return opt


class BaseWeightDecayOptimizer:
    def __init__(self, learning_rate, use_weight_decay, l2_weight, l2_mask):
        self.learning_rate = learning_rate
        self.use_weight_decay = use_weight_decay
        self.l2_weight = l2_weight
        self.l2_mask = l2_mask

    def _add_weight_decay_and_lr_transformations(self, transforms):
        if self.use_weight_decay:
            assert self.l2_weight != 0.
            transforms.append(transform.add_decayed_weights(weight_decay=self.l2_weight, mask=self.l2_mask))

        transforms.append(_scale_by_learning_rate(self.learning_rate))
        return transforms

    def create_transforms(self):
        """Method that returns a list of optax transformations to apply during optimization"""
        raise NotImplementedError('Do not use BaseWeightDecayOptimizer, create your own optimizer on top!')

    def make(self):
        return combine.chain(*self.create_transforms())


class Adam(BaseWeightDecayOptimizer):
    def __init__(self, learning_rate: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 eps_root: float = 0.,
                 use_weight_decay: bool = False,
                 l2_weight: float = 0.,
                 l2_mask: Optional[Union[Any, Callable[[Params], Any]]] = None):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.eps_root = eps_root
        super(Adam, self).__init__(learning_rate=learning_rate, use_weight_decay=use_weight_decay, l2_weight=l2_weight,
                                   l2_mask=l2_mask)

    def create_transforms(self):
        transforms = [transform.scale_by_adam(b1=self.b1, b2=self.b2, eps=self.eps, eps_root=self.eps_root)]
        return self._add_weight_decay_and_lr_transformations(transforms)


class Adamax(BaseWeightDecayOptimizer):
    def __init__(self, learning_rate: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 use_weight_decay: bool = False,
                 l2_weight: float = 0.,
                 l2_mask: Optional[Union[Any, Callable[[Params], Any]]] = None):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        super(Adamax, self).__init__(learning_rate=learning_rate, use_weight_decay=use_weight_decay, l2_weight=l2_weight,
                                     l2_mask=l2_mask)

    def create_transforms(self):
        transforms = [scale_by_adamax(b1=self.b1, b2=self.b2, eps=self.eps)]
        return self._add_weight_decay_and_lr_transformations(transforms)


class Radam(BaseWeightDecayOptimizer):
    def __init__(self, learning_rate: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 eps_root: float = 0.,
                 threshold: float = 5.0,
                 use_weight_decay: bool = False,
                 l2_weight: float = 0.,
                 l2_mask: Optional[Union[Any, Callable[[Params], Any]]] = None):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.eps_root = eps_root
        self.threshold = threshold
        super(Radam, self).__init__(learning_rate=learning_rate, use_weight_decay=use_weight_decay, l2_weight=l2_weight,
                                    l2_mask=l2_mask)

    def create_transforms(self):
        transforms = [transform.scale_by_radam(b1=self.b1, b2=self.b2, eps=self.eps, eps_root=self.eps_root, threshold=self.threshold)]
        return self._add_weight_decay_and_lr_transformations(transforms)


class SGD(BaseWeightDecayOptimizer):
    def __init__(self, learning_rate: ScalarOrSchedule,
                 momentum: Optional[float] = None,
                 nesterov: bool = False,
                 accumulator_dtype: Optional[Any] = None,
                 use_weight_decay: bool = False,
                 l2_weight: float = 0.,
                 l2_mask: Optional[Union[Any, Callable[[Params], Any]]] = None):
        self.momentum = momentum
        self.nesterov = nesterov
        self.accumulator_dtype = accumulator_dtype
        super(SGD, self).__init__(learning_rate=learning_rate, use_weight_decay=use_weight_decay, l2_weight=l2_weight,
                                  l2_mask=l2_mask)

    def create_transforms(self):
        transforms = [transform.trace(decay=self.momentum, nesterov=self.nesterov, accumulator_dtype=self.accumulator_dtype)
                      if self.momentum is not None else identity()]
        return self._add_weight_decay_and_lr_transformations(transforms)
