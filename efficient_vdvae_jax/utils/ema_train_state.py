import jax
import optax
from flax.training.train_state import TrainState

from typing import Any
from flax import core


class EMATrainState(TrainState):
    ema_decay: float
    ema_params: core.FrozenDict[str, Any]

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = jax.tree_map(lambda ema, p: ema * self.ema_decay + (1 - self.ema_decay) * p,
                                      self.ema_params, new_params)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, ema_params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            ema_decay=ema_decay,
            **kwargs,
        )
