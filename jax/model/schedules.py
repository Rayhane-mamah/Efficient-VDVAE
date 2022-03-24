import jax
import jax.numpy as jnp
import jax.lax as lax
from hparams import HParams

hparams = HParams.get_hparams_by_name("efficient_vdvae")


def get_lr_schedule(decay_scheme, init_lr, warmup_steps, decay_steps, decay_rate, decay_start, min_lr):
    if decay_scheme == 'noam':
        return NoamSchedule(init_lr, warmup_steps=warmup_steps)

    elif decay_scheme == 'exponential':
        return NarrowExponentialDecay(initial_learning_rate=init_lr,
                                      decay_steps=decay_steps,
                                      decay_rate=decay_rate,
                                      decay_start=decay_start,
                                      minimum_learning_rate=min_lr)

    elif decay_scheme == 'cosine':
        return NarrowCosineDecay(initial_learning_rate=init_lr,
                                 decay_steps=decay_steps,
                                 decay_start=decay_start,
                                 minimum_learning_rate=min_lr,
                                 warmup_steps=warmup_steps)

    elif decay_scheme == 'constant':
        return ConstantLearningRate(init_lr, warmup_steps=warmup_steps)

    else:
        raise NotImplementedError(f'{decay_scheme} is not implemented yet!')


class LogisticBetaSchedule:
    def __init__(self, activation_step, growth_rate):
        self.beta_max = 1.
        self.activation_step = activation_step
        self.growth_rate = growth_rate

    def __call__(self, step):
        return self.beta_max / (1. + jnp.exp(-self.growth_rate * (step - self.activation_step)))


class LinearBetaSchedule:
    def __init__(self, anneal_start, anneal_steps, beta_min):
        self.beta_max = 1.
        self.anneal_start = anneal_start
        self.anneal_steps = anneal_steps
        self.beta_min = beta_min

    def __call__(self, step):
        return jnp.clip((step - self.anneal_start) / (self.anneal_start + self.anneal_steps), a_min=self.beta_min, a_max=self.beta_max)


class GammaSchedule:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.num_groups = sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

    def __call__(self, kl_losses, resolutions, step_n, epsilon=0.):
        assert kl_losses.shape == resolutions.shape == (self.num_groups,)

        def schedule(kls):
            if hparams.loss.scaled_gamma:
                alpha_hat = kls / resolutions + epsilon
            else:
                alpha_hat = kls + epsilon
            alpha = self.num_groups * alpha_hat / jnp.sum(alpha_hat)

            return jnp.tensordot(lax.stop_gradient(alpha), kls, axes=1)

        def do_nothing(kls):
            return jnp.sum(kls)

        kl_loss = jax.lax.cond(step_n <= self.max_steps, schedule, do_nothing, operand=kl_losses)

        return kl_loss


class ConstantLearningRate:
    def __init__(self, init_lr, warmup_steps):
        super(ConstantLearningRate, self).__init__()

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        return self.init_lr * jnp.minimum(1., step / self.warmup_steps)


class NarrowExponentialDecay:
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, decay_start=0, minimum_learning_rate=None):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.decay_start = decay_start
        self.initial_learning_rate = initial_learning_rate
        self.minimum_learning_rate = minimum_learning_rate

    def __call__(self, step):
        return jnp.clip(self.initial_learning_rate * self.decay_rate ^ (step - self.decay_start / self.decay_steps),
                        a_min=self.minimum_learning_rate, a_max=self.initial_learning_rate)


class NarrowCosineDecay:
    def __init__(self, initial_learning_rate, decay_steps, decay_start=0, minimum_learning_rate=None, warmup_steps=4000):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = minimum_learning_rate / initial_learning_rate
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps

        assert self.warmup_steps <= self.decay_start

    def __call__(self, step):
        def decay(step_):
            step_ = step_ - self.decay_start
            step_ = jnp.minimum(step_, self.decay_steps)
            cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * step_ / self.decay_steps))
            decayed = (1. - self.alpha) * cosine_decay + self.alpha
            lr = self.initial_learning_rate * decayed
            return lr

        def do_nothing(step_):
            return self.initial_learning_rate * jnp.minimum(1., step / self.warmup_steps)

        lr = jax.lax.cond(step < self.decay_start, do_nothing, decay, operand=step)

        return lr


class NoamSchedule:
    def __init__(self, init_lr, warmup_steps=4000):
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = jnp.float32(step)
        arg1 = lax.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.init_lr * self.warmup_steps ** 0.5 * jnp.minimum(arg1, arg2)
