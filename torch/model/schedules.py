import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
import warnings
from hparams import HParams

hparams = HParams.get_hparams_by_name("global_local_memcodes")


class LogisticBetaSchedule:
    def __init__(self, activation_step, growth_rate):
        self.beta_max = 1.
        self.activation_step = activation_step
        self.growth_rate = growth_rate

    def __call__(self, step):
        return self.beta_max / (1. + torch.exp(-self.growth_rate * (step - self.activation_step)))


class LinearBetaSchedule:
    def __init__(self,  anneal_start, anneal_steps, beta_min):
        self.beta_max = 1.
        self.anneal_start = anneal_start
        self.anneal_steps = anneal_steps
        self.beta_min = beta_min

    def __call__(self, step):
        return torch.clamp(torch.tensor((step - self.anneal_start) / (self.anneal_start + self.anneal_steps)),
                           min=self.beta_min, max=self.beta_max)


class GammaSchedule:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.num_groups = sum(hparams.model.down_n_blocks_per_res) + len(hparams.model.down_strides)

    def __call__(self, kl_losses, avg_kl_losses, step_n, epsilon=0.):
        avg_kl_losses = torch.stack(avg_kl_losses, dim=0) * np.log(2)  # [n]
        assert kl_losses.size() == avg_kl_losses.size() == (self.num_groups,)

        if step_n <= self.max_steps:
            if hparams.loss.scaled_gamma:
                alpha_hat = (avg_kl_losses + epsilon)
            else:
                alpha_hat = kl_losses + epsilon
            alpha = self.num_groups * alpha_hat / torch.sum(alpha_hat)

            kl_loss = torch.tensordot(alpha.detach(), kl_losses, dims=1)

        else:
            kl_loss = torch.sum(kl_losses)

        return kl_loss


class ConstantLearningRate(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1, verbose=False):
        if warmup_steps != 0:
            self.warmup_steps = warmup_steps
        else:
            self.warmup_steps = 1
        super(ConstantLearningRate, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)
        # self.last_epoch = last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]

    def _get_closed_form_lr(self):
        return [v * (torch.minimum(torch.tensor(1.), torch.tensor(self.last_epoch / self.warmup_steps))) for v in
                self.base_lrs]


class NarrowExponentialDecay(_LRScheduler):
    def __init__(self, optimizer, decay_steps, decay_rate, decay_start,
                 minimum_learning_rate, last_epoch=-1, verbose=False):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.decay_start = decay_start
        self.minimum_learning_rate = minimum_learning_rate

        super(NarrowExponentialDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ^ (self.last_epoch - self.decay_start / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs

    def _get_closed_form_lr(self):
        lrs = [torch.clamp(base_lr * self.decay_rate ^ (self.last_epoch - self.decay_start / self.decay_steps),
                           min=self.minimum_learning_rate, max=base_lr) for base_lr in self.base_lrs]
        return lrs


class NarrowCosineDecay(CosineAnnealingLR):
    def __init__(self, optimizer, decay_steps, warmup_steps, decay_start=0, minimum_learning_rate=None, last_epoch=-1,
                 verbose=False):
        self.decay_steps = decay_steps
        self.decay_start = decay_start
        self.minimum_learning_rate = minimum_learning_rate
        self.warmup_steps = warmup_steps

        assert self.warmup_steps <= self.decay_start

        super(NarrowCosineDecay, self).__init__(optimizer=optimizer, last_epoch=last_epoch, T_max=decay_steps,
                                                eta_min=self.minimum_learning_rate)

    def get_lr(self):
        if self.last_epoch < self.decay_start:

            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(NarrowCosineDecay, self).get_lr()

    def _get_closed_form_lr(self):
        if self.last_epoch < self.decay_start:
            return [v * (torch.minimum(torch.tensor(1.), self.last_epoch / self.warmup_steps)) for v in self.base_lrs]
        else:
            return super(NarrowCosineDecay, self)._get_closed_form_lr()


class NoamSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps=4000, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        super(NoamSchedule, self).__init__(optimizer=optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [base_lr * self.warmup_steps ** 0.5 * torch.minimum(arg1, arg2) for base_lr in self.base_lrs]
