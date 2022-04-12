import time
import torch
import torch.distributed as dist
import numpy as np
from hparams import HParams
import torch_optimizer as optim

try:
    from .schedules import LogisticBetaSchedule, NoamSchedule, \
        NarrowExponentialDecay, ConstantLearningRate, GammaSchedule, LinearBetaSchedule, NarrowCosineDecay
    from .losses import StructureSimilarityIndexMap, KLDivergence, \
        DiscMixLogistic, BernoulliLoss
    from .ssim import SSIM
    from .adamax import Adamax

    from ..utils.utils import create_checkpoint_manager_and_load_if_exists, effective_pixels
except (ImportError, ValueError):
    from model.schedules import LogisticBetaSchedule, NoamSchedule, \
        NarrowExponentialDecay, ConstantLearningRate, GammaSchedule, LinearBetaSchedule, NarrowCosineDecay
    from model.losses import StructureSimilarityIndexMap, KLDivergence, \
        DiscMixLogistic, BernoulliLoss
    from model.ssim import SSIM
    from utils.utils import create_checkpoint_manager_and_load_if_exists, effective_pixels
    from model.adamax import Adamax

hparams = HParams.get_hparams_by_name("efficient_vdvae")

if hparams.loss.variation_schedule == 'None':
    kldiv_schedule = lambda x: torch.as_tensor(1.)
elif hparams.loss.variation_schedule == 'Logistic':
    kldiv_schedule = LogisticBetaSchedule(
        activation_step=hparams.loss.vae_beta_activation_steps,
        growth_rate=hparams.loss.vae_beta_growth_rate
    )

elif hparams.loss.variation_schedule == 'Linear':
    kldiv_schedule = LinearBetaSchedule(
        anneal_start=hparams.loss.vae_beta_anneal_start,
        anneal_steps=hparams.loss.vae_beta_anneal_steps,
        beta_min=hparams.loss.vae_beta_min,
    )

else:
    raise NotImplementedError(f'KL beta schedule {hparams.loss.variation_schedule} not known!!')

reconstruction_loss = DiscMixLogistic()
reconstruction_loss_bernoulli = BernoulliLoss()
kldiv_loss = KLDivergence()


def get_optimizer(model, type, learning_rate, beta_1, beta_2, epsilon,
                  weight_decay_rate, decay_scheme, warmup_steps, decay_steps, decay_rate, decay_start,
                  min_lr, last_epoch, checkpoint):
    if type == 'Adamax':
        opt = Adamax
        opt_kwargs = dict(
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay_rate)
    elif type == 'RAdam':
        opt = optim.RAdam
        opt_kwargs = dict(
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay_rate)
    elif type == 'Adam':
        opt = torch.optim.Adam
        opt_kwargs = dict(
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
            weight_decay=weight_decay_rate)

    else:
        raise ValueError(f'Optimizer {type} not known!!')

    if checkpoint['optimizer_state_dict'] is not None:
        opt = opt(params=model.parameters(), **opt_kwargs)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded Optimizer Checkpoint')
    else:
        opt = opt(params=model.parameters(), **opt_kwargs)

    if decay_scheme == 'noam':
        schedule = NoamSchedule(optimizer=opt, warmup_steps=warmup_steps, last_epoch=last_epoch)

    elif decay_scheme == 'exponential':
        schedule = NarrowExponentialDecay(optimizer=opt,
                                          decay_steps=decay_steps,
                                          decay_rate=decay_rate,
                                          decay_start=decay_start,
                                          minimum_learning_rate=min_lr,
                                          last_epoch=last_epoch)

    elif decay_scheme == 'cosine':
        schedule = NarrowCosineDecay(optimizer=opt,
                                     decay_steps=decay_steps,
                                     decay_start=decay_start,
                                     minimum_learning_rate=min_lr,
                                     last_epoch=last_epoch,
                                     warmup_steps=warmup_steps)

    elif decay_scheme == 'constant':
        schedule = ConstantLearningRate(optimizer=opt, last_epoch=last_epoch, warmup_steps=warmup_steps)

    else:
        raise NotImplementedError(f'{decay_scheme} is not implemented yet!')

    if checkpoint['scheduler_state_dict'] is not None:
        schedule.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Loaded Scheduler Checkpoint')

    return opt, schedule


gamma_schedule = GammaSchedule(max_steps=hparams.loss.gamma_max_steps)


def compute_loss(targets, predictions, posterior_dist_list, prior_kl_dist_list, step_n, global_batch_size):
    if hparams.data.dataset_source == 'binarized_mnist':
        feature_matching_loss, avg_feature_matching_loss, means, log_scales = reconstruction_loss_bernoulli(
            targets=targets,
            logits=predictions,
            global_batch_size=global_batch_size)
    else:
        feature_matching_loss, avg_feature_matching_loss, means, log_scales = reconstruction_loss(
            targets=targets,
            logits=predictions,
            global_batch_size=global_batch_size)

    global_variational_prior_losses, avg_global_varprior_losses = [], []
    for posterior_dist, prior_kl_dist in zip(posterior_dist_list, prior_kl_dist_list):
        global_variational_prior_loss, avg_global_varprior_loss = kldiv_loss(
            p=posterior_dist,
            q=prior_kl_dist,
            global_batch_size=global_batch_size
        )
        global_variational_prior_losses.append(global_variational_prior_loss)
        avg_global_varprior_losses.append(avg_global_varprior_loss)

    global_variational_prior_losses = torch.stack(global_variational_prior_losses, dim=0)

    if hparams.loss.use_gamma_schedule:
        global_variational_prior_loss = gamma_schedule(global_variational_prior_losses,
                                                       avg_global_varprior_losses,
                                                       step_n=step_n)
    else:
        global_variational_prior_loss = torch.sum(global_variational_prior_losses)

    global_var_loss = kldiv_schedule(step_n) * global_variational_prior_loss  # beta

    total_generator_loss = feature_matching_loss + global_var_loss

    scalar = np.log(2.)

    # True bits/dim kl div
    kl_div = torch.sum(global_variational_prior_losses) / scalar

    return avg_feature_matching_loss, avg_global_varprior_losses, total_generator_loss, means, log_scales, kl_div


def tensorboard_log(model, optimizer, global_step, writer, losses, outputs, targets, means=None, log_scales=None,
                    updates=None,
                    global_norm=None, train_steps_per_epoch=None, mode='train'):
    for key, value in losses.items():
        writer.add_scalar(f"Losses/{key}", value, global_step)

    writer.add_histogram("Distributions/target", targets, global_step, bins=20)
    writer.add_histogram("Distributions/output", torch.clamp(outputs, min=-1., max=1.), global_step, bins=20)

    if means is not None:
        assert log_scales is not None

        writer.add_histogram('OutputLayer/means', means, global_step, bins=30)
        writer.add_histogram('OutputLayer/log_scales', log_scales, global_step, bins=30)

    if mode == 'train':
        for variable in model.parameters():
            writer.add_histogram("Weights/{}".format(variable.name), variable, global_step)

        # Get the learning rate from the optimizer
        writer.add_scalar("Schedules/learning_rate", optimizer.param_groups[0]['lr'],
                          global_step)

        if updates is not None:
            for layer, update in updates.items():
                writer.add_scalar("Updates/{}".format(layer), update, global_step)

            max_updates = torch.max(torch.stack(list(updates.values())))
            assert global_norm is not None
            writer.add_scalar("Mean_Max_Updates/Global_norm", global_norm, global_step)
            writer.add_scalar("Mean_Max_Updates/Max_updates", max_updates, global_step)

    writer.flush()


def _global_norm(model):
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = torch.tensor(0.0)
    else:
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
    return total_norm


def gradient_clip(model):
    if hparams.optimizer.clip_gradient_norm:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    max_norm=hparams.optimizer.gradient_clip_norm_value)
    else:
        total_norm = _global_norm(model)
    return total_norm


def gradient_skip(global_norm):
    if hparams.optimizer.gradient_skip:
        if torch.any(torch.isnan(global_norm)) or global_norm >= hparams.optimizer.gradient_skip_threshold:
            skip = True
            gradient_skip_counter_delta = 1.

        else:
            skip = False
            gradient_skip_counter_delta = 0.

    else:
        skip = False
        gradient_skip_counter_delta = 0.

    return skip, gradient_skip_counter_delta


def plot_image(outputs, targets, step, writer):
    if hparams.data.dataset_source != 'binarized_mnist':
        targets = (targets + 1) / 2
        outputs = (outputs + 1) / 2
    writer.add_image(f"{step}/Original_{step}", targets, step)
    writer.add_image(f"{step}/Generated_{step}", outputs, step)


def _compiled_train_step(model, inputs, step_n):
    predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)
    avg_feature_matching_loss, avg_global_varprior_losses, total_generator_loss, means, \
    log_scales, kl_div = compute_loss(inputs,
                                      predictions,
                                      posterior_dist_list=posterior_dist_list,
                                      prior_kl_dist_list=prior_kl_dist_list,
                                      step_n=step_n,
                                      global_batch_size=hparams.train.batch_size // hparams.run.num_gpus)

    total_generator_loss.backward()

    total_norm = gradient_clip(model)
    skip, gradient_skip_counter_delta = gradient_skip(total_norm)

    outputs = model.module.top_down.sample(predictions)

    return outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, total_norm, \
           gradient_skip_counter_delta, skip


def train_step(model, ema_model, optimizer, inputs, step_n):
    outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, global_norm, \
    gradient_skip_counter_delta, skip = _compiled_train_step(model, inputs, step_n)

    if not skip:
        optimizer.step()
        update_ema(model, ema_model, hparams.train.ema_decay)

    optimizer.zero_grad()
    return outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, \
           global_norm, gradient_skip_counter_delta


def eval_step(model, inputs, step_n):
    with torch.no_grad():
        predictions, posterior_dist_list, prior_kl_dist_list = model(inputs)

        avg_feature_matching_loss, avg_global_varprior_losses, total_generator_loss, means, \
        log_scales, kl_div = compute_loss(inputs,
                                          predictions,
                                          posterior_dist_list=posterior_dist_list,
                                          prior_kl_dist_list=prior_kl_dist_list,
                                          step_n=step_n,
                                          global_batch_size=hparams.val.batch_size // hparams.run.num_gpus)

        outputs = model.module.top_down.sample(predictions)

    return outputs, avg_feature_matching_loss, avg_global_varprior_losses, kl_div, means, log_scales


def reconstruction_step(model, inputs, variates_masks=None, mode='recon'):
    model.eval()
    with torch.no_grad():
        predictions, posterior_dist_list, prior_kl_dist_list = model(inputs, variates_masks)

        if mode == 'recon':
            feature_matching_loss, global_varprior_losses, total_generator_loss, means, \
            log_scales, kl_div = compute_loss(inputs,
                                              predictions,
                                              posterior_dist_list=posterior_dist_list,
                                              prior_kl_dist_list=prior_kl_dist_list,
                                              step_n=max(hparams.loss.vae_beta_anneal_steps,
                                                         hparams.loss.gamma_max_steps) * 10.,
                                              # any number bigger than schedule is fine
                                              global_batch_size=hparams.synthesis.batch_size)

            if hparams.data.dataset_source == 'binarized_mnist':
                # Convert from bpd to nats for comparison
                kl_div = kl_div * np.log(2.) * effective_pixels()

            outputs = model.top_down.sample(predictions)

            return outputs, feature_matching_loss, kl_div
        elif mode == 'encode':
            return posterior_dist_list
        else:
            raise ValueError(f'Unknown Mode {mode}')


def generation_step(model, temperatures):
    y, prior_zs = model.top_down.sample_from_prior(hparams.synthesis.batch_size,
                                                   temperatures=temperatures)

    outputs = model.top_down.sample(y)
    return outputs, prior_zs


def encode_step(model, inputs):
    predictions, attn_weights_post_list, attn_weights_prior_kl_list, _, _, _, _, _ = model(inputs)
    return attn_weights_prior_kl_list


def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def train(model, ema_model, optimizer, schedule, train_dataset, val_dataset, checkpoint_start, tb_writer_train,
          tb_writer_val, checkpoint_path, device, rank):
    ssim_metric = StructureSimilarityIndexMap(image_channels=hparams.data.channels)
    global_step = checkpoint_start
    gradient_skip_counter = 0.

    model.train()

    # let all processes sync up before starting with a new epoch of training
    total_train_epochs = int(np.ceil(hparams.train.total_train_steps / len(train_dataset)))
    val_epoch = 0
    for epoch in range(0, total_train_epochs):
        train_dataset.sampler.set_epoch(epoch)
        if rank == 0:
            print(f'\nEpoch: {epoch + 1}')
        dist.barrier()
        for batch_n, train_inputs in enumerate(train_dataset):
            # update global step
            global_step += 1

            train_inputs = train_inputs.to(device, non_blocking=True)
            # torch.cuda.synchronize()
            start_time = time.time()
            train_outputs, train_feature_matching_loss, train_global_varprior_losses, train_kl_div, \
            global_norm, gradient_skip_counter_delta = train_step(model, ema_model, optimizer, train_inputs,
                                                                  global_step)
            # torch.cuda.synchronize()
            end_time = round((time.time() - start_time), 2)
            schedule.step()

            gradient_skip_counter += gradient_skip_counter_delta
            if hparams.data.dataset_source == 'binarized_mnist':
                # Convert from bpd to nats for comparison
                train_kl_div = train_kl_div * np.log(2.) * effective_pixels()

            train_var_loss = np.sum([v.detach().cpu() for v in train_global_varprior_losses])
            train_nelbo = train_kl_div + train_feature_matching_loss
            # global_norm = global_norm / (hparams.data.target_res * hparams.data.target_res * hparams.data.channels)
            if rank == 0:
                print(global_step,
                      ('Time/Step (sec)', end_time),
                      ('Reconstruction Loss', round(train_feature_matching_loss.detach().cpu().item(), 3)),
                      ('KL loss', round(train_kl_div.detach().cpu().item(), 3)),
                      ('nelbo', round(train_nelbo.detach().cpu().item(), 4)),
                      ('average KL loss', round(train_var_loss.item(), 3)),
                      ('Beta', round(kldiv_schedule(global_step).detach().cpu().item(), 4)),
                      ('N° active groups', np.sum([v.detach().cpu() >= hparams.metrics.latent_active_threshold
                                                   for v in train_global_varprior_losses])),
                      ('GradNorm', round(global_norm.detach().cpu().item(), 1)),
                      ('GradSkipCount', gradient_skip_counter),
                      # ('learning_rate', optimizer.param_groups[0]['lr']),
                      end="\r")

            if global_step % hparams.train.checkpoint_and_eval_interval_in_steps == 0 or global_step == 0:
                model.eval()
                # Compute SSIM at the end of the global_step
                train_ssim = ssim_metric(train_inputs, train_outputs,
                                         global_batch_size=hparams.train.batch_size // hparams.run.num_gpus)
                if rank == 0:
                    train_losses = {'reconstruction_loss': train_feature_matching_loss,
                                    'kl_div': train_kl_div,
                                    'average_kl_div': train_var_loss,
                                    'variational_beta': kldiv_schedule(global_step),
                                    'ssim': train_ssim,
                                    'nelbo': train_nelbo}

                    train_losses.update({f'latent_kl_{i}': v for i, v in enumerate(train_global_varprior_losses)})

                    print(
                        f'\nTrain Stats for global_step {global_step} | NELBO {train_nelbo:.6f} | '
                        f'SSIM: {train_ssim:.6f}')

                # Evaluate model
                val_feature_matching_losses = 0
                val_global_varprior_losses = None
                val_ssim = 0
                val_kl_divs = 0
                val_epoch += 1

                val_dataset.sampler.set_epoch(val_epoch)
                for val_step, val_inputs in enumerate(val_dataset):
                    # Val inputs contains val_Data and filenames
                    val_inputs = val_inputs.to(device, non_blocking=True)
                    val_outputs, val_feature_matching_loss, \
                    val_global_varprior_loss, val_kl_div, val_means, val_log_scales = eval_step(model,
                                                                                                inputs=val_inputs,
                                                                                                step_n=global_step)

                    val_ssim_per_batch = ssim_metric(val_inputs, val_outputs,
                                                     global_batch_size=hparams.val.batch_size // hparams.run.num_gpus)

                    val_feature_matching_losses += val_feature_matching_loss
                    val_ssim += val_ssim_per_batch
                    val_kl_divs += val_kl_div

                    if val_global_varprior_losses is None:
                        val_global_varprior_losses = val_global_varprior_loss
                    else:
                        val_global_varprior_losses = [u + v for u, v in
                                                      zip(val_global_varprior_losses, val_global_varprior_loss)]

                val_feature_matching_loss = val_feature_matching_losses / (val_step + 1)
                val_ssim = val_ssim / (val_step + 1)
                val_kl_div = val_kl_divs / (val_step + 1)

                if hparams.data.dataset_source == 'binarized_mnist':
                    # Convert from bpd to nats for comparison
                    val_kl_div = val_kl_div * np.log(2.) * effective_pixels()

                val_global_varprior_losses = [v / (val_step + 1) for v in val_global_varprior_losses]

                val_varprior_loss = np.sum([v.detach().cpu() for v in val_global_varprior_losses])
                val_nelbo = val_kl_div + val_feature_matching_loss

                if rank == 0:
                    val_losses = {'reconstruction_loss': val_feature_matching_loss,
                                  'kl_div': val_kl_div,
                                  'average_kl_div': val_varprior_loss,
                                  'ssim': val_ssim,
                                  'nelbo': val_nelbo}

                    val_losses.update({f'latent_kl_{i}': v for i, v in enumerate(val_global_varprior_losses)})

                    print(
                        f'Validation Stats for global_step {global_step} |'
                        f' Reconstruction Loss {val_feature_matching_loss:.4f} |'
                        f' KL Div {val_kl_div:.4f} |'f'NELBO {val_nelbo:.6f} |'
                        f'SSIM: {val_ssim:.6f}')

                    # Save checkpoint (only if better than best)
                    print(f'Saving checkpoint for global_step {global_step}..')

                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.module.state_dict(),
                        'ema_model_state_dict': ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': schedule.state_dict()
                    }, checkpoint_path)

                    # Tensorboard logging
                    print('Logging to Tensorboard..')
                    train_losses['skips_count'] = gradient_skip_counter / hparams.train.total_train_steps
                    tensorboard_log(model, optimizer, global_step, tb_writer_train, train_losses, train_outputs,
                                    train_inputs,
                                    global_norm=global_norm)
                    tensorboard_log(model, optimizer, global_step, tb_writer_val, val_losses, val_outputs, val_inputs,
                                    means=val_means, log_scales=val_log_scales, mode='val')

                    # Save artifacts
                    plot_image(train_outputs[0], train_inputs[0], global_step, writer=tb_writer_train)
                    plot_image(val_outputs[0], val_inputs[0], global_step, writer=tb_writer_val)
                model.train()
            dist.barrier()

            if global_step >= hparams.train.total_train_steps:
                print(f'Finished training after {global_step} steps!')
                exit()
