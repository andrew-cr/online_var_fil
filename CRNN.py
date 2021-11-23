# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Independent, Normal, MultivariateNormal, StudentT
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import core.amortised_models as amortised_models
import core.nonamortised_models as nonamortised_models
import core.networks as networks
import core.utils as utils
import math
import subprocess
import hydra
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import time
from pathlib import Path


def save_np(name, x):
    np.save(name, x)


@hydra.main(config_path='conf', config_name="CRNN")
def main(cfg):
    utils.save_git_hash(hydra.utils.get_original_cwd())
    device = cfg.device
    writer = SummaryWriter(os.getcwd())
    writer.add_text("config", Path('.hydra/config.yaml').read_text())

    seed = np.random.randint(0, 9999999) if cfg.seed is None else cfg.seed
    print("seed", seed)
    with open('seed.txt', 'w') as f:
        f.write(str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_np, y_np, xdim, ydim, F_fn, G_fn, p_0_dist = generate_data(cfg)
    x = torch.from_numpy(x_np).float().to(device)
    y = torch.from_numpy(y_np).float().to(device)

    saved_models_folder_name = 'saved_models'
    if cfg.model_training.save_models:
        os.mkdir(saved_models_folder_name)

    if cfg.model_training.func_type == 'Bootstrap_PF':
        model = nonamortised_models.BootstrapParticleFilter(
            device, xdim, ydim,
            F_fn, G_fn, p_0_dist,
            cfg.model_training.num_particles
        )
    elif cfg.model_training.func_type == 'EnKF':
        model = nonamortised_models.EnsembleKalmanFilter(
            device, xdim, ydim,
            F_fn, G_fn, p_0_dist,
            cfg.model_training.ensemble_size
        )
    elif cfg.model_training.func_type in ['Vx_t', 'JELBO', 'Ignore_Past', 'VJF', 'Offline']:
        q_hidden_dims = cfg.model_training.q_hidden_dims
        phi_t_init_method = cfg.model_training.phi_t_init_method

        def cond_q_mean_net_constructor():
            net_dims = [xdim] + list(q_hidden_dims) + [xdim]
            return networks.MLP(net_dims, nn.ReLU).to(device)

        print("cond_q_mean_net: ", cond_q_mean_net_constructor())
        if cfg.model_training.func_type == 'Vx_t':
            sigma = cfg.model_training.KRR_sigma
            lam = cfg.model_training.KRR_lambda
            train_sigma = cfg.model_training.KRR_train_sigma
            train_lam = cfg.model_training.KRR_train_lam
            assert cfg.model_training.approx_updates_start_t == 0

            def KRR_constructor():
                return nonamortised_models.KernelRidgeRegressor(nonamortised_models.MaternKernel(
                    sigma=sigma, lam=lam, train_sigma=train_sigma, train_lam=train_lam),
                    centre_elbo=cfg.model_training.KRR_centre_elbo).to(device)

            model = nonamortised_models.Vx_t_phi_t_Model(
                device, xdim, ydim,
                torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                F_fn, G_fn, p_0_dist,
                phi_t_init_method, cfg.model_training.window_size,
                KRR_constructor, cfg.model_training.KRR_init_sigma_median, cfg.model_training.approx_decay,
                cfg.model_training.approx_with_filter,
                num_params_to_store=9999999
            )

        elif cfg.model_training.func_type == 'JELBO':
            model = nonamortised_models.JELBO_Model(device, xdim, ydim,
                                                    torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                                                    cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                                                    F_fn, G_fn, p_0_dist,
                                                    phi_t_init_method, cfg.model_training.window_size,
                                                    num_params_to_store=9999999)

        elif cfg.model_training.func_type == 'Offline':
            model = nonamortised_models.JELBO_Model(device, xdim, ydim,
                                                    torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                                                    cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                                                    F_fn, G_fn, p_0_dist,
                                                    phi_t_init_method, cfg.data.num_data,
                                                    num_params_to_store=9999999)

        elif cfg.model_training.func_type == 'Ignore_Past':
            model = nonamortised_models.Ignore_Past_phi_t_Model(device, xdim, ydim,
                                                                torch.randn(xdim, device=device),
                                                                torch.zeros(xdim, device=device),
                                                                cond_q_mean_net_constructor,
                                                                torch.zeros(xdim, device=device),
                                                                F_fn, G_fn, p_0_dist,
                                                                phi_t_init_method, cfg.model_training.window_size,
                                                                num_params_to_store=9999999)

        elif cfg.model_training.func_type == 'VJF':
            model = nonamortised_models.VJF_Model(device, xdim, ydim,
                                                  torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                                                  cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                                                  F_fn, G_fn, p_0_dist,
                                                  phi_t_init_method, cfg.model_training.window_size,
                                                  num_params_to_store=9999999)
    elif cfg.model_training.func_type in ['kernel_amortised', 'net_amortised']:
        q_rnn_hidden_dim = cfg.model_training.q_rnn_hidden_dim
        q_rnn_num_layers = cfg.model_training.q_rnn_num_layers
        if cfg.model_training.q_rnn_type == "RNN":
            q_rnn = nn.RNN(ydim, q_rnn_hidden_dim, q_rnn_num_layers).to(device)
        elif cfg.model_training.q_rnn_type == "LSTM":
            q_rnn = nn.LSTM(ydim, q_rnn_hidden_dim, q_rnn_num_layers).to(device)
        elif cfg.model_training.q_rnn_type == "MLP":
            q_rnn = networks.MLPRNN(
                [ydim + 2 * xdim] + [q_rnn_hidden_dim] * q_rnn_num_layers + [2 * xdim], nn.ReLU
            ).to(device)

        q_hidden_dims = cfg.model_training.q_hidden_dims  # Hidden dims of q_t_net and cond_q_t_net

        if cfg.model_training.q_net_type == "MLP":
            q_t_net = networks.NormalNet(
                networks.MLP(
                    [q_rnn.hidden_size] + list(q_hidden_dims) + [2 * xdim],
                    nn.ReLU, device
                )
            ).to(device)
            cond_q_t_net = networks.NormalNet(
                networks.MLP(
                    [xdim + q_rnn.hidden_size] + list(q_hidden_dims) + [2 * xdim],
                    nn.ReLU, device
                )
            ).to(device)
        elif cfg.model_training.q_net_type == "Identity":
            q_t_net = networks.NormalNet(nn.Identity()).to(device)
            cond_q_t_net = networks.NormalNet(
                networks.MLP(
                    [xdim + q_rnn.hidden_size] + list(q_hidden_dims) + [2 * xdim], nn.ReLU
                )
            ).to(device)
        else:
            raise NotImplementedError

        if cfg.model_training.func_type == 'kernel_amortised':
            sigma = cfg.model_training.KRR_sigma
            lam = cfg.model_training.KRR_lambda
            train_sigma = cfg.model_training.KRR_train_sigma
            train_lam = cfg.model_training.KRR_train_lam

            def KRR_constructor():
                return nonamortised_models.KernelRidgeRegressor(nonamortised_models.MaternKernel(
                    sigma=sigma, lam=lam, train_sigma=train_sigma, train_lam=train_lam),
                    centre_elbo=cfg.model_training.KRR_centre_elbo).to(device)

            model = amortised_models.Kernel_Amortised_Model(device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net,
                                                            F_fn, G_fn, p_0_dist,
                                                            cfg.model_training.window_size,
                                                            cfg.model_training.rnn_window_size,
                                                            cfg.model_training.rnn_h_lambda,
                                                            KRR_constructor,
                                                            cfg.model_training.funcs_to_approx,
                                                            cfg.model_training.approx_updates_start_t,
                                                            cfg.model_training.approx_decay,
                                                            cfg.model_training.approx_with_filter)

        elif cfg.model_training.func_type == 'net_amortised':
            model = amortised_models.AmortizedModelBase(device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net,
                                                        F_fn, G_fn, p_0_dist,
                                                        cfg.model_training.window_size,
                                                        cfg.model_training.rnn_window_size,
                                                        cfg.model_training.rnn_h_lambda)

            theta_dim = len(replace_none(model.get_theta_params(flatten=True)))
            phi_dim = len(replace_none(model.get_phi_params(flatten=True)))

            net_output_dim = theta_dim + phi_dim + xdim + 1

            def net_constructor_split():
                net = net_constructor([xdim] + list(cfg.model_training.net_hidden_dims) + [net_output_dim])
                return nonamortised_models.NN_Func_Split(net, xdim, net_output_dim,
                                                         cfg.model_training.net_norm_decay).to(device)

            model = amortised_models.Net_Amortised_Model(device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net,
                                                         F_fn, G_fn, p_0_dist,
                                                         cfg.model_training.window_size,
                                                         cfg.model_training.rnn_window_size,
                                                         cfg.model_training.rnn_h_lambda,
                                                         net_constructor_split,
                                                         cfg.model_training.funcs_to_approx,
                                                         cfg.model_training.approx_updates_start_t,
                                                         cfg.model_training.approx_decay,
                                                         cfg.model_training.approx_with_filter)

            print("Approx net dim: ", sum(p.numel() for p in net_constructor_split().parameters()))


        theta_dim = len(utils.replace_none(model.get_theta_params(flatten=True)))
        phi_dim = len(utils.replace_none(model.get_phi_params(flatten=True)))
        rnn_phi_dim = len(utils.replace_none(model.get_rnn_phi_params(flatten=True)))
        q_t_net_phi_dim = len(utils.replace_none(model.get_q_t_net_phi_params(flatten=True)))

        print("Theta dim: ", theta_dim)
        print("Phi dim: ", phi_dim)
        print("Phi dim (RNN): ", rnn_phi_dim)
        print("Phi dim (q_t_net): ", q_t_net_phi_dim)

    else:
        raise ValueError('Unknown func_type type')

    filter_means = []
    filter_covs = []
    x_Tm1_means = []
    x_Tm1_covs = []

    filter_means_training = []
    filter_covs_training = []
    x_Tm1_means_training = []
    x_Tm1_covs_training = []


    true_elbos = []
    elbo_estimates = []
    times = []
    model_training_times = []
    approx_func_training_times = []

    losses = []
    approx_func_losses = []
    filter_RMSEs = []
    x_Tm1_RMSEs = []

    pbar = tqdm(range(0, cfg.data.num_data))

    if model.amortised:
        phi_optim = torch.optim.Adam(model.get_phi_params(), lr=cfg.model_training.phi_lr)
        phi_decay = utils.make_lr_scheduler(phi_optim, cfg.model_training.phi_lr_decay_type,
                                            getattr(cfg.model_training, "phi_lr_num_steps_oom_drop", None))

    for T in pbar:
        start_time = time.time()
        if cfg.model_training.func_type in ['Bootstrap_PF', 'EnKF']:
            model.advance_timestep(y[T, :])
            model.update(y[T, :])

            filter_stats = model.return_summary_stats()
            filter_mean = filter_stats[0].detach().cpu().numpy()
            filter_cov = filter_stats[1].detach().cpu().numpy()

            end_time = time.time()
            model_training_end_time = end_time

            if T > 0:
                x_Tm1_stats = model.return_summary_stats(t=T - 1)
                x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                x_Tm1_cov = x_Tm1_stats[1].detach().cpu().numpy()

        else:
            losses_T = []
            approx_func_losses_T = []
            filter_RMSEs_T = []
            x_Tm1_RMSEs_T = []

            sigma_T = []
            lambda_T = []

            inner_filter_means_training = []
            inner_filter_covs_training = []
            if T > 0:
                inner_x_Tm1_means_training = []
                inner_x_Tm1_covs_training = []

            model.advance_timestep(y[T, :])
            if not model.amortised:
                phi_t_optim = torch.optim.Adam(model.get_phi_T_params(), lr=cfg.model_training.phi_lr)
                phi_t_decay = utils.make_lr_scheduler(phi_t_optim, cfg.model_training.phi_lr_decay_type,
                                                      getattr(cfg.model_training, "phi_lr_num_steps_oom_drop", None))

            approx_func_loss_dict = {}
            for k in range(cfg.model_training.phi_iters):
                if not model.amortised:
                    phi_t_optim.zero_grad()
                    loss = model.populate_phi_grads(y, cfg.model_training.phi_minibatch_size)
                    phi_t_optim.step()
                    phi_t_decay.step()

                    filter_stats = model.return_summary_stats(y)
                    filter_mean = filter_stats[0].detach().cpu().numpy()
                    filter_cov = filter_stats[1].detach().cpu().numpy()
                    inner_filter_means_training.append(filter_mean)
                    inner_filter_covs_training.append(np.diag(filter_cov))
                    if T > 0:
                        x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=1000)
                        x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                        x_Tm1_cov = x_Tm1_stats[1].detach().cpu().numpy()
                        inner_x_Tm1_means_training.append(x_Tm1_mean)
                        inner_x_Tm1_covs_training.append(np.diag(x_Tm1_cov))
                    

                else:
                    phi_optim.zero_grad()
                    loss = model.populate_grads(y, cfg.model_training.phi_minibatch_size)
                    if cfg.model_training.phi_grad_clip_norm is not None:
                        phi_grad_norm = nn.utils.clip_grad_norm_(model.get_phi_params(), cfg.model_training.phi_grad_clip_norm)
                        writer.add_scalar("model_training/phi_grad_norm", phi_grad_norm, T*cfg.model_training.phi_iters+k)
                    phi_optim.step()
                    phi_decay.step()

                    writer.add_scalar("model_training/lr", phi_decay.get_last_lr()[0], T*cfg.model_training.phi_iters+k)

                pbar.set_postfix({"loss": loss.item(), **approx_func_loss_dict})
                losses_T.append(loss.item())

                if cfg.model_training.plot_diagnostics:
                    filter_stats = model.return_summary_stats(y)
                    filter_mean = filter_stats[0].detach().cpu().numpy()
                    filter_RMSEs_T.append(np.sqrt(np.mean((filter_mean - x_np[T]) ** 2)))

                    if T > 0:
                        x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=1000)
                        x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                        x_Tm1_RMSEs_T.append(np.sqrt(np.mean((x_Tm1_mean - x_np[T - 1]) ** 2)))

            # Offline models: train for some additional iterations with all data
            if T == cfg.data.num_data - 1:
                if cfg.model_training.func_type == 'Offline':
                    full_batch_means_training = []
                    full_batch_covs_training = []
                    phi_t_optim = torch.optim.Adam(model.get_phi_T_params(), lr=cfg.model_training.additional_phi_lr)
                    phi_t_decay = utils.make_lr_scheduler(phi_t_optim, cfg.model_training.phi_lr_decay_type,
                                                          getattr(cfg.model_training, "phi_lr_num_steps_oom_drop", None))
                    for k in tqdm(range(cfg.model_training.additional_phi_iters)):
                        phi_t_optim.zero_grad()
                        loss = model.populate_phi_grads(y, cfg.model_training.phi_minibatch_size)
                        phi_t_optim.step()
                        phi_t_decay.step()
                        pbar.set_postfix({"loss": loss.item()})
                        if k % 100 == 0:
                            means, covs = full_batch_stats(model)
                            full_batch_means_training.append(means)
                            full_batch_covs_training.append(covs)

                    save_np('full_batch_means_training.npy', np.array(full_batch_means_training))
                    save_np('full_batch_covs_training.npy', np.array(full_batch_covs_training))

                elif cfg.model_training.func_type == "Offline_amortised":
                    phi_optim = torch.optim.Adam(model.get_phi_params(), lr=cfg.model_training.additional_phi_lr)
                    phi_decay = utils.make_lr_scheduler(phi_optim, cfg.model_training.phi_lr_decay_type,
                                                        getattr(cfg.model_training, "phi_lr_num_steps_oom_drop", None))
                    for k in tqdm(range(cfg.model_training.additional_phi_iters)):
                        phi_optim.zero_grad()
                        loss = model.populate_grads(y, cfg.model_training.phi_minibatch_size)
                        if cfg.model_training.phi_grad_clip_norm is not None:
                            phi_grad_norm = nn.utils.clip_grad_norm_(model.get_phi_params(), cfg.model_training.phi_grad_clip_norm)
                            print("Phi grad norm: ", phi_grad_norm)
                            writer.add_scalar("model_training/phi_grad_norm", phi_grad_norm, cfg.data.num_data*cfg.model_training.phi_iters+k)
                        phi_optim.step()
                        phi_decay.step()
                        pbar.set_postfix({"loss": loss.item()})

                        writer.add_scalar("model_training/lr", phi_decay.get_last_lr()[0], cfg.data.num_data*cfg.model_training.phi_iters+k)

            model_training_end_time = time.time()

            filter_means_training.append(np.array(inner_filter_means_training))
            filter_covs_training.append(np.array(inner_filter_covs_training))
            if T > 0:
                x_Tm1_means_training.append(np.array(inner_x_Tm1_means_training))
                x_Tm1_covs_training.append(np.array(inner_x_Tm1_covs_training))

            if model.amortised:
                model.detach_rnn_hist_hn(y)

            # Train functional approx_funcs
            if cfg.model_training.func_type in ['Vx_t', 'kernel_amortised', 'net_amortised'] and \
                    T >= max(cfg.model_training.window_size - 1, cfg.model_training.approx_updates_start_t):
                approx_func_optim = torch.optim.Adam(model.get_func_t_params(), lr=cfg.model_training.approx_lr)

                if cfg.model_training.func_type in ['Vx_t', 'kernel_amortised']:
                    kernel_inputs, *kernel_targets = model.generate_training_dataset(
                         y, cfg.model_training.kernel_batch_size
                    )
                    model.update_func_t(kernel_inputs, *kernel_targets)

                    if cfg.model_training.KRR_train_sigma or cfg.model_training.KRR_train_lam:
                        kernel_inputs, *kernel_targets = model.generate_training_dataset(
                            y, cfg.model_training.kernel_train_set_size
                        )

                        for i in range(cfg.model_training.approx_iters):
                            idx = np.random.choice(np.arange(kernel_inputs.shape[0]),
                                                   (cfg.model_training.approx_minibatch_size,), replace=False)
                            approx_func_optim.zero_grad()
                            approx_func_loss, _, approx_func_loss_list = model.func_t_loss(kernel_inputs[idx, :],
                                                                       *[kernel_target[idx, :] for kernel_target in kernel_targets])
                            approx_func_loss.backward()
                            approx_func_optim.step()
                            approx_func_losses_T.append(approx_func_loss.item())
                            approx_func_loss_dict["approx_func_loss"] = approx_func_loss.item()
                            pbar.set_postfix({"loss": loss.item(), **approx_func_loss_dict})
                            sigma = model.approx_func_t.kernel.log_sigma.exp().item()
                            lam = model.approx_func_t.kernel.log_lam.exp().item()
                            sigma_T.append(sigma)
                            lambda_T.append(lam)
                            if i % cfg.model_training.approx_loss_write_freq == 0 and model.amortised:
                                writer.add_scalar("approx_func_training/loss", approx_func_loss.item(),
                                                  T * cfg.model_training.approx_iters + i)
                                for j, k in enumerate(["S", "T", "U", "V"]):
                                    writer.add_scalar("approx_func_training/loss_" + k, approx_func_loss_list[j].item(),
                                                      T * cfg.model_training.approx_iters + i)
                                writer.add_scalar("approx_func_training/kernel_sigma", sigma,
                                                  T * cfg.model_training.approx_iters + i)
                                writer.add_scalar("approx_func_training/kernal_lambda", lam,
                                                  T * cfg.model_training.approx_iters + i)

                elif cfg.model_training.func_type == 'net_amortised':
                    net_inputs, *net_targets = model.generate_training_dataset(
                         y, cfg.model_training.net_train_set_size
                    )
                    model.update_func_t(net_inputs, *net_targets)
                    for i in range(cfg.model_training.approx_iters):
                        idx = np.random.choice(np.arange(net_inputs.shape[0]),
                                               (cfg.model_training.approx_minibatch_size,), replace=False)
                        approx_func_optim.zero_grad()
                        approx_func_loss, _, approx_func_loss_list = model.func_t_loss(net_inputs[idx, :],
                                                                   *[net_target[idx, :] for net_target in net_targets])
                        approx_func_loss.backward()
                        approx_func_optim.step()
                        approx_func_losses_T.append(approx_func_loss.item())
                        approx_func_loss_dict["approx_func_loss"] = approx_func_loss.item()
                        pbar.set_postfix({"loss": loss.item(), **approx_func_loss_dict})
                        if i % cfg.model_training.approx_loss_write_freq == 0 and model.amortised:
                            writer.add_scalar("approx_func_training/loss",
                                              approx_func_loss.item(), T * cfg.model_training.approx_iters + i)
                            for j, k in enumerate(["S", "T", "U", "V"]):
                                writer.add_scalar("approx_func_training/loss_" + k, approx_func_loss_list[j].item(),
                                                  T * cfg.model_training.approx_iters + i)

            end_time = time.time()

            filter_stats = model.return_summary_stats(y)
            filter_mean = filter_stats[0].detach().cpu().numpy()
            filter_cov = filter_stats[1].detach().cpu().numpy()

            if T > 0:
                x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=1000)
                x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                x_Tm1_cov = x_Tm1_stats[1].detach().cpu().numpy()

            if cfg.model_training.plot_diagnostics:
                plt.plot(filter_RMSEs_T, label="filter_RMSEs_T")
                if T > 0:
                    plt.plot(x_Tm1_RMSEs_T, label="x_Tm1_RMSEs_T")
                plt.legend()
                plt.title(f"RMSEs (time {T})")
                plt.show()

            losses.append(losses_T)
            if cfg.model_training.func_type in ['Vx_t', 'kernel_amortised', 'net_amortised'] and \
                    T >= max(cfg.model_training.window_size - 1, cfg.model_training.approx_updates_start_t):
                approx_func_losses.append(approx_func_losses_T)
                if cfg.model_training.plot_diagnostics:
                    plt.plot(approx_func_losses_T)
                    plt.title(f"Approximator loss (time {T})")
                    plt.show()
                    if cfg.model_training.func_type in ['Vx_t', 'kernel_amortised']:
                        fig, ax1 = plt.subplots()
                        color = 'C1'
                        ax1.set_ylabel('sigma', color=color)
                        ax1.plot(sigma_T, color=color)
                        ax1.tick_params(axis='y', labelcolor=color)
                        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                        color = 'C2'
                        ax2.set_ylabel('lambda', color=color)  # we already handled the x-label with ax1
                        ax2.plot(lambda_T, color=color)
                        ax2.tick_params(axis='y', labelcolor=color)
                        fig.tight_layout()
                        plt.show()
            filter_RMSEs.append(filter_RMSEs_T)
            if T > 0:
                x_Tm1_RMSEs.append(x_Tm1_RMSEs_T)

            if cfg.model_training.evaluate_elbo:
                true_elbo = model.compute_elbo_loss(y, cfg.model_training.evaluate_elbo_num_samples).item()
                true_elbos.append(true_elbo)
                print("True negative ELBO: ", true_elbo)

                if cfg.model_training.func_type in ['Vx_t']:  # TODO: Amortization
                    elbo_estimate = model.populate_phi_grads(y, cfg.model_training.evaluate_elbo_num_samples).item()
                    elbo_estimates.append(elbo_estimate)
                    print("Negative ELBO estimate: ", elbo_estimate)
                elif cfg.model_training.func_type in ['JELBO', 'VJF']:
                    elbo_estimate = model.populate_phi_grads(y, cfg.model_training.evaluate_elbo_num_samples).item()
                    if T > 0:
                        elbo_estimate += elbo_estimates[-1]
                    elbo_estimates.append(elbo_estimate)
                    print("Negative ELBO estimate: ", elbo_estimate)
                elif cfg.model_training.func_type in ['Offline']:
                    pass # Just the true ELBO is ok
                else:
                    raise NotImplementedError

        # Logging
        filter_means.append(filter_mean)
        filter_covs.append(filter_cov)

        filter_RMSE = np.sqrt(np.mean((filter_mean - x_np[T]) ** 2))

        print("RMSE: ", filter_RMSE)
        writer.add_scalar("model_training/filter_RMSE", filter_RMSE, T)

        if T > 0:
            x_Tm1_means.append(x_Tm1_mean)
            x_Tm1_covs.append(x_Tm1_cov)
            smooth_RMSE = np.sqrt(np.mean((x_Tm1_mean - x_np[T - 1]) ** 2))

            print("RMSE (1-step smoothing): ", smooth_RMSE)
            writer.add_scalar("model_training/smooth_RMSE", smooth_RMSE, T - 1)

        times.append(end_time - start_time)
        model_training_times.append(model_training_end_time - start_time)
        approx_func_training_times.append(end_time - model_training_end_time)
        writer.add_scalar("model_training/time", times[-1], T)
        writer.add_scalar("model_training/model_training_time", model_training_times[-1], T)
        writer.add_scalar("model_training/approx_func_training_time", approx_func_training_times[-1], T)

        if (T % (round(max(cfg.data.num_data, 10) / 10)) == 0) or (T == cfg.data.num_data - 1):
            save_np('filter_means.npy', np.array(filter_means))
            save_np('filter_covs.npy', np.array(filter_covs))
            save_np('x_Tm1_means.npy', np.array(x_Tm1_means))
            save_np('x_Tm1_covs.npy', np.array(x_Tm1_covs))
            save_np('filter_RMSEs.npy', np.array(filter_RMSEs))
            save_np('x_Tm1_RMSEs.npy', np.array(x_Tm1_RMSEs))
            save_np('losses.npy', np.array(losses))
            save_np('true_elbos.npy', np.array(true_elbos))
            save_np('elbo_estimates.npy', np.array(elbo_estimates))
            save_np('times.npy', np.array(times))
            save_np('model_training_times.npy', np.array(model_training_times))
            save_np('approx_func_training_times.npy', np.array(approx_func_training_times))
            if cfg.model_training.func_type in ['Vx_t', 'kernel_amortised', 'net_amortised']:
                save_np('approx_func_losses.npy', np.array(approx_func_losses))
            save_np('filter_means_training.npy', np.array(filter_means_training))
            save_np('filter_covs_training.npy', np.array(filter_covs_training))
            save_np('x_Tm1_means_training.npy', np.array(x_Tm1_means_training))
            save_np('x_Tm1_covs_training.npy', np.array(x_Tm1_covs_training))

    if cfg.data.num_data <= 100:
        all_means, all_covs = full_batch_stats(model)
        save_np('full_batch_means.npy', all_means)
        save_np('full_batch_covs.npy', all_covs)
    else:
        print("T too big for full batch stats?!")


    print(f"Average filtering RMSE after t={cfg.data.num_data // 10}:",
          np.mean(np.sqrt(np.mean((np.array(filter_means) - x_np) ** 2, axis=1))[cfg.data.num_data // 10:]))
    print(f"Average 1-step smoothing RMSE after t={cfg.data.num_data // 10}:",
          np.mean(np.sqrt(np.mean((np.array(x_Tm1_means) - x_np[:-1, :]) ** 2, axis=1))[cfg.data.num_data // 10:]))

    if model.amortised:
        model.rnn_forward(y)
        x_samples_historical_rnn, _ = model.sample_joint_q_t(100,
                                                             cfg.data.num_data - 1, True)
        joint_means_historical_rnn = torch.stack(x_samples_historical_rnn).mean(1).detach().cpu().numpy()

        filter_means_historical_rnn = []
        filter_covs_historical_rnn = []
        for T in range(0, cfg.data.num_data):
            q_T_mean, q_T_std = model.compute_filter_stats(T=T)
            filter_means_historical_rnn.append(q_T_mean)
            filter_covs_historical_rnn.append(q_T_std.diag().square())
        filter_means_historical_rnn = torch.stack(filter_means_historical_rnn, dim=0).detach().cpu().numpy()
        filter_covs_historical_rnn = torch.stack(filter_covs_historical_rnn, dim=0).detach().cpu().numpy()

        model.rnn_forward_offline(y)
        x_samples_offline_rnn, _ = model.sample_joint_q_t(100,
                                                          cfg.data.num_data - 1, True)
        joint_means_offline_rnn = torch.stack(x_samples_offline_rnn).mean(1).detach().cpu().numpy()

        filter_means_offline_rnn = []
        filter_covs_offline_rnn = []
        for T in range(0, cfg.data.num_data):
            q_T_mean, q_T_std = model.compute_filter_stats(T=T)
            filter_means_offline_rnn.append(q_T_mean)
            filter_covs_offline_rnn.append(q_T_std.diag().square())
        filter_means_offline_rnn = torch.stack(filter_means_offline_rnn, dim=0).detach().cpu().numpy()
        filter_covs_offline_rnn = torch.stack(filter_covs_offline_rnn, dim=0).detach().cpu().numpy()

        plt.plot(x_np[:, 0], label="x (first dim)")
        plt.plot(filter_means_offline_rnn[:, 0], label="joint_means_offline_rnn (first dim)")
        plt.legend()
        plt.show()

        historical_rnn_filter_rmse = np.mean(
            np.sqrt(np.mean((np.array(filter_means_historical_rnn) - x_np) ** 2,
                            axis=1))[cfg.data.num_data // 10:])
        offline_rnn_filter_rmse = np.mean(np.sqrt(np.mean((np.array(filter_means_offline_rnn) - x_np) ** 2,
                                                          axis=1))[cfg.data.num_data // 10:])

        print(f"Average filter RMSE after t={cfg.data.num_data // 10} (historical RNN):", historical_rnn_filter_rmse)
        print(f"Average filter RMSE after t={cfg.data.num_data // 10} (offline RNN):", offline_rnn_filter_rmse)
        writer.add_scalar("test/historical_rnn_filter_rmse", historical_rnn_filter_rmse)
        writer.add_scalar("test/offline_rnn_filter_rmse", offline_rnn_filter_rmse)

        historical_rnn_joint_rmse = np.mean(np.sqrt(np.mean((np.array(joint_means_historical_rnn) - x_np) ** 2,
                                                            axis=1))[cfg.data.num_data // 10:])
        offline_rnn_joint_rmse = np.mean(np.sqrt(np.mean((np.array(joint_means_offline_rnn) - x_np) ** 2,
                                                         axis=1))[cfg.data.num_data // 10:])

        print(f"Average joint RMSE after t={cfg.data.num_data // 10} (historical RNN):", historical_rnn_joint_rmse)
        print(f"Average joint RMSE after t={cfg.data.num_data // 10} (offline RNN):", offline_rnn_joint_rmse)

        writer.add_scalar("test/historical_rnn_joint_rmse", historical_rnn_joint_rmse)
        writer.add_scalar("test/offline_rnn_smooth_rmse", offline_rnn_joint_rmse)

        joint_covs_offline_rnn = utils.sample_cov(torch.stack(x_samples_offline_rnn)).detach().cpu().numpy()
        joint_covs_historical_rnn = utils.sample_cov(torch.stack(x_samples_historical_rnn)).detach().cpu().numpy()

        save_np('filter_means_historical_rnn.npy', np.array(filter_means_historical_rnn))
        save_np('filter_covs_historical_rnn.npy', np.array(filter_covs_historical_rnn))
        save_np('filter_means_offline_rnn.npy', np.array(filter_means_offline_rnn))
        save_np('filter_covs_offline_rnn.npy', np.array(filter_covs_offline_rnn))

        save_np('joint_means_historical_rnn.npy', np.array(joint_means_historical_rnn))
        save_np('joint_covs_historical_rnn.npy', np.array(joint_covs_historical_rnn))
        save_np('joint_means_offline_rnn.npy', np.array(joint_means_offline_rnn))
        save_np('joint_covs_offline_rnn.npy', np.array(joint_covs_offline_rnn))

    # plt.plot(elbo_estimates, label="elbo estimates")
    # plt.plot(true_elbos, "--", label="true elbos")
    # plt.legend()
    # plt.show()

    if cfg.model_training.evaluate_final_elbo:
        if cfg.model_training.func_type in ['Vx_t', 'JELBO', 'Ignore_Past', 'VJF', 'Offline']:
            elbo_func = model.compute_elbo_loss
        elif cfg.model_training.func_type in ['kernel_amortised', 'net_amortised', 'AELBO2_amortised', 'Offline_amortised',
                                            'Analytic_amortised']:
            elbo_func = model.compute_elbo_loss_offline_rnn
        else:
            raise NotImplementedError

        final_neg_elbos = []
        for i in range(10):
            final_true_negative_elbo = elbo_func(y, cfg.model_training.evaluate_elbo_num_samples).item()
            final_neg_elbos.append(final_true_negative_elbo)
        final_neg_elbos = np.array(final_neg_elbos)
        print("Final true negative ELBO: ", np.mean(final_neg_elbos))
        save_np('final_true_negative_elbo.npy', final_neg_elbos)
        writer.add_scalar("test/final_true_negative_elbo", np.mean(final_neg_elbos))

    if cfg.model_training.save_models:
        torch.save(model.state_dict(), os.path.join(saved_models_folder_name, cfg.model_training.func_type + ".pt"))


def generate_data(cfg):
    device = cfg.device

    if cfg.data.data_name == "CRNN":
        xdim = cfg.data.dim
        ydim = cfg.data.dim

        grid_size = cfg.data.grid_size
        gamma = cfg.data.gamma
        tau = cfg.data.tau
        U_std = cfg.data.U_std
        V_scale = cfg.data.V_scale
        V_df = cfg.data.V_df

        W = torch.randn(xdim, xdim).to(device) / np.sqrt(xdim)
        G = torch.eye(ydim).to(device)
        mean_0 = torch.zeros(xdim).to(device)
        std_0 = U_std

        class F_module(nn.Module):
            # Nonlinear model, additive Gaussian noise
            def __init__(self):
                super().__init__()
                self.W = W

            def forward(self, input, t=None):
                mean = self.F_mean_fn(input, t)
                return Independent(Normal(mean, U_std), 1)

            def F_mean_fn(self, x, t):
                return x + grid_size * (-x + gamma * functional.linear(torch.tanh(x), self.W)) / tau

            def F_cov_fn(self, x, t):
                return torch.eye(xdim, device=device) * U_std ** 2

        class G_module(nn.Module):
            # Linear model, additive StudentT noise
            def __init__(self):
                super().__init__()

            def forward(self, input, t=None):
                mean = self.G_mean_fn(input, t)
                return Independent(StudentT(V_df, mean, V_scale), 1)

            def G_mean_fn(self, x, t):
                return functional.linear(x, G)

        class p_0_dist_module(nn.Module):
            def __init__(self):
                super().__init__()
                self.mean_0 = mean_0
                self.cov_0 = torch.eye(xdim, device=device) * std_0 ** 2

            def forward(self):
                return Independent(Normal(mean_0, std_0), 1)

        F_fn = F_module().to(device)
        G_fn = G_module().to(device)
        p_0_dist = p_0_dist_module().to(device)

        if cfg.data.path_to_data is None:
            data_gen = nonamortised_models.NonAmortizedModelBase(device, xdim, ydim, None, None, None, None, F_fn, G_fn,
                                                                 p_0_dist, None, 1)
            x, y = data_gen.generate_data(cfg.data.num_data)
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            save_np('x_data.npy', x_np)
            save_np('y_data.npy', y_np)
            save_np('W.npy', W.cpu().numpy())

        else:
            path_to_data = hydra.utils.to_absolute_path(cfg.data.path_to_data)
            x_np = np.load(os.path.join(path_to_data, 'x_data.npy'))
            y_np = np.load(os.path.join(path_to_data, 'y_data.npy'))

            W_np = np.load(os.path.join(path_to_data, 'W.npy'))
            W = torch.from_numpy(W_np).float().to(device)
            F_fn.W = W

    # plt.plot(x_np[:, :2], label="x (first 2 dims)")
    # plt.plot(y_np[:, :2], label="y (first 2 dims)")
    # plt.legend()
    # plt.show()

    return x_np, y_np, xdim, ydim, F_fn, G_fn, p_0_dist


def full_batch_stats(model, num_samples=1000):
    T = model.T
    dim = len(model.q_t_mean_list[T])
    samples = torch.zeros((T+1, num_samples, dim))
    samples[T, :, :] = model.q_t_mean_list[T] + \
        torch.exp(model.q_t_log_std_list[T]) * torch.randn(num_samples, dim)
    for t in range(T-1, -1, -1):
        samples[t, :, :] = model.cond_q_t_mean_net_list[t+1](samples[t+1, :, :]) + \
            torch.exp(model.cond_q_t_log_std_list[t+1]) * torch.randn(num_samples, dim)
    np_samples = samples.detach().numpy()
    means = np.mean(np_samples, axis=1)
    covs = utils.sample_cov(samples).detach().numpy()
    return means, covs
    

if __name__ == "__main__":
    main()
