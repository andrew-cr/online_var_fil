# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Independent, Normal, MultivariateNormal, StudentT
from tqdm import tqdm
import core.nonamortised_models as models
import core.utils as utils
import math
import subprocess
import hydra
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import time


def save_np(name, x):
    np.save(name, x)


@hydra.main(config_path='conf', config_name="CTRNN_train")
def main(cfg):
    utils.save_git_hash(hydra.utils.get_original_cwd())
    device = cfg.device

    seed = np.random.randint(0, 9999999) if cfg.seed is None else cfg.seed
    print("seed", seed)
    with open('seed.txt', 'w') as f:
        f.write(str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    x, y, x_np, y_np, xdim, ydim, F_fn, G_fn, p_0_dist = generate_data(cfg)

    saved_models_folder_name = 'saved_models'
    if cfg.model_training.save_models:
        os.mkdir(saved_models_folder_name)

    if cfg.model_training.func_type == 'Bootstrap_PF':
        model = models.BootstrapParticleFilter(
            device, xdim, ydim,
            F_fn, G_fn, p_0_dist,
            cfg.model_training.num_particles
        )
    elif cfg.model_training.func_type == 'EnKF':
        model = models.EnsembleKalmanFilter(
            device, xdim, ydim,
            F_fn, G_fn, p_0_dist,
            cfg.model_training.ensemble_size
        )
    elif cfg.model_training.func_type in ['Vx_t', 'JELBO', 'Ignore_Past', 'VJF']:
        q_hidden_dims = cfg.model_training.q_hidden_dims
        phi_t_init_method = cfg.model_training.phi_t_init_method

        def cond_q_mean_net_constructor():
            net_dims = [xdim] + list(q_hidden_dims) + [xdim]
            modules = []
            for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
                modules.append(nn.Linear(in_dim, out_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(net_dims[-2], net_dims[-1]))
            return nn.Sequential(*modules).to(device)

        print("cond_q_mean_net: ", cond_q_mean_net_constructor())
        if cfg.model_training.func_type == 'Vx_t':
            sigma = cfg.model_training.KRR_sigma
            lam = cfg.model_training.KRR_lambda
            train_sigma = cfg.model_training.KRR_train_sigma
            train_lam = cfg.model_training.KRR_train_lam

            def KRR_constructor():
                return models.KernelRidgeRegressor(models.MaternKernel(
                    sigma=sigma, lam=lam, train_sigma=train_sigma, train_lam=train_lam),
                    centre_elbo=cfg.model_training.KRR_centre_elbo).to(device)

            model = models.Vx_t_phi_t_Model(
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
            model = models.JELBO_Model(device, xdim, ydim,
                                       torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                                       cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                                       F_fn, G_fn, p_0_dist,
                                       phi_t_init_method, cfg.model_training.window_size,
                                       num_params_to_store=9999999)

        elif cfg.model_training.func_type == 'Ignore_Past':
            model = models.Ignore_Past_phi_t_Model(device, xdim, ydim,
                                                   torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                                                   cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                                                   F_fn, G_fn, p_0_dist,
                                                   phi_t_init_method, cfg.model_training.window_size,
                                                   num_params_to_store=9999999)

        elif cfg.model_training.func_type == 'VJF':
            model = models.VJF_Model(device, xdim, ydim,
                                     torch.randn(xdim, device=device), torch.zeros(xdim, device=device),
                                     cond_q_mean_net_constructor, torch.zeros(xdim, device=device),
                                     F_fn, G_fn, p_0_dist,
                                     phi_t_init_method, cfg.model_training.window_size,
                                     num_params_to_store=9999999)

    else:
        raise ValueError('Unknown func_type type')

    filter_means = []
    filter_covs = []
    x_Tm1_means = []
    x_Tm1_covs = []
    true_elbos = []
    elbo_estimates = []
    times = []

    losses = []
    approximator_losses = []
    filter_RMSEs = []
    x_Tm1_RMSEs = []

    pbar = tqdm(range(0, cfg.data.num_data))

    for T in pbar:
        start_time = time.time()
        if cfg.model_training.func_type in ['Bootstrap_PF', 'EnKF']:
            model.advance_timestep(y[T, :])
            model.update(y[T, :])

            filter_stats = model.return_summary_stats()
            filter_mean = filter_stats[0].detach().cpu().numpy()
            filter_cov = filter_stats[1].detach().cpu().numpy()

            end_time = time.time()

            if T > 0:
                x_Tm1_stats = model.return_summary_stats(t=T - 1)
                x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                x_Tm1_cov = x_Tm1_stats[1].detach().cpu().numpy()

        else:
            losses_T = []
            approximator_losses_T = []
            filter_RMSEs_T = []
            x_Tm1_RMSEs_T = []

            sigma_T = []
            lambda_T = []

            model.advance_timestep(y[T, :])
            model_phi_t_optim = torch.optim.Adam(model.get_phi_T_params(), lr=cfg.model_training.phi_lr)

            approximator_loss_dict = {}
            for k in range(cfg.model_training.phi_iters):
                model_phi_t_optim.zero_grad()
                loss = model.populate_phi_grads(y, cfg.model_training.phi_minibatch_size)
                model_phi_t_optim.step()

                pbar.set_postfix({"loss": loss.item(), **approximator_loss_dict})
                losses_T.append(loss.item())

                filter_stats = model.return_summary_stats(y)
                filter_mean = filter_stats[0].detach().cpu().numpy()
                filter_RMSEs_T.append(np.sqrt(np.mean((filter_mean - x_np[T]) ** 2)))

                if T > 0:
                    x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=1000)
                    x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                    x_Tm1_RMSEs_T.append(np.sqrt(np.mean((x_Tm1_mean - x_np[T - 1]) ** 2)))

            if cfg.model_training.func_type == 'Vx_t' and T >= cfg.model_training.window_size - 1:
                model.update_V_t(y, cfg.model_training.V_batch_size)
                Vx_optim = torch.optim.Adam(model.get_V_t_params(), lr=cfg.model_training.V_lr)
                for k in range(cfg.model_training.V_iters):
                    Vx_optim.zero_grad()
                    V_loss, _, _ = model.V_t_loss(y, cfg.model_training.V_minibatch_size)
                    V_loss.backward()
                    Vx_optim.step()
                    approximator_losses_T.append(V_loss.item())
                    approximator_loss_dict["V_loss"] = V_loss.item()
                    pbar.set_postfix({"loss": loss.item(), **approximator_loss_dict})
                    sigma_T.append(model.V_func_t.kernel.log_sigma.exp().item())
                    lambda_T.append(model.V_func_t.kernel.log_lam.exp().item())

            end_time = time.time()

            filter_stats = model.return_summary_stats(y)
            filter_mean = filter_stats[0].detach().cpu().numpy()
            filter_cov = filter_stats[1].detach().cpu().numpy()

            if T > 0:
                x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=10000)
                x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                x_Tm1_cov = x_Tm1_stats[1].detach().cpu().numpy()

            if cfg.model_training.plot_diagnostics:
                plt.plot(filter_RMSEs_T)
                if T > 0:
                    plt.plot(x_Tm1_RMSEs_T)
                plt.show()

            losses.append(losses_T)
            if cfg.model_training.func_type == 'Vx_t' and T >= cfg.model_training.window_size - 1:
                approximator_losses.append(approximator_losses_T)
                if cfg.model_training.plot_diagnostics:
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

                if cfg.model_training.func_type in ['Vx_t', 'Zx_t']:
                    elbo_estimate = model.populate_phi_grads(y, cfg.model_training.evaluate_elbo_num_samples).item()
                    elbo_estimates.append(elbo_estimate)
                    print("Negative ELBO estimate: ", elbo_estimate)
                elif cfg.model_training.func_type in ['JELBO', 'VJF']:
                    elbo_estimate = model.populate_phi_grads(y, cfg.model_training.evaluate_elbo_num_samples).item()
                    if T > 0:
                        elbo_estimate += elbo_estimates[-1]
                    elbo_estimates.append(elbo_estimate)
                    print("ELBO estimate: ", elbo_estimate)
                else:
                    raise NotImplementedError

        # Logging
        filter_means.append(filter_mean)
        filter_covs.append(filter_cov)

        print("RMSE: ", np.sqrt(np.mean((filter_mean - x_np[T]) ** 2)))

        if T > 0:
            x_Tm1_means.append(x_Tm1_mean)
            x_Tm1_covs.append(x_Tm1_cov)
            print("RMSE (1-step smoothing): ", np.sqrt(np.mean((x_Tm1_mean - x_np[T - 1]) ** 2)))
        times.append(end_time - start_time)

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
            if cfg.model_training.func_type in ['Vx_t']:
                save_np('approximator_losses.npy', np.array(approximator_losses))

    print(f"Average filtering RMSE after t={cfg.data.num_data // 10}:",
          np.mean(np.sqrt(np.mean((np.array(filter_means) - x_np) ** 2, axis=1))[cfg.data.num_data // 10:]))
    print(f"Average 1-step smoothing RMSE after t={cfg.data.num_data // 10}:",
          np.mean(np.sqrt(np.mean((np.array(x_Tm1_means) - x_np[:-1, :]) ** 2, axis=1))[cfg.data.num_data // 10:]))

    plt.plot(elbo_estimates, label="elbo estimates")
    plt.plot(true_elbos, "--", label="true elbos")
    plt.legend()
    plt.show()

    if cfg.model_training.func_type in ['Vx_t', 'JELBO', 'Ignore_Past', 'VJF']:
        if cfg.model_training.evaluate_final_elbo:
            final_true_elbo = model.compute_elbo_loss(y, cfg.model_training.evaluate_elbo_num_samples).item()
            print("Final true negative ELBO: ", final_true_elbo)
            save_np('final_true_elbo.npy', final_true_elbo)

    if cfg.model_training.save_models:
        torch.save(model.state_dict(), os.path.join(saved_models_folder_name, cfg.model_training.func_type + ".pt"))


def generate_data(cfg):
    device = cfg.device

    if cfg.data.data_name == "CTRNN":
        xdim = cfg.data.dim
        ydim = cfg.data.dim

        grid_size = cfg.data.grid_size
        gamma = cfg.data.gamma
        tau = cfg.data.tau
        U_std = cfg.data.U_std
        V_scale = cfg.data.V_scale
        V_df = cfg.data.V_df

        W = torch.randn(xdim, xdim).to(device) / np.sqrt(xdim)
        G = torch.eye(ydim)
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
            data_gen = models.NonAmortizedModelBase(device, xdim, ydim, None, None, None, None, F_fn, G_fn, p_0_dist,
                                                    None,
                                                    1)
            x, y = data_gen.generate_data(cfg.data.num_data)
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            save_np('x_data.npy', x_np)
            save_np('y_data.npy', y_np)
            save_np('W.npy', W.cpu().numpy())

        else:
            path_to_data = hydra.utils.to_absolute_path(cfg.data.path_to_data) + '/'
            x_np = np.load(os.path.join(path_to_data, 'x_data.npy'))
            y_np = np.load(os.path.join(path_to_data, 'y_data.npy'))
            x = torch.from_numpy(x_np).float().to(device)
            y = torch.from_numpy(y_np).float().to(device)

            W_np = np.load(os.path.join(path_to_data, 'W.npy'))
            W = torch.from_numpy(W_np).float().to(device)
            F_fn.W = W

    return x, y, x_np, y_np, xdim, ydim, F_fn, G_fn, p_0_dist

if __name__ == "__main__":
    main()
