# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.distributions import Independent, Normal, MultivariateNormal, StudentT
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


@hydra.main(config_path='conf', config_name="CRNNAmortized")
def main(cfg):
    utils.save_git_hash(hydra.utils.get_original_cwd())
    device = cfg.device

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
    elif cfg.model_training.func_type == 'kernel_amortised':
        q_rnn_hidden_dim = cfg.model_training.q_rnn_hidden_dim
        q_rnn_num_layers = cfg.model_training.q_rnn_num_layers

        assert cfg.model_training.q_net_type == "Identity"
        q_rnn = networks.MLPRNN(
            [ydim + 2 * xdim] + [q_rnn_hidden_dim] * q_rnn_num_layers + [2 * xdim], nn.ReLU
        ).to(device)

        q_hidden_dims = cfg.model_training.q_hidden_dims  # Hidden dims of q_t_net and cond_q_t_net

        q_t_net = networks.NormalNet(nn.Identity()).to(device)
        cond_q_t_net = networks.NormalNet(
            networks.MLP(
                [xdim + q_rnn.hidden_size] + list(q_hidden_dims) + [2 * xdim], nn.ReLU
            )
        ).to(device)

        sigma = cfg.model_training.KRR_sigma
        lam = cfg.model_training.KRR_lambda
        train_sigma = cfg.model_training.KRR_train_sigma
        train_lam = cfg.model_training.KRR_train_lam

        def KRR_constructor():
            return nonamortised_models.KernelRidgeRegressor(nonamortised_models.MaternKernel(
                sigma=sigma, lam=lam, train_sigma=train_sigma, train_lam=train_lam),
                centre_elbo=cfg.model_training.KRR_centre_elbo).to(device)

        model = amortised_models.Kernel_Amortised_Model(device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net,
                                                        F_fn, G_fn, p_0_dist, 1, 1, 0,
                                                        KRR_constructor,
                                                        cfg.model_training.funcs_to_approx,
                                                        0,
                                                        cfg.model_training.approx_decay,
                                                        cfg.model_training.approx_with_filter)

        theta_dim = len(utils.replace_none(model.get_theta_params(flatten=True)))
        phi_dim = len(utils.replace_none(model.get_phi_params(flatten=True)))
        rnn_phi_dim = len(utils.replace_none(model.get_rnn_phi_params(flatten=True)))
        q_t_net_phi_dim = len(utils.replace_none(model.get_q_t_net_phi_params(flatten=True)))

        print("Theta dim: ", theta_dim)
        print("Phi dim: ", phi_dim)
        print("Phi dim (RNN): ", rnn_phi_dim)
        print("Phi dim (q_t_net): ", q_t_net_phi_dim)
    

    phi_optim = torch.optim.Adam(model.get_phi_params(), lr=cfg.model_training.phi_lr)
    phi_decay = utils.make_lr_scheduler(phi_optim, cfg.model_training.phi_lr_decay_type,
                                        getattr(cfg.model_training, "phi_lr_num_steps_oom_drop", None))
    num_epochs = cfg.model_training.num_epochs


    for epoch in range(num_epochs):
        pbar = tqdm(range(cfg.data.num_data))

        filter_means = []
        filter_covs = []
        x_Tm1_means = []
        x_Tm1_covs = []

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
                model.advance_timestep(y[T, :])

                approx_func_loss_dict = {}
                for k in range(cfg.model_training.phi_iters):
                    phi_optim.zero_grad()
                    loss = model.populate_grads(y, cfg.model_training.phi_minibatch_size)
                    phi_optim.step()

                    pbar.set_postfix({"loss": loss.item(), **approx_func_loss_dict})

                model_training_end_time = time.time()

                model.detach_rnn_hist_hn(y)

                # Train functional approx_funcs
                approx_func_optim = torch.optim.Adam(model.get_func_t_params(), lr=cfg.model_training.approx_lr)

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
                        approx_func_loss, _, _ = model.func_t_loss(kernel_inputs[idx, :],
                                                                   *[kernel_target[idx, :] for kernel_target in kernel_targets])
                        approx_func_loss.backward()
                        approx_func_optim.step()
                        approx_func_loss_dict["approx_func_loss"] = approx_func_loss.item()
                        pbar.set_postfix({"loss": loss.item(), **approx_func_loss_dict})

                end_time = time.time()

                filter_stats = model.return_summary_stats(y)
                filter_mean = filter_stats[0].detach().cpu().numpy()
                filter_cov = filter_stats[1].detach().cpu().numpy()

                if T > 0:
                    x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=1000)
                    x_Tm1_mean = x_Tm1_stats[0].detach().cpu().numpy()
                    x_Tm1_cov = x_Tm1_stats[1].detach().cpu().numpy()

            # Logging
            filter_means.append(filter_mean)
            filter_covs.append(filter_cov)

            filter_RMSE = np.sqrt(np.mean((filter_mean - x_np[T]) ** 2))

            print("RMSE: ", filter_RMSE)

            if T > 0:
                x_Tm1_means.append(x_Tm1_mean)
                x_Tm1_covs.append(x_Tm1_cov)
                smooth_RMSE = np.sqrt(np.mean((x_Tm1_mean - x_np[T - 1]) ** 2))

                print("RMSE (1-step smoothing): ", smooth_RMSE)

        save_np('filter_means.npy', np.array(filter_means))
        save_np('filter_covs.npy', np.array(filter_covs))
        save_np('x_Tm1_means.npy', np.array(x_Tm1_means))
        save_np('x_Tm1_covs.npy', np.array(x_Tm1_covs))

        if model.amortised and epoch < num_epochs - 1:
            phi_decay.step()
            model.reset_timestep()

    print(f"Average filtering RMSE after t={cfg.data.num_data // 10}:",
          np.mean(np.sqrt(np.mean((np.array(filter_means) - x_np) ** 2, axis=1))[cfg.data.num_data // 10:]))
    print(f"Average 1-step smoothing RMSE after t={cfg.data.num_data // 10}:",
          np.mean(np.sqrt(np.mean((np.array(x_Tm1_means) - x_np[:-1, :]) ** 2, axis=1))[cfg.data.num_data // 10:]))

    if model.amortised:
        # Historical
        # Joint
        model.rnn_forward(y)
        x_samples_historical_rnn, _ = model.sample_joint_q_t(100,
                                                             cfg.data.num_data - 1, True)
        joint_means_historical_rnn = torch.stack(x_samples_historical_rnn).mean(1).detach().cpu().numpy()
        joint_covs_historical_rnn = utils.sample_cov(torch.stack(x_samples_historical_rnn)).detach().cpu().numpy()

        # Filter (plot)
        filter_means_historical_rnn = []
        filter_covs_historical_rnn = []
        for T in range(0, cfg.data.num_data):
            q_T_mean, q_T_std = model.compute_filter_stats(T=T)
            filter_means_historical_rnn.append(q_T_mean)
            filter_covs_historical_rnn.append(q_T_std.diag().square())
        filter_means_historical_rnn = torch.stack(filter_means_historical_rnn, dim=0).detach().cpu().numpy()
        filter_covs_historical_rnn = torch.stack(filter_covs_historical_rnn, dim=0).detach().cpu().numpy()

        for i in range(3):
            plt.plot(x_np[:, i], c="C"+str(i))
            plt.plot(filter_means_historical_rnn[:, i], ls="--", c="C"+str(i))
        plt.title("x vs filter_means_historical_rnn (first 3 dims)")
        plt.savefig("filter_means_historical_rnn.png")

        # Offline
        # Joint
        model.rnn_forward_offline(y)
        x_samples_offline_rnn, _ = model.sample_joint_q_t(100,
                                                          cfg.data.num_data - 1, True)
        joint_means_offline_rnn = torch.stack(x_samples_offline_rnn).mean(1).detach().cpu().numpy()
        joint_covs_offline_rnn = utils.sample_cov(torch.stack(x_samples_offline_rnn)).detach().cpu().numpy()

        # Filter (plot)
        filter_means_offline_rnn = []
        filter_covs_offline_rnn = []
        for T in range(0, cfg.data.num_data):
            q_T_mean, q_T_std = model.compute_filter_stats(T=T)
            filter_means_offline_rnn.append(q_T_mean)
            filter_covs_offline_rnn.append(q_T_std.diag().square())
        filter_means_offline_rnn = torch.stack(filter_means_offline_rnn, dim=0).detach().cpu().numpy()
        filter_covs_offline_rnn = torch.stack(filter_covs_offline_rnn, dim=0).detach().cpu().numpy()

        for i in range(3):
            plt.plot(x_np[:, i], c="C"+str(i))
            plt.plot(filter_means_offline_rnn[:, i], ls="--", c="C"+str(i))
        plt.title("x vs filter_means_offline_rnn (first 3 dims)")
        plt.savefig("filter_means_offline_rnn.png")

        historical_rnn_filter_rmse = np.mean(
            np.sqrt(np.mean((np.array(filter_means_historical_rnn) - x_np) ** 2,
                            axis=1))[cfg.data.num_data // 10:])
        offline_rnn_filter_rmse = np.mean(np.sqrt(np.mean((np.array(filter_means_offline_rnn) - x_np) ** 2,
                                                          axis=1))[cfg.data.num_data // 10:])

        print(f"Average filter RMSE after t={cfg.data.num_data // 10} (historical RNN):", historical_rnn_filter_rmse)
        print(f"Average filter RMSE after t={cfg.data.num_data // 10} (offline RNN):", offline_rnn_filter_rmse)

        historical_rnn_joint_rmse = np.mean(np.sqrt(np.mean((np.array(joint_means_historical_rnn) - x_np) ** 2,
                                                            axis=1))[cfg.data.num_data // 10:])
        offline_rnn_joint_rmse = np.mean(np.sqrt(np.mean((np.array(joint_means_offline_rnn) - x_np) ** 2,
                                                         axis=1))[cfg.data.num_data // 10:])

        print(f"Average joint RMSE after t={cfg.data.num_data // 10} (historical RNN):", historical_rnn_joint_rmse)
        print(f"Average joint RMSE after t={cfg.data.num_data // 10} (offline RNN):", offline_rnn_joint_rmse)


        save_np('filter_means_historical_rnn.npy', np.array(filter_means_historical_rnn))
        save_np('filter_covs_historical_rnn.npy', np.array(filter_covs_historical_rnn))
        save_np('filter_means_offline_rnn.npy', np.array(filter_means_offline_rnn))
        save_np('filter_covs_offline_rnn.npy', np.array(filter_covs_offline_rnn))

        save_np('joint_means_historical_rnn.npy', np.array(joint_means_historical_rnn))
        save_np('joint_covs_historical_rnn.npy', np.array(joint_covs_historical_rnn))
        save_np('joint_means_offline_rnn.npy', np.array(joint_means_offline_rnn))
        save_np('joint_covs_offline_rnn.npy', np.array(joint_covs_offline_rnn))

    if cfg.model_training.func_type == 'kernel_amortised':
        if cfg.model_training.evaluate_final_elbo:
            final_true_negative_elbo = model.compute_elbo_loss_offline_rnn(
                y, cfg.model_training.evaluate_elbo_num_samples).item()
            print("Final true negative ELBO: ", final_true_negative_elbo)
            save_np('final_true_negative_elbo.npy', final_true_negative_elbo)

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


    return x_np, y_np, xdim, ydim, F_fn, G_fn, p_0_dist


if __name__ == "__main__":
    main()
