# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from core.data_generation import GaussianHMM, construct_HMM_matrices
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


@hydra.main(config_path='conf', config_name="fig1a")
def main(cfg):
    assert cfg.data.diagFG
    utils.save_git_hash(hydra.utils.get_original_cwd())
    device = cfg.device

    seed = np.random.randint(0, 9999999) if cfg.seed is None else cfg.seed
    print("seed", seed)
    with open('seed.txt', 'w') as f:
        f.write(str(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_folder_name = 'saved_models'
    os.mkdir(saved_models_folder_name)

    DIM = cfg.data.dim

    if cfg.data.path_to_data is None:
        F, G, U, V = construct_HMM_matrices(dim=DIM,
                                            F_eigvals=np.random.uniform(
                                                cfg.data.F_min_eigval,
                                                cfg.data.F_max_eigval, (DIM)),
                                            G_eigvals=np.random.uniform(
                                                cfg.data.G_min_eigval,
                                                cfg.data.G_max_eigval, (DIM)),
                                            U_std=cfg.data.U_std,
                                            V_std=cfg.data.V_std,
                                            diag=cfg.data.diagFG)

        data_gen = GaussianHMM(xdim=DIM, ydim=DIM, F=F, G=G, U=U, V=V)
        x_np, y_np = data_gen.generate_data(cfg.data.num_data)

        save_np('datapoints.npy', np.stack((x_np, y_np)))
        save_np('F.npy', F)
        save_np('G.npy', G)
        save_np('U.npy', U)
        save_np('V.npy', V)
    else:
        path_to_data = hydra.utils.to_absolute_path(cfg.data.path_to_data) + '/'
        F, G, U, V = np.load(path_to_data + 'F.npy'), \
                        np.load(path_to_data + 'G.npy'), \
                        np.load(path_to_data + 'U.npy'), \
                        np.load(path_to_data + 'V.npy')
        xystack = np.load(path_to_data + 'datapoints.npy')
        x_np = xystack[0, :, :]
        y_np = xystack[1, :, :]

    kalman_xs = np.zeros((y_np.shape[0], DIM))
    kalman_Ps = np.zeros((y_np.shape[0], DIM, DIM))

    # For t=0
    kalman_Ps[0, :, :] = np.linalg.inv(np.eye(DIM) + G.T @ np.linalg.inv(V) @ G)
    kalman_xs[0, :] = kalman_Ps[0, :, :] @ G.T @ np.linalg.inv(V) @ y_np[0, :]

    kalman_filter = models.KalmanFilter(x_0=kalman_xs[0, :], P_0=kalman_Ps[0, :, :], F=F, G=G, U=U,
                                        V=V)

    for t in range(1, y_np.shape[0]):
        kalman_filter.update(y_np[t, :])
        kalman_xs[t, :] = kalman_filter.x
        kalman_Ps[t, :, :] = kalman_filter.P
    save_np('kalman_xs.npy', kalman_xs)
    save_np('kalman_Ps.npy', kalman_Ps)
    kalman_xs_pyt = torch.from_numpy(kalman_xs).float()
    kalman_Ps_pyt = torch.from_numpy(kalman_Ps).float()

    y = torch.from_numpy(y_np).float().to(device)

    F = torch.from_numpy(F).float().to(device)
    G = torch.from_numpy(G).float().to(device)

    U = torch.from_numpy(U).float().to(device)
    V = torch.from_numpy(V).float().to(device)
    mean_0 = torch.zeros(DIM).to(device)

    class F_Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('weight',
                nn.Parameter(torch.zeros(DIM)))
            self.F_mean_fn = lambda x, t: self.weight * x
            self.F_cov_fn = lambda x, t: U

        def forward(self, x, t=None):
            return Independent(Normal(self.F_mean_fn(x, t),
                torch.sqrt(torch.diag(U))), 1)

    class G_Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('weight',
                nn.Parameter(torch.zeros(DIM)))
            self.G_mean_fn = lambda x, t: self.weight * x

        def forward(self, x, t=None):
            return Independent(Normal(self.G_mean_fn(x, t),
                torch.sqrt(torch.diag(V))), 1)

    class p_0_dist_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_0 = mean_0

        def forward(self):
            return Independent(Normal(mean_0, 1.0), 1)

    F_fn = F_Module().to(device)
    F_fn.weight.data = torch.diag(F)
    G_fn = G_Module().to(device)
    G_fn.weight.data = torch.diag(G)
    p_0_dist = p_0_dist_module().to(device)

    def cond_q_mean_net_constructor():
        class MeanNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter('weight', nn.Parameter(torch.randn(DIM)))
                self.register_parameter('bias', nn.Parameter(torch.randn(DIM)))
            def forward(self, x):
                return self.weight * x + self.bias
        out = MeanNet()
        return out

    if cfg.model_training.func_type == 'Vx_t':

        sigma = cfg.model_training.KRR_sigma
        lam = cfg.model_training.KRR_lambda

        def KRR_constructor():
            return models.KernelRidgeRegressor(models.MaternKernel(
                sigma=sigma, lam=lam, train_sigma=True, train_lam=False)).to(device)

        model = models.Vx_t_phi_t_Model(
            device, DIM, DIM,
            torch.zeros(DIM, device=device), torch.ones(DIM, device=device),
            cond_q_mean_net_constructor, torch.ones(DIM, device=device),
            F_fn, G_fn, p_0_dist, cfg.model_training.phi_t_init_method,
            cfg.model_training.window_size,
            KRR_constructor, cfg.model_training.KRR_init_sigma_median,
            cfg.model_training.approx_decay,
            cfg.model_training.approx_with_filter,
            cfg.model_training.window_size + 1
        )
    else:
        raise ValueError('Unknown func_type type')

    filter_means = []
    filter_stds = []
    x_Tm1_means = []
    x_Tm1_covs = []
    joint_kls = []
    Z_losses = []
    times = []

    for T in tqdm(range(0, cfg.data.num_data)):

        start_time = time.time()

        if T == 0 or T == 1:
            decay = cfg.model_training.initial_phi_decay
            inner_iters = cfg.model_training.initial_phi_iters
            lr = cfg.model_training.initial_phi_lr
        else:
            decay = cfg.model_training.phi_decay
            inner_iters = cfg.model_training.phi_iters
            lr = cfg.model_training.phi_lr

        model.advance_timestep(y[T, :])
        model_phi_t_optim = torch.optim.Adam(model.get_phi_T_params(), lr=lr)
        phi_lr_decay = torch.optim.lr_scheduler.StepLR(model_phi_t_optim,
            1, decay)
        

        for k in range(inner_iters):
            model_phi_t_optim.zero_grad()
            model.populate_phi_grads(y, cfg.model_training.phi_minibatch_size)
            model_phi_t_optim.step()
            filter_means.append(model.q_t_mean_list[T].clone().detach().numpy())
            filter_stds.append(np.exp(model.q_t_log_std_list[T].detach().numpy()))
            if T > 0:
                joint_kls.append(utils.KL_between_q_and_p_linear_back_q(
                    model.q_t_mean_list[T].detach().numpy(),
                    model.cond_q_t_mean_net_list[T].bias.detach().numpy(),
                    torch.diag(model.cond_q_t_mean_net_list[T].weight).detach().numpy(),
                    torch.exp(2*model.q_t_log_std_list[T]).detach().numpy(),
                    torch.exp(2*model.cond_q_t_log_std_list[T]).detach().numpy(),
                    kalman_xs[T, :], kalman_Ps[T, :, :],
                    kalman_xs[T-1, :], kalman_Ps[T-1, :],
                    F.detach().numpy(), U.detach().numpy()
                ))
            phi_lr_decay.step()

        model.update_V_t(y, cfg.model_training.V_batch_size)
        Vx_optim = torch.optim.Adam(model.get_V_t_params(), lr=cfg.model_training.V_lr)
        for k in range(cfg.model_training.V_iters):
            Vx_optim.zero_grad()
            V_loss, _, _ = model.V_t_loss(y, cfg.model_training.V_minibatch_size)
            V_loss.backward()
            Vx_optim.step()

        end_time = time.time()

        # Logging
        # filter_means.append(model.q_t_mean_list[T].detach().numpy())
        # filter_stds.append(np.exp(model.q_t_log_std_list[T].detach().numpy()))

        times.append(end_time - start_time)


        if (T % (round(max(cfg.data.num_data, 10) / 10)) == 0) or (T == cfg.data.num_data - 1):
            save_np('filter_means.npy', np.array(filter_means))
            save_np('filter_stds.npy', np.array(filter_stds))
            save_np('joint_kls.npy', np.array(joint_kls))
            save_np('times.npy', np.array(times))
            plt.plot(joint_kls)
            plt.yscale('log')
            plt.show()


if __name__ == "__main__":
    main()