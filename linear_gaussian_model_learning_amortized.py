import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from core.data_generation import GaussianHMM, construct_HMM_matrices
from torch.distributions import Independent, Normal, MultivariateNormal
from tqdm import tqdm
import core.amortised_models as amortised_models
import core.nonamortised_models as nonamortised_models
import core.utils as utils
import math
import subprocess
import hydra
import os
import time

def save_np(name, x):
    np.save(name, x)

@hydra.main(config_path='conf', config_name="linearGaussianModelLearningAmortized")
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

    DIM = cfg.data.dim
    xdim, ydim = DIM, DIM

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

        save_np('x_data.npy', x_np)
        save_np('y_data.npy', y_np)
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

        x_np = np.load(path_to_data + 'x_data.npy')
        y_np = np.load(path_to_data + 'y_data.npy')

    F_init, G_init, _, _ = construct_HMM_matrices(dim=DIM,
                                            F_eigvals=np.random.uniform(
                                                cfg.data.F_min_eigval,
                                                cfg.data.F_max_eigval, (DIM)),
                                            G_eigvals=np.random.uniform(
                                                cfg.data.G_min_eigval,
                                                cfg.data.G_max_eigval, (DIM)),
                                            U_std=cfg.data.U_std,
                                            V_std=cfg.data.V_std,
                                            diag=cfg.data.diagFG)

    F_init = torch.from_numpy(F_init).float().to(device)
    G_init = torch.from_numpy(G_init).float().to(device)


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
            self.G_cov_fn = lambda x, t: V

        def forward(self, x, t=None):
            return Independent(Normal(self.G_mean_fn(x, t),
                                      torch.sqrt(torch.diag(V))), 1)

    class p_0_dist_module(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean_0 = mean_0
            self.cov_0 = torch.eye(DIM).to(device)

        def forward(self):
            return Independent(Normal(mean_0, 1.0), 1)

    F_fn = F_Module().to(device)
    G_fn = G_Module().to(device)
    p_0_dist = p_0_dist_module().to(device)

    F_fn.weight.data = torch.diag(F_init).data
    G_fn.weight.data = torch.diag(G_init).data

    EKF = nonamortised_models.ExtendedKalmanFilter(device, DIM, DIM, F_fn, G_fn, p_0_dist)
    for t in range(y_np.shape[0]):
        EKF.advance_timestep(y[t, :])
        EKF.update(y[t, :])
    kalman_xs = torch.stack(EKF.q_t_mean_list)
    kalman_Ps = torch.stack(EKF.q_t_cov_list)
    kalman_xs_tm1 = torch.stack(EKF.q_tm1_mean_list)
    kalman_Ps_tm1 = torch.stack(EKF.q_tm1_cov_list)

    kalman_xs_np = kalman_xs.detach().cpu().numpy()
    kalman_Ps_np = kalman_Ps.detach().cpu().numpy()
    kalman_xs_tm1_np = kalman_xs_tm1.detach().cpu().numpy()
    kalman_Ps_tm1_np = kalman_Ps_tm1.detach().cpu().numpy()

    save_np('kalman_xs.npy', kalman_xs_np)
    save_np('kalman_Ps.npy', kalman_Ps_np)
    save_np('kalman_xs_tm1.npy', kalman_xs_tm1_np)
    save_np('kalman_Ps_tm1.npy', kalman_Ps_tm1_np)

    q_rnn_hidden_dim = cfg.phi_training.q_rnn_hidden_dim
    q_hidden_dims = cfg.phi_training.q_hidden_dims
    q_rnn_num_layers = cfg.phi_training.q_rnn_num_layers
    q_rnn_constructor = lambda: nn.RNN(ydim, q_rnn_hidden_dim, q_rnn_num_layers).to(device)

    def net_constructor(net_dims):
        modules = []
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(net_dims[-2], net_dims[-1]))
        return nn.Sequential(*modules).to(device)

    class Normal_Net(nn.Module):
        def __init__(self, mlp_module):
            super().__init__()
            self.mlp_module = mlp_module

        def forward(self, x):
            x = self.mlp_module(x)
            mu, logsigma = torch.chunk(x, 2, dim=-1)
            sig = functional.softplus(logsigma)
            return mu, sig

    q_t_net_constructor = lambda: Normal_Net(
        net_constructor(
            [q_rnn_hidden_dim] + list(q_hidden_dims) + [2 * xdim]
        )
    ).to(device)
    cond_q_t_net_constructor = lambda: Normal_Net(
        net_constructor(
            [xdim + q_rnn_hidden_dim] + list(q_hidden_dims) + [2 * xdim]
        )
    ).to(device)


    if cfg.phi_training.func_type == 'separate_time_kernel_amortised':
        sigma = cfg.phi_training.KRR_sigma
        lam = cfg.phi_training.KRR_lambda
        train_sigma = cfg.phi_training.KRR_train_sigma
        train_lam = cfg.phi_training.KRR_train_lam


        def KRR_constructor():
            return nonamortised_models.KernelRidgeRegressor(nonamortised_models.MaternKernel(
                sigma=sigma, lam=lam, train_sigma=train_sigma, train_lam=train_lam),
                centre_elbo=cfg.phi_training.KRR_centre_elbo).to(device)

        class KRRWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.KRR = KRR_constructor()

            def forward(self, x, t):
                return self.KRR(x)

            def fit(self, train_input, *train_outputs):
                self.KRR.fit(train_input, *train_outputs)

            @property
            def kernel(self):
                return self.KRR.kernel

        model = amortised_models.SeparateTimeKernelAmortisedModel(
            device, xdim, ydim, q_rnn_constructor, q_t_net_constructor,
            cond_q_t_net_constructor, F_fn,
            G_fn, p_0_dist, cfg.phi_training.rnn_window_size,
            KRRWrapper, cfg.phi_training.approx_with_filter,
            'ST', 'FG'
        )
    else:
        raise NotImplementedError

    rmle = nonamortised_models.LinearRMLEDiagFG(np.zeros((DIM,1)), np.eye(DIM),
        F_init.detach().cpu().numpy().copy(),
        G_init.detach().cpu().numpy().copy(),
        U.cpu().detach().numpy().copy(), V.cpu().detach().numpy().copy(),
        cfg.theta_training.theta_lr,
        'FG')

    filter_means = []
    filter_covs = []
    x_Tm1_means = []
    x_Tm1_covs = []

    approx_func_losses = []

    rnn_hidden_states = []
    back_window_means = []
    back_window_xs = []

    means_during_training = []
    back_means_during_training = []
    covs_during_training = []

    Fs = []
    Gs = []
    rmle_Fs = []
    rmle_Gs = []
    rmle_xs = []
    rmle_Ps = []


    theta_optim = torch.optim.SGD([*F_fn.parameters(), *G_fn.parameters()],
        lr=cfg.theta_training.theta_lr
    )
    theta_decay = utils.make_lr_scheduler(theta_optim,
        cfg.theta_training.theta_lr_decay_type,
        cfg.theta_training.theta_lr_num_steps_oom_drop
    )

    for T in tqdm(range(0, cfg.data.num_data)):
        model.advance_timestep(y[T, :])

        phi_optim = torch.optim.Adam(model.get_phi_params(), lr=cfg.phi_training.phi_lr)
        phi_decay = utils.make_lr_scheduler(phi_optim, cfg.phi_training.phi_lr_decay_type,
                                            getattr(cfg.phi_training, "phi_lr_num_steps_oom_drop", None))
        means_during_training_inner = []
        back_means_during_training_inner = []
        covs_during_training_inner = []

        for k in range(cfg.phi_training.phi_iters):
            phi_optim.zero_grad()
            theta_optim.zero_grad()

            loss = model.populate_grads(y, cfg.phi_training.phi_minibatch_size)

            if cfg.phi_training.phi_grad_clip_norm is not None:
                phi_grad_norm = nn.utils.clip_grad_norm_(
                    model.get_phi_params(), cfg.phi_training.phi_grad_clip_norm)
            phi_optim.step()
            phi_decay.step()

            filter_stats = model.return_summary_stats(y)
            means_during_training_inner.append(filter_stats[0].detach().cpu().numpy())
            covs_during_training_inner.append(torch.diag(filter_stats[1]).detach().cpu().numpy())
            if T > 0:
                back_stats = model.return_summary_stats(y, t=T-1, num_samples=1028)
                back_means_during_training_inner.append(back_stats[0].detach().cpu().numpy())

            if k == cfg.phi_training.phi_iters - 1 and\
                T > cfg.theta_training.theta_updates_start_T:
                theta_optim.step()

                rmle.step_size = theta_decay.state_dict()['_last_lr'][0]
                rmle.advance_timestep(y[T, :].detach().numpy().copy().reshape((DIM,1)))
                rmle_Gs.append(rmle.G.copy())
                rmle_Fs.append(rmle.F.copy())
                Fs.append(F_fn.weight.clone().detach().numpy())
                Gs.append(G_fn.weight.clone().detach().numpy())

                theta_decay.step()

        means_during_training.append(np.array(means_during_training_inner))
        covs_during_training.append(np.array(covs_during_training_inner))
        if T > 0:
            back_means_during_training.append(np.array(back_means_during_training_inner))


        model.detach_rnn_hist_hn(y)
        

        approx_func_optim = torch.optim.Adam(model.get_func_t_params(),
            lr=cfg.phi_training.approx_lr)

        kernel_inputs, *kernel_targets = model.generate_training_dataset(
            y, cfg.phi_training.kernel_batch_size, cfg.phi_training.disperse_temp
        )
        model.update_func_t(kernel_inputs, *kernel_targets)

        if cfg.phi_training.KRR_train_sigma or cfg.phi_training.KRR_train_lam:
            kernel_inputs, *kernel_targets = model.generate_training_dataset(
                y, cfg.phi_training.kernel_train_set_size
            )

            approx_func_losses_T = []

            for i in range(cfg.phi_training.approx_iters):
                idx = np.random.choice(np.arange(kernel_inputs.shape[0]),
                                        (cfg.phi_training.approx_minibatch_size,),
                                        replace=False)
                approx_func_optim.zero_grad()

                approx_func_loss, _, approx_func_loss_list = \
                    model.func_t_loss(
                        kernel_inputs[idx, :],
                        *[kernel_target[idx, :] for kernel_target in kernel_targets]
                )

                approx_func_loss.backward()
                approx_func_optim.step()

                approx_func_losses_T.append(approx_func_loss.item())

            approx_func_losses.append(np.array(approx_func_losses_T))                



        # -------------- Logging ------------
        filter_stats = model.return_summary_stats(y)
        filter_means.append(filter_stats[0].detach().cpu().numpy())
        filter_covs.append(filter_stats[1].detach().cpu().numpy())

        rmle_xs.append(rmle.x.copy())
        rmle_Ps.append(rmle.P.copy())

        if T > 0:
            x_Tm1_stats = model.return_summary_stats(y, t=T - 1, num_samples=1000)
            x_Tm1_means.append(x_Tm1_stats[0].detach().cpu().numpy())
            x_Tm1_covs.append(x_Tm1_stats[1].detach().cpu().numpy())

        rnn_hidden_states.append(model.rnn_forward(y)[-1, 0, :].detach().cpu().numpy())

        if T > cfg.phi_training.back_log_length + 1:
            model.rnn_forward(y)
            N = 32
            x_t_samples, _ = model.sample_q_T(N)
            x_window_samples, _ = model.sample_q_t_cond_T(x_t_samples,
                cfg.phi_training.back_log_length)
            all_window_means = np.zeros((cfg.phi_training.back_log_length+1, xdim))
            all_window_means[-1, :] = np.mean(x_t_samples.detach().cpu().numpy(), axis=0)
            for i in range(cfg.phi_training.back_log_length):
                all_window_means[i, :] = \
                    np.mean(x_window_samples[i].detach().cpu().numpy(), axis=0)
            back_window_means.append(all_window_means)

            back_window_x = np.zeros((cfg.phi_training.back_log_length+1, xdim))
            for i in range(cfg.phi_training.back_log_length+1):
                back_window_x[i, :] = kalman_xs[T+i-cfg.phi_training.back_log_length, :].detach().cpu().numpy()
            back_window_xs.append(back_window_x)



        if (T % (round(max(cfg.data.num_data, cfg.phi_training.num_times_save_data)\
            / cfg.phi_training.num_times_save_data)) == 0) or\
            (T == cfg.data.num_data - 1):

            save_np('Gs.npy', np.array(Gs))
            save_np('Fs.npy', np.array(Fs))
            save_np('rmle_Gs.npy', np.array(rmle_Gs))
            save_np('rmle_Fs.npy', np.array(rmle_Fs))
            save_np('rmle_xs.npy', np.array(rmle_xs))
            save_np('rmle_Ps.npy', np.array(rmle_Ps))
            save_np('filter_means.npy', np.array(filter_means))
            save_np('filter_covs.npy', np.array(filter_covs))
            save_np('x_Tm1_means.npy', np.array(x_Tm1_means))
            save_np('x_Tm1_covs.npy', np.array(x_Tm1_covs))
            save_np('approx_func_losses.npy', np.array(approx_func_losses))
            save_np('rnn_hidden_states.npy', np.array(rnn_hidden_states))
            save_np('back_window_means.npy', np.array(back_window_means))
            save_np('back_window_xs.npy', np.array(back_window_xs))
            save_np('means_during_training.npy', np.array(means_during_training))
            save_np('back_means_during_training.npy', np.array(back_means_during_training))
            save_np('covs_during_training.npy', np.array(covs_during_training))


if __name__ == "__main__":
    main()