from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import core.utils as utils
from torch.distributions import MultivariateNormal, Independent, Normal, Categorical
from core.utils import gaussian_posterior, sample_cov


class NonAmortizedModelBase(nn.Module):
    """
        A generic base class for nonlinear models to be inherited from.
        Contains nonlinear model in the form
            x_t ~ F_fn (x_{t-1})
            y_t ~ G_fn (x_t)
        F, G are nn.Modules which return a distribution and can potentially contain learnable theta parameters
        The q lists contain nonamortized variational posteriors
            q_t(x_t) = N(x_t; q_T_mean, diag(q_T_std)^2)
            k_t(x_{t-1} | x_t) = N(x_{t-1}; cond_q_t_mean_net(x_t), diag(cond_q_t_std)^2)
            These will both be factorized Gaussians.
        cond_q_t_mean_net: Linear, MLP, etc: (xdim) -> (xdim)
    """

    def __init__(self, device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store=None):
        super().__init__()
        self.T = -1

        self.device = device
        self.xdim = xdim
        self.ydim = ydim

        # Inference model
        self.q_0_mean = q_0_mean
        self.q_0_log_std = q_0_log_std
        self.cond_q_mean_net_constructor = cond_q_mean_net_constructor
        self.cond_q_0_log_std = cond_q_0_log_std

        time_store_size = window_size + 1 if num_params_to_store is None else num_params_to_store
        self.q_t_mean_list = utils.TimeStore(None, time_store_size, 'ParameterList')
        self.q_t_log_std_list = utils.TimeStore(None, time_store_size, 'ParameterList')
        self.cond_q_t_mean_net_list = utils.TimeStore(None, time_store_size, 'ModuleList')
        self.cond_q_t_log_std_list = utils.TimeStore(None, time_store_size, 'ParameterList')

        # Generative model
        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist

        self.phi_t_init_method = phi_t_init_method
        self.window_size = window_size

    def advance_timestep(self, y_T):
        # Prepare the model when new data arrives, should be called for every T >= 0
        # The most recent window_size modules are optimized
        self.T += 1

        if self.phi_t_init_method == "last":
            if self.T == 0:
                self.q_t_mean_list.append(nn.Parameter(self.q_0_mean))
                self.q_t_log_std_list.append(nn.Parameter(self.q_0_log_std))
            else:
                self.q_t_mean_list.append(nn.Parameter(self.q_t_mean_list[self.T - 1].clone().detach()))
                self.q_t_log_std_list.append(nn.Parameter(self.q_t_log_std_list[self.T - 1].clone().detach()))

        elif self.phi_t_init_method == "pred":
            if self.T == 0:
                self.q_t_mean_list.append(nn.Parameter(self.q_0_mean))
                self.q_t_log_std_list.append(nn.Parameter(self.q_0_log_std))
            else:
                self.q_t_mean_list.append(nn.Parameter(self.F_fn.F_mean_fn(self.q_t_mean_list[self.T-1], self.T-1)
                                                       .clone().detach()))
                test_x = self.q_t_mean_list[self.T - 1].detach().clone().requires_grad_()
                F_jac = torch.autograd.functional.jacobian(partial(self.F_fn.F_mean_fn, t=self.T-1), test_x)
                F_cov = self.F_fn.F_cov_fn(self.q_t_mean_list[self.T-1], self.T-1)
                pred_cov = F_jac @ torch.diag((self.q_t_log_std_list[self.T - 1] * 2).exp()) @ F_jac.t() + F_cov

                self.q_t_log_std_list.append(nn.Parameter(pred_cov.detach().diag().log() / 2))

        elif self.phi_t_init_method == "EKF":
            if self.T == 0:
                pred_mean = self.p_0_dist.mean_0
                pred_cov = self.p_0_dist.cov_0
            else:
                pred_mean = self.F_fn.F_mean_fn(self.q_t_mean_list[self.T-1], self.T-1)
                test_x = self.q_t_mean_list[self.T - 1].detach().clone().requires_grad_()
                F_jac = torch.autograd.functional.jacobian(partial(self.F_fn.F_mean_fn, t=self.T-1), test_x)
                F_cov = self.F_fn.F_cov_fn(self.q_t_mean_list[self.T - 1], self.T - 1)
                pred_cov = F_jac @ torch.diag((self.q_t_log_std_list[self.T - 1] * 2).exp()) @ F_jac.t() + F_cov

            test_x = pred_mean.detach().clone().requires_grad_()
            G_jac = torch.autograd.functional.jacobian(partial(self.G_fn.G_mean_fn, t=self.T), test_x)
            G_cov = self.G_fn.G_cov_fn(pred_mean, self.T)
            q_T_mean, q_T_cov = gaussian_posterior(y_T, pred_mean, pred_cov, G_jac, G_cov,
                                                   G_fn=partial(self.G_fn.G_mean_fn, t=self.T))

            self.q_t_mean_list.append(nn.Parameter(q_T_mean.detach()))
            self.q_t_log_std_list.append(nn.Parameter(q_T_cov.detach().diag().log() / 2))
        else:
            assert False, "Invalid phi_t_init_method"

        self.cond_q_t_mean_net_list.append(self.cond_q_mean_net_constructor())
        if self.T == 0:
            self.cond_q_t_log_std_list.append(nn.Parameter(self.cond_q_0_log_std))
        else:
            self.cond_q_t_mean_net_list[self.T].load_state_dict(self.cond_q_t_mean_net_list[self.T - 1].state_dict())
            self.cond_q_t_log_std_list.append(nn.Parameter(self.cond_q_t_log_std_list[self.T - 1].clone().detach()))

        if self.T >= self.window_size:
            self.q_t_mean_list[self.T - self.window_size].requires_grad_(False)
            self.q_t_log_std_list[self.T - self.window_size].requires_grad_(False)
            self.cond_q_t_mean_net_list[self.T - self.window_size].requires_grad_(False)
            self.cond_q_t_log_std_list[self.T - self.window_size].requires_grad_(False)

    def get_phi_T_params(self):
        params = []
        params = params + [self.q_t_mean_list[self.T], self.q_t_log_std_list[self.T],
                            *self.cond_q_t_mean_net_list[self.T].parameters(), self.cond_q_t_log_std_list[self.T]]
        for t in range(max(0, self.T - self.window_size + 1), self.T):
            params = params + [*self.cond_q_t_mean_net_list[t].parameters(), self.cond_q_t_log_std_list[t]]
        return params

    def sample_q_T(self, num_samples, detach_x=False, T=None):
        if T is None:
            T = self.T
        assert T <= self.T
        q_T_mean = self.q_t_mean_list[T].expand(num_samples, self.xdim)
        q_T_std = self.q_t_log_std_list[T].exp().expand(num_samples, self.xdim)
        q_T_stats = [q_T_mean, q_T_std]

        eps_x_T = torch.randn(num_samples, self.xdim).to(self.device)
        x_T = q_T_mean + q_T_std * eps_x_T

        if detach_x:
            x_T = x_T.detach()

        return x_T, q_T_stats

    def sample_q_t_cond_T(self, x_T, num_steps_back, detach_x=False, T=None):
        if T is None:
            T = self.T
        assert T <= self.T
        num_samples = x_T.shape[0]
        if num_steps_back > T:
            print("Warning: num_steps_back > T")
        num_steps_back = min(num_steps_back, T)
        x_t_samples = [None] * num_steps_back
        all_cond_q_t_means = [None] * num_steps_back
        all_cond_q_t_stds = [None] * num_steps_back

        for t in range(T - 1, T - 1 - num_steps_back, -1):
            if t == T - 1:
                x_tp1 = x_T
            else:
                x_tp1 = x_t_samples[t - T + num_steps_back + 1]

            # We do not use the 0th entry in the conditional lists, so that at time t we are learning the
            # t-th entry in all the lists (q(x_t) and q(x_tm1|x_t))
            cond_q_t_mean = self.cond_q_t_mean_net_list[t + 1](x_tp1)
            cond_q_t_std = self.cond_q_t_log_std_list[t + 1].exp().expand(num_samples, self.xdim)

            eps_x_t = torch.randn(num_samples, self.xdim).to(self.device)
            x_t = cond_q_t_mean + cond_q_t_std * eps_x_t

            if detach_x:
                x_t = x_t.detach()

            x_t_samples[t - T + num_steps_back] = x_t
            all_cond_q_t_means[t - T + num_steps_back] = cond_q_t_mean
            all_cond_q_t_stds[t - T + num_steps_back] = cond_q_t_std

        all_cond_q_t_stats = [[mean, std] for mean, std in zip(all_cond_q_t_means, all_cond_q_t_stds)]

        return x_t_samples, all_cond_q_t_stats

    def sample_joint_q_t(self, num_samples, num_steps_back, detach_x=False, T=None):
        """
            Sample num_samples from
            q(x_T) \prod_{t= T - num_steps_back}^{T-1} k(x_t | x_{t+1})
            If detach_x is true then all x samples are detached
        """
        if T is None:
            T = self.T
        assert T <= self.T
        if num_steps_back > T:
            print("Warning: num_steps_back > T")
        num_steps_back = min(num_steps_back, T)

        x_T_samples, q_T_stats = self.sample_q_T(num_samples, detach_x=detach_x, T=T)

        x_t_samples, all_cond_q_t_stats = self.sample_q_t_cond_T(x_T_samples, num_steps_back, detach_x=detach_x, T=T)

        return x_t_samples + [x_T_samples], all_cond_q_t_stats + [q_T_stats]

    def compute_log_p_t(self, x_t, y_t, x_tm1=None, t=None):
        """
            Compute log p(x_t | x_{t-1}) and log p(y_t | x_t)
            t is set to self.T if not specified
        """
        if t is None:
            t = self.T
        if t == 0:
            log_p_x_t = self.p_0_dist().log_prob(x_t).unsqueeze(1)
        else:
            log_p_x_t = self.F_fn(x_tm1, t-1).log_prob(x_t).unsqueeze(1)
        log_p_y_t = self.G_fn(x_t, t).log_prob(y_t).unsqueeze(1)
        return {"log_p_x_t": log_p_x_t, "log_p_y_t": log_p_y_t}

    def compute_log_q_t(self, x_t, *q_t_stats):
        """
            Compute log q(x_t | x_{t+1}) (independent Gaussian inference model)
        """
        assert len(q_t_stats) == 2
        return Independent(Normal(*q_t_stats), 1).log_prob(x_t).unsqueeze(1)

    def compute_r_t(self, x_t, y_t, *q_tm1_stats, x_tm1=None, t=None):
        if t is None:
            t = self.T
        if t == 0:
            log_p_t = self.compute_log_p_t(x_t, y_t, t=0)
            log_p_x_t, log_p_y_t = log_p_t["log_p_x_t"], log_p_t["log_p_y_t"]
            r_t = log_p_x_t + log_p_y_t
        else:
            log_p_t = self.compute_log_p_t(x_t, y_t, x_tm1, t=t)
            log_p_x_t, log_p_y_t = log_p_t["log_p_x_t"], log_p_t["log_p_y_t"]
            log_q_x_tm1 = self.compute_log_q_t(x_tm1, *q_tm1_stats)
            r_t = log_p_x_t + log_p_y_t - log_q_x_tm1
        return r_t

    def sample_and_compute_r_t(self, y_t, num_samples, detach_x=False, t=None, disperse_temp=1):
        if t is None:
            t = self.T - self.window_size + 1
        x_t, q_t_stats = self.sample_joint_q_t(num_samples, self.T - t, detach_x=detach_x)
        x_t, q_t_stats = x_t[0], q_t_stats[0]
        x_t_dispersed = x_t + torch.randn_like(x_t) * np.sqrt(1/disperse_temp - 1) * q_t_stats[1]
        if t == 0:
            x_tm1 = None
            r_t = self.compute_r_t(x_t_dispersed, y_t, t=0)
        else:
            x_tm1, cond_q_tm1_stats = self.sample_q_t_cond_T(x_t_dispersed, 1, detach_x=detach_x, T=t)
            x_tm1, cond_q_tm1_stats = x_tm1[0], cond_q_tm1_stats[0]
            r_t = self.compute_r_t(x_t_dispersed, y_t, *cond_q_tm1_stats, x_tm1=x_tm1, t=t)
        return {"x_tm1": x_tm1, "x_t": x_t_dispersed, "r_t": r_t}

    def sample_and_compute_joint_r_t(self, y, num_samples, window_size, detach_x=False, only_return_first_r=False,
                                     T=None):
        """
            Sample from q(x_T) k(x_{(T-window_size):(T-1)} | x_T) and compute r_{(T-window_size+1):T}
            as well as other useful quantities
        """
        if T is None:
            T = self.T
        assert T <= self.T
        if window_size > T + 1:
            print("Warning: window_size > T + 1")
        window_size = min(window_size, T + 1)
        num_steps_back = min(window_size, T)
        x_samples, all_q_stats = self.sample_joint_q_t(num_samples, num_steps_back, detach_x=detach_x, T=T)
        ts = np.arange(T - window_size + 1, T + 1)  # correspond to r_t, length window_size
        x_t_idx = np.arange(T + 1) if window_size == T + 1 else \
            np.arange(1, window_size + 1)  # indices of x_t in x_samples to compute r_t, length window_size
        r_values = []  # r_t, length window_size

        for i, t in zip(x_t_idx, ts):
            x_t, q_t_stats = x_samples[i], all_q_stats[i]
            x_tm1, q_tm1_stats = (None, []) if t == 0 else (x_samples[i - 1], all_q_stats[i - 1])
            r_t = self.compute_r_t(x_t, y[t, :], *q_tm1_stats, x_tm1=x_tm1, t=t)
            r_values.append(r_t)

            if only_return_first_r:
                break

        log_q_x_T = self.compute_log_q_t(x_samples[-1], *all_q_stats[-1])

        return {"x_samples": x_samples, "all_q_stats": all_q_stats,
                "r_values": r_values, "log_q_x_T": log_q_x_T}

    def generate_data(self, T):
        # Generates hidden states and observations up to time T
        x = torch.zeros((T, self.xdim)).to(self.device)
        y = torch.zeros((T, self.ydim)).to(self.device)

        x[0, :] = self.p_0_dist().sample()

        for t in range(T):
            y_t = self.G_fn(x[t, :], t).sample()
            x_tp1 = self.F_fn(x[t, :], t).sample()

            y[t, :] = y_t
            if t < T-1:
                x[t+1, :] = x_tp1

        return x, y

    def compute_elbo_loss(self, y, num_samples):
        all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, self.T + 1)

        r_values = all_r_results["r_values"]
        sum_r = sum(r_values)
        log_q_x_T = all_r_results["log_q_x_T"]

        loss = - (sum_r - log_q_x_T).mean()
        return loss

    def return_summary_stats(self, y, t=None, num_samples=None):
        if t is None:
            t = self.T
        if t == self.T:
            x_t_mean = self.q_t_mean_list[self.T].detach().clone()
            x_t_cov = torch.diag(torch.exp(self.q_t_log_std_list[self.T].detach().clone() * 2))
        elif t == self.T - 1:
            joint_x_samples, _ = self.sample_joint_q_t(num_samples, 1)
            x_Tm1_samples = joint_x_samples[0]
            x_t_mean = x_Tm1_samples.mean(0).detach().clone()
            x_t_cov = sample_cov(x_Tm1_samples).detach().clone()
        return x_t_mean, x_t_cov


class Vx_t_phi_t_Model(NonAmortizedModelBase):
    def __init__(self, device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, G_fn, p_0_dist, phi_t_init_method, window_size,
                 V_func_constructor, init_sigma_median, approx_decay, approx_with_filter,
                 num_params_to_store=None):
        super().__init__(device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                         F_fn, G_fn, p_0_dist, phi_t_init_method, window_size,
                         num_params_to_store)
        self.V_func_constructor = V_func_constructor
        self.init_sigma_median = init_sigma_median
        # self.V_func_t = self.V_func_constructor()
        self.approx_decay = approx_decay
        self.approx_with_filter = approx_with_filter

    def advance_timestep(self, y_T):
        super().advance_timestep(y_T)
        if self.T == self.window_size - 1:
            self.V_func_t = self.V_func_constructor()

        elif self.T > self.window_size - 1:
            self.V_func_tm1 = self.V_func_t
            self.V_func_t = self.V_func_constructor()
            self.V_func_t.load_state_dict(self.V_func_tm1.state_dict())
            self.V_func_tm1.requires_grad_(False)
            self.V_func_tm1.eval()

    def update_V_t(self, y, num_samples, t=None, disperse_temp=1):
        """
            Updates V_t using q(x_{tm1:T})
        """
        if t is None:
            t = self.T - self.window_size + 1  # By default, we update V_t at time T (V_T when window_size=1)
        if t >= 0:
            # self.zero_grad()

            r_results = self.sample_and_compute_r_t(y[t, :], num_samples, t=t, disperse_temp=disperse_temp)
            x_t = r_results["x_t"]
            r_t = r_results["r_t"]

            if t == 0:
                if self.approx_with_filter:
                    log_q_t = self.compute_log_q_t(x_t, self.q_t_mean_list[t].detach(),
                                                   self.q_t_log_std_list[t].detach().exp())
                    r_t -= log_q_t
                dr_t_x_t = torch.autograd.grad(r_t.sum(), x_t, retain_graph=True)[0]
                Vx_tm1_dx_tm1_x_t = torch.zeros_like(dr_t_x_t)
                V_tm1 = torch.zeros_like(r_t)

            else:
                x_tm1 = r_results["x_tm1"]
                if self.approx_with_filter:
                    log_q_t = self.compute_log_q_t(x_t, self.q_t_mean_list[t].detach(),
                                                   self.q_t_log_std_list[t].detach().exp())
                    log_q_tm1 = self.compute_log_q_t(x_tm1, self.q_t_mean_list[t-1].detach(),
                                                     self.q_t_log_std_list[t-1].detach().exp())
                    r_t += (log_q_tm1 - log_q_t)
                dr_t_x_t = torch.autograd.grad(r_t.sum(), x_t, retain_graph=True)[0]
                with torch.no_grad():
                    Vx_tm1, V_tm1 = self.V_func_tm1(x_tm1)
                Vx_tm1_dx_tm1_x_t = torch.autograd.grad((Vx_tm1 * x_tm1).sum(),
                                                        x_t, retain_graph=True)[0]

            # Fit new V_func
            if self.init_sigma_median and t == 0:
                self.V_func_t.kernel.log_sigma.data = torch.tensor(
                    np.log(utils.estimate_median_distance(x_t)).astype(float)).to(self.device)
                print("Update bandwidth to ", self.V_func_t.kernel.log_sigma.exp().item())

            self.V_func_t.fit(x_t.detach(),
                              (dr_t_x_t + self.approx_decay * Vx_tm1_dx_tm1_x_t).detach(),
                              (r_t + self.approx_decay * V_tm1).detach())
        else:
            print("t < 0, V_t not updated")

    def V_t_loss(self, y, num_samples, t=None, disperse_temp=1):
        if t is None:
            t = self.T - self.window_size + 1  # By default, we update V_t at time T (V_T when window_size=1)
        if t >= 0:
            self.V_func_t.train()
            # self.zero_grad()

            r_results = self.sample_and_compute_r_t(y[t, :], num_samples, t=t, disperse_temp=disperse_temp)
            x_t = r_results["x_t"]
            r_t = r_results["r_t"]

            if t == 0:
                if self.approx_with_filter:
                    log_q_t = self.compute_log_q_t(x_t, self.q_t_mean_list[t].detach(),
                                                   self.q_t_log_std_list[t].detach().exp())
                    r_t -= log_q_t
                dr_t_x_t = torch.autograd.grad(r_t.sum(), x_t, retain_graph=True)[0]
                Vx_tm1_dx_tm1_x_t = torch.zeros_like(dr_t_x_t)
                V_tm1 = torch.zeros_like(r_t)

            else:
                x_tm1 = r_results["x_tm1"]
                if self.approx_with_filter:
                    log_q_t = self.compute_log_q_t(x_t, self.q_t_mean_list[t].detach(),
                                                   self.q_t_log_std_list[t].detach().exp())
                    log_q_tm1 = self.compute_log_q_t(x_tm1, self.q_t_mean_list[t - 1].detach(),
                                                     self.q_t_log_std_list[t - 1].detach().exp())
                    r_t += (log_q_tm1 - log_q_t)
                dr_t_x_t = torch.autograd.grad(r_t.sum(), x_t, retain_graph=True)[0]
                with torch.no_grad():
                    Vx_tm1, V_tm1 = self.V_func_tm1(x_tm1)
                Vx_tm1_dx_tm1_x_t = torch.autograd.grad((Vx_tm1 * x_tm1).sum(),
                                                        x_t, retain_graph=True)[0]

            preds = self.V_func_t(x_t.detach())[0]
            targets = (dr_t_x_t + self.approx_decay * Vx_tm1_dx_tm1_x_t).detach()

            # preds = self.V_func_t(x_t.detach())[1]
            # targets = (r_t + self.approx_decay * V_tm1).detach()

            return torch.mean((preds - targets) ** 2), preds, targets

        else:
            print("t < 0, V_t not updated")

    def populate_phi_grads(self, y, num_samples):
        # self.zero_grad()

        all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, min(self.window_size, self.T + 1))

        r_values = all_r_results["r_values"]
        sum_r = sum(r_values)
        log_q_x_T = all_r_results["log_q_x_T"]

        if self.T < self.window_size:
            Vx_tm1_x_tm1 = torch.zeros_like(sum_r)
            V_tm1 = torch.zeros_like(sum_r)

        else:
            x_tm1 = all_r_results["x_samples"][0]
            with torch.no_grad():
                Vx_tm1, V_tm1 = self.V_func_tm1(x_tm1)
            Vx_tm1_x_tm1 = (Vx_tm1 * x_tm1).sum(1, keepdim=True)
            if self.approx_with_filter:
                log_q_x_tm1 = self.compute_log_q_t(x_tm1, self.q_t_mean_list[self.T - self.window_size].detach(),
                                                   self.q_t_log_std_list[self.T - self.window_size].detach().exp())
                sum_r += log_q_x_tm1

        loss = - (sum_r + Vx_tm1_x_tm1 - log_q_x_T - Vx_tm1_x_tm1.detach() + V_tm1.detach()).mean()
        loss.backward()
        return loss

    def get_V_t_params(self):
        return self.V_func_t.parameters()


class Ignore_Past_phi_t_Model(NonAmortizedModelBase):
    """
        This model ignores V/Z computation, i.e. only compute r_{(T-window_size+1):T} and log_q_x_T.
    """

    def __init__(self, device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store=None):
        super().__init__(device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                         F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store)

    def populate_phi_grads(self, y, num_samples):
        # self.zero_grad()

        all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, min(self.window_size, self.T + 1))

        r_values = all_r_results["r_values"]
        sum_r = sum(r_values)
        log_q_x_T = all_r_results["log_q_x_T"]

        loss = - (sum_r - log_q_x_T).mean()
        loss.backward()
        return loss


class JELBO_Model(NonAmortizedModelBase):
    """
        At time step T optimizes the following objective
        E_{q(x_{(T-window_size):T})} [ log q_{T-window_size}(x_{T-window_size}) +
                                       log p(x_{(T-window_size+1):T}, y_{(T-window_size+1):T}|x_{T-window_size}) -
                                       log q_{(T-window_size+1):T}(x_{(T-window_size):T})]
    """

    def __init__(self, device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store=None):

        super().__init__(device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                         F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store)

    def populate_phi_grads(self, y, num_samples):
        # self.zero_grad()

        all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, min(self.window_size, self.T + 1))

        r_values = all_r_results["r_values"]
        sum_r = sum(r_values)
        log_q_x_T = all_r_results["log_q_x_T"]

        if self.T < self.window_size:
            loss = -(sum_r - log_q_x_T).mean()
        else:
            x_tm1 = all_r_results["x_samples"][0]
            log_q_x_tm1 = self.compute_log_q_t(x_tm1, self.q_t_mean_list[self.T - self.window_size].detach(),
                                               self.q_t_log_std_list[self.T - self.window_size].detach().exp())
            loss = - (sum_r - log_q_x_T + log_q_x_tm1).mean()

        loss.backward()
        return loss


class VJF_Model(NonAmortizedModelBase):
    def __init__(self, device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                 F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store=None):
        assert window_size == 1
        super().__init__(device, xdim, ydim, q_0_mean, q_0_log_std, cond_q_mean_net_constructor, cond_q_0_log_std,
                         F_fn, G_fn, p_0_dist, phi_t_init_method, window_size, num_params_to_store)

    def populate_phi_grads(self, y, num_samples):
        # self.zero_grad()

        x_T, q_T_stats = self.sample_q_T(num_samples)
        log_q_x_T = self.compute_log_q_t(x_T, *q_T_stats)
        if self.T == 0:
            log_p_t = self.compute_log_p_t(x_T, y[self.T])

        else:
            x_Tm1, q_Tm1_stats = self.sample_q_T(num_samples, detach_x=True, T=self.T-1)
            log_p_t = self.compute_log_p_t(x_T, y[self.T], x_Tm1)
        log_p_x_t, log_p_y_t = log_p_t["log_p_x_t"], log_p_t["log_p_y_t"]
        loss = - (log_p_x_t + log_p_y_t - log_q_x_T).mean()

        loss.backward()
        return loss

    def compute_elbo_loss(self, y, num_samples):
        elbo_loss = 0
        for T in range(self.T + 1):
            x_T, q_T_stats = self.sample_q_T(num_samples, T=T)
            log_q_x_T = self.compute_log_q_t(x_T, *q_T_stats)
            if T == 0:
                log_p_t = self.compute_log_p_t(x_T, y[T], t=T)

            else:
                x_Tm1, q_Tm1_stats = self.sample_q_T(num_samples, detach_x=True, T=T-1)
                log_p_t = self.compute_log_p_t(x_T, y[T], x_Tm1, t=T)
            log_p_x_t, log_p_y_t = log_p_t["log_p_x_t"], log_p_t["log_p_y_t"]
            loss = - (log_p_x_t + log_p_y_t - log_q_x_T).mean()
            elbo_loss += loss

        return elbo_loss


class KalmanFilter():
    def __init__(self, x_0, P_0, F, G, U, V):
        self.x, self.P, self.F, self.G, self.U, self.V = \
            x_0, P_0, F, G, U, V

    def update(self, y):
        xp = np.dot(self.F, self.x)
        Pp = np.matmul(np.matmul(self.F, self.P), np.transpose(self.F)) + self.U
        S = np.matmul(np.matmul(self.G, Pp), np.transpose(self.G)) + self.V
        K = np.matmul(np.matmul(Pp, np.transpose(self.G)), np.linalg.inv(S))
        z = y - np.dot(self.G, xp)

        self.x = xp + np.dot(K, z)
        self.P = np.matmul(np.eye(self.P.shape[0]) - np.matmul(K, self.G), Pp)


class ExtendedKalmanFilter(nn.Module):
    def __init__(self, device, xdim, ydim, F_fn, G_fn, p_0_dist):
        super().__init__()
        self.T = -1

        self.device = device
        self.xdim = xdim
        self.ydim = ydim

        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist

        self.q_t_mean_list = []
        self.q_t_cov_list = []
        self.q_tm1_mean_list = []
        self.q_tm1_cov_list = []

    def advance_timestep(self, y_T):
        self.T += 1

    def update(self, y_T):
        if self.T == 0:
            pred_mean = self.p_0_dist.mean_0
            pred_cov = self.p_0_dist.cov_0
        else:
            pred_mean = self.F_fn.F_mean_fn(self.q_t_mean_list[self.T - 1], self.T - 1)
            test_x = self.q_t_mean_list[self.T - 1].detach().clone().requires_grad_()
            F_jac = torch.autograd.functional.jacobian(partial(self.F_fn.F_mean_fn, t=self.T - 1), test_x)
            F_cov = self.F_fn.F_cov_fn(self.q_t_mean_list[self.T - 1], self.T - 1)
            pred_cov = F_jac @ self.q_t_cov_list[self.T - 1] @ F_jac.t() + F_cov

        test_x = pred_mean.detach().clone().requires_grad_()
        G_jac = torch.autograd.functional.jacobian(partial(self.G_fn.G_mean_fn, t=self.T), test_x)
        G_cov = self.G_fn.G_cov_fn(pred_mean, self.T)
        q_T_mean, q_T_cov = gaussian_posterior(y_T, pred_mean, pred_cov, G_jac, G_cov,
                                               G_fn=partial(self.G_fn.G_mean_fn, t=self.T))
        self.q_t_mean_list.append(q_T_mean)
        self.q_t_cov_list.append(q_T_cov)

        # 1-step smoothing
        if self.T > 0:
            J = self.q_t_cov_list[self.T - 1] @ F_jac.t() @ torch.linalg.inv(pred_cov)
            q_Tm1_cov = self.q_t_cov_list[self.T - 1] + J @ (q_T_cov - pred_cov) @ J.t()
            q_Tm1_mean = self.q_t_mean_list[self.T - 1] + J @ (q_T_mean - pred_mean)
            self.q_tm1_mean_list.append(q_Tm1_mean)
            self.q_tm1_cov_list.append(q_Tm1_cov)

    def return_summary_stats(self, t=None):
        if t is None:
            t = self.T
        if t == self.T:
            x_t_mean = self.q_t_mean_list[t]
            x_t_cov = self.q_t_cov_list[t]
        elif t == self.T - 1:
            x_t_mean = self.q_tm1_mean_list[t]
            x_t_cov = self.q_tm1_cov_list[t]
        return x_t_mean, x_t_cov


class EnsembleKalmanFilter(nn.Module):
    def __init__(self, device, xdim, ydim, F_fn, G_fn, p_0_dist, ensemble_size):
        super().__init__()
        self.T = -1

        self.device = device
        self.xdim = xdim
        self.ydim = ydim
        self.ensemble_size = ensemble_size

        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist
        self.x_Tm1 = None
        self.x_T = None

    def advance_timestep(self, y_T):
        self.T += 1
        if self.T > 0:
            self.x_Tm1 = self.x_T.clone().detach()

    def update(self, y_T):
        if self.T == 0:
            x_pred = self.p_0_dist().sample((self.ensemble_size,))
        else:
            x_pred = self.F_fn(self.x_Tm1, self.T-1).sample()
        y_pred = self.G_fn(x_pred, self.T).sample()
        cov_x_y = sample_cov(x_pred, y_pred)
        cov_y = sample_cov(y_pred)
        K = cov_x_y @ torch.linalg.inv(cov_y)
        self.x_T = x_pred + (y_T - y_pred) @ K.t()

        if self.T > 0:
            cov_x_Tm1_y = sample_cov(self.x_Tm1, y_pred)
            J = cov_x_Tm1_y @ torch.linalg.inv(cov_y)
            self.x_Tm1 = self.x_Tm1 + (y_T - y_pred) @ J.t()

    def return_summary_stats(self, t=None):
        if t is None:
            t = self.T
        if t == self.T:
            x_t = self.x_T
        elif t == self.T - 1:
            x_t = self.x_Tm1
        x_t_mean = x_t.mean(0)
        x_t_cov = sample_cov(x_t)
        return x_t_mean, x_t_cov


class BootstrapParticleFilter(nn.Module):
    def __init__(self, device, xdim, ydim, F_fn, G_fn, p_0_dist, num_particles):
        """
            Cond q is the locally optimal proposal
        """
        super().__init__()
        self.T = -1

        self.device = device
        self.xdim = xdim
        self.ydim = ydim
        self.num_particles = num_particles

        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist

        self.x_Tm1 = None
        self.x_T = None

        # Initial weights (1/num_particles)
        self.log_w = np.log(1 / self.num_particles) * torch.ones((self.num_particles, 1), device=self.device)

    def advance_timestep(self, y_T):
        self.T += 1
        if self.T > 0:
            self.x_Tm1 = self.x_T.clone().detach()

    def sample_q_T(self, y_T, num_samples, detach_x=False):
        if self.T == 0:
            pred_dist = self.p_0_dist()
            x_T = pred_dist.rsample((self.num_particles, ))

        else:
            pred_dist = self.F_fn(self.x_Tm1, self.T-1)
            x_T = pred_dist.rsample()

        if detach_x:
            x_T = x_T.detach()

        return x_T

    def compute_log_p_t(self, x_t, y_t):  # Only need for time T
        log_p_y_t = self.G_fn(x_t, self.T).log_prob(y_t).unsqueeze(1)
        return {"log_p_y_t": log_p_y_t}

    def update(self, y_T):
        if self.resample_criterion():
            print("resample")
            self.resample()

        self.x_T = self.sample_q_T(y_T, self.num_particles, detach_x=True)
        log_p_T = self.compute_log_p_t(self.x_T, y_T)
        self.log_w += log_p_T["log_p_y_t"]

    def resample(self):
        resampling_dist = Categorical(logits=self.log_w[:, 0])
        ancestors = resampling_dist.sample((self.num_particles,))
        self.x_Tm1 = self.x_Tm1[ancestors, :]
        self.log_w = np.log(1 / self.num_particles) * torch.ones((self.num_particles, 1), device=self.device)

    def resample_criterion(self):
        if self.T > 0:
            return utils.ess(self.log_w) <= self.num_particles / 2
        else:
            return False

    def return_summary_stats(self, t=None):
        if t is None:
            t = self.T
        normalized_w = functional.softmax(self.log_w, dim=0)
        if t == self.T:
            x_t = self.x_T
        elif t == self.T - 1:
            x_t = self.x_Tm1
        x_t_mean = (x_t * normalized_w).sum(0)
        x_t_cov = sample_cov(x_t, w=normalized_w)
        return x_t_mean, x_t_cov


class MaternKernel(nn.Module):
    def __init__(self, nets=[torch.nn.Identity()], sigma=1.0, lam=0.01, p=np.inf, train_sigma=False, train_lam=False):
        super().__init__()
        self.log_lam = nn.Parameter(torch.tensor(lam).log(), requires_grad=train_lam)
        self.log_sigma = nn.Parameter(torch.tensor(sigma).log(), requires_grad=train_sigma)
        self.kernel_networks = nn.ModuleList(nets)
        self.p = p

    def forward(self, x, y):
        return self.gram(x, y)

    def dgg(self, X, Y, g):
        FX = g(X)
        FY = g(Y)
        FK = utils.l2_distance(FX, FY)
        return FK

    def gram(self, X, Y):
        G = 0
        for k in self.kernel_networks:
            G = G + self.dgg(X, Y, k)

        if self.p == np.inf:
            G = (-G / (2 * self.log_sigma.exp() ** 2)).exp()
            return G
        else:
            distance = G.sqrt() / self.log_sigma.exp()
            exp_component = torch.exp(-np.sqrt(self.p * 2) * distance)

            if self.p == 0.5:
                constant_component = 1
            elif self.p == 1.5:
                constant_component = (np.sqrt(3) * distance).add(1)
            elif self.p == 2.5:
                constant_component = (np.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component


class KernelRidgeRegressor(nn.Module):
    def __init__(self, kernel, centre_elbo=False):
        # We approximate the elbo in the final entry of the KRR
        super().__init__()
        self.kernel = kernel
        self.centre_elbo = centre_elbo
        self.x_fit = None

    def fit(self, x_fit, *fs):
        self.nsleep = x_fit.shape[0]
        for f in fs:
            assert f.shape[0] == self.nsleep

        # self.K = (self.K + self.K.t()) / 2
        self.x_fit = x_fit
        # regressions share the same inverse of the gram matrix K(sleep_x, sleep_x)
        self.update_K()
        self.fs = list(fs)

    def forward(self, x, index=None):
        # similarity between sleep and wake data
        G = self.kernel(self.x_fit, x)
        if self.training:
            self.update_K()

        # this computes G.t() @ K^{-1}
        GKinv = torch.solve(G, self.K)[0].t()

        if not self.centre_elbo:
            if index is None:
                return [GKinv @ f for f in self.fs]
            else:
                return GKinv @ self.fs[index]
        else:
            if index is None:
                elbo_mean = self.fs[-1].mean(dim=0)
                return [GKinv @ f for f in self.fs[:-1]] + [GKinv @ (self.fs[-1] - elbo_mean) + elbo_mean]
            else:
                if index == -1 or index == len(self.fs) - 1:
                    elbo_mean = self.fs[-1].mean(dim=0)
                    return GKinv @ (self.fs[index] - elbo_mean) + elbo_mean
                else:
                    return GKinv @ self.fs[index]

        # dual_coefs = [torch.solve(f, self.K)[0] for f in self.fs]
        #
        # return [G.t() @ dual_coef for dual_coef in dual_coefs]

        # GKinv = G.t() @ torch.linalg.inv(self.K)

        # chol_K = torch.cholesky(self.K)
        # dual_coefs = [torch.cholesky_solve(f, chol_K)[0] for f in self.fs]

        # dual_coefs = [torch.tensor(linalg.solve(self.K.detach().cpu().numpy(), f.detach().cpu().numpy())).to(G.device)
        #               for f in self.fs]

        # return [G.t() @ dual_coef for dual_coef in dual_coefs]

    def update_K(self):
        self.K = self.kernel(self.x_fit, self.x_fit) + \
                 self.kernel.log_lam.exp() * torch.eye(self.nsleep, device=self.x_fit.device)

    def train(self, mode=True):
        super().train(mode)
        if not mode:
            if self.x_fit is not None: # update if we have an x_fit
                self.update_K()
        return self

class ThetaGrad():
    """
        Add on to a NonArmotizedModelBase that will learn theta gradients

        model_base: The base model that learns phi
        theta_func_constructor: Constructor that makes the function that estimates theta grads
        window_size: >=1 how far back the function estimator comes in
        get_theta_parameters: Function that returns the theta params to learn
        add_theta_grads_to_params: Function which takes tensor of size (theta_dim)
            and adds that tensor to the appropriate gradients of the appropriate 
            theta parameters.
    """
    def __init__(self, device, model_base, theta_func_constructor,
        window_size, theta_dim, get_theta_parameters, add_theta_grads_to_params):
        self.device = device
        self.mb = model_base

        assert window_size >= 1

        self.theta_func_constructor = theta_func_constructor
        self.theta_func_TmL = self.theta_func_constructor()
        self.theta_func_TmLm1 = self.theta_func_constructor()

        self.window_size = window_size
        self.theta_dim = theta_dim

        self.get_theta_parameters = get_theta_parameters
        self.add_theta_grads_to_params = add_theta_grads_to_params

    def advance_timestep(self):
        self.theta_func_TmLm1 = self.theta_func_TmL
        self.theta_func_TmL = self.theta_func_constructor()
        self.theta_func_TmL.load_state_dict(self.theta_func_TmLm1.state_dict())
        self.theta_func_TmLm1.requires_grad_(False)
        self.theta_func_TmLm1.eval()

    def get_theta_func_TmL_parameters(self):
        return self.theta_func_TmL.parameters()

    def generate_training_dataset(self, N, y):
        # Generates input-output pairs to train theta_func_TmL 

        assert self.mb.T >= self.window_size

        def s_to_flat(s):
            output = torch.zeros(self.theta_dim, device=self.device)
            culm_idx = 0
            for grad in s:
                grad = grad.flatten()
                output[culm_idx:culm_idx+grad.shape[0]] = grad
                culm_idx += grad.shape[0]
            assert culm_idx == self.theta_dim
            return output
            

        if self.mb.T > self.window_size:
            x_samples, _ = self.mb.sample_joint_q_t(N, self.window_size+1, True)

            s_batch = torch.zeros((N, self.theta_dim), device=self.device)
            for i in range(N):
                x_TmLm1 = x_samples[0][i:i+1, :]
                x_TmL = x_samples[1][i:i+1, :]
                log_p_t_dict = self.mb.compute_log_p_t(x_TmL,
                                                    y[self.mb.T-self.window_size, :],
                                                    x_TmLm1)
                log_p = log_p_t_dict['log_p_x_t'] + log_p_t_dict['log_p_y_t']
                s = s_to_flat(torch.autograd.grad(log_p[0, 0], self.get_theta_parameters(),
                    retain_graph=True))
                s_batch[i, :] = s

            theta_grads_TmLm1 = self.theta_func_TmLm1(x_samples[0])

            targets = (theta_grads_TmLm1 + s_batch).detach()

            inputs = x_samples[1].detach()

        elif self.mb.T == self.window_size:
            x_samples, _ = self.mb.sample_joint_q_t(N, self.window_size, True)

            s_batch = torch.zeros((N, self.theta_dim), device=self.device)
            for i in range(N):
                x_0 = x_samples[0][i:i+1, :]
                log_p_0_dict = self.mb.compute_log_p_t(x_0, y[0, :], t=0)
                log_p = log_p_0_dict['log_p_x_t'] + log_p_0_dict['log_p_y_t']
                grads = torch.autograd.grad(log_p[0, 0], self.get_theta_parameters(),
                    retain_graph=True, allow_unused=True)
                grads_list = []
                for idx, p in enumerate(self.get_theta_parameters()):
                    if grads[idx] is None:
                        grads_list.append(torch.zeros_like(p))
                    else:
                        grads_list.append(grads[idx])
                s = s_to_flat(grads_list)
                s_batch[i, :] = s

            targets = s_batch.detach()

            inputs = x_samples[0].detach()

        return inputs, targets

    def populate_theta_grads(self, N, y):
        if self.mb.T == 0:
            x_samples, _ = self.mb.sample_q_T(N, True)
            log_p_t = self.mb.compute_log_p_t(x_samples, y[0, :], t=0)
            s = log_p_t['log_p_x_t'] + log_p_t['log_p_y_t']
            torch.mean(s).backward()

        elif self.mb.T < self.window_size and self.mb.T > 0:
            x_samples_1, _ = self.mb.sample_joint_q_t(N, self.mb.T, True)
            self.mb.T -= 1
            if self.mb.T == 0:
                x_samples_2 = [self.mb.sample_q_T(N, True)[0]]
            else:
                x_samples_2, _ = self.mb.sample_joint_q_t(N, self.mb.T, True)
            self.mb.T += 1

            s_sum_1 = 0
            for i in range(self.mb.T+1):
                log_p_t = self.mb.compute_log_p_t(
                    x_samples_1[i], y[i, :],
                    x_samples_1[i-1] if i > 0 else None, 
                    t=i
                )
                s_sum_1 += log_p_t['log_p_x_t'] + log_p_t['log_p_y_t']
            
            s_sum_2 = 0
            for i in range(self.mb.T):
                log_p_t = self.mb.compute_log_p_t(
                    x_samples_2[i], y[i, :],
                    x_samples_2[i-1] if i > 0 else None,
                    t=i
                )
                s_sum_2 += log_p_t['log_p_x_t'] + log_p_t['log_p_y_t']

            torch.mean(s_sum_1 - s_sum_2).backward()

        elif self.mb.T >= self.window_size:
            x_samples_1, _ = self.mb.sample_joint_q_t(N, self.window_size, True)
            self.mb.T -= 1
            x_samples_2, _ = self.mb.sample_joint_q_t(N, self.window_size-1, True)
            self.mb.T += 1


            s_sum_1 = 0
            for i in range(self.window_size):
                log_p_t = self.mb.compute_log_p_t(
                    x_samples_1[self.window_size-i], y[self.mb.T-i, :], x_samples_1[self.window_size-1-i],
                    self.mb.T-i
                )
                s_sum_1 += log_p_t['log_p_x_t'] + log_p_t['log_p_y_t']

            s_sum_2 = 0
            for i in range(self.window_size-1):
                log_p_t = self.mb.compute_log_p_t(
                    x_samples_2[self.window_size-1-i], y[self.mb.T-1-i, :],
                    x_samples_2[self.window_size-2-i], self.mb.T-1-i
                )
                s_sum_2 += log_p_t['log_p_x_t'] + log_p_t['log_p_y_t']

            theta_grads_TmL_v1 = self.theta_func_TmL(x_samples_1[0])
            theta_grads_TmL_v2 = self.theta_func_TmL(x_samples_2[0])

            torch.mean(s_sum_1 - s_sum_2).backward()

            self.add_theta_grads_to_params(
                torch.mean(theta_grads_TmL_v1 - theta_grads_TmL_v2, dim=0)
            )

        for p in self.get_theta_parameters():
            p.grad = -p.grad


class ThetaGradGaussian(ThetaGrad):
    def generate_training_dataset(self, N, y):
        # Generates input-output pairs to train theta_func_TmL

        assert self.mb.T >= self.window_size

        if self.mb.T > self.window_size:
            x_samples, _ = self.mb.sample_joint_q_t(N, self.window_size+1, True)
            x_TmLm1_batch = x_samples[0]
            x_TmL_batch = x_samples[1]

            analytical_sbatch_F = - (
                        self.mb.F_fn.weight * x_TmLm1_batch ** 2 - x_TmL_batch * x_TmLm1_batch) / self.mb.F_fn.F_cov.diag()
            analytical_sbatch_G = - (
                        self.mb.G_fn.weight * x_TmL_batch ** 2 - x_TmL_batch * y[self.mb.T - self.window_size,
                                                                               :]) / self.mb.G_fn.G_cov.diag()
            s_batch = torch.cat([analytical_sbatch_F, analytical_sbatch_G], dim=1)

            theta_grads_TmLm1 = self.theta_func_TmLm1(x_TmLm1_batch)

            targets = (theta_grads_TmLm1 + s_batch).detach()

            inputs = x_TmL_batch.detach()

        elif self.mb.T == self.window_size:
            x_samples, _ = self.mb.sample_joint_q_t(N, self.window_size, True)
            x_TmL_batch = x_samples[0]
            analytical_sbatch_F = torch.zeros((N, self.mb.xdim), device=self.device)
            analytical_sbatch_G = - (
                        self.mb.G_fn.weight * x_TmL_batch ** 2 - x_TmL_batch * y[self.mb.T - self.window_size,
                                                                               :]) / self.mb.G_fn.G_cov.diag()
            s_batch = torch.cat([analytical_sbatch_F, analytical_sbatch_G], dim=1)

            targets = s_batch.detach()

            inputs = x_TmL_batch.detach()

        return inputs, targets


class ThetaGradVJF(ThetaGrad):
    def populate_theta_grads(self, N, y):
        if self.mb.T == 0:
            x_samples, _ = self.mb.sample_q_T(N, detach_x=True)
            log_p_t = self.mb.compute_log_p_t(x_samples, y[0, :], t=0)

        else:
            x_T, _ = self.mb.sample_q_T(N, detach_x=True)
            self.mb.T -= 1
            x_Tm1, _ = self.mb.sample_q_T(N, detach_x=True)
            self.mb.T += 1
            log_p_t = self.mb.compute_log_p_t(x_T, y[self.mb.T, :], x_Tm1, t=self.mb.T)

        s = log_p_t['log_p_x_t'] + log_p_t['log_p_y_t']
        torch.mean(-s).backward()


class NNFuncEstimator(nn.Module):
    def __init__(self, nn, input_dim, output_dim):
        super().__init__()
        self.nn = nn

        # The statistics of the inputs to the NN function
        self.register_buffer('input_mean', torch.zeros((input_dim)))
        self.register_buffer('input_std', torch.ones((input_dim)))

        # The statistics of the training data outputs
        self.register_buffer('output_mean', torch.zeros((output_dim)))
        self.register_buffer('output_std', torch.ones((output_dim)))

        self.initial_update_norm = True

    def update_normalization(self, train_inputs, train_outputs, decay):
        if self.initial_update_norm:
            decay = 0
            self.initial_update_norm = False

        self.input_mean = decay * self.input_mean + \
                          (1 - decay) * torch.mean(train_inputs, dim=0)
        self.input_std = decay * self.input_std + \
                         (1 - decay) * torch.std(train_inputs, dim=0)
        self.output_mean = decay * self.output_mean + \
                           (1 - decay) * torch.mean(train_outputs, dim=0)
        self.output_std = decay * self.output_std + \
                          (1 - decay) * torch.std(train_outputs, dim=0)

    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        outputs_norm = self.nn(x_norm)
        return self.output_mean + outputs_norm * self.output_std

class NN_Func_Polar(nn.Module):
    """
        A wrapper for a neural net gradient estimator that operates using a
        polar style system,
        i.e. it predicts a direction for the gradient and a magnitude rather
        than the gradient vector itself.
        net should have an output of dimension equal to gradient dim + 1
        So the first bits of the output specify the direction and the last bit
        is the log magnitude of the gradient.
    """
    def __init__(self, net, input_dim, grad_dim):
        super().__init__()
        self.net = net
        self.grad_dim = grad_dim

        self.register_buffer('input_mean', torch.zeros((input_dim)))
        self.register_buffer('input_std', torch.ones((input_dim)))
        self.register_buffer('log_mag_mean', torch.zeros((1)))
        self.register_buffer('log_mag_std', torch.ones((1)))

        self.initial_update_norm = True

    def update_normalization(self, train_inputs, train_outputs, decay):
        if self.initial_update_norm:
            decay = 0
            self.initial_update_norm = False

        self.input_mean = decay * self.input_mean + \
            (1 - decay) * torch.mean(train_inputs, dim=0)
        self.input_std = decay * self.input_std + \
            (1 - decay) * torch.std(train_inputs, dim=0)

        output_norms = torch.sum(torch.abs(train_outputs), dim=1)

        log_output_norms = torch.log(output_norms)
        if torch.isinf(log_output_norms).any():
            raise ValueError("Log output norms inf")

        self.log_mag_mean = decay * self.log_mag_mean + \
            (1 - decay) * torch.mean(log_output_norms)
        self.log_mag_std = decay * self.log_mag_std + \
            (1 - decay) * torch.std(log_output_norms)

    def forward(self, x):
        x_norm = (x - self.input_mean) / self.input_std
        net_out = self.net(x_norm)
        output_dir = net_out[..., 0:self.grad_dim]
        log_mag_norm = net_out[..., self.grad_dim:]
        log_mag = self.log_mag_mean + log_mag_norm * self.log_mag_std

        output_dir_norms = torch.sum(torch.abs(output_dir), dim=-1).unsqueeze(-1)
        return (output_dir / output_dir_norms) * torch.exp(log_mag)

class LinearRMLEDiagFG():
    """
        RMLE for linear Gaussian learning both F and G assuming everything
        is diagonal

        matrices_to_learn = 'F' or 'G' or 'FG'
    """
    def __init__(self, x_0, P_0, F_0, G_0, U, V, step_size, matrices_to_learn):
        """
            Vectors should be dx1
            Matrics should be dxd
        """
        self.x, self.P, self.F, self.G, self.U, self.V = \
            x_0, P_0, F_0, G_0, U, V

        self.d = x_0.shape[0]

        self.Gquad = np.zeros(self.d)
        self.Glin = np.zeros(self.d)
        self.Gconst = np.zeros(self.d)

        self.Fquad = np.zeros(self.d)
        self.Flin = np.zeros(self.d)
        self.Fconst = np.zeros(self.d)

        self.first_update = True

        self.step_size = step_size

        self.matrices_to_learn = matrices_to_learn

    def advance_timestep(self, y_T):
        """
            y_T should be (d,1)
        """
        if self.first_update:
            self.Gquad = - np.diag(self.G) / np.diag(self.V)
            self.Glin = y_T[:, 0] / np.diag(self.V)
            self.Gconst = np.zeros(self.d)

            self.Fquad = np.zeros(self.d)
            self.Flin = np.zeros(self.d)
            self.Fconst = np.zeros(self.d)
        else:
            triangle = (np.diag(self.P) * np.diag(self.F)) / \
                (np.diag(self.F)**2 * np.diag(self.P) + np.diag(self.U))
            self.Gconst = self.Gquad * np.diag(self.P) \
                - self.Gquad * np.diag(self.P) * np.diag(self.F) * triangle \
                + self.Gquad * self.x[:,0]**2 \
                - self.Gquad * 2 * self.x[:,0]**2 * triangle * np.diag(self.F) \
                + self.Gquad * triangle**2 * np.diag(self.F)**2 * self.x[:,0]**2 \
                - self.Glin * triangle * np.diag(self.F) * self.x[:,0] \
                + self.Glin * self.x[:,0] \
                + self.Gconst

            self.Glin = y_T[:, 0] / np.diag(self.V) \
                + self.Gquad * 2 * self.x[:,0] * triangle \
                - 2 * self.Gquad * triangle**2 * np.diag(self.F) * self.x[:,0] \
                + self.Glin * triangle

            self.Gquad = - np.diag(self.G) / np.diag(self.V) \
                + self.Gquad * triangle**2

            self.Fconst = (self.Fquad - np.diag(self.F)/np.diag(self.U)) * \
                (
                    np.diag(self.P) \
                    - triangle * np.diag(self.P) * np.diag(self.F) \
                    + self.x[:,0]**2 \
                    - 2 * self.x[:,0]**2 * triangle * np.diag(self.F) \
                    + triangle**2 * np.diag(self.F)**2 * self.x[:,0]**2
                ) \
                + self.Flin * self.x[:,0] \
                - self.Flin * triangle * np.diag(self.F) * self.x[:,0] \
                + self.Fconst

            self.Flin = self.x[:,0] / np.diag(self.U) \
                - triangle * np.diag(self.F) * self.x[:,0] / np.diag(self.U) \
                + (self.Fquad - np.diag(self.F)/np.diag(self.U)) * 2 * self.x[:,0] * triangle \
                - 2 * (self.Fquad - np.diag(self.F)/np.diag(self.U)) * triangle**2 * np.diag(self.F) * self.x[:,0] \
                + self.Flin * triangle

            self.Fquad = triangle / np.diag(self.U) \
                + (self.Fquad - np.diag(self.F)/np.diag(self.U)) * triangle**2

        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.U
        S = self.G @ Pp @ self.G.T + self.V
        K = Pp @ self.G.T @ np.linalg.inv(S)
        z = y_T - self.G @ xp

        self.x = xp + K @ z
        self.P = (np.eye(self.d) - K @ self.G) @ Pp

        G_grad_term = self.Gquad * (np.diag(self.P) + self.x[:,0]**2) \
            + self.Glin * self.x[:,0] + self.Gconst
        F_grad_term = self.Fquad * (np.diag(self.P) + self.x[:,0]**2) \
            + self.Flin * self.x[:,0] + self.Fconst
        if not self.first_update:
            if 'G' in self.matrices_to_learn:
                self.G = self.G + self.step_size * \
                    np.diag(G_grad_term - self.G_prev_grad_term) 
            if 'F' in self.matrices_to_learn:
                self.F = self.F + self.step_size * \
                    np.diag(F_grad_term - self.F_prev_grad_term)

        self.G_prev_grad_term = np.copy(G_grad_term)
        self.F_prev_grad_term = np.copy(F_grad_term)

        if self.first_update:
            self.first_update = False


class TrueSDiagFG(nn.Module):
    """
        For the linear Gaussian model, this is the analytic S function.
    """

    def __init__(self, dim, y, G_diag, U, V, matrices_to_learn):
        super().__init__()
        self.dim = dim
        self.U = U
        self.V = V
        self.matrices_to_learn = matrices_to_learn

        self.register_buffer('FTlin', torch.zeros(dim, dim))
        self.register_buffer('FTquad', torch.zeros(dim, dim, dim))
        self.register_buffer('Fconst', torch.zeros(dim))

        self.register_buffer('GTlin', torch.zeros(dim, dim))
        self.register_buffer('GTquad', torch.zeros(dim, dim, dim))
        self.register_buffer('Gconst', torch.zeros(dim))

        self.GTlin = torch.diag(y[0] / self.V.diag())
        self.GTquad = - torch.diag_embed(torch.diag(G_diag / self.V.diag()))

    def advance_timestep(self, y_T, F_diag, G_diag, qW, qb, qcov_diag):
        FTcross = torch.diag_embed(torch.diag(1 / self.U.diag()))
        FTm1quad = - torch.diag_embed(torch.diag(F_diag / self.U.diag()))
        self.Fconst = self.FTlin @ qb + \
                      torch.diagonal((self.FTquad + FTm1quad) @ qcov_diag.diag(), dim1=-2, dim2=-1).sum(-1) + \
                      qb.T @ (self.FTquad + FTm1quad) @ qb + \
                      self.Fconst
        self.FTlin = self.FTlin @ qW + \
                     qb.T @ (self.FTquad + FTm1quad).transpose(-2, -1) @ qW + \
                     qb.T @ (self.FTquad + FTm1quad) @ qW + \
                     qb.T @ FTcross
        self.FTquad = qW.T @ (self.FTquad + FTm1quad) @ qW + \
                      qW.T @ FTcross

        self.Gconst = self.GTlin @ qb + \
                      torch.diagonal(self.GTquad @ qcov_diag.diag(), dim1=-2, dim2=-1).sum(-1) + \
                      qb.T @ self.GTquad @ qb + \
                      self.Gconst
        self.GTlin = self.GTlin @ qW + \
                     qb.T @ self.GTquad.transpose(-2, -1) @ qW + \
                     qb.T @ self.GTquad @ qW + \
                     torch.diag(y_T / self.V.diag())
        self.GTquad = qW.T @ self.GTquad @ qW - \
                      torch.diag_embed(torch.diag(G_diag / self.V.diag()))

    def forward(self, x_T):
        N = x_T.shape[0]
        assert x_T.shape == torch.Size([N, self.dim])

        Fout = functional.linear(x_T, self.FTlin) + \
               functional.bilinear(x_T, x_T, self.FTquad) + \
               self.Fconst

        Gout = functional.linear(x_T, self.GTlin) + \
               functional.bilinear(x_T, x_T, self.GTquad) + \
               self.Gconst

        if self.matrices_to_learn == 'F':
            return Fout
        elif self.matrices_to_learn == 'G':
            return Gout
        elif self.matrices_to_learn == 'FG':
            return torch.cat([Fout, Gout], dim=1)
        else:
            raise ValueError(self.matrices_to_learn)
