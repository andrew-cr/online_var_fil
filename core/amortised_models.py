from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import core.nonamortised_models as nonamortised_models
import core.utils as utils
from torch.distributions import MultivariateNormal, Independent, Normal, Categorical
from core.utils import gaussian_posterior, sample_cov


class AmortizedModelBase(nonamortised_models.NonAmortizedModelBase):
    """
        A generic base class for nonlinear models to be inherited from.
        Contains nonlinear model in the form
            x_t ~ F_fn (x_{t-1})
            y_t ~ G_fn (x_t)
        F, G are nn.Modules which return a distribution and can potentially contain learnable theta parameters
        q distributions are amortized variational posteriors
            These will both be factorized Gaussians.
        q_rnn encodes observations y^t -> q_rnn_hidden_size
        q_t_net: q_rnn_hidden_size -> xdim, xdim (Gaussian mean, std)
        cond_q_t_net: xdim + q_rnn_hidden_size -> xdim, xdim (Gaussian mean, std)
    """

    def __init__(self, device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net, F_fn, G_fn, p_0_dist,
                 window_size, rnn_window_size, approx_rnn_grad=False,
                 theta_to_learn='FG'):
        nn.Module.__init__(self)
        self.T = -1

        self.device = device
        self.xdim = xdim
        self.ydim = ydim

        # Inference model
        self.q_rnn = q_rnn
        self.q_rnn_hidden_size = self.q_rnn.hidden_size
        self.q_t_net = q_t_net  # q_rnn_hidden_size -> xdim, xdim
        self.cond_q_t_net = cond_q_t_net  # xdim + q_rnn_hidden_size -> xdim, xdim
        self.q_rnn_hist_hn_list = []  # Store historical q_rnn hn, length T + 1
        self.q_rnn_hist_output_list = []  # Store historical q_rnn output, length T + 1

        # Generative model
        self.F_fn = F_fn
        self.G_fn = G_fn
        self.p_0_dist = p_0_dist

        self.window_size = window_size
        self.rnn_window_size = rnn_window_size
        self.approx_rnn_grad = approx_rnn_grad

        self.amortised = True

        self.theta_to_learn = theta_to_learn

    def advance_timestep(self, y):
        """
            Prepare the model when new data arrives, should be called for every T >= 0
        """
        self.T += 1
        self.q_rnn_hist_hn_list.append(None)  # length T + 1
        self.q_rnn_hist_output_list.append(None)  # length T + 1

    def reset_timestep(self):
        self.T = -1
        self.q_rnn_hist_hn_list = []
        self.q_rnn_hist_output_list = []

    def detach_rnn_hist_hn(self, y, t=None):
        # Detach and store the last rnn state in the window, should be called at the end of each T
        if t is None:
            t = self.T - self.rnn_window_size + 1  # Detach and store h_t
        if t >= 0:
            if t == 0:
                output, hn = self.q_rnn(y[0, :].view(1, 1, self.ydim))
            elif t >= 1:
                output, hn = self.q_rnn(y[t, :].view(1, 1, self.ydim), self.q_rnn_hist_hn_list[t - 1])

            if not self.approx_rnn_grad:  # detach historical rnn state
                # length self.T - self.rnn_window_size + 1, last time self.T - self.rnn_window_size
                self.q_rnn_hist_output_list[t] = output.detach()
                if isinstance(hn, tuple):
                    hn, cn = hn
                    self.q_rnn_hist_hn_list[t] = (hn.detach(), cn.detach())
                else:
                    self.q_rnn_hist_hn_list[t] = hn.detach()
            else:
                raise NotImplementedError

    def get_theta_params(self, flatten=False):
        params = []
        if 'F' in self.theta_to_learn:
            params += [*self.F_fn.parameters()]
        if 'G' in self.theta_to_learn:
            params += [*self.G_fn.parameters()]
        if len(params) == 0:
            return None
        if not flatten:
            return params
        else:
            return nn.utils.parameters_to_vector(params)

    def get_phi_params(self, flatten=False):
        params = [*self.q_rnn.parameters(), *self.q_t_net.parameters(), *self.cond_q_t_net.parameters()]
        if len(params) == 0:
            return None
        if not flatten:
            return params
        else:
            return nn.utils.parameters_to_vector(params)

    def get_rnn_phi_params(self, flatten=False):
        params = [*self.q_rnn.parameters()]
        if len(params) == 0:
            return None
        if not flatten:
            return params
        else:
            return nn.utils.parameters_to_vector(params)

    def get_q_t_net_phi_params(self, flatten=False):
        params = [*self.q_t_net.parameters(), *self.cond_q_t_net.parameters()]
        if len(params) == 0:
            return None
        if not flatten:
            return params
        else:
            return nn.utils.parameters_to_vector(params)

    def rnn_forward(self, y, t_start=None):
        """
            Run RNN forward from t_start, assumes that t_start is 0 or rnn state at t_start - 1 is already stored
        """
        if t_start is None:
            t_start = max(self.T - self.rnn_window_size + 1, 0)  # time of first y

        assert t_start >= 0

        if t_start == 0:
            self.rnn_outputs, _ = self.q_rnn(y[t_start:(self.T + 1), :].view(self.T - t_start + 1, 1, self.ydim))
        else:
            # output shape (self.rnn_window_size, 1, hidden)
            self.rnn_outputs, _ = self.q_rnn(y[t_start:(self.T + 1), :].view(self.T - t_start + 1, 1, self.ydim),
                                             self.q_rnn_hist_hn_list[t_start - 1])
        return self.rnn_outputs

    def rnn_forward_offline(self, y):
        """
            Run RNN forward from time 0 (in an offline manner) through entire y
        """
        assert y.shape[0] == self.T + 1
        self.rnn_outputs, _ = self.q_rnn(y.view(self.T + 1, 1, self.ydim))
        return self.rnn_outputs

    def compute_filter_stats(self, T=None):
        """
            Compute the q_T(x_T) (filtering) statistics at time T
            Need to call rnn_forward before calling this function
        """
        if T is None:
            T = self.T
        assert T <= self.T
        t_start = self.T - self.rnn_outputs.shape[0] + 1  # time index of first y obs in rnn
        if T < t_start:  # case: sample historical q_T
            q_T_stats = self.q_t_net(self.q_rnn_hist_output_list[T][0, 0, :])

        else:  # case: sample recent q_T in the window, we use self.rnn_outputs
            q_T_stats = self.q_t_net(self.rnn_outputs[T - t_start, 0, :])

        return q_T_stats

    def sample_q_T(self, num_samples, detach_x=False, T=None):
        """
            Sample from q_T(x_T) and return its statistics
            Need to call rnn_forward before calling this function
        """
        if T is None:
            T = self.T
        assert T <= self.T
        q_T_stats = self.compute_filter_stats(T)
        q_T_mean, q_T_std = q_T_stats

        eps_x_T = torch.randn(num_samples, self.xdim).to(self.device)
        x_T = q_T_mean + q_T_std * eps_x_T

        if detach_x:
            x_T = x_T.detach()

        return x_T, q_T_stats

    def sample_q_t_cond_T(self, x_T, num_steps_back, detach_x=False, T=None):
        """
            Sample from q(x_{(T-num_steps_back):(T-1)}|x_T) and return their statistics
            Need to call rnn_forward before calling this function
        """
        if T is None:
            T = self.T
        assert T <= self.T
        num_samples = x_T.shape[0]
        t_start = self.T - self.rnn_outputs.shape[0] + 1  # time index of first y obs in rnn
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

            if t < t_start:  # case: sample historical q_t, use rnn state from stored output list
                rnn_output = self.q_rnn_hist_output_list[t][0, 0, :]
            else:  # case: sample recent q_t in the window, we use self.rnn_outputs
                rnn_output = self.rnn_outputs[t - t_start, 0, :]
            cond_q_t_mean, cond_q_t_std = self.cond_q_t_net(torch.cat([
                x_tp1, rnn_output.expand(num_samples, self.q_rnn_hidden_size)], dim=1))

            eps_x_t = torch.randn(num_samples, self.xdim).to(self.device)
            x_t = cond_q_t_mean + cond_q_t_std * eps_x_t

            if detach_x:
                x_t = x_t.detach()

            x_t_samples[t - T + num_steps_back] = x_t
            all_cond_q_t_means[t - T + num_steps_back] = cond_q_t_mean
            all_cond_q_t_stds[t - T + num_steps_back] = cond_q_t_std

        all_cond_q_t_stats = [[mean, std] for mean, std in zip(all_cond_q_t_means, all_cond_q_t_stds)]

        return x_t_samples, all_cond_q_t_stats

    # sample_joint_q_t, compute_log_p_t, compute_log_q_t, compute_r_t, sample_and_compute_r_t,
    # sample_and_compute_joint_r_t, generate_data are inherited from NonAmortizedModelBase

    def compute_elbo_loss_offline_rnn(self, y, num_samples):
        """
            Compute negative ELBO loss of the entire y
        """
        self.rnn_forward_offline(y)
        all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, self.T + 1)

        r_values = all_r_results["r_values"]
        sum_r = sum(r_values)
        log_q_x_T = all_r_results["log_q_x_T"]

        loss = - (sum_r - log_q_x_T).mean()
        return loss

    def return_summary_stats(self, y, t=None, num_samples=None):
        if t is None:
            t = self.T
        self.rnn_forward(y)
        if t == self.T:
            q_t_stats = self.compute_filter_stats()
            x_t_mean = q_t_stats[0].detach().clone()
            x_t_cov = torch.diag(q_t_stats[1].detach().clone() ** 2)
        elif t == self.T - 1:
            joint_x_samples, _ = self.sample_joint_q_t(num_samples, 1)
            x_Tm1_samples = joint_x_samples[0]
            x_t_mean = x_Tm1_samples.mean(0)
            x_t_cov = sample_cov(x_Tm1_samples)
        return x_t_mean, x_t_cov


class Kernel_Amortised_Model(AmortizedModelBase):
    def __init__(self, device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net, F_fn, G_fn, p_0_dist,
                 window_size, rnn_window_size, rnn_h_lambda,
                 approx_func_constructor, funcs_to_approx, approx_updates_start_t, approx_decay, approx_with_filter):
        super().__init__(device, xdim, ydim, q_rnn, q_t_net, cond_q_t_net, F_fn, G_fn, p_0_dist,
                         window_size, rnn_window_size)
        self.rnn_h_lambda = rnn_h_lambda
        self.approx_func_constructor = approx_func_constructor
        self.funcs_to_approx = funcs_to_approx  # subset of 'STU', by default we estimate V
        self.approx_updates_start_t = approx_updates_start_t  # t of first fitted approx_func_t
        self.approx_decay = approx_decay
        self.approx_with_filter = approx_with_filter

        self.theta_dim = len(utils.replace_none(self.get_theta_params(flatten=True)))
        self.phi_dim = len(utils.replace_none(self.get_phi_params(flatten=True)))

    def advance_timestep(self, y_T):
        super().advance_timestep(y_T)
        if self.T == self.approx_updates_start_t + self.window_size - 1:
            self.approx_func_t = self.approx_func_constructor()

        elif self.T > self.approx_updates_start_t + self.window_size - 1:
            self.approx_func_tm1 = self.approx_func_t
            self.approx_func_t = self.approx_func_constructor()
            self.approx_func_t.load_state_dict(self.approx_func_tm1.state_dict())
            self.approx_func_tm1.requires_grad_(False)
            self.approx_func_tm1.eval()

    def generate_training_dataset(self, y, num_samples, t=None, disperse_temp=1):
        """
            Generate training dataset for updating approx_func_t
            Returns x variable x_t and the corresponding values of S_t, T_t, U_t, V_t at x_t
        """
        if t is None:
            t = self.T - self.window_size + 1  # By default, we update func_t at time T (func_T when window_size=1)
        if t >= self.approx_updates_start_t:  # approx_updates_start_t is t of first fitted approx_func_t
            self.rnn_forward(y)

            r_results = self.sample_and_compute_r_t(y[t, :], num_samples, t=t, disperse_temp=disperse_temp)
            x_t = r_results["x_t"]
            r_t = r_results["r_t"]

            if t == 0:
                if self.approx_with_filter:
                    # Adjust r_t to the form in the paper
                    log_q_t = self.compute_log_q_t(x_t, *self.compute_filter_stats(t))
                    r_t -= log_q_t
            else:
                x_tm1 = r_results["x_tm1"]
                if self.approx_with_filter:
                    log_q_t = self.compute_log_q_t(x_t, *self.compute_filter_stats(t))
                    log_q_tm1 = self.compute_log_q_t(x_tm1, *self.compute_filter_stats(t-1))
                    r_t += (log_q_tm1 - log_q_t)

            # t_t = \nabla_{x_t} r_t(x_{t-1}(phi, x_t), x_t), shape (num_samples, xdim)
            if 'T' in self.funcs_to_approx:
                t_t = torch.autograd.grad(r_t.sum(), x_t, retain_graph=True)[0]
            else:
                t_t = torch.zeros((num_samples, self.xdim), device=self.device)

            s_t = torch.zeros((num_samples, self.theta_dim), device=self.device)
            u_t = torch.zeros((num_samples, self.phi_dim), device=self.device)

            # s_t = \nabla_theta r_t(x_{t-1}(phi, x_t), x_t), shape (num_samples, theta_dim)
            if self.theta_dim > 0 and 'S' in self.funcs_to_approx:
                for i in range(num_samples):
                    s_t[i, :] = nn.utils.parameters_to_vector(
                        torch.autograd.grad(r_t[i, 0], self.get_theta_params(), retain_graph=True))

            # u_t = \nabla_phi r_t(x_{t-1}(phi, x_t), x_t), shape (num_samples, phi_dim)
            if 'U' in self.funcs_to_approx:
                for i in range(num_samples):
                    u_t[i, :] = nn.utils.parameters_to_vector(
                        torch.autograd.grad(r_t[i, 0] + 0 * self.get_phi_params(flatten=True).mean(),
                                            self.get_phi_params(), retain_graph=True))

            # S_tm1 = \nabla_theta V_{t-1} (x_{t-1})
            # T_tm1 = \nabla_{x_{t-1}} V_{t-1} (x_{t-1})
            # U_tm1 = \nabla_phi V_{t-1} (x_{t-1})
            if t == self.approx_updates_start_t:
                S_tm1, T_tm1, U_tm1, V_tm1 = torch.zeros_like(s_t), torch.zeros_like(t_t), torch.zeros_like(u_t), \
                                             torch.zeros_like(r_t)
                T_tm1_dx_tm1_x_t = torch.zeros_like(t_t)
                T_tm1_dx_tm1_phi = torch.zeros_like(u_t)
            else:
                with torch.no_grad():
                    S_tm1, T_tm1, U_tm1, V_tm1 = self.approx_func_tm1(x_tm1)

                # T_tm1_dx_tm1_x_t = T_tm1 * dx_{t-1}(phi, x_t) / dx_t, shape (num_samples, xdim)
                if 'T' in self.funcs_to_approx:
                    T_tm1_dx_tm1_x_t = torch.autograd.grad((T_tm1 * x_tm1).sum(),
                                                           x_t, retain_graph=True)[0]
                else:
                    T_tm1_dx_tm1_x_t = torch.zeros_like(t_t)

                # T_tm1_dx_tm1_phi = T_tm1 * dx_{t-1}(phi, x_t) / dphi, shape (num_samples, phi_dim)
                T_tm1_dx_tm1_phi = torch.zeros_like(u_t)
                if 'U' in self.funcs_to_approx:
                    for i in range(num_samples):
                        T_tm1_dx_tm1_phi[i, :] = nn.utils.parameters_to_vector(
                            torch.autograd.grad((T_tm1[i, :] * x_tm1[i, :]).sum() +
                                                0 * self.get_phi_params(flatten=True).mean(),
                                                self.get_phi_params(), retain_graph=True))

            return x_t.detach(), (s_t + self.approx_decay * S_tm1).detach(), \
                   (t_t + self.approx_decay * T_tm1_dx_tm1_x_t).detach(), \
                   (u_t + self.approx_decay * (U_tm1 + T_tm1_dx_tm1_phi)).detach(), \
                   (r_t + self.approx_decay * V_tm1).detach()

    def update_func_t(self, train_input, *train_outputs, t=None):
        """
            Fit the kernel function approx_func_t
        """
        if t is None:
            t = self.T - self.window_size + 1  # By default, we update func_t at time T (func_T when window_size=1)
        if t >= self.approx_updates_start_t:  # approx_updates_start_t is t of first fitted approx_func_t
            # Fit new approx_func
            if t == self.approx_updates_start_t:
                self.approx_func_t.kernel.log_sigma.data = torch.tensor(
                    np.log(utils.estimate_median_distance(train_input)).astype(float)).float().to(self.device)
                print("Update bandwidth to ", self.approx_func_t.kernel.log_sigma.exp().item())

            self.approx_func_t.fit(train_input, *train_outputs)
        else:
            print("t < approx_updates_start_t, func_t not updated")

    def func_t_loss(self, train_input, *train_outputs):
        """
            Compute MSE loss of approx_func_t
            The losses between S, T, U, V are balanced by weights
            Return the averaged loss, preds = approx_func_t(train_input), and componentwise losses
        """
        preds = self.approx_func_t(train_input)  # TODO: make this consistent with non-amortised model
        losses = []
        for i in range(len(train_outputs)):
            if preds[i].shape[1] > 0:
                losses.append(torch.mean((preds[i] - train_outputs[i]) ** 2))
            else:
                losses.append(torch.tensor(0.).to(self.device))
        # output_dim = sum([train_outputs[i].shape[1] for i in range(len(train_outputs))])
        weights = [0.25] * 4
        loss = sum([losses[i] * weights[i] for i in range(len(train_outputs))])
        return loss, preds, losses

    def populate_grads(self, y, num_samples):
        """
            Compute RMLE training objective at time T:
            E_{q(x_{T-window_size:T}|y^T)} [ V(x_{T-window_size}) + sum_{l=1}^window_size r_{T-window_size+l} ] -
            E_{q(x_{T-window_size}|y^{T-window_size})} [ V(x_{T-window_size}) ]
            (This is approximately ELBO_T - ELBO_{T - window_size})
        """
        self.rnn_forward(y)

        if self.T < self.window_size:
            all_r_results = self.sample_and_compute_joint_r_t(y, num_samples, self.T + 1)

            sum_r = sum(all_r_results["r_values"])
            log_q_x_T = all_r_results["log_q_x_T"]

            loss = - (sum_r - log_q_x_T).mean()

        elif self.T >= self.window_size:
            all_r_results_1 = self.sample_and_compute_joint_r_t(y, num_samples, self.window_size)
            x_samples_1 = all_r_results_1["x_samples"]
            x_samples_2, _ = self.sample_q_T(num_samples, T=self.T - self.window_size)
            x_samples_2 = [x_samples_2]

            sum_r = sum(all_r_results_1["r_values"])
            log_q_x_T = all_r_results_1["log_q_x_T"]

            x_tm1_1 = x_samples_1[0]

            if self.T <= self.approx_updates_start_t + self.window_size - 1:  # Do not have approx_func_tm1 before
                # AELBO-2
                q_tm1_mean, q_tm1_std = self.compute_filter_stats(self.T - self.window_size)
                q_tm1_mean, q_tm1_std = q_tm1_mean.detach(), q_tm1_std.detach()
                log_q_x_tm1 = self.compute_log_q_t(x_tm1_1, q_tm1_mean, q_tm1_std)
                loss = - (sum_r + log_q_x_tm1 - log_q_x_T).mean()
            else:
                if self.approx_with_filter:
                    log_q_x_tm1 = self.compute_log_q_t(x_tm1_1, *self.compute_filter_stats(self.T - self.window_size))
                    sum_r += log_q_x_tm1

                with torch.no_grad():
                    S_tm1_1, T_tm1_1, U_tm1_1, V_tm1_1 = self.approx_func_tm1(x_tm1_1)

                x_tm1_2 = x_samples_2[0]
                with torch.no_grad():
                    S_tm1_2, T_tm1_2, U_tm1_2, _ = self.approx_func_tm1(x_tm1_2)

                V_sum_1 = torch.zeros_like(sum_r)  # shape (num_samples, 1)
                V_sum_2 = torch.zeros_like(sum_r)  # shape (num_samples, 1)

                if self.theta_dim > 0 and 'S' in self.funcs_to_approx:
                    V_sum_1 += (S_tm1_1 * self.get_theta_params(flatten=True)).sum(1, keepdim=True)
                    V_sum_2 += (S_tm1_2 * self.get_theta_params(flatten=True)).sum(1, keepdim=True)

                if 'T' in self.funcs_to_approx:
                    V_sum_1 += (T_tm1_1 * x_tm1_1).sum(1, keepdim=True)
                    if self.approx_with_filter:
                        V_sum_2 += (T_tm1_2 * x_tm1_2).sum(1, keepdim=True)

                if 'U' in self.funcs_to_approx:
                    V_sum_1 += (U_tm1_1 * self.get_phi_params(flatten=True)).sum(1, keepdim=True)
                    V_sum_2 += (U_tm1_2 * self.get_phi_params(flatten=True)).sum(1, keepdim=True)

                V_diff = V_sum_1 - V_sum_2

                # The loss gradient corresponds to the RMLE training objective, the loss value tracks ELBO_T
                loss = - (sum_r + V_diff - log_q_x_T - V_diff.detach() + V_tm1_1.detach()).mean()

        # Regularize RNN output state
        loss += self.rnn_h_lambda * torch.mean(self.rnn_outputs**2, 0).sum()

        loss.backward()
        return loss

    def get_func_t_params(self):
        return self.approx_func_t.parameters()


class Net_Amortised_Model(Kernel_Amortised_Model):
    def update_func_t(self, train_input, *train_outputs, t=None):
        """
            Update normalization of the NN function approx_func_t
        """
        if t is None:
            t = self.T - self.window_size + 1  # By default, we update func_t at time T (func_T when window_size=1)
        if t >= self.approx_updates_start_t:  # approx_updates_start_t is t of first fitted approx_func_t
            self.approx_func_t.update_normalization(train_input, *train_outputs)
        else:
            print("t < approx_updates_start_t, func_t not updated")

    def func_t_loss(self, train_input, *train_outputs):  # TODO: Can use a different loss here
        """
            Compute MSE loss of approx_func_t
            The losses between S, T, U, V are balanced by weights
            Return the averaged loss, preds = approx_func_t(train_input), and componentwise losses
        """
        preds = self.approx_func_t(train_input)  # TODO: make this consistent with non-amortised model
        losses = []
        for i in range(len(train_outputs)):
            if preds[i].shape[1] > 0:
                losses.append(torch.mean((preds[i] - train_outputs[i]) ** 2))
            else:
                losses.append(torch.tensor(0.).to(self.device))
        # output_dim = sum([train_outputs[i].shape[1] for i in range(len(train_outputs))])
        weights = [0.25] * 4
        loss = sum([losses[i] * weights[i] for i in range(len(train_outputs))])
        return loss, preds, losses


class SeparateTimeKernelAmortisedModel(AmortizedModelBase):
    def __init__(self, device, xdim, ydim, q_rnn_constructor,
                q_t_net_constructor, cond_q_t_net_constructor, F_fn,
                 G_fn, p_0_dist, rnn_window_size, approx_func_constructor,
                 approx_with_filter, funcs_to_approx, theta_to_learn):

        self.q_rnn_constructor = q_rnn_constructor
        self.q_t_net_constructor = q_t_net_constructor
        self.cond_q_t_net_constructor = cond_q_t_net_constructor

        super().__init__(device, xdim, ydim, q_rnn_constructor(),
            q_t_net_constructor(), cond_q_t_net_constructor(), F_fn,
            G_fn, p_0_dist, 1, rnn_window_size, theta_to_learn=theta_to_learn)


        self.approx_func_constructor = approx_func_constructor
        self.approx_with_filter = approx_with_filter
        self.funcs_to_approx = funcs_to_approx # Subset of 'ST'
        def replace_none(x):
            return x if x is not None else []
        self.theta_dim = len(replace_none(self.get_theta_params(flatten=True)))

    def advance_timestep(self, y_T):
        super().advance_timestep(y_T)

        if self.T == 0:
            self.approx_func_t = self.approx_func_constructor()
        else:
            self.approx_func_tm1 = self.approx_func_t
            self.approx_func_t = self.approx_func_constructor()
            self.approx_func_t.load_state_dict(self.approx_func_tm1.state_dict())
            self.approx_func_tm1.requires_grad_(False)
            self.approx_func_tm1.eval()

            self.q_tm1_rnn = self.q_rnn_constructor()
            self.q_tm1_rnn.load_state_dict(self.q_rnn.state_dict())
            self.q_tm1_net = self.q_t_net_constructor()
            self.q_tm1_net.load_state_dict(self.q_t_net.state_dict())

    def generate_training_dataset(self, y, num_samples, disperse_temp=1):
        t = self.T
        self.rnn_forward(y)

        r_results = self.sample_and_compute_r_t(y[t, :], num_samples, t=t,
            disperse_temp=disperse_temp)
        x_t = r_results["x_t"]
        r_t = r_results["r_t"]

        if t>0:
            x_tm1 = r_results["x_tm1"]

        if self.approx_with_filter and t == 0:
            r_t -= self.compute_log_q_t(x_t, *self.compute_filter_stats(t))

        if self.approx_with_filter and t > 0:

            t_start = max(self.T - self.rnn_window_size, 0)
            if t_start == 0:
                rnn_tm1_outputs, _ = self.q_tm1_rnn(y[t_start:(self.T + 1), :].view(self.T - t_start + 1, 1, self.ydim))
            else:
                rnn_tm1_outputs, _ = self.q_tm1_rnn(y[t_start:(self.T + 1), :].view(self.T - t_start + 1, 1, self.ydim),
                                                self.q_rnn_hist_hn_list[t_start - 1])
            q_tm1_stats = self.q_tm1_net(rnn_tm1_outputs[-2, 0, :])
            q_tm1_stats = (q_tm1_stats[0].detach(), q_tm1_stats[1].detach())
            log_q_x_tm1 = self.compute_log_q_t(x_tm1, *q_tm1_stats)

            log_q_t = self.compute_log_q_t(x_t, *self.compute_filter_stats(t))
            r_t += (log_q_x_tm1 - log_q_t)



        # t_t = \nabla_{x_t} r_t(x_{t-1}(phi, x_t), x_t), shape (num_samples, xdim)
        t_t = torch.autograd.grad(r_t.sum(), x_t, retain_graph=True)[0]

        s_t = torch.zeros((num_samples, self.theta_dim), device=self.device)
        if self.theta_dim > 0 and 'S' in self.funcs_to_approx:
            for i in range(num_samples):
                if t == 0 and \
                   'F' in self.theta_to_learn and \
                   not 'G' in self.theta_to_learn:
                   s_t[i, :] = torch.zeros((self.theta_dim,), device=self.device)
                else:
                    autograd_out =  torch.autograd.grad(r_t[i, 0], self.get_theta_params(),
                            retain_graph=True, allow_unused=True)
                    grads_list = []
                    for idx, p in enumerate(self.get_theta_params()):
                        if autograd_out[idx] is None:
                            grads_list.append(torch.zeros_like(p))
                        else:
                            grads_list.append(autograd_out[idx])
                    s_t[i, :] = nn.utils.parameters_to_vector(
                        grads_list
                    )

        if t == 0:
            T_tm1_dx_tm1_x_t = torch.zeros_like(t_t)
            S_tm1 = torch.zeros_like(s_t)
        else:
            with torch.no_grad():
                S_tm1, T_tm1 = self.approx_func_tm1(x_tm1, self.T-1)
            T_tm1_dx_tm1_x_t = torch.autograd.grad((T_tm1 * x_tm1).sum(),
                x_t)[0]


        return x_t.detach(),\
               (s_t + S_tm1).detach(),\
               (t_t + T_tm1_dx_tm1_x_t).detach()


    def update_func_t(self, train_input, *train_outputs):
        if self.T == 0:
            self.approx_func_t.kernel.log_sigma.data = torch.tensor(
                np.log(utils.estimate_median_distance(train_input)).astype(float)).float().to(self.device)
            print("Update bandwidth to ", self.approx_func_t.kernel.log_sigma.exp().item())
        self.approx_func_t.fit(train_input, *train_outputs)

    def func_t_loss(self, train_input, *train_outputs):
        preds = self.approx_func_t(train_input, self.T)  # TODO: make this consistent with non-amortised model
        losses = []
        for i in range(len(train_outputs)):
            if preds[i].shape[1] > 0:
                losses.append(torch.mean((preds[i] - train_outputs[i]) ** 2))
            else:
                losses.append(torch.tensor(0.).to(self.device))
        # output_dim = sum([train_outputs[i].shape[1] for i in range(len(train_outputs))])
        weights = [0.25] * 4
        loss = sum([losses[i] * weights[i] for i in range(len(train_outputs))])
        return loss, preds, losses

    def populate_grads(self, y, num_samples):

        self.rnn_forward(y)

        if self.T == 0:
            x_0, q_0_stats = self.sample_q_T(num_samples)
            r_t = self.compute_r_t(x_0, y[0, :], None)
            r_t -= self.compute_log_q_t(x_0, *q_0_stats)
            loss = -r_t.mean()
        else:
            r_results = self.sample_and_compute_joint_r_t(y, num_samples, 1)
            x_samples_1 = r_results["x_samples"]
            assert len(r_results["r_values"]) == 1
            assert len(x_samples_1) == 2

            r_t = r_results["r_values"][0] - r_results["log_q_x_T"]

            t_start = max(self.T - self.rnn_window_size, 0)
            if t_start == 0:
                rnn_tm1_outputs, _ = self.q_tm1_rnn(y[t_start:(self.T + 1), :].view(self.T - t_start + 1, 1, self.ydim))
            else:
                rnn_tm1_outputs, _ = self.q_tm1_rnn(y[t_start:(self.T + 1), :].view(self.T - t_start + 1, 1, self.ydim),
                                                self.q_rnn_hist_hn_list[t_start - 1])
            q_tm1_stats = self.q_tm1_net(rnn_tm1_outputs[-2, 0, :])
            q_tm1_stats = (q_tm1_stats[0].detach(), q_tm1_stats[1].detach())
            log_q_x_tm1 = self.compute_log_q_t(x_samples_1[0], *q_tm1_stats)

            if self.approx_with_filter:
                r_t += log_q_x_tm1

            x_samples_2 = q_tm1_stats[0] + \
                torch.randn_like(x_samples_1[1]).to(self.device) * \
                q_tm1_stats[1]
            x_samples_2 = [x_samples_2]



            with torch.no_grad():
                S_tm1_1, T_tm1_1 = self.approx_func_tm1(x_samples_1[0], self.T-1)

            with torch.no_grad():
                S_tm1_2, _ = self.approx_func_tm1(x_samples_2[0], self.T-1)

            Tdx = (T_tm1_1 * x_samples_1[0]).sum(1, keepdim=True)

            if self.theta_dim > 0 and 'S' in self.funcs_to_approx:
                S_dtheta_1 = (S_tm1_1 * self.get_theta_params(flatten=True)).sum(1, keepdim=True)
                S_dtheta_2 = (S_tm1_2 * self.get_theta_params(flatten=True)).sum(1, keepdim=True)
                loss = -(r_t + Tdx + S_dtheta_1 - S_dtheta_2).mean()
            else:
                loss = -(r_t + Tdx).mean()


        loss.backward()
        return loss

    def get_func_t_params(self):
        return self.approx_func_t.parameters()

class SeparateTimeAnalyticAmortisedModel(AmortizedModelBase):
    def __init__(self, device, xdim, ydim, q_rnn_constructor,
        q_t_net_constructor, cond_q_t_net_constructor, F_fn,
        G_fn, p_0_dist, rnn_window_size, backrollout_size):

        self.q_rnn_constructor = q_rnn_constructor
        self.q_t_net_constructor = q_t_net_constructor
        self.cond_q_t_net_constructor = cond_q_t_net_constructor

        super().__init__(device, xdim, ydim, q_rnn_constructor(),
            q_t_net_constructor(), cond_q_t_net_constructor(),
            F_fn, G_fn, p_0_dist, 1, rnn_window_size)

        self.backrollout_size = backrollout_size

        self.rnn_history = []

        self.q_rnn_list = utils.TimeStore(None,
            backrollout_size + 2,
            "ModuleList")
        self.q_t_net_list = utils.TimeStore(None,
            backrollout_size + 2,
            "ModuleList")
        self.cond_q_t_net_list = utils.TimeStore(None,
            backrollout_size + 2,
            "ModuleList")

    def advance_timestep(self, y_T):
        self.T += 1

        self.q_rnn_list.append(self.q_rnn_constructor())
        self.q_t_net_list.append(self.q_t_net_constructor())
        self.cond_q_t_net_list.append(self.cond_q_t_net_constructor())

        if self.T == 0:
            self.q_rnn_list[self.T].load_state_dict(self.q_rnn.state_dict())
            self.q_t_net_list[self.T].load_state_dict(self.q_t_net.state_dict())
            self.cond_q_t_net_list[self.T].load_state_dict(self.cond_q_t_net.state_dict())
        else:
            self.q_rnn_list[self.T-1].load_state_dict(self.q_rnn.state_dict())
            self.q_rnn_list[self.T].load_state_dict(self.q_rnn.state_dict())
            self.q_t_net_list[self.T-1].load_state_dict(self.q_t_net.state_dict())
            self.q_t_net_list[self.T].load_state_dict(self.q_t_net.state_dict())
            self.cond_q_t_net_list[self.T-1].load_state_dict(self.cond_q_t_net.state_dict())
            self.cond_q_t_net_list[self.T].load_state_dict(self.cond_q_t_net.state_dict())

        self.rnn_history.append(None)
        self.q_rnn_hist_hn_list.append(None)
        self.q_rnn_hist_output_list.append(None)

    def detach_rnn_hist_hn(self, y):
        super().detach_rnn_hist_hn(y)
        t_start = max(self.T - self.backrollout_size - self.rnn_window_size, 0)
        if t_start >= 1:
            if t_start == 1:
                rnn_init = torch.zeros(self.q_rnn_list[self.T].num_layers, 1,
                    self.q_rnn_list[self.T].hidden_size).to(self.device)
            else:
                rnn_init = self.rnn_history[t_start - 2]

            _, h = self.q_rnn_list[self.T](y[t_start-1, :].view(1, 1, self.ydim),
                rnn_init)

            self.rnn_history[t_start-1] = h.detach()

    def populate_grads(self, y, num_samples):
        if self.T < self.backrollout_size + 1:
            return torch.tensor(0.0).to(self.device)

        self.zero_grad()

        x_samples = {}
        q_back_stats = {}

        # x_samples[t] = x_t(\eps_t, x_{t+1}, \phi_{t+1})
        # except x_samples[T] = x_T(\eps_T, \phi_T)

        # q_back_stats[t] = q_{t+1}^{\phi_{t+1}}(x_t | x_{t+1})

        t_start = max(self.T - self.backrollout_size - self.rnn_window_size, 0)

        if t_start == 0:
            rnn_init = torch.zeros(self.q_rnn_list[self.T].num_layers, 1,
                self.q_rnn_list[self.T].hidden_size).to(self.device)
        else:
            rnn_init = self.rnn_history[t_start - 1]

        # sample x_T using phi_T
        time_T_phi_T_rnn_outputs, _ = self.q_rnn_list[self.T](\
            y[t_start:(self.T+1), :].view(self.T-t_start+1, 1, self.ydim),
            rnn_init
        )
        x_T_mean, x_T_std = self.q_t_net_list[self.T](time_T_phi_T_rnn_outputs[-1, 0, :])

        x_T_samples = x_T_mean + \
            x_T_std * torch.randn(num_samples, self.xdim).to(self.device)

        x_samples[self.T] = x_T_samples
        q_T_stats = (x_T_mean, x_T_std)

        # sample x_k using phi_{k+1}
        for k in range(self.T-1, self.T-self.backrollout_size-1, -1):
            # k = [T-1, T-2, ..., T-L] with L = backrollout_size

            time_k_phi_kp1_rnn_outputs, _ = self.q_rnn_list[k+1](\
                y[t_start:(k+1), :].view(k+1-t_start, 1, self.ydim),
                rnn_init    
            )

            x_k_mean, x_k_std = self.cond_q_t_net_list[k+1](
                torch.cat([
                    x_samples[k+1],
                    time_k_phi_kp1_rnn_outputs[-1, 0, :]\
                        .expand(num_samples, self.q_rnn_list[self.T].hidden_size)
                ], dim=1)
            )

            x_k_samples = x_k_mean + \
                x_k_std * torch.randn(num_samples, self.xdim).to(self.device)

            x_samples[k] = x_k_samples
            q_back_stats[k] = (x_k_mean, x_k_std)

        r_sum = 0
        for k in range(self.T-self.backrollout_size+1, self.T+1, 1):
            # k = [T-L+1, T-L+2, ..., T] L = backrollout_size

            log_p_k = self.compute_log_p_t(x_samples[k], y[k, :],
                x_samples[k-1], k)
            r_sum += log_p_k["log_p_x_t"] + log_p_k["log_p_y_t"]

            r_sum -= self.compute_log_q_t(x_samples[k-1],
                *q_back_stats[k-1])

        r_sum -= self.compute_log_q_t(x_samples[self.T],
            *q_T_stats)

        mean_r = r_sum.mean()
        mean_r.backward()

        q_rnn_grads = {}
        q_net_grads = {}
        cond_q_net_grads = {}
        def add_to_dict(dict, key, val):
            if key in dict:
                dict[key] += val
            else:
                dict[key] = val
            return dict


        for k in range(self.T - self.backrollout_size + 1, self.T+1, 1):
            for name, p in self.q_rnn_list[k].named_parameters():
                q_rnn_grads = add_to_dict(q_rnn_grads, name, p.grad)
            for name, p in self.cond_q_t_net_list[k].named_parameters():
                cond_q_net_grads = add_to_dict(cond_q_net_grads, name, p.grad)

        # Only t=T has grads for q_net
        for name, p in self.q_t_net_list[self.T].named_parameters():
            q_net_grads = add_to_dict(q_net_grads, name, p.grad)


        # Set to negative of the sum in order to maximize
        for name, p in self.q_rnn.named_parameters():
            p.grad = -q_rnn_grads[name]
        for name, p in self.q_t_net.named_parameters():
            p.grad = -q_net_grads[name]
        for name, p in self.cond_q_t_net.named_parameters():
            p.grad = -cond_q_net_grads[name]

        return mean_r