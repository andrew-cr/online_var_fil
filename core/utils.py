from operator import mul
import numpy as np
import torch
import torch.nn as nn
import pandas
import glob
import yaml
import subprocess
import os

from scipy.spatial.distance import pdist

def back_1_joint_smoothing_prob(x_t, x_t_1, K_t_mean, K_t_covar, K_t_1_mean, \
    K_t_1_covar, F, U):
    """
        Evaluates log p(x_{t-1:t}|y_{1:t}) at the given samples
        Using pytorch

        x_t (N, dim)
        x_t_1 (N, dim)
        K_t_mean etc are Kalman filter statistics

        Returns (N)
    """
    log_p_x_t = log_normal_type3(x_t, K_t_mean, torch.logdet(K_t_covar),
        torch.inverse(K_t_covar))

    t_1_covar = torch.inverse(torch.inverse(K_t_1_covar) + \
        F.T @ torch.inverse(U) @ F)
    t_1_mean = (t_1_covar @ (torch.inverse(K_t_1_covar) @ K_t_1_mean + \
        (F.T @ torch.inverse(U) @ x_t.T).T).unsqueeze(2))[:, :, 0]

    log_p_x_t_1 = log_normal_type4(x_t_1, t_1_mean, torch.logdet(t_1_covar),
        torch.inverse(t_1_covar))

    return log_p_x_t + log_p_x_t_1


def KL_between_q_and_p_linear_back_q(q_mu_t, q_b_t, q_W_t, q_sigma_t,
    q_sigma_tm1, kalman_mu_t, kalman_cov_t, kalman_mu_tm1, kalman_cov_tm1,
    F, U):
    """
        Calculates analytic value of KL(q(x_tm1, x_t) || p(x_tm1, x_t | y_{1:t}))
        When q back 1 stats have a linear dependence on x_t

    """
    d = q_mu_t.shape[0]
    q_joint_mean = np.zeros(2*d)
    q_joint_mean[0:d] = q_b_t + q_W_t @ q_mu_t
    q_joint_mean[d:] = q_mu_t

    q_joint_cov = np.zeros((2*d, 2*d))
    q_joint_cov[0:d, 0:d] = np.diag(q_sigma_tm1) + q_W_t @ np.diag(q_sigma_t) @ q_W_t.T
    q_joint_cov[0:d, d:] = q_W_t @ np.diag(q_sigma_t)
    q_joint_cov[d:, 0:d] = (q_W_t @ np.diag(q_sigma_t)).T
    q_joint_cov[d:, d:] = np.diag(q_sigma_t)

    tmp = np.linalg.inv(np.linalg.inv(kalman_cov_tm1) + F.T @ np.linalg.inv(U) @ F)

    p_joint_mean = np.zeros(2*d)
    p_joint_mean[0:d] = tmp @ np.linalg.inv(kalman_cov_tm1) @ kalman_mu_tm1 + \
        tmp @ F.T @ np.linalg.inv(U) @ kalman_mu_t
    p_joint_mean[d:] = kalman_mu_t

    p_joint_cov = np.zeros((2*d, 2*d))
    p_joint_cov[0:d, 0:d] = tmp + tmp @ F.T @ np.linalg.inv(U) @ kalman_cov_t @ \
        np.linalg.inv(U) @ F @ tmp
    p_joint_cov[0:d, d:] = tmp @ F.T @ np.linalg.inv(U) @ kalman_cov_t
    p_joint_cov[d:, 0:d] = p_joint_cov[0:d, d:].T
    p_joint_cov[d:, d:] = kalman_cov_t

    return multivar_gaussian_kl(q_joint_mean, q_joint_cov, p_joint_mean, p_joint_cov)




def multivar_gaussian_kl(mean_0, var_0, mean_1, var_1):
    # computes KL( N(mean_0, var_0) || N(mean_1, var_1))
    # mean (dim) var (dim x dim)
    trace_term = np.trace(np.linalg.inv(var_1) @ var_0)
    dot_term = np.dot(mean_1 - mean_0, np.dot(np.linalg.inv(var_1), mean_1-mean_0))
    log_term = np.linalg.slogdet(var_1)[1] - np.linalg.slogdet(var_0)[1]
    return 0.5 * (trace_term + dot_term - mean_0.shape[0] + log_term)



def save_git_hash(cwd):
    git_hash = subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"],
        cwd=cwd)
    git_hash = git_hash.decode("utf-8")
    with open('git_hash.txt', 'w') as f:
        f.write(git_hash)



def log_normal_type3(x, mean, logdet_cov, inv_cov):
    """
        Calculates a batch of log N(x; mean, cov)
        Expected shapes
        x (N, dim)
        mean (dim)
        logdet_cov ()
        inv_cov (dim, dim)
        Output (N)
    """
    log_p = -(x.shape[1]/2) * np.log(2*np.pi) -\
        0.5 * logdet_cov - \
        0.5 * torch.sum(
            (x - mean) * ((inv_cov @ (x - mean).T).T),
            dim=1
        )
    return log_p

def log_normal_type4(x, mean, logdet_cov, inv_cov):
    """
        Calculates a batch of log N(x; mean, cov)
        Expected shapes
        x (N, dim)
        mean (N, dim)
        logdet_cov ()
        inv_cov (dim, dim)
        Output (N)
    """
    return log_normal_type3(x, mean, logdet_cov, inv_cov)


def make_matrix(dim, min_eigval, max_eigval, diag=True):
    eigvals = np.random.uniform(min_eigval, max_eigval, dim)
    if diag or dim == 1:
        return np.diag(eigvals)
    else:
        from scipy.stats import ortho_group
        Q = ortho_group.rvs(dim)
        return Q @ np.diag(eigvals) @ np.linalg.inv(Q)

def gaussian_posterior(y, prior_mean, prior_cov, G, V, G_fn=None):
    # Compute posterior mean, cov of N(x; prior_mean, prior_cov) N(y; G @ x, V)
    # y: (*, ydim), prior_mean: (*, xdim), prior_cov: (*, xdim, xdim),
    # G: (*, ydim, xdim), V: (*, ydim, ydim)
    K = prior_cov @ G.transpose(-2, -1) @ torch.linalg.inv(G @ prior_cov @ G.transpose(-2, -1) + V)
    I_KG = torch.eye(prior_cov.shape[-1], device=prior_cov.device) - K @ G
    post_cov = I_KG @ prior_cov
    if G_fn is None:
        post_mean = (prior_mean.unsqueeze(-2) @ I_KG.transpose(-2, -1) +
                     y.unsqueeze(-2) @ K.transpose(-2, -1)).squeeze(-2)
    else:
        post_mean = prior_mean + ((y - G_fn(prior_mean)).unsqueeze(-2) @ K.transpose(-2, -1)).squeeze(-2)
    return post_mean, post_cov

def sample_cov(x, y=None, w=None):
    assert len(x.shape) == 2
    if w is None:
        w = 1 / x.shape[0] * torch.ones((x.shape[0], 1)).to(x.device)
    else:
        w = w.view(x.shape[0], 1)
    x_centred = x - (x * w).sum(0)
    if y is None:
        y_centred = x_centred
    else:
        y_centred = y - (y * w).sum(0)
    cov = (w * x_centred).t() @ y_centred / (1 - (w**2).sum())
    return cov

def ess(log_w):
    ess_num = 2 * torch.logsumexp(log_w, 0)
    ess_denom = torch.logsumexp(2 * log_w, 0)
    log_ess = ess_num - ess_denom
    print("ESS: ", log_ess.exp().item())
    return log_ess.exp()

def l2_distance(FX, FY, stable=True):
    if stable:
        FK = torch.sum((FX[:, None, :] - FY[None, :, :]) ** 2, -1)
    else:
        FK = (FX ** 2).sum(-1, keepdim=True)
        FK = FK + (FY ** 2).sum(-1, keepdim=True).t()
        FK -= 2 * (FX[:, None, :] * FY[None, :, :]).sum(-1)
    return FK

def estimate_median_distance(data):
    return np.median(pdist(data.detach().cpu().numpy()))

def RK4_step(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

class TimeStore(nn.Module):
    """
        Acts like a ParameterList/ModuleList that just keeps getting longer but
        under the hood it only stores the most recent N values.
        Does not support removing items from the list.
    """
    def __init__(self, first_val, N, type):
        super().__init__()

        if type == "ParameterList":
            self.list = nn.ParameterList([first_val] if first_val else [])
        elif type == "ModuleList":
            self.list = nn.ModuleList([first_val] if first_val else [])
        else:
            raise ValueError("TimeStore unknown type: " + type)
        self.len = 1 if first_val else 0
        self.N = N

    def append(self, val):
        self.list.append(val)
        self.len += 1
        if len(self.list) > self.N:
            self.list = self.list[-self.N:]

    def __getitem__(self, index):
        adjusted_idx = index - self.len + len(self.list)

        if index < 0:
            raise ValueError("TimeStore does not support negative indexing")
        if index >= self.len:
            raise ValueError("TimeStore index {} is too large".format(index))
        if adjusted_idx < 0:
            raise ValueError("TimeStore attempting to access deleted entry")

        return self.list[adjusted_idx]