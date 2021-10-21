#%%
import numpy as np

class GaussianHMM():
    """
    Gaussian hidden markov model:
    x_t are states, y_t are observations

    x_t = F x_{t-1} + \mu_t    \mu_t ~ N(0, U)    for  t = {1, 2, ...}
    y_t = G x_t     + \nu_t    \nu_t ~ N(0, V)    for  t = {0, 1, 2, ...}

    x_0 sampled from N(0, I)
    """
    def __init__(self, xdim, ydim, F, G, U, V):
        self.xdim = xdim
        self.ydim = ydim
        self.x_0 = np.random.randn(xdim)
        self.F = F
        self.G = G
        self.U = U
        self.V = V

    def generate_data(self, T):
        # Generates hidden states and observations up to time T
        self.x = np.zeros((T, self.xdim))
        self.y = np.zeros((T, self.ydim))

        self.x[0, :] = self.x_0

        for t in range(T):
            nu_t = self._sample_zero_normal(self.V)
            y_t = np.dot(self.G, self.x[t, :]) + nu_t

            mu_t = self._sample_zero_normal(self.U)
            x_tp1 = np.dot(self.F, self.x[t, :]) + mu_t

            self.y[t, :] = y_t
            if t < T-1:
                self.x[t+1, :] = x_tp1

        return self.x, self.y

    def _sample_zero_normal(self, cov):
        dim = cov.shape[0]
        # if dim > 1:
        sample = np.random.multivariate_normal(
            mean=np.zeros((dim)),
            cov=cov
        )
        # else:
        #     sample = np.random.normal(0, np.sqrt(cov))
        return sample

def construct_HMM_matrices(dim, F_eigvals, G_eigvals, U_std=0.3, V_std=0.3,
    diag=False):
    # Makes x and y have the same dimension
    # If diag = True then F and G will be diagonal

    def make_matrix(dim, eigvals):
        Q = np.random.randn(dim,dim)
        return np.matmul(np.matmul(Q, np.diag(eigvals)), np.linalg.inv(Q))

    if diag:
        F = np.diag(F_eigvals)
        G = np.diag(G_eigvals)
    else:
        F = make_matrix(dim, F_eigvals)
        G = make_matrix(dim, G_eigvals)
    U = U_std**2 * np.eye(dim)
    V = V_std**2 * np.eye(dim)
    return F, G, U, V


class NonlinearGaussianHMM:
    def __init__(self, xdim, ydim, F_fn, G_fn, U, V, Q_fn, R_fn, mean_0, cov_0):
        self.xdim = xdim
        self.ydim = ydim
        self.F_fn = F_fn
        self.G_fn = G_fn
        self.U = U
        self.V = V
        self.Q_fn = Q_fn
        self.R_fn = R_fn
        self.mean_0 = mean_0
        self.cov_0 = cov_0

    def generate_data(self, T):
        # Generates hidden states and observations up to time T
        self.x = np.zeros((T, self.xdim))
        self.y = np.zeros((T, self.ydim))

        self.x[0, :] = np.random.multivariate_normal(self.mean_0, self.cov_0)

        for t in range(T):
            nu_t = self._sample_zero_normal(self.V)
            y_t = self.G_fn(self.x[t, :]) + self.R_fn(self.x[t, :]) @ nu_t

            mu_t = self._sample_zero_normal(self.U)
            x_tp1 = self.F_fn(self.x[t, :]) + self.Q_fn(self.x[t, :]) @ mu_t

            self.y[t, :] = y_t
            if t < T-1:
                self.x[t+1, :] = x_tp1

        return self.x, self.y

    def _sample_zero_normal(self, cov):
        dim = cov.shape[0]
        sample = np.random.multivariate_normal(
            mean=np.zeros((dim)),
            cov=cov
        )
        return sample
