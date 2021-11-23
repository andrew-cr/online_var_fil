import torch
import torch.nn as nn
import torch.nn.functional as functional


def MLP(net_dims, nonlinearity):
    modules = []
    for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
        modules.append(nn.Linear(in_dim, out_dim))
        modules.append(nonlinearity())
    modules.append(nn.Linear(net_dims[-2], net_dims[-1]))
    return nn.Sequential(*modules)


class NormalNet(nn.Module):
    def __init__(self, net_module):
        super().__init__()
        self.net_module = net_module

    def forward(self, x):
        x = self.net_module(x)
        mu, logsigma = torch.chunk(x, 2, dim=-1)
        sig = functional.softplus(logsigma)
        return mu, sig


class MLPRNN(nn.Module):
    """
    Recursive MLP module: h_n = MLP(input_n, h_{n-1})
    """
    def __init__(self, net_dims, nonlinearity):
        super().__init__()
        self.net_dims = net_dims  # net_dims[0] = ydim + 2*xdim, net_dims[-1] = 2*xdim
        self.MLP_module = MLP(net_dims, nonlinearity)
        self.hidden_size = net_dims[-1]  # 2*xdim

    def forward(self, input, h_0=None):
        # input shape (L, N, ydim)
        # h_0 shape (1, N, 2*xdim)
        # output shape (L, N, 2*xdim), (1, N, 2*xdim)
        assert self.net_dims[0] == input.shape[-1] + self.net_dims[-1]
        if h_0 is None:
            h_0 = torch.zeros((1, input.shape[1], self.net_dims[-1]), device=input.device)
        assert self.net_dims[-1] == h_0.shape[-1]
        L = input.shape[0]
        output = []
        for t in range(L):
            h_n = self.MLP_module(torch.cat([input[t].unsqueeze(0), h_0], dim=-1))
            output.append(h_n)
            h_0 = h_n
        return torch.cat(output, dim=0), h_n
